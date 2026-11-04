#!/usr/bin/env python3
"""Reference logits of a (reduced-depth) GLM-5.3-Flash checkpoint, for Legion's glm5next tests.

The trunk is Hugging Face transformers' own `Glm5NextTextModel`
(`transformers/models/glm5_next/modeling_glm5_next.py`, Apache-2.0, transformers >= 5.16) in
float32 on the CPU, eager attention, with the torch fallbacks of its KDA and conv kernels. The
weights are the checkpoint's: BF16 tensors widened to F32, NVFP4 ones (ModelOpt: E2M1 codes,
E4M3 per-16 scales, F32 `weight_scale_2`) dequantized exactly. Activations stay F32 (W4A16:
`input_scale` is not applied). The routed experts are dequantized per expert when a token
routes to one (`LazyExperts`, the same math as `Glm5NextTextExperts`), so a 288-expert layer
does not need 29 GB of F32 weights.

transformers does not run the MTP block (`layers.{num_hidden_layers}`); `mtp()` builds it from
transformers' own attention and MoE modules the way llama.cpp's `glm5next` `graph_mtp` and
SGLang's `DeepseekModelNextN` run it: `eh_proj([enorm(embed(t)) | hnorm(h)])`, then one DSA
attention + MoE layer with a plain residual (no hyper-connections), then `shared_head.norm` and
the LM head. Row `p` pairs token `fed[p]` with the trunk's final hidden state (after the final
norm) at `p - 1`, zeros at `p = 0`, as Legion's MTP passes do; `fed` is the prompt plus the
trunk's greedy pick after it.

    reference.py --ckpt DIR --ids IDS.txt --out DIR [--topk K] [--rows N] [--mtp]

Writes to --out: `logits.f32` (the last --rows positions' logits, F32 `[rows][vocab]`),
`argmax.i32` (every position's), `hidden.f32` (every position's final hidden state), and with
--mtp `mtp_hidden.f32` / `mtp_argmax.i32` / `mtp_pmax.f32` (every MTP row's), plus `meta.json`.
--topk overrides `index_topk` (a small budget makes the indexer select at short context).
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers.models.glm5_next import modeling_glm5_next as M
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

PREFIX = "model.language_model."

# E2M1 code -> value.
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


class Ckpt:
    def __init__(self, d):
        self.dir = d
        with open(os.path.join(d, "model.safetensors.index.json")) as f:
            self.map = json.load(f)["weight_map"]
        self.files = {}

    def has(self, name):
        return name in self.map

    def raw(self, name):
        f = self.map[name]
        if f not in self.files:
            self.files[f] = safe_open(os.path.join(self.dir, f), framework="pt")
        return self.files[f].get_tensor(name)

    def weight(self, base):
        """`base.weight` as F32, dequantized when it is NVFP4."""
        w = self.raw(base + ".weight")
        if not self.has(base + ".weight_scale"):
            return w.float()
        s = self.raw(base + ".weight_scale").float()
        s2 = self.raw(base + ".weight_scale_2").float()
        rows = w.shape[0]
        codes = torch.stack([w & 0xF, w >> 4], dim=-1).reshape(rows, -1).long()
        v = E2M1[codes]
        return v * s.repeat_interleave(16, dim=1) * s2

    def vec(self, name):
        return self.raw(name).float()


class LazyExperts(nn.Module):
    """`Glm5NextTextExperts.forward` with each expert's weights read when a token routes to it."""

    def __init__(self, ck, prefix, n, limit):
        super().__init__()
        self.ck, self.prefix, self.n, self.limit = ck, prefix, n, limit

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        for e in sorted(set(top_k_index.flatten().tolist())):
            top_k_pos, token_idx = torch.where((top_k_index == e).T)
            p = "%s.%d." % (self.prefix, e)
            gate = F.linear(hidden_states[token_idx], self.ck.weight(p + "gate_proj"))
            up = F.linear(hidden_states[token_idx], self.ck.weight(p + "up_proj"))
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            cur = F.linear(F.silu(gate) * up, self.ck.weight(p + "down_proj"))
            cur = cur * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, cur.to(final.dtype))
        return final


def set_param(mod, name, t):
    parts = name.split(".")
    for p in parts[:-1]:
        mod = getattr(mod, p)
    old = getattr(mod, parts[-1])
    assert tuple(old.shape) == tuple(t.shape), (name, tuple(old.shape), tuple(t.shape))
    if isinstance(old, nn.Parameter):
        setattr(mod, parts[-1], nn.Parameter(t.contiguous(), requires_grad=False))
    else:
        setattr(mod, parts[-1], t.contiguous())


def load_mlp(ck, mlp, src, cfg):
    """Dense MLP, or MoE (router, lazy experts, shared expert)."""
    if isinstance(mlp, M.Glm5NextTextMoE):
        set_param(mlp, "gate.weight", ck.vec(src + "gate.weight"))
        set_param(mlp, "gate.e_score_correction_bias", ck.vec(src + "gate.e_score_correction_bias"))
        mlp.experts = LazyExperts(ck, src + "experts", cfg.n_routed_experts, cfg.swiglu_limit)
        for p in ("gate_proj", "up_proj", "down_proj"):
            set_param(mlp, "shared_experts.%s.weight" % p, ck.weight(src + "shared_experts." + p))
    else:
        for p in ("gate_proj", "up_proj", "down_proj"):
            set_param(mlp, "%s.weight" % p, ck.weight(src + p))


def load_attention(ck, attn, src):
    """A DSA (MLA + indexer) attention module."""
    for p in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
        set_param(attn, p + ".weight", ck.weight(src + p))
    for p in ("q_a_layernorm", "kv_a_layernorm"):
        set_param(attn, p + ".weight", ck.vec(src + p + ".weight"))
    ix = src + "indexer."
    for p in ("wq_b", "wk", "weights_proj"):
        set_param(attn, "indexer.%s.weight" % p, ck.weight(ix + p))
    set_param(attn, "indexer.k_norm.weight", ck.vec(ix + "k_norm.weight"))
    set_param(attn, "indexer.k_norm.bias", ck.vec(ix + "k_norm.bias"))
    for p in ("index_kpool_compress_ape", "index_kpool_compress_gate"):
        set_param(attn, "indexer." + p, ck.vec(ix + p))


def load_kda(ck, attn, src):
    for p in ("q_proj", "k_proj", "v_proj", "b_proj", "g_a_proj", "g_b_proj", "o_proj"):
        set_param(attn, p + ".weight", ck.weight(src + p))
    set_param(attn, "forget_gate.f_a_proj.weight", ck.weight(src + "f_a_proj"))
    set_param(attn, "forget_gate.f_b_proj.weight", ck.weight(src + "f_b_proj"))
    set_param(attn, "forget_gate.dt_bias", ck.vec(src + "dt_bias"))
    set_param(attn, "forget_gate.A_log", ck.vec(src + "A_log"))
    conv = torch.cat([ck.vec(src + "%s_conv1d.weight" % c) for c in "qkv"], dim=0)
    set_param(attn, "conv1d.weight", conv)
    set_param(attn, "o_norm.weight", ck.vec(src + "o_norm.weight"))


def build_trunk(ck, cfg):
    with torch.device("meta"):
        model = M.Glm5NextTextModel(cfg)
    model.embed_tokens = nn.Embedding.from_pretrained(ck.vec(PREFIX + "embed_tokens.weight"))
    set_param(model, "norm.weight", ck.vec(PREFIX + "norm.weight"))
    for il, layer in enumerate(model.layers):
        src = "%slayers.%d." % (PREFIX, il)
        for which in ("attn", "ffn"):
            for p in ("fn", "base", "scale"):
                set_param(layer, "%s_hc.%s" % (which, p), ck.vec(src + "hc_%s_%s" % (which, p)))
        set_param(layer, "input_layernorm.weight", ck.vec(src + "input_layernorm.weight"))
        set_param(layer, "post_attention_layernorm.weight", ck.vec(src + "post_attention_layernorm.weight"))
        if layer.block_type == "linear_attention":
            load_kda(ck, layer.self_attn, src + "self_attn.")
        else:
            load_attention(ck, layer.self_attn, src + "self_attn.")
        load_mlp(ck, layer.mlp, src + "mlp.", cfg)
    leftover = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    leftover += [n for n, b in model.named_buffers() if b.device.type == "meta"]
    assert not leftover, leftover
    return model.eval()


class Mtp(nn.Module):
    """The NextN block: see the module docs."""

    def __init__(self, ck, cfg, il):
        super().__init__()
        src = "%slayers.%d." % (PREFIX, il)
        # transformers' modules index per-layer config lists by layer: give layer `il` a DSA
        # attention with its own indexer and a MoE.
        c = Glm5NextTextConfig(**cfg.to_dict())
        c.layer_types = list(cfg.layer_types) + ["indexed_attention"]
        c.mlp_layer_types = list(cfg.mlp_layer_types) + ["sparse"]
        c.indexer_types = list(cfg.indexer_types) + ["full"]
        c._attn_implementation = "eager"
        with torch.device("meta"):
            self.attn = M.Glm5NextTextAttention(c, il)
            self.mlp = M.Glm5NextTextMoE(c)
            self.input_layernorm = M.Glm5NextTextRMSNorm(c.hidden_size, c.rms_norm_eps)
            self.post_attention_layernorm = M.Glm5NextTextRMSNorm(c.hidden_size, c.rms_norm_eps)
            self.enorm = M.Glm5NextTextRMSNorm(c.hidden_size, c.rms_norm_eps)
            self.hnorm = M.Glm5NextTextRMSNorm(c.hidden_size, c.rms_norm_eps)
            self.head_norm = M.Glm5NextTextRMSNorm(c.hidden_size, c.rms_norm_eps)
            self.eh_proj = nn.Linear(2 * c.hidden_size, c.hidden_size, bias=False)
        load_attention(ck, self.attn, src + "self_attn.")
        load_mlp(ck, self.mlp, src + "mlp.", c)
        set_param(self, "input_layernorm.weight", ck.vec(src + "input_layernorm.weight"))
        set_param(self, "post_attention_layernorm.weight", ck.vec(src + "post_attention_layernorm.weight"))
        set_param(self, "enorm.weight", ck.vec(src + "enorm.weight"))
        set_param(self, "hnorm.weight", ck.vec(src + "hnorm.weight"))
        set_param(self, "head_norm.weight", ck.vec(src + "shared_head.norm.weight"))
        set_param(self, "eh_proj.weight", ck.weight(src + "eh_proj"))
        self.config = c

    def forward(self, embeds, h):
        x = self.eh_proj(torch.cat([self.enorm(embeds), self.hnorm(h)], dim=-1))
        mask = torch.ones(x.shape[:2], dtype=torch.bool)
        res = x
        a, _, _ = self.attn(self.input_layernorm(x), attention_mask=mask)
        x = res + a
        res = x
        x = res + self.mlp(self.post_attention_layernorm(x))
        return self.head_norm(x)


def write(path, t, dtype):
    t.detach().contiguous().to(dtype).numpy().tofile(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ids", required=True, help="token ids, whitespace-separated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, help="override index_topk")
    ap.add_argument("--rows", type=int, default=8, help="logits rows written (the last ones)")
    ap.add_argument("--mtp", action="store_true", help="also run the MTP block")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)

    with open(os.path.join(a.ckpt, "config.json")) as f:
        tc = json.load(f)["text_config"]
    cfg = Glm5NextTextConfig(**tc)
    if a.topk:
        cfg.index_topk = a.topk
    cfg._attn_implementation = "eager"
    ck = Ckpt(a.ckpt)
    ids = [int(x) for x in open(a.ids).read().split()]
    os.makedirs(a.out, exist_ok=True)

    t0 = time.time()
    model = build_trunk(ck, cfg)
    head = ck.weight("lm_head")
    t1 = time.time()
    out = model(input_ids=torch.tensor([ids]), use_cache=False)
    h = out.last_hidden_state[0]
    logits = F.linear(h, head)
    t2 = time.time()
    rows = min(a.rows, len(ids))
    write(os.path.join(a.out, "logits.f32"), logits[-rows:], torch.float32)
    write(os.path.join(a.out, "argmax.i32"), logits.argmax(-1), torch.int32)
    write(os.path.join(a.out, "hidden.f32"), h, torch.float32)
    meta = {"ids": ids, "rows": rows, "vocab": logits.shape[-1], "layers": cfg.num_hidden_layers,
            "index_topk": cfg.index_topk, "load_s": t1 - t0, "trunk_s": t2 - t1}

    n_mtp = tc.get("num_nextn_predict_layers", 0) or 0
    if a.mtp and n_mtp:
        mtp = Mtp(ck, cfg, cfg.num_hidden_layers)
        # Row p: token fed[p] (the prompt, then the trunk's greedy pick), h at p - 1.
        fed = ids + [int(logits[-1].argmax())]
        hh = torch.cat([torch.zeros(1, h.shape[-1]), h], dim=0)
        x = mtp(model.embed_tokens(torch.tensor([fed])), hh[None])[0]
        ml = F.linear(x, head)
        p = ml.softmax(-1)
        write(os.path.join(a.out, "mtp_hidden.f32"), x, torch.float32)
        write(os.path.join(a.out, "mtp_argmax.i32"), ml.argmax(-1), torch.int32)
        write(os.path.join(a.out, "mtp_pmax.f32"), p.max(-1).values, torch.float32)
        meta["mtp_fed"] = fed
        meta["mtp_s"] = time.time() - t2
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("ids", "mtp_fed")}))
    print("argmax:", logits.argmax(-1).tolist())


if __name__ == "__main__":
    main()
