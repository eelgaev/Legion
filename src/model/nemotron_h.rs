//! Nemotron-H (`nemotron_h` dense and `nemotron_h_moe` in GGUF, e.g. Nemotron 3.5 Lightning
//! 30B-A3B): NVIDIA's hybrid of Mamba-2 layers, a few attention layers and squared-ReLU FFN or
//! MoE layers. Every layer is one block of one kind (`attn_norm` RMS norm, the block, residual
//! add); the attention layers have no positional encoding.
//!
//! It runs on the core executor: the Mamba-2 layers keep their conv history and SSM state in
//! the same recurrent-state slots as the DeltaNet layers (conv width `d_inner + 2 × n_group ×
//! d_state` is the same formula; the SSM state is `[d_state, head_dim, n_head]` per slot), the
//! attention layers use the paged KV pool, and batching, prefix caching and speculative
//! rollback work unchanged.
//!
//! The graph follows llama.cpp's `src/models/nemotron-h.cpp` (also used for
//! `nemotron_h_moe`), `llm_build_mamba_base::build_mamba2_layer` and `llm_graph_context`'s
//! `build_ffn` / `build_moe_ffn` (MIT), op for op; weight matmuls go through [`crate::model::core::Linear`]
//! (NVFP4 precision hints and per-tensor `.scale`s, FP8 weights).

pub mod tiny;

use std::ops::Range;

use crate::ggml::{Context, Graph, Tensor, types};
use crate::gguf::GgufFile;

use crate::model::core::arch::same_as;
use crate::model::core::mtp::MtpGraph;
use crate::model::core::state::KvPool;
use crate::model::core::{
    Arch, ArchRef, GRAPH_SIZE, GroupShape, Hparams, Model, ModelError, Pass, Recurrent, StatePool,
    Tap, cols, concat_cols, stable_fusions,
};

/// What one layer of a Nemotron-H stack is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Block {
    /// Mamba-2 mixer (recurrent).
    Mamba,
    /// Self-attention, no RoPE.
    Attention,
    /// Squared-ReLU FFN: routed experts plus a shared expert in `nemotron_h_moe`, a plain
    /// up/down MLP in `nemotron_h`.
    Ffn,
}

/// The Nemotron-H specifics of [`Hparams`] (its [`Hparams::arch`]). The `ssm_*` fields of
/// [`Hparams`] describe the Mamba-2 mixer: `ssm_d_state` its state size, `ssm_n_group` its
/// B/C groups, `ssm_dt_rank` its heads.
#[derive(Debug, Clone, PartialEq)]
pub struct NemotronParams {
    /// Per main-pass layer.
    pub blocks: Vec<Block>,
    /// Dense FFN size per layer (`feed_forward_length`; 0 for non-FFN layers).
    pub n_ff: Vec<i64>,
    /// Shared expert size (MoE).
    pub n_ff_shexp: i64,
    /// Renormalize the selected experts' weights to sum 1.
    pub weights_norm: bool,
    /// Multiplier of the routed experts' weights (`routed_scaling_factor`).
    pub weights_scale: f32,
    /// Epsilon of the MTP head's final layer norm.
    pub norm_eps: f32,
}

/// Reads a per-layer key that is either one number for every layer or an array.
fn per_layer(g: &GgufFile, key: &str, n: usize) -> Result<Vec<u64>, ModelError> {
    let v = g.get(key).ok_or_else(|| ModelError::Metadata(key.into()))?;
    match v.as_array() {
        Some(a) => {
            let out: Option<Vec<u64>> = a.iter().map(|x| x.as_u64()).collect();
            let out = out.ok_or_else(|| ModelError::Metadata(key.into()))?;
            if out.len() < n {
                return Err(ModelError::Metadata(key.into()));
            }
            Ok(out)
        }
        None => {
            let x = v.as_u64().ok_or_else(|| ModelError::Metadata(key.into()))?;
            Ok(vec![x; n])
        }
    }
}

/// [`Hparams`] of a `nemotron_h` / `nemotron_h_moe` GGUF.
pub(super) fn parse(g: &GgufFile, arch: &str) -> Result<Hparams, ModelError> {
    let moe = arch == "nemotron_h_moe";
    let key = |k: &str| format!("{arch}.{k}");
    let u = |k: &str| -> Result<u64, ModelError> {
        g.get_u64(&key(k))
            .ok_or_else(|| ModelError::Metadata(key(k)))
    };
    let f = |k: &str| -> Option<f32> { g.get(&key(k)).and_then(|v| v.as_f64()).map(|v| v as f32) };
    let n_all = u("block_count")? as usize;
    let nextn = g.get_u64(&key("nextn_predict_layers")).unwrap_or(0) as usize;
    let n_layer = n_all - nextn;
    let head_kv = per_layer(g, &key("attention.head_count_kv"), n_all)?;
    let n_ff = per_layer(g, &key("feed_forward_length"), n_all)?;
    // llama.cpp's rule (`llama_model_nemotron_h::load_arch_hparams`): a layer is recurrent iff
    // it has neither kv heads nor an FFN; attention iff it has no FFN.
    let blocks: Vec<Block> = (0..n_layer)
        .map(|il| match (head_kv[il], n_ff[il]) {
            (0, 0) => Block::Mamba,
            (_, 0) => Block::Attention,
            _ => Block::Ffn,
        })
        .collect();
    let n_head_kv = head_kv[..n_layer].iter().copied().max().unwrap_or(0) as i64;
    let n_vocab = g
        .tensor("token_embd.weight")
        .map(|t| t.shape[1] as i64)
        .ok_or_else(|| ModelError::Tensor("token_embd.weight".into()))?;
    let norm_eps = f("attention.layer_norm_epsilon");
    let rms_eps = f("attention.layer_norm_rms_epsilon")
        .or(norm_eps)
        .ok_or_else(|| ModelError::Metadata(key("attention.layer_norm_rms_epsilon")))?;
    let flag = |k: &str| g.get(&key(k)).and_then(|v| v.as_bool());
    Ok(Hparams {
        n_layer,
        n_embd: u("embedding_length")? as i64,
        n_head: u("attention.head_count")? as i64,
        n_head_kv,
        head_dim: u("attention.key_length")? as i64,
        // No RoPE: the attention layers are position-free.
        n_rot: 0,
        rope_sections: [0; 4],
        rope_freq_base: f("rope.freq_base").unwrap_or(10_000.0),
        n_ctx_train: u("context_length")? as i32,
        rms_eps,
        moe,
        n_expert: if moe { u("expert_count")? as i64 } else { 0 },
        n_expert_used: if moe {
            u("expert_used_count")? as i64
        } else {
            0
        },
        // Unused: `blocks` says which layers are which.
        full_attention_interval: 1,
        ssm_d_conv: u("ssm.conv_kernel")? as i64,
        ssm_d_state: u("ssm.state_size")? as i64,
        ssm_n_group: u("ssm.group_count")? as i64,
        ssm_dt_rank: u("ssm.time_step_rank")? as i64,
        ssm_d_inner: u("ssm.inner_size")? as i64,
        n_vocab,
        arch: ArchRef::new(NemotronParams {
            blocks,
            n_ff: n_ff[..n_layer].iter().map(|&x| x as i64).collect(),
            n_ff_shexp: g
                .get_u64(&key("expert_shared_feed_forward_length"))
                .unwrap_or(0) as i64,
            weights_norm: flag("expert_weights_norm").unwrap_or(false),
            weights_scale: f("expert_weights_scale").unwrap_or(0.0),
            norm_eps: norm_eps.unwrap_or(rms_eps),
        }),
    })
}

/// The Nemotron-H hyperparameters of `hp`, if it is a Nemotron-H model.
pub fn params(hp: &Hparams) -> Option<&NemotronParams> {
    hp.arch.as_any().downcast_ref::<NemotronParams>()
}

impl Arch for NemotronParams {
    fn name(&self) -> &'static str {
        "nemotron_h"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn same(&self, other: &dyn Arch) -> bool {
        same_as(self, other)
    }

    fn is_recurrent(&self, _hp: &Hparams, il: usize) -> bool {
        self.blocks[il] == Block::Mamba
    }

    fn is_attention(&self, _hp: &Hparams, il: usize) -> bool {
        self.blocks[il] == Block::Attention
    }

    fn recurrent(&self) -> Recurrent {
        Recurrent::Mamba2
    }

    fn slot_ids(&self) -> bool {
        true
    }

    fn layers<'a>(
        &self,
        m: &'a Model,
        pass: &Pass<'a>,
        x: Tensor<'a>,
        capture: &[usize],
        layers: Range<usize>,
    ) -> Result<(Tensor<'a>, Vec<(usize, Tensor<'a>)>), ModelError> {
        assert!(
            layers == (0..m.hp.n_layer),
            "Nemotron-H runs as one pipeline stage"
        );
        // No RoPE reads the positions; keep them in the graph so the input gets memory.
        pass.graph.borrow_mut().expand(pass.pos);
        m.nemotron_layers(pass, x, capture)
    }

    fn capture_normed(&self) -> bool {
        true
    }

    fn mtp(&self) -> bool {
        true
    }

    fn build_mtp<'a>(
        &self,
        m: &'a Model,
        ctx: &'a Context,
        kv: &'a KvPool,
        rec: &'a StatePool,
        shape: &'a [GroupShape],
    ) -> Result<(Graph<'a>, MtpGraph<'a>), ModelError> {
        m.build_mtp_nemotron(ctx, kv, rec, shape)
    }
}

impl Model {
    /// Every layer of a Nemotron-H stack over the embeddings `x`; returns the last layer's
    /// output (before the final norm) and the outputs of the layers in `capture`.
    #[allow(clippy::type_complexity)]
    fn nemotron_layers<'a>(
        &'a self,
        pass: &Pass<'a>,
        mut x: Tensor<'a>,
        capture: &[usize],
    ) -> Result<(Tensor<'a>, Vec<(usize, Tensor<'a>)>), ModelError> {
        let ctx = pass.ctx;
        let hp = &self.hp;
        let nm = params(hp).expect("a Nemotron-H model");
        let mut hidden = Vec::new();
        for il in 0..hp.n_layer {
            let h = ctx.mul(
                ctx.rms_norm(x, hp.rms_eps),
                self.lw(il, "attn_norm.weight")?,
            );
            pass.tap.tap("attn_norm", il, h);
            let out = match nm.blocks[il] {
                Block::Mamba => self.mamba2(pass, il, h)?,
                Block::Attention => self.nemotron_attention(pass, il, h)?,
                Block::Ffn if hp.moe => self.nemotron_moe(
                    ctx,
                    &pass.tap,
                    &pass.graph,
                    il,
                    h,
                    pass.n,
                    !pass.kv.is_host(),
                )?,
                Block::Ffn => self.nemotron_ffn(ctx, il, h)?,
            };
            x = ctx.add(out, x);
            pass.tap.tap("nemotron_h_block_out", il, x);
            if capture.contains(&il) {
                hidden.push((il, x));
            }
        }
        Ok((x, hidden))
    }

    /// A Mamba-2 mixer (`build_mamba2_layer`): in-projection to z | xBC | dt, causal conv over
    /// xBC with the slot's history, selective scan from the slot's state, `y + D x` gated by
    /// silu(z), grouped RMS norm, out-projection.
    fn mamba2<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        h: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        let (ctx, tap, n) = (pass.ctx, &pass.tap, pass.n);
        let hp = &self.hp;
        let d_inner = hp.ssm_d_inner;
        let d_state = hp.ssm_d_state;
        let n_head = hp.ssm_dt_rank;
        let head_dim = d_inner / n_head;
        let n_group = hp.ssm_n_group;
        let conv_dim = d_inner + 2 * n_group * d_state;
        let blk = |name: &str| format!("blk.{il}.{name}");

        // {n_embd, d_in_proj} @ {n_embd, n} => {d_in_proj, n}
        let zxbcdt = self.lmm(ctx, il, "ssm_in.weight", h)?;
        let d_in_proj = zxbcdt.ne()[0];
        let es = zxbcdt.element_size();
        let row = zxbcdt.nb()[1];
        let conv_w = self.lw(il, "ssm_conv1d.weight")?;
        let conv_b = self.w(&blk("ssm_conv1d.bias")).ok();
        let dt_b = self.lw(il, "ssm_dt.bias")?;
        let a = self.lw(il, "ssm_a")?;
        let d = self.lw(il, "ssm_d")?;

        let mut ys = Vec::new();
        for (off, sh) in pass.groups() {
            let (nt, ns) = (sh.n_tok as i64, sh.n_seqs as i64);
            let ng = nt * ns;
            let g = ctx.reshape_3d(cols(ctx, zxbcdt, off, ng), d_in_proj, nt, ns);
            let xbc = ctx.view_3d(
                g,
                [conv_dim, nt, ns],
                [row, row * nt as usize],
                d_inner as usize * es,
            );
            let dt = ctx.view_3d(
                g,
                [n_head, nt, ns],
                [row, row * nt as usize],
                (2 * d_inner + 2 * n_group * d_state) as usize * es,
            );

            // {d_conv - 1 + nt, conv_dim, ns}; the last d_conv - 1 columns are the next step's
            // history.
            let conv_state = pass.rec.conv_slots(ctx, il, sh.slot, sh.n_seqs);
            // The transposed inputs made contiguous first: ggml's CUDA concat of a strided
            // source is ~10x slower than the copy plus a contiguous concat (nsys, 16 sequences).
            let conv_x = ctx.concat(conv_state, ctx.cont(ctx.transpose(xbc)), 0);
            let cnb = conv_x.nb();
            let history = ctx.view_3d(
                conv_x,
                conv_state.ne()[..3].try_into().unwrap(),
                [cnb[1], cnb[2]],
                nt as usize * cnb[0],
            );
            if !sh.verify {
                pass.graph.borrow_mut().expand(ctx.cpy(history, conv_state));
            }
            let mut xbc = ctx.ssm_conv(conv_x, conv_w);
            if let Some(b) = conv_b {
                xbc = ctx.add(xbc, b);
            }
            let xbc = ctx.silu(xbc);
            tap.tap("conv_output_silu", il, xbc);

            let xs = xbc.element_size();
            let xnb = xbc.nb();
            let x = ctx.view_4d(
                xbc,
                [head_dim, n_head, nt, ns],
                [head_dim as usize * xs, xnb[1], xnb[2]],
                0,
            );
            let b = ctx.view_4d(
                xbc,
                [d_state, n_group, nt, ns],
                [d_state as usize * xs, xnb[1], xnb[2]],
                d_inner as usize * xs,
            );
            let c = ctx.view_4d(
                xbc,
                [d_state, n_group, nt, ns],
                [d_state as usize * xs, xnb[1], xnb[2]],
                (d_inner + n_group * d_state) as usize * xs,
            );
            let dt = ctx.add(ctx.cont(dt), dt_b);

            let state = pass.rec.ssm_slots(ctx, il, sh.slot, sh.n_seqs);
            let ids = pass.seq_ids(ctx, sh.n_seqs);
            let y_ssm = ctx.ssm_scan(state, x, dt, a, b, c, ids, 1);
            let ys_es = y_ssm.element_size();
            let x_elems = (head_dim * n_head * ng) as usize;
            if sh.verify {
                let spec = pass.spec.expect("checked in build");
                spec.store_mamba(
                    ctx,
                    &mut pass.graph.borrow_mut(),
                    il,
                    sh.slot,
                    conv_x,
                    [x, dt, b, c],
                );
            } else {
                // A decode step's states land in place; others are copied back.
                let in_place =
                    sh.slot == 0 && sh.n_tok == 1 && pass.rec.decode_output(il, y_ssm, ng as usize);
                if !in_place {
                    let sn = state.ne();
                    let final_state = ctx.view_4d(
                        y_ssm,
                        sn,
                        [
                            sn[0] as usize * ys_es,
                            (sn[0] * sn[1]) as usize * ys_es,
                            (sn[0] * sn[1] * sn[2]) as usize * ys_es,
                        ],
                        x_elems * ys_es,
                    );
                    pass.graph.borrow_mut().expand(ctx.cpy(final_state, state));
                }
            }
            let hd = head_dim as usize * ys_es;
            let y = ctx.view_4d(
                y_ssm,
                [head_dim, n_head, nt, ns],
                [hd, hd * n_head as usize, hd * (n_head * nt) as usize],
                0,
            );
            let y = ctx.add(y, ctx.mul(x, d));
            tap.tap("mamba2_y_add_d", il, y);
            ys.push(ctx.reshape_2d(y, d_inner, ng));
        }
        let y = concat_cols(ctx, ys);
        let z = ctx.view_2d(zxbcdt, [d_inner, n], row, 0);
        let y = ctx.swiglu_split(ctx.cont(z), y);
        // Grouped RMS norm.
        let y = ctx.reshape_3d(y, d_inner / n_group, n_group, n);
        let y = ctx.mul(ctx.rms_norm(y, hp.rms_eps), self.lw(il, "ssm_norm.weight")?);
        let y = ctx.reshape_2d(y, d_inner, n);
        let out = self.lmm(ctx, il, "ssm_out.weight", y)?;
        tap.tap("mamba_out", il, out);
        Ok(out)
    }

    /// Attention without positional encoding (`build_attention_layer`).
    fn nemotron_attention<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        h: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        let (ctx, tap, n) = (pass.ctx, &pass.tap, pass.n);
        let hp = &self.hp;
        let d = hp.head_dim;
        let q = ctx.reshape_3d(self.lmm(ctx, il, "attn_q.weight", h)?, d, hp.n_head, n);
        let k = ctx.reshape_3d(self.lmm(ctx, il, "attn_k.weight", h)?, d, hp.n_head_kv, n);
        let v = ctx.reshape_3d(self.lmm(ctx, il, "attn_v.weight", h)?, d, hp.n_head_kv, n);
        tap.tap("Qcur", il, q);
        tap.tap("Kcur", il, k);
        tap.tap("Vcur", il, v);
        pass.kv
            .write(ctx, &mut pass.graph.borrow_mut(), il, k, v, pass.cells);
        let o = self.attend(pass, il, q, None)?;
        let out = self.lmm(ctx, il, "attn_output.weight", o)?;
        tap.tap("attn_out", il, out);
        Ok(out)
    }

    /// Routed experts plus the shared expert, both squared ReLU (`build_ffn_layer`'s MoE branch
    /// and `build_moe_ffn` with sigmoid gating and a selection bias).
    ///
    /// `fold`: apply the down experts' per-expert NVFP4 scale to the routing weights instead of
    /// to the expert outputs (`down(a) · s · w = down(a) · (s · w)`): the same value up to
    /// rounding, without an elementwise pass over `[n_embd, n_used, n]` per layer. Only the down
    /// scale: unscaled up outputs, squared, overflow the F16 scales of the next matmul's 8-bit
    /// activations. Off where results must match llama.cpp bit for bit.
    #[allow(clippy::too_many_arguments)]
    fn nemotron_moe<'a>(
        &'a self,
        ctx: &'a Context,
        tap: &Tap<'a>,
        graph: &std::cell::RefCell<Graph<'a>>,
        il: usize,
        h: Tensor<'a>,
        n: i64,
        fold: bool,
    ) -> Result<Tensor<'a>, ModelError> {
        let hp = &self.hp;
        let nm = params(hp).expect("a Nemotron-H model");
        let used = hp.n_expert_used;
        let n_expert = hp.n_expert;

        let logits = self.lmm(ctx, il, "ffn_gate_inp.weight", h)?;
        tap.tap("ffn_moe_logits", il, logits);
        let probs = ctx.sigmoid(logits);
        tap.tap("ffn_moe_probs", il, probs);
        let selection = ctx.add(probs, self.lw(il, "exp_probs_b.bias")?);
        let selected = ctx.argsort_top_k(selection, used as i32);
        tap.tap("ffn_moe_topk", il, selected);
        let mut w = ctx.get_rows(ctx.reshape_3d(probs, 1, n_expert, n), selected);
        if nm.weights_norm {
            let w2 = ctx.reshape_2d(w, used, n);
            let sum = ctx.clamp(ctx.sum_rows(w2), 6.103_515_6e-5, f32::INFINITY);
            w = ctx.reshape_3d(ctx.div(w2, sum), 1, used, n);
        }
        if nm.weights_scale != 0.0 && nm.weights_scale != 1.0 {
            w = ctx.scale(w, nm.weights_scale);
        }
        tap.tap("ffn_moe_weights_scaled", il, w);

        let x = ctx.reshape_3d(h, hp.n_embd, 1, n);
        let up_w = format!("blk.{il}.ffn_up_exps.weight");
        let down_w = format!("blk.{il}.ffn_down_exps.weight");
        let up_scale = self.w(&format!("blk.{il}.ffn_up_exps.scale")).ok();
        let down_scale = self.w(&format!("blk.{il}.ffn_down_exps.scale")).ok();
        // Up, relu² and down: one W4A16 pair of ops when the kernels take them (the activation
        // in the up matmul's epilogue), else the graph's.
        let routed = |down_scale: Option<Tensor<'a>>| -> Result<Tensor<'a>, ModelError> {
            let (wu, wd) = (self.w(&up_w)?, self.w(&down_w)?);
            if let Some(down) = self.linear.relu_sqr_ffn_id(
                ctx,
                (&up_w, wu, up_scale),
                (&down_w, wd, down_scale),
                x,
                selected,
            ) {
                return Ok(down);
            }
            let up = self.lmm_id(ctx, il, "ffn_up_exps.weight", x, selected)?;
            let act = ctx.sqr(ctx.relu(up));
            self.linear
                .mul_mat_id(ctx, &down_w, wd, down_scale, act, selected)
        };
        // Prefill batches with 4-bit activations: up, relu², down, the weights and the sum in
        // two ops (the down scale applied per expert in the GEMM).
        let weighted = self.linear.relu_sqr_ffn_id_weighted(
            ctx,
            (&up_w, self.w(&up_w)?, up_scale),
            (&down_w, self.w(&down_w)?, down_scale),
            x,
            selected,
            w,
        );
        let out = if let Some(out) = weighted {
            out
        } else {
            let weighted = match down_scale {
                Some(sd) if fold => {
                    // Per expert s_down, picked per (token, slot) like the weights and folded
                    // into them before the expert matmuls (so the graph after them keeps the
                    // shape ggml's CUDA backend fuses: mul_mat_id, then mul by the weights).
                    let f = ctx.repeat_4d(ctx.reshape_3d(sd, 1, n_expert, 1), [1, n_expert, n, 1]);
                    let w = ctx.mul(w, ctx.get_rows(f, selected));
                    ctx.mul(routed(None)?, w)
                }
                _ => ctx.mul(routed(down_scale)?, w), // [n_embd, used, n]
            };
            let nb = weighted.nb();
            let mut out = ctx.view_2d(weighted, [hp.n_embd, n], nb[2], 0);
            for e in 1..used as usize {
                out = ctx.add(out, ctx.view_2d(weighted, [hp.n_embd, n], nb[2], e * nb[1]));
            }
            // ggml-cuda's one-rounding fusion in every layer (`--stable-fusions`, #472).
            stable_fusions::pin_weighted_sum(graph, weighted, out);
            if used == 1 {
                out = ctx.cont(out);
            }
            out
        };
        tap.tap("ffn_moe_out", il, out);

        let sh = self.relu_sqr_ffn(ctx, il, "ffn_up_shexp", "ffn_down_shexp", h)?;
        tap.tap("ffn_shexp", il, sh);
        let out = ctx.add(out, sh);
        tap.tap("ffn_out", il, out);
        Ok(out)
    }

    /// The dense squared-ReLU MLP of `nemotron_h`.
    fn nemotron_ffn<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        h: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        let blk = |name: &str| format!("blk.{il}.{name}");
        let up_bias = self.w(&blk("ffn_up.bias")).ok();
        let mut out = match up_bias {
            None => self.relu_sqr_ffn(ctx, il, "ffn_up", "ffn_down", h)?,
            Some(b) => {
                let up = ctx.add(self.lmm(ctx, il, "ffn_up.weight", h)?, b);
                let act = ctx.sqr(ctx.relu(up));
                self.lmm(ctx, il, "ffn_down.weight", act)?
            }
        };
        if let Ok(b) = self.w(&blk("ffn_down.bias")) {
            out = ctx.add(out, b);
        }
        Ok(out)
    }

    /// `down @ relu(up @ h)²` for layer `il`'s weights `<up>.weight` and `<down>.weight`: the
    /// fused W4A16 pair where the kernels take it ([`crate::model::core::Linear::relu_sqr_ffn`]),
    /// else the graph's matmuls, relu and sqr.
    fn relu_sqr_ffn<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        up: &str,
        down: &str,
        h: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        let (un, dn) = (
            format!("blk.{il}.{up}.weight"),
            format!("blk.{il}.{down}.weight"),
        );
        let (wu, wd) = (self.w(&un)?, self.w(&dn)?);
        let su = self.w(&format!("blk.{il}.{up}.scale")).ok();
        let sd = self.w(&format!("blk.{il}.{down}.scale")).ok();
        if let Some(out) = self
            .linear
            .relu_sqr_ffn(ctx, (&un, wu, su), (&dn, wd, sd), h)
        {
            return Ok(out);
        }
        let act = ctx.sqr(ctx.relu(self.lmm(ctx, il, &format!("{up}.weight"), h)?));
        self.lmm(ctx, il, &format!("{down}.weight"), act)
    }

    /// One pass of Nemotron-H's MTP head (llama.cpp's `llama_model_nemotron_h_moe::graph_mtp`,
    /// MIT): block `n_layer` joins `enorm(embedding)` and `hnorm(h)` through `eh_proj`, then an
    /// attention sub-layer (its own KV cache) and a MoE sub-layer, each with a residual, and a
    /// final layer norm whose output is both the next draft's `h` and, through the target's LM
    /// head, the draft's logits. `h` at a position is the target's last-layer output after the
    /// final norm (`h_nextn`) at the position before, or the head's own output when drafting on.
    fn build_mtp_nemotron<'a>(
        &'a self,
        ctx: &'a Context,
        kv: &'a KvPool,
        rec: &'a StatePool,
        shape: &'a [GroupShape],
    ) -> Result<(Graph<'a>, MtpGraph<'a>), ModelError> {
        let hp = &self.hp;
        let nm = params(hp).expect("a Nemotron-H model");
        let (e, il) = (hp.n_embd, hp.n_layer);
        let n: i64 = shape.iter().map(|g| g.tokens() as i64).sum();
        let n_out: i64 = shape.iter().map(|g| g.out_rows() as i64).sum();
        let mut inputs = self.new_inputs(ctx, shape, false);
        let h = ctx
            .new_tensor(types::F32, &[e, n])
            .named("mtp_h")
            .as_input();
        let pass = Pass {
            ctx,
            graph: std::cell::RefCell::new(ctx.new_graph(GRAPH_SIZE)),
            tap: Tap {
                ctx,
                on: false,
                taps: Default::default(),
            },
            kv,
            rec,
            saved: None,
            spec: None,
            shape,
            pos: inputs.pos,
            cells: inputs.cells,
            ring_cells: inputs.ring_cells,
            attn: inputs.attn.clone(),
            out_rows: inputs.out_rows,
            ple: None,
            group: Vec::new(),
            seq_ids: None,
            n,
            exclusive: false,
            split: None,
        };
        // No RoPE reads the positions; keep them in the graph so the input gets memory.
        pass.graph.borrow_mut().expand(inputs.pos);
        let w = |name: &str| self.lw(il, name);
        let tok = ctx.get_rows(self.w("token_embd.weight")?, inputs.tokens);
        let hn = ctx.mul(ctx.rms_norm(h, hp.rms_eps), w("nextn.hnorm.weight")?);
        let en = ctx.mul(ctx.rms_norm(tok, hp.rms_eps), w("nextn.enorm.weight")?);
        let cur = self.mm(
            ctx,
            &format!("blk.{il}.nextn.eh_proj.weight"),
            ctx.concat(en, hn, 0),
        )?;

        let x = ctx.mul(ctx.rms_norm(cur, hp.rms_eps), w("attn_norm.weight")?);
        let cur = ctx.add(self.nemotron_attention(&pass, il, x)?, cur);
        let x = ctx.mul(
            ctx.rms_norm(cur, hp.rms_eps),
            w("post_attention_norm.weight")?,
        );
        let cur = ctx.add(
            self.nemotron_moe(ctx, &pass.tap, &pass.graph, il, x, n, !kv.is_host())?,
            cur,
        );
        let cur = ctx.mul(
            ctx.norm(cur, nm.norm_eps),
            w("nextn.shared_head_norm.weight")?,
        );

        let mut graph = pass.graph.into_inner();
        if n_out == 0 {
            graph.expand(cur);
            return Ok((
                graph,
                MtpGraph {
                    inputs,
                    h,
                    pmax: None,
                    h_out: None,
                },
            ));
        }
        let cur = match inputs.out_rows {
            Some(rows) => ctx.get_rows(cur, rows),
            None => cur,
        };
        let h_out = cur.as_output();
        let logits = self.mm(ctx, "output.weight", cur)?;
        let argmax = ctx.argmax(logits).as_output();
        let probs = ctx.soft_max(logits);
        let pmax = ctx.get_rows(
            ctx.reshape_3d(probs, 1, hp.n_vocab, n_out),
            ctx.reshape_2d(argmax, 1, n_out),
        );
        let pmax = ctx.reshape_1d(pmax, n_out).as_output();
        graph.expand(h_out);
        graph.expand(argmax);
        graph.expand(pmax);
        inputs.logits = Some(logits);
        inputs.argmax = Some(argmax);
        Ok((
            graph,
            MtpGraph {
                inputs,
                h,
                pmax: Some(pmax),
                h_out: Some(h_out),
            },
        ))
    }
}
