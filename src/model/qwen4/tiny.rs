//! A tiny `qwen4exp` (Qwen3.8-Flash-Next layout): hyper-connections (4 streams), the
//! per-layer n-gram embedding in layer 1, sparse attention (pools of 4, a budget of 2 pools,
//! so attention drops keys past position 11) in the attention layers, a 16-expert MoE, and the
//! byte-level tokenizer. With `mtp`, also its MTP head as a GGUF of its own, as the released
//! model ships it: block `n_layer` (dense attention, MoE) plus the `nextn.*` tensors, sharing
//! the target's embeddings and output head.

use std::path::Path;

use crate::engine::sampling::Rng;
use crate::ggml::{self, ffi, types};
use crate::gguf::{GgufError, GgufWriter, Value};
use crate::model::qwen35::tiny::*;

const ARCH: &str = "qwen4exp";
pub const N_LAYER: usize = 4;
const E: i64 = 64;
const HC: i64 = 4;
const HC_RANK: i64 = 32;
const N_HEAD: i64 = 4;
const N_HEAD_KV: i64 = 2;
const HEAD_DIM: i64 = 32;
const SSM_STATE: i64 = 16;
const SSM_GROUPS: i64 = 2;
const SSM_HEADS: i64 = 4;
const N_EXPERT: i64 = 8;
const N_EXPERT_USED: i64 = 2;
const N_FF: i64 = 32;
const PLE_LAYER: usize = 1;
const PLE_DIM: i64 = 32;
const PLE_HEADS: usize = 4;
const IDX_DIM: i64 = 16;
const KPOOL: i32 = 4;
const TOP_K: u32 = 8;

struct Builder {
    g: GgufWriter,
    rng: Rng,
    /// The indexer's key size and query heads.
    idx: (i64, u32),
    /// The attention heads' size.
    head_dim: i64,
    /// The experts' (and the shared expert's) FFN rows.
    n_ff: i64,
    /// The DeltaNet heads' size (keys and values).
    ssm_state: i64,
}

/// The indexer shape of [`write`]: small.
const IDX_SMALL: (i64, u32) = (IDX_DIM, 2);
/// The indexer shape of [`write_cuda_indexer`]: the one ggml-cuda's lightning indexer takes
/// (key size 128, 4 heads), so the model also runs where that op runs.
const IDX_CUDA: (i64, u32) = (128, 4);
/// [`write_cuda_indexer`]'s attention head size: one the paged sparse-attention kernels take.
const HEAD_DIM_CUDA: i64 = 128;

impl Builder {
    fn key(&mut self, k: &str, v: Value) -> Result<(), GgufError> {
        self.g.set(&format!("{ARCH}.{k}"), &v)?;
        Ok(())
    }

    /// `bias + scale * U(-1, 1)`, stored as `ty`.
    fn tensor(
        &mut self,
        name: &str,
        ty: ffi::ggml_type,
        shape: &[i64],
        scale: f32,
        bias: f32,
    ) -> Result<(), GgufError> {
        let n: i64 = shape.iter().product();
        let v: Vec<f32> = (0..n)
            .map(|_| bias + scale * (self.rng.next_f64() as f32 * 2.0 - 1.0))
            .collect();
        let bytes = if ty == types::F32 {
            v.iter().flat_map(|x| x.to_ne_bytes()).collect()
        } else {
            ggml::quantize(ty, &v, shape[0] as usize)
        };
        self.g.add_tensor(name, ty, shape, &bytes)?;
        Ok(())
    }

    /// A weight `[n_in, n_out]` scaled for unit-variance outputs.
    fn weight(&mut self, name: &str, ty: ffi::ggml_type, shape: &[i64]) -> Result<(), GgufError> {
        let s = (3.0 / shape[0] as f32).sqrt();
        self.tensor(name, ty, shape, s, 0.0)
    }

    fn norm(&mut self, name: &str, n: i64) -> Result<(), GgufError> {
        self.tensor(name, types::F32, &[n], 0.1, 1.0)
    }

    fn hc_block(&mut self, prefix: &str, inject: bool) -> Result<(), GgufError> {
        let hc_dim = HC * E;
        self.norm(&format!("{prefix}_norm.weight"), hc_dim)?;
        self.weight(
            &format!("{prefix}_down.weight"),
            types::Q8_0,
            &[hc_dim, HC_RANK],
        )?;
        self.weight(
            &format!("{prefix}_up.weight"),
            types::Q8_0,
            &[HC_RANK, hc_dim],
        )?;
        if inject {
            self.tensor(
                &format!("{prefix}_inject.weight"),
                types::F32,
                &[hc_dim, HC],
                0.05,
                0.0,
            )?;
        }
        Ok(())
    }

    fn attention(&mut self, il: usize, indexer: bool) -> Result<(), GgufError> {
        let t = |s: &str| format!("blk.{il}.{s}");
        self.weight(
            &t("attn_q.weight"),
            types::Q8_0,
            &[E, 2 * N_HEAD * self.head_dim],
        )?;
        self.weight(
            &t("attn_k.weight"),
            types::Q8_0,
            &[E, N_HEAD_KV * self.head_dim],
        )?;
        self.weight(
            &t("attn_v.weight"),
            types::Q8_0,
            &[E, N_HEAD_KV * self.head_dim],
        )?;
        self.weight(
            &t("attn_output.weight"),
            types::Q8_0,
            &[N_HEAD * self.head_dim, E],
        )?;
        let hd = self.head_dim;
        self.norm(&t("attn_q_norm.weight"), hd)?;
        self.norm(&t("attn_k_norm.weight"), hd)?;
        if indexer {
            let (d, heads) = self.idx;
            self.norm(&t("indexer.k_norm.weight"), d)?;
            self.norm(&t("indexer.q_norm.weight"), d)?;
            self.weight(&t("indexer.k_proj.weight"), types::BF16, &[E, d])?;
            self.weight(
                &t("indexer.q_proj.weight"),
                types::BF16,
                &[E, heads as i64 * d],
            )?;
        }
        Ok(())
    }

    fn moe(&mut self, il: usize) -> Result<(), GgufError> {
        let t = |s: &str| format!("blk.{il}.{s}");
        let ff = self.n_ff;
        self.weight(&t("ffn_gate_inp.weight"), types::F32, &[E, N_EXPERT])?;
        self.weight(&t("ffn_gate_inp_shexp.weight"), types::F32, &[E])?;
        self.weight(&t("ffn_up_exps.weight"), types::Q8_0, &[E, ff, N_EXPERT])?;
        self.weight(&t("ffn_gate_exps.weight"), types::Q8_0, &[E, ff, N_EXPERT])?;
        self.weight(&t("ffn_down_exps.weight"), types::Q8_0, &[ff, E, N_EXPERT])?;
        self.weight(&t("ffn_up_shexp.weight"), types::Q8_0, &[E, ff])?;
        self.weight(&t("ffn_gate_shexp.weight"), types::Q8_0, &[E, ff])?;
        self.weight(&t("ffn_down_shexp.weight"), types::Q8_0, &[ff, E])
    }

    /// The metadata both files carry; `blocks` is `block_count`, `nextn` the MTP layers.
    fn common(&mut self, blocks: usize, nextn: u32) -> Result<(), GgufError> {
        self.g
            .set("general.architecture", &Value::String(ARCH.into()))?;
        self.g
            .set("general.name", &Value::String("tiny-qwen4exp".into()))?;
        byte_tokenizer(&mut self.g)?;
        let u = Value::U32;
        self.key("block_count", u(blocks as u32))?;
        if nextn > 0 {
            self.key("nextn_predict_layers", u(nextn))?;
        }
        self.key("context_length", u(4096))?;
        self.key("embedding_length", u(E as u32))?;
        self.key("attention.head_count", u(N_HEAD as u32))?;
        self.key("attention.head_count_kv", u(N_HEAD_KV as u32))?;
        self.key("attention.key_length", u(self.head_dim as u32))?;
        self.key("attention.value_length", u(self.head_dim as u32))?;
        self.key("rope.dimension_count", u(16))?;
        self.key(
            "rope.dimension_sections",
            Value::Array([3, 3, 2, 0].into_iter().map(Value::I32).collect()),
        )?;
        self.key("rope.freq_base", Value::F32(10_000_000.0))?;
        self.key("attention.layer_norm_rms_epsilon", Value::F32(1e-6))?;
        self.key("expert_count", u(N_EXPERT as u32))?;
        self.key("expert_used_count", u(N_EXPERT_USED as u32))?;
        self.key("expert_feed_forward_length", u(self.n_ff as u32))?;
        self.key("expert_shared_feed_forward_length", u(self.n_ff as u32))?;
        self.key("ssm.conv_kernel", u(4))?;
        self.key("ssm.state_size", u(self.ssm_state as u32))?;
        self.key("ssm.group_count", u(SSM_GROUPS as u32))?;
        self.key("ssm.time_step_rank", u(SSM_HEADS as u32))?;
        self.key("ssm.inner_size", u((self.ssm_state * SSM_HEADS) as u32))?;
        self.key("full_attention_interval", u(4))?;
        self.key("hyper_connection.count", u(HC as u32))?;
        self.key("hyper_connection.low_rank", u(HC_RANK as u32))?;
        self.key("attention.indexer.head_count", u(self.idx.1))?;
        self.key("attention.indexer.key_length", u(self.idx.0 as u32))?;
        self.key("attention.indexer.top_k", u(TOP_K))?;
        // Sparse in the main attention layers, dense in the MTP block (as released).
        self.key(
            "attention.compress_ratios",
            Value::Array(
                (0..blocks)
                    .map(|il| {
                        Value::I32(if il < N_LAYER && (il + 1) % 4 == 0 {
                            KPOOL
                        } else {
                            0
                        })
                    })
                    .collect(),
            ),
        )
    }
}

/// Writes the tiny target model to `path` and, with `mtp`, its MTP head there.
pub fn write(path: impl AsRef<Path>, mtp: Option<&Path>, seed: u64) -> Result<(), GgufError> {
    write_with(path, mtp, seed, IDX_SMALL, HEAD_DIM, false)
}

/// [`write`] with experts of twice the FFN rows and DeltaNet heads of twice the size, so that
/// two ranks of tensor parallelism over machines (#733) each take whole Q8_0 blocks of every
/// expert's down projection and of the DeltaNet output projection.
pub fn write_tp(path: impl AsRef<Path>, mtp: Option<&Path>, seed: u64) -> Result<(), GgufError> {
    write_with(path, mtp, seed, IDX_SMALL, HEAD_DIM, true)
}

/// [`write`] with the sparse-attention indexer in the shape ggml-cuda's lightning indexer
/// supports and attention heads the paged sparse-attention kernels take (128), for tests of
/// the model on a GPU.
pub fn write_cuda_indexer(
    path: impl AsRef<Path>,
    mtp: Option<&Path>,
    seed: u64,
) -> Result<(), GgufError> {
    write_with(path, mtp, seed, IDX_CUDA, HEAD_DIM_CUDA, false)
}

/// `wide`: [`write_tp`]'s shapes.
fn write_with(
    path: impl AsRef<Path>,
    mtp: Option<&Path>,
    seed: u64,
    idx: (i64, u32),
    head_dim: i64,
    wide: bool,
) -> Result<(), GgufError> {
    let (n_ff, ssm_state) = if wide {
        (2 * N_FF, 2 * SSM_STATE)
    } else {
        (N_FF, SSM_STATE)
    };
    let mut b = Builder {
        g: GgufWriter::new(),
        rng: Rng::new(seed),
        idx,
        head_dim,
        n_ff,
        ssm_state,
    };
    b.common(N_LAYER, 0)?;
    let sizes: Vec<u64> = (0..PLE_HEADS as u64).map(|h| 1009 + 2 * h).collect();
    let offsets: Vec<u64> = sizes
        .iter()
        .scan(0, |acc, &s| {
            let o = *acc;
            *acc += s;
            Some(o)
        })
        .collect();
    let ple_rows = (offsets[PLE_HEADS - 1] + sizes[PLE_HEADS - 1]) as i64;
    let u = Value::U32;
    b.key(
        "ple.layers",
        Value::Array(vec![Value::I32(PLE_LAYER as i32)]),
    )?;
    b.key("ple.ngram_size", u(3))?;
    b.key("ple.heads_per_ngram", u(2))?;
    b.key("ple.conv_kernel", u(4))?;
    b.key("ple.eos_token_id", u(EOS))?;
    b.key("embedding_length_per_layer_input", u(PLE_DIM as u32))?;
    b.key(
        "ple.layer_multipliers",
        Value::Array(
            [23703573157769u64, 20109073645365, 8052911324071]
                .into_iter()
                .map(Value::U64)
                .collect(),
        ),
    )?;
    b.key(
        "ple.head_offsets",
        Value::Array(offsets.into_iter().map(Value::U64).collect()),
    )?;
    b.key(
        "ple.head_vocab_sizes",
        Value::Array(sizes.into_iter().map(Value::U64).collect()),
    )?;

    let n_vocab = 256 + SPECIAL_TOKENS.len() as i64;
    let hc_dim = HC * E;
    b.tensor("token_embd.weight", types::Q8_0, &[E, n_vocab], 1.0, 0.0)?;
    b.weight("output.weight", types::Q8_0, &[E, n_vocab])?;
    b.hc_block("output_hc", false)?;
    b.tensor(
        "per_layer_token_embd.weight",
        types::IQ4_NL,
        &[PLE_DIM, ple_rows],
        1.0,
        0.0,
    )?;
    let ss = b.ssm_state;
    let (ks, vs) = (ss * SSM_GROUPS, ss * SSM_HEADS);
    for il in 0..N_LAYER {
        let t = |s: &str| format!("blk.{il}.{s}");
        b.hc_block(&t("hc_attn"), true)?;
        b.hc_block(&t("hc_ffn"), true)?;
        if (il + 1) % 4 != 0 {
            b.weight(&t("attn_qkv.weight"), types::Q8_0, &[E, 2 * ks + vs])?;
            b.weight(&t("attn_gate.weight"), types::Q8_0, &[E, vs])?;
            b.tensor(
                &t("ssm_conv1d.weight"),
                types::F32,
                &[4, 2 * ks + vs],
                0.5,
                0.0,
            )?;
            b.tensor(&t("ssm_dt.bias"), types::F32, &[SSM_HEADS], 0.5, 0.0)?;
            b.tensor(&t("ssm_a"), types::F32, &[SSM_HEADS], 0.5, -1.0)?;
            b.weight(&t("ssm_alpha.weight"), types::F32, &[E, SSM_HEADS])?;
            b.weight(&t("ssm_beta.weight"), types::F32, &[E, SSM_HEADS])?;
            b.norm(&t("ssm_norm.weight"), ss)?;
            b.weight(&t("ssm_out.weight"), types::Q8_0, &[vs, E])?;
        } else {
            b.attention(il, true)?;
        }
        if il == PLE_LAYER {
            let pe = PLE_DIM * PLE_HEADS as i64;
            b.tensor(&t("ple_conv1d.weight"), types::F32, &[4, hc_dim], 0.5, 0.0)?;
            b.weight(&t("ple_key.weight"), types::Q8_0, &[pe, hc_dim])?;
            b.weight(&t("ple_value.weight"), types::Q8_0, &[pe, E])?;
            b.norm(&t("ple_norm_conv.weight"), hc_dim)?;
            b.norm(&t("ple_norm_key.weight"), hc_dim)?;
            b.norm(&t("ple_norm_query.weight"), hc_dim)?;
        }
        b.moe(il)?;
    }
    b.g.write(path)?;

    let Some(mtp) = mtp else {
        return Ok(());
    };
    let mut b = Builder {
        g: GgufWriter::new(),
        rng: Rng::new(seed ^ 0x5eed),
        idx,
        head_dim,
        n_ff,
        ssm_state,
    };
    b.common(N_LAYER + 1, 1)?;
    b.g.set("qwen4exp.nextn_shared_target_tensors", &Value::Bool(true))?;
    let il = N_LAYER;
    let t = |s: &str| format!("blk.{il}.{s}");
    b.hc_block(&t("hc_attn"), true)?;
    b.hc_block(&t("hc_ffn"), true)?;
    b.attention(il, true)?;
    b.moe(il)?;
    b.weight(&t("nextn.eh_proj.weight"), types::Q8_0, &[2 * E, E])?;
    b.norm(&t("nextn.enorm.weight"), E)?;
    b.norm(&t("nextn.hnorm.weight"), hc_dim)?;
    b.hc_block(&t("nextn.hc_head"), false)?;
    b.g.write(mtp)
}
