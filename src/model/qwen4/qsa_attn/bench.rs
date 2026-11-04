//! The prompt kernels timed at qwen4exp's shape (`legion qsa-bench`, #828), and the synthetic
//! data the tests share with it.

use super::*;
#[cfg(feature = "cuda")]
use crate::ggml::Backend;
use crate::model::core::Hparams;
#[cfg(feature = "cuda")]
use crate::model::core::{KvPool, KvType, KvTypes};

/// A one-layer attention model's hyperparameters (layer 0 an attention layer).
pub(crate) fn hparams(head_dim: i64, n_head: i64, n_head_kv: i64) -> Hparams {
    Hparams {
        n_layer: 1,
        n_embd: head_dim * n_head,
        n_head,
        n_head_kv,
        head_dim,
        n_rot: head_dim as i32,
        rope_sections: [0; 4],
        rope_freq_base: 1e4,
        n_ctx_train: 4096,
        rms_eps: 1e-6,
        moe: false,
        n_expert: 0,
        n_expert_used: 0,
        // Layer 0 is an attention layer.
        full_attention_interval: 1,
        ssm_d_conv: 4,
        ssm_d_state: 16,
        ssm_n_group: 1,
        ssm_dt_rank: 1,
        ssm_d_inner: 16,
        n_vocab: 16,
        arch: crate::model::core::ArchRef::new(crate::model::qwen35::Qwen35),
    }
}

/// Deterministic values in [-1, 1).
pub(crate) fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

pub(crate) fn f16_bytes(v: &[f32]) -> Vec<u8> {
    let mut h = vec![0u16; v.len()];
    // SAFETY: equal lengths.
    unsafe { ffi::ggml_fp32_to_fp16_row(v.as_ptr(), h.as_mut_ptr(), v.len() as i64) };
    h.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

/// FNV-1a over the bits of `v`.
pub(crate) fn bits_hash(v: &[f32]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
        (h ^ x.to_bits() as u64).wrapping_mul(0x100_0000_01b3)
    })
}

/// Each token's pools in [`prompt_ms`]: `n_tok` rows of 512 sorted, distinct pools below the
/// token's own (pools of 4; `n_ctx - n_tok + t` is token `t`'s position). `local`: the tokens
/// of each run of 64 share 448 of their pools (a prompt's neighbours pick mostly the same
/// pools, so their K / V rows meet in L2), the other 64 their own; else every token picks
/// pseudo-random pools of the whole context.
#[cfg(feature = "cuda")]
pub(crate) fn prompt_pools(n_ctx: usize, n_tok: usize, local: bool) -> Vec<Vec<usize>> {
    let r = noise(n_tok * 512, 4);
    let shared = noise(n_tok.div_ceil(64) * 448, 5);
    (0..n_tok)
        .map(|t| {
            let below = (n_ctx - n_tok + t) / 4;
            let pick = |x: f32| ((x.abs() * below as f32) as usize).min(below - 1);
            let mut pools: Vec<usize> = if local {
                let g = t / 64 * 448;
                let first = (n_ctx - n_tok + t / 64 * 64) / 4;
                (0..448)
                    .map(|j| ((shared[g + j].abs() * first as f32) as usize).min(first - 1))
                    .chain((0..64).map(|j| pick(r[t * 512 + j])))
                    .collect()
            } else {
                (0..512).map(|j| pick(r[t * 512 + j])).collect()
            };
            pools.sort_unstable();
            pools.dedup();
            pools
        })
        .collect()
}

/// One kernel's time in [`prompt_ms`].
#[cfg(feature = "cuda")]
pub struct Timing {
    pub kernel: PromptKernel,
    /// Milliseconds per call: the best round's mean over its `reps` calls.
    pub ms: f64,
    /// [`bits_hash`] of the output: equal for kernels that agree bit for bit.
    pub bits: u64,
}

/// Times [`attention`] on `backend` at qwen4exp's shape (head size 256, 24 heads over 2 kv
/// heads) with caches of `types`: `n_tok` tokens at the end of an `n_ctx` context, every token
/// with the pools of [`prompt_pools`] (2,048 keys or fewer) plus its tail. Each round runs every
/// kernel of `kernels` in turn, one warm-up call and `reps` timed ones. Leaves the launches on
/// [`PromptKernel::Auto`].
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn prompt_ms(
    backend: &mut Backend,
    types: KvTypes,
    n_ctx: usize,
    n_tok: usize,
    reps: usize,
    rounds: usize,
    local: bool,
    kernels: &[PromptKernel],
) -> Vec<Timing> {
    use crate::ggml::ComputeAllocator;
    let n_blocks = n_ctx / KV_BLOCK + 1;
    let hp = hparams(256, 24, 2);
    let kv = KvPool::new_ring(&hp, backend.buffer_type(), n_blocks, &[0], None, types).unwrap();
    let (kp, vp) = kv.pools(0);
    let fill = |t: Tensor<'_>, ty: KvType, seed: u64| {
        let x = noise(t.elements(), seed);
        if ty == KvType::F16 {
            t.set_bytes(&f16_bytes(&x));
        } else {
            t.set_bytes(&crate::ggml::quantize(ty.ggml(), &x, t.ne()[0] as usize));
        }
    };
    fill(kp, types.k, 1);
    fill(vp, types.v, 2);
    let ctx = Context::new(16, 16);
    let q = ctx
        .new_tensor(types::F32, &[256, 24, n_tok as i64, 1])
        .as_input();
    let table = ctx.new_tensor(types::I32, &[n_blocks as i64, 1]).as_input();
    let cap = 512 * 4 + 3;
    let sel = ctx
        .new_tensor(types::I32, &[1 + cap as i64, n_tok as i64])
        .as_input();
    let att = attention(&ctx, q, kp, vp, table, sel).as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(att);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    q.set_f32(&noise(q.elements(), 3));
    table.set_i32(&(0..n_blocks as i32).collect::<Vec<_>>());
    // Every token: its pools' members, then its tail.
    let mut rows = Vec::with_capacity(n_tok * (1 + cap));
    for (t, pools) in prompt_pools(n_ctx, n_tok, local).iter().enumerate() {
        let p = n_ctx - n_tok + t;
        let mut row: Vec<i32> = pools
            .iter()
            .flat_map(|&b| (0..4).map(move |k| (b * 4 + k) as i32))
            .collect();
        row.extend(((p + 1) / 4 * 4..=p).map(|x| x as i32));
        let n = row.len();
        rows.push(n as i32);
        rows.extend(row);
        rows.resize(rows.len() + cap - n, -1);
    }
    sel.set_i32(&rows);
    let mut out: Vec<Timing> = kernels
        .iter()
        .map(|&kernel| Timing {
            kernel,
            ms: f64::INFINITY,
            bits: 0,
        })
        .collect();
    for _ in 0..rounds.max(1) {
        for o in out.iter_mut() {
            set_prompt_kernel(o.kernel);
            backend.compute(&graph).unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                backend.compute(&graph).unwrap();
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3 / reps.max(1) as f64;
            o.ms = o.ms.min(ms);
            o.bits = bits_hash(&att.to_f32());
        }
    }
    set_prompt_kernel(PromptKernel::Auto);
    out
}
