//! GLM's sparse MLA prompt kernels timed at Maya-S's shape (`legion mla-bench`, #862): 64 heads
//! of 512 latent values, every token attending to the members of 512 pools of 4 below it (top-k
//! 2,048) plus its own pool's tail.

use super::*;
use crate::ggml::{Backend, ComputeAllocator, Graph};
use crate::model::core::KV_BLOCK;

const N_HEAD: i64 = 64;
const D: i64 = 512;
const KPOOL: usize = 4;
const TOP_POOLS: usize = 512;
/// Positions a selection row holds: the pools' members and a tail.
pub const CAP: usize = TOP_POOLS * KPOOL + KPOOL - 1;

/// Deterministic values in [-1, 1).
pub fn noise(n: usize, seed: u64) -> Vec<f32> {
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

/// `v` as F16 bytes (ggml's rounding).
pub fn f16_bytes(v: &[f32]) -> Vec<u8> {
    let mut h = vec![0u16; v.len()];
    // SAFETY: equal lengths.
    unsafe { ffi::ggml_fp32_to_fp16_row(v.as_ptr(), h.as_mut_ptr(), v.len() as i64) };
    h.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

/// FNV-1a over the bits of `v`.
pub fn bits_hash(v: &[f32]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
        (h ^ x.to_bits() as u64).wrapping_mul(0x100_0000_01b3)
    })
}

/// The selection rows (I32 `[1 + CAP, n_tok]`: count, positions, -1s) of `n_tok` tokens at the
/// end of an `n_ctx` context: each token's whole pools up to its position (all of them up to 512, else 512
/// sorted distinct ones), their members, then its own pool's tail. `local`: the tokens of each
/// run of 64 share 448 of their pools (a prompt's neighbours pick mostly the same pools), the
/// other 64 their own; else every token picks pseudo-random pools of the whole context.
pub fn selections(n_ctx: usize, n_tok: usize, local: bool) -> Vec<i32> {
    let r = noise(n_tok * TOP_POOLS, 4);
    let shared = noise(n_tok.div_ceil(64) * 448, 5);
    let mut rows = Vec::with_capacity(n_tok * (1 + CAP));
    for t in 0..n_tok {
        let p = n_ctx - n_tok + t;
        let below = (p + 1) / KPOOL;
        let mut pools: Vec<usize> = if below <= TOP_POOLS {
            (0..below).collect()
        } else {
            let pick = |x: f32, n: usize| ((x.abs() * n as f32) as usize).min(n - 1);
            let first = (n_ctx - n_tok + t / 64 * 64) / KPOOL;
            if local && first > TOP_POOLS {
                let g = t / 64 * 448;
                (0..448)
                    .map(|j| pick(shared[g + j], first))
                    .chain((0..64).map(|j| pick(r[t * TOP_POOLS + j], below)))
                    .collect()
            } else {
                (0..TOP_POOLS)
                    .map(|j| pick(r[t * TOP_POOLS + j], below))
                    .collect()
            }
        };
        pools.sort_unstable();
        pools.dedup();
        let mut row: Vec<i32> = pools
            .iter()
            .flat_map(|&b| (0..KPOOL).map(move |k| (b * KPOOL + k) as i32))
            .collect();
        row.extend((below * KPOOL..=p).map(|x| x as i32));
        let n = row.len();
        rows.push(n as i32);
        rows.extend(row);
        rows.resize(rows.len() + CAP - n, -1);
    }
    rows
}

/// One kernel's time in [`prompt_ms`].
pub struct Timing {
    pub kernel: MlaKernel,
    /// Milliseconds per call: the best round's mean over its `reps` calls.
    pub ms: f64,
    /// [`bits_hash`] of the output: equal for kernels that agree bit for bit.
    pub bits: u64,
}

/// Times [`mla_attn_prompt`] on `backend` at Maya-S's shape: `n_tok` tokens at the end of an
/// `n_ctx` context with the selections of [`selections`], the latents in an F16 cache whose
/// blocks the table lists in reverse. Each round runs every kernel of `kernels` in turn, one
/// warm-up call and `reps` timed ones.
pub fn prompt_ms(
    backend: &mut Backend,
    n_ctx: usize,
    n_tok: usize,
    reps: usize,
    rounds: usize,
    local: bool,
    kernels: &[MlaKernel],
) -> Vec<Timing> {
    let n_blocks = n_ctx.div_ceil(KV_BLOCK);
    let data = Context::new(8, 0);
    let q = data.new_tensor(types::F32, &[D, N_HEAD, n_tok as i64]);
    let cache = data.new_tensor(types::F16, &[D, (n_blocks * KV_BLOCK) as i64]);
    let table = data.new_tensor(types::I32, &[n_blocks as i64, 1]);
    let sel = data.new_tensor(types::I32, &[1 + CAP as i64, n_tok as i64]);
    let after = data.new_tensor(types::F32, &[1]);
    data.alloc_tensors(backend.buffer_type()).unwrap();
    q.set_f32(&noise(q.elements(), 3));
    cache.set_bytes(&f16_bytes(&noise(cache.elements(), 1)));
    table.set_i32(&(0..n_blocks as i32).rev().collect::<Vec<_>>());
    sel.set_i32(&selections(n_ctx, n_tok, local));
    after.set_f32(&[0.0]);
    let scale = 1.0 / 192f32.sqrt();
    let ctxs: Vec<Context> = kernels.iter().map(|_| Context::new(4, 16)).collect();
    let outs: Vec<(Tensor<'_>, Graph<'_>)> = ctxs
        .iter()
        .zip(kernels)
        .map(|(ctx, &kernel)| {
            let srcs = [q, cache, table, sel, after];
            let out = mla_attn_prompt(ctx, srcs, n_tok as i64, scale, kernel).as_output();
            let mut graph = ctx.new_graph(16);
            graph.expand(out);
            ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
                .unwrap();
            (out, graph)
        })
        .collect();
    let mut timings: Vec<Timing> = kernels
        .iter()
        .map(|&kernel| Timing {
            kernel,
            ms: f64::INFINITY,
            bits: 0,
        })
        .collect();
    for _ in 0..rounds.max(1) {
        for ((out, graph), timing) in outs.iter().zip(timings.iter_mut()) {
            backend.compute(graph).unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                backend.compute(graph).unwrap();
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3 / reps.max(1) as f64;
            timing.ms = timing.ms.min(ms);
            timing.bits = bits_hash(&out.to_f32());
        }
    }
    timings
}
