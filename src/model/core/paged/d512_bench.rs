//! Heads of 512 at 8 query heads a kv head (Gemma 4's global layers) on the warp-pair kernel of
//! [`set_d512_pair`] (#1181) against `paged_prefill_mma`: the op over the same inputs with each
//! kernel choice for the calling thread ([`set_d512_pair_here`]), every output value compared bit
//! for bit, and the time of each; the split over the key blocks (`--paged-verify-split`, other
//! bits) is timed for reference. Used by `legion paged-qm-bench --d512`, which runs them from a
//! release build on the GPU it is given.

use super::qm_bench::{edge_cases, op};
use super::wmma_bench::Group;
use super::*;
use crate::ggml::Backend;
use crate::model::core::KvPool;

/// The kernel choices: `main` (`paged_prefill_mma`), `auto` ([`d512_pair`]'s pick), `pair` (the
/// pair kernel for every group), `split` (over the key blocks); measurement only, wrong output:
/// `h1` (an A fragment loaded every other MMA), `h2` (ldmatrix.x2, half the bytes), `h0` (one load
/// a chunk; ptxas merges the identical MMAs), `h0v` (as h0, every MMA kept); exact but slower:
/// `cols` / `cols2` (`paged_prefill_cols`, 16 query columns a warp, three / two stages).
pub const KERNELS: [&str; 10] = [
    "main", "auto", "pair", "split", "h1", "h2", "h0", "cols", "cols2", "h0v",
];

/// Whether `kernel` runs `g` on the pair kernel on a GPU of `sms` SMs.
pub fn on_pair(g: &Group, kernel: &str, sms: usize) -> bool {
    match kernel {
        "auto" => d512_pair(g.n_tok, g.n_past.len(), g.heads.2, sms),
        "pair" => true,
        _ => false,
    }
}

/// One run of the op for `g` over `kv` with `kernel` ([`KERNELS`]): its output and its
/// milliseconds (best of `rounds` rounds of `reps` runs after the first; 0 without).
pub fn run(
    backend: &mut Backend,
    kv: &KvPool,
    g: &Group,
    kernel: &str,
    reps: usize,
    rounds: usize,
) -> (Vec<f32>, f64) {
    let (d, n_head, n_head_kv) = g.heads;
    let mut scratch = [1, 1, 1, 1];
    let mode = match kernel {
        "main" => 0,
        "auto" => -1,
        "pair" => 1,
        "h1" => 2,
        "h2" => 3,
        "h0" => 4,
        "cols" => 5,
        "cols2" => 6,
        "h0v" => 7,
        "split" => {
            let (d, h, hk) = (d as i64, n_head as i64, n_head_kv as i64);
            scratch = split_scratch_shape(d, h, hk, g.n_tok, g.n_past.len());
            0
        }
        _ => panic!("no kernel {kernel}; one of {KERNELS:?}"),
    };
    set_d512_pair_here(Some(mode));
    let out = op(backend, kv, g, scratch, reps, rounds);
    set_d512_pair_here(None);
    out
}

/// The edge cases ([`super::qm_bench::edges`]) at heads of 512, 32 / 4 and 16 / 2 query / kv
/// heads, with `kernel` against `paged_prefill_mma`. Returns (cases, cases the pair kernel ran on
/// a GPU of `sms` SMs, values compared, values that differ), printing each failing case.
pub fn edges(backend: &mut Backend, kernel: &str, sms: usize) -> (usize, usize, usize, usize) {
    edge_cases(backend, 512, &[(32, 4), (16, 2)], |b, kv, g| {
        let (a, _) = run(b, kv, g, "main", 0, 0);
        let (x, _) = run(b, kv, g, kernel, 0, 0);
        assert!(
            a.iter().chain(&x).all(|v| v.is_finite()),
            "non-finite output"
        );
        let differ = a
            .iter()
            .zip(&x)
            .filter(|(p, q)| p.to_bits() != q.to_bits())
            .count();
        (differ, a.len(), on_pair(g, kernel, sms))
    })
}
