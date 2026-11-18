//! Heads of 256 at 6 query heads a kv head (Qwen3.8-27B's attention layers) on the wide-grid
//! kernel of [`verify_grid`] (#689) against `paged_prefill_mma`: the op over the same inputs with
//! each kernel choice for the calling thread ([`set_verify_grid_here`]), every output value
//! compared bit for bit, and the time of each; the split over the key blocks
//! (`--paged-verify-split`, other bits) is timed for reference. Used by `legion paged-qm-bench
//! --grid`, which runs them from a release build on the GPU it is given.

use super::qm_bench::{edge_cases, op};
use super::wmma_bench::Group;
use super::*;
use crate::ggml::Backend;
use crate::model::core::KvPool;

/// The kernel choices: `main` (`paged_prefill_mma`), `auto` ([`verify_grid`]'s pick), `grid`
/// (the wide-grid kernel for every group it covers), `split` (over the key blocks).
pub const KERNELS: [&str; 7] = ["main", "auto", "grid", "q1", "q1r", "q2r", "split"];

/// Whether `kernel` runs `g` on the wide-grid kernel on a GPU of `sms` SMs.
pub fn on_grid(g: &Group, kernel: &str, sms: usize) -> bool {
    match kernel {
        "auto" => verify_grid(g.n_tok, g.n_past.len(), g.heads.2, sms),
        "grid" | "q1" | "q1r" | "q2r" => g.n_tok > 1,
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
        "grid" => 1,
        "q1" => 2,
        "q1r" => 3,
        "q2r" => 4,
        "split" => {
            let (d, h, hk) = (d as i64, n_head as i64, n_head_kv as i64);
            scratch = split_scratch_shape(d, h, hk, g.n_tok, g.n_past.len());
            0
        }
        _ => panic!("no kernel {kernel}; one of {KERNELS:?}"),
    };
    set_verify_grid_here(Some(mode));
    let out = op(backend, kv, g, scratch, reps, rounds);
    set_verify_grid_here(None);
    out
}

/// The edge cases ([`super::qm_bench::edges`]) at heads of 256, 24 / 4, 12 / 2 and 6 / 1 query /
/// kv heads, with `kernel` against `paged_prefill_mma`. Returns (cases, cases the wide-grid kernel
/// ran on a GPU of `sms` SMs, values compared, values that differ), printing each failing case.
pub fn edges(backend: &mut Backend, kernel: &str, sms: usize) -> (usize, usize, usize, usize) {
    edge_cases(backend, 256, &[(24, 4), (12, 2), (6, 1)], |b, kv, g| {
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
        (differ, a.len(), on_grid(g, kernel, sms))
    })
}
