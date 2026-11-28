//! Heads of 256 at 6 query heads a kv head (Qwen3.8-27B's attention layers) on the wide-grid
//! kernels of [`verify_grid_kind`] (#689) against `paged_prefill_mma`: the op over the same
//! inputs with each kernel choice for the calling thread ([`set_verify_grid_here`]), every output
//! value compared bit for bit, and the time of each; the split over the key blocks
//! (`--paged-verify-split`, other bits) is timed for reference, and so is the split at fixed
//! places (`--paged-verify-split-keys`, #1133) on [`verify_grid_kind`]'s pick and on
//! `paged_prefill_mma`, which must give each other's bits. Used by `legion paged-qm-bench
//! --grid`, which runs them from a release build on the GPU it is given.

use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use super::qm_bench::{edge_cases, op};
use super::wmma_bench::Group;
use super::*;
use crate::ggml::Backend;
use crate::model::core::KvPool;

/// The kernel choices: `main` (`paged_prefill_mma`), `auto` ([`verify_grid_kind`]'s pick),
/// `pair` / `quad` (the pair / quad kernel for every group they cover), `split` (over the key
/// blocks), `quad0` (the quad kernel without its page prefetch), `fixed` / `fixedmma` (split at
/// fixed places, every [`set_split_keys`] keys, on `auto`'s pick / on `paged_prefill_mma`).
pub const KERNELS: [&str; 8] = [
    "main", "auto", "pair", "quad", "split", "quad0", "fixed", "fixedmma",
];

/// Keys per run of the `fixed` kernels ([`set_split_keys`]).
static SPLIT_KEYS: AtomicUsize = AtomicUsize::new(16384);

/// Keys per run of the `fixed` / `fixedmma` kernels (a multiple of 64; 16384 unless set).
pub fn set_split_keys(keys: usize) {
    SPLIT_KEYS.store(keys, AtomicOrdering::Relaxed);
}

/// Whether `kernel` splits the keys (other bits than `main`'s where a row's keys span more
/// than one part).
pub fn is_split(kernel: &str) -> bool {
    matches!(kernel, "split" | "fixed" | "fixedmma")
}

/// Whether `kernel` runs `g` on a wide-grid kernel on a GPU of `sms` SMs.
pub fn on_grid(g: &Group, kernel: &str, sms: usize) -> bool {
    match kernel {
        "auto" | "fixed" => verify_grid(g.n_tok, g.n_past.len(), g.heads.2, sms),
        "pair" | "quad" | "quad0" => g.n_tok > 1,
        _ => false,
    }
}

/// The [`set_verify_grid_here`] choice of `kernel` ([`KERNELS`] but `split`; `fixed` as `auto`,
/// `fixedmma` as `main`).
pub fn mode(kernel: &str) -> i32 {
    match kernel {
        "main" | "fixedmma" => 0,
        "auto" | "fixed" => -1,
        "pair" => 1,
        "quad" => 2,
        "quad0" => 3,
        _ => panic!("no kernel {kernel}; one of {KERNELS:?} but split"),
    }
}

/// [`mode`] of `kernel` for the calling thread's launches; `""` follows the default again.
pub fn set_here(kernel: &str) {
    set_verify_grid_here((!kernel.is_empty()).then(|| mode(kernel)));
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
    let fixed = matches!(kernel, "fixed" | "fixedmma");
    let mode = match kernel {
        "split" => {
            let (d, h, hk) = (d as i64, n_head as i64, n_head_kv as i64);
            scratch = split_scratch_shape(d, h, hk, g.n_tok, g.n_past.len());
            0
        }
        k => mode(k),
    };
    if fixed {
        set_verify_split_keys_here(Some(SPLIT_KEYS.load(AtomicOrdering::Relaxed)));
        let (d, h) = (d as i64, n_head as i64);
        scratch = fixed_split_shape(d, h, g.n_tok, g.n_past.len(), g.width).unwrap_or(scratch);
    }
    set_verify_grid_here(Some(mode));
    let out = op(backend, kv, g, scratch, reps, rounds);
    set_verify_grid_here(None);
    set_verify_split_keys_here(None);
    out
}

/// The fixed split's checks for `kernel`'s output `x` of `g` against `main`'s `a` (any other
/// kernel: none): every row whose visible keys lie in one run has `main`'s bits, the other
/// fixed kernel gives `x`'s bits, and the first sequence alone gives its rows of `x`. Returns
/// the values that differ, printing each failed check.
pub fn fixed_check(
    backend: &mut Backend,
    kv: &KvPool,
    g: &Group,
    kernel: &str,
    a: &[f32],
    x: &[f32],
) -> usize {
    let other = match kernel {
        "fixed" => "fixedmma",
        "fixedmma" => "fixed",
        _ => return 0,
    };
    let f = SPLIT_KEYS.load(AtomicOrdering::Relaxed);
    let row = g.heads.0 * g.heads.1;
    let first = |pos: usize| match g.window {
        0 => 0,
        w => (pos + 1).saturating_sub(w),
    };
    let differ = |p: &[f32], q: &[f32]| {
        p.iter()
            .zip(q)
            .filter(|(u, v)| u.to_bits() != v.to_bits())
            .count()
    };
    let mut bad = 0;
    // Rows within one run: main's bits.
    let mut one_run = 0;
    for (i, (p, q)) in a.chunks(row).zip(x.chunks(row)).enumerate() {
        let (s, t) = (i / g.n_tok, i % g.n_tok);
        let pos = g.n_past[s] + t;
        if first(pos) / f == pos / f {
            one_run += differ(p, q);
        }
    }
    if one_run > 0 {
        println!("  {kernel}: {one_run} values of rows within one run differ from main");
    }
    bad += one_run;
    let (y, _) = run(backend, kv, g, other, 0, 0);
    let pair = differ(x, &y);
    if pair > 0 {
        println!("  {kernel}: {pair} values differ from {other}");
    }
    bad += pair;
    if g.n_past.len() > 1 {
        let one = Group {
            n_past: &g.n_past[..1],
            tables: &g.tables[..1],
            ..*g
        };
        let (z, _) = run(backend, kv, &one, kernel, 0, 0);
        let alone = differ(&x[..z.len()], &z);
        if alone > 0 {
            println!("  {kernel}: {alone} values of sequence 0 differ from it alone");
        }
        bad += alone;
    }
    bad
}

/// The edge cases ([`super::qm_bench::edges`]) at heads of 256, 24 / 4, 12 / 2 and 6 / 1 query /
/// kv heads, with `kernel` against `paged_prefill_mma`. Returns (cases, cases the wide-grid kernel
/// ran on a GPU of `sms` SMs, values compared, values that differ), printing each failing case.
/// The fixed split's kernels are checked against `fixedmma` instead (the same split's bits).
pub fn edges(backend: &mut Backend, kernel: &str, sms: usize) -> (usize, usize, usize, usize) {
    let base = if kernel == "fixed" {
        "fixedmma"
    } else {
        "main"
    };
    edge_cases(backend, 256, &[(24, 4), (12, 2), (6, 1)], |b, kv, g| {
        let (a, _) = run(b, kv, g, base, 0, 0);
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
