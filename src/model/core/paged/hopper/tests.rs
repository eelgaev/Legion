//! The Hopper prefill kernel's plan (CPU: shared memory, descriptors, which ops it covers) and,
//! on a GPU of compute capability 9.0 with an sm_90a build only, its output against the f64
//! oracle and against the default kernel (`paged_prefill_mma`) on the same inputs. The GPU tests
//! skip (pass) on any other device: no Hopper GPU ran them yet.

use super::*;

/// The sizes the kernel file asserts (`smem_bytes<128>`, `smem_bytes<256>`), within what a block
/// may opt into.
#[test]
fn smem_plan_matches_the_kernel() {
    assert_eq!(smem_bytes(128), 164_992);
    assert_eq!(smem_bytes(256), 197_696);
    for d in HEAD_SIZES {
        assert!(smem_bytes(d) <= SMEM_LIMIT, "d {d}: {}", smem_bytes(d));
        // Q and every stage start on 1,024-byte swizzle atoms.
        assert_eq!((ROWS * d * 2) % 1024, 0);
        assert_eq!((KV_BLOCK * d * 2) % 1024, 0);
    }
    assert_eq!(stages(128), 4);
    assert_eq!(stages(256), 2);
}

#[test]
fn descriptors() {
    assert_eq!(desc_k_hi(), crate::model::wgmma_gemm::desc_hi());
    let v = desc_v_hi();
    assert_eq!(
        (v >> 16) & 0x3FFF,
        8192 >> 4,
        "LBO: one 64-dim column of 64 keys"
    );
    assert_eq!((v >> 32) & 0x3FFF, 1024 >> 4, "SBO: one 8-key atom");
    assert_eq!((v >> 49) & 0x7, 0, "base offset");
    assert_eq!(v >> 62, 1, "128-byte swizzle");
    assert_eq!(v & 0x3FFF, 0, "the kernel ORs in the address");
}

/// The 128-byte swizzle as the hardware applies it to an address inside a 1,024-byte aligned
/// tile: the 16-byte chunk (bits 4-6) XOR the 128-byte row within the atom (bits 7-9).
fn hw_swizzle(addr: usize) -> usize {
    addr ^ (((addr >> 7) & 7) << 4)
}

/// V's element (key, dim) where the TMA puts it (64-dim columns of 64 keys, a 128-byte
/// swizzled row per key) and where wgmma reads it as element (n = dim, k = key) of `O += P V`
/// behind [`desc_v_hi`] at the kernel's slice start (+2,048 bytes per 16 keys). The MN-major
/// 128-byte canonical layout: 64 values of n contiguous, n blocks of 64 at LBO, 8 rows of k at
/// 128 bytes, k blocks of 8 at SBO, swizzled.
#[test]
fn v_layout_matches_the_mn_major_descriptor() {
    use crate::model::wgmma_gemm::swizzle_128b;
    let lbo = ((desc_v_hi() >> 16) & 0x3FFF) as usize * 16;
    let sbo = ((desc_v_hi() >> 32) & 0x3FFF) as usize * 16;
    for d in HEAD_SIZES {
        for key in 0..KV_BLOCK {
            for dim in 0..d {
                let tma = (dim / 64) * KV_BLOCK * SPAN + swizzle_128b(key, (dim % 64) * 2);
                let (slice, k) = (key / 16, key % 16);
                let addr = slice * 2048
                    + (dim / 64) * lbo
                    + (k / 8) * sbo
                    + (k % 8) * SPAN
                    + (dim % 64) * 2;
                assert_eq!(tma, hw_swizzle(addr), "d {d}, key {key}, dim {dim}");
            }
        }
    }
}

/// Q's rows as the consumers write them (the same layout as the TMA's for K) against where
/// wgmma reads them behind [`desc_k_hi`] at the kernel's slice start (+32 bytes per 16 dims
/// inside a 64-dim column, the column's bytes per 64 dims). The K-major 128-byte canonical
/// layout: a row's 64 values contiguous, 8 rows at 128 bytes, row blocks of 8 at SBO, swizzled.
#[test]
fn qk_layout_matches_the_k_major_descriptor() {
    use crate::model::wgmma_gemm::swizzle_128b;
    let sbo = ((desc_k_hi() >> 32) & 0x3FFF) as usize * 16;
    for (rows, d) in [(ROWS, 128), (ROWS, 256), (KV_BLOCK, 128), (KV_BLOCK, 256)] {
        for row in 0..rows {
            for dim in 0..d {
                // The kernel's Q store: 16-byte chunk u of the row in column u / 8.
                let u = dim / 8;
                let stored = (u / 8) * rows * SPAN
                    + row * SPAN
                    + (((u % 8) ^ (row % 8)) << 4)
                    + (dim % 8) * 2;
                assert_eq!(
                    stored,
                    (dim / 64) * rows * SPAN + swizzle_128b(row, (dim % 64) * 2)
                );
                let slice = dim / 16;
                let start = (slice / 4) * rows * SPAN + (slice % 4) * 32;
                let addr = start + (row / 8) * sbo + (row % 8) * SPAN + (dim % 16) * 2;
                assert_eq!(
                    stored,
                    hw_swizzle(addr),
                    "rows {rows}, row {row}, dim {dim}"
                );
            }
        }
    }
}

#[test]
fn tokens_per_cta_by_gqa() {
    let got: Vec<usize> = [1, 2, 4, 6, 8, 12, 16]
        .iter()
        .map(|&g| tokens_per_cta(g))
        .collect();
    assert_eq!(got, [128, 64, 32, 21, 16, 10, 8]);
}

fn shape() -> Shape {
    Shape {
        d: 256,
        n_head: 24,
        n_head_kv: 4,
        n_tok: 512,
        f16: true,
        compact: false,
        tiered: false,
        max_split: 1,
        k_cell: 256 * 4 * 2,
        v_cell: 256 * 4 * 2,
        k_cells: 64 * 1024,
        v_cells: 64 * 1024,
    }
}

#[test]
fn covers_prefill_on_f16_caches_only() {
    assert!(covers(&shape()));
    assert!(covers(&Shape {
        d: 128,
        n_head: 32,
        n_head_kv: 2,
        k_cell: 128 * 2 * 2,
        v_cell: 128 * 2 * 2,
        ..shape()
    }));
    let off = [
        Shape { d: 512, ..shape() },
        Shape { d: 64, ..shape() },
        Shape {
            n_tok: 1,
            ..shape()
        },
        Shape {
            f16: false,
            ..shape()
        },
        Shape {
            compact: true,
            ..shape()
        },
        Shape {
            tiered: true,
            ..shape()
        },
        Shape {
            max_split: 4,
            ..shape()
        },
        Shape {
            n_head: 25,
            ..shape()
        },
        Shape {
            n_head_kv: 0,
            ..shape()
        },
        Shape {
            k_cell: 256 * 4 * 2 + 8,
            ..shape()
        },
        Shape {
            v_cell: 100,
            ..shape()
        },
        Shape {
            k_cells: 1 << 31,
            ..shape()
        },
    ];
    for s in off {
        assert!(!covers(&s), "{s:?}");
    }
}

/// Off unless asked for: the process setting (`--hopper-attn`) and a thread's own.
#[test]
fn off_by_default() {
    assert!(!enabled());
    set_enabled_here(Some(true));
    assert!(enabled());
    set_enabled_here(None);
    assert!(!enabled());
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::super::super::ring_tests::{ring_case, seams};
    use super::super::super::tests::{
        cuda_long_context, cuda_prefill_edges, f16_bytes, hparams, max_abs_diff, noise,
    };
    use super::super::super::*;
    use super::super::{launches, set_enabled_here};
    use crate::ggml::{Backend, ComputeAllocator, DeviceKind, device_by_kind};
    use crate::model::core::KvPool;

    fn gpu() -> Backend {
        device_by_kind(DeviceKind::Gpu)
            .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
            .expect("no GPU device")
            .init()
            .unwrap()
    }

    /// The kernel for this thread while the guard lives; `None` (skip) without a Hopper GPU
    /// and an sm_90a build.
    struct On;
    impl On {
        fn here() -> Option<On> {
            if !crate::model::wgmma_gemm::supported() {
                eprintln!("skipped: needs a GPU of compute capability 9.0 and an sm_90a build");
                return None;
            }
            set_enabled_here(Some(true));
            Some(On)
        }
    }
    impl Drop for On {
        fn drop(&mut self) {
            set_enabled_here(None);
        }
    }

    /// Several sequences over shuffled blocks, ragged tiles, mid-block windows, GQA 1 to 16 (the
    /// default kernel takes the 512-wide heads).
    #[test]
    #[ignore = "needs a Hopper GPU (compute capability 9.0)"]
    fn hopper_prefill_edges_match_oracle() {
        let _backend = gpu();
        let Some(_on) = On::here() else {
            return;
        };
        let before = launches();
        cuda_prefill_edges();
        assert!(launches() > before, "the Hopper kernel did not run");
    }

    /// 8K keys at heads of 128 and 256, with and without a window.
    #[test]
    #[ignore = "needs a Hopper GPU (compute capability 9.0)"]
    fn hopper_long_context_matches_oracle() {
        let _backend = gpu();
        let Some(_on) = On::here() else {
            return;
        };
        let before = launches();
        for (n_head, n_head_kv) in [(16, 4), (24, 2), (24, 4)] {
            cuda_long_context(256, n_head, n_head_kv, 1.0 / 16.0, 0);
        }
        cuda_long_context(256, 16, 8, 1.0 / 16.0, 1024);
        cuda_long_context(128, 32, 2, 1.0 / 11.3, 0);
        assert!(launches() > before, "the Hopper kernel did not run");
    }

    /// Sliding-window rings, Gemma 4's sliding shape and window.
    #[test]
    #[ignore = "needs a Hopper GPU (compute capability 9.0)"]
    fn hopper_ring_matches_full_cache_and_oracle() {
        let mut backend = gpu();
        let Some(_on) = On::here() else {
            return;
        };
        let before = launches();
        seams(&mut backend, 5e-4);
        for n_past in [[1024, 1536, 2047], [2048, 2560, 3071]] {
            ring_case(&mut backend, 256, 16, 8, 1024, 512, &n_past, 512, 5e-4);
        }
        assert!(launches() > before, "the Hopper kernel did not run");
    }

    /// The op's output with the Hopper kernel on or off (the default, `paged_prefill_mma`) for
    /// the given case; V in [-1, 1).
    #[allow(clippy::too_many_arguments)]
    fn prefill_out(
        backend: &mut Backend,
        hopper: bool,
        d: usize,
        n_head: usize,
        n_head_kv: usize,
        n_tok: usize,
        n_past: &[usize],
        tables: &[Vec<u32>],
        scale: f32,
        window: usize,
    ) -> Vec<f32> {
        set_enabled_here(Some(hopper));
        let hp = hparams(d as i64, n_head as i64, n_head_kv as i64);
        let n_blocks = tables.iter().flatten().max().unwrap() + 1;
        let kv = KvPool::new(&hp, backend.buffer_type(), n_blocks as usize).unwrap();
        let (kp, vp) = kv.pools(0);
        kp.set_bytes(&f16_bytes(&noise(kp.elements(), 31)));
        vp.set_bytes(&f16_bytes(&noise(vp.elements(), 32)));
        let ns = n_past.len();
        let width = tables.iter().map(Vec::len).max().unwrap();
        let ctx = Context::new(64, 64);
        let q = ctx
            .new_tensor(
                types::F32,
                &[d as i64, n_head as i64, n_tok as i64, ns as i64],
            )
            .as_input();
        let tab = ctx
            .new_tensor(types::I32, &[width as i64, ns as i64])
            .as_input();
        let past = ctx.new_tensor(types::I32, &[ns as i64]).as_input();
        let scratch_ne = scratch_shape(d as i64, n_head as i64, n_head_kv as i64, n_tok, ns);
        let scratch = ctx.new_tensor(types::F32, &scratch_ne).as_input();
        let out = attention_with(&ctx, q, kp, vp, tab, past, scratch, scale, window, 0).as_output();
        let mut graph = ctx.new_graph(64);
        graph.expand(out);
        ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
            .unwrap();
        let qmul = 2.0 * scale * (d as f32).sqrt().min(16.0);
        let qv: Vec<f32> = noise(q.elements(), 33).iter().map(|x| qmul * x).collect();
        q.set_f32(&qv);
        let mut t = Vec::new();
        for (s, tb) in tables.iter().enumerate() {
            t.extend(tb.iter().map(|&x| x as i32));
            t.resize((s + 1) * width, 0);
        }
        tab.set_i32(&t);
        past.set_i32(&n_past.iter().map(|&p| p as i32).collect::<Vec<_>>());
        backend.compute(&graph).unwrap();
        set_enabled_here(None);
        out.to_f32()
    }

    /// The Hopper kernel against the default kernel on the same inputs: every head size and GQA
    /// ratio it covers, ragged chunks, windows, several sequences. Both round each weight to F16
    /// against a running max (within 2^-12 of the exact sum with |V| < 1), so within 2^-11.
    #[test]
    #[ignore = "needs a Hopper GPU (compute capability 9.0)"]
    fn hopper_matches_default_kernel() {
        let mut backend = gpu();
        let Some(_on) = On::here() else {
            return;
        };
        let tables = [vec![5, 0, 9, 3, 7], vec![2, 10], vec![8, 1, 6, 4]];
        for (d, n_head, n_head_kv) in [
            (128, 32, 2),
            (128, 8, 8),
            (128, 16, 1),
            (256, 16, 8),
            (256, 16, 4),
            (256, 24, 4),
            (256, 24, 2),
        ] {
            // 120 tokens: a whole tile at GQA 1 and several at the rest, within every table.
            for (n_tok, window) in [(37, 0), (70, 100), (5, 64), (120, 0)] {
                let n_past = [190, 3, 120];
                let scale = 1.0 / (d as f32).sqrt();
                let before = launches();
                let want = prefill_out(
                    &mut backend,
                    false,
                    d,
                    n_head,
                    n_head_kv,
                    n_tok,
                    &n_past,
                    &tables,
                    scale,
                    window,
                );
                assert_eq!(
                    launches(),
                    before,
                    "the default kernel's run took the Hopper one"
                );
                let got = prefill_out(
                    &mut backend,
                    true,
                    d,
                    n_head,
                    n_head_kv,
                    n_tok,
                    &n_past,
                    &tables,
                    scale,
                    window,
                );
                assert!(launches() > before, "the Hopper kernel did not run");
                let max_diff = max_abs_diff(&got, &want);
                println!(
                    "d {d}, heads {n_head}/{n_head_kv}, {n_tok} tokens, window {window}: max \
                     |hopper - default| {max_diff:.2e}"
                );
                assert!(
                    max_diff < 5e-4,
                    "d {d}, heads {n_head}/{n_head_kv}, {n_tok} tokens, window {window}: \
                     {max_diff}"
                );
            }
        }
    }
}
