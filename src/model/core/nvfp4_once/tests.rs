//! The op's shape checks, tiles and CPU reference (the GPU bits against ggml's MMVQ:
//! `legion nvfp4-once-bench --check` on the GPU box).

use super::bench::{nvfp4_bytes, routing};
use super::*;
use crate::ggml::{Backend, ComputeAllocator};

fn noise(n: usize, seed: u64) -> Vec<f32> {
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

#[test]
fn fits_shape_takes_the_mmvq_verify_steps() {
    let ctx = Context::new(64, 0);
    let gate = ctx.new_tensor(types::NVFP4, &[2560, 640, 512]);
    let down = ctx.new_tensor(types::NVFP4, &[640, 2560, 512]);
    let q8 = ctx.new_tensor(types::Q8_0, &[2560, 640, 512]);
    let x = |k, n_x, n| ctx.new_tensor(types::F32, &[k, n_x, n]);
    let ids = |n| ctx.new_tensor(types::I32, &[10, n]);
    for n in 2..=8 {
        assert!(fits_shape(gate, x(2560, 1, n), ids(n)), "gate at {n}");
        assert!(fits_shape(down, x(640, 10, n), ids(n)), "down at {n}");
    }
    // One token, ggml's MMQ steps, another type, a column count that is neither 1 nor n_used.
    assert!(!fits_shape(gate, x(2560, 1, 1), ids(1)));
    assert!(!fits_shape(gate, x(2560, 1, 9), ids(9)));
    assert!(!fits_shape(q8, x(2560, 1, 4), ids(4)));
    assert!(!fits_shape(down, x(640, 3, 4), ids(4)));
    // On no GPU (the default until the engine sets one) nothing is used.
    assert!(!usable(gate, None, x(2560, 1, 4), ids(4)));
}

#[test]
fn compute_capability_12_takes_the_ops() {
    // The GB10 and the RTX 50 (#1200); nothing else, and no GPU.
    assert!(gpu_takes(Some((12, 1))));
    assert!(gpu_takes(Some((12, 0))));
    for cc in [(7, 0), (8, 9), (9, 0), (10, 0), (11, 0)] {
        assert!(!gpu_takes(Some(cc)), "{cc:?}");
    }
    assert!(!gpu_takes(None));
}

#[test]
fn the_scratch_holds_every_fitting_step() {
    // Flash-Next's down projection at 8 tokens, the biggest activations the op takes there.
    const { assert!(10 * 8 * 640 / 32 <= SCRATCH_BLOCKS) };
    const { assert!(8 * 2560 / 32 <= SCRATCH_BLOCKS) };
    const { assert!(10 * MAX_TOKENS <= MAX_PAIRS) };
}

#[test]
fn tiles_cover_the_kernel_table_and_divide_the_rows() {
    assert!(DEFAULT_GLU_TILE < TILES.len());
    for k in [640, 2560, 4096] {
        assert!(default_tile(k) < TILES.len());
    }
    for t in TILES {
        assert_eq!(
            ROW_MULTIPLE % (WARPS * t.rows as i64),
            0,
            "{t:?}: fits_shape takes rows in multiples of {ROW_MULTIPLE}"
        );
    }
    // The table: the quantization, then per tile the matmul and the SwiGLU at 1 and 3 K steps,
    // each for 2, 4 and 8 pairs per pass.
    let mut seen: Vec<usize> = (0..TILES.len())
        .flat_map(|t| {
            [false, true].into_iter().flat_map(move |glu| {
                [1, 3]
                    .into_iter()
                    .flat_map(move |i| (0..3).map(move |u| kernel_index(t, glu, i, u)))
            })
        })
        .collect();
    seen.sort_unstable();
    assert_eq!(seen, (1..=12 * TILES.len()).collect::<Vec<_>>());
    assert_eq!([1, 2, 3, 4, 5, 8, 12].map(users), [0, 0, 1, 1, 2, 2, 2]);
    assert_eq!(iters(2560), 1);
    assert_eq!(iters(640), 1);
    force_iters(Some(3));
    assert_eq!(iters(640), 3);
    force_iters(Some(1));
    assert_eq!(iters(2560), 1);
    force_iters(None);
    assert_eq!(iters(2560), DEFAULT_ITERS);
    force_tile(Some(3));
    assert_eq!(tile(true, 2560), 3);
    assert_eq!(tile(false, 640), 3);
    force_tile(Some(99));
    assert_eq!(tile(false, 2560), TILES.len() - 1);
    force_tile(None);
    assert_eq!(tile(true, 2560), DEFAULT_GLU_TILE);
    assert_eq!(tile(false, 640), default_tile(640));
}

#[test]
fn warp_kernels_follow_the_tiles_in_the_table() {
    let ws = warps();
    assert_eq!(ws.len(), 16);
    let mut seen: Vec<usize> = ws
        .iter()
        .flat_map(|&w| [w.kernel(false), w.kernel(true)])
        .collect();
    seen.sort_unstable();
    let first = 1 + 12 * TILES.len();
    assert_eq!(seen, (first..first + 32).collect::<Vec<_>>());
    // The table's order: matmul then SwiGLU, then 1, 2, 4, 8 pairs a pass, per pair then
    // once, 8 then 12 CTAs an SM.
    let w = |users, pre, occ7| Warp { users, pre, occ7 };
    assert_eq!(w(1, false, false).kernel(false), first);
    assert_eq!(w(1, false, true).kernel(false), first + 1);
    assert_eq!(w(1, true, false).kernel(false), first + 2);
    assert_eq!(w(2, false, false).kernel(false), first + 4);
    assert_eq!(w(8, true, true).kernel(true), first + 31);
    assert_eq!(w(4, true, false).name(), "w u4 pre o8");
    assert_eq!(ROW_MULTIPLE % WARP_ROWS, 0);
}

#[test]
fn hybrid_kernels_follow_the_warp_kernels_in_the_table() {
    let hs = hybrids();
    assert_eq!(hs.len(), 9);
    let mut seen: Vec<usize> = hs
        .iter()
        .flat_map(|&h| [h.kernel(false), h.kernel(true)])
        .collect();
    seen.sort_unstable();
    let first = 1 + 12 * TILES.len() + 32;
    assert_eq!(seen, (first..first + 18).collect::<Vec<_>>());
    // The table's order: matmul then SwiGLU, then at most 2, 4, 8 pairs a pass, 8, 6, 4 CTAs.
    let h = |umax, occ| Hybrid { umax, occ };
    assert_eq!(h(2, 8).kernel(false), first);
    assert_eq!(h(2, 6).kernel(false), first + 1);
    assert_eq!(h(4, 8).kernel(false), first + 3);
    assert_eq!(h(2, 8).kernel(true), first + 9);
    assert_eq!(h(8, 4).kernel(true), first + 17);
    assert_eq!(h(4, 6).name(), "h u4 o6");
}

#[test]
fn forcing_a_family_and_the_defaults() {
    let w = |users, pre, occ7| Warp { users, pre, occ7 };
    // Forcing: a warp kernel, a hybrid, the tiles, then the defaults again.
    force_family(Some(Family::Warp(w(2, true, true))));
    assert_eq!(family(true, 4), Family::Warp(w(2, true, true)));
    let h = Hybrid { umax: 8, occ: 4 };
    force_family(Some(Family::Hybrid(h)));
    assert_eq!(family(false, 2), Family::Hybrid(h));
    // The down projection's own family over that one, the SwiGLU's unchanged.
    let d = Family::Warp(w(1, false, false));
    force_matmul_family(Some(d));
    assert_eq!(family(false, 2), d);
    assert_eq!(family(true, 2), Family::Hybrid(h));
    force_matmul_family(None);
    assert_eq!(family(false, 2), Family::Hybrid(h));
    force_family(Some(Family::Tiles));
    assert_eq!(family(false, 4), Family::Tiles);
    force_family(None);
    // The defaults: the hybrid at 4 CTAs an SM for the SwiGLU, the warp kernel at 8 CTAs for
    // the down projection, one pair a pass at 2 tokens.
    assert_eq!(
        default_family(true, 3),
        Family::Hybrid(Hybrid { umax: 4, occ: 4 })
    );
    assert_eq!(default_family(false, 2), Family::Warp(w(1, false, false)));
    assert_eq!(default_family(false, 3), Family::Warp(w(2, false, false)));
    for n in 2..=8 {
        assert_eq!(family(true, n), default_family(true, n), "{n} tokens");
        assert_eq!(family(false, n), default_family(false, n), "{n} tokens");
    }
}

#[test]
fn glu_fits_needs_matching_weights_and_scales() {
    let ctx = Context::new(64, 0);
    let gate = ctx.new_tensor(types::NVFP4, &[2560, 640, 512]);
    let up = ctx.new_tensor(types::NVFP4, &[2560, 640, 512]);
    let other = ctx.new_tensor(types::NVFP4, &[2560, 1280, 512]);
    let s = ctx.new_tensor(types::F32, &[512]);
    let s_bad = ctx.new_tensor(types::F32, &[256]);
    let x = ctx.new_tensor(types::F32, &[2560, 1, 4]);
    let ids = ctx.new_tensor(types::I32, &[10, 4]);
    assert!(glu_fits((gate, None), (up, None), x, ids));
    assert!(glu_fits((gate, Some(s)), (up, Some(s)), x, ids));
    assert!(!glu_fits((gate, Some(s)), (up, None), x, ids));
    assert!(!glu_fits((gate, Some(s_bad)), (up, Some(s_bad)), x, ids));
    assert!(!glu_fits((gate, None), (other, None), x, ids));
    assert!(scale_fits(gate, None));
    assert!(!scale_fits(gate, Some(s_bad)));
    // Rows that are not a multiple of a CTA's rows at the biggest tile.
    let odd = ctx.new_tensor(types::NVFP4, &[2560, 648, 512]);
    assert!(!fits_shape(odd, x, ids));
    // On no GPU (the default until the engine sets one) nothing is used.
    assert!(!glu_usable((gate, None), (up, None), x, ids));
}

#[test]
fn routing_picks_distinct_experts_per_token_from_the_pool() {
    let ids = routing(10, 6, 512, 16, 7);
    assert_eq!(ids.len(), 60);
    for t in ids.chunks(10) {
        let mut s = t.to_vec();
        s.sort_unstable();
        s.dedup();
        assert_eq!(s.len(), 10);
    }
    let unique: std::collections::HashSet<_> = ids.iter().collect();
    assert!(unique.len() <= 16);
}

/// The op on the CPU is the dequantized product, row by row, for both activation layouts and
/// with tokens sharing experts.
#[test]
fn cpu_op_is_the_dequantized_product() {
    let (k, rows, n_expert, n_used, n) = (128usize, 8usize, 6usize, 3usize, 4usize);
    let row_bytes = crate::ggml::row_size(types::NVFP4, k);
    for n_x in [1usize, n_used] {
        let wq = nvfp4_bytes(row_bytes * rows * n_expert, 3);
        let xs = noise(k * n_x * n, 5);
        let ids = routing(n_used, n, n_expert, 4, 11);
        let mut be = Backend::cpu().unwrap();
        let wctx = Context::new(1, 0);
        let w = wctx.new_tensor(types::NVFP4, &[k as i64, rows as i64, n_expert as i64]);
        wctx.alloc_tensors(be.buffer_type()).unwrap();
        w.set_bytes(&wq);
        let ctx = Context::new(8, 16);
        let x = ctx.new_tensor(types::F32, &[k as i64, n_x as i64, n as i64]);
        let it = ctx.new_tensor(types::I32, &[n_used as i64, n as i64]);
        ctx.alloc_tensors(be.buffer_type()).unwrap();
        x.set_f32(&xs);
        it.set_i32(&ids);
        let out = mul_mat_id(&ctx, w, x, it, None);
        assert_eq!(out.ne(), [rows as i64, n_used as i64, n as i64, 1]);
        let mut g = ctx.new_graph(16);
        g.expand(out);
        ctx.alloc_graph(&g, ComputeAllocator::new(be.buffer_type()))
            .unwrap();
        be.compute(&g).unwrap();
        let got = out.to_f32();
        for t in 0..n {
            for j in 0..n_used {
                let e = ids[t * n_used + j] as usize;
                let col = &xs[(t * n_x + if n_x == 1 { 0 } else { j }) * k..][..k];
                for r in 0..rows {
                    let at = (e * rows + r) * row_bytes;
                    let wr = crate::ggml::dequantize(types::NVFP4, &wq[at..at + row_bytes], k);
                    let want: f32 = wr.iter().zip(col).map(|(a, b)| a * b).sum();
                    let v = got[(t * n_used + j) * rows + r];
                    assert_eq!(v.to_bits(), want.to_bits(), "n_x {n_x} t {t} j {j} r {r}");
                }
            }
        }
    }
}

/// The SwiGLU op on the CPU is `silu(gate * s_gate) * (up * s_up)` of the dequantized products,
/// and the matmul op with a scale its product times the expert's scale.
#[test]
fn cpu_ops_apply_the_scales_and_the_swiglu() {
    let (k, rows, n_expert, n_used, n) = (128usize, 32usize, 5usize, 2usize, 3usize);
    let row_bytes = crate::ggml::row_size(types::NVFP4, k);
    let wq = [7, 9].map(|seed| nvfp4_bytes(row_bytes * rows * n_expert, seed));
    let xs = noise(k * n, 13);
    let ids = routing(n_used, n, n_expert, 3, 17);
    let sv = [noise(n_expert, 19), noise(n_expert, 23)];
    let mut be = Backend::cpu().unwrap();
    let wctx = Context::new(4, 0);
    let shape = [k as i64, rows as i64, n_expert as i64];
    let (wg, wu) = (
        wctx.new_tensor(types::NVFP4, &shape),
        wctx.new_tensor(types::NVFP4, &shape),
    );
    let (sg, su) = (
        wctx.new_tensor(types::F32, &[n_expert as i64]),
        wctx.new_tensor(types::F32, &[n_expert as i64]),
    );
    wctx.alloc_tensors(be.buffer_type()).unwrap();
    wg.set_bytes(&wq[0]);
    wu.set_bytes(&wq[1]);
    sg.set_f32(&sv[0]);
    su.set_f32(&sv[1]);
    let ctx = Context::new(8, 16);
    let x = ctx.new_tensor(types::F32, &[k as i64, 1, n as i64]);
    let it = ctx.new_tensor(types::I32, &[n_used as i64, n as i64]);
    ctx.alloc_tensors(be.buffer_type()).unwrap();
    x.set_f32(&xs);
    it.set_i32(&ids);
    let act = swiglu_id(&ctx, (wg, Some(sg)), (wu, Some(su)), x, it);
    let scaled = mul_mat_id(&ctx, wu, x, it, Some(su));
    assert_eq!(act.ne(), [rows as i64, n_used as i64, n as i64, 1]);
    let mut g = ctx.new_graph(16);
    g.expand(act);
    g.expand(scaled);
    ctx.alloc_graph(&g, ComputeAllocator::new(be.buffer_type()))
        .unwrap();
    be.compute(&g).unwrap();
    let (got, got_scaled) = (act.to_f32(), scaled.to_f32());
    for t in 0..n {
        for j in 0..n_used {
            let e = ids[t * n_used + j] as usize;
            let col = &xs[t * k..][..k];
            for r in 0..rows {
                let at = (e * rows + r) * row_bytes;
                let dot = |w: &[u8]| -> f32 {
                    let wr = crate::ggml::dequantize(types::NVFP4, &w[at..at + row_bytes], k);
                    wr.iter().zip(col).map(|(a, b)| a * b).sum()
                };
                let up = dot(&wq[1]) * sv[1][e];
                let gate = dot(&wq[0]) * sv[0][e];
                let want = up * (gate / (1.0 + (-gate).exp()));
                let i = (t * n_used + j) * rows + r;
                assert_eq!(got[i].to_bits(), want.to_bits(), "t {t} j {j} r {r}");
                assert_eq!(got_scaled[i].to_bits(), up.to_bits(), "t {t} j {j} r {r}");
            }
        }
    }
}

#[test]
fn the_router_plans_only_steps_the_ops_take() {
    for n in MIN_TOKENS..=MAX_TOKENS {
        assert!(plan_shape(10, n), "{n} tokens");
    }
    assert!(!plan_shape(10, 1));
    assert!(!plan_shape(10, MAX_TOKENS + 1));
    assert!(!plan_shape(0, 4));
    // More pairs than the plan holds.
    assert!(!plan_shape(MAX_PAIRS / 2 + 1, 2));
    // Switched off (the default), no router writes one.
    assert!(!plan_fits(10, 3));
}

#[test]
fn planned_router_ids_are_marked_and_the_same_on_the_cpu() {
    let mut be = Backend::cpu().unwrap();
    let ctx = Context::new(16, 16);
    let logits = ctx.new_tensor(types::F32, &[16, 3]);
    ctx.alloc_tensors(be.buffer_type()).unwrap();
    logits.set_f32(&noise(16 * 3, 7));
    let (plain, _) = super::super::fused::router(&ctx, logits, 4);
    let (planned, _) = super::super::fused::router_planned(&ctx, logits, 4, true, false);
    let one = ctx.view_2d(logits, [16, 1], logits.nb()[1], 0);
    let (one_tok, _) = super::super::fused::router_planned(&ctx, one, 4, true, false);
    // SAFETY: live nodes of `ctx`.
    unsafe {
        assert!(!super::super::fused::writes_plan(plain.raw()));
        assert!(super::super::fused::writes_plan(planned.raw()));
        // A one-token step is no step of the ops: no plan.
        assert!(!super::super::fused::writes_plan(one_tok.raw()));
    }
    let mut g = ctx.new_graph(16);
    g.expand(plain);
    g.expand(planned);
    ctx.alloc_graph(&g, ComputeAllocator::new(be.buffer_type()))
        .unwrap();
    be.compute(&g).unwrap();
    assert_eq!(plain.to_i32(), planned.to_i32());
}
