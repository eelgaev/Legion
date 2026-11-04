use super::*;
use crate::ggml::{Backend, ComputeAllocator};

/// The op's CPU implementation against ggml's `mul_mat_id` on the CPU: same experts, same
/// outputs (up to ggml quantizing the activations), for shared and per-slot activations.
#[test]
fn cpu_reference_matches_ggml_mul_mat_id() {
    let (k, rows, n_expert, n_used, n_tok) = (256usize, 24usize, 6usize, 3usize, 2usize);
    let wf: Vec<f32> = (0..k * rows * n_expert)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 400.0)
        .collect();
    // Maya's quants (#559) whose ggml quantizers need no importance matrix too.
    for ty in [
        types::Q4_K,
        types::Q5_K,
        types::Q5_1,
        types::Q6_K,
        types::Q8_0,
        types::Q2_K,
        types::Q3_K,
        types::IQ3_XXS,
        types::IQ3_S,
        types::IQ4_XS,
    ] {
        let wq = crate::ggml::quantize(ty, &wf, k);
        for n_x in [1, n_used] {
            let xs: Vec<f32> = (0..k * n_x * n_tok)
                .map(|i| ((i * 13 % 29) as f32 - 14.0) / 7.3)
                .collect();
            let ids = [5i32, 0, 2, 2, 4, 5];
            let run = |ours: bool| {
                let mut b = Backend::cpu().unwrap();
                b.set_threads(2);
                let weights = Context::new(1, 0);
                let w = weights.new_tensor(ty, &[k as i64, rows as i64, n_expert as i64]);
                weights.alloc_tensors(b.buffer_type()).unwrap();
                w.set_bytes(&wq);
                let ctx = Context::new(8, 16);
                let x = ctx.new_tensor(types::F32, &[k as i64, n_x as i64, n_tok as i64]);
                let it = ctx.new_tensor(types::I32, &[n_used as i64, n_tok as i64]);
                ctx.alloc_tensors(b.buffer_type()).unwrap();
                x.set_f32(&xs);
                it.set_i32(&ids);
                let out = if ours {
                    mul_mat_id(&ctx, w, x, it)
                } else {
                    ctx.mul_mat_id(w, x, it)
                };
                let mut g = ctx.new_graph(16);
                g.expand(out);
                ctx.alloc_graph(&g, ComputeAllocator::new(b.buffer_type()))
                    .unwrap();
                b.compute(&g).unwrap();
                assert_eq!(out.shape(), vec![rows as i64, n_used as i64, n_tok as i64]);
                out.to_f32()
            };
            let (ours, theirs) = (run(true), run(false));
            let scale = theirs.iter().fold(0f32, |m, v| m.max(v.abs()));
            let diff = ours
                .iter()
                .zip(&theirs)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                diff < 0.02 * scale,
                "{} n_x {n_x}: max |diff| {diff} of {scale}",
                crate::ggml::type_name(ty)
            );
        }
    }
}

/// The SwiGLU op's CPU implementation against ggml's two `mul_mat_id`s, silu and mul.
#[test]
fn cpu_glu_matches_ggml() {
    glu_against_ggml(0.0);
}

/// The clamped SwiGLU (GLM-5's `swiglu_limit`) against ggml's products clamped, then silu
/// and mul; a limit small enough to clamp many of them.
#[test]
fn cpu_glu_clamped_matches_ggml() {
    glu_against_ggml(0.75);
}

fn glu_against_ggml(limit: f32) {
    let (k, rows, n_expert, n_used, n_tok) = (256usize, 24usize, 6usize, 3usize, 2usize);
    let mk = |seed: usize| -> Vec<u8> {
        let wf: Vec<f32> = (0..k * rows * n_expert)
            .map(|i| (((i + seed) * 37 % 101) as f32 - 50.0) / 400.0)
            .collect();
        crate::ggml::quantize(types::Q4_K, &wf, k)
    };
    let (gq, uq) = (mk(0), mk(7));
    let xs: Vec<f32> = (0..k * n_tok)
        .map(|i| ((i * 13 % 29) as f32 - 14.0) / 7.3)
        .collect();
    let ids = [5i32, 0, 2, 2, 4, 5];
    let run = |ours: bool, limit: f32| {
        let mut b = Backend::cpu().unwrap();
        b.set_threads(2);
        let weights = Context::new(2, 0);
        let shape = [k as i64, rows as i64, n_expert as i64];
        let g = weights.new_tensor(types::Q4_K, &shape);
        let u = weights.new_tensor(types::Q4_K, &shape);
        weights.alloc_tensors(b.buffer_type()).unwrap();
        g.set_bytes(&gq);
        u.set_bytes(&uq);
        let ctx = Context::new(8, 32);
        let x = ctx.new_tensor(types::F32, &[k as i64, 1, n_tok as i64]);
        let it = ctx.new_tensor(types::I32, &[n_used as i64, n_tok as i64]);
        ctx.alloc_tensors(b.buffer_type()).unwrap();
        x.set_f32(&xs);
        it.set_i32(&ids);
        let out = if ours {
            glu_tiered_clamped(&ctx, g, u, x, it, 0, limit)
        } else {
            let (mut gate, mut up) = (ctx.mul_mat_id(g, x, it), ctx.mul_mat_id(u, x, it));
            if limit > 0.0 {
                gate = ctx.clamp(gate, f32::NEG_INFINITY, limit);
                up = ctx.clamp(up, -limit, limit);
            }
            ctx.mul(ctx.silu(gate), up)
        };
        let mut gr = ctx.new_graph(32);
        gr.expand(out);
        ctx.alloc_graph(&gr, ComputeAllocator::new(b.buffer_type()))
            .unwrap();
        b.compute(&gr).unwrap();
        out.to_f32()
    };
    let (ours, theirs) = (run(true, limit), run(false, limit));
    if limit > 0.0 {
        assert_ne!(theirs, run(false, 0.0), "the limit clamps nothing");
    }
    let scale = theirs.iter().fold(0f32, |m, v| m.max(v.abs()));
    let diff = ours
        .iter()
        .zip(&theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(diff < 0.03 * scale, "max |diff| {diff} of {scale}");
}

#[test]
fn tiles_fit_the_loads_in_flight() {
    // Qwen3.8-Flash-Next: gate/up Q4_K rows of 2560 (1440 bytes), down Q5_1 of 640 (480).
    assert_eq!(rows_per_cta(1440, 1), 8);
    assert_eq!(rows_per_cta(1440, 2), 4);
    assert_eq!(rows_per_cta(480, 1), 32);
    assert_eq!(rows_per_cta(680, 1), 16);
    assert_eq!(rows_per_cta(2720, 1), 4);
    // GLM-5.3-Flash: gate/up Q4_K rows of 4096 (2304 bytes) take 2-row tiles together.
    assert_eq!(rows_per_cta(2304, 1), 4);
    assert_eq!(rows_per_cta(2304, 2), 2);
    assert_eq!(q8_1_column_bytes(2560), 2880);
}

#[test]
fn q4k_unit_activations_in_the_scratch() {
    // Qwen3.8-Flash-Next's gate/up: K 2560, 160 units of 32 bytes per column after the Q8_1 ones.
    assert_eq!(scratch_bytes(types::Q8_0, 2560, 3), 3 * 2880);
    assert_eq!(scratch_bytes(types::Q5_1, 640, 3), 3 * 720);
    assert_eq!(q4k_units_at(2560, 3), 3 * 2880);
    assert_eq!(scratch_bytes(types::Q4_K, 2560, 3), 3 * 2880 + 3 * 5120);
    // 16-aligned: a Q4_K row's Q8_1 column is 288 bytes per 256 values.
    assert_eq!(q4k_units_at(256, 1), 288);
    // Not a whole number of Q4_K blocks: Q8_1 only.
    assert_eq!(scratch_bytes(types::Q4_K, 2592, 2), 2 * 2916);

    let mut o = operands(0x1000, 0x8000, 2560, 512);
    o.ty = types::Q4_K;
    o.xq_bytes = scratch_bytes(types::Q4_K, 2560, 4);
    for (n_tok, on) in [
        (1, false),
        (2, false),
        (3, true),
        (4, true),
        (5, false),
        (8, false),
    ] {
        o.n_tok = n_tok;
        assert_eq!(q4k_units_for(&o), on, "{n_tok} tokens");
    }
    o.n_tok = 3;
    // A scratch without them (smaller), a single matrix, another type: off.
    o.xq_bytes = q4k_units_at(2560, 3);
    assert!(!q4k_units_for(&o));
    o.xq_bytes = scratch_bytes(types::Q4_K, 2560, 3);
    assert!(q4k_units_for(&o));
    o.w2 = std::ptr::null();
    assert!(!q4k_units_for(&o));
    o.w2 = 0x8000 as *const u8;
    o.ty = types::Q5_K;
    assert!(!q4k_units_for(&o));
}

#[test]
fn q4k_units_scratch_only_when_switched_on() {
    // Off (the default): the scratch is the Q8_1 columns alone, the graph's layout as before.
    let weights = Context::new(1, 0);
    let w = weights.new_tensor(types::Q4_K, &[2560, 512, 8]);
    let ctx = Context::new(8, 16);
    let x = ctx.new_tensor(types::F32, &[2560, 1, 3]);
    let ids = ctx.new_tensor(types::I32, &[8, 3]);
    assert_eq!(q8_1_scratch(&ctx, w, x, ids).ne()[0], 3 * 2880);
    set_q4k_units(true);
    let on = q8_1_scratch(&ctx, w, x, ids).ne()[0];
    set_q4k_units(false);
    assert_eq!(on as usize, scratch_bytes(types::Q4_K, 2560, 3));
}

/// Operands of a launch for [`launch_check`]: Q8_0 rows of `k`, `rows` per expert, contiguous.
fn operands(w: usize, w2: usize, k: usize, rows: usize) -> Operands {
    let nb1 = k / 32 * 34;
    Operands {
        ty: types::Q8_0,
        xq: std::ptr::null_mut(),
        xq_bytes: 0,
        w: w as *const u8,
        w2: w2 as *const u8,
        w_nb: [34, nb1, nb1 * rows, nb1 * rows * 4],
        x: std::ptr::null(),
        x_nb: [4, 4 * k, 4 * k, 4 * k],
        n_x: 1,
        ids: std::ptr::null(),
        ids_nb1: 8,
        dst: std::ptr::null_mut(),
        dst_nb: [4, 4 * rows, 8 * rows, 8 * rows],
        k,
        rows,
        n_used: 2,
        n_tok: 1,
        n_w: 4,
        vram_below: 0,
        split: overlap::NONE,
        mode: 0,
        stage: Staging {
            buf: std::ptr::null_mut(),
            bytes: 0,
            slots: 0,
            slot: 0,
            down: std::ptr::null(),
            down_nb2: 0,
        },
        limit: 0.0,
        id_base: 0,
        zero_skipped: false,
        map: std::ptr::null(),
        n_map: 0,
        ahead: None,
        halves: 0,
        pad: std::ptr::null(),
    }
}

/// `--tp-moe-ep` (#798): two ranks' shares of a layer's experts, each through the SwiGLU op
/// and the down op with the router's ids as they are ([`glu_share`], [`mul_mat_id_share`]).
/// Each pair's rows are the whole stack's on the rank that holds its expert, bit for bit, and
/// 0 on the other; so the ranks' weighted sums add up to the whole MoE.
#[test]
fn cpu_shares_split_the_pairs() {
    let (k, n_ff, n_expert, n_used, n_tok) = (256usize, 256usize, 6usize, 3usize, 3usize);
    let mk = |seed: usize, k: usize, rows: usize| -> Vec<u8> {
        let wf: Vec<f32> = (0..k * rows * n_expert)
            .map(|i| (((i + seed) * 37 % 101) as f32 - 50.0) / 400.0)
            .collect();
        crate::ggml::quantize(types::Q4_K, &wf, k)
    };
    let (gq, uq, dq) = (mk(0, k, n_ff), mk(7, k, n_ff), mk(3, n_ff, k));
    let xs: Vec<f32> = (0..k * n_tok)
        .map(|i| ((i * 13 % 29) as f32 - 14.0) / 7.3)
        .collect();
    // Token 1's experts all on the first share, token 2's on the second; one id twice.
    let ids = [5i32, 0, 2, 2, 1, 0, 4, 3, 5];
    // Experts `which` of the stack (slot `i` holds `which[i]`) as a share: `first` (contiguous
    // `first..`) or with `map` the router's ids through a slot table (`--tp-moe-ep-place`);
    // `share` false: the whole stack through the plain ops.
    let run = |which: &[usize], share: bool, map: bool| {
        let mut b = Backend::cpu().unwrap();
        b.set_threads(2);
        let n = which.len();
        let weights = Context::new(3, 0);
        let slice = |q: &[u8]| {
            let per = q.len() / n_expert;
            which
                .iter()
                .flat_map(|&e| &q[e * per..][..per])
                .copied()
                .collect::<Vec<u8>>()
        };
        let g = weights.new_tensor(types::Q4_K, &[k as i64, n_ff as i64, n as i64]);
        let u = weights.new_tensor(types::Q4_K, &[k as i64, n_ff as i64, n as i64]);
        let d = weights.new_tensor(types::Q4_K, &[n_ff as i64, k as i64, n as i64]);
        weights.alloc_tensors(b.buffer_type()).unwrap();
        g.set_bytes(&slice(&gq));
        u.set_bytes(&slice(&uq));
        d.set_bytes(&slice(&dq));
        let ctx = Context::new(8, 32);
        let x = ctx.new_tensor(types::F32, &[k as i64, 1, n_tok as i64]);
        let it = ctx.new_tensor(types::I32, &[n_used as i64, n_tok as i64]);
        let mt = ctx.new_tensor(types::I32, &[n_expert as i64]);
        ctx.alloc_tensors(b.buffer_type()).unwrap();
        x.set_f32(&xs);
        it.set_i32(&ids);
        let slots: Vec<i32> = (0..n_expert)
            .map(|e| which.iter().position(|&w| w == e).map_or(-1, |s| s as i32))
            .collect();
        mt.set_i32(&slots);
        let s = Share {
            first: if map { 0 } else { which[0] },
            map: map.then_some(mt),
        };
        let out = if share {
            mul_mat_id_share(&ctx, d, glu_share(&ctx, g, u, x, it, s), it, s)
        } else {
            mul_mat_id(&ctx, d, glu_tiered(&ctx, g, u, x, it, 0), it)
        };
        let mut gr = ctx.new_graph(32);
        gr.expand(out);
        ctx.alloc_graph(&gr, ComputeAllocator::new(b.buffer_type()))
            .unwrap();
        b.compute(&gr).unwrap();
        assert_eq!(out.shape(), vec![k as i64, n_used as i64, n_tok as i64]);
        out.to_f32()
    };
    let whole = run(&(0..n_expert).collect::<Vec<_>>(), false, false);
    let half = n_expert / 2;
    // Contiguous halves (#798), then scattered ones through the map (#803), stored out of
    // order.
    let splits: [(Vec<usize>, Vec<usize>, bool); 2] = [
        ((0..half).collect(), (half..n_expert).collect(), false),
        (vec![4, 0, 3], vec![5, 2, 1], true),
    ];
    for (lo, hi, map) in splits {
        let (a, b) = (run(&lo, true, map), run(&hi, true, map));
        for (pair, &e) in ids.iter().enumerate() {
            let rows = |v: &[f32]| v[pair * k..][..k].to_vec();
            let (own, other) = if lo.contains(&(e as usize)) {
                (&a, &b)
            } else {
                (&b, &a)
            };
            assert!(rows(&whole).iter().any(|&v| v != 0.0), "pair {pair}");
            assert_eq!(
                rows(own),
                rows(&whole),
                "pair {pair} (expert {e}, map {map})"
            );
            assert!(rows(other).iter().all(|&v| v == 0.0), "pair {pair}");
        }
    }
}

/// The launch guard refuses what would fault on the GPU. Regression: on 2026-10-17 a GLU
/// test with Q8_0 rows of K = 640 (680 bytes) and 2561 rows per expert put expert 1 at an
/// 8-byte offset, and the tile kernel's 16-byte loads raised Xid 13 (misaligned address) on
/// every SM of the GPU. [`fits`] has kept graphs off such shapes since; now no launch can
/// reach the kernel with them either.
#[test]
fn launch_check_refuses_what_the_kernels_cannot_read() {
    let check = |o: &Operands| {
        let mats = if o.w2.is_null() { 1 } else { 2 };
        let r = rows_per_cta(o.w_nb[1], mats);
        launch_check(o, r, mats * (r * o.w_nb[1]).next_multiple_of(16))
    };
    // Qwen3.8-Flash-Next's down projection shape (as Q8_0) is fine, alone and as a GLU pair.
    check(&operands(0x1000, 0, 640, 2560)).unwrap();
    check(&operands(0x1000, 0x8000, 640, 2560)).unwrap();
    // The Xid 13 case: experts 2561 rows x 680 bytes apart, not a multiple of 16.
    let e = check(&operands(0x1000, 0x8000, 640, 2561)).unwrap_err();
    assert!(e.contains("stride"), "{e}");
    // Misaligned bases.
    assert!(check(&operands(0x1008, 0, 640, 2560)).is_err());
    assert!(check(&operands(0x1000, 0x8008, 640, 2560)).is_err());
    // K not a multiple of 32 (a warp per Q8_1 block).
    let mut o = operands(0x1000, 0, 640, 2560);
    o.k = 624;
    assert!(check(&o).is_err());
    // A tile past the loads in flight.
    let o = operands(0x1000, 0, 640, 2560);
    assert!(launch_check(&o, 32, 32 * 680).is_err());
    // An activation scratch the IQ types' 16-byte quant loads cannot read.
    let mut o = operands(0x1000, 0, 640, 2560);
    o.xq = 0x2004 as *mut u8;
    let e = check(&o).unwrap_err();
    assert!(e.contains("scratch"), "{e}");
}

/// Ids outside the expert tensor (a corrupt or padded routing entry) leave their pair's row at
/// zero on the CPU instead of reading out of bounds, and the other pairs are unchanged.
#[test]
fn cpu_skips_ids_out_of_range() {
    let (k, rows, n_expert, n_used, n_tok) = (64usize, 8usize, 3usize, 2usize, 2usize);
    let wf: Vec<f32> = (0..k * rows * n_expert)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 400.0)
        .collect();
    let wq = crate::ggml::quantize(types::Q8_0, &wf, k);
    let xs: Vec<f32> = (0..k * n_tok)
        .map(|i| ((i * 13 % 29) as f32 - 14.0) / 7.3)
        .collect();
    let run = |ids: [i32; 4]| {
        let mut b = Backend::cpu().unwrap();
        b.set_threads(2);
        let weights = Context::new(1, 0);
        let w = weights.new_tensor(types::Q8_0, &[k as i64, rows as i64, n_expert as i64]);
        weights.alloc_tensors(b.buffer_type()).unwrap();
        w.set_bytes(&wq);
        let ctx = Context::new(8, 16);
        let x = ctx.new_tensor(types::F32, &[k as i64, 1, n_tok as i64]);
        let it = ctx.new_tensor(types::I32, &[n_used as i64, n_tok as i64]);
        ctx.alloc_tensors(b.buffer_type()).unwrap();
        x.set_f32(&xs);
        it.set_i32(&ids);
        let out = mul_mat_id(&ctx, w, x, it);
        let mut gr = ctx.new_graph(16);
        gr.expand(out);
        ctx.alloc_graph(&gr, ComputeAllocator::new(b.buffer_type()))
            .unwrap();
        b.compute(&gr).unwrap();
        out.to_f32()
    };
    let good = run([2, 0, 1, 2]);
    let bad = run([2, -1, 1, n_expert as i32]);
    for (pair, ok) in [(0, true), (1, false), (2, true), (3, false)] {
        let (g, b) = (&good[pair * rows..][..rows], &bad[pair * rows..][..rows]);
        if ok {
            assert_eq!(g, b, "pair {pair}");
        } else {
            assert!(b.iter().all(|&v| v == 0.0), "pair {pair}: {b:?}");
        }
    }
}

/// The split ops (#735) on the CPU: an expert's two halves (SwiGLU rows, then down over the
/// matching K half) add up to the whole expert's output; a whole id in a split op is the
/// whole expert's output and a -1 zero.
#[test]
fn split_halves_add_up_to_the_whole_expert() {
    let (k, n_ff, n_expert, n_used, n_tok) = (256usize, 512usize, 4usize, 3usize, 2usize);
    let mk = |ty, n: usize, k: usize, seed: usize| {
        let wf: Vec<f32> = (0..n)
            .map(|i| (((i + seed) * 37 % 101) as f32 - 50.0) / 400.0)
            .collect();
        crate::ggml::quantize(ty, &wf, k)
    };
    let xs: Vec<f32> = (0..k * n_tok)
        .map(|i| ((i * 13 % 29) as f32 - 14.0) / 7.3)
        .collect();
    let ids = [3i32, 0, 2, 1, 3, 0];
    for (tg, td) in [(types::IQ3_XXS, types::IQ3_XXS), (types::Q2_K, types::Q3_K)] {
        let (gq, uq) = (
            mk(tg, k * n_ff * n_expert, k, 0),
            mk(tg, k * n_ff * n_expert, k, 7),
        );
        let dq = mk(td, n_ff * k * n_expert, n_ff, 3);
        // `side`: None the whole ops, Some(s) the split ops' half s.
        let run = |ids: &[i32], side: Option<usize>| {
            let mut b = Backend::cpu().unwrap();
            b.set_threads(2);
            let weights = Context::new(3, 0);
            let gu = [k as i64, n_ff as i64, n_expert as i64];
            let g = weights.new_tensor(tg, &gu);
            let u = weights.new_tensor(tg, &gu);
            let d = weights.new_tensor(td, &[n_ff as i64, k as i64, n_expert as i64]);
            weights.alloc_tensors(b.buffer_type()).unwrap();
            g.set_bytes(&gq);
            u.set_bytes(&uq);
            d.set_bytes(&dq);
            let ctx = Context::new(8, 32);
            let x = ctx.new_tensor(types::F32, &[k as i64, 1, n_tok as i64]);
            let it = ctx.new_tensor(types::I32, &[n_used as i64, n_tok as i64]);
            ctx.alloc_tensors(b.buffer_type()).unwrap();
            x.set_f32(&xs);
            it.set_i32(ids);
            let out = match side {
                None => {
                    let act = glu_tiered_clamped(&ctx, g, u, x, it, 0, 0.75);
                    mul_mat_id_tiered(&ctx, d, act, it, 0)
                }
                Some(s) => {
                    let act = glu_split_clamped(&ctx, g, u, x, it, 0, 0.75, s);
                    mul_mat_id_split(&ctx, d, act, it, 0, s)
                }
            };
            let mut gr = ctx.new_graph(32);
            gr.expand(out);
            ctx.alloc_graph(&gr, ComputeAllocator::new(b.buffer_type()))
                .unwrap();
            b.compute(&gr).unwrap();
            out.to_f32()
        };
        let whole = run(&ids, None);
        let halves = ids.map(|e| e | HALF_BIT);
        let (h0, h1) = (run(&halves, Some(0)), run(&halves, Some(1)));
        let scale = whole.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(scale > 0.0);
        for (i, w) in whole.iter().enumerate() {
            assert!(
                (h0[i] + h1[i] - w).abs() <= 1e-4 * scale,
                "{}: output {i}: {} + {} vs {w}",
                crate::ggml::type_name(td),
                h0[i],
                h1[i]
            );
        }
        assert_ne!(h0, h1, "the halves are not the same");
        // Mixed: pair 0 whole, pair 1 skipped, the rest halves.
        let mut mixed = halves;
        mixed[0] = ids[0];
        mixed[1] = -1;
        let m = run(&mixed, Some(1));
        assert_eq!(&m[..k], &whole[..k], "a whole id in a split op");
        assert!(m[k..2 * k].iter().all(|&v| v == 0.0), "a skipped id");
        assert_eq!(&m[2 * k..], &h1[2 * k..], "the other halves");
    }
}

/// [`split_check`] takes GLM's shapes in halves and refuses what `expert_mv_split` cannot
/// read.
#[test]
fn split_check_refuses_what_the_split_kernel_cannot_read() {
    // Maya-S's down projection: IQ2_S rows of K = 2048 (656 bytes, halves of 328).
    let iq2s = |k: usize, rows: usize| {
        let mut o = operands(0x1000, 0, k, rows);
        o.ty = types::IQ2_S;
        o.w_nb[1] = crate::ggml::row_size(types::IQ2_S, k);
        o.halves = 1;
        o
    };
    split_check(&iq2s(2048, 4096), 4, 1).unwrap();
    // A plain op (no half), a third half, a type with decoded-once dot products.
    let mut o = iq2s(2048, 4096);
    o.halves = 0;
    assert!(split_check(&o, 4, 1).is_err());
    o.halves = 3;
    assert!(split_check(&o, 4, 1).is_err());
    let mut o = operands(0x1000, 0, 2048, 4096);
    o.halves = 2;
    assert!(split_check(&o, 4, 1).is_err());
    // K halves that are not whole blocks (K = 256 + 256 + 256: 1.5 blocks a half).
    assert!(split_check(&iq2s(768, 4096), 4, 1).is_err());
    // SwiGLU halves of 2048 rows in tiles of 4: whole; of 6: not.
    let mut o = iq2s(4096, 2048);
    o.ty = types::IQ2_XXS;
    o.w_nb[1] = crate::ggml::row_size(types::IQ2_XXS, 4096);
    split_check(&o, 4, 2).unwrap();
    assert!(split_check(&o, 6, 2).is_err());
}

/// `Dot`'s codes are the kernel's `DOT_*` values, and `auto` is the default.
#[test]
fn dot_codes_match_the_kernel() {
    assert_eq!(Dot::default(), Dot::Auto);
    let codes: Vec<usize> = [Dot::Auto, Dot::PerUser, Dot::Once, Dot::Sub16]
        .into_iter()
        .map(Dot::code)
        .collect();
    assert_eq!(codes, [0, 1, 2, 3]);
}

/// The pipelined tile kernel's default (#816): Flash-Next's gate/up Q4_K, down Q5_1 and MTP
/// layer Q8_0 at the step sizes measured faster, the tile kernel (0) elsewhere; a set value
/// applies to every op.
#[test]
fn pipe_tiles_default_and_override() {
    let auto = |ty, glu| {
        (1..=8)
            .map(|n| auto_pipe_tiles(ty, glu, n))
            .collect::<Vec<_>>()
    };
    assert_eq!(auto(types::Q4_K, true), [1, 4, 4, 4, 0, 0, 0, 0]);
    assert_eq!(auto(types::Q5_1, false), [1, 4, 4, 4, 0, 0, 0, 0]);
    assert_eq!(auto(types::Q8_0, false), [4, 4, 4, 0, 0, 0, 0, 0]);
    assert_eq!(auto(types::Q8_0, true), [0; 8]);
    assert_eq!(auto(types::Q4_K, false), [0; 8]);
    assert_eq!(pipe_tiles(types::Q4_K, true, 3), 4);
    set_pipe_tiles(Some(2));
    assert_eq!(pipe_tiles(types::Q8_0, false, 3), 2);
    set_pipe_tiles(Some(0));
    assert_eq!(pipe_tiles(types::Q4_K, true, 3), 0);
    set_pipe_tiles(None);
    assert_eq!(pipe_tiles(types::Q4_K, true, 3), 4);
}

/// The link workers go only to launches whose tile kernel takes both VRAM and host experts:
/// part 0 or 3, the VRAM experts not handed to another kernel, no `--expert-mv-prefetch`.
#[test]
fn link_workers_only_for_mixed_tile_launches() {
    // Off by default: no launch gets workers.
    assert_eq!(LINK_CTAS, 0);
    assert_eq!(link_ctas_for(64, 320, 3, 0, false), 0);
    // Only this test sets it (the CUDA launch alone reads it).
    set_link_ctas(64);
    let n = 64;
    // 64 cache slots of 256 experts (n_w: slots + experts).
    let (below, n_w) = (64, 320);
    assert_eq!(link_ctas_for(below, n_w, 0, 0, false), n);
    assert_eq!(link_ctas_for(below, n_w, 3, 0, false), n);
    for part in [1, 2, 4] {
        assert_eq!(link_ctas_for(below, n_w, part, 0, false), 0, "part {part}");
    }
    assert_eq!(link_ctas_for(below, n_w, 0, 1, false), 0);
    assert_eq!(link_ctas_for(below, n_w, 0, 2, false), 0);
    assert_eq!(link_ctas_for(below, n_w, 0, 0, true), 0);
    // No cache (all host) or every expert resident: nothing to keep apart.
    assert_eq!(link_ctas_for(0, 256, 0, 0, false), 0);
    assert_eq!(link_ctas_for(256, 256, 0, 0, false), 0);
    set_link_ctas(LINK_CTAS);
}

/// #819: the SwiGLU op from Q8_1 blocks written already reads them in place of its scratch,
/// keeps the scratch as a source (the graph's layout), and computes what [`glu_tiered`] does.
#[test]
fn glu_preq_reads_given_blocks_and_keeps_the_scratch() {
    let (k, rows, n_expert, n_used, n_tok) = (256usize, 24usize, 6usize, 3usize, 2usize);
    let mk = |seed: usize| -> Vec<u8> {
        let wf: Vec<f32> = (0..k * rows * n_expert)
            .map(|i| (((i + seed) * 37 % 101) as f32 - 50.0) / 400.0)
            .collect();
        crate::ggml::quantize(types::Q4_K, &wf, k)
    };
    let (gq, uq) = (mk(0), mk(7));
    let xs: Vec<f32> = (0..k * n_tok)
        .map(|i| ((i * 13 % 29) as f32 - 14.0) / 7.3)
        .collect();
    let ids = [5i32, 0, 2, 2, 4, 5];
    let q_elems = crate::model::core::q8_group::quantized_elements(k as i64, n_tok as i64);
    let run = |preq: bool| {
        let mut b = Backend::cpu().unwrap();
        b.set_threads(2);
        let weights = Context::new(2, 0);
        let shape = [k as i64, rows as i64, n_expert as i64];
        let g = weights.new_tensor(types::Q4_K, &shape);
        let u = weights.new_tensor(types::Q4_K, &shape);
        weights.alloc_tensors(b.buffer_type()).unwrap();
        g.set_bytes(&gq);
        u.set_bytes(&uq);
        let ctx = Context::new(8, 32);
        let x = ctx.new_tensor(types::F32, &[k as i64, 1, n_tok as i64]);
        let it = ctx.new_tensor(types::I32, &[n_used as i64, n_tok as i64]);
        let xq = ctx.new_tensor(types::F32, &[q_elems]);
        ctx.alloc_tensors(b.buffer_type()).unwrap();
        x.set_f32(&xs);
        it.set_i32(&ids);
        let out = if preq {
            assert!(preq_shapes(g, x, it, xq));
            // Too few blocks, or a column per pair: not taken.
            let short = ctx.view_1d(xq, q_elems - 1, 0);
            assert!(!preq_shapes(g, x, it, short));
            let per_pair = ctx.new_tensor(types::F32, &[k as i64, n_used as i64, n_tok as i64]);
            assert!(!preq_shapes(g, per_pair, it, xq));
            let out = glu_tiered_preq(&ctx, g, u, x, it, 0, xq);
            // SAFETY: a node just made.
            let raw = unsafe { &*out.raw() };
            assert_eq!(raw.src[3], xq.raw(), "the given blocks");
            assert_eq!(raw.src[4], u.raw(), "up");
            // SAFETY: the scratch source of the node.
            let scratch = unsafe { &*raw.src[5] };
            assert_eq!(scratch.type_, types::I8);
            assert!(scratch.ne[0] as usize >= q8_1_column_bytes(k) * n_tok);
            out
        } else {
            glu_tiered(&ctx, g, u, x, it, 0)
        };
        let mut gr = ctx.new_graph(32);
        gr.expand(out);
        ctx.alloc_graph(&gr, ComputeAllocator::new(b.buffer_type()))
            .unwrap();
        b.compute(&gr).unwrap();
        out.to_f32()
    };
    let (a, b) = (run(true), run(false));
    assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()));
}

/// The Q4_K activation hold (#816 rounds 6 and 8) is on by default and can be turned off and on again.
#[test]
fn q4k_hold_is_the_default() {
    assert!(q4k_hold());
    set_q4k_hold(false);
    assert!(!q4k_hold());
    set_q4k_hold(true);
    assert!(q4k_hold());
}

#[test]
fn balance_is_the_default() {
    assert!(balance());
    set_balance(false);
    assert!(!balance());
    set_balance(true);
    assert!(balance());
}

#[test]
fn half_tiles_is_the_default() {
    assert!(half_tiles());
    set_half_tiles(false);
    assert!(!half_tiles());
    set_half_tiles(true);
    assert!(half_tiles());
}

#[test]
fn half_tiles_take_the_q5_1_down_at_1_to_3_tokens() {
    let at = |ty, glu| {
        (1..=5)
            .map(|n| half_tiles_at(ty, glu, n))
            .collect::<Vec<_>>()
    };
    assert_eq!(at(types::Q5_1, false), [true, true, true, false, false]);
    assert_eq!(at(types::Q5_1, true), [false; 5]);
    assert_eq!(at(types::Q8_0, false), [false; 5]);
    assert_eq!(at(types::Q4_K, true), [false; 5]);
}

/// `--moe-pad-skip-tiered` (#814): with a verify step's row map the SwiGLU and down ops skip
/// the dead padding pairs only (a padding row's pair of an expert no real row chose: rows 0);
/// every other pair, a padding row's pair of a real row's expert too, is the plain ops' bits.
#[test]
fn cpu_row_map_skips_dead_padding_pairs() {
    let (k, n_ff, n_expert, n_used, n_tok) = (256usize, 256usize, 6usize, 2usize, 4usize);
    let mk = |seed: usize, k: usize, rows: usize| -> Vec<u8> {
        let wf: Vec<f32> = (0..k * rows * n_expert)
            .map(|i| (((i + seed) * 37 % 101) as f32 - 50.0) / 400.0)
            .collect();
        crate::ggml::quantize(types::Q4_K, &wf, k)
    };
    let (gq, uq, dq) = (mk(0, k, n_ff), mk(7, k, n_ff), mk(3, n_ff, k));
    let xs: Vec<f32> = (0..k * n_tok)
        .map(|i| ((i * 13 % 29) as f32 - 14.0) / 7.3)
        .collect();
    // Two sequences of two rows: the first's both real, the second's row 3 padding. Row 3's
    // expert 4 is row 2's too (live), its expert 5 no real row's (dead).
    let ids = [0i32, 1, 2, 3, 1, 4, 4, 5];
    let map = [0i32, 1, 2, 2];
    let run = |pad: bool| {
        let mut b = Backend::cpu().unwrap();
        b.set_threads(2);
        let weights = Context::new(3, 0);
        let g = weights.new_tensor(types::Q4_K, &[k as i64, n_ff as i64, n_expert as i64]);
        let u = weights.new_tensor(types::Q4_K, &[k as i64, n_ff as i64, n_expert as i64]);
        let d = weights.new_tensor(types::Q4_K, &[n_ff as i64, k as i64, n_expert as i64]);
        weights.alloc_tensors(b.buffer_type()).unwrap();
        g.set_bytes(&gq);
        u.set_bytes(&uq);
        d.set_bytes(&dq);
        let ctx = Context::new(8, 32);
        let x = ctx.new_tensor(types::F32, &[k as i64, 1, n_tok as i64]);
        let it = ctx.new_tensor(types::I32, &[n_used as i64, n_tok as i64]);
        let rows = ctx.new_tensor(types::I32, &[n_tok as i64]);
        ctx.alloc_tensors(b.buffer_type()).unwrap();
        x.set_f32(&xs);
        it.set_i32(&ids);
        rows.set_i32(&map);
        assert!(pad_fits(rows, it));
        let pad = pad.then_some(rows);
        let act = glu_tiered_pad(&ctx, g, u, x, it, 0, pad);
        let out = mul_mat_id_tiered_pad(&ctx, d, act, it, 0, pad);
        let mut gr = ctx.new_graph(32);
        gr.expand(out);
        ctx.alloc_graph(&gr, ComputeAllocator::new(b.buffer_type()))
            .unwrap();
        b.compute(&gr).unwrap();
        assert_eq!(out.shape(), vec![k as i64, n_used as i64, n_tok as i64]);
        (act.to_f32(), out.to_f32())
    };
    let (plain_act, plain) = run(false);
    let (skip_act, skip) = run(true);
    let dead = 3 * n_used + 1;
    for pair in 0..n_used * n_tok {
        let (a, o) = (pair * n_ff..(pair + 1) * n_ff, pair * k..(pair + 1) * k);
        assert!(plain[o.clone()].iter().any(|&v| v != 0.0), "pair {pair}");
        if pair == dead {
            assert!(skip_act[a].iter().all(|&v| v == 0.0), "dead pair's SwiGLU");
            assert!(skip[o].iter().all(|&v| v == 0.0), "dead pair's down");
        } else {
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&skip_act[a.clone()]),
                bits(&plain_act[a]),
                "pair {pair}"
            );
            assert_eq!(bits(&skip[o.clone()]), bits(&plain[o]), "pair {pair}");
        }
    }
}

/// The row map's source rides in the external params of an op of no split only, so a plain
/// node is the same node as before (no source, param 3 zero).
#[test]
fn row_map_param_is_off_without_a_map() {
    let weights = Context::new(2, 0);
    let w = weights.new_tensor(types::Q4_K, &[256, 64, 4]);
    let ctx = Context::new(16, 0);
    let x = ctx.new_tensor(types::F32, &[256, 1, 3]);
    let ids = ctx.new_tensor(types::I32, &[2, 3]);
    let rows = ctx.new_tensor(types::I32, &[3]);
    assert!(pad_fits(rows, ids));
    assert!(!pad_fits(ctx.new_tensor(types::I32, &[4]), ids));
    assert!(!pad_fits(ctx.new_tensor(types::F32, &[3]), ids));
    let plain = mul_mat_id_tiered(&ctx, w, x, ids, 0);
    let padded = mul_mat_id_tiered_pad(&ctx, w, x, ids, 0, Some(rows));
    // SAFETY: nodes of this context.
    let (p, q) = unsafe { (&*plain.raw(), &*padded.raw()) };
    assert!(p.src[4].is_null());
    assert_eq!(q.src[4], rows.raw());
    assert_eq!(crate::ggml::external_params(p)[P_PAD], 0);
    assert_eq!(crate::ggml::external_params(q)[P_PAD], 4);
}
