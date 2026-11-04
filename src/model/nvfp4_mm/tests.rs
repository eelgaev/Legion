use super::*;

#[test]
fn token_tiles_cover_the_batch() {
    assert_eq!(tokens_tile(1, 4), 1);
    assert_eq!(tokens_tile(8, 4), 1);
    assert_eq!(tokens_tile(9, 4), 2);
    assert_eq!(tokens_tile(17, 4), 4);
    assert_eq!(tokens_tile(100, 4), 4);
    assert_eq!(tokens_tile(129, 16), 16);
}

#[test]
fn dense_kernel_choice() {
    // Decode batches: the mv kernel with the smallest token tile.
    assert_eq!(
        pick_dense(5120, 17408, 16, 48, MV_MAX_N),
        Kernel::Mv {
            nt: 2,
            rtw: 1,
            kw: 8,
            rpt: 1
        }
    );
    // Prefill: 256-row tiles once there are three waves of them, 128-row tiles before.
    assert_eq!(
        pick_dense(5120, 17408, 64, 48, MV_MAX_N),
        Kernel::Gemm { mi: 1, nt: 8 }
    );
    assert_eq!(
        pick_dense(5120, 17408, 2048, 48, MV_MAX_N),
        Kernel::Gemm { mi: 2, nt: 16 }
    );
    // --nvfp4-wide-mv: 33-64 tokens on the mv kernel's decode tile (8 warps), 64 tokens per
    // CTA, 16 rows up to 40 tokens and 32 above; up to 32 as without it, more on the GEMM.
    let mv = |nt, rtw| Kernel::Mv {
        nt,
        rtw,
        kw: 8,
        rpt: 1,
    };
    // 24-32 tokens: four row tiles per CTA (the same sums).
    let rpt4 = Kernel::Mv {
        nt: 4,
        rtw: 1,
        kw: 8,
        rpt: 4,
    };
    assert_eq!(pick_dense(5120, 17408, 23, 48, MV_MAX_N), mv(4, 1));
    assert_eq!(pick_dense(5120, 17408, 24, 48, MV_MAX_N), rpt4);
    assert_eq!(pick_dense(17408, 5120, 32, 48, MV_MAX_N), rpt4);
    assert_eq!(pick_dense(5120, 17408, 32, 48, WIDE_MV_MAX_N), rpt4);
    assert_eq!(pick_dense(5120, 17408, 33, 48, WIDE_MV_MAX_N), mv(8, 1));
    assert_eq!(pick_dense(5120, 17408, 40, 48, WIDE_MV_MAX_N), mv(8, 1));
    assert_eq!(pick_dense(5120, 17408, 41, 48, WIDE_MV_MAX_N), mv(8, 2));
    assert_eq!(pick_dense(17408, 5120, 64, 48, WIDE_MV_MAX_N), mv(8, 2));
    assert_eq!(
        pick_dense(5120, 17408, 65, 48, WIDE_MV_MAX_N),
        Kernel::Gemm { mi: 1, nt: 16 }
    );
    for n in 1..=32 {
        assert_eq!(
            pick_dense(5120, 17408, n, 48, WIDE_MV_MAX_N),
            pick_dense(5120, 17408, n, 48, MV_MAX_N)
        );
    }
}

#[test]
fn grouped_kernel_choice() {
    // 16 decode tokens x 6 experts over 128: about one pair per expert, the mv kernel.
    assert!(matches!(
        pick_grouped(2688, 1856, 96, 128),
        Kernel::Mv { nt: 1, rpt: 8, .. }
    ));
    // Qwen3.8-Flash-Next's down projection at 16 decode tokens: 128 rows per CTA.
    let kernel = pick_grouped(640, 2560, 160, 512);
    assert_eq!(kernel.tile(), (128, 8));
    // A 4096-token prefill chunk: 192 pairs per expert, the GEMM's widest tiles.
    assert_eq!(
        pick_grouped(2688, 1856, 4096 * 6, 128),
        Kernel::Gemm { mi: 1, nt: 16 }
    );
    // Qwen3.8-Flash-Next's down projection (k = 640, 512 experts) in a 4096-token prefill
    // chunk: 256 rows by 64 tokens, two CTAs per SM (#824); its gate and up keep the defaults.
    let down = pick_grouped(640, 2560, 4096 * 10, 512);
    assert_eq!(down, Kernel::Gemm { mi: 2, nt: 8 });
    assert_eq!(grouped_gemm_tune(640, 512, 2, 8, false), GEMM_DOWN);
    assert_eq!(
        pick_grouped(2560, 640, 4096 * 10, 512),
        Kernel::Gemm { mi: 1, nt: 16 }
    );
    assert_eq!(grouped_gemm_tune(2560, 512, 2, 16, true), GEMM_DEFAULT);
    assert_eq!(grouped_gemm_tune(640, 512, 2, 8, true), GEMM_DEFAULT);
    // Fewer experts (Nemotron 3.5's 128) or a longer k: unchanged.
    assert_eq!(
        pick_grouped(640, 2560, 4096 * 10, 128),
        Kernel::Gemm { mi: 1, nt: 16 }
    );
    assert_eq!(grouped_gemm_tune(640, 128, 2, 8, false), GEMM_DEFAULT);
    assert_eq!(grouped_gemm_tune(1856, 512, 2, 8, false), GEMM_DEFAULT);
}

#[test]
fn grouped_cutoff_and_mv_tile() {
    let mv = |nt| Kernel::Mv {
        nt,
        rtw: 1,
        kw: 8,
        rpt: GROUPED_MV_RPT,
    };
    // The defaults: Qwen3.8-Flash-Next's gate/up (10 of 512 experts) at a 1,024-token prompt
    // step, 20 pairs per expert, on the mv kernel at 32-token tiles.
    assert_eq!(expert_mv_max(), MV_MAX_N);
    assert_eq!(expert_mv_tile(), 4);
    assert_eq!(
        pick_grouped_at(2560, 640, 10240, 512, MV_MAX_N, 4),
        pick_grouped(2560, 640, 10240, 512)
    );
    assert_eq!(pick_grouped_at(2560, 640, 10240, 512, MV_MAX_N, 4), mv(4));
    // Narrower widest tiles: 16 or 8 tokens; a decode step keeps one 8-token tile.
    assert_eq!(pick_grouped_at(2560, 640, 10240, 512, MV_MAX_N, 2), mv(2));
    assert_eq!(pick_grouped_at(2560, 640, 10240, 512, MV_MAX_N, 1), mv(1));
    assert_eq!(pick_grouped_at(2560, 640, 160, 512, MV_MAX_N, 2), mv(1));
    // A lower cutoff: the step on the GEMM from 9 pairs per expert, a decode step stays.
    assert!(matches!(
        pick_grouped_at(2560, 640, 10240, 512, 8, 4),
        Kernel::Gemm { .. }
    ));
    assert_eq!(pick_grouped_at(2560, 640, 4096, 512, 8, 4), mv(1));
    assert_eq!(pick_grouped_at(640, 2560, 160, 512, 1, 4), mv(1));
    assert!(matches!(
        pick_grouped_at(640, 2560, 160, 512, 0, 4),
        Kernel::Gemm { .. }
    ));
}

#[test]
fn split_modes() {
    let mv = |nt| Kernel::Mv {
        nt,
        rtw: 1,
        kw: 8,
        rpt: GROUPED_MV_RPT,
    };
    let at = |shape, glu, step_only, mv_max, mode| {
        plan_grouped_at(2560, 640, shape, glu, step_only, mv_max, 4, mode)
    };
    let down =
        |shape, mv_max, mode| plan_grouped_at(640, 2560, shape, false, false, mv_max, 4, mode);
    // Qwen3.8-Flash-Next (10 of 512 experts): a 1,024-token prompt step, a 600-token one, a
    // 128-token verify step (16 sequences of 8) and a 16-sequence decode step.
    let (p1024, p600, verify, decode) = (
        [10240, 1024, 512],
        [6000, 600, 512],
        [1280, 128, 512],
        [160, 16, 512],
    );
    // The default mode and cutoff: the step-wide choice, the mv kernel up to 32 pairs.
    assert_eq!(expert_mv_split(), MvSplit::Step);
    assert_eq!(
        plan_grouped(2560, 640, 10240, 1024, 512, true, false),
        Grouped::One(mv(4))
    );
    for mode in [
        MvSplit::Step,
        MvSplit::Expert,
        MvSplit::Down,
        MvSplit::ExpertDown,
    ] {
        for shape in [p1024, p600, verify, decode] {
            for glu in [false, true] {
                let step = Grouped::One(pick_grouped_at(2560, 640, shape[0], 512, MV_MAX_N, 4));
                if mode == MvSplit::Step || shape[1] < SPLIT_MIN_TOKENS {
                    // The default's kernels at the default cutoff: decode and verify steps in
                    // every mode, prompt steps in the step mode.
                    assert_eq!(
                        at(shape, glu, false, MV_MAX_N, mode),
                        step,
                        "{mode:?} {shape:?}"
                    );
                }
                // A row map or a clamp: the step-wide choice at any mode.
                assert_eq!(
                    at(shape, glu, true, 8, mode),
                    Grouped::One(pick_grouped_at(2560, 640, shape[0], 512, 8, 4))
                );
            }
        }
    }
    // Step: a 1,024-token step at cutoff 8 all on the GEMM.
    assert!(matches!(
        at(p1024, true, false, 8, MvSplit::Step),
        Grouped::One(Kernel::Gemm { .. })
    ));
    // Expert: the gate/up and the down split at the cutoff, the mv tile holding `cut` tokens.
    assert_eq!(
        at(p1024, true, false, 8, MvSplit::Expert),
        Grouped::Split {
            cut: 8,
            mv: mv(1),
            gemm: Kernel::Gemm { mi: 2, nt: 8 }
        }
    );
    assert_eq!(
        down(p600, 16, MvSplit::Expert),
        Grouped::Split {
            cut: 16,
            mv: mv(2),
            gemm: Kernel::Gemm { mi: 2, nt: 8 }
        }
    );
    // The default cutoff split too: the experts of more than 32 pairs on the GEMM.
    assert_eq!(
        down(p1024, MV_MAX_N, MvSplit::Expert),
        Grouped::Split {
            cut: 32,
            mv: mv(4),
            gemm: Kernel::Gemm { mi: 2, nt: 8 }
        }
    );
    // Down: the gate/up as the default's, the down step-wide at the cutoff.
    assert_eq!(
        at(p1024, true, false, 8, MvSplit::Down),
        Grouped::One(mv(4))
    );
    assert!(matches!(
        down(p1024, 8, MvSplit::Down),
        Grouped::One(Kernel::Gemm { mi: 2, nt: 8 })
    ));
    // ExpertDown: the gate/up as the default's, the down split.
    assert_eq!(
        at(p1024, true, false, 8, MvSplit::ExpertDown),
        Grouped::One(mv(4))
    );
    assert!(matches!(
        down(p1024, 8, MvSplit::ExpertDown),
        Grouped::Split { cut: 8, .. }
    ));
    // Past 32 pairs per expert (a 2,048-token step): the GEMM for every expert in every mode.
    for mode in [MvSplit::Expert, MvSplit::ExpertDown] {
        assert!(matches!(
            down([20480, 2048, 512], 8, mode),
            Grouped::One(Kernel::Gemm { .. })
        ));
    }
}

#[test]
fn split_tile_bounds() {
    // 10,240 pairs, cut 8: the GEMM's 64-token tiles, one partial tile per expert of 9+ pairs
    // (at most 512); the mv's one 8-token tile per expert of up to 8 pairs.
    assert_eq!(split_tiles(10240, 64, 512, 8, true), 160 + 512);
    assert_eq!(split_tiles(10240, 8, 512, 8, false), 512);
    // A cut wider than the mv tile: ceil(cut / cols) tiles per expert.
    assert_eq!(split_tiles(10240, 8, 512, 16, false), 1024);
    // Few pairs: at most the pairs' tiles, never more than the unsplit bound.
    assert_eq!(split_tiles(300, 64, 512, 8, true), 4 + 33);
    assert_eq!(
        split_tiles(300, 8, 512, 32, false),
        grouped_tiles(300, 8, 512, false)
    );
    for (pairs, cols, cut) in [(6000, 64, 4), (6000, 32, 16), (2560, 8, 1)] {
        for busy in [false, true] {
            assert!(
                split_tiles(pairs, cols, 512, cut, busy) <= grouped_tiles(pairs, cols, 512, false)
            );
        }
    }
}

#[test]
fn split_mode_names() {
    use clap::ValueEnum;
    for (mode, name) in [
        (MvSplit::Step, "step"),
        (MvSplit::Expert, "expert"),
        (MvSplit::Down, "down"),
        (MvSplit::ExpertDown, "expert-down"),
    ] {
        assert_eq!(mode.to_possible_value().unwrap().get_name(), name);
        assert_eq!(MvSplit::from_code(mode.code()), mode);
    }
}

#[test]
fn glu_gemm_tiles() {
    // Short prompt steps on the GEMM (--nvfp4-expert-mv-max): 64-token tiles, four stages.
    for per in [1, 5, 20, MV_MAX_N] {
        assert_eq!(glu_gemm_tile(2560, 512, per), (8, GEMM_GLU_SHORT));
    }
    assert!(gemm_tune_fits(2, 8, GEMM_GLU_SHORT));
    // Past the mv kernel's default cutoff: the earlier choice.
    assert_eq!(glu_gemm_tile(2560, 512, 33), (8, GEMM_DEFAULT));
    assert_eq!(glu_gemm_tile(2560, 512, 80), (16, GEMM_DEFAULT));
    assert_eq!(glu_gemm_tile(2560, 512, 400), (16, GEMM_DEFAULT));
}

#[test]
fn gemm_tune_codes() {
    assert_eq!(GEMM_DEFAULT.code(), 31);
    assert_eq!(GEMM_DOWN.code(), 22);
    for v in [21, 22, 31, 32, 41] {
        assert_eq!(GemmTune::from_code(v).code(), v);
    }
    assert!(gemm_tune_fits(2, 8, GEMM_DOWN));
    assert!(gemm_tune_fits(2, 16, GEMM_DEFAULT));
    assert!(gemm_tune_fits(1, 4, GEMM_DEFAULT));
    assert!(!gemm_tune_fits(1, 4, GEMM_DOWN));
    // 256 rows by 128 tokens: two stages fit one CTA, not two; four not at all.
    assert!(gemm_tune_fits(2, 16, GemmTune::from_code(21)));
    assert!(!gemm_tune_fits(2, 16, GEMM_DOWN));
    assert!(!gemm_tune_fits(2, 16, GemmTune::from_code(41)));
}

#[test]
fn bf16_rounding() {
    assert_eq!(bf16_bits(1.0), 0x3f80);
    // Ties to even: 1 + 2^-8 rounds down to 1, 1 + 3 * 2^-8 up to 1 + 2^-6.
    assert_eq!(bf16_bits(f32::from_bits(0x3f80_8000)), 0x3f80);
    assert_eq!(bf16_bits(f32::from_bits(0x3f81_8000)), 0x3f82);
    assert_eq!(bf16_bits(f32::from_bits(0x3f80_8001)), 0x3f81);
    assert_eq!(bf16_bits(f32::NAN), 0x7fc0);
}

/// The down matmul reads the SwiGLU op's plan only on that op's whole result with the same
/// routing (the same ids, or copies of the same ones); the CPU ops (no plan) still give the
/// unfused chain's values.
#[test]
fn down_shares_the_swiglu_plan() {
    use crate::ggml::{Backend, ComputeAllocator};
    let (k, m, n_expert, n_used, n_tok) = (128usize, 64usize, 8usize, 2usize, 3usize);
    let mut cpu = Backend::cpu().unwrap();
    let weights = Context::new(4, 0);
    let wg = weights.new_tensor(types::NVFP4, &[k as i64, m as i64, n_expert as i64]);
    let wu = weights.new_tensor(types::NVFP4, &[k as i64, m as i64, n_expert as i64]);
    let wd = weights.new_tensor(types::NVFP4, &[m as i64, k as i64, n_expert as i64]);
    weights.alloc_tensors(cpu.buffer_type()).unwrap();
    let block = |i: usize| -> [u8; 36] {
        let mut b = [0u8; 36];
        b[..4].copy_from_slice(&[0x30, 0x38, 0x28, 0x34]);
        for (j, v) in b[4..].iter_mut().enumerate() {
            *v = ((i * 7 + j * 13) % 251) as u8;
        }
        b
    };
    let fill = |w: Tensor<'_>, seed: usize| {
        let bytes: Vec<u8> = (0..w.nbytes() / 36).flat_map(|i| block(i + seed)).collect();
        w.set_bytes(&bytes);
    };
    fill(wg, 0);
    fill(wu, 5);
    fill(wd, 11);
    let ctx = Context::new(64, 128);
    let x = ctx
        .new_tensor(types::F32, &[k as i64, 1, n_tok as i64])
        .as_input();
    let ids = ctx
        .new_tensor(types::I32, &[n_used as i64, n_tok as i64])
        .as_input();
    let other = ctx
        .new_tensor(types::I32, &[n_used as i64, n_tok as i64])
        .as_input();
    let act = swiglu_id(&ctx, wg, wu, x, ids, None, None);
    let shared = mul_mat_id(&ctx, wd, act, ids, None);
    let routed_apart = mul_mat_id(&ctx, wd, act, other, None);
    let up = mul_mat_id(&ctx, wu, x, ids, None);
    let gate = mul_mat_id(&ctx, wg, x, ids, None);
    let chain = mul_mat_id(&ctx, wd, ctx.mul(ctx.silu(gate), up), ids, None);
    let flag = |t: Tensor<'_>| {
        // SAFETY: a view of a live node.
        unsafe { (*(*t.raw()).view_src).op_params[P_PLAN] }
    };
    assert_eq!(flag(shared), 1);
    assert_eq!(flag(routed_apart), 0);
    assert_eq!(flag(chain), 0);
    // A copy of the same ids (each op makes its own of a strided `ids`).
    let wide = ctx.new_tensor(types::I32, &[2 * n_used as i64, n_tok as i64]);
    let strided = ctx.view_2d(wide, [n_used as i64, n_tok as i64], wide.nb()[1], 0);
    let act2 = swiglu_id(&ctx, wg, wu, x, strided, None, None);
    assert_eq!(flag(mul_mat_id(&ctx, wd, act2, strided, None)), 1);

    let (shared, chain) = (shared.as_output(), chain.as_output());
    let mut graph = ctx.new_graph(64);
    graph.expand(shared);
    graph.expand(chain);
    ctx.alloc_graph(&graph, ComputeAllocator::new(cpu.buffer_type()))
        .unwrap();
    let xs: Vec<f32> = (0..k * n_tok)
        .map(|i| ((i * 37) % 17) as f32 / 8.0 - 1.0)
        .collect();
    x.set_f32(&xs);
    let routing: Vec<i32> = [1, 6, 3, 1, 7, 0].to_vec();
    ids.set_i32(&routing);
    cpu.compute(&graph).unwrap();
    let (a, b) = (shared.to_f32(), chain.to_f32());
    let top = b.iter().fold(0f32, |t, v| t.max(v.abs()));
    assert!(top > 0.0);
    // ggml's CPU silu has its own exp: one BF16 rounding of the product apart at most.
    for (a, b) in a.iter().zip(&b) {
        assert!((a - b).abs() <= 2e-2 * top, "{a} against {b}");
    }
}

#[test]
fn id_layout_keeps_the_plan_aligned() {
    for esize in [4, 2] {
        let (out, plan) = id_layout(1856, 6, 3, 128, esize);
        assert_eq!(out % 256, 0);
        assert!(out >= 1856 * 6 * 3 * esize);
        // Offsets, pairs, then the token tiles of 8, 16, 32, 64 and 128 pairs (nvfp4_plan).
        assert_eq!(plan, 129 + 18 + 5);
        assert_eq!(plan, plan_len(128, 18));
    }
}

#[test]
fn every_grouped_token_tile_has_a_plan_count() {
    // The plan counts tiles of 8 << j pairs (j < PLAN_TILES); a grouped launch at any other
    // token tile would scan the plan in every CTA past the routing's last tile (#836).
    let counted = |cols: i64| (0..PLAN_TILES).any(|j| cols == 8 << j);
    for (k, n_expert) in [(2560, 512), (640, 512), (2048, 128), (4096, 64)] {
        for n_tok in [1, 4, 16, 36, 144, 256, 1024, 2048, 4096, 8192, 32768] {
            let pairs = 10 * n_tok;
            for mv_max in [0, MV_MAX_N, i64::MAX] {
                for mv_nt in [1, 2, 4] {
                    let (_, cols) = pick_grouped_at(k, 640, pairs, n_expert, mv_max, mv_nt).tile();
                    assert!(counted(cols), "k {k} pairs {pairs} mv_max {mv_max}: {cols}");
                }
            }
            let per = pairs / pairs.min(n_expert).max(1);
            let cols = 8 * glu_gemm_tile(k, n_expert, per).0 as i64;
            assert!(counted(cols), "glu k {k} per {per}: {cols}");
        }
    }
}

#[test]
fn gemm_bench_times_each_default_on_its_exact_grid() {
    let variants = gemm_bench::variants();
    for op in [gemm_bench::Op::Glu, gemm_bench::Op::Down] {
        let of_op: Vec<_> = variants.iter().filter(|v| v.op == op).collect();
        // The default first (the bit checks' reference), its exact grid right after it.
        assert!(of_op[0].is_default());
        assert!(of_op[1].exact && !of_op[1].is_default());
        assert_eq!(of_op.iter().filter(|v| v.exact).count(), 1);
        assert!(of_op[1].name().ends_with("exact"));
    }
}

/// A row map routes each token's pairs as the token it names: the named tokens' experts on the
/// token's own input, and every token that names itself the same values as without the map.
#[test]
fn row_map_routes_padding_as_the_named_row() {
    use crate::ggml::{Backend, ComputeAllocator};
    let (k, m, n_expert, n_used, n_tok) = (128usize, 64usize, 8usize, 2usize, 4usize);
    let mut cpu = Backend::cpu().unwrap();
    let weights = Context::new(4, 0);
    let wg = weights.new_tensor(types::NVFP4, &[k as i64, m as i64, n_expert as i64]);
    let wu = weights.new_tensor(types::NVFP4, &[k as i64, m as i64, n_expert as i64]);
    let wd = weights.new_tensor(types::NVFP4, &[m as i64, k as i64, n_expert as i64]);
    weights.alloc_tensors(cpu.buffer_type()).unwrap();
    let fill = |w: Tensor<'_>, seed: usize| {
        let bytes: Vec<u8> = (0..w.nbytes() / 36)
            .flat_map(|i| {
                let mut b = [0u8; 36];
                b[..4].copy_from_slice(&[0x30, 0x38, 0x28, 0x34]);
                for (j, v) in b[4..].iter_mut().enumerate() {
                    *v = (((i + seed) * 7 + j * 13) % 251) as u8;
                }
                b
            })
            .collect();
        w.set_bytes(&bytes);
    };
    fill(wg, 0);
    fill(wu, 5);
    fill(wd, 11);
    let ctx = Context::new(64, 128);
    let input = |ty, ne: &[i64]| ctx.new_tensor(ty, ne).as_input();
    let x = input(types::F32, &[k as i64, 1, n_tok as i64]);
    let ids = input(types::I32, &[n_used as i64, n_tok as i64]);
    let mapped = input(types::I32, &[n_used as i64, n_tok as i64]);
    let rows = input(types::I32, &[n_tok as i64]);
    let down = |act, ids| mul_mat_id(&ctx, wd, act, ids, None).as_output();
    let glu_rows = |ids| swiglu_id_rows(&ctx, wg, wu, x, ids, None, None, Some(rows));
    let plain = down(swiglu_id(&ctx, wg, wu, x, ids, None, None), ids);
    let with_map = down(glu_rows(ids), ids);
    let named = down(glu_rows(ids), mapped);
    let as_named = down(swiglu_id(&ctx, wg, wu, x, mapped, None, None), mapped);
    let mut graph = ctx.new_graph(64);
    for t in [plain, with_map, named, as_named] {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(cpu.buffer_type()))
        .unwrap();
    let xs: Vec<f32> = (0..k * n_tok)
        .map(|i| ((i * 37) % 17) as f32 / 8.0 - 1.0)
        .collect();
    x.set_f32(&xs);
    // Tokens 2 and 3 are padding: they take token 0's experts.
    let routing = [1, 6, 3, 2, 7, 0, 5, 4];
    let map = [0usize, 1, 0, 0];
    let routed: Vec<i32> = map
        .iter()
        .flat_map(|&r| routing[r * n_used..(r + 1) * n_used].to_vec())
        .collect();
    ids.set_i32(&routing);
    mapped.set_i32(&routed);
    rows.set_i32(&map.map(|r| r as i32));
    cpu.compute(&graph).unwrap();
    let per_tok = k * n_used;
    let (p, w) = (plain.to_f32(), with_map.to_f32());
    assert_eq!(
        p[..2 * per_tok],
        w[..2 * per_tok],
        "the real tokens' values"
    );
    assert_ne!(
        p[2 * per_tok..],
        w[2 * per_tok..],
        "the padding's experts changed"
    );
    assert_eq!(named.to_f32(), as_named.to_f32());
}

/// Ids of -1 on nodes marked by [`skips`] (`--tp-moe-ep`'s other ranks' experts, #1176): those
/// pairs' SwiGLU and down rows are zeros, and every other pair's are the bits of the same
/// routing with the -1s on an expert.
#[test]
fn skipped_ids_give_zero_rows_and_keep_the_others() {
    use crate::ggml::{Backend, ComputeAllocator};
    let (k, m, n_expert, n_used, n_tok) = (128usize, 64usize, 8usize, 2usize, 4usize);
    let mut cpu = Backend::cpu().unwrap();
    let weights = Context::new(4, 0);
    let wg = weights.new_tensor(types::NVFP4, &[k as i64, m as i64, n_expert as i64]);
    let wu = weights.new_tensor(types::NVFP4, &[k as i64, m as i64, n_expert as i64]);
    let wd = weights.new_tensor(types::NVFP4, &[m as i64, k as i64, n_expert as i64]);
    weights.alloc_tensors(cpu.buffer_type()).unwrap();
    let fill = |w: Tensor<'_>, seed: usize| {
        let bytes: Vec<u8> = (0..w.nbytes() / 36)
            .flat_map(|i| {
                let mut b = [0u8; 36];
                b[..4].copy_from_slice(&[0x30, 0x38, 0x28, 0x34]);
                for (j, v) in b[4..].iter_mut().enumerate() {
                    *v = (((i + seed) * 7 + j * 13) % 251) as u8;
                }
                b
            })
            .collect();
        w.set_bytes(&bytes);
    };
    fill(wg, 0);
    fill(wu, 5);
    fill(wd, 11);
    let ctx = Context::new(64, 128);
    let input = |ty, ne: &[i64]| ctx.new_tensor(ty, ne).as_input();
    let x = input(types::F32, &[k as i64, 1, n_tok as i64]);
    let ids = input(types::I32, &[n_used as i64, n_tok as i64]);
    let skipped = input(types::I32, &[n_used as i64, n_tok as i64]);
    let act = swiglu_id(&ctx, wg, wu, x, ids, None, None);
    let plain = mul_mat_id(&ctx, wd, act, ids, None).as_output();
    let act_skip = skips(swiglu_id(&ctx, wg, wu, x, skipped, None, None));
    let skip = skips(mul_mat_id(&ctx, wd, act_skip, skipped, None)).as_output();
    // SAFETY: the views' nodes, made above.
    unsafe {
        assert!(skipping(&*(*act_skip.raw()).view_src));
        assert!(skipping(&*(*skip.raw()).view_src));
        assert!(!skipping(&*(*plain.raw()).view_src));
    }
    // The SwiGLU's BF16 rows, as F32.
    let f32_of = |t| ctx.cast(ctx.cont(t), types::F32).as_output();
    let (act, act_skip) = (f32_of(act), f32_of(act_skip));
    let mut graph = ctx.new_graph(64);
    for t in [act, plain, act_skip, skip] {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(cpu.buffer_type()))
        .unwrap();
    let xs: Vec<f32> = (0..k * n_tok)
        .map(|i| ((i * 37) % 17) as f32 / 8.0 - 1.0)
        .collect();
    x.set_f32(&xs);
    let routing = [1, 6, 3, 2, 7, 0, 5, 4];
    // A rank holding experts 0..4: the others' ids -1.
    let slots: Vec<i32> = routing
        .iter()
        .map(|&e| if e < 4 { e } else { -1 })
        .collect();
    ids.set_i32(&routing);
    skipped.set_i32(&slots);
    cpu.compute(&graph).unwrap();
    let rows = |t: Tensor<'_>, len: usize| -> Vec<Vec<f32>> {
        t.to_f32().chunks(len).map(<[f32]>::to_vec).collect()
    };
    let (a, a_s) = (rows(act, m), rows(act_skip, m));
    let (d, d_s) = (rows(plain, k), rows(skip, k));
    for (q, &e) in slots.iter().enumerate() {
        if e < 0 {
            assert!(a_s[q].iter().all(|&v| v == 0.0), "pair {q}: SwiGLU row");
            assert!(d_s[q].iter().all(|&v| v == 0.0), "pair {q}: down row");
        } else {
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&a[q]), bits(&a_s[q]), "pair {q}: SwiGLU row");
            assert_eq!(bits(&d[q]), bits(&d_s[q]), "pair {q}: down row");
        }
    }
}

/// A skipping plan's grid always has a tile past the experts' tiles of its real pairs, so the
/// CTAs that zero the padding pairs exist: for every split of the real pairs over the experts,
/// the tiles they take (`ceil(count / cols)` each) are fewer than `grouped_tiles` with a spare.
#[test]
fn skipping_grids_keep_a_spare_tile() {
    let mut x = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for _ in 0..2000 {
        let n_expert = [8i64, 64, 512][(next() % 3) as usize];
        let cols = [8i64, 16, 32][(next() % 3) as usize];
        let pairs = 1 + (next() % 2000) as i64;
        let real = (next() % pairs as u64) as i64;
        let mut count = vec![0i64; n_expert as usize];
        // Skewed: most pairs on a few experts, as padding remapped onto a first row's.
        for _ in 0..real {
            let e = if next() % 2 == 0 {
                next() % 4
            } else {
                next() % n_expert as u64
            };
            count[e as usize] += 1;
        }
        let tiles: i64 = count.iter().map(|c| (c + cols - 1) / cols).sum();
        assert!(
            tiles < grouped_tiles(pairs, cols, n_expert, true) as i64,
            "pairs {pairs} real {real} cols {cols} experts {n_expert}: {tiles} tiles"
        );
        assert_eq!(
            grouped_tiles(pairs, cols, n_expert, false),
            (pairs / cols + pairs.min(n_expert)) as i32
        );
    }
}
