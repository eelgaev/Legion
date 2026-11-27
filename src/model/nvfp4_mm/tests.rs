use super::*;

/// [`mul_mat_id_mv`]'s [`P_TILE`] values (#1221 r9): every variant back from its value, and
/// `mul_mat_id_with`'s 0 and 1 none.
#[test]
fn tile_params_round_trip() {
    assert_eq!(tile_variant(0), None);
    assert_eq!(tile_variant(1), None);
    for g in [GLU_MV, GLU_MV_TILE, DOWN_MV_TILE] {
        assert_eq!(tile_variant(tile_param(g)), Some(g));
    }
    let wide = GroupedMv {
        rtw: 15,
        rpt: 255,
        sb: 255,
        pft: 15,
    };
    assert_eq!(tile_variant(tile_param(wide)), Some(wide));
}

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
    // 25-32 tokens on matrices of 4096 rows or more: four row tiles of 32 rows (#1181).
    let rtw2 = Kernel::Mv {
        nt: 4,
        rtw: 2,
        kw: 8,
        rpt: 4,
    };
    assert_eq!(pick_dense(17408, 5120, 32, 48, MV_MAX_N), rtw2);
    assert_eq!(pick_dense(5120, 17408, 25, 48, MV_MAX_N), rtw2);
    assert_eq!(pick_dense(5376, 4096, 32, 48, MV_MAX_N), rtw2);
    assert_eq!(pick_dense(5376, 2048, 32, 48, MV_MAX_N), rpt4);
    assert_eq!(pick_dense(5120, 17408, 32, 48, WIDE_MV_MAX_N), rtw2);
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
    // The dense GEMM's 64-token tiles (#1181): warps in the thousands.
    assert_eq!(GEMM_VERIFY.code(), 4042);
    for v in [4031, 4042, 8032, 8041] {
        assert_eq!(GemmTune::from_code(v).code(), v);
    }
    assert_eq!(GEMM_VERIFY.rows(128), 64);
    assert_eq!(GemmTune::from_code(8041).rows(128), 128);
    assert_eq!(GEMM_DEFAULT.rows(256), 256);
    assert!(!gemm_tune_fits(1, 8, GEMM_VERIFY));
    // The streaming GEMM (#1181): its own code space (... 99), the default's rows.
    assert_eq!(GEMM_STREAM.code(), 411_099);
    assert_eq!(GEMM_STREAM_GLU.code(), 412_099);
    for t in [GEMM_STREAM, GEMM_STREAM_GLU, GemmTune::stream(1, 2, 3)] {
        assert!(t.is_stream());
        assert_eq!(GemmTune::from_code(t.code()), t);
        assert_eq!(t.rows(128), 128);
        assert!(!gemm_tune_fits(1, 8, t));
    }
    // A diagnostic variant (the bench's): the thousands above the shape.
    assert_eq!(GemmTune::from_code(1_411_099).warps, 1411);
    // CTAs of a set warp count (the bench's): ten thousands above the shape, still the stream.
    assert_eq!(GemmTune::from_code(60_412_099).warps, 60_412);
    assert!(GemmTune::from_code(120_411_099).is_stream());
    assert!(!GEMM_VERIFY.is_stream() && !GEMM_DEFAULT.is_stream());
}

#[test]
fn dense_gemm_tune_at_33_64_tokens() {
    // Gemma 4 31B's verify steps of 5-8 users (33-64 rows): the streaming GEMM from 4096
    // rows and for the fused GLU; the down projection (21504:5376, rows 4x as long as it has
    // rows) on #180's tune up to 48 tokens.
    for n in 33..=64 {
        assert_eq!(
            pick_dense(5376, 8192, n, 48, MV_MAX_N),
            Kernel::Gemm { mi: 1, nt: 8 }
        );
        assert_eq!(dense_gemm_tune(5376, 8192, n, 8, false), GEMM_STREAM);
        assert_eq!(dense_gemm_tune(5376, 4096, n, 8, false), GEMM_STREAM);
        assert_eq!(dense_gemm_tune(4096, 5376, n, 8, false), GEMM_STREAM);
        assert_eq!(dense_gemm_tune(5376, 16384, n, 8, false), GEMM_STREAM);
        assert_eq!(dense_gemm_tune(5376, 2048, n, 8, false), GEMM_DEFAULT);
        assert_eq!(dense_gemm_tune(5376, 21504, n, 8, true), GEMM_STREAM_GLU);
        // o 8192:5376 and a rank's down 10752:5376 over two machines: the stream.
        assert_eq!(dense_gemm_tune(8192, 5376, n, 8, false), GEMM_STREAM);
        assert_eq!(dense_gemm_tune(10752, 5376, n, 8, false), GEMM_STREAM);
        let down = if n <= STREAM_LONG_K_MAX_N {
            GEMM_VERIFY
        } else {
            GEMM_STREAM
        };
        assert_eq!(dense_gemm_tune(21504, 5376, n, 8, false), down, "n {n}");
        // Past GEMM_VERIFY's rows: the stream still.
        assert_eq!(dense_gemm_tune(65536, 16384, n, 8, false), GEMM_STREAM);
    }
    for n in [1, 32, 65, 128, 4096] {
        assert_eq!(
            dense_gemm_tune(5376, 8192, n, tokens_tile(n, 16).max(4), false),
            GEMM_DEFAULT
        );
        assert_eq!(
            dense_gemm_tune(5376, 21504, n, tokens_tile(n, 16).max(4), true),
            GEMM_DEFAULT
        );
    }
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

/// `mul_mat_id_with`'s `plan_of` (#1221 r7): read only on a whole `mul_mat_id` result of the
/// same routing (the same ids, or copies of the same ones) that made its own plan, at up to
/// `PLAN_SHARE_MAX_TOKENS` tokens, that node then the last source; `tile` marks the node.
#[test]
fn down_reads_an_earlier_mul_mat_ids_plan() {
    let (k, m, n_expert, n_used) = (128i64, 64i64, 8i64, 2i64);
    let cpu = crate::ggml::Backend::cpu().unwrap();
    let weights = Context::new(4, 0);
    let wgu = weights.new_tensor(types::NVFP4, &[k, 2 * m, n_expert]);
    let wd = weights.new_tensor(types::NVFP4, &[m, k, n_expert]);
    let scale = weights.new_tensor(types::F32, &[n_expert]);
    // Host memory: what `supports` takes on any machine.
    weights.alloc_tensors(cpu.buffer_type()).unwrap();
    let ctx = Context::new(256, 0);
    let node = |t: Tensor<'_>| {
        // SAFETY: a view of a live node.
        unsafe { (*t.raw()).view_src }
    };
    // SAFETY: a live node.
    let flags = |t: Tensor<'_>| unsafe {
        let p = &(*node(t)).op_params;
        (p[P_PLAN], p[P_TILE])
    };
    // SAFETY: a live node.
    let src = |t: Tensor<'_>, i: usize| unsafe { (*node(t)).src[i] };
    let none = std::ptr::null_mut();
    for n_tok in [1i64, 3, 64, 65] {
        let x = ctx.new_tensor(types::F32, &[k, 1, n_tok]);
        let ids = ctx.new_tensor(types::I32, &[n_used, n_tok]);
        let other = ctx.new_tensor(types::I32, &[n_used, n_tok]);
        let a = ctx.new_tensor(types::BF16, &[m, n_used, n_tok]);
        let shares = n_tok <= PLAN_SHARE_MAX_TOKENS;
        let shared = if shares { PLAN_OF_SOURCE } else { 0 };
        let gu = mul_mat_id_with(&ctx, wgu, x, ids, Some(scale), true, None);
        assert_eq!(flags(gu), (0, 1));
        let down = mul_mat_id_with(&ctx, wd, a, ids, Some(scale), true, Some(gu));
        assert_eq!(flags(down), (shared, 1));
        let from = if shares { node(gu) } else { none };
        assert_eq!(src(down, 4), from);
        // Another routing, a part of the result, or a node on another's plan: its own plan.
        let apart = mul_mat_id_with(&ctx, wd, a, other, Some(scale), false, Some(gu));
        assert_eq!(flags(apart), (0, 0));
        let part = ctx.view_3d(gu, [2 * m, 1, n_tok], [gu.nb()[1], gu.nb()[2]], 0);
        let d = mul_mat_id_with(&ctx, wd, a, ids, None, true, Some(part));
        assert_eq!(flags(d), (0, 1));
        let d = mul_mat_id_with(&ctx, wd, a, ids, None, true, Some(down));
        assert_eq!(flags(d), (0, 1));
        // A copy of the same ids (each op makes its own of a strided `ids`).
        let wide = ctx.new_tensor(types::I32, &[2 * n_used, n_tok]);
        let strided = ctx.view_2d(wide, [n_used, n_tok], wide.nb()[1], 0);
        let gu2 = mul_mat_id_with(&ctx, wgu, x, strided, None, true, None);
        let d = mul_mat_id_with(&ctx, wd, a, strided, None, true, Some(gu2));
        assert_eq!(flags(d), (shared, 1));
        let from = if shares { node(gu2) } else { none };
        assert_eq!(src(d, 3), from);
        // `mul_mat_id` itself stays as it was.
        assert_eq!(flags(mul_mat_id(&ctx, wd, a, ids, None)), (0, 0));
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
    // The diagnostic GEMMs (#1176 r35) are never the reference, each named once.
    let diags: Vec<_> = variants.iter().filter(|v| v.diag != 0).collect();
    assert_eq!(diags.len(), gemm_bench::DIAGS.len());
    assert!(
        diags
            .iter()
            .all(|v| v.op == gemm_bench::Op::Glu && !v.is_default())
    );
    assert!(diags.iter().all(|v| !v.name().ends_with('?')));
    // The prefetch runs (#1176 r36): each op's default GEMM at every run, named apart.
    for op in [gemm_bench::Op::Glu, gemm_bench::Op::Down] {
        let pfs: Vec<_> = variants
            .iter()
            .filter(|v| v.op == op && v.pf != 0)
            .collect();
        assert_eq!(pfs.len(), gemm_bench::PFS.len());
        assert!(pfs.iter().all(|v| !v.is_default() && v.diag == 0));
        let names: std::collections::HashSet<_> = pfs.iter().map(|v| v.name()).collect();
        assert_eq!(names.len(), pfs.len());
    }
}

/// A recorded routing replays as rank 1 of 2 sees it: its experts as local indices, the other
/// rank's as -1, decode records left out, and chunks of the requested tokens over every layer.
#[test]
fn gemm_bench_route_replays_one_ranks_experts() {
    use crate::tp_net::ep_place::write_record;
    let mut bytes = Vec::new();
    // Two layers, 8 experts, 2 per token: a 4-token prompt step each, then a 1-token decode step.
    for il in 0..2 {
        let ids: Vec<i32> = (0..8).map(|i| ((i * 3 + il) % 8) as i32).collect();
        write_record(&mut bytes, il, 8, 2, &ids).unwrap();
    }
    for il in 0..2 {
        write_record(&mut bytes, il, 8, 2, &[7, 0]).unwrap();
    }
    let r = gemm_bench::load_route(&bytes, 1, 2, 2, 2).unwrap();
    assert_eq!((r.n_expert, r.n_used, r.layers.len()), (4, 2, 2));
    // Layer 0's ids 0 3 6 1 4 7 2 5: experts 4-7 are rank 1's 0-3.
    assert_eq!(r.layers[0], vec![-1, -1, 2, -1, 0, 3, -1, 1]);
    assert_eq!(r.layers[1].len(), 8);
    // Chunks of 2 tokens: 2 per layer, spread over the 4 tokens (chunks 0 and 1 of 2), in
    // sets of 2 layers.
    let sets = gemm_bench::route_sets(&r, 2, 2).unwrap();
    assert_eq!(sets.len(), 2);
    assert_eq!(sets[0][0], vec![-1, -1, 2, -1]);
    assert_eq!(sets[1][0], vec![0, 3, -1, 1]);
    assert!(gemm_bench::route_sets(&r, 2, 5).is_err());
    assert!(gemm_bench::load_route(&bytes, 0, 2, 9, 2).is_err());
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

/// The nodes [`unread_nodes`] gives for a MoE tail that skips zero-weight rows (#1176): a
/// [`skips`] down matmul on its SwiGLU's plan, and that SwiGLU when the down matmul alone reads
/// it; nothing for unmarked nodes or a SwiGLU with a row map (its padding pairs are read).
#[test]
fn unread_skips_only_on_marked_expert_chains() {
    let (k, m, n_expert, n_used, n_tok) = (128i64, 64i64, 8i64, 2i64, 4i64);
    let cpu = crate::ggml::Backend::cpu().unwrap();
    let weights = Context::new(4, 0);
    let wg = weights.new_tensor(types::NVFP4, &[k, m, n_expert]);
    let wu = weights.new_tensor(types::NVFP4, &[k, m, n_expert]);
    let wd = weights.new_tensor(types::NVFP4, &[m, k, n_expert]);
    // In host memory: `supports` takes them on any machine (unallocated weights would need a
    // GPU with BF16 tensor cores).
    weights.alloc_tensors(cpu.buffer_type()).unwrap();
    let ctx = Context::new(64, 128);
    let x = ctx.new_tensor(types::F32, &[k, 1, n_tok]);
    let ids = ctx.new_tensor(types::I32, &[n_used, n_tok]);
    let rows = ctx.new_tensor(types::I32, &[n_tok]);
    let node = |t: Tensor<'_>| {
        // SAFETY: the result views made below; their nodes are valid tensors.
        unsafe { (*t.raw()).view_src }
    };
    let act = skips(swiglu_id(&ctx, wg, wu, x, ids, None, None));
    let down = skips(mul_mat_id(&ctx, wd, act, ids, None));
    // SAFETY: nodes made above.
    unsafe {
        assert_eq!(
            unread_nodes(node(down), |_| true),
            vec![node(down), node(act)]
        );
        assert_eq!(unread_nodes(node(down), |_| false), vec![node(down)]);
        // Not marked by `skips`: its zeros are someone's reads.
        let glu = swiglu_id(&ctx, wg, wu, x, ids, None, None);
        let plain = mul_mat_id(&ctx, wd, glu, ids, None);
        assert!(unread_nodes(node(plain), |_| true).is_empty());
        // A row map's padding pairs: their rows are read.
        let padded = skips(swiglu_id_rows(&ctx, wg, wu, x, ids, None, None, Some(rows)));
        let on_rows = skips(mul_mat_id(&ctx, wd, padded, ids, None));
        assert!(unread_nodes(node(on_rows), |_| true).is_empty());
        let n = node(down);
        leave_unread(n);
        assert!(unread(&*n));
        assert!(!unread(&*node(on_rows)));
    }
}

/// Expert ids `[n_used, n_tok]` in which expert `e` has `counts[e]` pairs: the pairs laid out
/// one expert after another, slot by slot (slot `s` of token `t` is the `s * n_tok + t`-th), so
/// no token picks an expert twice while no count is above `n_tok`.
fn counted_routing(counts: &[usize], n_used: usize, n_tok: usize) -> Vec<i32> {
    assert_eq!(counts.iter().sum::<usize>(), n_used * n_tok, "pairs");
    assert!(
        counts.iter().all(|&c| c <= n_tok),
        "a count above the tokens"
    );
    let flat: Vec<i32> = counts
        .iter()
        .enumerate()
        .flat_map(|(e, &c)| std::iter::repeat_n(e as i32, c))
        .collect();
    let mut ids = vec![0; n_used * n_tok];
    for t in 0..n_tok {
        for s in 0..n_used {
            ids[t * n_used + s] = flat[s * n_tok + t];
        }
    }
    ids
}

/// Per-expert pair counts of `n_used * n_tok` pairs over `n_expert` experts: the `special`
/// counts on experts spread over the range, the rest in small counts (1-12) on the others.
fn spread_counts(special: &[usize], n_expert: usize, n_used: usize, n_tok: usize) -> Vec<usize> {
    let mut counts = vec![0; n_expert];
    let mut spare: Vec<usize> = (0..n_expert).collect();
    for (i, &c) in special.iter().enumerate() {
        let e = i * 37 % n_expert;
        counts[e] = c;
        spare.retain(|&s| s != e);
    }
    let mut left = n_used * n_tok - special.iter().sum::<usize>();
    let fill = [3, 5, 9, 1, 12, 7];
    let mut i = 0;
    while left > 0 {
        let e = spare[i % spare.len()];
        let c = fill[i % fill.len()].min(left).min(n_tok - counts[e]);
        counts[e] += c;
        left -= c;
        i += 1;
    }
    counts
}

/// The bit test's routings (#1183): experts of 1, 15, 16, 17, 64 and more pairs (each count's
/// last token tile partial at 8, 16, 32, 64 and 128 tokens) give each expert its count, and no
/// token an expert twice.
#[test]
fn counted_routing_gives_each_expert_its_pairs() {
    let special = [
        1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 200, 300,
    ];
    let (n_expert, n_used, n_tok) = (512, 10, 400);
    let counts = spread_counts(&special, n_expert, n_used, n_tok);
    let ids = counted_routing(&counts, n_used, n_tok);
    let mut got = vec![0; n_expert];
    for row in ids.chunks_exact(n_used) {
        let mut seen = row.to_vec();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), n_used, "a token picks an expert twice: {row:?}");
        for &e in row {
            got[e as usize] += 1;
        }
    }
    assert_eq!(got, counts);
    for (i, &c) in special.iter().enumerate() {
        assert_eq!(counts[i * 37 % n_expert], c);
    }
}

/// The activation prefetch (#1183) is the down projection's grouped GEMM tile's only: not the
/// SwiGLU's, not a down projection of narrow k or few experts.
#[test]
fn down_x_prefetch_only_on_the_down_gemm() {
    assert_eq!(DOWN_XPF, 4);
    assert_eq!(DOWN_XPF >> 1, DOWN_XPF_AHEAD);
    // Qwen3.8-Flash-Next's down projection (k 640, 512 experts) at prompt steps.
    assert_eq!(grouped_gemm_xpf(640, 512, 2, 8, false), DOWN_XPF);
    // Other tiles and the SwiGLU GEMM: none.
    assert_eq!(grouped_gemm_xpf(640, 512, 2, 16, false), 0);
    assert_eq!(grouped_gemm_xpf(640, 512, 1, 8, false), 0);
    assert_eq!(grouped_gemm_xpf(2560, 512, 2, 16, true), 0);
    assert_eq!(grouped_gemm_xpf(2560, 512, 2, 8, true), 0);
    // Not a wide down projection's shape.
    assert_eq!(grouped_gemm_xpf(2560, 512, 2, 8, false), 0);
    assert_eq!(grouped_gemm_xpf(640, 64, 2, 8, false), 0);
}

/// The CUDA tests of the activation prefetch and the down epilogue share the process-wide
/// kernel overrides and switches (so does the epilogue's dispatch test).
static XPF_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(feature = "cuda")]
fn xpf_gpu() -> crate::ggml::Backend {
    use crate::ggml::{DeviceKind, device_by_kind};
    device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap()
}

/// Every override the activation prefetch tests set, back at its default.
#[cfg(feature = "cuda")]
fn xpf_reset() {
    force_kernel(None);
    force_gemm_tune(None, None);
    force_gemm_xpf(None);
    force_gemm_wst(None);
    set_glu_wide_store(true);
    set_glu_store_evict_first(false);
    set_expert_mv_max(MV_MAX_N);
}

/// Random BF16 values (bits) of `n` elements.
#[cfg(feature = "cuda")]
fn xpf_bf16(n: usize, seed: u64) -> Vec<u8> {
    let mut rng = gemm_bench::Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    (0..n)
        .flat_map(|_| bf16_bits(rng.value()).to_le_bytes())
        .collect()
}

/// One bit test case: a routed SwiGLU FFN of `n_expert` NVFP4 experts (`hidden` by `ffn`),
/// `n_used` per token, the routing `ids`; the down projection's input of `n_x` rows a token.
#[cfg(feature = "cuda")]
struct XpfCase {
    name: &'static str,
    hidden: usize,
    ffn: usize,
    n_expert: usize,
    n_used: usize,
    n_x: usize,
    ids: Vec<i32>,
}

/// The activation prefetches the bit test compares with none (0): own rows only, tiles ahead
/// only, both, the default, and a distance past every routed pair.
#[cfg(feature = "cuda")]
const XPF_CHECKED: [i32; 8] = [1, 2, 3, 9, DOWN_XPF, 10 << 1, 1 | 10 << 1, 1 | 1000 << 1];

/// Runs `graph` under every activation prefetch of [`XPF_CHECKED`] (and the default) and checks
/// that its outputs `outs` are the bits of the run without one.
#[cfg(feature = "cuda")]
fn xpf_check(
    gpu: &mut crate::ggml::Backend,
    graph: &crate::ggml::Graph<'_>,
    outs: &[Tensor<'_>],
    what: &str,
) {
    let read = |outs: &[Tensor<'_>]| -> Vec<Vec<u8>> { outs.iter().map(|o| o.bytes()).collect() };
    force_gemm_xpf(Some(0));
    gpu.compute(graph).unwrap();
    let want = read(outs);
    for xpf in XPF_CHECKED.map(Some).into_iter().chain([None]) {
        force_gemm_xpf(xpf);
        gpu.compute(graph).unwrap();
        for (i, (got, want)) in read(outs).iter().zip(&want).enumerate() {
            assert_eq!(got.len(), want.len());
            let differ = got.iter().zip(want).filter(|(a, b)| a != b).count();
            assert_eq!(
                differ, 0,
                "{what}, output {i}, xpf {xpf:?}: {differ} bytes differ"
            );
        }
    }
    force_gemm_xpf(None);
}

/// The down projection (F32 and BF16 rows) and the SwiGLU GEMM of `case` on `gpu`, each tile
/// shape the grouped GEMM has, with and without the activation prefetch: the same bits.
#[cfg(feature = "cuda")]
fn xpf_case(gpu: &mut crate::ggml::Backend, case: &XpfCase) {
    use crate::ggml::ComputeAllocator;
    let (hidden, ffn, n_expert, n_used) = (case.hidden, case.ffn, case.n_expert, case.n_used);
    let n_tok = case.ids.len() / n_used;
    let blocks = |k: usize, m: usize| k / 64 * 36 * m * n_expert;
    let weights = Context::new(8, 0);
    let wd = weights.new_tensor(types::NVFP4, &[ffn as i64, hidden as i64, n_expert as i64]);
    let wg = weights.new_tensor(types::NVFP4, &[hidden as i64, ffn as i64, n_expert as i64]);
    let wu = weights.new_tensor(types::NVFP4, &[hidden as i64, ffn as i64, n_expert as i64]);
    let scales = [0, 1, 2].map(|_| weights.new_tensor(types::F32, &[n_expert as i64]));
    weights.alloc_tensors(gpu.buffer_type()).unwrap();
    wd.set_bytes(&gemm_bench::weights(blocks(ffn, hidden), 3));
    wg.set_bytes(&gemm_bench::weights(blocks(hidden, ffn), 4));
    wu.set_bytes(&gemm_bench::weights(blocks(hidden, ffn), 5));
    for (i, s) in scales.iter().enumerate() {
        let v: Vec<f32> = (0..n_expert)
            .map(|e| 0.002 + ((e * 7 + i * 3) % 1000) as f32 * 1e-5)
            .collect();
        s.set_f32(&v);
    }
    // The down projection, F32 and BF16 rows in one graph.
    let ctx = Context::new(16, 32);
    let xd = ctx
        .new_tensor(types::BF16, &[ffn as i64, case.n_x as i64, n_tok as i64])
        .as_input();
    let id = ctx
        .new_tensor(types::I32, &[n_used as i64, n_tok as i64])
        .as_input();
    let down_f32 = mul_mat_id_act(&ctx, wd, xd, id, Some(scales[0]), Act::None).as_output();
    let down_bf16 = mul_mat_id_act(&ctx, wd, xd, id, Some(scales[0]), Act::Bf16).as_output();
    let mut down = ctx.new_graph(32);
    down.expand(down_f32);
    down.expand(down_bf16);
    ctx.alloc_graph(&down, ComputeAllocator::new(gpu.buffer_type()))
        .unwrap();
    xd.set_bytes(&xpf_bf16(ffn * case.n_x * n_tok, 9));
    id.set_i32(&case.ids);
    // The grouped GEMM's tiles (the default down tile first) and the default choice.
    for kernel in [
        Some(Kernel::Gemm { mi: 2, nt: 8 }),
        Some(Kernel::Gemm { mi: 2, nt: 16 }),
        Some(Kernel::Gemm { mi: 1, nt: 8 }),
        Some(Kernel::Gemm { mi: 1, nt: 4 }),
        None,
    ] {
        force_kernel(kernel);
        let what = format!("{} down {kernel:?}", case.name);
        xpf_check(gpu, &down, &[down_f32, down_bf16], &what);
    }
    force_kernel(None);
    // The SwiGLU GEMM at its 64- and 128-token tiles, every expert on the GEMM.
    let ctx = Context::new(16, 16);
    let xg = ctx
        .new_tensor(types::BF16, &[hidden as i64, 1, n_tok as i64])
        .as_input();
    let id = ctx
        .new_tensor(types::I32, &[n_used as i64, n_tok as i64])
        .as_input();
    let glu = swiglu_id(&ctx, wg, wu, xg, id, Some(scales[1]), Some(scales[2])).as_output();
    let mut swiglu = ctx.new_graph(16);
    swiglu.expand(glu);
    ctx.alloc_graph(&swiglu, ComputeAllocator::new(gpu.buffer_type()))
        .unwrap();
    xg.set_bytes(&xpf_bf16(hidden * n_tok, 11));
    id.set_i32(&case.ids);
    set_expert_mv_max(0);
    for nt in [8, 16] {
        force_gemm_tune(None, Some(nt));
        xpf_check(
            gpu,
            &swiglu,
            &[glu],
            &format!("{} swiglu nt {nt}", case.name),
        );
    }
    xpf_reset();
}

/// The activation prefetch (#1183) keeps every grouped GEMM's bits: the down projection (F32
/// and BF16 rows, every tile shape) and the SwiGLU GEMM, with and without it, own rows and tiles
/// ahead, over hand-made and random routings (experts of 1, 15, 16, 17, 64 and more pairs, last
/// tiles partial at every tile width; a down projection of 320 rows: a partial row tile).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_down_x_prefetch_keeps_bits() {
    let _lock = XPF_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut gpu = xpf_gpu();
    let special = [
        1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 200, 300,
    ];
    let small = [1, 15, 16, 17, 64, 65, 130];
    let cases = [
        // Qwen3.8-Flash-Next's routed experts (hidden 2560, 640 per expert, 512, top 10).
        XpfCase {
            name: "flash-next counted",
            hidden: 2560,
            ffn: 640,
            n_expert: 512,
            n_used: 10,
            n_x: 10,
            ids: counted_routing(&spread_counts(&special, 512, 10, 400), 10, 400),
        },
        XpfCase {
            name: "flash-next random 512",
            hidden: 2560,
            ffn: 640,
            n_expert: 512,
            n_used: 10,
            n_x: 10,
            ids: gemm_bench::routing(10, 1500, 512, 512, 7),
        },
        XpfCase {
            name: "flash-next random 64",
            hidden: 2560,
            ffn: 640,
            n_expert: 512,
            n_used: 10,
            n_x: 10,
            ids: gemm_bench::routing(10, 700, 512, 64, 8),
        },
        // A small shape: 320 output rows (a partial row tile), one input row per token.
        XpfCase {
            name: "small shared x",
            hidden: 320,
            ffn: 256,
            n_expert: 24,
            n_used: 4,
            n_x: 1,
            ids: counted_routing(&spread_counts(&small, 24, 4, 200), 4, 200),
        },
        XpfCase {
            name: "small",
            hidden: 320,
            ffn: 256,
            n_expert: 24,
            n_used: 4,
            n_x: 4,
            ids: counted_routing(&spread_counts(&small, 24, 4, 200), 4, 200),
        },
    ];
    for case in &cases {
        xpf_case(&mut gpu, case);
        eprintln!("{}: the same bits", case.name);
    }
}

/// The down projection's grouped GEMM (Qwen3.8-Flash-Next, 512 experts, top 10, BF16 rows as
/// `--nvfp4-expert-rows-bf16`) at 12,288- and 8,067-token prompt steps without (old) and with
/// the activation prefetch (#1183), and the SwiGLU GEMM's at the same steps: microseconds a
/// layer (median and best of the rounds, the configurations taking turns, each round starting
/// one further on), the bytes moved a second at the median, and whether the bits are the old
/// ones. `*` marks the default.
/// `cargo test --release --features cuda --lib -- --ignored cuda_down_x_prefetch_bench
/// --nocapture`.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU; a benchmark"]
fn cuda_down_x_prefetch_bench() {
    use crate::ggml::ComputeAllocator;
    let _lock = XPF_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut gpu = xpf_gpu();
    let (hidden, ffn, n_expert, n_used) = (2560usize, 640usize, 512usize, 10usize);
    const LAYERS: usize = 4;
    const ROUNDS: usize = 5;
    let blocks = |k: usize, m: usize| k / 64 * 36 * m * n_expert;
    let weights = Context::new(8, 0);
    let wd = weights.new_tensor(types::NVFP4, &[ffn as i64, hidden as i64, n_expert as i64]);
    let wg = weights.new_tensor(types::NVFP4, &[hidden as i64, ffn as i64, n_expert as i64]);
    let wu = weights.new_tensor(types::NVFP4, &[hidden as i64, ffn as i64, n_expert as i64]);
    let scales = [0, 1, 2].map(|_| weights.new_tensor(types::F32, &[n_expert as i64]));
    weights.alloc_tensors(gpu.buffer_type()).unwrap();
    wd.set_bytes(&gemm_bench::weights(blocks(ffn, hidden), 3));
    wg.set_bytes(&gemm_bench::weights(blocks(hidden, ffn), 4));
    wu.set_bytes(&gemm_bench::weights(blocks(hidden, ffn), 5));
    for s in scales {
        s.set_f32(&vec![0.004; n_expert]);
    }
    let down_xpfs: [(&str, i32); 11] = [
        ("old (none)", 0),
        ("own rows", 1),
        ("own + 1 ahead", 1 | 1 << 1),
        ("own + 2 ahead", 1 | 2 << 1),
        ("own + 4 ahead", 1 | 4 << 1),
        ("own + 10 ahead", 1 | 10 << 1),
        ("2 ahead only", 2 << 1),
        ("4 ahead only", 4 << 1),
        ("6 ahead only", 6 << 1),
        ("10 ahead only", 10 << 1),
        ("16 ahead only", 16 << 1),
    ];
    let glu_xpfs: [(&str, i32); 4] = [
        ("old (none)", 0),
        ("own rows", 1),
        ("own + 1 ahead", 1 | 1 << 1),
        ("own + 10 ahead", 1 | 10 << 1),
    ];
    for n_tok in [12288usize, 8067] {
        for glu in [false, true] {
            let ctx = Context::new(16 + 24 * LAYERS, 16 * LAYERS + 16);
            let (k, x_rows) = if glu { (hidden, 1) } else { (ffn, n_used) };
            let x = ctx
                .new_tensor(types::BF16, &[k as i64, x_rows as i64, n_tok as i64])
                .as_input();
            let ids: Vec<_> = (0..LAYERS)
                .map(|_| {
                    ctx.new_tensor(types::I32, &[n_used as i64, n_tok as i64])
                        .as_input()
                })
                .collect();
            let mut graph = ctx.new_graph(16 * LAYERS + 16);
            let outs: Vec<_> = ids
                .iter()
                .map(|&id| {
                    let y = if glu {
                        swiglu_id(&ctx, wg, wu, x, id, Some(scales[1]), Some(scales[2]))
                    } else {
                        mul_mat_id_act(&ctx, wd, x, id, Some(scales[0]), Act::Bf16)
                    }
                    .as_output();
                    graph.expand(y);
                    y
                })
                .collect();
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .unwrap();
            x.set_bytes(&xpf_bf16(k * x_rows * n_tok, 5 + n_tok as u64));
            for (l, id) in ids.iter().enumerate() {
                id.set_i32(&gemm_bench::routing(
                    n_used,
                    n_tok,
                    n_expert,
                    n_expert,
                    1 + l as u64,
                ));
            }
            let variants: &[(&str, i32)] = if glu { &glu_xpfs } else { &down_xpfs };
            // A warm-up run of each (its bits against the old one's), then the rounds, the
            // variants taking turns.
            let mut want = Vec::new();
            let mut same = vec![true; variants.len()];
            for (vi, &(_, xpf)) in variants.iter().enumerate() {
                force_gemm_xpf(Some(xpf));
                gpu.compute(&graph).unwrap();
                let got: Vec<Vec<u8>> = outs.iter().map(|o| o.bytes()).collect();
                if vi == 0 {
                    want = got;
                } else {
                    same[vi] = got == want;
                }
            }
            let nv = variants.len();
            let mut times = vec![Vec::with_capacity(ROUNDS); nv];
            for round in 0..ROUNDS {
                for i in 0..nv {
                    let vi = (round + i) % nv;
                    force_gemm_xpf(Some(variants[vi].1));
                    let t = std::time::Instant::now();
                    gpu.compute(&graph).unwrap();
                    times[vi].push(t.elapsed().as_secs_f64() / LAYERS as f64);
                }
            }
            force_gemm_xpf(None);
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            let median: Vec<f64> = times.iter().map(|t| t[t.len() / 2]).collect();
            let best: Vec<f64> = times.iter().map(|t| t[0]).collect();
            // The default of this op ([`grouped_gemm_xpf`]; the SwiGLU's is none).
            let default = if glu {
                0
            } else {
                grouped_gemm_xpf(ffn as i64, n_expert as i64, 2, 8, false)
            };
            // Bytes a layer: every expert's weights, each pair's input row and output row.
            let pairs = (n_used * n_tok) as f64;
            let (w_bytes, x_bytes, out_bytes) = if glu {
                (
                    2.0 * blocks(hidden, ffn) as f64,
                    (n_tok * hidden * 2) as f64,
                    pairs * ffn as f64 * 2.0,
                )
            } else {
                (
                    blocks(ffn, hidden) as f64,
                    pairs * ffn as f64 * 2.0,
                    pairs * hidden as f64 * 2.0,
                )
            };
            let bytes = w_bytes + x_bytes + out_bytes;
            let op = if glu { "swiglu" } else { "down (bf16 rows)" };
            eprintln!(
                "{op} at {n_tok} tokens ({:.0} MB a layer: {:.0} weights, {:.0} in, {:.0} out)",
                bytes / 1e6,
                w_bytes / 1e6,
                x_bytes / 1e6,
                out_bytes / 1e6
            );
            for (vi, &(name, xpf)) in variants.iter().enumerate() {
                eprintln!(
                    "  {}{name:<16} xpf {xpf:>3}: median {:>8.1} us/layer ({:>5.3}x old), \
                     min {:>8.1} ({:>5.3}x)  {:>6.1} GB/s  {}",
                    if xpf == default { '*' } else { ' ' },
                    median[vi] * 1e6,
                    median[0] / median[vi],
                    best[vi] * 1e6,
                    best[0] / best[vi],
                    bytes / median[vi] / 1e9,
                    if same[vi] { "same bits" } else { "BITS DIFFER" }
                );
            }
            assert!(same.iter().all(|&s| s), "{op} at {n_tok}: bits differ");
        }
    }
    xpf_reset();
}

/// The staged epilogue (#1183) is the down projection's grouped GEMM tile's only: not the
/// SwiGLU's, not a down projection of wide k or few experts; streaming stores for BF16 rows
/// only; the switches turn them off. Holds [`XPF_LOCK`]: the switches are process-wide.
#[test]
fn down_wide_store_only_on_the_down_gemm() {
    let _lock = XPF_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(DOWN_WST_BF16, WST_WIDE | WST_CS);
    assert_eq!(DOWN_WST_F32, WST_WIDE);
    assert_eq!(WST_WIDE & WST_CS, 0);
    for bf16 in [false, true] {
        let down = if bf16 { DOWN_WST_BF16 } else { DOWN_WST_F32 };
        // Qwen3.8-Flash-Next's down projection (k 640, 512 experts) at prompt steps.
        assert_eq!(grouped_gemm_wst(640, 512, 2, 8, false, bf16), down);
        // Other tiles and the SwiGLU GEMM: none.
        assert_eq!(grouped_gemm_wst(640, 512, 2, 16, false, bf16), 0);
        assert_eq!(grouped_gemm_wst(640, 512, 1, 8, false, bf16), 0);
        assert_eq!(grouped_gemm_wst(2560, 512, 2, 16, true, bf16), 0);
        assert_eq!(grouped_gemm_wst(2560, 512, 2, 8, true, bf16), 0);
        // Not a wide down projection's shape.
        assert_eq!(grouped_gemm_wst(2560, 512, 2, 8, false, bf16), 0);
        assert_eq!(grouped_gemm_wst(640, 64, 2, 8, false, bf16), 0);
    }
    // --no-nvfp4-down-store-evict-first: staged only, either rows.
    set_down_store_evict_first(false);
    assert_eq!(grouped_gemm_wst(640, 512, 2, 8, false, true), WST_WIDE);
    assert_eq!(grouped_gemm_wst(640, 512, 2, 8, false, false), WST_WIDE);
    // --no-nvfp4-down-wide-store: neither, whatever the other says.
    for cs in [false, true] {
        set_down_store_evict_first(cs);
        set_down_wide_store(false);
        assert_eq!(grouped_gemm_wst(640, 512, 2, 8, false, true), 0);
        assert_eq!(grouped_gemm_wst(640, 512, 2, 8, false, false), 0);
        set_down_wide_store(true);
    }
    set_down_store_evict_first(true);
}

/// The epilogues the staged-store bit test compares with the fragments' own stores (0).
#[cfg(feature = "cuda")]
const WST_CHECKED: [i32; 3] = [WST_WIDE, WST_CS, WST_WIDE | WST_CS];

/// One staged-store bit test case: a down projection of `n_expert` NVFP4 experts of `m` rows by
/// `k`, `n_used` per token, the routing `ids`, `n_x` input rows a token.
#[cfg(feature = "cuda")]
struct WstCase {
    name: &'static str,
    m: usize,
    k: usize,
    n_expert: usize,
    n_used: usize,
    n_x: usize,
    ids: Vec<i32>,
}

/// `case`'s down projection on the down tile ([`GEMM_DOWN`], forced: the small shapes are no
/// default down projection) with F32, BF16 and squared-ReLU BF16 rows, without and with the
/// activation prefetch: every epilogue of [`WST_CHECKED`] (and the defaults, with and without
/// the BF16 rows' streaming stores) stores the bits of the fragments' own stores. The outputs
/// are filled with a pattern before each run, so a value an epilogue leaves out shows.
#[cfg(feature = "cuda")]
fn wst_case(gpu: &mut crate::ggml::Backend, case: &WstCase) {
    use crate::ggml::ComputeAllocator;
    let (m, k, n_expert, n_used) = (case.m, case.k, case.n_expert, case.n_used);
    let n_tok = case.ids.len() / n_used;
    let weights = Context::new(4, 0);
    let wd = weights.new_tensor(types::NVFP4, &[k as i64, m as i64, n_expert as i64]);
    let scale = weights.new_tensor(types::F32, &[n_expert as i64]);
    weights.alloc_tensors(gpu.buffer_type()).unwrap();
    wd.set_bytes(&gemm_bench::weights(k / 64 * 36 * m * n_expert, 13));
    let s: Vec<f32> = (0..n_expert)
        .map(|e| 0.002 + (e * 7 % 1000) as f32 * 1e-5)
        .collect();
    scale.set_f32(&s);
    let ctx = Context::new(16, 32);
    let x = ctx
        .new_tensor(types::BF16, &[k as i64, case.n_x as i64, n_tok as i64])
        .as_input();
    let id = ctx
        .new_tensor(types::I32, &[n_used as i64, n_tok as i64])
        .as_input();
    let outs = [Act::None, Act::Bf16, Act::ReluSqr]
        .map(|act| mul_mat_id_act(&ctx, wd, x, id, Some(scale), act).as_output());
    let mut graph = ctx.new_graph(32);
    for o in outs {
        graph.expand(o);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
        .unwrap();
    x.set_bytes(&xpf_bf16(k * case.n_x * n_tok, 17));
    id.set_i32(&case.ids);
    let run = |gpu: &mut crate::ggml::Backend| -> Vec<Vec<u8>> {
        for o in outs {
            o.set_bytes(&vec![0xa5; o.nbytes()]);
        }
        gpu.compute(&graph).unwrap();
        outs.iter().map(|o| o.bytes()).collect()
    };
    force_kernel(Some(Kernel::Gemm { mi: 2, nt: 8 }));
    force_gemm_tune(Some(GEMM_DOWN), None);
    for xpf in [Some(0), None] {
        force_gemm_xpf(xpf);
        force_gemm_wst(Some(0));
        let want = run(gpu);
        // The forced epilogues, then the defaults (per output type) with the streaming
        // stores on and off.
        let tries = WST_CHECKED
            .map(|w| (Some(w), true))
            .into_iter()
            .chain([(None, true), (None, false)]);
        for (wst, cs) in tries {
            force_gemm_wst(wst);
            set_down_store_evict_first(cs);
            for (i, (got, want)) in run(gpu).iter().zip(&want).enumerate() {
                assert_eq!(got.len(), want.len());
                let differ = got.iter().zip(want).filter(|(a, b)| a != b).count();
                assert_eq!(
                    differ, 0,
                    "{}, output {i} (F32, BF16, relu^2), xpf {xpf:?}, wst {wst:?}, \
                     evict-first {cs}: {differ} bytes differ",
                    case.name
                );
            }
        }
    }
    set_down_store_evict_first(true);
    xpf_reset();
}

/// The down tile's staged 16-byte stores and streaming stores (#1183) keep every output bit:
/// F32, BF16 and squared-ReLU BF16 rows, with and without the activation prefetch, over
/// hand-made routings (experts of 1, 15, 16, 17, 63, 64, 65 and 200 pairs: last token tiles
/// partial and full), random ones over 512 and 64 experts, one input row per token, and down
/// projections of 320 rows (a partial row tile) and 324 (rows not a multiple of 8: the BF16
/// rows' element-wise tail).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_down_wide_store_keeps_bits() {
    let _lock = XPF_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut gpu = xpf_gpu();
    let special = [1, 15, 16, 17, 63, 64, 65, 200];
    let small = [1, 15, 16, 17, 63, 64, 65];
    let cases = [
        // Qwen3.8-Flash-Next's down projection (2560 rows, k 640, 512 experts, top 10).
        WstCase {
            name: "flash-next counted",
            m: 2560,
            k: 640,
            n_expert: 512,
            n_used: 10,
            n_x: 10,
            ids: counted_routing(&spread_counts(&special, 512, 10, 400), 10, 400),
        },
        WstCase {
            name: "flash-next random 512",
            m: 2560,
            k: 640,
            n_expert: 512,
            n_used: 10,
            n_x: 10,
            ids: gemm_bench::routing(10, 1500, 512, 512, 7),
        },
        WstCase {
            name: "flash-next random 64",
            m: 2560,
            k: 640,
            n_expert: 512,
            n_used: 10,
            n_x: 10,
            ids: gemm_bench::routing(10, 700, 512, 64, 8),
        },
        WstCase {
            name: "flash-next shared x",
            m: 2560,
            k: 640,
            n_expert: 512,
            n_used: 10,
            n_x: 1,
            ids: gemm_bench::routing(10, 300, 512, 512, 9),
        },
        // Small shapes: 320 rows (a partial row tile) and 324 (not a multiple of 8).
        WstCase {
            name: "small 320 rows",
            m: 320,
            k: 256,
            n_expert: 24,
            n_used: 4,
            n_x: 4,
            ids: counted_routing(&spread_counts(&small, 24, 4, 200), 4, 200),
        },
        WstCase {
            name: "small 324 rows",
            m: 324,
            k: 256,
            n_expert: 24,
            n_used: 4,
            n_x: 4,
            ids: counted_routing(&spread_counts(&small, 24, 4, 200), 4, 200),
        },
        WstCase {
            name: "small 324 rows shared x",
            m: 324,
            k: 128,
            n_expert: 24,
            n_used: 4,
            n_x: 1,
            ids: counted_routing(&spread_counts(&small, 24, 4, 200), 4, 200),
        },
    ];
    for case in &cases {
        wst_case(&mut gpu, case);
        eprintln!("{}: the same bits", case.name);
    }
}

/// The down projection's grouped GEMM (Qwen3.8-Flash-Next, 512 experts, top 10) at 12,288- and
/// 8,067-token prompt steps, BF16 rows (`--nvfp4-expert-rows-bf16`) and F32 rows, with the
/// fragments' own stores (old) and each staged / streaming epilogue (#1183), the activation
/// prefetch at its default: microseconds a layer (median and best of the rounds, the
/// configurations taking turns, each round starting one further on), the bytes moved a second
/// at the median, and whether the bits are the old ones. `*` marks the default.
/// `cargo test --release --features cuda --lib -- --ignored cuda_down_wide_store_bench
/// --nocapture`.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU; a benchmark"]
fn cuda_down_wide_store_bench() {
    use crate::ggml::ComputeAllocator;
    let _lock = XPF_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    xpf_reset();
    let mut gpu = xpf_gpu();
    let (hidden, ffn, n_expert, n_used) = (2560usize, 640usize, 512usize, 10usize);
    const LAYERS: usize = 4;
    const ROUNDS: usize = 5;
    let blocks = ffn / 64 * 36 * hidden * n_expert;
    let weights = Context::new(4, 0);
    let wd = weights.new_tensor(types::NVFP4, &[ffn as i64, hidden as i64, n_expert as i64]);
    let scale = weights.new_tensor(types::F32, &[n_expert as i64]);
    weights.alloc_tensors(gpu.buffer_type()).unwrap();
    wd.set_bytes(&gemm_bench::weights(blocks, 3));
    scale.set_f32(&vec![0.004; n_expert]);
    let variants: [(&str, i32); 4] = [
        ("old (fragments)", 0),
        ("wide", WST_WIDE),
        ("old + cs", WST_CS),
        ("wide + cs", WST_WIDE | WST_CS),
    ];
    for n_tok in [12288usize, 8067] {
        for act in [Act::Bf16, Act::None] {
            let default =
                grouped_gemm_wst(ffn as i64, n_expert as i64, 2, 8, false, act != Act::None);
            let ctx = Context::new(16 + 24 * LAYERS, 16 * LAYERS + 16);
            let x = ctx
                .new_tensor(types::BF16, &[ffn as i64, n_used as i64, n_tok as i64])
                .as_input();
            let ids: Vec<_> = (0..LAYERS)
                .map(|_| {
                    ctx.new_tensor(types::I32, &[n_used as i64, n_tok as i64])
                        .as_input()
                })
                .collect();
            let mut graph = ctx.new_graph(16 * LAYERS + 16);
            let outs: Vec<_> = ids
                .iter()
                .map(|&id| {
                    let y = mul_mat_id_act(&ctx, wd, x, id, Some(scale), act).as_output();
                    graph.expand(y);
                    y
                })
                .collect();
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .unwrap();
            x.set_bytes(&xpf_bf16(ffn * n_used * n_tok, 5 + n_tok as u64));
            for (l, id) in ids.iter().enumerate() {
                id.set_i32(&gemm_bench::routing(
                    n_used,
                    n_tok,
                    n_expert,
                    n_expert,
                    1 + l as u64,
                ));
            }
            // A warm-up run of each (its bits against the old one's), then the rounds, the
            // variants taking turns.
            let mut want = Vec::new();
            let mut same = vec![true; variants.len()];
            for (vi, &(_, wst)) in variants.iter().enumerate() {
                force_gemm_wst(Some(wst));
                gpu.compute(&graph).unwrap();
                let got: Vec<Vec<u8>> = outs.iter().map(|o| o.bytes()).collect();
                if vi == 0 {
                    want = got;
                } else {
                    same[vi] = got == want;
                }
            }
            let nv = variants.len();
            let mut times = vec![Vec::with_capacity(ROUNDS); nv];
            for round in 0..ROUNDS {
                for i in 0..nv {
                    let vi = (round + i) % nv;
                    force_gemm_wst(Some(variants[vi].1));
                    let t = std::time::Instant::now();
                    gpu.compute(&graph).unwrap();
                    times[vi].push(t.elapsed().as_secs_f64() / LAYERS as f64);
                }
            }
            force_gemm_wst(None);
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            let median: Vec<f64> = times.iter().map(|t| t[t.len() / 2]).collect();
            let best: Vec<f64> = times.iter().map(|t| t[0]).collect();
            // Bytes a layer: every expert's weights, each pair's input row and output row.
            let pairs = (n_used * n_tok) as f64;
            let f32_rows = act == Act::None;
            let esize = if f32_rows { 4.0 } else { 2.0 };
            let (w_bytes, x_bytes, out_bytes) = (
                blocks as f64,
                pairs * ffn as f64 * 2.0,
                pairs * hidden as f64 * esize,
            );
            let bytes = w_bytes + x_bytes + out_bytes;
            let rows = if f32_rows { "f32" } else { "bf16" };
            eprintln!(
                "down ({rows} rows) at {n_tok} tokens ({:.0} MB a layer: {:.0} weights, {:.0} in, \
                 {:.0} out)",
                bytes / 1e6,
                w_bytes / 1e6,
                x_bytes / 1e6,
                out_bytes / 1e6
            );
            for (vi, &(name, wst)) in variants.iter().enumerate() {
                eprintln!(
                    "  {}{name:<16} wst {wst}: median {:>8.1} us/layer ({:>5.3}x old), \
                     min {:>8.1} ({:>5.3}x)  {:>6.1} GB/s  {}",
                    if wst == default { '*' } else { ' ' },
                    median[vi] * 1e6,
                    median[0] / median[vi],
                    best[vi] * 1e6,
                    best[0] / best[vi],
                    bytes / median[vi] / 1e9,
                    if same[vi] { "same bits" } else { "BITS DIFFER" }
                );
            }
            assert!(
                same.iter().all(|&s| s),
                "down ({rows} rows) at {n_tok}: bits differ"
            );
        }
    }
    xpf_reset();
}

/// The SwiGLU GEMM's staged epilogue (#1183): its 64- and 128-token tiles of the default kernel
/// and the 64-token tile of [`GEMM_GLU_SHORT`], staged and plain by default, streaming only under
/// [`set_glu_store_evict_first`]; [`set_glu_wide_store`] off turns both off. Holds
/// [`XPF_LOCK`]: the switches are process-wide.
#[test]
fn glu_wide_store_on_the_swiglu_tiles() {
    let _lock = XPF_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(GLU_WST, WST_WIDE);
    let staged = [(16, GEMM_DEFAULT), (8, GEMM_DEFAULT), (8, GEMM_GLU_SHORT)];
    for (nt, tune) in staged {
        assert_eq!(grouped_glu_wst(nt, tune), GLU_WST, "nt {nt} {tune:?}");
    }
    // The tiles prefill picks: Qwen3.8-Flash-Next's gate / up (k 2560, 512 experts) at 33-64,
    // 65-128 and 129+ pairs an expert, and at up to 32 (on the GEMM under an mv max of 0).
    for per in [8, 32, 33, 64, 65, 128, 240] {
        let (nt, tune) = glu_gemm_tile(2560, 512, per);
        assert_eq!(grouped_glu_wst(nt, tune), GLU_WST, "{per} pairs an expert");
    }
    // Other tiles and tunes: none.
    let plain = [
        (4, GEMM_DEFAULT),
        (16, GEMM_GLU_SHORT),
        (16, GEMM_DOWN),
        (8, GEMM_DOWN),
        (
            16,
            GemmTune {
                stages: 3,
                ctas: 2,
                warps: 0,
            },
        ),
        (8, GEMM_STREAM_GLU),
    ];
    for (nt, tune) in plain {
        assert_eq!(grouped_glu_wst(nt, tune), 0, "nt {nt} {tune:?}");
    }
    // --nvfp4-glu-store-evict-first: streaming stores on top.
    set_glu_store_evict_first(true);
    for (nt, tune) in staged {
        assert_eq!(grouped_glu_wst(nt, tune), GLU_WST | WST_CS);
    }
    assert_eq!(grouped_glu_wst(4, GEMM_DEFAULT), 0);
    // --no-nvfp4-glu-wide-store: neither, whatever the other says.
    for cs in [false, true] {
        set_glu_store_evict_first(cs);
        set_glu_wide_store(false);
        for (nt, tune) in staged {
            assert_eq!(grouped_glu_wst(nt, tune), 0);
        }
        set_glu_wide_store(true);
    }
    set_glu_store_evict_first(false);
    // The down projection's epilogue does not move with the SwiGLU's switches.
    assert_eq!(grouped_gemm_wst(640, 512, 2, 8, false, true), DOWN_WST_BF16);
}

/// The SwiGLU epilogues the bit test compares with the fragments' own stores (0); [`WST_CS`]
/// alone is the plain kernel.
#[cfg(feature = "cuda")]
const GLU_WST_CHECKED: [i32; 3] = [WST_WIDE, WST_CS, WST_WIDE | WST_CS];

/// The SwiGLU clamps the bit test runs: none, and two limits (GLM-5.3-Flash's clamp) that clip
/// some of the scaled sums of the test's weights.
#[cfg(feature = "cuda")]
const GLU_LIMITS: [f32; 3] = [0.0, 0.05, 0.5];

/// One SwiGLU staged-store bit test case: `n_expert` NVFP4 gate and up experts of `m` rows by
/// `k`, `n_used` per token, the routing `ids`.
#[cfg(feature = "cuda")]
struct GluWstCase {
    name: &'static str,
    m: usize,
    k: usize,
    n_expert: usize,
    n_used: usize,
    ids: Vec<i32>,
}

/// `case`'s routed SwiGLU on the GEMM (every expert: an mv max of 0) at the 128- and 64-token
/// tiles of the default kernel and the 64-token tile of [`GEMM_GLU_SHORT`], without and with
/// the clamp ([`GLU_LIMITS`]): every epilogue of [`GLU_WST_CHECKED`] (and the defaults, with
/// and without the streaming stores) stores the bits of the fragments' own stores. The outputs
/// are filled with a pattern before each run, so a value an epilogue leaves out shows.
#[cfg(feature = "cuda")]
fn glu_wst_case(gpu: &mut crate::ggml::Backend, case: &GluWstCase) {
    use crate::ggml::ComputeAllocator;
    let (m, k, n_expert, n_used) = (case.m, case.k, case.n_expert, case.n_used);
    let n_tok = case.ids.len() / n_used;
    let blocks = k / 64 * 36 * m * n_expert;
    let weights = Context::new(8, 0);
    let wg = weights.new_tensor(types::NVFP4, &[k as i64, m as i64, n_expert as i64]);
    let wu = weights.new_tensor(types::NVFP4, &[k as i64, m as i64, n_expert as i64]);
    let scales = [0, 1].map(|_| weights.new_tensor(types::F32, &[n_expert as i64]));
    weights.alloc_tensors(gpu.buffer_type()).unwrap();
    wg.set_bytes(&gemm_bench::weights(blocks, 21));
    wu.set_bytes(&gemm_bench::weights(blocks, 22));
    for (i, s) in scales.iter().enumerate() {
        let v: Vec<f32> = (0..n_expert)
            .map(|e| 0.002 + ((e * 7 + i * 3) % 1000) as f32 * 1e-5)
            .collect();
        s.set_f32(&v);
    }
    let ctx = Context::new(16, 32);
    let x = ctx
        .new_tensor(types::BF16, &[k as i64, 1, n_tok as i64])
        .as_input();
    let id = ctx
        .new_tensor(types::I32, &[n_used as i64, n_tok as i64])
        .as_input();
    let outs = GLU_LIMITS.map(|limit| {
        swiglu_id_clamped(&ctx, wg, wu, x, id, Some(scales[0]), Some(scales[1]), limit).as_output()
    });
    let mut graph = ctx.new_graph(32);
    for o in outs {
        graph.expand(o);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
        .unwrap();
    x.set_bytes(&xpf_bf16(k * n_tok, 23));
    id.set_i32(&case.ids);
    let run = |gpu: &mut crate::ggml::Backend| -> Vec<Vec<u8>> {
        for o in outs {
            o.set_bytes(&vec![0xa5; o.nbytes()]);
        }
        gpu.compute(&graph).unwrap();
        outs.iter().map(|o| o.bytes()).collect()
    };
    set_expert_mv_max(0);
    for (nt, tune) in [(16, GEMM_DEFAULT), (8, GEMM_DEFAULT), (8, GEMM_GLU_SHORT)] {
        force_gemm_tune(Some(tune), Some(nt));
        force_gemm_wst(Some(0));
        let want = run(gpu);
        let clipped = want[0].iter().zip(&want[1]).filter(|(a, b)| a != b).count();
        eprintln!(
            "  {} nt {nt} {tune:?}: limit {} moves {clipped} of {} bytes",
            case.name,
            GLU_LIMITS[1],
            want[0].len()
        );
        // The forced epilogues, then the defaults with the streaming stores off and on.
        let tries = GLU_WST_CHECKED
            .map(|w| (Some(w), false))
            .into_iter()
            .chain([(None, false), (None, true)]);
        for (wst, cs) in tries {
            force_gemm_wst(wst);
            set_glu_store_evict_first(cs);
            for (i, (got, want)) in run(gpu).iter().zip(&want).enumerate() {
                assert_eq!(got.len(), want.len());
                let differ = got.iter().zip(want).filter(|(a, b)| a != b).count();
                assert_eq!(
                    differ, 0,
                    "{}, nt {nt} {tune:?}, limit {}, wst {wst:?}, evict-first {cs}: {differ} \
                     bytes differ",
                    case.name, GLU_LIMITS[i]
                );
            }
        }
    }
    set_glu_store_evict_first(false);
    xpf_reset();
}

/// The SwiGLU GEMM's staged 16-byte stores and streaming stores (#1183) keep every output bit:
/// 128- and 64-token tiles (the default kernel's and [`GEMM_GLU_SHORT`]'s), without and with
/// GLM-5.3-Flash's clamp, over hand-made routings (experts of 1, 15, 16, 17, 63, 64, 65, 127,
/// 128, 129 and 200 pairs: last token tiles partial and full), random ones over 512 and 64
/// experts, and SwiGLUs of 320 rows (a partial row tile) and 324 (rows not a multiple of 8: the
/// element-wise tail).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_glu_wide_store_keeps_bits() {
    let _lock = XPF_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut gpu = xpf_gpu();
    let special = [1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 200];
    let small = [1, 15, 16, 17, 63, 64, 65, 129];
    let cases = [
        // Qwen3.8-Flash-Next's gate / up (640 rows each, k 2560, 512 experts, top 10).
        GluWstCase {
            name: "flash-next counted",
            m: 640,
            k: 2560,
            n_expert: 512,
            n_used: 10,
            ids: counted_routing(&spread_counts(&special, 512, 10, 400), 10, 400),
        },
        GluWstCase {
            name: "flash-next random 512",
            m: 640,
            k: 2560,
            n_expert: 512,
            n_used: 10,
            ids: gemm_bench::routing(10, 1500, 512, 512, 7),
        },
        GluWstCase {
            name: "flash-next random 64",
            m: 640,
            k: 2560,
            n_expert: 512,
            n_used: 10,
            ids: gemm_bench::routing(10, 700, 512, 64, 8),
        },
        // Small shapes: 320 rows (a partial row tile) and 324 (not a multiple of 8).
        GluWstCase {
            name: "small 320 rows",
            m: 320,
            k: 256,
            n_expert: 24,
            n_used: 4,
            ids: counted_routing(&spread_counts(&small, 24, 4, 200), 4, 200),
        },
        GluWstCase {
            name: "small 324 rows",
            m: 324,
            k: 256,
            n_expert: 24,
            n_used: 4,
            ids: counted_routing(&spread_counts(&small, 24, 4, 200), 4, 200),
        },
    ];
    for case in &cases {
        glu_wst_case(&mut gpu, case);
        eprintln!("{}: the same bits", case.name);
    }
}

/// The routed SwiGLU's grouped GEMM (Qwen3.8-Flash-Next: 640 gate and 640 up rows, k 2560, 512
/// experts, top 10) at 12,288- and 8,067-token prompt steps with the fragments' own stores
/// (old), the staged stores (wide) and staged streaming ones (wide + cs), on its own and followed
/// by the down projection that reads its rows (BF16 rows, its epilogue at the default):
/// microseconds a layer (median and best of 5 rounds, the configurations taking turns, each
/// round starting one further on), the bytes moved a second at the median (the SwiGLU's alone),
/// and whether the bits are the old ones. `*` marks the default.
/// `cargo test --release --features cuda --lib -- --ignored cuda_glu_wide_store_bench
/// --nocapture`.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU; a benchmark"]
fn cuda_glu_wide_store_bench() {
    use crate::ggml::ComputeAllocator;
    let _lock = XPF_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    xpf_reset();
    let mut gpu = xpf_gpu();
    let (hidden, ffn, n_expert, n_used) = (2560usize, 640usize, 512usize, 10usize);
    const LAYERS: usize = 4;
    const ROUNDS: usize = 5;
    let blocks = |k: usize, m: usize| k / 64 * 36 * m * n_expert;
    let weights = Context::new(8, 0);
    let wd = weights.new_tensor(types::NVFP4, &[ffn as i64, hidden as i64, n_expert as i64]);
    let wg = weights.new_tensor(types::NVFP4, &[hidden as i64, ffn as i64, n_expert as i64]);
    let wu = weights.new_tensor(types::NVFP4, &[hidden as i64, ffn as i64, n_expert as i64]);
    let scales = [0, 1, 2].map(|_| weights.new_tensor(types::F32, &[n_expert as i64]));
    weights.alloc_tensors(gpu.buffer_type()).unwrap();
    wd.set_bytes(&gemm_bench::weights(blocks(ffn, hidden), 3));
    wg.set_bytes(&gemm_bench::weights(blocks(hidden, ffn), 4));
    wu.set_bytes(&gemm_bench::weights(blocks(hidden, ffn), 5));
    for s in scales {
        s.set_f32(&vec![0.004; n_expert]);
    }
    let variants: [(&str, i32); 3] = [
        ("old (fragments)", 0),
        ("wide", WST_WIDE),
        ("wide + cs", WST_WIDE | WST_CS),
    ];
    for n_tok in [12288usize, 8067] {
        let per = (n_used * n_tok) as i64 / n_expert as i64;
        let (nt, tune) = glu_gemm_tile(hidden as i64, n_expert as i64, per);
        let default = grouped_glu_wst(nt, tune);
        for chain in [false, true] {
            let ctx = Context::new(16 + 32 * LAYERS, 24 * LAYERS + 16);
            let x = ctx
                .new_tensor(types::BF16, &[hidden as i64, 1, n_tok as i64])
                .as_input();
            let ids: Vec<_> = (0..LAYERS)
                .map(|_| {
                    ctx.new_tensor(types::I32, &[n_used as i64, n_tok as i64])
                        .as_input()
                })
                .collect();
            let mut graph = ctx.new_graph(24 * LAYERS + 16);
            let outs: Vec<_> = ids
                .iter()
                .map(|&id| {
                    let glu = swiglu_id(&ctx, wg, wu, x, id, Some(scales[1]), Some(scales[2]));
                    let y = if chain {
                        mul_mat_id_act(&ctx, wd, glu, id, Some(scales[0]), Act::Bf16)
                    } else {
                        glu
                    }
                    .as_output();
                    graph.expand(y);
                    y
                })
                .collect();
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .unwrap();
            x.set_bytes(&xpf_bf16(hidden * n_tok, 5 + n_tok as u64));
            for (l, id) in ids.iter().enumerate() {
                id.set_i32(&gemm_bench::routing(
                    n_used,
                    n_tok,
                    n_expert,
                    n_expert,
                    1 + l as u64,
                ));
            }
            // A warm-up run of each (its bits against the old one's), then the rounds, the
            // variants taking turns. The forced epilogue is the SwiGLU's; the down tile's is
            // its default either way (a forced one would apply to it too).
            let force = |wst: i32| {
                force_gemm_wst(None);
                set_glu_wide_store(wst & WST_WIDE != 0);
                set_glu_store_evict_first(wst & WST_CS != 0);
            };
            let mut want = Vec::new();
            let mut same = vec![true; variants.len()];
            for (vi, &(_, wst)) in variants.iter().enumerate() {
                force(wst);
                gpu.compute(&graph).unwrap();
                let got: Vec<Vec<u8>> = outs.iter().map(|o| o.bytes()).collect();
                if vi == 0 {
                    want = got;
                } else {
                    same[vi] = got == want;
                }
            }
            let nv = variants.len();
            let mut times = vec![Vec::with_capacity(ROUNDS); nv];
            for round in 0..ROUNDS {
                for i in 0..nv {
                    let vi = (round + i) % nv;
                    force(variants[vi].1);
                    let t = std::time::Instant::now();
                    gpu.compute(&graph).unwrap();
                    times[vi].push(t.elapsed().as_secs_f64() / LAYERS as f64);
                }
            }
            set_glu_wide_store(true);
            set_glu_store_evict_first(false);
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            let median: Vec<f64> = times.iter().map(|t| t[t.len() / 2]).collect();
            let best: Vec<f64> = times.iter().map(|t| t[0]).collect();
            // The SwiGLU's bytes a layer: both weights, each token's input row, each pair's
            // BF16 output row.
            let pairs = (n_used * n_tok) as f64;
            let (w_bytes, x_bytes, out_bytes) = (
                2.0 * blocks(hidden, ffn) as f64,
                (n_tok * hidden * 2) as f64,
                pairs * ffn as f64 * 2.0,
            );
            let bytes = w_bytes + x_bytes + out_bytes;
            let op = if chain { "swiglu + down" } else { "swiglu" };
            eprintln!(
                "{op} at {n_tok} tokens, tile nt {nt} {tune:?} (swiglu {:.0} MB a layer: {:.0} \
                 weights, {:.0} in, {:.0} out)",
                bytes / 1e6,
                w_bytes / 1e6,
                x_bytes / 1e6,
                out_bytes / 1e6
            );
            for (vi, &(name, wst)) in variants.iter().enumerate() {
                eprintln!(
                    "  {}{name:<16} wst {wst}: median {:>8.1} us/layer ({:>5.3}x old), \
                     min {:>8.1} ({:>5.3}x)  {:>6.1} GB/s  {}",
                    if wst == default { '*' } else { ' ' },
                    median[vi] * 1e6,
                    median[0] / median[vi],
                    best[vi] * 1e6,
                    best[0] / best[vi],
                    bytes / median[vi] / 1e9,
                    if same[vi] { "same bits" } else { "BITS DIFFER" }
                );
            }
            assert!(same.iter().all(|&s| s), "{op} at {n_tok}: bits differ");
        }
    }
    xpf_reset();
}
