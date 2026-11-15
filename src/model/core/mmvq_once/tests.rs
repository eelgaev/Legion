//! The ops' shape checks, launch choices and CPU reference, and on a GPU of compute capability
//! 12.x their output bytes against ggml's `mul_mat_id` (quantize_q8_1 + MMVQ) for random
//! Q4_K / Q5_1 experts and ids (and the other types the ops take).

use super::bench::quantized_rows;
use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::nvfp4_once::bench::routing;

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
fn fits_shape_takes_flash_next_decode_and_verify_steps() {
    let ctx = Context::new(64, 0);
    let gate = ctx.new_tensor(types::Q4_K, &[2560, 640, 512]);
    let gate5 = ctx.new_tensor(types::Q5_K, &[2560, 640, 512]);
    let down = ctx.new_tensor(types::Q5_1, &[640, 2560, 512]);
    let down8 = ctx.new_tensor(types::Q8_0, &[640, 2560, 512]);
    let nvfp4 = ctx.new_tensor(types::NVFP4, &[2560, 640, 512]);
    let x = |k, n_x, n| ctx.new_tensor(types::F32, &[k, n_x, n]);
    let ids = |n| ctx.new_tensor(types::I32, &[10, n]);
    for n in 1..=8 {
        assert!(fits_shape(gate, x(2560, 1, n), ids(n)), "gate at {n}");
        assert!(fits_shape(gate5, x(2560, 1, n), ids(n)), "q5_K gate at {n}");
        assert!(fits_shape(down, x(640, 10, n), ids(n)), "down at {n}");
        assert!(fits_shape(down8, x(640, 10, n), ids(n)), "q8_0 down at {n}");
        assert!(glu_fits(gate, gate, x(2560, 1, n), ids(n)));
    }
    // ggml's MMQ steps, a type the ops do not take, a column count that is neither 1 nor
    // n_used, a width that does not match.
    assert!(!fits_shape(gate, x(2560, 1, 9), ids(9)));
    assert!(!fits_shape(nvfp4, x(2560, 1, 4), ids(4)));
    assert!(!fits_shape(down, x(640, 3, 4), ids(4)));
    assert!(!fits_shape(down, x(2560, 10, 4), ids(4)));
    // Gate and up of different types.
    assert!(!glu_fits(gate, gate5, x(2560, 1, 4), ids(4)));
    // On no GPU (the default until the engine sets one) nothing is used.
    assert!(!usable(gate, None, x(2560, 1, 4), ids(4)));
    assert!(!glu_usable(gate, gate, x(2560, 1, 4), ids(4)));
}

#[test]
fn fits_shape_needs_contiguous_rows_and_room_in_the_scratch() {
    let ctx = Context::new(64, 0);
    let ids = |n_used, n| ctx.new_tensor(types::I32, &[n_used, n]);
    // A view of every other row: rows not contiguous.
    let w = ctx.new_tensor(types::Q4_K, &[2560, 640, 8]);
    let nb = w.nb();
    let strided = ctx.view_3d(w, [2560, 320, 8], [2 * nb[1], nb[2]], 0);
    let x = ctx.new_tensor(types::F32, &[2560, 1, 4]);
    assert!(!fits_shape(strided, x, ids(10, 4)));
    // More pairs than the plan holds, more activation blocks than the scratch holds.
    let big = ctx.new_tensor(types::Q8_0, &[8192, 64, 64]);
    let x = ctx.new_tensor(types::F32, &[8192, 1, 8]);
    assert!(!fits_shape(big, x, ids(17, 8)));
    let x = ctx.new_tensor(types::F32, &[8192, 8, 8]);
    assert!(!fits_shape(big, x, ids(8, 8)));
}

#[test]
fn compute_capability_12_takes_the_ops() {
    assert!(gpu_takes(Some((12, 1))));
    assert!(gpu_takes(Some((12, 0))));
    for cc in [(7, 0), (8, 9), (9, 0), (10, 0), (11, 0)] {
        assert!(!gpu_takes(Some(cc)), "{cc:?}");
    }
    assert!(!gpu_takes(None));
}

#[test]
fn each_gpu_of_a_mixed_box_answers_for_itself() {
    use crate::hw::per_device::test_cc;
    for engine in [false, true] {
        let picks: Vec<bool> = (0..6)
            .map(|d| gpu_for_by(Some(d), engine, test_cc))
            .collect();
        assert_eq!(
            picks,
            [true, false, false, false, true, false],
            "engine {engine}"
        );
        assert_eq!(gpu_for_by(None, engine, test_cc), engine);
    }
}

#[test]
fn the_ops_are_off_unless_the_debug_switch_turns_them_on() {
    for v in [None, Some("0"), Some("off"), Some(""), Some("2")] {
        assert_eq!(mode_of(v), Mode::Off, "{v:?}");
    }
    for v in ["1", "on", "true", "yes", " 1 "] {
        assert_eq!(mode_of(Some(v)), Mode::On, "{v:?}");
    }
    assert_eq!(mode_of(Some("all")), Mode::All);
    assert_eq!(mode_of(Some("split")), Mode::Split);
}

#[test]
fn the_on_mode_takes_the_swiglu_op_up_to_3_tokens_and_no_single_matmuls() {
    for n in 1..=MAX_TOKENS {
        assert_eq!(mode_takes_glu(Mode::On, n), n <= GLU_TOKENS, "{n}");
        assert!(mode_takes_glu(Mode::All, n));
        assert!(!mode_takes_glu(Mode::Split, n));
        assert!(!mode_takes_glu(Mode::Off, n));
    }
    assert!(!mode_takes_mul_mat(Mode::On));
    assert!(!mode_takes_mul_mat(Mode::Off));
    assert!(mode_takes_mul_mat(Mode::All));
    assert!(mode_takes_mul_mat(Mode::Split));
    // The mode round-trips through its static (every value set_mode stores).
    for m in [Mode::Off, Mode::All, Mode::Split, Mode::On] {
        assert_eq!(
            match m as u8 {
                0 => Mode::On,
                2 => Mode::All,
                3 => Mode::Split,
                _ => Mode::Off,
            },
            m
        );
    }
}

#[test]
fn the_scratch_and_the_plan_hold_every_flash_next_step() {
    const { assert!(10 * MAX_TOKENS * 640 / 32 <= SCRATCH_BLOCKS) };
    const { assert!(MAX_TOKENS * 2560 / 32 <= SCRATCH_BLOCKS) };
    const { assert!(10 * MAX_TOKENS <= MAX_PAIRS) };
    // A SwiGLU's ring of one row of each matmul fits a CTA's shared memory on sm_120 (99 KB).
    const { assert!(STAGES * 2 * MAX_ROW_BYTES <= RING_BYTES) };
    const { assert!(RING_BYTES <= 99 * 1024) };
}

#[test]
fn stage_rows_fill_the_stage_in_whole_warps() {
    let t = |stage_bytes, warps| Tuning { stage_bytes, warps };
    // Flash-Next's Q4_K rows (1,440 B): 5 in 8 KB, 4 with 4 warps; 11 in 16 KB, 8 with 8 warps.
    assert_eq!(stage_rows(t(8192, 4), 1440, 640), 4);
    assert_eq!(stage_rows(t(16384, 8), 1440, 640), 8);
    // Its Q5_1 down rows (480 B): 17 in 8 KB, 16 with 4 warps.
    assert_eq!(stage_rows(t(8192, 4), 480, 2560), 16);
    // Fewer rows fitting than warps: as many as fit, at least one; never more than the rows.
    assert_eq!(stage_rows(t(4096, 8), 1440, 640), 2);
    assert_eq!(stage_rows(t(1024, 4), 1440, 640), 1);
    assert_eq!(stage_rows(t(65536, 4), 480, 6), 6);
}

#[test]
fn forced_tunings_round_trip_and_none_is_the_default() {
    for t in super::bench::BENCH_TUNINGS
        .into_iter()
        .chain([DEFAULT_TUNING, DEFAULT_TUNING_STEPS])
    {
        assert_eq!(unpacked(packed(Some(t))), Some(t));
    }
    assert_eq!(unpacked(packed(None)), None);
    // Warps outside 1-8 are clamped (the kernels' launch bounds are 256 threads).
    let many = Tuning {
        stage_bytes: 8192,
        warps: 32,
    };
    assert_eq!(unpacked(packed(Some(many))).map(|t| t.warps), Some(8));
}

#[test]
fn one_token_and_longer_steps_have_their_own_defaults() {
    assert_eq!(default_tuning(1), DEFAULT_TUNING);
    for n in 2..=MAX_TOKENS as usize {
        assert_eq!(default_tuning(n), DEFAULT_TUNING_STEPS, "{n}");
    }
}

#[test]
fn copy_units_divide_every_address_and_stride() {
    assert_eq!(copy_unit(&[0x1000, 1440, 1440 * 640]), 16);
    assert_eq!(copy_unit(&[0x1008, 480]), 8);
    assert_eq!(copy_unit(&[0x1000, 680]), 8);
    assert_eq!(copy_unit(&[0x1000, 2100]), 4);
    assert_eq!(copy_unit(&[0x1002, 1440]), 0);
}

#[test]
fn kernel_indices_cover_the_table_once() {
    let mut seen: Vec<usize> = (0..TYPES.len())
        .flat_map(|t| {
            [false, true]
                .into_iter()
                .flat_map(move |glu| [1, 2].map(|n| kernel_index(t, glu, n)))
        })
        .collect();
    // The 1-token kernel and the 2-8 token kernel are the same at every n above 1.
    assert_eq!(kernel_index(0, false, 2), kernel_index(0, false, 8));
    seen.sort();
    assert_eq!(seen, (1..1 + 4 * TYPES.len()).collect::<Vec<_>>());
}

/// The CPU op for one weight type: the dequantized product row by row (with the SwiGLU, silu of
/// the gate's times the up's), for both activation layouts.
fn check_cpu(ty: ffi::ggml_type, k: usize) {
    let (rows, n_expert, n_used, n) = (8usize, 6usize, 3usize, 4usize);
    let row_bytes = crate::ggml::row_size(ty, k);
    for (n_x, glu) in [(1usize, false), (n_used, false), (1, true)] {
        let wq = quantized_rows(ty, k, rows * n_expert, 3);
        let gq = quantized_rows(ty, k, rows * n_expert, 4);
        let xs = noise(k * n_x * n, 5);
        let ids = routing(n_used, n, n_expert, 4, 11);
        let mut be = Backend::cpu().unwrap();
        let wctx = Context::new(2, 0);
        let shape = [k as i64, rows as i64, n_expert as i64];
        let (w, g) = (wctx.new_tensor(ty, &shape), wctx.new_tensor(ty, &shape));
        wctx.alloc_tensors(be.buffer_type()).unwrap();
        w.set_bytes(&wq);
        g.set_bytes(&gq);
        let ctx = Context::new(8, 16);
        let x = ctx.new_tensor(types::F32, &[k as i64, n_x as i64, n as i64]);
        let it = ctx.new_tensor(types::I32, &[n_used as i64, n as i64]);
        ctx.alloc_tensors(be.buffer_type()).unwrap();
        x.set_f32(&xs);
        it.set_i32(&ids);
        let out = if glu {
            swiglu_id(&ctx, g, w, x, it)
        } else {
            mul_mat_id(&ctx, w, x, it)
        };
        assert_eq!(out.ne(), [rows as i64, n_used as i64, n as i64, 1]);
        let mut graph = ctx.new_graph(16);
        graph.expand(out);
        ctx.alloc_graph(&graph, ComputeAllocator::new(be.buffer_type()))
            .unwrap();
        be.compute(&graph).unwrap();
        let got = out.to_f32();
        for t in 0..n {
            for j in 0..n_used {
                let e = ids[t * n_used + j] as usize;
                let col = &xs[(t * n_x + if n_x == 1 { 0 } else { j }) * k..][..k];
                for r in 0..rows {
                    let at = (e * rows + r) * row_bytes;
                    let dot = |q: &[u8]| {
                        let wr = crate::ggml::dequantize(ty, &q[at..at + row_bytes], k);
                        wr.iter().zip(col).map(|(a, b)| a * b).sum::<f32>()
                    };
                    let mut want = dot(&wq);
                    if glu {
                        let gate = dot(&gq);
                        want *= gate / (1.0 + (-gate).exp());
                    }
                    let v = got[(t * n_used + j) * rows + r];
                    assert_eq!(
                        v.to_bits(),
                        want.to_bits(),
                        "n_x {n_x} glu {glu} t {t} j {j} r {r}"
                    );
                }
            }
        }
    }
}

#[test]
fn cpu_ops_are_the_dequantized_products() {
    check_cpu(types::Q4_K, 512);
    check_cpu(types::Q5_1, 128);
}

/// On a GPU of compute capability 12.x: the ops' output bytes equal ggml's `mul_mat_id`
/// (quantize_q8_1 + MMVQ, then ggml's `silu` and `mul`) for random experts and ids, at 1-8
/// tokens, tokens sharing experts and not, Flash-Next's Q4_K gate/up and Q5_1 down and the
/// other types its layers have (Q5_K, Q6_K gate/up; Q8_0 down), at the default tuning and at
/// others. Returns without checking where there is no such GPU (the PC, CI).
#[cfg(feature = "cuda")]
#[test]
fn gpu_ops_have_ggmls_bits() {
    use super::bench::{self, Config};
    if crate::ggml::cuda_device_count() == 0 || !gpu_takes(crate::hw::per_device::cc(0)) {
        eprintln!("skipped: no GPU of compute capability 12.x");
        return;
    }
    let mut gpu = crate::ggml::backend_by_name("CUDA0").unwrap();
    let small = Config {
        n_expert: 24,
        layers: 2,
        toks: vec![1, 2, 3, 5, 8],
        pools: vec![10, 24],
        sets: 2,
        rounds: 1,
        tunings: vec![
            Tuning {
                stage_bytes: 2048,
                warps: 4,
            },
            Tuning {
                stage_bytes: 16 * 1024,
                warps: 8,
            },
        ],
        ..Config::default()
    };
    let types = [
        (types::Q4_K, types::Q5_1),
        (types::Q5_K, types::Q8_0),
        (types::Q6_K, types::Q5_1),
    ];
    for (glu_ty, down_ty) in types {
        let cfg = Config {
            glu_ty,
            down_ty,
            ..small.clone()
        };
        let rows = bench::run(&mut gpu, &cfg).unwrap();
        for r in &rows {
            assert_eq!(
                r.differ,
                0,
                "{}/{} {} {} toks {} pool {}",
                crate::ggml::type_name(glu_ty),
                crate::ggml::type_name(down_ty),
                r.part.name(),
                r.variant,
                r.n_tok,
                r.pool
            );
        }
    }
}
