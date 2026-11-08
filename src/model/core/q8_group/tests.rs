//! The grouped Q8_0 matvec op's shape rules, ggml's launch shape, and its CPU implementation.
//! The GPU bits against ggml's `mul_mat`: the server's greedy chat hashes and `--deterministic`
//! runs on a GPU machine.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::mmq_preq::tests::{noise, q8_0_matrix};

#[test]
fn ggml_launch_shape_for_q8_0_on_7x_8x() {
    // calc_nwarps / calc_rows_per_block (MMVQ_PARAMETERS_GENERIC and _TURING, Q8_0).
    for n in 1..=4 {
        assert_eq!(ggml_warps(n), 4);
    }
    for n in 5..=8 {
        assert_eq!(ggml_warps(n), 2);
    }
    assert_eq!(ggml_rows_per_block(1), 1);
    for n in 2..=8 {
        assert_eq!(ggml_rows_per_block(n), 2);
    }
}

#[test]
fn fits_q8_0_weights_of_1_to_8_columns() {
    let ctx = Context::new(64, 0);
    let qkv = ctx.new_tensor(types::Q8_0, &[2560, 10240]);
    let gate = ctx.new_tensor(types::Q8_0, &[2560, 6144]);
    let col = |n: i64| ctx.new_tensor(types::F32, &[2560, n]);
    for n in 1..=MAX_TOKENS {
        assert!(fits_all(&[qkv, gate], col(n)), "{n} columns");
    }
    assert!(!fits(qkv, col(9)));
    // One weight is ggml's own pair; five do not fit the op.
    assert!(!fits_all(&[qkv], col(3)));
    assert!(!fits_all(&[qkv, gate, qkv, gate, qkv], col(3)));
    // Odd rows (two rows per CTA from two columns), another type, a mismatched k.
    let odd = ctx.new_tensor(types::Q8_0, &[2560, 6143]);
    assert!(fits(odd, col(1)));
    assert!(!fits(odd, col(2)));
    assert!(!fits(ctx.new_tensor(types::Q4_0, &[2560, 6144]), col(3)));
    assert!(!fits(ctx.new_tensor(types::Q8_0, &[5120, 6144]), col(3)));
    // One column with a short row: ggml's small-K variant (k under 1024).
    let short = ctx.new_tensor(types::Q8_0, &[640, 2560]);
    assert!(!fits(short, ctx.new_tensor(types::F32, &[640, 1])));
    assert!(fits(short, ctx.new_tensor(types::F32, &[640, 2])));
    // A strided input (columns apart) fits; values apart do not.
    let wide = ctx.new_tensor(types::F32, &[5120, 3]);
    assert!(fits(qkv, ctx.view_2d(wide, [2560, 3], 5120 * 4, 0)));
    let t = ctx.transpose(ctx.new_tensor(types::F32, &[3, 2560]));
    assert!(!fits(qkv, t));
}

#[test]
fn gb10_takes_2_to_8_columns() {
    // ggml's GB10 mmvq widens only the one-column CTA (halve_iters): that stays on ggml there.
    assert!(!cols_on(true, 1));
    for n in 2..=MAX_TOKENS {
        assert!(cols_on(true, n), "{n} columns");
    }
    for n in 1..=MAX_TOKENS {
        assert!(cols_on(false, n), "{n} columns");
    }
    assert!(!cols_on(false, 0));
    assert!(!cols_on(false, MAX_TOKENS + 1));
    assert!(!cols_on(true, MAX_TOKENS + 1));
    // Without a GPU this is not a GB10 nor any 12.x, and every column count is the device's.
    assert!(!gb10());
    assert!(!sm12x());
    assert!(device_takes_cols(1));
    // A GB10 keeps the mixer's Q8_1 blocks and the experts reading them off.
    assert_eq!(GB10_NO_PARTS, PART_HC | PART_EXPERTS);
    assert_eq!(
        GB10_NO_PARTS & (PART_ATTN | PART_SHEXP | PART_SHEXP_DOWN),
        0
    );
}

#[test]
fn compute_capabilities_with_ggml_mmvq_launches() {
    // Volta to Ada (ggml's Turing and generic tables), the RTX 50 (generic, #1200) and the GB10.
    for cc in [70, 75, 80, 86, 87, 89, 120, 121] {
        assert!(cc_supported(cc), "{cc}");
    }
    // Hopper and others: not checked against ggml there.
    for cc in [0, 61, 90, 100, 103, 110, 122] {
        assert!(!cc_supported(cc), "{cc}");
    }
}

#[test]
fn gb10_only_with_the_flag() {
    assert!(mode_on_device(Mode::Default, false));
    assert!(!mode_on_device(Mode::Default, true));
    assert!(mode_on_device(Mode::All, true));
    assert!(mode_on_device(Mode::All, false));
    assert!(!mode_on_device(Mode::Off, false));
    assert!(!mode_on_device(Mode::Off, true));
}

#[test]
fn off_without_a_gpu() {
    // The engine sets the GPU on load; the tests never do (and run without CUDA here).
    let ctx = Context::new(4, 0);
    let a = ctx.new_tensor(types::Q8_0, &[2560, 64]);
    let b = ctx.new_tensor(types::Q8_0, &[2560, 32]);
    let x = ctx.new_tensor(types::F32, &[2560, 3]);
    assert!(fits_all(&[a, b], x));
    assert!(!takes(&[a, b], x));
}

#[test]
fn quantize_block_is_ggml_q8_1() {
    let x = noise(32, 3);
    let (d, q) = quantize_block(&x);
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let i = x.iter().position(|v| v.abs() == amax).unwrap();
    assert_eq!(q[i].unsigned_abs(), 127);
    // The scale as the F16 the dot products read.
    // SAFETY: plain conversions.
    let d16 = unsafe { ffi::ggml_fp16_to_fp32(ffi::ggml_fp32_to_fp16(amax / 127.0)) };
    assert_eq!(d.to_bits(), d16.to_bits());
    assert_eq!(quantize_block(&[0.0; 32]), (0.0, [0; 32]));
}

/// `mul_mat` on the CPU against the product of the dequantized operands in f64, for every
/// weight of the group (each result its own contiguous `[rows, n]`).
#[test]
fn cpu_mul_mat_is_the_dequantized_product() {
    let (k, n) = (1024usize, 3usize);
    let rows = [64usize, 32, 16];
    let wbs: Vec<Vec<u8>> = rows
        .iter()
        .enumerate()
        .map(|(i, &r)| q8_0_matrix(k, r, 7 + i as u64))
        .collect();
    let x = noise(k * n, 8);

    let ctx = Context::new(16, 16);
    let ws: Vec<_> = rows
        .iter()
        .map(|&r| {
            ctx.new_tensor(types::Q8_0, &[k as i64, r as i64])
                .as_input()
        })
        .collect();
    let xt = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
    let ys: Vec<_> = mul_mat(&ctx, &ws, xt)
        .into_iter()
        .map(|y| ctx.cont(y).as_output())
        .collect();
    let mut graph = ctx.new_graph(32);
    for &y in &ys {
        graph.expand(y);
    }
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    for (w, wb) in ws.iter().zip(&wbs) {
        w.set_bytes(wb);
    }
    xt.set_f32(&x);
    backend.compute(&graph).unwrap();

    for ((y, wb), &m) in ys.iter().zip(&wbs).zip(&rows) {
        let got = y.to_f32();
        assert_eq!(got.len(), m * n);
        let mut worst = 0f64;
        for r in 0..m {
            for j in 0..n {
                let (mut want, mut mag) = (0f64, 0f64);
                for b in 0..k / 32 {
                    let (dx, qx) = quantize_block(&x[j * k + b * 32..j * k + b * 32 + 32]);
                    let blk = &wb[(r * k / 32 + b) * Q8_0_BYTES..];
                    // SAFETY: a plain conversion.
                    let dw =
                        unsafe { ffi::ggml_fp16_to_fp32(u16::from_le_bytes([blk[0], blk[1]])) };
                    for v in 0..32 {
                        let t = dw as f64 * blk[2 + v] as i8 as f64 * dx as f64 * qx[v] as f64;
                        want += t;
                        mag += t.abs();
                    }
                }
                // F32 sums of k / 32 block terms: off by a few ulps of their magnitude.
                let g = got[j * m + r] as f64;
                worst = worst.max((g - want).abs() / mag);
            }
        }
        assert!(worst < 1e-5, "rel {worst:e}");
    }
}

/// `swiglu` on the CPU: `silu(gate @ x) * (up @ x)` of `mul_mat`'s two results, and the input's
/// Q8_1 node (`quantized`) as `quantize_block` gives each block, made once per input.
#[test]
fn cpu_swiglu_and_quantized_input() {
    let (k, n, m) = (1024usize, 2usize, 64usize);
    let (gb, ub) = (q8_0_matrix(k, m, 21), q8_0_matrix(k, m, 22));
    let x = noise(k * n, 23);

    let ctx = Context::new(32, 16);
    let gate = ctx
        .new_tensor(types::Q8_0, &[k as i64, m as i64])
        .as_input();
    let up = ctx
        .new_tensor(types::Q8_0, &[k as i64, m as i64])
        .as_input();
    let xt = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
    let parts: Vec<_> = mul_mat(&ctx, &[gate, up], xt)
        .into_iter()
        .map(|y| ctx.cont(y).as_output())
        .collect();
    let sh = ctx.cont(swiglu(&ctx, gate, up, xt)).as_output();
    assert!(has_quantized(&ctx, xt));
    let q = quantized(&ctx, xt).as_output();
    let mut graph = ctx.new_graph(32);
    for &y in parts.iter().chain([&sh, &q]) {
        graph.expand(y);
    }
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    gate.set_bytes(&gb);
    up.set_bytes(&ub);
    xt.set_f32(&x);
    backend.compute(&graph).unwrap();

    let (g, u, s) = (parts[0].to_f32(), parts[1].to_f32(), sh.to_f32());
    for i in 0..m * n {
        let want = u[i] * (g[i] / (1.0 + (-g[i]).exp()));
        assert_eq!(s[i].to_bits(), want.to_bits(), "{i}");
    }
    let qb = q.bytes();
    for t in 0..n {
        for b in 0..k / 32 {
            let (d, qs) = quantize_block(&x[t * k + b * 32..t * k + b * 32 + 32]);
            let o = &qb[(t * k / 32 + b) * Q8_1_BYTES..];
            // SAFETY: a plain conversion.
            let d16 = unsafe { ffi::ggml_fp16_to_fp32(u16::from_le_bytes([o[0], o[1]])) };
            assert_eq!(d16.to_bits(), d.to_bits());
            assert!(o[4..36].iter().zip(&qs).all(|(&a, &b)| a as i8 == b));
        }
    }
}

/// `mul_mat_each`: one node per weight, each the shape of ggml's `mul_mat` node and reading
/// the input's Q8_1 node, with the bits of `mul_mat`'s grouped node.
#[test]
fn cpu_mul_mat_each_is_one_node_per_weight_with_the_group_bits() {
    let (k, n) = (1024usize, 3usize);
    let rows = [64usize, 32];
    let wbs: Vec<Vec<u8>> = rows
        .iter()
        .enumerate()
        .map(|(i, &r)| q8_0_matrix(k, r, 31 + i as u64))
        .collect();
    let x = noise(k * n, 33);

    let ctx = Context::new(32, 16);
    let ws: Vec<_> = rows
        .iter()
        .map(|&r| {
            ctx.new_tensor(types::Q8_0, &[k as i64, r as i64])
                .as_input()
        })
        .collect();
    let xt = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
    assert!(!has_quantized(&ctx, xt));
    let q = quantized(&ctx, xt);
    let each = mul_mat_each(&ctx, &ws, xt);
    for (y, &m) in each.iter().zip(&rows) {
        // Not views: nodes of their own, ggml's matmul shape, the shared Q8_1 node read.
        assert_eq!(y.ne(), [m as i64, n as i64, 1, 1]);
        assert_eq!(y.srcs().len(), 3);
        assert_eq!(y.srcs()[1].raw(), q.raw());
    }
    assert_ne!(each[0].raw(), each[1].raw());
    let each: Vec<_> = each.into_iter().map(|y| y.as_output()).collect();
    let grouped: Vec<_> = mul_mat(&ctx, &ws, xt)
        .into_iter()
        .map(|y| ctx.cont(y).as_output())
        .collect();
    let mut graph = ctx.new_graph(32);
    for &y in each.iter().chain(&grouped) {
        graph.expand(y);
    }
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    for (w, wb) in ws.iter().zip(&wbs) {
        w.set_bytes(wb);
    }
    xt.set_f32(&x);
    backend.compute(&graph).unwrap();

    for (a, b) in each.iter().zip(&grouped) {
        let (a, b) = (a.to_f32(), b.to_f32());
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()));
    }
}

#[test]
fn takes_each_needs_the_q8_1_node_and_a_gpu() {
    let ctx = Context::new(8, 0);
    let qkv = ctx.new_tensor(types::Q8_0, &[2560, 10240]);
    let gate = ctx.new_tensor(types::Q8_0, &[2560, 6144]);
    let x = ctx.new_tensor(types::F32, &[2560, 3]);
    assert!(!takes_each(&ctx, &[qkv, gate], x));
    quantized(&ctx, x);
    // The Q8_1 node is there now, but the op is off without a GPU.
    assert!(has_quantized(&ctx, x));
    assert!(!takes_each(&ctx, &[qkv, gate], x));
}

#[test]
fn default_mode_runs_every_part_on_qwen4exp_only() {
    let parts = [
        PART_DELTA,
        PART_DELTA_AB,
        PART_ATTN,
        PART_SHEXP,
        PART_HC,
        PART_EXPERTS,
        PART_SHEXP_DOWN,
    ];
    let on = |mode, mask, arch| -> Vec<bool> {
        parts
            .iter()
            .map(|&b| part_in(mode, mask, arch, b))
            .collect()
    };
    // The default: every part on qwen4exp (the DeltaNet qkv and gate one node each), nothing
    // elsewhere.
    assert_eq!(on(Mode::Default, NO_MASK, "qwen4exp"), [true; 7]);
    assert_eq!(on(Mode::Default, NO_MASK, "gemma4"), [false; 7]);
    // --q8-group: all parts everywhere; --no-q8-group: none.
    assert_eq!(on(Mode::All, NO_MASK, "gemma4"), [true; 7]);
    assert_eq!(on(Mode::Off, NO_MASK, "qwen4exp"), [false; 7]);
    // The diagnostic mask replaces the parts, not the architectures.
    assert_eq!(
        on(Mode::Default, 1, "qwen4exp"),
        [true, false, false, false, false, false, false]
    );
    assert_eq!(on(Mode::Default, 1, "gemma4"), [false; 7]);
    assert_eq!(
        on(Mode::All, 4, "gemma4"),
        [false, false, true, false, false, false, false]
    );
    assert_eq!(
        on(Mode::Default, 31, "qwen4exp"),
        [true, true, true, true, true, false, false]
    );
    // Round 5's default: everything but the shared expert's down projection on the slot.
    assert_eq!(
        on(Mode::Default, 63, "qwen4exp"),
        [true, true, true, true, true, true, false]
    );
    assert_eq!(on(Mode::Off, 127, "qwen4exp"), [false; 7]);
}

#[test]
fn swiglu_down_is_off_without_a_gpu() {
    let ctx = Context::new(8, 0);
    let gate = ctx.new_tensor(types::Q8_0, &[2560, 640]);
    let up = ctx.new_tensor(types::Q8_0, &[2560, 640]);
    let down = ctx.new_tensor(types::Q8_0, &[640, 2560]);
    let x = ctx.new_tensor(types::F32, &[2560, 3]);
    assert!(fits_all(&[gate, up], x));
    assert!(!takes_swiglu_down(gate, up, down, x));
}

#[test]
fn slot_limits_cover_the_flash_next_shared_expert() {
    // n_ff 640 at every step size the op takes: 20 row blocks, 160 blocks at 8 columns.
    let m = 640;
    assert!(m / GROUP <= SLOT_ROW_BLOCKS);
    assert!(m / GROUP * MAX_TOKENS <= SLOT_BLOCKS);
    // A down projection of n_ff 640 at one column is ggml's small-K variant: not taken.
    let ctx = Context::new(4, 0);
    let down = ctx.new_tensor(types::Q8_0, &[640, 2560]);
    assert!(!weight_fits(down, 640, 1));
    assert!(weight_fits(down, 640, 2));
    assert!(!weight_fits(down, 512, 2));
}

/// `swiglu_down` on the CPU: two nodes, the SwiGLU's of `swiglu`'s size and the down
/// projection's of ggml's `mul_mat` size reading it, with the bits of `swiglu` and then
/// `mul_mat_each` of its output.
#[test]
fn cpu_swiglu_down_is_swiglu_then_the_down_matvec() {
    let (k, n, m, rows) = (1024usize, 3usize, 64usize, 96usize);
    let (gb, ub) = (q8_0_matrix(k, m, 41), q8_0_matrix(k, m, 42));
    let db = q8_0_matrix(m, rows, 43);
    let x = noise(k * n, 44);

    let ctx = Context::new(32, 16);
    let gate = ctx
        .new_tensor(types::Q8_0, &[k as i64, m as i64])
        .as_input();
    let up = ctx
        .new_tensor(types::Q8_0, &[k as i64, m as i64])
        .as_input();
    let down = ctx
        .new_tensor(types::Q8_0, &[m as i64, rows as i64])
        .as_input();
    let xt = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
    let ours = swiglu_down(&ctx, gate, up, down, xt, 70);
    assert_eq!(ours.ne(), [rows as i64, n as i64, 1, 1]);
    let sh = ours.srcs()[0];
    assert_eq!(sh.ne(), [m as i64, n as i64, 1, 1]);
    assert_eq!(ours.srcs()[1].raw(), sh.raw());
    assert_eq!(ours.srcs()[2].raw(), down.raw());
    let ours = ours.as_output();
    let sh_ref = ctx.cont(swiglu(&ctx, gate, up, xt));
    let theirs = mul_mat_each(&ctx, &[down], sh_ref)[0].as_output();
    let mut graph = ctx.new_graph(32);
    graph.expand(ours);
    graph.expand(theirs);
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    gate.set_bytes(&gb);
    up.set_bytes(&ub);
    down.set_bytes(&db);
    xt.set_f32(&x);
    backend.compute(&graph).unwrap();

    let (a, b) = (ours.to_f32(), theirs.to_f32());
    assert_eq!(a.len(), rows * n);
    assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()));
}
