use super::*;
use crate::ggml::{Backend, ComputeAllocator};

/// The op's CPU reference against the graph `hc_mix` builds (grouped RMS norm, down, SiLU, up,
/// gated mix, inject) on ggml's CPU backend, which rounds its activations to Q8_0: equal up to
/// that rounding.
#[test]
fn cpu_reference_matches_the_graph() {
    reference_against_graph(types::Q8_0, 0.02);
}

/// The same with BF16 down and up matrices (the NVFP4 checkpoint's mixers, `--hc-mix-bf16`):
/// ggml's CPU matmul rounds its activations to BF16 there, the reference keeps them F32.
#[test]
fn cpu_reference_matches_the_graph_bf16() {
    reference_against_graph(types::BF16, 0.02);
}

/// BF16 mixers take the op only when opted in; Q8_0 ones always (within the tile sizes); other
/// pairs never.
#[test]
fn bf16_weights_only_when_opted_in() {
    let ctx = Context::new(8, 0);
    let (k, rank) = (10240i64, 320i64);
    let pair = |ty| {
        (
            ctx.new_tensor(ty, &[k, rank]),
            ctx.new_tensor(ty, &[rank, k]),
        )
    };
    let (dq, uq) = pair(types::Q8_0);
    let (db, ub) = pair(types::BF16);
    let (df, uf) = pair(types::F16);
    let mixed = weights_fit(dq, ub, k, 4);
    let off = (weights_fit(dq, uq, k, 4), weights_fit(db, ub, k, 4));
    set_mix_bf16(true);
    let on = (weights_fit(dq, uq, k, 4), weights_fit(db, ub, k, 4));
    let f16 = weights_fit(df, uf, k, 4);
    set_mix_bf16(false);
    assert_eq!(
        (off, on, mixed, f16),
        ((true, false), (true, true), false, false)
    );
}

/// On CUDA the Q8_0 kernels with their loads ahead (the default, #814) write the same bytes as
/// the one-unit-at-a-time ones with a separate quantize kernel, for 1 to 8 tokens: at
/// Flash-Next's shape (each thread's units in one batch, the up rows in one pass), at one with
/// several batches a thread and lane and two passes of up rows (6 streams), and at one with
/// short batches and part of a warp's up rows idle (3 streams).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_loads_ahead_is_bit_for_bit() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for (e, hc, rank) in [(2560i64, 4i64, 320i64), (3072, 6, 1024), (1024, 3, 64)] {
        for n in 1..=MAX_TOKENS {
            set_mix_loads_ahead(true);
            let ahead = cuda_case(&mut gpu, e, hc, rank, n);
            set_mix_loads_ahead(false);
            let one = cuda_case(&mut gpu, e, hc, rank, n);
            set_mix_loads_ahead(true);
            for (i, what) in ["mixed", "inject"].iter().enumerate() {
                let diff = ahead[i].iter().zip(&one[i]).filter(|(a, b)| a != b).count();
                assert_eq!(
                    diff, 0,
                    "{e} x {hc}, rank {rank}, {n} tokens: {diff} {what} bytes moved"
                );
            }
        }
    }
}

/// The kernels' `l2_ahead` bits: `down` 1, `up` 2, `both` 3, `off` 0 (the default).
#[test]
fn l2_prefetch_codes() {
    let codes = [
        L2Prefetch::Off,
        L2Prefetch::Down,
        L2Prefetch::Up,
        L2Prefetch::Both,
    ]
    .map(L2Prefetch::code);
    assert_eq!(codes, [0, 1, 2, 3]);
    assert_eq!(L2Prefetch::default(), L2Prefetch::Off);
}

/// The kernels' variant bits (#814): a forced variant as it is; the defaults `V_DEFAULTS` 8 with
/// `V_NORM_REGS` 2, `V_DOWN_SPLIT` 4, `V_DOWN_HOIST` 16 and `V_UP_STAGE` 32 as switched on.
#[test]
fn variant_codes() {
    assert_eq!(variant_code(-1, true, true, true, true), 62);
    assert_eq!(variant_code(-1, true, true, false, false), 14);
    assert_eq!(variant_code(-1, true, false, false, false), 10);
    assert_eq!(variant_code(-1, false, true, false, false), 12);
    assert_eq!(variant_code(-1, false, false, false, false), 8);
    assert_eq!(variant_code(-1, false, false, true, false), 24);
    assert_eq!(variant_code(-1, false, false, false, true), 40);
    assert_eq!(variant_code(0, true, true, true, true), 0);
    assert_eq!(variant_code(6, false, false, false, false), 6);
}

/// On CUDA every L2 prefetch mode (#814) writes the same bytes as none, for 1 to 8 tokens and
/// the shapes of [`cuda_loads_ahead_is_bit_for_bit`].
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_l2_prefetch_is_bit_for_bit() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for (e, hc, rank) in [(2560i64, 4i64, 320i64), (3072, 6, 1024), (1024, 3, 64)] {
        for n in 1..=MAX_TOKENS {
            set_mix_l2_prefetch(L2Prefetch::Off);
            let off = cuda_case(&mut gpu, e, hc, rank, n);
            for mode in [L2Prefetch::Down, L2Prefetch::Up, L2Prefetch::Both] {
                set_mix_l2_prefetch(mode);
                let on = cuda_case(&mut gpu, e, hc, rank, n);
                assert_eq!(on, off, "{e} x {hc}, rank {rank}, {n} tokens, {mode:?}");
            }
            set_mix_l2_prefetch(L2Prefetch::Off);
        }
    }
}

/// [`mix`] on `backend` with Q8_0 down and up matrices and an inject: the mixed and inject
/// bytes.
#[cfg(feature = "cuda")]
fn cuda_case(backend: &mut Backend, e: i64, hc: i64, rank: i64, n: i64) -> [Vec<u8>; 2] {
    use crate::model::core::mmq_preq::tests::{noise, q8_0_matrix};
    let k = e * hc;
    let ctx = Context::new(16, 64);
    let x = ctx.new_tensor(types::F32, &[e, hc, n]).as_input();
    let norm = ctx.new_tensor(types::F32, &[k]).as_input();
    let down = ctx.new_tensor(types::Q8_0, &[k, rank]).as_input();
    let up = ctx.new_tensor(types::Q8_0, &[rank, k]).as_input();
    let inj = ctx.new_tensor(types::F32, &[k, hc]).as_input();
    let (mixed, inject) = mix(&ctx, x, norm, down, up, Some(inj), 1e-6);
    let outs = [ctx.cont(mixed), ctx.cont(inject.unwrap())].map(|t| t.as_output());
    let mut graph = ctx.new_graph(64);
    for t in outs {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(
        &noise((k * n) as usize, 91)
            .iter()
            .map(|v| v * 4.0)
            .collect::<Vec<_>>(),
    );
    norm.set_f32(
        &noise(k as usize, 92)
            .iter()
            .map(|v| 1.0 + 0.2 * v)
            .collect::<Vec<_>>(),
    );
    down.set_bytes(&q8_0_matrix(k as usize, rank as usize, 93));
    up.set_bytes(&q8_0_matrix(rank as usize, k as usize, 94));
    inj.set_f32(
        &noise((k * hc) as usize, 95)
            .iter()
            .map(|v| v * 0.01)
            .collect::<Vec<_>>(),
    );
    backend.compute(&graph).unwrap();
    outs.map(|t| t.bytes())
}

fn reference_against_graph(ty: ffi::ggml_type, tol: f32) {
    let (e, hc, rank, n) = (64i64, 4i64, 32i64, 3i64);
    let k = e * hc;
    let val = |i: usize, m: usize, d: f32| ((i * m % 97) as f32 - 48.0) / d;
    let down_f: Vec<f32> = (0..(k * rank) as usize)
        .map(|i| val(i, 31, 900.0))
        .collect();
    let up_f: Vec<f32> = (0..(rank * k) as usize)
        .map(|i| val(i, 17, 400.0))
        .collect();
    let norm_f: Vec<f32> = (0..k as usize).map(|i| 1.0 + val(i, 7, 500.0)).collect();
    let inj_f: Vec<f32> = (0..(k * hc) as usize).map(|i| val(i, 13, 700.0)).collect();
    let x_f: Vec<f32> = (0..(k * n) as usize).map(|i| val(i, 29, 30.0)).collect();
    let run = |ours: bool| -> (Vec<f32>, Vec<f32>) {
        let mut b = Backend::cpu().unwrap();
        b.set_threads(2);
        let weights = Context::new(4, 0);
        let down = weights.new_tensor(ty, &[k, rank]);
        let up = weights.new_tensor(ty, &[rank, k]);
        let norm = weights.new_tensor(types::F32, &[k]);
        let inj = weights.new_tensor(types::F32, &[k, hc]);
        weights.alloc_tensors(b.buffer_type()).unwrap();
        down.set_bytes(&crate::ggml::quantize(ty, &down_f, k as usize));
        up.set_bytes(&crate::ggml::quantize(ty, &up_f, rank as usize));
        norm.set_f32(&norm_f);
        inj.set_f32(&inj_f);
        let ctx = Context::new(16, 64);
        let x = ctx.new_tensor(types::F32, &[e, hc, n]);
        ctx.alloc_tensors(b.buffer_type()).unwrap();
        x.set_f32(&x_f);
        let (mixed, inject) = if ours {
            let (m, i) = mix(&ctx, x, norm, down, up, Some(inj), 1e-6);
            (m, i.unwrap())
        } else {
            let xn = ctx.mul(ctx.rms_norm(x, 1e-6), ctx.reshape_2d(norm, e, hc));
            let xn = ctx.reshape_2d(xn, k, n);
            let lo = ctx.silu(ctx.scale(ctx.mul_mat(down, xn), 1.0 / hc as f32));
            let gate = ctx.mul_mat(up, lo);
            let m = ctx.hc_pre_gated(
                ctx.reshape_3d(xn, e, hc, n),
                ctx.reshape_3d(gate, e, hc, n),
                1.0 / hc as f32,
            );
            (m, ctx.mul_mat(inj, xn))
        };
        let mixed = ctx.cont(mixed);
        let inject = ctx.cont(inject);
        let mut g = ctx.new_graph(64);
        g.expand(mixed);
        g.expand(inject);
        ctx.alloc_graph(&g, ComputeAllocator::new(b.buffer_type()))
            .unwrap();
        b.compute(&g).unwrap();
        (mixed.to_f32(), inject.to_f32())
    };
    let ((am, ai), (bm, bi)) = (run(true), run(false));
    let rel = |a: &[f32], b: &[f32]| {
        let s = b.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
            / s
    };
    assert_eq!((am.len(), ai.len()), ((e * n) as usize, (hc * n) as usize));
    assert!(rel(&am, &bm) < tol, "mixed: {}", rel(&am, &bm));
    assert!(rel(&ai, &bi) < 1e-4, "inject: {}", rel(&ai, &bi));
}
