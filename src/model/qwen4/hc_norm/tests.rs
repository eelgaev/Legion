//! The mixer front against ggml's `mul(rms_norm(x), gamma)` and `mul_mat(w, xn)` on the CPU;
//! on CUDA (ignored test) against the CPU implementation.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};

/// Deterministic values in [-1, 1).
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

/// `x` rounded to F16 as ggml rounds F32 to F16, as bytes.
fn f16_bytes(x: &[f32]) -> Vec<u8> {
    // SAFETY: a pure conversion.
    x.iter()
        .flat_map(|&v| unsafe { ffi::ggml_fp32_to_fp16(v) }.to_ne_bytes())
        .collect()
}

/// Deterministic Q8_0 rows: `n` values, scales around 0.01, quants in [-127, 127].
fn q8_0_bytes(n: usize, seed: u64) -> Vec<u8> {
    let v = noise(n + n / 32, seed);
    let mut out = Vec::with_capacity(n / 32 * 34);
    for b in 0..n / 32 {
        // SAFETY: a pure conversion.
        let d = unsafe { ffi::ggml_fp32_to_fp16(0.01 + 0.005 * v[n + b]) };
        out.extend_from_slice(&d.to_ne_bytes());
        out.extend((0..32).map(|i| ((v[b * 32 + i] * 127.0) as i8) as u8));
    }
    out
}

/// The values of Q8_0 rows (`d · q`, exact in F32).
fn q8_0_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks(34)
        .flat_map(|b| {
            // SAFETY: a pure conversion.
            let d = unsafe { ffi::ggml_fp16_to_fp32(u16::from_ne_bytes([b[0], b[1]])) };
            b[2..].iter().map(move |&q| f32::from(q as i8) * d)
        })
        .collect()
}

/// Max |a - b| over max |b|.
fn rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
        / scale
}

/// `hc_norm` and ggml's `mul(rms_norm(x), gamma)` + `mul_mat(w, xn)` on `backend`: (op xn, op
/// inject, chain xn, chain inject).
fn hc_norm_case(
    backend: &mut Backend,
    e: i64,
    hc: i64,
    n_inj: i64,
    n: i64,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let ctx = Context::new(32, 32);
    let x = ctx.new_tensor(types::F32, &[e, hc, n]).as_input();
    let g = ctx.new_tensor(types::F32, &[e * hc]).as_input();
    let w = ctx.new_tensor(types::F32, &[e * hc, n_inj]).as_input();
    let (xn, inj) = hc_norm(&ctx, x, g, w, 1e-6);
    let (xn, inj) = (ctx.cont(xn).as_output(), ctx.cont(inj).as_output());
    let cxn = ctx.reshape_2d(
        ctx.mul(ctx.rms_norm(x, 1e-6), ctx.reshape_2d(g, e, hc)),
        e * hc,
        n,
    );
    let cinj = ctx.mul_mat(w, cxn).as_output();
    let cxn = cxn.as_output();
    let mut graph = ctx.new_graph(32);
    for t in [xn, inj, cxn, cinj] {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(&noise((e * hc * n) as usize, 11));
    g.set_f32(&noise((e * hc) as usize, 12));
    w.set_f32(&noise((e * hc * n_inj) as usize, 13));
    backend.compute(&graph).unwrap();
    (xn.to_f32(), inj.to_f32(), cxn.to_f32(), cinj.to_f32())
}

#[test]
fn hc_norm_matches_ggml_chain() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for (e, hc, n_inj, n) in [(256, 4, 4, 9), (64, 2, 3, 1)] {
        let (xn, inj, cxn, cinj) = hc_norm_case(&mut backend, e, hc, n_inj, n);
        let r = rel(&xn, &cxn);
        assert!(r < 1e-5, "xn {e}x{hc}: {r:e}");
        let r = rel(&inj, &cinj);
        assert!(r < 1e-5, "inject {e}x{hc}: {r:e}");
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_hc_norm_matches_cpu() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let mut cpu = Backend::cpu().unwrap();
    for (e, hc, n_inj, n) in [(2560, 4, 4, 300), (64, 2, 3, 1)] {
        let (xn, inj, _, _) = hc_norm_case(&mut gpu, e, hc, n_inj, n);
        let (cxn, cinj, _, _) = hc_norm_case(&mut cpu, e, hc, n_inj, n);
        let r = rel(&xn, &cxn);
        assert!(r < 1e-5, "xn {e}x{hc}: {r:e}");
        let r = rel(&inj, &cinj);
        assert!(r < 1e-5, "inject {e}x{hc}: {r:e}");
    }
}

/// `hc_post_norm` and the chain it replaces (ggml's `scale`, `sigmoid`, `scale`, `hc_post`, then
/// [`hc_norm`]) on `backend`: (op streams, op xn, op inject, chain streams, chain xn, chain
/// inject).
type Six = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
fn post_norm_case(backend: &mut Backend, e: i64, hc: i64, n_inj: i64, n: i64) -> Six {
    let ctx = Context::new(64, 64);
    let res = ctx.new_tensor(types::F32, &[e, hc, n]).as_input();
    let out = ctx.new_tensor(types::F32, &[e, n]).as_input();
    let inj = ctx.new_tensor(types::F32, &[hc, n]).as_input();
    let g = ctx.new_tensor(types::F32, &[e * hc]).as_input();
    let w = ctx.new_tensor(types::F32, &[e * hc, n_inj]).as_input();
    let p = hc_post_norm(&ctx, res, out, inj, g, w, 1e-6, None, false);
    let (r, xn, ij) = (p.res, p.xn, p.inject);
    let (r, xn, ij) = (
        ctx.cont(r).as_output(),
        ctx.cont(xn).as_output(),
        ctx.cont(ij).as_output(),
    );
    let post = ctx.scale(ctx.sigmoid(ctx.scale(inj, 1.0 / hc as f32)), 2.0);
    let cr = ctx.hc_post(out, res, post);
    let (cxn, cij) = hc_norm(&ctx, cr, g, w, 1e-6);
    let (cr, cxn, cij) = (
        ctx.cont(cr).as_output(),
        ctx.cont(cxn).as_output(),
        ctx.cont(cij).as_output(),
    );
    let mut graph = ctx.new_graph(64);
    for t in [r, xn, ij, cr, cxn, cij] {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    res.set_f32(&noise((e * hc * n) as usize, 21));
    out.set_f32(&noise((e * n) as usize, 22));
    inj.set_f32(&noise((hc * n) as usize, 23));
    g.set_f32(&noise((e * hc) as usize, 24));
    w.set_f32(&noise((e * hc * n_inj) as usize, 25));
    backend.compute(&graph).unwrap();
    (
        r.to_f32(),
        xn.to_f32(),
        ij.to_f32(),
        cr.to_f32(),
        cxn.to_f32(),
        cij.to_f32(),
    )
}

#[test]
fn hc_post_norm_matches_the_chain_on_the_cpu() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for (e, hc, n_inj, n) in [(256, 4, 4, 9), (64, 2, 3, 1)] {
        let (r, xn, ij, cr, cxn, cij) = post_norm_case(&mut backend, e, hc, n_inj, n);
        for (name, a, b) in [
            ("streams", &r, &cr),
            ("xn", &xn, &cxn),
            ("inject", &ij, &cij),
        ] {
            let d = rel(a, b);
            assert!(d < 1e-5, "{name} {e}x{hc}: {d:e}");
        }
    }
}

/// On CUDA the fused op only reorders the work: bit for bit the chain's results, except on
/// sm_7x (V100) by default, where nvcc rounds the sums of squares differently in the two
/// kernels: `hc_norm_kernel` adds every fourth square with FMUL + FADD (two roundings) and the
/// others with FFMA (one), `hc_post_norm` every square with FFMA; so a stream's scale can be an
/// ulp apart (measured: xn rel 1.95e-7, inject 1-4 ulps). Making them agree moves the default
/// greedy output, so the default takes that ulp on sm_7x and `--hc-norm-fma`
/// ([`set_norm_fma`]) rounds `hc_norm` as `hc_post_norm`: then exact on every GPU, with the
/// register kernels and with the generic one (#409, #376).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_hc_post_norm_is_bit_identical_to_the_chain() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let sm7x = crate::residency::cuda::caps(0).is_some_and(|c| c.cc_major == 7);
    for (fma, generic) in [(false, false), (true, false), (true, true)] {
        super::force_norm_fma(fma);
        super::force_generic_post_norm(generic);
        for (e, hc, n_inj, n) in [(2560, 4, 4, 300), (2560, 4, 4, 4096), (64, 2, 3, 1)] {
            let (r, xn, ij, cr, cxn, cij) = post_norm_case(&mut gpu, e, hc, n_inj, n);
            for (name, a, b) in [
                ("streams", &r, &cr),
                ("xn", &xn, &cxn),
                ("inject", &ij, &cij),
            ] {
                let same = a
                    .iter()
                    .zip(b.iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits());
                // sm_7x without the FMA form: the streams stay exact, xn and inject within
                // the inconsistent rounding.
                let bound = match name {
                    "xn" => 3e-7,
                    "inject" => 1e-6,
                    _ => 0.0,
                };
                assert!(
                    same || (sm7x && !fma && rel(a, b) <= bound),
                    "{name} {e}x{hc}x{n} (fma {fma}, generic {generic}): differs (rel {:e})",
                    rel(a, b)
                );
            }
        }
    }
    super::force_norm_fma(false);
    super::force_generic_post_norm(false);
}

/// On CUDA the FMA form of `hc_norm` (`--hc-norm-fma`) is still the norm: within rounding of
/// the CPU reference; and on GPUs other than sm_7x it is the default kernel, bit for bit.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_hc_norm_fma_matches_cpu() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let mut cpu = Backend::cpu().unwrap();
    let sm7x = crate::residency::cuda::caps(0).is_some_and(|c| c.cc_major == 7);
    for (e, hc, n_inj, n) in [(2560, 4, 4, 300), (64, 2, 3, 1)] {
        let (dxn, dinj, _, _) = hc_norm_case(&mut gpu, e, hc, n_inj, n);
        super::force_norm_fma(true);
        let (xn, inj, _, _) = hc_norm_case(&mut gpu, e, hc, n_inj, n);
        super::force_norm_fma(false);
        let (cxn, cinj, _, _) = hc_norm_case(&mut cpu, e, hc, n_inj, n);
        let r = rel(&xn, &cxn);
        assert!(r < 1e-5, "xn {e}x{hc}: {r:e}");
        let r = rel(&inj, &cinj);
        assert!(r < 1e-5, "inject {e}x{hc}: {r:e}");
        if !sm7x {
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&xn), bits(&dxn), "xn {e}x{hc}: the FMA form moved");
            assert_eq!(
                bits(&inj),
                bits(&dinj),
                "inject {e}x{hc}: the FMA form moved"
            );
        }
    }
}

/// On CUDA the register kernel (qwen4exp's 2560 x 4 streams; on sm_70 its own variant, #439)
/// and the default one (sm_80+: the sm_70 variant's layout with sm_80 rounding, #483) give the
/// generic kernel's bits; and the time of each at a 4K chunk.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_hc_post_norm_register_kernel_matches_the_generic_one() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let (e, hc, n_inj, n) = (2560i64, 4i64, 4i64, 4096i64);
    let ctx = Context::new(16, 16);
    let res = ctx.new_tensor(types::F32, &[e, hc, n]).as_input();
    let out = ctx.new_tensor(types::F32, &[e, n]).as_input();
    let inj = ctx.new_tensor(types::F32, &[hc, n]).as_input();
    let g = ctx.new_tensor(types::F32, &[e * hc]).as_input();
    let w = ctx.new_tensor(types::F32, &[e * hc, n_inj]).as_input();
    let down = ctx.new_tensor(types::Q8_0, &[e * hc, 320]).as_input();
    let p = hc_post_norm(&ctx, res, out, inj, g, w, 1e-6, Some(down), false);
    let (x16, d16) = (p.xn16.expect("xn16"), p.down16.expect("down16"));
    let (r, xn, ij) = (p.res.as_output(), p.xn.as_output(), p.inject.as_output());
    let (x16, d16) = (x16.as_output(), d16.as_output());
    let mut graph = ctx.new_graph(16);
    for t in [r, xn, ij, x16, d16] {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
        .unwrap();
    res.set_f32(&noise((e * hc * n) as usize, 31));
    out.set_f32(&noise((e * n) as usize, 32));
    inj.set_f32(&noise((hc * n) as usize, 33));
    g.set_f32(&noise((e * hc) as usize, 34));
    w.set_f32(&noise((e * hc * n_inj) as usize, 35));
    let q8 = q8_0_bytes((e * hc * 320) as usize, 36);
    down.set_bytes(&q8);
    let want16 = f16_bytes(&q8_0_values(&q8));
    let mut run = |mode: i32| {
        super::force_post_norm_kernel(mode);
        gpu.compute(&graph).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            gpu.compute(&graph).unwrap();
        }
        let ms = t0.elapsed().as_secs_f64() * 100.0;
        let xn = xn.to_f32();
        // Every kernel's F16 xn: its F32 xn rounded as ggml rounds F32 to F16.
        assert!(x16.bytes() == f16_bytes(&xn), "kernel {mode}: xn16");
        assert!(d16.bytes() == want16, "kernel {mode}: down16");
        (r.to_f32(), xn, ij.to_f32(), ms)
    };
    let (gr, gxn, gij, g_ms) = run(1);
    let (rr, rxn, rij, r_ms) = run(2);
    let (dr, dxn, dij, d_ms) = run(0);
    super::force_post_norm_kernel(0);
    println!(
        "hc_post_norm {e}x{hc}x{n}: generic {g_ms:.3} ms, registers {r_ms:.3} ms, \
         default {d_ms:.3} ms"
    );
    for (kernel, (sr, sxn, sij)) in [("register", (rr, rxn, rij)), ("default", (dr, dxn, dij))] {
        for (name, a, b) in [
            ("streams", &sr, &gr),
            ("xn", &sxn, &gxn),
            ("inject", &sij, &gij),
        ] {
            let same = a
                .iter()
                .zip(b.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits());
            assert!(
                same,
                "{name}: {kernel} kernel differs (rel {:e})",
                rel(a, b)
            );
        }
    }
}

/// `hc_post_norm` without and with the F16 xn on `backend`, and the mixer's down projection
/// both ways (Q8_0 `[e · hc, m]` weights): `mul_mat(down, xn)` and `mul_mat(down16, xn16)`.
struct Xn16Case {
    /// (streams, xn, inject) of the op without and with xn16.
    plain: [Vec<u32>; 3],
    with: [Vec<u32>; 3],
    xn: Vec<f32>,
    xn16: Vec<u8>,
    /// The op's F16 down weight, and the weight's values rounded to F16 as ggml rounds them.
    down16: Vec<u8>,
    want16: Vec<u8>,
    lo: Vec<f32>,
    lo16: Vec<f32>,
}

fn xn16_case(backend: &mut Backend, e: i64, hc: i64, n_inj: i64, n: i64, m: i64) -> Xn16Case {
    let ctx = Context::new(64, 64);
    let res = ctx.new_tensor(types::F32, &[e, hc, n]).as_input();
    let out = ctx.new_tensor(types::F32, &[e, n]).as_input();
    let inj = ctx.new_tensor(types::F32, &[hc, n]).as_input();
    let g = ctx.new_tensor(types::F32, &[e * hc]).as_input();
    let w = ctx.new_tensor(types::F32, &[e * hc, n_inj]).as_input();
    let down = ctx.new_tensor(types::Q8_0, &[e * hc, m]).as_input();
    let p = hc_post_norm(&ctx, res, out, inj, g, w, 1e-6, None, false);
    let q = hc_post_norm(&ctx, res, out, inj, g, w, 1e-6, Some(down), false);
    let lo = ctx.mul_mat(down, p.xn).as_output();
    let (d16, x16) = (q.down16.expect("down16"), q.xn16.expect("xn16"));
    let lo16 = ctx.mul_mat(d16, x16).as_output();
    let outs = [p.res, p.xn, p.inject, q.res, q.xn, q.inject].map(|t| ctx.cont(t).as_output());
    let (d16, x16) = (ctx.cont(d16).as_output(), ctx.cont(x16).as_output());
    let mut graph = ctx.new_graph(64);
    for t in outs.iter().chain([&lo, &lo16, &x16, &d16]) {
        graph.expand(*t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    res.set_f32(&noise((e * hc * n) as usize, 41));
    out.set_f32(&noise((e * n) as usize, 42));
    inj.set_f32(&noise((hc * n) as usize, 43));
    g.set_f32(&noise((e * hc) as usize, 44));
    w.set_f32(&noise((e * hc * n_inj) as usize, 45));
    let q8 = q8_0_bytes((e * hc * m) as usize, 46);
    down.set_bytes(&q8);
    backend.compute(&graph).unwrap();
    let bits = |t: Tensor<'_>| t.to_f32().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    Xn16Case {
        plain: [bits(outs[0]), bits(outs[1]), bits(outs[2])],
        with: [bits(outs[3]), bits(outs[4]), bits(outs[5])],
        xn: outs[1].to_f32(),
        xn16: x16.bytes(),
        down16: d16.bytes(),
        want16: f16_bytes(&q8_0_values(&q8)),
        lo: lo.to_f32(),
        lo16: lo16.to_f32(),
    }
}

/// The F16 xn changes nothing else the op writes, and is xn rounded as ggml rounds it.
#[test]
fn hc_post_norm_xn16_is_xn_rounded_on_the_cpu() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for (e, hc, n_inj, n) in [(256, 4, 4, 9), (64, 2, 3, 3)] {
        let c = xn16_case(&mut backend, e, hc, n_inj, n, 32);
        assert!(c.plain == c.with, "{e}x{hc}x{n}: outputs moved");
        assert!(c.xn16 == f16_bytes(&c.xn), "{e}x{hc}x{n}: xn16");
        assert!(c.down16 == c.want16, "{e}x{hc}x{n}: down16");
        // The CPU's Q8_0 matmul quantizes xn to Q8_0 instead: only close.
        let r = rel(&c.lo16, &c.lo);
        assert!(r < 5e-2, "{e}x{hc}x{n}: down projection {r:e}");
    }
}

/// On a V100 (7.0) ggml-cuda runs `mul_mat(down, xn)` of 64 or more columns as a cuBLAS F16
/// GEMM after rounding xn to F16, so the down projection from the op's F16 xn is bit for bit
/// the same (qwen4exp's 10240 x 320 mixer, the steps the model hands it: 64+ tokens); on every
/// GPU the op's other outputs do not move and xn16 is xn rounded as ggml rounds it.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_hc_post_norm_xn16_gives_ggmls_down_projection() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let v100 = crate::residency::cuda::caps(0).is_some_and(|c| (c.cc_major, c.cc_minor) == (7, 0));
    for n in [XN16_MIN_TOKENS, 300, 2058, 4096] {
        let c = xn16_case(&mut gpu, 2560, 4, 4, n, 320);
        assert!(c.plain == c.with, "{n} tokens: outputs moved");
        assert!(c.xn16 == f16_bytes(&c.xn), "{n} tokens: xn16");
        assert!(c.down16 == c.want16, "{n} tokens: down16");
        let same =
            c.lo.iter()
                .zip(&c.lo16)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        println!(
            "{n} tokens: down projection {} (rel {:e})",
            if same { "bit-identical" } else { "differs" },
            rel(&c.lo16, &c.lo)
        );
        assert!(
            same || !v100,
            "{n} tokens: the down projection moved on a V100"
        );
    }
}

/// `hc_post_norm` without and with the BF16 xn of a BF16 down weight on `backend` (#469), and
/// the mixer's down projection both ways (`[e · hc, m]` weights): `mul_mat(down, xn)` and
/// `mul_mat(down, xnb)`.
struct XnbCase {
    /// (streams, xn, inject) of the op without and with xnb.
    plain: [Vec<u32>; 3],
    with: [Vec<u32>; 3],
    xn: Vec<f32>,
    xnb: Vec<u8>,
    lo: Vec<u32>,
    lob: Vec<u32>,
}

fn xnb_case(backend: &mut Backend, e: i64, hc: i64, n_inj: i64, n: i64, m: i64) -> XnbCase {
    let ctx = Context::new(64, 64);
    let res = ctx.new_tensor(types::F32, &[e, hc, n]).as_input();
    let out = ctx.new_tensor(types::F32, &[e, n]).as_input();
    let inj = ctx.new_tensor(types::F32, &[hc, n]).as_input();
    let g = ctx.new_tensor(types::F32, &[e * hc]).as_input();
    let w = ctx.new_tensor(types::F32, &[e * hc, n_inj]).as_input();
    let down = ctx.new_tensor(types::BF16, &[e * hc, m]).as_input();
    let p = hc_post_norm(&ctx, res, out, inj, g, w, 1e-6, None, false);
    let q = hc_post_norm(&ctx, res, out, inj, g, w, 1e-6, Some(down), false);
    assert!(q.xn16.is_none() && q.down16.is_none() && q.xq.is_none());
    let xb = q.xnb.expect("xnb");
    assert_eq!((xb.ty(), xb.ne()), (types::BF16, [e * hc, n, 1, 1]));
    let lo = ctx.mul_mat(down, p.xn).as_output();
    let lob = ctx.mul_mat(down, xb).as_output();
    let outs = [p.res, p.xn, p.inject, q.res, q.xn, q.inject].map(|t| ctx.cont(t).as_output());
    let xb = ctx.cont(xb).as_output();
    let mut graph = ctx.new_graph(64);
    for t in outs.iter().chain([&lo, &lob, &xb]) {
        graph.expand(*t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    res.set_f32(&noise((e * hc * n) as usize, 51));
    out.set_f32(&noise((e * n) as usize, 52));
    inj.set_f32(&noise((hc * n) as usize, 53));
    g.set_f32(&noise((e * hc) as usize, 54));
    w.set_f32(&noise((e * hc * n_inj) as usize, 55));
    let d: Vec<u8> = noise((e * hc * m) as usize, 56)
        .iter()
        .flat_map(|&v| crate::model::nvfp4_mm::bf16_bits(0.05 * v).to_ne_bytes())
        .collect();
    down.set_bytes(&d);
    backend.compute(&graph).unwrap();
    let bits = |t: Tensor<'_>| t.to_f32().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    XnbCase {
        plain: [bits(outs[0]), bits(outs[1]), bits(outs[2])],
        with: [bits(outs[3]), bits(outs[4]), bits(outs[5])],
        xn: outs[1].to_f32(),
        xnb: xb.bytes(),
        lo: bits(lo),
        lob: bits(lob),
    }
}

/// xn rounded to BF16 (ties to even), as bytes.
fn bf16_bytes(x: &[f32]) -> Vec<u8> {
    x.iter()
        .flat_map(|&v| crate::model::nvfp4_mm::bf16_bits(v).to_ne_bytes())
        .collect()
}

/// The BF16 xn changes nothing else the op writes, is xn rounded as ggml rounds it, and the
/// down projection from it is ggml's from the F32 xn (ggml rounds xn to BF16 itself).
#[test]
fn hc_post_norm_xnb_is_xn_rounded_on_the_cpu() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for (e, hc, n_inj, n) in [(256, 4, 4, 9), (64, 2, 3, 3)] {
        let c = xnb_case(&mut backend, e, hc, n_inj, n, 32);
        assert!(c.plain == c.with, "{e}x{hc}x{n}: outputs moved");
        assert!(c.xnb == bf16_bytes(&c.xn), "{e}x{hc}x{n}: xnb");
        assert!(c.lo == c.lob, "{e}x{hc}x{n}: down projection moved");
    }
}

/// On sm_80+ ggml-cuda runs a BF16 weight's `mul_mat` of more than 16 columns as a cuBLAS BF16
/// GEMM after rounding xn to BF16, so the down projection from the op's BF16 xn is bit for bit
/// the same (qwen4exp's 10240 x 320 mixer, the steps the model hands it: 128+ tokens), and the
/// op's other outputs do not move (#469).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_hc_post_norm_xnb_gives_ggmls_down_projection() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    if !crate::model::fp8::bf16_mma() {
        println!("no BF16 tensor cores: skipped");
        return;
    }
    for generic in [false, true] {
        force_generic_post_norm(generic);
        for n in [XNB_MIN_TOKENS, 300, 2058, 4096] {
            let c = xnb_case(&mut gpu, 2560, 4, 4, n, 320);
            assert!(c.plain == c.with, "{n} tokens: outputs moved");
            assert!(c.xnb == bf16_bytes(&c.xn), "{n} tokens: xnb");
            assert!(c.lo == c.lob, "{n} tokens: the down projection moved");
        }
    }
    force_generic_post_norm(false);
}

/// `hc_post_norm` with and without `quant` on the same inputs, and ggml's quantization of the
/// former's `xn`, on `backend`: (streams, xn, inject, xq) of each op and ggml's blocks (the
/// blocks only, not the padding). `gamma` has an all-zero group, a group of tiny values (flushed
/// to zero) and one of huge values, besides noise.
type Quant = ([Vec<u8>; 4], [Vec<u8>; 3], Vec<u8>);
fn post_norm_q_case(backend: &mut Backend, e: i64, hc: i64, n_inj: i64, n: i64) -> Quant {
    use crate::model::core::mmq_preq::{self, tests::ggml_quantize};
    let ctx = Context::new(64, 64);
    let res = ctx.new_tensor(types::F32, &[e, hc, n]).as_input();
    let out = ctx.new_tensor(types::F32, &[e, n]).as_input();
    let inj = ctx.new_tensor(types::F32, &[hc, n]).as_input();
    let g = ctx.new_tensor(types::F32, &[e * hc]).as_input();
    let w = ctx.new_tensor(types::F32, &[e * hc, n_inj]).as_input();
    let q = hc_post_norm(&ctx, res, out, inj, g, w, 1e-6, None, true);
    let gq = ggml_quantize(&ctx, q.xn).as_output();
    let [r, xn, ij, xq] =
        [q.res, q.xn, q.inject, q.xq.expect("xq")].map(|t| ctx.cont(t).as_output());
    let p = hc_post_norm(&ctx, res, out, inj, g, w, 1e-6, None, false);
    let [r2, xn2, ij2] = [p.res, p.xn, p.inject].map(|t| ctx.cont(t).as_output());
    let mut graph = ctx.new_graph(64);
    for t in [r, xn, ij, xq, gq, r2, xn2, ij2] {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    res.set_f32(&noise((e * hc * n) as usize, 41));
    out.set_f32(&noise((e * n) as usize, 42));
    inj.set_f32(&noise((hc * n) as usize, 43));
    let mut gamma = noise((e * hc) as usize, 44);
    gamma[64..96].fill(0.0);
    gamma[160..192].iter_mut().for_each(|v| *v *= 1e-38);
    gamma[256..288].iter_mut().for_each(|v| *v *= 1e30);
    g.set_f32(&gamma);
    w.set_f32(&noise((e * hc * n_inj) as usize, 45));
    backend.compute(&graph).unwrap();
    let data = ((e * hc) / mmq_preq::BLOCK * n) as usize * mmq_preq::BLOCK_BYTES;
    let blocks = |t: Tensor<'_>| t.bytes()[..data].to_vec();
    (
        [r.bytes(), xn.bytes(), ij.bytes(), blocks(xq)],
        [r2.bytes(), xn2.bytes(), ij2.bytes()],
        blocks(gq),
    )
}

#[test]
fn hc_post_norm_q_on_the_cpu_adds_the_reference_quantization() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    let ([r, xn, ij, xq], [r2, xn2, ij2], gq) = post_norm_q_case(&mut backend, 256, 4, 4, 9);
    assert_eq!((r, xn, ij), (r2, xn2, ij2), "the F32 outputs moved");
    assert_eq!(xq, gq, "not quantize_ref's blocks");
}

/// The rebuilt node holds the scales (`hc` floats a token) where `xn` was, the rest moved up
/// by the difference; without them the layout is as before.
#[test]
fn rebuilt_layout_holds_scales_in_place_of_xn() {
    let (k, n, n_inj) = (10240usize, 4097usize, 4usize);
    let xn = PostNormLayout::new(k, n, n_inj, 0, false, true, None);
    let sc = PostNormLayout::new(k, n, n_inj, 0, false, true, Some((4, false)));
    assert_eq!((xn.xn, sc.xn), (k * n, k * n));
    assert_eq!(xn.inject, 2 * k * n);
    assert_eq!(sc.inject, k * n + 4 * n);
    assert!(sc.xq.is_multiple_of(256) && sc.xq >= (sc.inject + n_inj * n) * 4);
    assert_eq!(xn.floats - xn.xq / 4, sc.floats - sc.xq / 4, "xq's size");
    // The kernel's 32-bit offsets: up to 209,633 tokens of 10240 values in 4 streams.
    assert!(rebuild_fits(10240, 4, 8192));
    assert!(rebuild_fits(10240, 4, 209_628));
    assert!(!rebuild_fits(10240, 4, 209_800));
}

/// The padded rebuilt node with a BF16 `xn` (#836) is [`hc_post_norm`]'s node part for part:
/// the scales where `xn` was, the BF16 `xn` and the inject weights at the same offsets, the
/// same size, for every token count (an odd one put the packed form's BF16 `xn` 16 bytes
/// aligned only).
#[test]
fn padded_xnb_layout_is_the_written_layout() {
    for n in [1usize, 129, 300, 2048, 2051] {
        let (k, n_inj) = (10240usize, 4usize);
        let main = PostNormLayout::new(k, n, n_inj, 0, true, false, None);
        let padded = PostNormLayout::new(k, n, n_inj, 0, true, false, Some((k, false)));
        let packed = PostNormLayout::new(k, n, n_inj, 0, true, false, Some((4, false)));
        assert_eq!(
            (main.xn, main.xn16, main.inject, main.floats),
            (padded.xn, padded.xn16, padded.inject, padded.floats),
            "{n} tokens"
        );
        assert!((main.xn16 * 4).is_multiple_of(256));
        if n % 2 == 1 {
            assert!(!(packed.xn16 * 4).is_multiple_of(32), "{n} tokens");
        }
    }
}

/// On the CPU the rebuilt op writes the streams, inject weights and quantized `xn` of the op
/// with `xq`, and scales from which `(res · scale) · gamma` is its `xn`, bit for bit.
#[test]
fn hc_post_norm_rebuilt_on_the_cpu() {
    let (e, hc, n_inj, n) = (256i64, 4i64, 4i64, 9i64);
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    let ctx = Context::new(64, 64);
    let res = ctx.new_tensor(types::F32, &[e, hc, n]).as_input();
    let out = ctx.new_tensor(types::F32, &[e, n]).as_input();
    let inj = ctx.new_tensor(types::F32, &[hc, n]).as_input();
    let g = ctx.new_tensor(types::F32, &[e * hc]).as_input();
    let w = ctx.new_tensor(types::F32, &[e * hc, n_inj]).as_input();
    let q = hc_post_norm(&ctx, res, out, inj, g, w, 1e-6, None, true);
    let p = hc_post_norm_rebuilt(&ctx, res, out, inj, g, w, 1e-6, false);
    let [r, xn, ij] = [q.res, q.xn, q.inject].map(|t| ctx.cont(t).as_output());
    let [r2, sc, ij2] = [p.res, p.scales, p.inject].map(|t| ctx.cont(t).as_output());
    let (xq, xq2) = (q.xq.expect("xq").as_output(), p.xq.as_output());
    let mut graph = ctx.new_graph(64);
    for t in [r, xn, ij, r2, sc, ij2, xq, xq2] {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    res.set_f32(&noise((e * hc * n) as usize, 81));
    out.set_f32(&noise((e * n) as usize, 82));
    inj.set_f32(&noise((hc * n) as usize, 83));
    let gamma = noise((e * hc) as usize, 84);
    g.set_f32(&gamma);
    w.set_f32(&noise((e * hc * n_inj) as usize, 85));
    backend.compute(&graph).unwrap();
    assert!(
        r.bytes() == r2.bytes() && ij.bytes() == ij2.bytes(),
        "streams or inject moved"
    );
    let data = ((e * hc) / mmq_preq::BLOCK * n) as usize * mmq_preq::BLOCK_BYTES;
    assert!(xq.bytes()[..data] == xq2.bytes()[..data], "xq moved");
    let (streams, scales, xn) = (r2.to_f32(), sc.to_f32(), xn.to_f32());
    let k = (e * hc) as usize;
    for (i, x) in xn.iter().enumerate() {
        let (t, c) = (i / k, i % k / e as usize);
        let y = (streams[i] * scales[t * hc as usize + c]) * gamma[i % k];
        assert_eq!(y.to_bits(), x.to_bits(), "value {i}");
    }
}

/// `x` rounded to BF16 (ties to even) and widened back.
fn bf16_round(x: f32) -> f32 {
    f32::from_bits(u32::from(crate::model::nvfp4_mm::bf16_bits(x)) << 16)
}

/// The rebuilt op with BF16 streams (#826) on the CPU, from F32 streams (`res16` false) or
/// BF16 ones holding the same values, next to the F32 op: the new streams (widened to F32),
/// scales, inject weights and `xq` of both, as bytes.
fn rebuilt_bf16_case(
    backend: &mut Backend,
    e: i64,
    res16: bool,
    zero_out: bool,
) -> ([Vec<u8>; 4], [Vec<u8>; 4]) {
    let (hc, n_inj, n) = (4i64, 4i64, 9i64);
    let ctx = Context::new(64, 64);
    let res = ctx.new_tensor(types::F32, &[e, hc, n]).as_input();
    let res16_in = ctx.new_tensor(types::BF16, &[e, hc, n]).as_input();
    let out = ctx.new_tensor(types::F32, &[e, n]).as_input();
    let inj = ctx.new_tensor(types::F32, &[hc, n]).as_input();
    let g = ctx.new_tensor(types::F32, &[e * hc]).as_input();
    let w = ctx.new_tensor(types::F32, &[e * hc, n_inj]).as_input();
    let streams = if res16 { res16_in } else { res };
    let f = hc_post_norm_rebuilt(&ctx, res, out, inj, g, w, 1e-6, false);
    let b = hc_post_norm_rebuilt(&ctx, streams, out, inj, g, w, 1e-6, true);
    assert_eq!(b.res.ty(), types::BF16);
    let outs = [f, b].map(|p| {
        let r = ctx.cast(p.res, types::F32);
        [r, p.scales, p.inject, p.xq].map(|t| ctx.cont(t).as_output())
    });
    let mut graph = ctx.new_graph(64);
    for t in outs.iter().flatten() {
        graph.expand(*t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let r: Vec<f32> = noise((e * hc * n) as usize, 91)
        .into_iter()
        .map(bf16_round)
        .collect();
    res.set_f32(&r);
    if res16 {
        res16_in.set_bytes(&bf16_bytes(&r));
    }
    let o = noise((e * n) as usize, 92);
    out.set_f32(&if zero_out { vec![0.0; o.len()] } else { o });
    inj.set_f32(&noise((hc * n) as usize, 93));
    g.set_f32(&noise((e * hc) as usize, 94));
    w.set_f32(&noise((e * hc * n_inj) as usize, 95));
    backend.compute(&graph).unwrap();
    let data = ((e * hc) / mmq_preq::BLOCK * n) as usize * mmq_preq::BLOCK_BYTES;
    let [f, b] = outs.map(|o| {
        let mut v = o.map(|t| t.bytes());
        v[3].truncate(data);
        v
    });
    (f, b)
}

/// With BF16 streams the new streams are the F32 op's rounded to BF16, from F32 or BF16
/// streams alike; where the F32 op's are BF16 values already (nothing added), every output is
/// the F32 op's, bit for bit.
fn check_rebuilt_bf16(backend: &mut Backend, e: i64) {
    for res16 in [false, true] {
        let (f, b) = rebuilt_bf16_case(backend, e, res16, false);
        let rounded: Vec<u8> = floats(&f[0])
            .into_iter()
            .flat_map(|x| bf16_round(x).to_ne_bytes())
            .collect();
        assert!(
            b[0] == rounded,
            "res16 {res16}: streams not the F32 ones rounded"
        );
        assert!(b[0] != f[0], "res16 {res16}: nothing rounded");
        let (f, b) = rebuilt_bf16_case(backend, e, res16, true);
        assert!(f == b, "res16 {res16}: BF16 values moved the op");
    }
}

/// [`check_rebuilt_bf16`] on the CPU (the reference).
#[test]
fn hc_post_norm_rebuilt_bf16_streams_on_the_cpu() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    check_rebuilt_bf16(&mut cpu, 256);
}

/// [`check_rebuilt_bf16`] on CUDA (sm_80+) at qwen4exp's shape: `hc_post_norm_s16_kernel`
/// against the default kernel.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_hc_post_norm_rebuilt_bf16_streams() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    check_rebuilt_bf16(&mut gpu, 2560);
}

/// The rebuilt BF16-xn op (#836) with BF16 streams (`--hc-bf16`, #1183) next to the F32 one on
/// `n` tokens, from F32 streams (`res16` false) or BF16 ones holding the same values: the new
/// streams (widened to F32), scales, inject weights and BF16 xn of both, as F32 values.
fn rebuilt_xnb_bf16_case(
    backend: &mut Backend,
    e: i64,
    n: i64,
    res16: bool,
    zero_out: bool,
) -> [[Vec<f32>; 4]; 2] {
    let (hc, n_inj, m) = (4i64, 4i64, 32i64);
    let ctx = Context::new(64, 64);
    let res = ctx.new_tensor(types::F32, &[e, hc, n]).as_input();
    let res16_in = ctx.new_tensor(types::BF16, &[e, hc, n]).as_input();
    let out = ctx.new_tensor(types::F32, &[e, n]).as_input();
    let inj = ctx.new_tensor(types::F32, &[hc, n]).as_input();
    let g = ctx.new_tensor(types::F32, &[e * hc]).as_input();
    let w = ctx.new_tensor(types::F32, &[e * hc, n_inj]).as_input();
    let down = ctx.new_tensor(types::BF16, &[e * hc, m]).as_input();
    let streams = if res16 { res16_in } else { res };
    let f = hc_post_norm_rebuilt_xnb(&ctx, res, out, inj, g, w, 1e-6, down, false);
    let b = hc_post_norm_rebuilt_xnb(&ctx, streams, out, inj, g, w, 1e-6, down, true);
    assert_eq!(b.res.ty(), types::BF16);
    let outs = [f, b].map(|p| {
        [p.res, p.scales, p.inject, p.xnb].map(|t| ctx.cont(ctx.cast(t, types::F32)).as_output())
    });
    let mut graph = ctx.new_graph(64);
    for t in outs.iter().flatten() {
        graph.expand(*t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let r: Vec<f32> = noise((e * hc * n) as usize, 61)
        .into_iter()
        .map(bf16_round)
        .collect();
    res.set_f32(&r);
    if res16 {
        res16_in.set_bytes(&bf16_bytes(&r));
    }
    let o = noise((e * n) as usize, 62);
    out.set_f32(&if zero_out { vec![0.0; o.len()] } else { o });
    inj.set_f32(&noise((hc * n) as usize, 63));
    g.set_f32(&noise((e * hc) as usize, 64));
    w.set_f32(&noise((e * hc * n_inj) as usize, 65));
    down.set_bytes(&bf16_bytes(&noise((e * hc * m) as usize, 66)));
    backend.compute(&graph).unwrap();
    outs.map(|o| o.map(|t| t.to_f32()))
}

/// With BF16 streams the new streams are the F32 op's rounded to BF16 (the same FMA, then
/// RNE), from F32 or BF16 streams alike. `exact`: the other outputs from the same streams
/// (nothing added) are the F32 op's bit for bit (the CPU reference, one sum order); else
/// within F32 rounding of the sum order (the CUDA kernel's pair layout).
fn check_rebuilt_xnb_bf16(backend: &mut Backend, e: i64, ns: &[i64], exact: bool) {
    for &n in ns {
        for res16 in [false, true] {
            let at = format!("{n} tokens, res16 {res16}");
            let [f, b] = rebuilt_xnb_bf16_case(backend, e, n, res16, false);
            let rounded: Vec<f32> = f[0].iter().map(|&x| bf16_round(x)).collect();
            assert!(
                b[0].iter()
                    .zip(&rounded)
                    .all(|(x, y)| x.to_bits() == y.to_bits()),
                "{at}: streams not the F32 ones rounded"
            );
            assert!(b[0] != f[0], "{at}: nothing rounded");
            let [f, b] = rebuilt_xnb_bf16_case(backend, e, n, res16, true);
            for (i, name) in ["streams", "scales", "inject", "xnb"].iter().enumerate() {
                if exact || i == 0 {
                    assert!(
                        f[i].iter()
                            .zip(&b[i])
                            .all(|(x, y)| x.to_bits() == y.to_bits()),
                        "{at}: {name} moved"
                    );
                } else {
                    // xnb: a scale's last bit may flip a BF16 rounding (2^-8 relative).
                    let tol = if i == 3 { 1e-2 } else { 1e-5 };
                    let r = rel(&b[i], &f[i]);
                    assert!(r < tol, "{at}: {name} off by {r}");
                }
            }
        }
    }
}

/// [`check_rebuilt_xnb_bf16`] on the CPU (the reference).
#[test]
fn hc_post_norm_rebuilt_xnb_bf16_streams_on_the_cpu() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    check_rebuilt_xnb_bf16(&mut cpu, 256, &[9], true);
}

/// [`check_rebuilt_xnb_bf16`] on CUDA (sm_80+) at qwen4exp's shape: `hc_post_norm_b16_kernel`
/// against the default kernel.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_hc_post_norm_rebuilt_xnb_bf16_streams() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    if !crate::model::fp8::bf16_mma() {
        println!("no BF16 tensor cores: skipped");
        return;
    }
    check_rebuilt_xnb_bf16(&mut gpu, 2560, &[1, 9, 300], false);
}

/// F32 values from their bytes.
fn floats(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_ne_bytes(*c))
        .collect()
}

/// On CUDA (sm_80+) every `hc_post_norm` kernel writing the quantized `xn` as well (the default
/// one, the register one, the generic one) writes ggml-cuda's `quantize_mmq_q8_1` bytes of its
/// `xn`, and its F32 outputs are those of the kernel without the quantization, bit for bit.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_hc_post_norm_q_writes_ggml_quantization() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    assert!(
        crate::residency::cuda::caps(0).is_some_and(|c| c.cc_major >= 8),
        "sm_80+ only"
    );
    for mode in [0, 2, 1] {
        super::force_post_norm_kernel(mode);
        for n in [9, 300, 4096] {
            let ([r, xn, ij, xq], [r2, xn2, ij2], gq) = post_norm_q_case(&mut gpu, 2560, 4, 4, n);
            let diff = xq.iter().zip(&gq).filter(|(a, b)| a != b).count();
            assert_eq!(
                diff, 0,
                "kernel {mode}, {n} tokens: {diff} bytes differ from ggml's"
            );
            assert!(
                r == r2 && xn == xn2 && ij == ij2,
                "kernel {mode}, {n} tokens: F32 outputs moved"
            );
        }
    }
    super::force_post_norm_kernel(0);
}

/// On CUDA the matmul from the quantized `xn` is ggml-cuda's `mul_mat` of `xn`, bit for bit
/// (qwen4exp's down matmul: Q8_0 10240 -> 320), from 9 tokens (ggml's first MMQ size) up; and
/// the time of each path (the norm and the matmul).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_mmq_preq_is_ggml_mul_mat() {
    use crate::ggml::{DeviceKind, device_by_kind};
    use crate::model::core::mmq_preq::{self, tests::q8_0_matrix};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let (e, hc, n_inj, rows) = (2560i64, 4i64, 4i64, 320i64);
    let wb = q8_0_matrix((e * hc) as usize, rows as usize, 51);
    for n in [9, 48, 128, 256, 512, 1000, 2048, 4096] {
        // ours: the quantizing norm and the matmul from its blocks; ggml: the norm and mul_mat
        let mut run = |quant: bool| {
            let ctx = Context::new(32, 32);
            let res = ctx.new_tensor(types::F32, &[e, hc, n]).as_input();
            let out = ctx.new_tensor(types::F32, &[e, n]).as_input();
            let inj = ctx.new_tensor(types::F32, &[hc, n]).as_input();
            let g = ctx.new_tensor(types::F32, &[e * hc]).as_input();
            let w = ctx.new_tensor(types::F32, &[e * hc, n_inj]).as_input();
            let down = ctx.new_tensor(types::Q8_0, &[e * hc, rows]).as_input();
            let p = hc_post_norm(&ctx, res, out, inj, g, w, 1e-6, None, quant);
            let y = match p.xq {
                Some(xq) => mmq_preq::mul_mat(&ctx, down, xq, p.xn),
                None => ctx.mul_mat(down, p.xn),
            };
            let y = y.as_output();
            let mut graph = ctx.new_graph(32);
            graph.expand(y);
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .unwrap();
            res.set_f32(&noise((e * hc * n) as usize, 52));
            out.set_f32(&noise((e * n) as usize, 53));
            inj.set_f32(&noise((hc * n) as usize, 54));
            g.set_f32(&noise((e * hc) as usize, 55));
            w.set_f32(&noise((e * hc * n_inj) as usize, 56));
            down.set_bytes(&wb);
            gpu.compute(&graph).unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                gpu.compute(&graph).unwrap();
            }
            (y.bytes(), t0.elapsed().as_secs_f64() * 100.0)
        };
        let (ours, ours_ms) = run(true);
        let (ggml, ggml_ms) = run(false);
        println!(
            "down matmul of {n} tokens: norm + mul_mat {ggml_ms:.3} ms, \
             quantizing norm + mmq_preq {ours_ms:.3} ms"
        );
        assert!(ours == ggml, "{n} tokens: the matmul differs from ggml's");
    }
}

/// `--hc-fuse` (#774) on a V100: [`hc_post_norm`] taking over the sum of a fused MoE op with a
/// gated shared expert (the op then skips its sum kernel) gives the bits of the op's own sum,
/// ggml's sigmoid, mul and add of the shared expert, and the plain combine-and-norm, with and
/// without the F16 copies. Other GPUs: the kernel refuses (skipped).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_hc_post_norm_moe_sum_is_bit_identical() {
    use crate::ggml::{DeviceKind, device_by_kind};
    use crate::model::moe_ffn::{ExpertRefs, PairOut, Shexp, moe_ffn, moe_ffn_shexp};
    if !crate::residency::cuda::caps(0).is_some_and(|c| (c.cc_major, c.cc_minor) == (7, 0)) {
        eprintln!("not a V100: skipped");
        return;
    }
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .expect("no GPU device")
        .init()
        .unwrap();
    let (e, hc, n_inj, f, s, used) = (2560i64, 4i64, 4i64, 256i64, 8i64, 3i64);
    let wctx = Context::new(4, 0);
    let up = wctx.new_tensor(types::Q8_0, &[e, f, s]);
    let gate = wctx.new_tensor(types::Q8_0, &[e, f, s]);
    let down = wctx.new_tensor(types::Q8_0, &[f, e, s]);
    let mixer = wctx.new_tensor(types::Q8_0, &[e * hc, 64]);
    wctx.alloc_tensors(gpu.buffer_type()).unwrap();
    up.set_bytes(&q8_0_bytes((e * f * s) as usize, 61));
    gate.set_bytes(&q8_0_bytes((e * f * s) as usize, 62));
    down.set_bytes(&q8_0_bytes((e * f * s) as usize, 63));
    mixer.set_bytes(&q8_0_bytes((e * hc * 64) as usize, 64));
    for n in [70i64, 1000] {
        for f16 in [false, true] {
            // fused: the norm makes the sum; else the op's sum, ggml's shared expert chain and
            // the plain norm
            let run = |gpu: &mut Backend, fused: bool| {
                let ctx = Context::new(64, 64);
                let x = ctx.new_tensor(types::F32, &[e, n]).as_input();
                let ids = ctx.new_tensor(types::I32, &[used, n]).as_input();
                let wr = ctx.new_tensor(types::F32, &[used, n]).as_input();
                let sh = ctx.new_tensor(types::F32, &[e, n]).as_input();
                let sg = ctx.new_tensor(types::F32, &[1, n]).as_input();
                let res = ctx.new_tensor(types::F32, &[e, hc, n]).as_input();
                let inj = ctx.new_tensor(types::F32, &[hc, n]).as_input();
                let g = ctx.new_tensor(types::F32, &[e * hc]).as_input();
                let w = ctx.new_tensor(types::F32, &[e * hc, n_inj]).as_input();
                let refs = ExpertRefs {
                    up,
                    gate,
                    down,
                    table: None,
                };
                let out = if fused {
                    let shexp = Some(Shexp { sh, g: sg });
                    moe_ffn_shexp(&ctx, x, ids, wr, refs, PairOut::F32, shexp)
                } else {
                    let plain = moe_ffn(&ctx, x, ids, wr, refs, PairOut::F32);
                    ctx.add(plain, ctx.mul(sh, ctx.sigmoid(sg)))
                };
                let dn = f16.then_some(mixer);
                let kind = PostNormKind {
                    down: dn,
                    quant: false,
                    rebuild: false,
                    moe: fused,
                    streams16: false,
                    padded: false,
                };
                let (node, at) = post_norm_node(&ctx, res, out, inj, g, w, 1e-6, kind);
                let k = e * hc;
                let parts = [
                    ctx.view_1d(node, k * n, 0),
                    ctx.view_1d(node, k * n, at.xn * 4),
                    ctx.view_1d(node, n_inj * n, at.inject * 4),
                ];
                let mut outs: Vec<_> = parts.iter().map(|&p| ctx.cont(p).as_output()).collect();
                if f16 {
                    let half = ctx.view_1d(node, k * n / 2, at.xn16 * 4);
                    outs.push(ctx.cont(half).as_output());
                }
                let mut graph = ctx.new_graph(64);
                for &t in &outs {
                    graph.expand(t);
                }
                ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                    .unwrap();
                x.set_f32(&noise((e * n) as usize, 65));
                let picks: Vec<i32> = (0..n * used)
                    .map(|i| ((i / used * 5 + i % used * 3) % s) as i32)
                    .collect();
                ids.set_i32(&picks);
                let weights: Vec<f32> = noise((used * n) as usize, 66)
                    .iter()
                    .map(|v| v.abs() + 0.1)
                    .collect();
                wr.set_f32(&weights);
                sh.set_f32(&noise((e * n) as usize, 67));
                let logits: Vec<f32> = noise(n as usize, 68).iter().map(|v| v * 24.0).collect();
                sg.set_f32(&logits);
                res.set_f32(&noise((e * hc * n) as usize, 69));
                inj.set_f32(&noise((hc * n) as usize, 70));
                g.set_f32(&noise((e * hc) as usize, 71));
                w.set_f32(&noise((e * hc * n_inj) as usize, 72));
                gpu.compute(&graph).unwrap();
                outs.iter().map(|t| t.bytes()).collect::<Vec<_>>()
            };
            let ours = run(&mut gpu, true);
            let theirs = run(&mut gpu, false);
            for (i, name) in ["streams", "xn", "inject", "xn16"]
                .iter()
                .enumerate()
                .take(ours.len())
            {
                assert!(
                    ours[i] == theirs[i],
                    "{name}, {n} tokens, f16 {f16}: differs"
                );
            }
        }
    }
}

/// `--hc-fuse` on sm_80+ (#826): [`hc_post_norm`] taking over the sum of a
/// fused MoE op with a gated shared expert gives the bits of the op's own sum (its `reduce4`
/// path) and the combine-and-norm after it, plain, with `xq` and in the rebuilt form (the
/// streams' scales instead of `xn`, the GB10's prompt steps). Other GPUs: skipped.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_hc_post_norm_moe_sum_sm80_is_bit_identical() {
    use crate::ggml::{DeviceKind, device_by_kind};
    use crate::model::moe_ffn::{ExpertRefs, PairOut, Shexp, moe_ffn_shexp};
    if !crate::residency::cuda::caps(0).is_some_and(|c| c.cc_major >= 8) {
        eprintln!("not an sm_80+ GPU: skipped");
        return;
    }
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .expect("no GPU device")
        .init()
        .unwrap();
    let (e, hc, n_inj, f, s, used) = (2560i64, 4i64, 4i64, 256i64, 8i64, 3i64);
    let wctx = Context::new(3, 0);
    let up = wctx.new_tensor(types::Q8_0, &[e, f, s]);
    let gate = wctx.new_tensor(types::Q8_0, &[e, f, s]);
    let down = wctx.new_tensor(types::Q8_0, &[f, e, s]);
    wctx.alloc_tensors(gpu.buffer_type()).unwrap();
    up.set_bytes(&q8_0_bytes((e * f * s) as usize, 81));
    gate.set_bytes(&q8_0_bytes((e * f * s) as usize, 82));
    down.set_bytes(&q8_0_bytes((e * f * s) as usize, 83));
    // (quant, rebuild): plain, xq with xn, xq with the scales
    for n in [70i64, 1000] {
        for (quant, rebuild) in [(false, false), (true, false), (true, true)] {
            // fused: the norm makes the sum; else the op's own sum and the same norm
            let run = |gpu: &mut Backend, fused: bool| {
                let ctx = Context::new(64, 64);
                let x = ctx.new_tensor(types::F32, &[e, n]).as_input();
                let ids = ctx.new_tensor(types::I32, &[used, n]).as_input();
                let wr = ctx.new_tensor(types::F32, &[used, n]).as_input();
                let sh = ctx.new_tensor(types::F32, &[e, n]).as_input();
                let sg = ctx.new_tensor(types::F32, &[1, n]).as_input();
                let res = ctx.new_tensor(types::F32, &[e, hc, n]).as_input();
                let inj = ctx.new_tensor(types::F32, &[hc, n]).as_input();
                let g = ctx.new_tensor(types::F32, &[e * hc]).as_input();
                let w = ctx.new_tensor(types::F32, &[e * hc, n_inj]).as_input();
                let refs = ExpertRefs {
                    up,
                    gate,
                    down,
                    table: None,
                };
                let shexp = Some(Shexp { sh, g: sg });
                let out = moe_ffn_shexp(&ctx, x, ids, wr, refs, PairOut::F32, shexp);
                let kind = PostNormKind {
                    down: None,
                    quant,
                    rebuild,
                    moe: fused,
                    streams16: false,
                    // The scales `hc` floats a token (the parts below).
                    padded: false,
                };
                let (node, at) = post_norm_node(&ctx, res, out, inj, g, w, 1e-6, kind);
                let k = e * hc;
                let xn_floats = if rebuild { hc * n } else { k * n };
                let mut parts = vec![
                    ctx.view_1d(node, k * n, 0),
                    ctx.view_1d(node, xn_floats, at.xn * 4),
                    ctx.view_1d(node, n_inj * n, at.inject * 4),
                ];
                if quant {
                    parts.push(ctx.view_1d(node, (at.floats - at.xq / 4) as i64, at.xq));
                }
                let outs: Vec<_> = parts.iter().map(|&p| ctx.cont(p).as_output()).collect();
                let mut graph = ctx.new_graph(64);
                for &t in &outs {
                    graph.expand(t);
                }
                ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                    .unwrap();
                x.set_f32(&noise((e * n) as usize, 85));
                let picks: Vec<i32> = (0..n * used)
                    .map(|i| ((i / used * 5 + i % used * 3) % s) as i32)
                    .collect();
                ids.set_i32(&picks);
                let weights: Vec<f32> = noise((used * n) as usize, 86)
                    .iter()
                    .map(|v| v.abs() + 0.1)
                    .collect();
                wr.set_f32(&weights);
                sh.set_f32(&noise((e * n) as usize, 87));
                let logits: Vec<f32> = noise(n as usize, 88).iter().map(|v| v * 24.0).collect();
                sg.set_f32(&logits);
                res.set_f32(&noise((e * hc * n) as usize, 89));
                inj.set_f32(&noise((hc * n) as usize, 90));
                g.set_f32(&noise((e * hc) as usize, 91));
                w.set_f32(&noise((e * hc * n_inj) as usize, 92));
                gpu.compute(&graph).unwrap();
                outs.iter().map(|t| t.bytes()).collect::<Vec<_>>()
            };
            let ours = run(&mut gpu, true);
            let theirs = run(&mut gpu, false);
            for (i, name) in ["streams", "xn or scales", "inject", "xq"]
                .iter()
                .enumerate()
                .take(ours.len())
            {
                assert!(
                    ours[i] == theirs[i],
                    "{name}, {n} tokens, quant {quant} rebuild {rebuild}: differs"
                );
            }
        }
    }
}

/// A MoE layer's tail inputs and the rebuilt BF16-xn norm's after it, for [`fold_case`]: `e`
/// columns, `k` experts, `n` tokens, 4 streams and 4 inject outputs.
struct FoldIn {
    e: i64,
    k: i64,
    n: i64,
    experts: Vec<f32>,
    weights: Vec<f32>,
    sh: Vec<f32>,
    /// Three logits a token, the gate's the second (a row of a wider matmul, as in the models).
    joined: Vec<f32>,
    res: Vec<f32>,
    inj: Vec<f32>,
    gamma: Vec<f32>,
    wi: Vec<f32>,
    down: Vec<u8>,
    /// The experts' rows as BF16 cast to F32 (`--nvfp4-expert-rows-bf16`, #1183): `experts`
    /// then holds BF16 values.
    bf16: bool,
    /// BF16 streams in and out (`--hc-bf16`, #1183): `res` then holds BF16 values.
    streams16: bool,
}

const FOLD_HC: i64 = 4;
const FOLD_INJ: i64 = 4;
const FOLD_M: i64 = 16;

fn fold_in(e: i64, k: i64, n: i64) -> FoldIn {
    let kk = e * FOLD_HC;
    FoldIn {
        e,
        k,
        n,
        experts: noise((e * k * n) as usize, 101),
        weights: noise((k * n) as usize, 102)
            .iter()
            .map(|v| v.abs() + 0.05)
            .collect(),
        sh: noise((e * n) as usize, 103),
        joined: noise((3 * n) as usize, 104)
            .iter()
            .map(|v| v * 8.0)
            .collect(),
        res: noise((kk * n) as usize, 105),
        inj: noise((FOLD_HC * n) as usize, 106),
        gamma: noise(kk as usize, 107),
        wi: noise((kk * FOLD_INJ) as usize, 108),
        down: noise((kk * FOLD_M) as usize, 109)
            .iter()
            .flat_map(|&v| crate::model::nvfp4_mm::bf16_bits(0.05 * v).to_ne_bytes())
            .collect(),
        bf16: false,
        streams16: false,
    }
}

/// How [`fold_case`] gets the norm's block output.
enum Tail {
    /// The tail's nodes, the graph planned with the fold's relayout (`fold`) or without;
    /// `out_read`: the tail's output is a graph output too.
    Planned { fold: bool, out_read: bool },
    /// The tail's nodes, the norm folded by hand with this sum.
    Manual(FoldSum),
    /// The block output as an input.
    Given(Vec<f32>),
}

/// The fold's relayout ([`crate::model::core::node_fuse::fold::RELAYOUT`]) whatever
/// [`set_moe_fold`] says (other tests run meanwhile), as on CPU graphs.
fn fold_relayout() -> crate::ggml::post_alloc::Relayout {
    /// # Safety
    /// None needed: reads nothing.
    unsafe fn always(_: *mut ffi::ggml_cgraph, _: bool) -> bool {
        true
    }
    crate::ggml::post_alloc::Relayout {
        wants: always,
        ..crate::model::core::node_fuse::fold::RELAYOUT
    }
}

/// The norm's outputs (streams, scales, inject weights, BF16 xn) as bytes, and whether the
/// tail was folded (its gate's add gone).
fn fold_case(c: &FoldIn, tail: Tail) -> ([Vec<u8>; 4], bool) {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    fold_case_on(&mut backend, c, tail)
}

/// [`fold_case`] on `backend`.
fn fold_case_on(backend: &mut Backend, c: &FoldIn, tail: Tail) -> ([Vec<u8>; 4], bool) {
    let (e, k, n) = (c.e, c.k, c.n);
    let kk = e * FOLD_HC;
    let ctx = Context::new(256, 256);
    let rows16 = ctx.new_tensor(types::BF16, &[e, k, n]).as_input();
    let experts = if c.bf16 {
        ctx.cast(rows16, types::F32)
    } else {
        ctx.new_tensor(types::F32, &[e, k, n]).as_input()
    };
    let wr = ctx.new_tensor(types::F32, &[1, k, n]).as_input();
    let sh = ctx.new_tensor(types::F32, &[e, n]).as_input();
    let joined = ctx.new_tensor(types::F32, &[3, n]).as_input();
    let sg = ctx.view_2d(joined, [1, n], joined.nb()[1], 4);
    let given = matches!(tail, Tail::Given(_));
    let out = if given {
        ctx.new_tensor(types::F32, &[e, n]).as_input()
    } else {
        let weighted = ctx.mul(experts, wr);
        let nb = weighted.nb();
        let view = |j: i64| ctx.view_2d(weighted, [e, n], nb[2], j as usize * nb[1]);
        let mut sum = ctx.add(view(0), view(1));
        for j in 2..k {
            sum = ctx.add(sum, view(j));
        }
        let out = ctx.add(sum, ctx.mul(sh, ctx.sigmoid(sg)));
        match tail {
            Tail::Planned { out_read: true, .. } => out.as_output(),
            _ => out,
        }
    };
    let res_ty = if c.streams16 { types::BF16 } else { types::F32 };
    let res = ctx.new_tensor(res_ty, &[e, FOLD_HC, n]).as_input();
    let inj = ctx.new_tensor(types::F32, &[FOLD_HC, n]).as_input();
    let gamma = ctx.new_tensor(types::F32, &[kk]).as_input();
    let wi = ctx.new_tensor(types::F32, &[kk, FOLD_INJ]).as_input();
    let down = ctx.new_tensor(types::BF16, &[kk, FOLD_M]).as_input();
    let p = hc_post_norm_rebuilt_xnb(&ctx, res, out, inj, gamma, wi, 1e-6, down, c.streams16);
    let outs = [p.res, p.scales, p.inject, p.xnb].map(|t| ctx.cont(t).as_output());
    let mut graph = ctx.new_graph(256);
    for t in outs {
        graph.expand(t);
    }
    if matches!(tail, Tail::Planned { out_read: true, .. }) {
        graph.expand(out);
    }
    let allocator = ComputeAllocator::new(backend.buffer_type());
    let allocator = match tail {
        Tail::Planned { fold: true, .. } => allocator.with_relayout(fold_relayout()),
        _ => allocator,
    };
    ctx.alloc_graph(&graph, allocator).unwrap();
    if let Tail::Manual(sum) = &tail {
        // SAFETY: the norm's node of the allocated graph, its inputs graph inputs.
        unsafe {
            let node = (*p.res.raw()).view_src;
            assert!(fold_target(node, false), "the norm takes a fold");
            fold_moe(
                node,
                experts.raw(),
                wr.raw(),
                k as usize,
                *sum,
                sh.raw(),
                sg.raw(),
            );
        }
    }
    // SAFETY: a node of the graph.
    let folded = !given && unsafe { (*out.raw()).op } == ffi::ggml_op_GGML_OP_NONE;
    if let Tail::Given(v) = &tail {
        out.set_f32(v);
    } else {
        if c.bf16 {
            rows16.set_bytes(&bf16_bytes(&c.experts));
        } else {
            experts.set_f32(&c.experts);
        }
        wr.set_f32(&c.weights);
        sh.set_f32(&c.sh);
        joined.set_f32(&c.joined);
    }
    if c.streams16 {
        res.set_bytes(&bf16_bytes(&c.res));
    } else {
        res.set_f32(&c.res);
    }
    inj.set_f32(&c.inj);
    gamma.set_f32(&c.gamma);
    wi.set_f32(&c.wi);
    down.set_bytes(&c.down);
    backend.compute(&graph).unwrap();
    (outs.map(|t| t.bytes()), folded)
}

/// The tail's block output with each step rounded as `sum` says, the gate's sigmoid as the
/// CPU ops compute it.
fn tail_ref(c: &FoldIn, sum: FoldSum) -> Vec<f32> {
    let (e, k, n) = (c.e as usize, c.k as usize, c.n as usize);
    let mut out = vec![0f32; e * n];
    for t in 0..n {
        let gv = c.joined[t * 3 + 1];
        let s = 1.0 / (1.0 + (-gv).exp());
        for col in 0..e {
            let v = |j: usize| c.experts[(t * k + j) * e + col];
            let w = |j: usize| c.weights[t * k + j];
            let mut acc = v(0) * w(0);
            for j in 1..k {
                acc = match sum {
                    FoldSum::Plain => acc + v(j) * w(j),
                    FoldSum::Fma => v(j).mul_add(w(j), acc),
                };
            }
            out[t * e + col] = acc + c.sh[t * e + col] * s;
        }
    }
    out
}

/// The fold's relayout folds a CPU graph's MoE tail into the rebuilt BF16-xn norm after it
/// (#836) and plans the graph again, the tail's inputs kept alive for the norm: its outputs are
/// the unfolded graph's bit for bit (and those of the norm reading the tail's values as an
/// input).
#[test]
fn moe_tail_folds_into_the_norm_with_its_bits() {
    for (e, k, n) in [(64, 4, 3), (32, 10, 5), (64, 2, 1)] {
        let c = fold_in(e, k, n);
        let planned = |fold| Tail::Planned {
            fold,
            out_read: false,
        };
        let (want, plain) = fold_case(&c, planned(false));
        let (got, folded) = fold_case(&c, planned(true));
        assert!(!plain && folded, "{e}x{k}x{n}: {plain} {folded}");
        assert!(got == want, "{e}x{k}x{n}: the norm's outputs moved");
        let (given, _) = fold_case(&c, Tail::Given(tail_ref(&c, FoldSum::Plain)));
        assert!(got == given, "{e}x{k}x{n}: not the tail's values");
    }
}

/// BF16 expert rows read back as F32 (`--nvfp4-expert-rows-bf16`, #1183): the fold reads the
/// BF16 rows itself and the cast goes too, with the unfolded graph's bits.
#[test]
fn moe_tail_of_bf16_rows_folds_with_its_bits() {
    for (e, k, n) in [(64, 4, 3), (32, 10, 5)] {
        let mut c = fold_in(e, k, n);
        c.experts = c.experts.iter().map(|&v| bf16_round(v)).collect();
        c.bf16 = true;
        let planned = |fold| Tail::Planned {
            fold,
            out_read: false,
        };
        let (want, plain) = fold_case(&c, planned(false));
        let (got, folded) = fold_case(&c, planned(true));
        assert!(!plain && folded, "{e}x{k}x{n}: {plain} {folded}");
        assert!(got == want, "{e}x{k}x{n}: the norm's outputs moved");
        let (given, _) = fold_case(&c, Tail::Given(tail_ref(&c, FoldSum::Plain)));
        assert!(got == given, "{e}x{k}x{n}: not the tail's values");
    }
}

/// BF16 streams (`--hc-bf16`, #1183), with F32 or BF16 expert rows: the tail folds into the
/// BF16-stream norm with the unfolded graph's bits, and the new streams are BF16.
fn check_tail_folds_into_bf16_streams(backend: &mut Backend, shapes: &[(i64, i64, i64)]) {
    for &(e, k, n) in shapes {
        for rows16 in [false, true] {
            let mut c = fold_in(e, k, n);
            c.res = c.res.iter().map(|&v| bf16_round(v)).collect();
            c.streams16 = true;
            if rows16 {
                c.experts = c.experts.iter().map(|&v| bf16_round(v)).collect();
                c.bf16 = true;
            }
            let planned = |fold| Tail::Planned {
                fold,
                out_read: false,
            };
            let (want, plain) = fold_case_on(backend, &c, planned(false));
            let (got, folded) = fold_case_on(backend, &c, planned(true));
            let at = format!("{e}x{k}x{n} rows16 {rows16}");
            assert!(!plain && folded, "{at}: {plain} {folded}");
            assert_eq!(
                want[0].len(),
                (e * FOLD_HC * n * 2) as usize,
                "{at}: BF16 streams"
            );
            assert!(got == want, "{at}: the norm's outputs moved");
        }
    }
}

/// [`check_tail_folds_into_bf16_streams`] on the CPU (the reference).
#[test]
fn moe_tail_folds_into_bf16_streams_with_its_bits() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    check_tail_folds_into_bf16_streams(&mut cpu, &[(64, 4, 3), (32, 10, 5)]);
}

/// [`crate::model::core::node_fuse::fold::fold_graph`] turns the tail's nodes into views
/// without sources, and its undo puts the graph back as built.
#[test]
fn moe_fold_undoes() {
    use crate::ggml::post_alloc::Relaid;
    let (e, k, n) = (64, 4, 2);
    let ctx = Context::new(256, 256);
    let experts = ctx.new_tensor(types::F32, &[e, k, n]).as_input();
    let wr = ctx.new_tensor(types::F32, &[1, k, n]).as_input();
    let sh = ctx.new_tensor(types::F32, &[e, n]).as_input();
    let joined = ctx.new_tensor(types::F32, &[3, n]).as_input();
    let sg = ctx.view_2d(joined, [1, n], joined.nb()[1], 4);
    let weighted = ctx.mul(experts, wr);
    let nb = weighted.nb();
    let view = |j: i64| ctx.view_2d(weighted, [e, n], nb[2], j as usize * nb[1]);
    let mut sum = ctx.add(view(0), view(1));
    for j in 2..k {
        sum = ctx.add(sum, view(j));
    }
    let out = ctx.add(sum, ctx.mul(sh, ctx.sigmoid(sg)));
    let kk = e * FOLD_HC;
    let res = ctx.new_tensor(types::F32, &[e, FOLD_HC, n]).as_input();
    let inj = ctx.new_tensor(types::F32, &[FOLD_HC, n]).as_input();
    let gamma = ctx.new_tensor(types::F32, &[kk]).as_input();
    let wi = ctx.new_tensor(types::F32, &[kk, FOLD_INJ]).as_input();
    let down = ctx.new_tensor(types::BF16, &[kk, FOLD_M]).as_input();
    let p = hc_post_norm_rebuilt_xnb(&ctx, res, out, inj, gamma, wi, 1e-6, down, false);
    let mut graph = ctx.new_graph(256);
    for t in [p.res, p.scales, p.inject, p.xnb] {
        graph.expand(ctx.cont(t).as_output());
    }
    let backend = Backend::cpu().unwrap();
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    // SAFETY: an allocated graph that has not run; its nodes valid tensors.
    unsafe {
        let norm = (*p.res.raw()).view_src;
        let before = (*out.raw()).op;
        let src1 = (*norm).src[1];
        let (undo, folded, fma) =
            crate::model::core::node_fuse::fold::fold_graph(graph.raw(), false);
        assert_eq!((folded, fma), (1, 0));
        assert_eq!((*out.raw()).op, ffi::ggml_op_GGML_OP_NONE);
        assert!((*out.raw()).src.iter().all(|s| s.is_null()));
        assert_eq!((*norm).src[1], experts.raw());
        assert!(undo.holds(graph.raw()));
        undo.undo(graph.raw());
        assert_eq!((*out.raw()).op, before);
        assert_eq!((*norm).src[1], src1);
        assert!(fold_target(norm, false), "the norm as built again");
    }
}

/// Each fold sum rounds as its nodes do: ggml's `mul` and adds, and ggml-cuda's fused FMA sum.
#[test]
fn folded_sums_round_as_their_nodes() {
    let c = fold_in(64, 6, 4);
    for sum in [FoldSum::Plain, FoldSum::Fma] {
        let (got, _) = fold_case(&c, Tail::Manual(sum));
        let (want, _) = fold_case(&c, Tail::Given(tail_ref(&c, sum)));
        assert!(got == want, "{sum:?}");
    }
}

/// A tail whose output is read elsewhere too stays as it was.
#[test]
fn moe_tail_read_elsewhere_is_not_folded() {
    let c = fold_in(64, 4, 2);
    let (got, folded) = fold_case(
        &c,
        Tail::Planned {
            fold: true,
            out_read: true,
        },
    );
    assert!(!folded);
    let (want, _) = fold_case(
        &c,
        Tail::Planned {
            fold: false,
            out_read: true,
        },
    );
    assert!(got == want);
}

/// #1176: a stage graph captures an `hc_post_norm` node, also one that reads a deferred
/// reduce-scatter's partials: its kernel reads their addresses from the communicator's device
/// slot, which the exchange writes on the stream, so the launch takes no per-exchange host
/// state.
#[test]
fn a_deferred_reduce_scatter_reader_is_captured() {
    let (e, n) = (64, 4);
    let kk = e * FOLD_HC;
    let ctx = Context::new(64, 64);
    let out = ctx.new_tensor(types::F32, &[e, n]).as_input();
    let res = ctx.new_tensor(types::F32, &[e, FOLD_HC, n]).as_input();
    let inj = ctx.new_tensor(types::F32, &[FOLD_HC, n]).as_input();
    let gamma = ctx.new_tensor(types::F32, &[kk]).as_input();
    let wi = ctx.new_tensor(types::F32, &[kk, FOLD_INJ]).as_input();
    let down = ctx.new_tensor(types::BF16, &[kk, FOLD_M]).as_input();
    let p = hc_post_norm_rebuilt_xnb(&ctx, res, out, inj, gamma, wi, 1e-6, down, false);
    // SAFETY: the norm's node, just made; its params are the op's own.
    unsafe {
        let node = (*p.res.raw()).view_src;
        let safe = || {
            let op = crate::ggml::op_of(node).expect("an external op node");
            crate::model::core::stage_ops::capture_safe(op, &*node)
        };
        assert!(safe(), "a plain input sum");
        crate::ggml::set_external_param(node, P_RS, 1);
        assert!(safe(), "a deferred reduce-scatter");
    }
}

#[test]
fn each_gpu_of_a_mixed_box_answers_for_itself() {
    use crate::hw::per_device::test_cc;
    for engine in [false, true] {
        // The F16 `xn`: the V100 only; the BF16 one: the sm_80+ GPUs. Whatever the engine's GPU.
        let xn16: Vec<bool> = (0..6)
            .map(|d| gpu_for_by(Some(d), engine, xn16_takes, true, test_cc))
            .collect();
        assert_eq!(
            xn16,
            [false, false, true, false, false, false],
            "engine {engine}"
        );
        let xnb: Vec<bool> = (0..6)
            .map(|d| gpu_for_by(Some(d), engine, xnb_takes, true, test_cc))
            .collect();
        assert_eq!(
            xnb,
            [true, true, false, true, true, false],
            "engine {engine}"
        );
        // A changed cuBLAS compute type keeps ggml's conversion on every GPU.
        for d in 0..6 {
            assert!(!gpu_for_by(Some(d), engine, xn16_takes, false, test_cc));
            assert!(!gpu_for_by(Some(d), engine, xnb_takes, false, test_cc));
        }
        // A weight that names no device: the engine's answer.
        assert_eq!(gpu_for_by(None, engine, xn16_takes, true, test_cc), engine);
        assert_eq!(gpu_for_by(None, engine, xnb_takes, false, test_cc), engine);
    }
}
