//! The fused expert op against an f64 oracle of the same weights, and against the graph's
//! `mul_mat_id` path as a wiring check.

use super::*;
use crate::ggml::{Backend, BufferType, ComputeAllocator};

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

struct Case {
    gu: ffi::ggml_type,
    down: ffi::ggml_type,
    n_embd: usize,
    n_ff: usize,
    n_expert: usize,
    n_used: usize,
    n_tok: usize,
    /// Extra slots of 0xFF bytes (NaN when read as weights), half before the experts and half
    /// after them: the kernels must never read a slot the table does not name, nor past an
    /// expert's last row (ggml's MMQ reads up to 512 values past a row's end).
    spare: usize,
    /// How the CUDA kernels keep per-pair outputs.
    pair: PairOut,
    /// Whether the op may take GLM-5.3-Flash's IQ2 GEMMs ([`glm_kernels`]); such cases are
    /// Maya-S-sized and skip the f64 oracle (an empty one).
    glm: bool,
    /// ... and their int8 prompt GEMMs ([`glm_q8`]).
    q8: bool,
}

/// Random blocks of the low-bit formats (whose quantizers want an importance matrix), every
/// F16 scale set to 0.01: any codes are valid, and the oracle reads the dequantized values.
fn random_blocks(ty: ffi::ggml_type, n: usize, seed: u64) -> Option<Vec<u8>> {
    // (values per block, bytes per block, byte offsets of F16 scales)
    let (qk, bytes): (usize, usize) = match ty {
        types::IQ2_XXS => (256, 66),
        types::IQ2_S => (256, 82),
        types::IQ2_XS => (256, 74),
        types::IQ1_M => (256, 56),
        types::Q2_0 => (64, 18),
        // Project Maya's GLM-5.3-Flash formats (#559).
        types::IQ3_XXS => (256, 98),
        types::IQ3_S => (256, 110),
        types::IQ4_XS => (256, 136),
        types::Q2_K => (256, 84),
        types::Q3_K => (256, 110),
        _ => return None,
    };
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut out: Vec<u8> = (0..n / qk * bytes)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 32) as u8
        })
        .collect();
    let d = 0x211Fu16; // 0.01 in F16
    for b in out.chunks_mut(bytes) {
        if ty == types::IQ1_M {
            // The block scale lives in the top nibbles of the four u16 scale words.
            for (i, w) in b[48..56].chunks_mut(2).enumerate() {
                let nib = (d >> (4 * i)) & 0xF;
                let v = (u16::from_le_bytes([w[0], w[1]]) & 0x0FFF) | (nib << 12);
                w.copy_from_slice(&v.to_le_bytes());
            }
        } else if ty == types::Q2_K {
            // d and dmin after the scales and quants.
            b[80..82].copy_from_slice(&d.to_le_bytes());
            b[82..84].copy_from_slice(&d.to_le_bytes());
        } else if ty == types::Q3_K {
            b[108..110].copy_from_slice(&d.to_le_bytes());
        } else {
            b[..2].copy_from_slice(&d.to_le_bytes());
        }
    }
    Some(out)
}

/// Weights of `n_slots` slots, quantized, plus their dequantized values per slot.
fn weights(
    ty: ffi::ggml_type,
    k: usize,
    m: usize,
    n_slots: usize,
    seed: u64,
) -> (Vec<u8>, Vec<f32>) {
    let q = match random_blocks(ty, k * m * n_slots, seed) {
        Some(q) => q,
        None => {
            let vals: Vec<f32> = noise(k * m * n_slots, seed)
                .iter()
                .map(|v| v * 0.1)
                .collect();
            crate::ggml::quantize(ty, &vals, k)
        }
    };
    let deq = crate::ggml::dequantize(ty, &q, k * m * n_slots);
    (q, deq)
}

/// Runs the op (and the mul_mat_id path) on `backend` and returns (op, graph path, oracle).
fn run(
    c: &Case,
    backend: &mut Backend,
    weights_buft: BufferType,
) -> (Vec<f32>, Vec<f32>, Vec<f64>) {
    let n_slots = c.n_expert + c.spare;
    let lo = c.spare / 2;
    let poison = |(mut q, d): (Vec<u8>, Vec<f32>), slot_bytes: usize| {
        q[..lo * slot_bytes].fill(0xFF);
        q[(lo + c.n_expert) * slot_bytes..].fill(0xFF);
        (q, d)
    };
    let gu_slot = crate::ggml::row_size(c.gu, c.n_embd) * c.n_ff;
    let d_slot = crate::ggml::row_size(c.down, c.n_ff) * c.n_embd;
    let (uq, ud) = poison(weights(c.gu, c.n_embd, c.n_ff, n_slots, 1), gu_slot);
    let (gq, gd) = poison(weights(c.gu, c.n_embd, c.n_ff, n_slots, 2), gu_slot);
    let (dq, dd) = poison(weights(c.down, c.n_ff, c.n_embd, n_slots, 3), d_slot);
    // Expert e lives in slot table[e]: reversed, between the spare slots.
    let table: Vec<i32> = (0..c.n_expert)
        .map(|e| (lo + c.n_expert - 1 - e) as i32)
        .collect();

    let wctx = Context::new(8, 0);
    let (e, f, s) = (c.n_embd as i64, c.n_ff as i64, n_slots as i64);
    let up = wctx.new_tensor(c.gu, &[e, f, s]);
    let gate = wctx.new_tensor(c.gu, &[e, f, s]);
    let down = wctx.new_tensor(c.down, &[f, e, s]);
    wctx.alloc_tensors(weights_buft).unwrap();
    up.set_bytes(&uq);
    gate.set_bytes(&gq);
    down.set_bytes(&dq);

    let x: Vec<f32> = noise(c.n_embd * c.n_tok, 4);
    let mut ids = Vec::new();
    for t in 0..c.n_tok {
        // distinct experts per token, skewed toward the low ids
        let mut picked = Vec::new();
        let mut h = t as u64 * 7 + 3;
        while picked.len() < c.n_used {
            h = h
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (h >> 33) as usize % (c.n_expert * c.n_expert);
            let ex = ((r as f64).sqrt() as usize).min(c.n_expert - 1);
            if !picked.contains(&(ex as i32)) {
                picked.push(ex as i32);
            }
        }
        ids.extend(picked);
    }
    let w: Vec<f32> = noise(c.n_used * c.n_tok, 5)
        .iter()
        .map(|v| v.abs() + 0.1)
        .collect();

    let ctx = Context::new(64, 64);
    let (nt, nu) = (c.n_tok as i64, c.n_used as i64);
    let xt = ctx.new_tensor(types::F32, &[e, nt]).as_input();
    let idt = ctx.new_tensor(types::I32, &[nu, nt]).as_input();
    let wt = ctx.new_tensor(types::F32, &[nu, nt]).as_input();
    let tab = ctx.new_tensor(types::I32, &[c.n_expert as i64]).as_input();
    let refs = ExpertRefs {
        up,
        gate,
        down,
        table: Some(tab),
    };
    let op = glm_kernels(moe_ffn(&ctx, xt, idt, wt, refs, c.pair), c.glm);
    let op = glm_q8(op, c.q8).as_output();
    // The graph's path (qwen35's moe), with the router ids mapped through the table.
    let slots = ctx.reshape_2d(
        ctx.get_rows(
            ctx.reshape_2d(tab, 1, c.n_expert as i64),
            ctx.reshape_1d(idt, nu * nt),
        ),
        nu,
        nt,
    );
    let x3 = ctx.reshape_3d(xt, e, 1, nt);
    let graph_path = {
        let u = ctx.mul_mat_id(up, x3, slots);
        let g = ctx.mul_mat_id(gate, x3, slots);
        let act = ctx.mul(ctx.silu(g), u);
        let ex = ctx.mul(
            ctx.mul_mat_id(down, act, slots),
            ctx.reshape_3d(wt, 1, nu, nt),
        );
        let nb = ex.nb();
        let mut out = ctx.view_2d(ex, [e, nt], nb[2], 0);
        for j in 1..c.n_used {
            out = ctx.add(out, ctx.view_2d(ex, [e, nt], nb[2], j * nb[1]));
        }
        ctx.cont(out).as_output()
    };
    let mut graph = ctx.new_graph(64);
    graph.expand(op);
    graph.expand(graph_path);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    xt.set_f32(&x);
    idt.set_i32(&ids);
    wt.set_f32(&w);
    tab.set_i32(&table);
    backend.compute(&graph).unwrap();

    if c.glm {
        return (op.to_f32(), graph_path.to_f32(), Vec::new());
    }
    // f64 oracle on the dequantized weights
    let mut want = vec![0f64; c.n_embd * c.n_tok];
    let (ue, de) = (c.n_embd * c.n_ff, c.n_ff * c.n_embd);
    for t in 0..c.n_tok {
        let xr = &x[t * c.n_embd..(t + 1) * c.n_embd];
        for j in 0..c.n_used {
            let slot = table[ids[t * c.n_used + j] as usize] as usize;
            let wgt = w[t * c.n_used + j] as f64;
            let h: Vec<f64> = (0..c.n_ff)
                .map(|r| {
                    let row = |d: &[f32]| -> f64 {
                        let base = slot * ue + r * c.n_embd;
                        (0..c.n_embd)
                            .map(|i| d[base + i] as f64 * xr[i] as f64)
                            .sum()
                    };
                    let (g, u) = (row(&gd), row(&ud));
                    g / (1.0 + (-g).exp()) * u
                })
                .collect();
            for r in 0..c.n_embd {
                let base = slot * de + r * c.n_ff;
                let y: f64 = (0..c.n_ff).map(|i| dd[base + i] as f64 * h[i]).sum();
                want[t * c.n_embd + r] += wgt * y;
            }
        }
    }
    (op.to_f32(), graph_path.to_f32(), want)
}

/// Max |a - b| over max |b|.
fn rel(a: &[f32], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0f64, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - y).abs())
        .fold(0f64, f64::max)
        / scale
}

fn cases() -> Vec<Case> {
    let base = Case {
        gu: types::Q4_K,
        down: types::Q5_1,
        n_embd: 256,
        n_ff: 128,
        n_expert: 8,
        n_used: 3,
        n_tok: 70,
        spare: 4,
        pair: PairOut::F32,
        glm: false,
        q8: false,
    };
    vec![
        Case { ..base },
        Case {
            gu: types::Q5_K,
            down: types::Q8_0,
            n_tok: 5,
            ..base
        },
        Case {
            gu: types::Q8_0,
            down: types::Q5_1,
            n_embd: 128,
            spare: 0,
            ..base
        },
        // Q8_0 down rows of 6 blocks (204 bytes): every other row starts only 4-byte aligned,
        // which moves the GEMM's 64-bit Q8_0 word loads onto its 32-bit path (#808).
        Case {
            gu: types::Q8_0,
            down: types::Q8_0,
            n_ff: 192,
            n_tok: 40,
            ..base
        },
        // The low-bit formats of an IQ2_XS-class Qwen3.8-Flash-Next GGUF.
        Case {
            gu: types::IQ2_S,
            down: types::Q2_0,
            ..base
        },
        Case {
            gu: types::IQ2_XXS,
            down: types::Q2_0,
            n_tok: 5,
            ..base
        },
        Case {
            gu: types::IQ1_M,
            down: types::Q2_0,
            ..base
        },
        // Project Maya's GLM-5.3-Flash formats (#559): Maya-S's experts, its NextN block's,
        // Maya-M/L's; down rows of whole 256-value super-blocks.
        Case {
            gu: types::IQ2_XXS,
            down: types::IQ3_XXS,
            n_ff: 256,
            ..base
        },
        Case {
            gu: types::Q2_K,
            down: types::Q3_K,
            n_ff: 256,
            n_tok: 5,
            ..base
        },
        Case {
            gu: types::IQ3_S,
            down: types::IQ4_XS,
            n_ff: 256,
            ..base
        },
        // DeepSeek-V4.1-Flash GSQ-RCO's IQ2_XS among its IQ2_XXS / IQ1_M experts (#584).
        Case {
            gu: types::IQ2_XS,
            down: types::IQ1_M,
            n_ff: 256,
            ..base
        },
        Case {
            gu: types::IQ2_XXS,
            down: types::IQ2_XS,
            n_ff: 256,
            n_tok: 5,
            ..base
        },
    ]
}

#[test]
fn cpu_matches_oracle_and_graph_path() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for c in cases() {
        let buft = backend.buffer_type();
        let (op, graph, want) = run(&c, &mut backend, buft);
        let label = format!("types {}/{}, {} tokens", c.gu, c.down, c.n_tok);
        // The reference: F32 weights and activations, F32 sums.
        let r = rel(&op, &want);
        assert!(r < 1e-5, "{label}: op vs oracle {r:e}");
        // ggml's CPU path quantizes the activations (Q8_K / Q8_0): a wiring check only.
        // Random Q2_K blocks' mins cancel much of each sum: a looser bound there.
        let g = rel(&graph, &want);
        let tol = if c.gu == types::Q2_K { 5e-2 } else { 2e-2 };
        assert!(g < tol, "{label}: graph path vs oracle {g:e}");
    }
}

/// The CUDA kernels: FP16 activations and SwiGLU rows, F32 sums; weights in VRAM through a
/// permuted table.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_matches_oracle() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut backend = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let mut all = cases();
    all.push(Case {
        n_tok: 300,
        ..cases().remove(0)
    });
    all.push(wide_case());
    for c in all {
        let buft = backend.buffer_type();
        let (op, graph, want) = run(&c, &mut backend, buft);
        let label = format!("types {}/{}, {} tokens", c.gu, c.down, c.n_tok);
        let r = rel(&op, &want);
        let g = rel(&graph, &want);
        eprintln!("{label}: op vs oracle {r:e}, graph path vs oracle {g:e}");
        assert!(r < 2e-3, "{label}: op vs oracle {r:e}");
    }
}

/// Q4_K gate/up two super-blocks wide and 512 interleaved rows, Q5_1 down two 128-value stages
/// long (the GB10's fragment kernels take both: whole stages along K, whole row blocks), full and
/// partial tiles.
#[cfg(feature = "cuda")]
fn wide_case() -> Case {
    Case {
        n_embd: 512,
        n_ff: 256,
        n_tok: 300,
        ..cases().remove(0)
    }
}

/// The op's output bits on the current GPU, as one hash.
#[cfg(feature = "cuda")]
fn cuda_output_hash(c: &Case) -> u64 {
    use crate::ggml::{DeviceKind, device_by_kind};
    use std::hash::{Hash, Hasher};
    let mut backend = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let buft = backend.buffer_type();
    let (op, _, _) = run(c, &mut backend, buft);
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for v in &op {
        v.to_bits().hash(&mut h);
    }
    h.finish()
}

/// The GEMM kernels the dispatch picks for the same weights give the same bits: the default
/// (sm_80+: the fragment kernels on the GB10 for Q4_K / Q5_K gate/up and Q5_1 / Q8_0 down,
/// gemm_raw_kernel elsewhere) against [`set_raw_staging`]`(false)` (`--no-moe-raw-staging`:
/// gemm_kernel, loads straight from global memory). Q5_K / Q8_0 (#836 part 10) at
/// Qwen3.8-Flash-Next's n_ff of 640 (Q8_0 rows of 680 bytes, only 8-byte aligned), at verify
/// step sizes and a prompt step.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_kernels_agree_bitwise() {
    // Back to the default even when an assert fails.
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            set_raw_staging(true);
        }
    }
    let _reset = Reset;
    // the F32 per-pair outputs (the default) and the 16-bit ones (--moe-pair-out, #825: the
    // fragment down kernel stores them from its registers)
    let q5k = |n_tok| Case {
        gu: types::Q5_K,
        down: types::Q8_0,
        n_ff: 640,
        n_tok,
        ..wide_case()
    };
    let mut all = vec![wide_case(), q5k(16), q5k(80), q5k(300)];
    all.push(Case {
        n_ff: 256,
        ..q5k(80)
    });
    for c in all {
        for pair in [PairOut::F32, PairOut::F16, PairOut::Bf16] {
            let case = Case { pair, ..c };
            let label = format!(
                "types {}/{}, n_ff {}, {} tokens, {pair:?}",
                c.gu, c.down, c.n_ff, c.n_tok
            );
            set_raw_staging(true);
            let mine = cuda_output_hash(&case);
            set_raw_staging(false);
            let theirs = cuda_output_hash(&case);
            eprintln!("{label}: default kernels: {mine:016x}; --no-moe-raw-staging: {theirs:016x}");
            assert_eq!(mine, theirs, "{label}");
        }
    }
}

/// [`glm_kernels`] marks the op's node; the op leaves it unmarked by default.
#[test]
fn glm_kernels_marks_the_node() {
    let ctx = Context::new(64, 0);
    let (e, f, s, n) = (4096, 2048, 8, 5);
    let refs = || ExpertRefs {
        up: ctx.new_tensor(types::IQ2_XXS, &[e, f, s]),
        gate: ctx.new_tensor(types::IQ2_XXS, &[e, f, s]),
        down: ctx.new_tensor(types::IQ2_S, &[f, e, s]),
        table: None,
    };
    let op = || {
        let x = ctx.new_tensor(types::F32, &[e, n]);
        let ids = ctx.new_tensor(types::I32, &[2, n]);
        let w = ctx.new_tensor(types::F32, &[2, n]);
        moe_ffn(&ctx, x, ids, w, refs(), PairOut::F32)
    };
    // SAFETY: views of fresh moe_ffn nodes.
    let marked = |t: Tensor<'_>| unsafe { (*(*t.raw()).view_src).op_params[P_GLM] };
    assert_eq!(marked(op()), 0);
    assert_eq!(marked(glm_kernels(op(), true)), 1);
    assert_eq!(marked(glm_kernels(glm_kernels(op(), true), false)), 0);
    // the int8 GEMMs (#603): their own mark, off by default, independent of the kernels' one
    // SAFETY: as above.
    let q8 = |t: Tensor<'_>| unsafe { (*(*t.raw()).view_src).op_params[P_GLM_Q8] };
    assert_eq!(q8(glm_kernels(op(), true)), 0);
    let both = glm_q8(glm_kernels(op(), true), true);
    assert_eq!((marked(both), q8(both)), (1, 1));
    assert_eq!(q8(glm_q8(glm_q8(op(), true), false)), 0);
}

/// GLM-5.3-Flash's int8 prompt GEMMs (#603, opt-in; IQ2_S and IQ3_XXS down, #594): within the
/// activations' 8-bit rounding of the F16 GEMMs over full and partial tiles (the table, poisoned
/// spare slots), not their bits; a 16-token step (the decode kernels' size) keeps the F16 path's
/// bits.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_glm_q8_close_to_f16() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut backend = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let buft = backend.buffer_type();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for (down, n_tok) in [
        (types::IQ2_S, 16),
        (types::IQ2_S, 17),
        (types::IQ2_S, 150),
        (types::IQ2_S, 700),
        (types::IQ3_XXS, 16),
        (types::IQ3_XXS, 150),
        (types::IQ3_XXS, 700),
    ] {
        let base = Case {
            gu: types::IQ2_XXS,
            down,
            n_embd: 4096,
            n_ff: 2048,
            n_used: 4,
            n_tok,
            spare: 2,
            glm: true,
            ..cases().remove(0)
        };
        let (f16, _, _) = run(&base, &mut backend, buft);
        let (q8, _, _) = run(&Case { q8: true, ..base }, &mut backend, buft);
        assert!(
            q8.iter().all(|v| v.is_finite()),
            "{down:?} down, {n_tok} tokens: non-finite outputs"
        );
        if n_tok <= 16 {
            assert_eq!(
                bits(&q8),
                bits(&f16),
                "{down:?} {n_tok} tokens: not the F16 path"
            );
            continue;
        }
        let scale = f16.iter().fold(0f32, |m, v| m.max(v.abs()));
        let diff = q8
            .iter()
            .zip(&f16)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "{down:?} down, {n_tok} tokens: int8 vs F16 GEMMs max |diff| {diff:e} of max |out| {scale:e}"
        );
        assert!(
            diff < 2e-2 * scale,
            "{down:?} {n_tok} tokens: {diff:e} vs {scale:e}"
        );
        assert!(q8 != f16, "{down:?} {n_tok} tokens: no effect");
    }
}

/// GLM-5.3-Flash's IQ2 GEMMs (#571; sm_80+, Maya-S's row sizes: IQ2_XXS gate/up 4096 wide, IQ2_S
/// or IQ3_XXS down 2048 wide) give the shared kernels' bits, over full and partial tiles, with
/// the table and poisoned spare slots; 1-16 token steps take the decode kernel (#594), longer
/// ones gemm_glm_kernel, IQ3_XXS down included (#853).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_glm_kernels_agree_bitwise() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut backend = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let buft = backend.buffer_type();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for down in [types::IQ2_S, types::IQ3_XXS] {
        for n_tok in [150, 700, 1, 3, 16] {
            let base = Case {
                gu: types::IQ2_XXS,
                down,
                n_embd: 4096,
                n_ff: 2048,
                n_used: 4,
                n_tok,
                spare: 2,
                glm: true,
                ..cases().remove(0)
            };
            let (glm, graph, _) = run(&base, &mut backend, buft);
            let (shared, _, _) = run(&Case { glm: false, ..base }, &mut backend, buft);
            let diff = glm
                .iter()
                .zip(&graph)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "{down:?} down, {n_tok} tokens: GLM kernels vs graph path: max |diff| {diff:e}"
            );
            assert!(glm.iter().all(|v| v.is_finite()), "non-finite outputs");
            assert_eq!(bits(&glm), bits(&shared), "{down:?} down, {n_tok} tokens");
        }
    }
}

/// The decode kernel's IQ2_S decoder (#871, `GlmIq2sDec` in `moe_gemm_glm.cuh`) computes the
/// selectors `GlmIq2s`'s table held from its per-row spread code words, and each row's copy of
/// the codebook stays in its own four banks. A host model of the device expressions.
#[test]
fn glm_iq2s_dec_selectors_and_layout() {
    // GlmIq2s::tables' entry for a 16-bit code; glm_spread, glm_sel
    let sel_table = |code: u32, tg: u32| {
        let cc = (code >> (4 * tg)) & 15;
        0x1010 + (cc & 3) * 0x22 + (cc >> 2) * 0x2200
    };
    let spread = |code: u32| {
        (0..4).fold(0u32, |w, tg| {
            w | (((code >> (4 * tg)) & 3) << (2 * tg))
                | (((code >> (4 * tg + 2)) & 3) << (8 + 2 * tg))
        })
    };
    let sel = |w: u32, tg: u32| ((w >> (2 * tg)) & 0x303) * 0x22 + 0x1010;
    for code in 0..1u32 << 16 {
        let w = spread(code);
        assert!(w <= 0xFFFF);
        for tg in 0..4 {
            assert_eq!(sel(w, tg), sel_table(code, tg), "code {code:#x} tg {tg}");
        }
    }

    // row g's entry e at byte 16 g + 2 e + 14 (e & ~7) of the 16 KB table
    let mut used = [false; 8 * 1024 * 2];
    for g in 0..8usize {
        for e in 0..1024usize {
            let a = 16 * g + 2 * e + 14 * (e & !7);
            assert!(a + 2 <= used.len(), "row {g} entry {e} past the table");
            assert!(
                (4 * g..4 * g + 4).contains(&(a / 4 % 32)),
                "row {g} entry {e} bank"
            );
            assert!(!used[a] && !used[a + 1], "row {g} entry {e} overlaps");
            used[a] = true;
            used[a + 1] = true;
        }
    }
}

/// Per-pair outputs kept in F16 / BF16 (`--moe-pair-out`): the same sums with each pair's
/// output rounded once, so within a few roundings of 16 bits of the F32 path's error.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_pair_out_formats() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut backend = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let base = Case {
        n_tok: 300,
        ..cases().remove(0)
    };
    let buft = backend.buffer_type();
    let (f32_out, _, want) = run(&base, &mut backend, buft);
    let r32 = rel(&f32_out, &want);
    for (pair, bound) in [(PairOut::F16, 2e-3), (PairOut::Bf16, 1e-2)] {
        let (out, _, _) = run(&Case { pair, ..base }, &mut backend, buft);
        let r = rel(&out, &want);
        eprintln!("{pair:?}: op vs oracle {r:e} (F32: {r32:e})");
        assert!(r < bound, "{pair:?}: op vs oracle {r:e}");
        assert!(out != f32_out, "{pair:?}: no effect");
    }
}

/// Times the CUDA op at Qwen3.8-Flash-Next's shape (2560 -> 640, 512 experts, 10 per token,
/// Q4_K gate/up, Q5_1 down) for 256-8K-token steps, with skewed and with uniform routing.
/// Prints milliseconds and the products' rate.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU; a timing, not a check"]
fn cuda_moe_speed() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut backend = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let (n_embd, n_ff, n_expert, n_used) = (2560usize, 640usize, 512usize, 10usize);
    // The model's types (Q4_K / Q5_1 in most layers), or LEGION_MOE_SPEED_TYPES=gu,down (GGML
    // type numbers) for the others (Q5_K = 13, Q8_0 = 8).
    let (gu_ty, down_ty) = std::env::var("LEGION_MOE_SPEED_TYPES")
        .ok()
        .and_then(|v| {
            let (a, b) = v.split_once(',')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or((types::Q4_K, types::Q5_1));
    let wctx = Context::new(8, 0);
    let (e, f, s) = (n_embd as i64, n_ff as i64, n_expert as i64);
    let up = wctx.new_tensor(gu_ty, &[e, f, s]);
    let gate = wctx.new_tensor(gu_ty, &[e, f, s]);
    let down = wctx.new_tensor(down_ty, &[f, e, s]);
    wctx.alloc_tensors(backend.buffer_type()).unwrap();
    for (i, (t, k)) in [(up, n_embd), (gate, n_embd), (down, n_ff)]
        .into_iter()
        .enumerate()
    {
        let rows = t.elements() / k;
        // 61 quantized rows, picked by a hash of the row's index: the timing does not depend on
        // the values, and the output hash below sees rows and experts mixed up.
        let pool: Vec<Vec<u8>> = (0..61)
            .map(|s| crate::ggml::quantize(t.ty(), &noise(k, 9 + 100 * i as u64 + s), k))
            .collect();
        let mut all = Vec::with_capacity(pool[0].len() * rows);
        for r in 0..rows as u64 {
            let h = r.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32;
            all.extend_from_slice(&pool[(h % 61) as usize]);
        }
        t.set_bytes(&all);
    }
    for skewed in [true, false] {
        for n_tok in [256usize, 1024, 4096, 8192] {
            let ctx = Context::new(64, 128); // room for the four graphs below
            let xt = ctx.new_tensor(types::F32, &[e, n_tok as i64]).as_input();
            let idt = ctx
                .new_tensor(types::I32, &[n_used as i64, n_tok as i64])
                .as_input();
            let wt = ctx
                .new_tensor(types::F32, &[n_used as i64, n_tok as i64])
                .as_input();
            let act = ctx
                .new_tensor(types::F32, &[f, n_used as i64, n_tok as i64])
                .as_input();
            let refs = ExpertRefs {
                up,
                gate,
                down,
                table: None,
            };
            let op = moe_ffn(&ctx, xt, idt, wt, refs, PairOut::F32).as_output();
            // ggml's mul_mat_id (MMQ) on the same weights and routing, gate/up and down apart:
            // the graph path's rates for the same products.
            let x3 = ctx.reshape_3d(xt, e, 1, n_tok as i64);
            let mm_up = ctx.mul_mat_id(up, x3, idt).as_output();
            let mm_gate = ctx.mul_mat_id(gate, x3, idt).as_output();
            let mm_down = ctx.mul_mat_id(down, act, idt).as_output();
            fn graph_of<'a>(ctx: &'a Context, nodes: &[Tensor<'a>]) -> crate::ggml::Graph<'a> {
                let mut g = ctx.new_graph(16);
                for n in nodes {
                    g.expand(*n);
                }
                g
            }
            let all = graph_of(&ctx, &[op, mm_up, mm_gate, mm_down]);
            ctx.alloc_graph(&all, ComputeAllocator::new(backend.buffer_type()))
                .unwrap();
            xt.set_f32(&noise(n_embd * n_tok, 4));
            act.set_f32(&noise(n_ff * n_used * n_tok, 7));
            let r = noise(n_tok * n_used * 4, 6);
            let mut ids = Vec::new();
            for t in 0..n_tok {
                let mut picked: Vec<i32> = Vec::new();
                let mut i = 0;
                while picked.len() < n_used {
                    // Skewed: squares of uniform numbers favour low ids.
                    let u = r[(t * n_used * 4 + i) % r.len()].abs();
                    let u = if skewed { u * u } else { u };
                    let ex = ((u * n_expert as f32) as usize).min(n_expert - 1) as i32;
                    let ex = if picked.contains(&ex) {
                        (ex + 1 + i as i32 * 37) % n_expert as i32
                    } else {
                        ex
                    };
                    if !picked.contains(&ex) {
                        picked.push(ex);
                    }
                    i += 1;
                }
                ids.extend(picked);
            }
            idt.set_i32(&ids);
            wt.set_f32(&vec![0.1; n_used * n_tok]);
            let routing = if skewed { "skewed" } else { "uniform" };
            let flop = 2.0 * (n_embd * n_ff * n_used * n_tok) as f64;
            for (name, nodes, flop) in [
                ("moe_ffn", &[op][..], 3.0 * flop),
                ("mul_mat_id gate/up", &[mm_up, mm_gate][..], 2.0 * flop),
                ("mul_mat_id down", &[mm_down][..], flop),
            ] {
                let g = graph_of(&ctx, nodes);
                backend.compute(&g).unwrap();
                let t0 = std::time::Instant::now();
                for _ in 0..5 {
                    backend.compute(&g).unwrap();
                }
                let ms = t0.elapsed().as_secs_f64() * 1e3 / 5.0;
                // The op's output: FNV-1a over its bits, equal across builds when the kernels
                // are bitwise the same.
                let hash = if name == "moe_ffn" {
                    let h = op.to_f32().iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
                        (h ^ v.to_bits() as u64).wrapping_mul(0x100_0000_01b3)
                    });
                    format!(", output hash {h:016x}")
                } else {
                    String::new()
                };
                println!(
                    "{name} {routing} {n_tok} tokens: {ms:.2} ms, {:.1} TFLOP/s{hash}",
                    flop / ms / 1e9
                );
            }
        }
    }
}

/// Routing entries the weights don't hold (ids -1 and `n_expert`, and an expert whose table
/// slot is past the weights) are left out on CUDA instead of indexing the counters and the
/// weights out of bounds: each token's output equals that of the same routing with those
/// pairs pointed at a real expert and weighted 0, bit for bit. Run it under
/// `compute-sanitizer --tool memcheck` as well.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_skips_experts_out_of_range() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut backend = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    skips_experts_out_of_range(&mut backend);
}

/// The CPU implementation leaves the same pairs out ([`cuda_skips_experts_out_of_range`]).
#[test]
fn cpu_skips_experts_out_of_range() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    skips_experts_out_of_range(&mut backend);
}

fn skips_experts_out_of_range(backend: &mut Backend) {
    let (n_embd, n_ff, n_expert, n_used, n_tok) = (256usize, 128usize, 8usize, 3usize, 70usize);
    let n_slots = n_expert;
    let (uq, _) = weights(types::Q4_K, n_embd, n_ff, n_slots, 1);
    let (gq, _) = weights(types::Q4_K, n_embd, n_ff, n_slots, 2);
    let (dq, _) = weights(types::Q5_1, n_ff, n_embd, n_slots, 3);
    // Expert 7's slot is past the weights.
    let mut table: Vec<i32> = (0..n_expert as i32).collect();
    table[7] = n_slots as i32;
    let x = noise(n_embd * n_tok, 4);
    let good: Vec<i32> = (0..n_tok * n_used)
        .map(|i| ((i / n_used * 5 + i % n_used * 3) % (n_expert - 1)) as i32)
        .collect();
    let w0 = noise(n_used * n_tok, 5)
        .iter()
        .map(|v| v.abs() + 0.1)
        .collect::<Vec<_>>();
    // Every fifth pair broken three ways; the reference weights those pairs 0 instead.
    let (mut bad, mut w_ref) = (good.clone(), w0.clone());
    for i in (0..n_tok * n_used).step_by(5) {
        bad[i] = [-1, n_expert as i32, 7][i / 5 % 3];
        w_ref[i] = 0.0;
    }
    let run = |backend: &mut Backend, ids: &[i32], w: &[f32]| {
        let wctx = Context::new(4, 0);
        let (e, f, s) = (n_embd as i64, n_ff as i64, n_slots as i64);
        let up = wctx.new_tensor(types::Q4_K, &[e, f, s]);
        let gate = wctx.new_tensor(types::Q4_K, &[e, f, s]);
        let down = wctx.new_tensor(types::Q5_1, &[f, e, s]);
        wctx.alloc_tensors(backend.buffer_type()).unwrap();
        up.set_bytes(&uq);
        gate.set_bytes(&gq);
        down.set_bytes(&dq);
        let ctx = Context::new(16, 16);
        let (nt, nu) = (n_tok as i64, n_used as i64);
        let xt = ctx.new_tensor(types::F32, &[e, nt]).as_input();
        let idt = ctx.new_tensor(types::I32, &[nu, nt]).as_input();
        let wt = ctx.new_tensor(types::F32, &[nu, nt]).as_input();
        let tab = ctx.new_tensor(types::I32, &[n_expert as i64]).as_input();
        let refs = ExpertRefs {
            up,
            gate,
            down,
            table: Some(tab),
        };
        let op = moe_ffn(&ctx, xt, idt, wt, refs, PairOut::F32).as_output();
        let mut graph = ctx.new_graph(16);
        graph.expand(op);
        ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
            .unwrap();
        xt.set_f32(&x);
        idt.set_i32(ids);
        wt.set_f32(w);
        tab.set_i32(&table);
        backend.compute(&graph).unwrap();
        op.to_f32()
    };
    let got = run(backend, &bad, &w0);
    let want = run(backend, &good, &w_ref);
    assert!(got.iter().all(|v| v.is_finite()), "non-finite output");
    assert_eq!(got.len(), want.len());
    let differ = got
        .iter()
        .zip(&want)
        .filter(|(a, b)| a.to_bits() != b.to_bits() && !(**a == 0.0 && **b == 0.0))
        .count();
    assert_eq!(differ, 0, "{differ} values differ");
}

/// What [`shexp_case`] computes.
#[derive(Clone, Copy, PartialEq)]
enum ShexpPath {
    /// [`moe_ffn_shexp`].
    Fold,
    /// ggml's `add(moe_ffn, mul(sh, sigmoid(g)))` (the GPU test's reference).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    Chain,
    /// [`moe_ffn`] alone.
    Plain,
    /// [`moe_ffn_shexp`] with its sum handed to a reader ([`defer_sum`]; the CPU writes it
    /// anyway).
    Deferred,
}

/// [`shexp_case`]'s shared expert and gate logits (`[2, n]`, of which row 1 is read).
fn shexp_inputs(n_embd: usize, n_tok: usize) -> (Vec<f32>, Vec<f32>) {
    // Gate logits over a wide range: sigmoid saturates at both ends.
    let g2 = noise(2 * n_tok, 17).iter().map(|v| v * 24.0).collect();
    (noise(n_embd * n_tok, 16), g2)
}

/// The op with a gated shared expert ([`Shexp`]) on `backend`, `n_tok` tokens of 256 values,
/// through a table or not, as `path` computes it (each in a graph of its own: the allocator
/// may reuse a view's node once the view is read). The gate logits are a strided view (row 1
/// of `[2, n]`), as a joined matmul's are.
fn shexp_case(backend: &mut Backend, n_tok: usize, table: bool, path: ShexpPath) -> Vec<f32> {
    let (n_embd, n_ff, n_expert, n_used) = (256usize, 128usize, 8usize, 3usize);
    let (uq, _) = weights(types::Q8_0, n_embd, n_ff, n_expert, 11);
    let (gq, _) = weights(types::Q8_0, n_embd, n_ff, n_expert, 12);
    let (dq, _) = weights(types::Q5_1, n_ff, n_embd, n_expert, 13);
    let wctx = Context::new(4, 0);
    let (e, f, s) = (n_embd as i64, n_ff as i64, n_expert as i64);
    let up = wctx.new_tensor(types::Q8_0, &[e, f, s]);
    let gate = wctx.new_tensor(types::Q8_0, &[e, f, s]);
    let down = wctx.new_tensor(types::Q5_1, &[f, e, s]);
    wctx.alloc_tensors(backend.buffer_type()).unwrap();
    up.set_bytes(&uq);
    gate.set_bytes(&gq);
    down.set_bytes(&dq);
    let ctx = Context::new(32, 32);
    let (nt, nu) = (n_tok as i64, n_used as i64);
    let xt = ctx.new_tensor(types::F32, &[e, nt]).as_input();
    let idt = ctx.new_tensor(types::I32, &[nu, nt]).as_input();
    let wt = ctx.new_tensor(types::F32, &[nu, nt]).as_input();
    let tab = ctx.new_tensor(types::I32, &[n_expert as i64]).as_input();
    let sht = ctx.new_tensor(types::F32, &[e, nt]).as_input();
    let g2 = ctx.new_tensor(types::F32, &[2, nt]).as_input();
    let g = ctx.view_2d(g2, [1, nt], 8, 4);
    let refs = ExpertRefs {
        up,
        gate,
        down,
        table: table.then_some(tab),
    };
    let shexp = Shexp { sh: sht, g };
    assert!(shexp_fits(&shexp, e, nt));
    let out = match path {
        ShexpPath::Fold => moe_ffn_shexp(&ctx, xt, idt, wt, refs, PairOut::F32, Some(shexp)),
        ShexpPath::Plain => moe_ffn(&ctx, xt, idt, wt, refs, PairOut::F32),
        ShexpPath::Deferred => {
            let out = moe_ffn_shexp(&ctx, xt, idt, wt, refs, PairOut::F32, Some(shexp));
            defer_sum(out);
            out
        }
        ShexpPath::Chain => {
            let plain = moe_ffn(&ctx, xt, idt, wt, refs, PairOut::F32);
            ctx.add(plain, ctx.mul(sht, ctx.sigmoid(ctx.cont(g))))
        }
    }
    .as_output();
    let mut graph = ctx.new_graph(32);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    xt.set_f32(&noise(n_embd * n_tok, 14));
    let ids: Vec<i32> = (0..n_tok * n_used)
        .map(|i| ((i / n_used * 5 + i % n_used * 3) % n_expert) as i32)
        .collect();
    idt.set_i32(&ids);
    let w: Vec<f32> = noise(n_used * n_tok, 15)
        .iter()
        .map(|v| v.abs() + 0.1)
        .collect();
    wt.set_f32(&w);
    if table {
        tab.set_i32(&(0..n_expert as i32).collect::<Vec<_>>());
    }
    let (sh, gv) = shexp_inputs(n_embd, n_tok);
    if path != ShexpPath::Plain {
        sht.set_f32(&sh);
        g2.set_f32(&gv);
    }
    backend.compute(&graph).unwrap();
    out.to_f32()
}

/// On the CPU the op adds the shared expert as its reference says: the plain op's sum plus
/// `sh · sigmoid(g)`, the product and the sum rounded apart.
#[test]
fn cpu_adds_the_shared_expert() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(3);
    let (n_embd, n_tok) = (256, 7);
    let (sh, g2) = shexp_inputs(n_embd, n_tok);
    for table in [false, true] {
        let fold = shexp_case(&mut cpu, n_tok, table, ShexpPath::Fold);
        let plain = shexp_case(&mut cpu, n_tok, table, ShexpPath::Plain);
        for (i, (got, p)) in fold.iter().zip(&plain).enumerate() {
            let s = 1.0 / (1.0 + (-g2[2 * (i / n_embd) + 1]).exp());
            let want = p + sh[i] * s;
            assert_eq!(got.to_bits(), want.to_bits(), "value {i}, table {table}");
        }
    }
}

/// A deferred sum (#774, `--hc-fuse`): [`deferrable`] takes a fresh op's whole F32 sum only,
/// [`defer_sum`] marks it once, and the CPU op still writes the sum, the same bits.
#[test]
fn defer_sum_marks_the_node_and_the_cpu_still_sums() {
    let ctx = Context::new(64, 0);
    let (e, f, s, n) = (256, 128, 8, 5);
    let refs = || ExpertRefs {
        up: ctx.new_tensor(types::Q8_0, &[e, f, s]),
        gate: ctx.new_tensor(types::Q8_0, &[e, f, s]),
        down: ctx.new_tensor(types::Q8_0, &[f, e, s]),
        table: None,
    };
    let op = |pair| {
        let x = ctx.new_tensor(types::F32, &[e, n]);
        let ids = ctx.new_tensor(types::I32, &[2, n]);
        let w = ctx.new_tensor(types::F32, &[2, n]);
        moe_ffn(&ctx, x, ids, w, refs(), pair)
    };
    let out = op(PairOut::F32);
    assert!(deferrable(out) && !sum_deferred(out));
    // not the op's whole sum: a later view, another tensor, F16 pair rows
    assert!(!deferrable(ctx.view_2d(
        out,
        [e, n - 1],
        e as usize * 4,
        e as usize * 4
    )));
    assert!(!deferrable(ctx.new_tensor(types::F32, &[e, n])));
    assert!(!deferrable(ctx.scale(out, 2.0)));
    assert!(!deferrable(op(PairOut::F16)));
    defer_sum(out);
    assert!(sum_deferred(out) && !deferrable(out));
    // the shared expert's bit stays apart
    // SAFETY: a view of a fresh moe_ffn node.
    assert_eq!(
        unsafe { (*(*out.raw()).view_src).op_params[P_SHEXP] },
        SUM_DEFERRED
    );
    // what the reader must keep alive: the ids, then the table and the shared expert's two
    let raw = |ts: Vec<Tensor<'_>>| ts.iter().map(|t| t.raw()).collect::<Vec<_>>();
    let node = out.srcs()[0];
    assert_eq!(raw(sum_srcs(out)), raw(vec![node.srcs()[1]]));
    let x = ctx.new_tensor(types::F32, &[e, n]);
    let ids = ctx.new_tensor(types::I32, &[2, n]);
    let w = ctx.new_tensor(types::F32, &[2, n]);
    let table = ctx.new_tensor(types::I32, &[16]);
    let (sh, g) = (
        ctx.new_tensor(types::F32, &[e, n]),
        ctx.new_tensor(types::F32, &[1, n]),
    );
    let tabled = ExpertRefs {
        table: Some(table),
        ..refs()
    };
    let out = moe_ffn_shexp(&ctx, x, ids, w, tabled, PairOut::F32, Some(Shexp { sh, g }));
    defer_sum(out);
    assert_eq!(raw(sum_srcs(out)), raw(vec![ids, table, sh, g]));

    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(3);
    for table in [false, true] {
        let fold = shexp_case(&mut cpu, 7, table, ShexpPath::Fold);
        let deferred = shexp_case(&mut cpu, 7, table, ShexpPath::Deferred);
        let same = fold
            .iter()
            .zip(&deferred)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(same, "table {table}");
    }
}

/// On CUDA the op with the shared expert is ggml's sigmoid, mul and add after the op, bit for
/// bit, from 70 to 4096 tokens.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_shexp_sum_is_ggml_bit_for_bit() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for n in [70, 300, 4096] {
        for table in [false, true] {
            let fold = shexp_case(&mut gpu, n, table, ShexpPath::Fold);
            let chain = shexp_case(&mut gpu, n, table, ShexpPath::Chain);
            let moved = fold
                .iter()
                .zip(&chain)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(moved, 0, "{n} tokens, table {table}: {moved} values moved");
        }
    }
}

/// The down GEMM's own per-token sums ([`set_down_sum`], opt-in; #826: on the GB10's
/// fragment down kernel) give the sum kernel's bits: Q5_1 and Q8_0 down, full and partial tiles,
/// with and without a gated shared expert and a slot table. Elsewhere both runs take the sum
/// kernel.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_down_sum_is_the_reduce_bit_for_bit() {
    use crate::ggml::{DeviceKind, device_by_kind};
    // Back to the default even when an assert fails.
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            set_down_sum(false);
        }
    }
    let _reset = Reset;
    let q5k = |n_tok| Case {
        gu: types::Q5_K,
        down: types::Q8_0,
        n_ff: 640,
        n_tok,
        ..wide_case()
    };
    for c in [wide_case(), q5k(16), q5k(80), q5k(300)] {
        set_down_sum(true);
        let mine = cuda_output_hash(&c);
        set_down_sum(false);
        let theirs = cuda_output_hash(&c);
        assert_eq!(
            mine, theirs,
            "types {}/{}, {} tokens",
            c.gu, c.down, c.n_tok
        );
    }
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for n in [70, 300, 4096] {
        for table in [false, true] {
            for path in [ShexpPath::Fold, ShexpPath::Plain] {
                set_down_sum(true);
                let mine = shexp_case(&mut gpu, n, table, path);
                set_down_sum(false);
                let theirs = shexp_case(&mut gpu, n, table, path);
                let moved = mine
                    .iter()
                    .zip(&theirs)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(moved, 0, "{n} tokens, table {table}: {moved} values moved");
            }
        }
    }
}

/// Project Maya's quants (#559) are GLM-only: [`supported_glm`] takes them, [`supported`] (every
/// other model's check) refuses them as before; both take the formats they took before.
#[test]
fn maya_types_are_glm_only() {
    let check = |gu, down| {
        let ctx = Context::new(8, 0);
        let up = ctx.new_tensor(gu, &[512, 256, 4]);
        let gate = ctx.new_tensor(gu, &[512, 256, 4]);
        let down = ctx.new_tensor(down, &[256, 512, 4]);
        let e = ExpertRefs {
            up,
            gate,
            down,
            table: None,
        };
        (supported(&e), supported_glm(&e))
    };
    assert_eq!(check(types::Q4_K, types::Q6_K), (true, true));
    assert_eq!(check(types::IQ2_XXS, types::IQ2_S), (true, true));
    for (gu, down) in [
        (types::IQ2_XXS, types::IQ3_XXS),
        (types::Q2_K, types::Q3_K),
        (types::IQ3_S, types::IQ4_XS),
        // Not a Maya quant: DeepSeek-V4.1-Flash's (#584), also through supported_glm only.
        (types::IQ2_XS, types::IQ2_XXS),
        (types::IQ1_M, types::IQ2_XS),
    ] {
        assert_eq!(check(gu, down), (false, true), "{gu}/{down}");
    }
    // Still not a format the kernels read.
    assert_eq!(check(types::IQ4_NL, types::IQ2_S), (false, false));
}

/// The fragment GEMMs' block-row cap: decode and verify steps of Qwen3.8-Flash-Next's 512
/// experts (10 used) get it where the uncapped grid is larger; prefill keeps one row per tile.
#[cfg(feature = "cuda")]
#[test]
fn frag_ycap_policy() {
    let cap = |rows: i64| {
        let pairs = rows * 10;
        cuda::frag_ycap(pairs, 512, pairs / TILE_PAIRS + pairs.min(512))
    };
    // 1 row (+ 8 drafts): 91 tiles at most, under the cap
    assert_eq!(cap(9), 0);
    // C=4 and C=16 verify steps (36 and 144 rows)
    assert_eq!(cap(36), 256 + 5);
    assert_eq!(cap(144), 256 + 22);
    // prefill: at least four pairs per expert
    assert_eq!(cap(205), 0);
    assert_eq!(cap(8192), 0);
}
