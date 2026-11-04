//! The grouped op's bits against ggml's on one GPU (#819): for each weight group (Q8_0 `[k, m_i]`
//! read with one input) and step size, one graph holds ggml's `mul_mat` of every weight and the
//! op's results, and with `glu` (two weights of the same rows: gate, up) ggml's `swiglu_split` of
//! the pair's matmuls (which ggml-cuda fuses into one matvec at one column) against the op's
//! SwiGLU. Used by the hidden `legion q8-group-check`
//! subcommand (a build tree's binary on the GPU box).

use super::{GROUP, Q8_0_BYTES, fits_all, mul_mat, swiglu};
use crate::ggml::{Backend, ComputeAllocator, Context, Tensor, ffi, types};

/// One group at one step size: outputs compared and how many differ in their bits.
#[derive(Clone, Debug)]
pub struct Check {
    pub rows: Vec<usize>,
    pub cols: usize,
    /// Whether the outputs are the SwiGLU of the pair (else each matmul's).
    pub glu: bool,
    pub outputs: usize,
    pub differing: usize,
    /// Largest absolute difference.
    pub max_diff: f32,
}

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

/// A Q8_0 matrix of `blocks` blocks: int8 in -127..=127, scales in [0.001, 0.011).
fn weights(blocks: usize, seed: u64) -> Vec<u8> {
    let qs = noise(blocks * GROUP as usize, seed);
    let ds = noise(blocks, seed + 1);
    let mut out = Vec::with_capacity(blocks * Q8_0_BYTES);
    for (b, d) in ds.iter().enumerate() {
        // SAFETY: a plain conversion.
        let h = unsafe { ffi::ggml_fp32_to_fp16(0.006 + 0.005 * d) };
        out.extend_from_slice(&h.to_le_bytes());
        for q in &qs[b * GROUP as usize..(b + 1) * GROUP as usize] {
            out.push(((q * 127.0).round() as i8) as u8);
        }
    }
    out
}

fn compare(a: &[f32], b: &[f32]) -> (usize, f32) {
    a.iter().zip(b).fold((0, 0f32), |(n, m), (p, q)| {
        (
            n + usize::from(p.to_bits() != q.to_bits()),
            m.max((p - q).abs()),
        )
    })
}

/// Every group of `groups` (row counts, read with one `k`-value input) at every step size of
/// `cols` on `gpu`: the op against ggml's `mul_mat`, and with `glu` (groups of two) the SwiGLU
/// of both against ggml's.
pub fn check(
    gpu: &mut Backend,
    k: usize,
    groups: &[Vec<usize>],
    cols: &[usize],
    glu: bool,
) -> Result<Vec<Check>, String> {
    let mut out = Vec::new();
    for (gi, rows) in groups.iter().enumerate() {
        for &n in cols {
            let ctx = Context::new(64, 64);
            let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
            let ws: Vec<Tensor<'_>> = rows
                .iter()
                .map(|&m| {
                    ctx.new_tensor(types::Q8_0, &[k as i64, m as i64])
                        .as_input()
                })
                .collect();
            if !fits_all(&ws, x) {
                return Err(format!(
                    "rows {rows:?}, {n} columns: the op does not take them"
                ));
            }
            let pair = glu && ws.len() == 2 && rows[0] == rows[1];
            let mut graph = ctx.new_graph(64);
            let refs: Vec<Tensor<'_>> = ws.iter().map(|&w| ctx.mul_mat(w, x)).collect();
            let (refs, ops): (Vec<_>, Vec<_>) = if pair {
                (
                    vec![ctx.swiglu_split(refs[0], refs[1])],
                    vec![swiglu(&ctx, ws[0], ws[1], x)],
                )
            } else {
                (refs, mul_mat(&ctx, &ws, x))
            };
            let refs: Vec<_> = refs.into_iter().map(|t| t.as_output()).collect();
            let ops: Vec<_> = ops.into_iter().map(|t| t.as_output()).collect();
            for &t in refs.iter().chain(&ops) {
                graph.expand(t);
            }
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .map_err(|e| e.to_string())?;
            // Some all-zero groups and one large value: ggml's quantization's corner cases.
            let mut xs = noise(k * n, 1000 + (gi * 16 + n) as u64);
            xs[32..64].fill(0.0);
            xs[k / 2 + 7] = 300.0;
            x.set_f32(&xs);
            for (i, w) in ws.iter().enumerate() {
                let blocks = w.nbytes() / Q8_0_BYTES;
                w.set_bytes(&weights(blocks, 2000 + (gi * 8 + i) as u64));
            }
            gpu.compute(&graph).map_err(|e| e.to_string())?;
            let (mut differing, mut max_diff, mut outputs) = (0, 0f32, 0);
            for (r, o) in refs.iter().zip(&ops) {
                let (a, b) = (o.to_f32(), r.to_f32());
                let (d, m) = compare(&a, &b);
                differing += d;
                max_diff = max_diff.max(m);
                outputs += a.len();
            }
            out.push(Check {
                rows: rows.clone(),
                cols: n,
                glu: pair,
                outputs,
                differing,
                max_diff,
            });
        }
    }
    Ok(out)
}

/// qwen4exp's hyper-connection mixer (`n_embd` `e`, `hc` streams, `rank`, Q8_0 down and up) at
/// each step size of `cols` on `gpu`: the Q8_1 blocks it writes beside the block input
/// ([`super::set_quantized`]) against [`super::quantized`]'s of the same values. Each entry:
/// the step size, the blocks' bytes and how many differ (`None`: the mixer did not write them).
pub fn check_hc(
    gpu: &mut Backend,
    e: usize,
    hc: usize,
    rank: usize,
    cols: &[usize],
) -> Result<Vec<(usize, usize, Option<usize>)>, String> {
    use crate::model::qwen4::hc as mixer;
    let k = e * hc;
    let mut out = Vec::new();
    for &n in cols {
        let ctx = Context::new(64, 64);
        let x = ctx
            .new_tensor(types::F32, &[e as i64, hc as i64, n as i64])
            .as_input();
        let norm = ctx.new_tensor(types::F32, &[k as i64]).as_input();
        let down = ctx
            .new_tensor(types::Q8_0, &[k as i64, rank as i64])
            .as_input();
        let up = ctx
            .new_tensor(types::Q8_0, &[rank as i64, k as i64])
            .as_input();
        if !mixer::fits(x, norm, down, up, None) {
            return Err(format!("{n} columns: the mixer op does not take them"));
        }
        let (mixed, _) = mixer::mix(&ctx, x, norm, down, up, None, 1e-6);
        let written = super::has_quantized(&ctx, mixed);
        let ours = super::quantized(&ctx, mixed).as_output();
        // The same values through a copy, which has no blocks of its own yet: the quantize op.
        let copy = ctx.cont(ctx.scale(mixed, 1.0));
        let theirs = super::quantized(&ctx, copy).as_output();
        let mut graph = ctx.new_graph(64);
        graph.expand(ours);
        graph.expand(theirs);
        ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
            .map_err(|e| e.to_string())?;
        x.set_f32(&noise(k * n, 3000 + n as u64));
        norm.set_f32(
            &noise(k, 4000)
                .iter()
                .map(|v| 1.0 + 0.5 * v)
                .collect::<Vec<_>>(),
        );
        down.set_bytes(&weights(down.nbytes() / Q8_0_BYTES, 5000));
        up.set_bytes(&weights(up.nbytes() / Q8_0_BYTES, 6000));
        gpu.compute(&graph).map_err(|e| e.to_string())?;
        let bytes = super::quantized_bytes(e as i64, n as i64);
        let (a, b) = (ours.bytes(), theirs.bytes());
        let differing = a[..bytes]
            .iter()
            .zip(&b[..bytes])
            .filter(|(p, q)| p != q)
            .count();
        out.push((n, bytes, written.then_some(differing)));
    }
    Ok(out)
}

/// A shared expert's SwiGLU and down projection (Q8_0 gate and up `[k, m]`, down `[m, rows]`)
/// at each step size of `cols` on `gpu`: [`super::swiglu_down`] (the down projection reading
/// the SwiGLU's Q8_1 blocks from its slot) against ggml's `mul_mat` of `swiglu_split` of the
/// pair's matmuls. Each entry: the step size, the outputs and how many differ in their bits
/// (`None`: the op does not take the shapes).
pub fn check_swiglu_down(
    gpu: &mut Backend,
    k: usize,
    m: usize,
    rows: usize,
    cols: &[usize],
) -> Result<Vec<(usize, usize, Option<usize>)>, String> {
    let mut out = Vec::new();
    for &n in cols {
        let ctx = Context::new(64, 64);
        let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
        let gate = ctx
            .new_tensor(types::Q8_0, &[k as i64, m as i64])
            .as_input();
        let up = ctx
            .new_tensor(types::Q8_0, &[k as i64, m as i64])
            .as_input();
        let down = ctx
            .new_tensor(types::Q8_0, &[m as i64, rows as i64])
            .as_input();
        if !super::takes_swiglu_down(gate, up, down, x) {
            out.push((n, 0, None));
            continue;
        }
        // Two layers' slots, the second pair reading the first's output: each its own blocks.
        let ours = super::swiglu_down(&ctx, gate, up, down, x, 3);
        let ours2 = super::swiglu_down(&ctx, gate, up, down, x, 4).as_output();
        let ours = ours.as_output();
        let sh = ctx.swiglu_split(ctx.mul_mat(gate, x), ctx.mul_mat(up, x));
        let theirs = ctx.mul_mat(down, sh).as_output();
        let mut graph = ctx.new_graph(64);
        graph.expand(ours);
        graph.expand(ours2);
        graph.expand(theirs);
        ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
            .map_err(|e| e.to_string())?;
        let mut xs = noise(k * n, 7000 + n as u64);
        xs[32..64].fill(0.0);
        x.set_f32(&xs);
        for (i, w) in [gate, up, down].iter().enumerate() {
            w.set_bytes(&weights(w.nbytes() / Q8_0_BYTES, 8000 + i as u64));
        }
        // Twice: the slots' counters must be back at zero after the first run.
        let mut differing = 0;
        for _ in 0..2 {
            gpu.compute(&graph).map_err(|e| e.to_string())?;
            let t = theirs.to_f32();
            differing += compare(&ours.to_f32(), &t).0 + compare(&ours2.to_f32(), &t).0;
        }
        out.push((n, 4 * rows * n, Some(differing)));
    }
    Ok(out)
}
