//! The paged op over a compact cache (#551, [`attention_granules`]): per cell and kv head one
//! granule of `d + r` values, K the first `d` of them and V the `d` from `r` on (a view of the
//! same rows), against the f64 oracle on those K and V.

use super::tests::{Shape, f16_bytes, noise, oracle};
use super::*;
use crate::ggml::{Backend, ComputeAllocator};

/// The op on `backend` over `n_blocks` blocks of granules, `n_tok` tokens per sequence
/// (`n_past`, `tables` per sequence): its output and the oracle's.
#[allow(clippy::too_many_arguments)]
fn granules(
    backend: &mut Backend,
    sh: &Shape,
    r: usize,
    n_past: &[usize],
    tables: &[Vec<u32>],
) -> (Vec<f32>, Vec<f32>) {
    let (d, n_head, n_head_kv, n_tok) = (sh.d, sh.n_head, sh.n_head_kv, sh.n_tok);
    let hs = d + r;
    let row = hs * n_head_kv;
    let cells = (tables.iter().flatten().max().unwrap() + 1) as usize * KV_BLOCK;
    let pool = Context::new(1, 0);
    // One pad cell: V's last head ends r values into it.
    let g = pool.new_tensor(types::F16, &[row as i64, cells as i64 + 1]);
    pool.alloc_tensors(backend.buffer_type()).unwrap();
    let mut gdat = noise(g.elements(), 31);
    g.set_bytes(&f16_bytes(&gdat));
    round_f16(&mut gdat);
    // The oracle's K and V, `[d × n_head_kv]` per cell.
    let (mut kdat, mut vdat) = (Vec::new(), Vec::new());
    for c in 0..cells {
        for h in 0..n_head_kv {
            let at = c * row + h * hs;
            kdat.extend_from_slice(&gdat[at..at + d]);
            vdat.extend_from_slice(&gdat[at + r..at + r + d]);
        }
    }

    let ns = n_past.len();
    let width = tables.iter().map(Vec::len).max().unwrap();
    let ctx = Context::new(64, 64);
    let k = ctx.view_2d(g, [row as i64, cells as i64], row * 2, 0);
    let v = ctx.view_2d(g, [row as i64, cells as i64], row * 2, r * 2);
    let q = ctx
        .new_tensor(
            types::F32,
            &[d as i64, n_head as i64, n_tok as i64, ns as i64],
        )
        .as_input();
    let tab = ctx
        .new_tensor(types::I32, &[width as i64, ns as i64])
        .as_input();
    let past = ctx.new_tensor(types::I32, &[ns as i64]).as_input();
    let scratch_ne = scratch_shape(d as i64, n_head as i64, n_head_kv as i64, n_tok, ns);
    let scratch = ctx.new_tensor(types::F32, &scratch_ne).as_input();
    let out = attention_granules(
        &ctx,
        q,
        k,
        v,
        tab,
        past,
        scratch,
        sh.scale as f32,
        sh.window,
        None,
        hs,
    )
    .as_output();
    let mut graph = ctx.new_graph(64);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let qmul = 2.0 * sh.scale as f32 * (d as f32).sqrt().min(16.0);
    let qv: Vec<f32> = noise(q.elements(), 33).iter().map(|x| qmul * x).collect();
    q.set_f32(&qv);
    let mut t = Vec::new();
    for (s, tb) in tables.iter().enumerate() {
        assert!(tb.len() * KV_BLOCK >= n_past[s] + n_tok);
        t.extend(tb.iter().map(|&x| x as i32));
        t.resize((s + 1) * width, 0);
    }
    tab.set_i32(&t);
    past.set_i32(&n_past.iter().map(|&p| p as i32).collect::<Vec<_>>());
    backend.compute(&graph).unwrap();
    let want = oracle(sh, &qv, &kdat, &vdat, n_past, tables);
    (out.to_f32(), want)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn shape(d: usize, n_head: usize, n_head_kv: usize, n_tok: usize, window: usize) -> Shape {
    Shape {
        d,
        n_head,
        n_head_kv,
        n_tok,
        scale: 1.0 / (d as f64).sqrt(),
        window,
    }
}

#[test]
fn cpu_granules_match_oracle() {
    let mut b = Backend::cpu().unwrap();
    b.set_threads(4);
    let tables = [vec![4, 1], vec![0, 5, 2]];
    for (n_tok, window) in [(1, 0), (5, 0), (3, 40)] {
        let sh = shape(64, 8, 2, n_tok, window);
        let (got, want) = granules(&mut b, &sh, 32, &[70, 120], &tables);
        let diff = max_diff(&got, &want);
        assert!(diff < 1e-5, "{n_tok} tokens, window {window}: {diff}");
    }
}

/// Gemma 4 31B's global heads (512 wide, 128 rotated, 32 over 4 kv heads): decode and prefill
/// chunks on the GPU's kernels.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_granules_match_oracle() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut b = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let tables = [vec![5, 0, 9, 3, 7], vec![2, 10], vec![8, 1, 6, 4]];
    for (n_tok, window) in [(1, 0), (37, 0), (70, 100)] {
        let sh = shape(512, 32, 4, n_tok, window);
        let (got, want) = granules(&mut b, &sh, 128, &[190, 3, 120], &tables);
        let diff = max_diff(&got, &want);
        println!("granules, {n_tok} tokens, window {window}: max |diff| {diff:.2e}");
        // As `cuda_check`'s bound.
        assert!(diff < 5e-4, "{n_tok} tokens, window {window}: {diff}");
    }
}
