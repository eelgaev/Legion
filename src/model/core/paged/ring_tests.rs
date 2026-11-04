//! The paged attention op over sliding-window rings ([`crate::model::core::KvRing`]) against
//! the same attention over every position: the f64 oracle, and the op on a full cache. A ring
//! is filled as the engine fills it (each step's positions written in order, later ones
//! overwriting the ring entries of earlier ones), and the cases put chunk starts, chunk ends
//! and decode positions on and around block boundaries and ring wraps.

use super::tests::{Shape, f16_bytes, noise, oracle};
use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::{KV_BLOCK, KvRing};

/// Sequences at `n_past` (one batch, `n_tok` tokens each) over rings for `window` and steps of
/// `step` tokens: the ring op equals the op over the full cache bit for bit, and the oracle
/// to rounding (`tol`).
#[allow(clippy::too_many_arguments)]
pub(super) fn ring_case(
    backend: &mut Backend,
    d: usize,
    n_head: usize,
    n_head_kv: usize,
    window: usize,
    step: usize,
    n_past: &[usize],
    n_tok: usize,
    tol: f32,
) {
    assert!(n_tok <= step);
    let ns = n_past.len();
    let ring = KvRing::for_window(window, step, ns);
    let row = d * n_head_kv;
    // Every sequence's K and V at every position, F16-rounded.
    let len: Vec<usize> = n_past.iter().map(|p| p + n_tok).collect();
    let width = len.iter().max().unwrap().div_ceil(KV_BLOCK);
    let n_full = width * ns;
    // Full cache: sequence s's block i is pool block (i * ns + s) * m % n_full, shuffled by a
    // multiplier m coprime to n_full.
    let gcd = |mut a: usize, mut b: usize| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    let m = [7, 11, 13, 17, 19, 23]
        .into_iter()
        .find(|&m| gcd(m, n_full) == 1)
        .unwrap();
    let tables: Vec<Vec<u32>> = (0..ns)
        .map(|s| {
            (0..width)
                .map(|i| ((i * ns + s) * m % n_full) as u32)
                .collect()
        })
        .collect();
    let cells_full = n_full * KV_BLOCK;
    let cells_ring = ring.sets * ring.positions();
    let (mut k_full, mut v_full) = (vec![0f32; cells_full * row], vec![0f32; cells_full * row]);
    let (mut k_ring, mut v_ring) = (vec![0f32; cells_ring * row], vec![0f32; cells_ring * row]);
    for (s, &l) in len.iter().enumerate() {
        let kd = noise(l * row, 31 + s as u64);
        let vd = noise(l * row, 47 + s as u64);
        for p in 0..l {
            let full = (tables[s][p / KV_BLOCK] as usize * KV_BLOCK + p % KV_BLOCK) * row;
            let rc = ring.cell(s, p) * row;
            for (pool, at, src) in [
                (&mut k_full, full, &kd),
                (&mut v_full, full, &vd),
                (&mut k_ring, rc, &kd),
                (&mut v_ring, rc, &vd),
            ] {
                pool[at..at + row].copy_from_slice(&src[p * row..(p + 1) * row]);
            }
        }
    }
    // Stale entries of the ring (positions overwritten) are what a real ring holds; the full
    // cache's never-written cells stay 0.
    let ctx = Context::new(64, 64);
    let input = |ne: &[i64], ty| ctx.new_tensor(ty, ne).as_input();
    let q = input(
        &[d as i64, n_head as i64, n_tok as i64, ns as i64],
        types::F32,
    );
    let kf = input(&[row as i64, cells_full as i64], types::F16);
    let vf = input(&[row as i64, cells_full as i64], types::F16);
    let kr = input(&[row as i64, cells_ring as i64], types::F16);
    let vr = input(&[row as i64, cells_ring as i64], types::F16);
    let tf = input(&[width as i64, ns as i64], types::I32);
    let tr = input(&[ring.blocks as i64, ns as i64], types::I32);
    let past = input(&[ns as i64], types::I32);
    let scratch_ne = scratch_shape(d as i64, n_head as i64, n_head_kv as i64, n_tok, ns);
    let scratch = input(&scratch_ne, types::F32);
    let scale = 1.0 / (d as f32).sqrt();
    let full = attention_with(&ctx, q, kf, vf, tf, past, scratch, scale, window, 0).as_output();
    let ringed = attention_with(
        &ctx,
        q,
        kr,
        vr,
        tr,
        past,
        scratch,
        scale,
        window,
        ring.blocks,
    )
    .as_output();
    let mut graph = ctx.new_graph(64);
    graph.expand(full);
    graph.expand(ringed);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let qv: Vec<f32> = noise(q.elements(), 5).iter().map(|x| 2.0 * x).collect();
    q.set_f32(&qv);
    for (t, data) in [
        (kf, &mut k_full),
        (vf, &mut v_full),
        (kr, &mut k_ring),
        (vr, &mut v_ring),
    ] {
        t.set_bytes(&f16_bytes(data));
        let mut h = vec![0u16; data.len()];
        // SAFETY: equal lengths.
        unsafe {
            ffi::ggml_fp32_to_fp16_row(data.as_ptr(), h.as_mut_ptr(), data.len() as i64);
            ffi::ggml_fp16_to_fp32_row(h.as_ptr(), data.as_mut_ptr(), data.len() as i64);
        }
    }
    let tfull: Vec<i32> = tables.iter().flatten().map(|&b| b as i32).collect();
    tf.set_i32(&tfull);
    let tring: Vec<i32> = (0..ns)
        .flat_map(|s| (0..ring.blocks).map(move |i| ring.block(s, i) as i32))
        .collect();
    tr.set_i32(&tring);
    past.set_i32(&n_past.iter().map(|&p| p as i32).collect::<Vec<_>>());
    backend.compute(&graph).unwrap();

    let label = format!(
        "d {d}, heads {n_head}/{n_head_kv}, window {window}, ring {} blocks, n_past {n_past:?}, \
         {n_tok} tokens",
        ring.blocks
    );
    let (a, b) = (ringed.to_f32(), full.to_f32());
    assert!(
        a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
        "{label}: ring and full cache differ"
    );
    let sh = Shape {
        d,
        n_head,
        n_head_kv,
        n_tok,
        scale: scale as f64,
        window,
    };
    let want = oracle(&sh, &qv, &k_full, &v_full, n_past, &tables);
    let diff = a
        .iter()
        .zip(&want)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(diff < tol, "{label}: max |ring - oracle| {diff}");
}

/// Chunk starts on and next to block boundaries and ring wraps (a ring of 4 blocks here: a
/// 100-position window and 37-token steps), decode at the seams, several sequences per batch.
pub(super) fn seams(backend: &mut Backend, tol: f32) {
    let cases: [(&[usize], usize); 5] = [
        (&[0, 63, 64], 37),
        (&[190, 255, 256], 37),
        (&[219, 300, 511], 37),
        (&[255, 256, 319, 320, 1000], 1),
        (&[611], 5),
    ];
    for (d, n_head, n_head_kv) in [(128, 8, 2), (256, 16, 8)] {
        for (n_past, n_tok) in cases {
            ring_case(backend, d, n_head, n_head_kv, 100, 37, n_past, n_tok, tol);
        }
    }
    // The window a whole number of blocks, the step one token (decode only).
    let decode: &[usize] = &[127, 128, 191, 192, 700];
    ring_case(backend, 128, 8, 2, 128, 1, decode, 1, tol);
}

#[test]
fn ring_matches_full_cache_and_oracle() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    seams(&mut backend, 1e-4);
}

/// The CUDA kernels (decode and prefill) over rings, and Gemma 4's shapes with its window.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_ring_matches_full_cache_and_oracle() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut backend = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    // The prefill kernels' F16 weights against the running max: see tests.rs.
    seams(&mut backend, 5e-4);
    for n_past in [[1024, 1536, 2047], [2048, 2560, 3071]] {
        ring_case(&mut backend, 256, 16, 8, 1024, 512, &n_past, 512, 5e-4);
    }
    ring_case(
        &mut backend,
        256,
        16,
        8,
        1024,
        512,
        &[1023, 1024, 2600, 5000],
        1,
        5e-4,
    );
}
