//! The safe ggml layer on the CPU backend (no GPU needed).

use legion::ggml::{
    self, Backend, ComputeAllocator, Context, DeviceKind, GgmlError, devices, types,
};

/// Deterministic pseudo-random values in [-1, 1).
fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2654435761).max(1);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// out[n][m] = sum_k w[m][k] * x[n][k]  (ggml's mul_mat convention)
fn reference_matmul(w: &[f32], x: &[f32], k: usize, m: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for j in 0..n {
        for i in 0..m {
            out[j * m + i] = (0..k).map(|p| w[i * k + p] * x[j * k + p]).sum();
        }
    }
    out
}

fn assert_close(got: &[f32], want: &[f32], rel: f32) {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= rel * scale,
            "element {i}: got {g}, want {w} (scale {scale})"
        );
    }
}

#[test]
fn cpu_device_is_listed_and_starts() {
    let devs = devices();
    assert!(devs.iter().any(|d| d.kind() == DeviceKind::Cpu), "{devs:?}");
    let backend = Backend::cpu().unwrap();
    assert_eq!(backend.device().kind(), DeviceKind::Cpu);
    assert!(backend.buffer_type().is_host());
    assert!(!backend.name().is_empty());
}

/// Runs `weights (k×m, type ty) @ x (k×n, f32)` on the CPU backend.
fn matmul_on_cpu(ty: u32, w_bytes: &[u8], x: &[f32], k: usize, m: usize, n: usize) -> Vec<f32> {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);

    let weights = Context::new(1, 0);
    let w = weights.new_tensor(ty, &[k as i64, m as i64]).named("w");
    assert!(weights.alloc_tensors(backend.buffer_type()).unwrap() >= w_bytes.len());
    w.set_bytes(w_bytes);

    let ctx = Context::new(8, 16);
    let input = ctx.new_tensor(types::F32, &[k as i64, n as i64]).named("x");
    ctx.alloc_tensors(backend.buffer_type()).unwrap();
    input.set_f32(x);
    let out = ctx.mul_mat(w, input).named("out");
    let mut graph = ctx.new_graph(16);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    backend.compute(&graph).unwrap();
    assert_eq!(out.shape(), vec![m as i64, n as i64]);
    out.to_f32()
}

#[test]
fn f32_matmul_matches_reference() {
    let (k, m, n) = (64, 32, 5);
    let w = values(k * m, 1);
    let x = values(k * n, 2);
    let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let got = matmul_on_cpu(types::F32, &bytes, &x, k, m, n);
    assert_close(&got, &reference_matmul(&w, &x, k, m, n), 1e-5);
}

#[test]
fn quantized_matmul_matches_dequantized_reference() {
    // k = 256: one q4_K super-block per row.
    let (k, m, n) = (256, 16, 3);
    let w = values(k * m, 3);
    let x = values(k * n, 4);
    for ty in [types::Q8_0, types::Q4_K, types::Q6_K] {
        let q = ggml::quantize(ty, &w, k);
        assert_eq!(q.len(), ggml::row_size(ty, k) * m);
        // Reference uses the dequantized weights, so only the activation rounding
        // inside ggml's quantized dot product (q8 activations) is left as error.
        let wd = ggml::dequantize(ty, &q, k * m);
        let got = matmul_on_cpu(ty, &q, &x, k, m, n);
        assert_close(&got, &reference_matmul(&wd, &x, k, m, n), 2e-2);
    }
}

#[test]
fn multi_op_graph() {
    // y = silu(rms_norm(a) * g) + b, row-wise over k
    let (k, rows) = (32, 4);
    let a = values(k * rows, 5);
    let g = values(k, 6);
    let b = values(k * rows, 7);
    let eps = 1e-6;

    let mut backend = Backend::cpu().unwrap();
    let ctx = Context::new(16, 32);
    let ta = ctx.new_tensor(types::F32, &[k as i64, rows as i64]);
    let tg = ctx.new_tensor(types::F32, &[k as i64]);
    let tb = ctx.new_tensor(types::F32, &[k as i64, rows as i64]);
    ctx.alloc_tensors(backend.buffer_type()).unwrap();
    ta.set_f32(&a);
    tg.set_f32(&g);
    tb.set_f32(&b);
    let y = ctx.add(ctx.silu(ctx.mul(ctx.rms_norm(ta, eps), tg)), tb);
    let mut graph = ctx.new_graph(32);
    graph.expand(y);
    assert!(graph.nodes() >= 4);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    backend.compute(&graph).unwrap();

    let mut want = Vec::with_capacity(k * rows);
    for r in 0..rows {
        let row = &a[r * k..(r + 1) * k];
        let rms = (row.iter().map(|v| v * v).sum::<f32>() / k as f32 + eps).sqrt();
        for i in 0..k {
            let v = row[i] / rms * g[i];
            want.push(v / (1.0 + (-v).exp()) + b[r * k + i]);
        }
    }
    assert_close(&y.to_f32(), &want, 1e-5);
}

#[test]
fn embedding_lookup_with_get_rows() {
    let (dim, vocab) = (8, 10);
    let table = values(dim * vocab, 8);
    let mut backend = Backend::cpu().unwrap();
    let ctx = Context::new(8, 8);
    let emb = ctx.new_tensor(types::F32, &[dim as i64, vocab as i64]);
    let ids = ctx.new_tensor(types::I32, &[3]);
    ctx.alloc_tensors(backend.buffer_type()).unwrap();
    emb.set_f32(&table);
    ids.set_i32(&[7, 0, 7]);
    let rows = ctx.get_rows(emb, ids);
    let mut graph = ctx.new_graph(8);
    graph.expand(rows);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    backend.compute(&graph).unwrap();
    let got = rows.to_f32();
    assert_eq!(&got[..dim], &table[7 * dim..8 * dim]);
    assert_eq!(&got[dim..2 * dim], &table[..dim]);
    assert_eq!(&got[2 * dim..], &table[7 * dim..8 * dim]);
}

#[test]
fn allocator_is_reused_across_steps_and_guards_misuse() {
    let mut backend = Backend::cpu().unwrap();
    let weights = Context::new(1, 0);
    let w = weights.new_tensor(types::F32, &[16, 16]);
    weights.alloc_tensors(backend.buffer_type()).unwrap();
    w.set_f32(&values(256, 9));

    let mut allocator = ComputeAllocator::new(backend.buffer_type());
    for step in 0..3 {
        let ctx = Context::new(8, 8);
        let x = ctx.new_tensor(types::F32, &[16, 1 + step]);
        ctx.alloc_tensors(backend.buffer_type()).unwrap();
        x.set_f32(&values(16 * (1 + step as usize), 10 + step as u32));
        let y = ctx.mul_mat(w, x);
        let mut graph = ctx.new_graph(8);
        graph.expand(y);
        ctx.alloc_graph(&graph, allocator).unwrap();
        backend.compute(&graph).unwrap();
        assert_eq!(y.elements(), 16 * (1 + step as usize));

        // A second allocation in the same context is refused.
        let again = ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()));
        assert!(matches!(again, Err(GgmlError::GraphAlreadyAllocated)));

        allocator = ctx.into_allocator().unwrap();
        assert!(allocator.buffer_size() > 0);
    }

    // A graph from another context is refused.
    let a = Context::new(4, 4);
    let b = Context::new(4, 4);
    let t = a.new_tensor(types::F32, &[4]);
    let mut graph = a.new_graph(4);
    graph.expand(a.scale(t, 2.0));
    assert!(matches!(
        b.alloc_graph(&graph, allocator),
        Err(GgmlError::ForeignGraph)
    ));
}

#[test]
#[should_panic(expected = "has no memory yet")]
fn reading_an_unallocated_tensor_panics() {
    let ctx = Context::new(1, 0);
    ctx.new_tensor(types::F32, &[4]).to_f32();
}

/// Values written and read through a host tensor (`set_f32_via` / `to_f32_via`) are the plain
/// ones, the host tensor bigger than they need; on `buft` with a host tensor in `host`.
fn round_trip_via(buft: ggml::BufferType, host: ggml::BufferType) {
    let ctx = Context::new(1, 0);
    let t = ctx.new_tensor(types::F32, &[1000]);
    ctx.alloc_tensors(buft).unwrap();
    let via_ctx = Context::new(1, 0);
    let via = via_ctx.new_tensor(types::F32, &[1300]);
    via_ctx.alloc_tensors(host).unwrap();
    let v = values(1000, 7);
    t.set_f32_via(&v, via);
    assert_eq!(t.to_f32(), v);
    t.set_f32(&values(1000, 8));
    assert_eq!(t.to_f32_via(via), values(1000, 8));
}

#[test]
fn values_round_trip_through_a_host_tensor() {
    let cpu = Backend::cpu().unwrap().buffer_type();
    round_trip_via(cpu, cpu);
}

/// On CUDA: through the device's pinned host memory.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_values_round_trip_through_pinned_memory() {
    let dev = ggml::device_by_kind(DeviceKind::Gpu).expect("no GPU");
    round_trip_via(
        dev.buffer_type(),
        dev.host_buffer_type().expect("pinned host memory"),
    );
}

/// A padded compute allocator (long prompts' graphs, each a little bigger than the last) keeps
/// its buffer for a slightly bigger next graph, where a plain one grows again; the results are
/// the same either way.
#[test]
fn padded_allocator_absorbs_a_slightly_bigger_graph() {
    let mut backend = Backend::cpu().unwrap();
    let buft = backend.buffer_type();
    // out = mul_mat(w, x) over n columns: the intermediate grows with n.
    let mut run = |alloc: ComputeAllocator, n: usize| -> (ComputeAllocator, Vec<f32>, usize) {
        let k = 64;
        let weights = Context::new(1, 0);
        let w = weights.new_tensor(types::F32, &[k as i64, 32]);
        weights.alloc_tensors(buft).unwrap();
        w.set_f32(&values(k * 32, 1));
        let ctx = Context::new(8, 16);
        let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]);
        ctx.alloc_tensors(buft).unwrap();
        x.set_f32(&values(k * n, 2));
        let out = ctx.scale(ctx.mul_mat(w, x), 2.0);
        let mut graph = ctx.new_graph(16);
        graph.expand(out);
        ctx.alloc_graph(&graph, alloc).unwrap();
        backend.compute(&graph).unwrap();
        let got = out.to_f32();
        let size = ctx.allocator_size();
        (ctx.into_allocator().unwrap(), got, size)
    };
    let (alloc, small, size_small) = run(ComputeAllocator::padded(buft), 100);
    let (_, big, size_big) = run(alloc, 101);
    assert_eq!(
        size_small, size_big,
        "the padded buffer held the next graph"
    );
    let (alloc, small_plain, size_plain) = run(ComputeAllocator::new(buft), 100);
    assert!(
        size_small > size_plain,
        "padded {size_small} plain {size_plain}"
    );
    let (_, big_plain, size_plain_big) = run(alloc, 101);
    assert!(size_plain_big > size_plain, "a plain buffer grows");
    assert_eq!(small, small_plain);
    assert_eq!(big, big_plain);
}

/// A reserved compute allocator (#491: the spare kept through an expert-cache lend) has its
/// buffer before any graph: a graph that fits runs in it without growing it, with the same
/// results as from a fresh allocator; one that needs more grows it.
fn reserved_allocator_holds_a_graph_that_fits(buft: ggml::BufferType, mut backend: Backend) {
    let k = 64;
    let weights = Context::new(1, 0);
    let w = weights.new_tensor(types::F32, &[k as i64, 32]);
    weights.alloc_tensors(buft).unwrap();
    w.set_f32(&values(k * 32, 1));
    let mut run = |alloc: ComputeAllocator, n: usize| -> (ComputeAllocator, Vec<f32>, usize) {
        let ctx = Context::new(8, 16);
        let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]);
        ctx.alloc_tensors(buft).unwrap();
        x.set_f32(&values(k * n, 2));
        let out = ctx.scale(ctx.mul_mat(w, x), 2.0);
        let mut graph = ctx.new_graph(16);
        graph.expand(out);
        ctx.alloc_graph(&graph, alloc).unwrap();
        backend.compute(&graph).unwrap();
        let got = out.to_f32();
        let size = ctx.allocator_size();
        (ctx.into_allocator().unwrap(), got, size)
    };
    let bytes = 1 << 20;
    let spare = ComputeAllocator::reserved(buft, bytes).unwrap();
    assert!(spare.buffer_size() >= bytes, "{}", spare.buffer_size());
    let reserved = spare.buffer_size();
    let (spare, got, size) = run(spare, 100);
    assert_eq!(size, reserved, "a graph that fits grew the buffer");
    let (_, want, plain) = run(ComputeAllocator::new(buft), 100);
    assert!(plain < reserved, "the test graph needs {plain}");
    assert_eq!(got, want);
    // 32 x 20000 floats out (2.5 MB): more than the buffer.
    let (_, _, grown) = run(spare, 20000);
    assert!(grown > reserved, "{grown} <= {reserved}");
}

#[test]
fn reserved_allocator_holds_a_graph_that_fits_on_cpu() {
    let backend = Backend::cpu().unwrap();
    reserved_allocator_holds_a_graph_that_fits(backend.buffer_type(), backend);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_reserved_allocator_holds_a_graph_that_fits() {
    let dev = ggml::device_by_kind(DeviceKind::Gpu).expect("no GPU");
    let backend = dev.init().unwrap();
    reserved_allocator_holds_a_graph_that_fits(backend.buffer_type(), backend);
}
