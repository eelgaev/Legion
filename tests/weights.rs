use legion::ggml::{self, Backend, ComputeAllocator, Context, types};
use legion::gguf::{GgufFile, GgufWriter, Value};
use legion::load_io::Options;
use legion::weights::Weights;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

fn write_model(dir: &std::path::Path) -> (std::path::PathBuf, Vec<f32>, Vec<u8>) {
    let path = dir.join("model.gguf");
    let emb: Vec<f32> = (0..32).map(|i| i as f32 * 0.5).collect();
    let w: Vec<f32> = (0..64 * 4).map(|i| ((i % 7) as f32 - 3.0) / 3.0).collect();
    let wq = ggml::quantize(types::Q8_0, &w, 64);
    let mut g = GgufWriter::new();
    g.set("general.architecture", &Value::String("test".into()))
        .unwrap();
    g.add_tensor("token_embd.weight", types::F32, &[8, 4], &f32_bytes(&emb))
        .unwrap();
    g.add_tensor("blk.0.ffn_up.weight", types::Q8_0, &[64, 4], &wq)
        .unwrap();
    g.write(&path).unwrap();
    (path, emb, wq)
}

#[test]
fn loads_every_tensor_with_its_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let (path, emb, wq) = write_model(dir.path());
    let gguf = GgufFile::open(&path).unwrap();
    let backend = Backend::cpu().unwrap();
    let weights = Weights::load(&gguf, |_| backend.buffer_type()).unwrap();

    assert_eq!(weights.len(), 2);
    let e = weights.get("token_embd.weight").unwrap();
    assert_eq!(e.shape(), vec![8, 4]);
    assert_eq!(e.to_f32(), emb);
    let up = weights.get("blk.0.ffn_up.weight").unwrap();
    assert_eq!(up.ty(), types::Q8_0);
    assert_eq!(up.bytes(), wq);
    assert!(weights.get("missing").is_none());

    let placement = weights.placement();
    assert_eq!(placement.len(), 1);
    assert_eq!(placement[0].1, (emb.len() * 4 + wq.len()) as u64);
}

#[test]
fn loaded_weights_feed_a_graph() {
    let dir = tempfile::tempdir().unwrap();
    let (path, _, wq) = write_model(dir.path());
    let gguf = GgufFile::open(&path).unwrap();
    let mut backend = Backend::cpu().unwrap();
    let weights = Weights::load(&gguf, |_| backend.buffer_type()).unwrap();

    let ctx = Context::new(4, 8);
    let x = ctx.new_tensor(types::F32, &[64, 1]);
    ctx.alloc_tensors(backend.buffer_type()).unwrap();
    let xs: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
    x.set_f32(&xs);
    let y = ctx.mul_mat(weights.get("blk.0.ffn_up.weight").unwrap(), x);
    let mut graph = ctx.new_graph(8);
    graph.expand(y);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    backend.compute(&graph).unwrap();

    let wd = ggml::dequantize(types::Q8_0, &wq, 64 * 4);
    for (row, got) in y.to_f32().iter().enumerate() {
        let want: f32 = (0..64).map(|k| wd[row * 64 + k] * xs[k]).sum();
        assert!((got - want).abs() < 0.05, "row {row}: {got} vs {want}");
    }
}

#[test]
fn two_hundred_tensors_load_identically_with_one_and_four_threads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("many.gguf");
    let mut expected = Vec::new();
    let mut g = GgufWriter::new();
    g.set("general.architecture", &Value::String("test".into()))
        .unwrap();
    for i in 0..200usize {
        // Varying sizes, including one big enough to be split into chunks.
        let n = if i == 7 { 300_000 } else { 16 + (i * 37) % 900 };
        let vals: Vec<f32> = (0..n).map(|k| (i * 1000 + k) as f32 * 0.25).collect();
        let name = format!("blk.{i}.w");
        g.add_tensor(&name, types::F32, &[n as i64], &f32_bytes(&vals))
            .unwrap();
        expected.push((name, vals));
    }
    g.write(&path).unwrap();

    let gguf = GgufFile::open(&path).unwrap();
    let backend = Backend::cpu().unwrap();
    for threads in [1, 4] {
        let weights = Weights::load_with(
            &gguf,
            |_| backend.buffer_type(),
            &Options::with_threads(threads),
        )
        .unwrap();
        assert_eq!(weights.len(), 200);
        for (name, vals) in &expected {
            assert_eq!(
                &weights.get(name).unwrap().to_f32(),
                vals,
                "{name} @ {threads}"
            );
        }
    }
}

/// Loads a real model into host memory: `LEGION_TEST_GGUF=/path/model.gguf cargo test --release
/// --test weights -- --ignored --nocapture`.
#[test]
#[ignore = "needs LEGION_TEST_GGUF pointing at a real model"]
fn loads_a_real_model() {
    let path = std::env::var("LEGION_TEST_GGUF").expect("set LEGION_TEST_GGUF");
    let gguf = GgufFile::open(&path).unwrap();
    let backend = Backend::cpu().unwrap();
    let rss0 = rss_kib();
    let t0 = std::time::Instant::now();
    let weights = Weights::load(&gguf, |_| backend.buffer_type()).unwrap();
    println!(
        "RSS +{} MiB after load, {:?}",
        (rss_kib() - rss0) / 1024,
        weights.stats()
    );
    let secs = t0.elapsed().as_secs_f64();
    let bytes: u64 = weights.placement().iter().map(|p| p.1).sum();
    println!(
        "{} tensors, {:.2} GiB in {secs:.1} s ({:.2} GiB/s)",
        weights.len(),
        bytes as f64 / (1u64 << 30) as f64,
        bytes as f64 / (1u64 << 30) as f64 / secs
    );
    assert_eq!(weights.len(), gguf.tensor_count());
}

fn no_mmap() -> Options<'static> {
    let mut o = Options::with_threads(2);
    o.mmap = false;
    o
}

#[test]
fn mmap_maps_cpu_tensors_and_matches_the_copying_loader() {
    let dir = tempfile::tempdir().unwrap();
    let (path, emb, wq) = write_model(dir.path());
    let gguf = GgufFile::open(&path).unwrap();
    let backend = Backend::cpu().unwrap();

    let copied = Weights::load_with(&gguf, |_| backend.buffer_type(), &no_mmap()).unwrap();
    assert_eq!(copied.stats().mapped, 0);
    assert_eq!(copied.stats().copied, 2);

    let mapped =
        Weights::load_with(&gguf, |_| backend.buffer_type(), &Options::with_threads(2)).unwrap();
    assert_eq!(mapped.stats().mapped, 2);
    assert_eq!(mapped.stats().mmap_fallback, 0);
    assert_eq!(mapped.get("token_embd.weight").unwrap().to_f32(), emb);
    assert_eq!(mapped.get("blk.0.ffn_up.weight").unwrap().bytes(), wq);
    for name in ["token_embd.weight", "blk.0.ffn_up.weight"] {
        assert_eq!(
            mapped.get(name).unwrap().bytes(),
            copied.get(name).unwrap().bytes()
        );
    }
}

/// Loads, computes, drops (unmapping the file), and does it again: the second round must see
/// intact data and nothing may touch freed memory.
#[test]
fn drop_unmaps_and_reload_works() {
    let dir = tempfile::tempdir().unwrap();
    let (path, _, wq) = write_model(dir.path());
    let gguf = GgufFile::open(&path).unwrap();
    let mut backend = Backend::cpu().unwrap();
    let xs: Vec<f32> = (0..64).map(|i| (i as f32).cos()).collect();
    let mut first: Option<Vec<f32>> = None;
    for _ in 0..3 {
        let weights = Weights::load(&gguf, |_| backend.buffer_type()).unwrap();
        assert_eq!(weights.stats().mapped, 2);
        let ctx = Context::new(4, 8);
        let x = ctx.new_tensor(types::F32, &[64, 1]);
        ctx.alloc_tensors(backend.buffer_type()).unwrap();
        x.set_f32(&xs);
        let y = ctx.mul_mat(weights.get("blk.0.ffn_up.weight").unwrap(), x);
        let mut graph = ctx.new_graph(8);
        graph.expand(y);
        ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
            .unwrap();
        backend.compute(&graph).unwrap();
        let out = y.to_f32();
        assert_eq!(weights.get("blk.0.ffn_up.weight").unwrap().bytes(), wq);
        match &first {
            None => first = Some(out),
            Some(f) => assert_eq!(&out, f),
        }
        drop(ctx);
        drop(weights);
    }
}

#[test]
fn misaligned_or_out_of_range_tensors_are_not_mapped() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("raw.bin");
    let vals: Vec<f32> = (0..64).map(|i| i as f32).collect();
    std::fs::write(&path, f32_bytes(&vals)).unwrap();
    let file = std::fs::File::open(&path).unwrap();

    let ctx = Context::new(4, 0);
    ctx.new_tensor(types::F32, &[4]).named("aligned");
    ctx.new_tensor(types::F32, &[4]).named("misaligned");
    ctx.new_tensor(types::F32, &[4]).named("past_end");
    let done = ctx
        .map_tensors_from_file(
            &file,
            &[("aligned", 32), ("misaligned", 16), ("past_end", 4096)],
        )
        .unwrap();
    assert_eq!(done, vec![0]);
    assert_eq!(ctx.tensor("aligned").unwrap().to_f32(), vals[8..12]);
    // The leftovers are still unallocated; the normal allocator takes them.
    let backend = Backend::cpu().unwrap();
    ctx.alloc_tensors(backend.buffer_type()).unwrap();
    ctx.tensor("misaligned").unwrap().set_f32(&[1.0; 4]);
    // Writes to a mapped tensor are private: the file is unchanged.
    ctx.tensor("aligned").unwrap().set_f32(&[9.0; 4]);
    assert_eq!(std::fs::read(&path).unwrap(), f32_bytes(&vals));
}

fn rss_kib() -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap();
    let l = s.lines().find(|l| l.starts_with("VmRSS:")).unwrap();
    l.split_whitespace().nth(1).unwrap().parse().unwrap()
}

/// Warm load time and resident memory with and without mmap, and identical logits:
/// `LEGION_TEST_GGUF=.../Qwen3.5-4B-Q8_0.gguf cargo test --release --test weights --
/// --ignored --nocapture mmap_real_model`.
#[test]
#[ignore = "needs LEGION_TEST_GGUF pointing at a real qwen35 model"]
fn mmap_real_model_logits_and_numbers() {
    use legion::model::core::Model;
    use legion::tokenizer::Tokenizer;
    let path = std::env::var("LEGION_TEST_GGUF").expect("set LEGION_TEST_GGUF");
    let gguf = GgufFile::open(&path).unwrap();
    let ids = Tokenizer::from_gguf(&gguf)
        .unwrap()
        .encode("The capital of France is", true);
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(8);
    let mut logits = Vec::new();
    for mmap in [false, true] {
        let mut opts = Options::with_threads(8);
        opts.mmap = mmap;
        let rss0 = rss_kib();
        let t0 = std::time::Instant::now();
        let model = Model::load_with(&gguf, |_| backend.buffer_type(), &opts).unwrap();
        let secs = t0.elapsed().as_secs_f64();
        let rss_load = rss_kib();
        let l = model.prompt_logits(&mut backend, &ids).unwrap();
        println!(
            "mmap={mmap}: load {secs:.3} s, RSS +{} MiB after load, +{} MiB after a forward pass, {:?}",
            (rss_load - rss0) / 1024,
            (rss_kib() - rss0) / 1024,
            model.weights().stats()
        );
        if mmap {
            assert_eq!(model.weights().stats().mmap_fallback, 0);
        }
        logits.push(l);
    }
    assert_eq!(logits[0], logits[1], "logits differ with mmap");
}

/// A file holding `exps` (Q5_1 `[640, 64, 4]`: rows of 640, not a multiple of ggml-cuda's
/// 512-element row padding) followed by `tail` bytes of 0xFF, which read as F16 scales are NaN.
fn write_experts_then_ff(dir: &std::path::Path, tail: usize) -> (std::path::PathBuf, Vec<u8>) {
    let w: Vec<f32> = (0..640 * 64 * 4)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
        .collect();
    let q = ggml::quantize(types::Q5_1, &w, 640);
    let path = dir.join("exps.bin");
    let mut bytes = q.clone();
    bytes.extend(std::iter::repeat_n(0xFFu8, tail));
    std::fs::write(&path, &bytes).unwrap();
    (path, q)
}

/// A mapped quantized tensor gets its CUDA row padding zeroed in the private mapping, and only
/// there: the file keeps its bytes.
#[test]
fn mapping_zeroes_the_cuda_row_padding_privately() {
    let dir = tempfile::tempdir().unwrap();
    let (path, q) = write_experts_then_ff(dir.path(), 4096);
    let file = std::fs::File::open(&path).unwrap();
    let pad = ggml::mapped_tail_padding(types::Q5_1, 640);
    // Up to the next multiple of 512 elements: 384 of Q5_1, 24 bytes per 32.
    assert_eq!(pad, 288);
    assert_eq!(ggml::mapped_tail_padding(types::Q5_1, 2560), 0);
    assert_eq!(ggml::mapped_tail_padding(types::F32, 640), 0);

    let ctx = Context::new(2, 0);
    let t = ctx.new_tensor(types::Q5_1, &[640, 64, 4]).named("exps");
    assert_eq!(
        ctx.map_tensors_from_file(&file, &[("exps", 0)]).unwrap(),
        vec![0]
    );
    let data = t.host_bytes().unwrap();
    assert_eq!(data, &q[..]);
    // SAFETY: the mapping covers the whole file, which extends 4096 bytes past the tensor.
    let after = unsafe { std::slice::from_raw_parts(data.as_ptr().add(q.len()), pad + 1) };
    assert!(after[..pad].iter().all(|&b| b == 0), "padding not zeroed");
    assert_eq!(after[pad], 0xFF, "zeroed past the padding");
    assert_eq!(std::fs::read(&path).unwrap()[q.len()], 0xFF);
}

/// Padding past the mapping's last page (here the tensor ends the file on a page boundary) is
/// not there to read: the tensor is left for the reader. Padding that overlaps another mapped
/// tensor is that tensor's data and stays as it is.
#[test]
fn row_padding_past_the_file_or_inside_a_neighbour() {
    let dir = tempfile::tempdir().unwrap();
    let (path, q) = write_experts_then_ff(dir.path(), 0);
    assert!(q.len().is_multiple_of(4096));
    let file = std::fs::File::open(&path).unwrap();
    let ctx = Context::new(2, 0);
    ctx.new_tensor(types::Q5_1, &[640, 64, 4]).named("exps");
    assert!(
        ctx.map_tensors_from_file(&file, &[("exps", 0)])
            .unwrap()
            .is_empty()
    );

    let (path, q) = write_experts_then_ff(dir.path(), 4096);
    let file = std::fs::File::open(&path).unwrap();
    let ctx = Context::new(2, 0);
    ctx.new_tensor(types::Q5_1, &[640, 64, 4]).named("exps");
    let next = ctx.new_tensor(types::I32, &[64]).named("next");
    let done = ctx
        .map_tensors_from_file(&file, &[("exps", 0), ("next", q.len() as u64)])
        .unwrap();
    assert_eq!(done, vec![0, 1]);
    assert!(next.host_bytes().unwrap().iter().all(|&b| b == 0xFF));
}

/// The case the padding is for: a MUL_MAT_ID on CUDA over experts mapped from a file (as
/// `--host-experts` runs them), with more tokens than ggml's one-pass kernels take (8), so the
/// tiled kernel (MMQ) runs and reads the last expert's last row up to the row padding. With the
/// file's next bytes there (here 0xFF, NaN as F16 scales) every output of a token routed to the
/// last expert was NaN; zeroed, the result matches the CPU's.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_mul_mat_id_over_mapped_experts() {
    use legion::ggml::{DeviceKind, device_by_kind};
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = write_experts_then_ff(dir.path(), 4096);
    let file = std::fs::File::open(&path).unwrap();
    let weights = Context::new(1, 0);
    let exps = weights.new_tensor(types::Q5_1, &[640, 64, 4]).named("exps");
    assert_eq!(
        weights
            .map_tensors_from_file(&file, &[("exps", 0)])
            .unwrap(),
        vec![0]
    );

    let n_tok: i32 = 9;
    let xs: Vec<f32> = (0..640 * n_tok)
        .map(|i| ((i % 13) as f32 - 6.0) / 6.0)
        .collect();
    // Token t uses experts 3 (the last) and t % 3.
    let ids: Vec<i32> = (0..n_tok).flat_map(|t| [3, t % 3]).collect();
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU")
        .init()
        .unwrap();
    let mut cpu = Backend::cpu().unwrap();
    let mut outs = Vec::new();
    for backend in [&mut gpu, &mut cpu] {
        let ctx = Context::new(8, 8);
        let x = ctx
            .new_tensor(types::F32, &[640, 1, n_tok as i64])
            .as_input();
        let id = ctx.new_tensor(types::I32, &[2, n_tok as i64]).as_input();
        let y = ctx.mul_mat_id(exps, x, id).as_output();
        let mut graph = ctx.new_graph(8);
        graph.expand(y);
        ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
            .unwrap();
        x.set_f32(&xs);
        id.set_i32(&ids);
        backend.compute(&graph).unwrap();
        outs.push(y.to_f32());
    }
    let (on_gpu, on_cpu) = (&outs[0], &outs[1]);
    assert!(
        on_gpu.iter().all(|v| v.is_finite()),
        "non-finite CUDA output"
    );
    let scale = on_cpu.iter().fold(0f32, |m, v| m.max(v.abs()));
    let diff = on_gpu
        .iter()
        .zip(on_cpu)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    println!("max |diff| {diff:.2e} of max |y| {scale:.2e}");
    assert!(diff <= 2e-2 * scale, "CUDA vs CPU: max |diff| {diff}");
}
