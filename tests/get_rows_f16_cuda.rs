//! `Context::get_rows_f16` (the kernel writes F16 rows) against `get_rows` then a cast to F16,
//! bit for bit, on a CUDA device: F16 rows are copied, the quantized KV cache types (Q8_0, Q4_0, Q5_1, IQ4_NL) dequantized. The
//! tensor-parallel cell gather reads the KV cache this way.
//!
//! `cargo test --release --features cuda --test get_rows_f16_cuda -- --ignored`

#![cfg(feature = "cuda")]

use legion::ggml::{Context, DeviceKind, device_by_kind, ffi, quantize, types};

#[test]
#[ignore = "needs a CUDA device"]
fn get_rows_f16_matches_get_rows_and_cast() {
    let gpu = device_by_kind(DeviceKind::Gpu).expect("a GPU");
    let mut backend = gpu.init().unwrap();
    let (row, n_rows, n_ids) = (512usize, 300usize, 777usize);
    let values: Vec<f32> = (0..row * n_rows)
        .map(|i| ((i * 2654435761usize) % 1000) as f32 / 250.0 - 2.0)
        .collect();
    let ids: Vec<i32> = (0..n_ids)
        .map(|i| ((i * 37 + 11) % n_rows) as i32)
        .collect();
    for ty in [
        types::F16,
        types::Q8_0,
        ffi::ggml_type_GGML_TYPE_Q4_0,
        types::Q5_1,
        types::IQ4_NL,
    ] {
        let ctx = Context::new(8, 0);
        let src = ctx.new_tensor(ty, &[row as i64, n_rows as i64]);
        let idx = ctx.new_tensor(types::I32, &[n_ids as i64]);
        ctx.alloc_tensors(backend.buffer_type()).unwrap();
        src.set_bytes(&quantize(ty, &values, row));
        idx.set_i32(&ids);
        let g = Context::new(16, 16);
        let direct = g.get_rows_f16(src, idx);
        let cast = g.cast(g.get_rows(src, idx), types::F16);
        let mut graph = g.new_graph(16);
        graph.expand(direct);
        graph.expand(cast);
        g.alloc_graph(
            &graph,
            legion::ggml::ComputeAllocator::new(backend.buffer_type()),
        )
        .unwrap();
        backend.compute(&graph).unwrap();
        assert_eq!(direct.ne(), cast.ne());
        assert!(direct.bytes() == cast.bytes(), "type {ty}: rows differ");
    }
}
