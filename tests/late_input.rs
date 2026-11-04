//! A graph input written while the graph runs (`Backend::compute_kept_with`): on CUDA the fill
//! runs between the graph's segments, just before the input's first reader; elsewhere before
//! the graph. Either way the graph reads the filled values, and the fill runs once.
//!
//! The CUDA case: `cargo test --release --features cuda --test late_input -- --ignored`

use std::os::raw::c_void;

use legion::ggml::{
    Backend, ComputeAllocator, Context, ExternalOp, KeptGraph, KeptParts, LateInput, ffi, types,
};

unsafe fn cpu_noop(_: &ffi::ggml_tensor, _: usize, _: usize) {}

unsafe fn cuda_noop(_: &ffi::ggml_tensor, _: *mut c_void) -> Result<(), String> {
    Ok(())
}

/// Does nothing: on CUDA it splits the graph into segments around it.
static SPLIT: ExternalOp = ExternalOp {
    name: "test_split",
    cpu: cpu_noop,
    cuda: Some(cuda_noop),
};

/// `y = 3a + 2x` with an external op between `3a` and `x`'s reader, `x` filled late; checked
/// over two runs of the kept graph with different `x`.
fn late_input(backend: &mut Backend) {
    let ctx = Context::new(16, 16);
    let a = ctx.new_tensor(types::F32, &[4]).named("a").as_input();
    let x = ctx.new_tensor(types::F32, &[4]).named("x").as_input();
    let b = ctx.scale(a, 3.0);
    let split = ctx.external(&SPLIT, types::F32, [1, 1, 1, 1], &[b]);
    let y = ctx.add(b, ctx.scale(x, 2.0)).as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(split);
    graph.expand(y);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let parts = KeptParts::new(&graph, &[a, x, y]);
    // SAFETY: the graph reads only tensors of `ctx`.
    let kept = unsafe { KeptGraph::new(ctx, parts) };
    for (round, sync) in [(0.0f32, true), (10.0, false)] {
        kept.tensor(0).set_f32(&[1.0; 4]);
        let xs: Vec<f32> = (1..=4).map(|i| i as f32 + round).collect();
        let mut fills = 0;
        let mut fill = || {
            fills += 1;
            kept.tensor(1).set_f32(&xs);
        };
        let late = LateInput {
            tensor: kept.tensor(1),
            fill: &mut fill,
        };
        backend.compute_kept_with(&kept, sync, Some(late)).unwrap();
        backend.synchronize();
        let want: Vec<f32> = xs.iter().map(|x| 3.0 + 2.0 * x).collect();
        assert_eq!(kept.tensor(2).to_f32(), want, "sync {sync}");
        assert_eq!(fills, 1, "sync {sync}");
    }
}

#[test]
fn cpu_fills_before_the_graph() {
    late_input(&mut Backend::cpu().unwrap());
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a CUDA GPU"]
fn cuda_fills_between_segments() {
    late_input(&mut legion::ggml::backend_by_name("CUDA0").unwrap());
}
