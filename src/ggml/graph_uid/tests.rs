use super::*;
use crate::ggml::{
    Backend, ComputeAllocator, Context, DeviceKind, KeptGraph, KeptParts, SharedAllocator,
    device_by_kind, meta, types,
};

/// The mirror is ggml's struct: the fields read through it are what ggml's getters say, and the
/// node array ggml lays out right behind the struct starts at the mirror's size, so `uid`, the
/// last field, is where ggml keeps it.
#[test]
fn mirror_matches_ggml_cgraph() {
    let ctx = Context::new(8, 32);
    let x = ctx.new_tensor(types::F32, &[4]);
    let mut graph = ctx.new_graph(32);
    graph.expand(ctx.scale(ctx.add(x, x), 2.0));
    let g = graph.raw();
    // SAFETY: a live graph ggml made; reads only.
    unsafe {
        let m = &*(g as *const CGraph);
        assert_eq!(m.size, ffi::ggml_graph_size(g));
        assert_eq!(m.n_nodes, ffi::ggml_graph_n_nodes(g));
        assert_eq!(m.n_nodes, 2);
        assert_eq!(m.nodes, ffi::ggml_graph_nodes(g));
        assert_eq!(m.nodes as usize, g as usize + std::mem::size_of::<CGraph>());
        // Then the leaves, then the use counts (ggml_new_graph_custom's order).
        assert_eq!(m.leafs, m.nodes.add(32));
        assert_eq!(m.use_counts as usize, m.leafs.add(32) as usize);
        assert!(m.grads.is_null() && m.grad_accs.is_null());
        assert!(m.visited_hash_set.size >= 64);
        assert_eq!(m.order, 0, "left to right");
        assert_eq!(uid(g), 0, "ggml leaves it unset");
        set_uid(g, 42);
        assert_eq!(uid(g), 42);
        // The fields before it are untouched.
        assert_eq!(m.n_nodes, 2);
        assert_eq!(m.order, 0);
    }
}

#[test]
fn ids_are_new_and_apart_from_ggml_s() {
    let (a, b) = (next_uid(), next_uid());
    assert_ne!(a, b);
    assert!(a >= 1 << 63 && b >= 1 << 63);
}

/// `y = 2 x + 1` over 8 floats (x first, then y), kept, its intermediates from `shared`.
fn kept(shared: &SharedAllocator, scale: f32) -> KeptGraph {
    let ctx = Context::new(8, 8);
    let parts = {
        let x = ctx.new_tensor(types::F32, &[8]).as_input();
        let y = ctx.scale_bias(x, scale, 1.0).as_output();
        let mut graph = ctx.new_graph(8);
        graph.expand(y);
        ctx.alloc_graph_shared(&graph, shared).unwrap();
        KeptParts::new(&graph, &[x, y])
    };
    // SAFETY: the graph reads no tensor outside its own context.
    unsafe { KeptGraph::new(ctx, parts) }
}

fn run(backend: &mut Backend, g: &KeptGraph, scale: f32, base: f32) {
    backend.ready_kept(g).unwrap();
    let x: Vec<f32> = (0..8).map(|i| base + i as f32).collect();
    g.tensor(0).set_f32(&x);
    backend.compute_kept(g).unwrap();
    let want: Vec<f32> = x.iter().map(|v| scale * v + 1.0).collect();
    assert_eq!(g.tensor(1).to_f32(), want);
}

/// On a meta backend a kept graph gets an id when first made ready, keeps it over runs of its
/// own, and gets a new one when it is initialized again after another graph ran over its
/// buffer: the meta backend then rebuilds its per-device subgraphs, and the results stay right.
#[test]
fn meta_kept_graph_keeps_its_id_until_initialized_again() {
    let cpu = device_by_kind(DeviceKind::Cpu).unwrap();
    let dev = meta::meta_device(&[cpu, cpu], Box::new(|_, _, _| meta::Split::Mirrored)).unwrap();
    let mut backend = dev.init().unwrap();
    let shared: SharedAllocator = std::sync::Arc::new(std::sync::Mutex::new(
        ComputeAllocator::new(backend.buffer_type()),
    ));
    let a = kept(&shared, 2.0);
    let b = kept(&shared, 3.0);
    // SAFETY: live kept graphs; reads only.
    let id = |g: &KeptGraph| unsafe { uid(g.graph().raw()) };
    assert_eq!(id(&a), 0);
    run(&mut backend, &a, 2.0, 0.0);
    let first = id(&a);
    assert_ne!(first, 0);
    for i in 1..4 {
        run(&mut backend, &a, 2.0, i as f32 * 10.0);
        assert_eq!(id(&a), first, "runs of its own keep the id");
    }
    run(&mut backend, &b, 3.0, 5.0);
    run(&mut backend, &a, 2.0, 7.0);
    assert_ne!(id(&a), first, "initialized again: a new id");
    let second = id(&a);
    run(&mut backend, &a, 2.0, 9.0);
    assert_eq!(id(&a), second);
}

/// Not on a meta backend: graphs stay as ggml made them (single-device paths unchanged).
#[test]
fn other_backends_leave_the_id_unset() {
    let mut backend = Backend::cpu().unwrap();
    let shared: SharedAllocator = std::sync::Arc::new(std::sync::Mutex::new(
        ComputeAllocator::new(backend.buffer_type()),
    ));
    let a = kept(&shared, 2.0);
    run(&mut backend, &a, 2.0, 0.0);
    run(&mut backend, &a, 2.0, 1.0);
    // SAFETY: a live kept graph; reads only.
    assert_eq!(unsafe { uid(a.graph().raw()) }, 0);
}
