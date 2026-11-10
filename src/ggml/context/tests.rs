use super::{ComputeAllocator, Context, Graph};
use crate::ggml::{Backend, BufferType, ffi, types};

/// Each node's offset in its compute buffer: the graph's memory layout.
fn layout(graph: &Graph<'_>) -> Vec<usize> {
    let g = graph.raw();
    // SAFETY: an allocated graph; every node has its buffer and data.
    unsafe {
        (0..ffi::ggml_graph_n_nodes(g))
            .map(|i| {
                let t = ffi::ggml_graph_node(g, i);
                (*t).data as usize - ffi::ggml_backend_buffer_get_base((*t).buffer) as usize
            })
            .collect()
    }
}

/// A step-like graph over `n` rows: the same nodes for every `n`, larger with it, so a larger
/// step's plan fits a smaller one.
fn step(ctx: &Context, n: i64) -> Graph<'_> {
    let x = ctx.new_tensor(types::F32, &[256, n]);
    let w = ctx.new_tensor(types::F32, &[256]);
    let a = ctx.scale(x, 2.0);
    let b = ctx.scale(w, 3.0);
    let c = ctx.add(a, b);
    let d = ctx.scale(b, 0.5);
    let e = ctx.mul(c, d);
    let f = ctx.scale(d, 0.25);
    let g = ctx.add(a, f);
    let h = ctx.mul(e, g);
    let mut graph = ctx.new_graph(64);
    graph.expand(ctx.add(h, f));
    graph
}

/// The layout of a 2-row step from `allocator` after it allocated an 8-row step, and the
/// layout of a 2-row step from an allocator that served nothing before.
fn after_a_larger_step(buft: BufferType, fresh: bool) -> (Vec<usize>, Vec<usize>) {
    let new = || ComputeAllocator::new(buft).with_fresh_plans(fresh);
    let ctx = Context::new(64, 64);
    let large = step(&ctx, 8);
    ctx.alloc_graph(&large, new()).unwrap();
    let used = ctx.into_allocator().expect("its own allocator");
    let ctx = Context::new(64, 64);
    let small = step(&ctx, 2);
    ctx.alloc_graph(&small, used).unwrap();
    let alone = Context::new(64, 64);
    let small_alone = step(&alone, 2);
    alone.alloc_graph(&small_alone, new()).unwrap();
    (layout(&small), layout(&small_alone))
}

/// #459: ggml's allocator reuses its last plan for a graph that fits it, so a step's layout
/// (and with it which ops ggml-cuda fuses) followed the graphs built before it.
#[test]
fn reused_allocator_keeps_the_last_graphs_plan() {
    let buft = Backend::cpu().unwrap().buffer_type();
    let (reused, alone) = after_a_larger_step(buft, false);
    assert_ne!(
        reused, alone,
        "the 8-row step's offsets, not the 2-row step's own"
    );
}

/// With fresh plans (`--deterministic`) a graph's layout depends on the graph alone.
#[test]
fn fresh_plans_lay_a_graph_out_as_a_new_allocator_does() {
    let buft = Backend::cpu().unwrap().buffer_type();
    let (reused, alone) = after_a_larger_step(buft, true);
    assert_eq!(reused, alone);
}

/// A fresh plan keeps the allocator's buffer when the graph fits it: no reallocation, so
/// kept graphs sharing the buffer stay valid.
#[test]
fn fresh_plan_keeps_a_buffer_that_fits() {
    let buft = Backend::cpu().unwrap().buffer_type();
    let shared = std::sync::Arc::new(std::sync::Mutex::new(
        ComputeAllocator::new(buft).with_fresh_plans(true),
    ));
    let ctx = Context::new(64, 64);
    let large = step(&ctx, 8);
    assert!(
        ctx.alloc_graph_shared(&large, &shared).unwrap(),
        "first: grew"
    );
    let ctx2 = Context::new(64, 64);
    let small = step(&ctx2, 2);
    assert!(
        !ctx2.alloc_graph_shared(&small, &shared).unwrap(),
        "fits: did not grow"
    );
    let ctx3 = Context::new(64, 64);
    let larger = step(&ctx3, 16);
    assert!(ctx3.alloc_graph_shared(&larger, &shared).unwrap(), "grew");
}

/// `cast_shared` gives one node per source and type in a context; views and other contexts
/// get their own, and `shared_cast` finds only what `cast_shared` made.
#[test]
fn cast_shared_reuses_one_node_per_source_and_type() {
    let ctx = Context::new(16, 0);
    let x = ctx.scale(ctx.new_tensor(types::F32, &[64, 8]), 2.0);
    let y = ctx.scale(x, 3.0);
    assert!(
        ctx.shared_cast(x, types::BF16).is_none(),
        "nothing made yet"
    );
    let a = ctx.cast_shared(x, types::BF16);
    assert_eq!(a.ty(), types::BF16);
    assert_eq!(a.ne(), x.ne());
    assert_eq!(
        ctx.cast_shared(x, types::BF16).raw(),
        a.raw(),
        "same source, same node"
    );
    assert_eq!(
        ctx.shared_cast(x, types::BF16).map(|t| t.raw()),
        Some(a.raw())
    );
    assert_ne!(
        ctx.cast_shared(x, types::F16).raw(),
        a.raw(),
        "another type"
    );
    assert_ne!(
        ctx.cast_shared(y, types::BF16).raw(),
        a.raw(),
        "another source"
    );
    assert!(ctx.shared_cast(y, types::F16).is_none());
    let v = ctx.view_2d(x, [64, 4], x.nb()[1], 0);
    let v1 = ctx.cast_shared(v, types::BF16);
    assert_ne!(
        ctx.cast_shared(v, types::BF16).raw(),
        v1.raw(),
        "views: a cast each"
    );
    assert!(ctx.shared_cast(v, types::BF16).is_none());
    let ctx2 = Context::new(4, 0);
    let x2 = ctx2.new_tensor(types::F32, &[64, 8]);
    assert!(ctx2.shared_cast(x2, types::BF16).is_none(), "per context");
}

/// A reshape of a node with a shared cast takes that cast reshaped (#469); without one, or for
/// another view, a cast of its own.
#[test]
fn cast_shared_of_a_reshape_reshapes_the_sources_cast() {
    let ctx = Context::new(32, 0);
    let x = ctx.scale(ctx.new_tensor(types::F32, &[64, 8]), 2.0);
    let r = ctx.reshape_3d(x, 64, 1, 8);
    let own = ctx.cast_shared(r, types::BF16);
    assert_ne!(own.raw(), r.raw());
    assert!(
        ctx.shared_cast(x, types::BF16).is_none(),
        "nothing shared yet"
    );
    let c = ctx.cast_shared(x, types::BF16);
    let rc = ctx.cast_shared(r, types::BF16);
    assert_eq!((rc.ty(), rc.ne()), (types::BF16, r.ne()));
    // SAFETY: live nodes of this context.
    let (op, src) = unsafe { ((*rc.raw()).op, (*rc.raw()).src[0]) };
    assert_eq!(op, ffi::ggml_op_GGML_OP_RESHAPE, "a reshape of the cast");
    assert_eq!(src, c.raw(), "of the shared cast");
    let v = ctx.view_2d(x, [64, 8], x.nb()[1], 0);
    let vc = ctx.cast_shared(v, types::BF16);
    // SAFETY: a live node of this context.
    assert_eq!(unsafe { (*vc.raw()).op }, ffi::ggml_op_GGML_OP_CPY);
}

#[test]
fn set_shared_cast_replaces_the_cast_for_its_source_and_type() {
    let ctx = Context::new(16, 0);
    let x = ctx.scale(ctx.new_tensor(types::F32, &[64, 8]), 2.0);
    let direct = ctx.new_tensor(types::BF16, &[64, 8]);
    ctx.set_shared_cast(x, types::BF16, direct);
    assert_eq!(ctx.cast_shared(x, types::BF16).raw(), direct.raw());
    let other = ctx.new_tensor(types::BF16, &[64, 8]);
    ctx.set_shared_cast(x, types::BF16, other);
    assert_eq!(
        ctx.shared_cast(x, types::BF16).map(|t| t.raw()),
        Some(other.raw())
    );
    assert!(ctx.shared_cast(x, types::F16).is_none(), "per type");
    // A view's cast is shared once set, though cast_shared never makes one for it.
    let v = ctx.view_2d(x, [64, 4], x.nb()[1], 0);
    let vd = ctx.new_tensor(types::BF16, &[64, 4]);
    ctx.set_shared_cast(v, types::BF16, vd);
    assert_eq!(ctx.cast_shared(v, types::BF16).raw(), vd.raw());
}

#[test]
#[should_panic(expected = "set_shared_cast")]
fn set_shared_cast_rejects_another_shape() {
    let ctx = Context::new(8, 0);
    let x = ctx.new_tensor(types::F32, &[64, 8]);
    ctx.set_shared_cast(x, types::BF16, ctx.new_tensor(types::BF16, &[64, 4]));
}

/// `shared_node` builds one node per source and key in a context; views get their own.
#[test]
fn shared_node_reuses_one_node_per_source_and_key() {
    let ctx = Context::new(16, 0);
    let x = ctx.scale(ctx.new_tensor(types::F32, &[64, 8]), 2.0);
    let y = ctx.scale(x, 3.0);
    let mut made = 0;
    let mut node = |a, key| {
        ctx.shared_node(a, key, || {
            made += 1;
            ctx.scale(a, 0.5)
        })
    };
    let a = node(x, "q");
    assert_eq!(node(x, "q").raw(), a.raw(), "same source and key");
    assert_ne!(node(x, "r").raw(), a.raw(), "another key");
    assert_ne!(node(y, "q").raw(), a.raw(), "another source");
    let v = ctx.view_2d(x, [64, 4], x.nb()[1], 0);
    let v1 = node(v, "q");
    assert_ne!(node(v, "q").raw(), v1.raw(), "views: a node each");
    assert_eq!(made, 5);
}

/// `set_shared_node` replaces what `shared_node` returns for its source and key, views
/// included, and leaves other keys alone.
#[test]
fn set_shared_node_replaces_the_node_for_its_source_and_key() {
    let ctx = Context::new(16, 0);
    let x = ctx.scale(ctx.new_tensor(types::F32, &[64, 8]), 2.0);
    let direct = ctx.scale(x, 0.25);
    ctx.set_shared_node(x, "q", direct);
    fn never<'a>() -> crate::ggml::Tensor<'a> {
        panic!("set_shared_node gave one")
    }
    assert_eq!(ctx.shared_node(x, "q", never).raw(), direct.raw());
    let other = ctx.scale(x, 0.5);
    ctx.set_shared_node(x, "q", other);
    assert_eq!(ctx.shared_node(x, "q", never).raw(), other.raw());
    assert_ne!(
        ctx.shared_node(x, "r", || ctx.scale(x, 3.0)).raw(),
        other.raw()
    );
    let v = ctx.view_2d(x, [64, 4], x.nb()[1], 0);
    let vd = ctx.scale(v, 0.5);
    ctx.set_shared_node(v, "q", vd);
    assert_eq!(ctx.shared_node(v, "q", never).raw(), vd.raw());
}

/// `move_after` puts the last nodes added right after the anchor, the others in their order.
#[test]
fn move_after_puts_the_last_nodes_after_the_anchor() {
    let ctx = Context::new(16, 16);
    let x = ctx.new_tensor(types::F32, &[4]).as_input();
    let a = ctx.scale(x, 2.0);
    let b = ctx.scale(a, 3.0);
    let c = ctx.scale(b, 4.0);
    let d = ctx.scale(a, 5.0);
    let mut graph = ctx.new_graph(16);
    graph.expand(c);
    let from = graph.nodes();
    graph.expand(d);
    graph.move_after(a, from);
    let order: Vec<_> = (0..graph.nodes())
        // SAFETY: indices in range.
        .map(|i| unsafe { ffi::ggml_graph_node(graph.raw(), i as i32) })
        .collect();
    assert_eq!(order, [a, d, b, c].map(|t| t.raw()));
}

/// A moved node may not read a node it would then run ahead of.
#[test]
#[should_panic(expected = "reads a node between")]
fn move_after_keeps_sources_ahead() {
    let ctx = Context::new(16, 16);
    let x = ctx.new_tensor(types::F32, &[4]).as_input();
    let a = ctx.scale(x, 2.0);
    let b = ctx.scale(a, 3.0);
    let d = ctx.scale(b, 5.0);
    let mut graph = ctx.new_graph(16);
    graph.expand(b);
    let from = graph.nodes();
    graph.expand(d);
    graph.move_after(a, from);
}

/// One run of `x * 2` (x an input, filled with 1 once, after the allocation), then a second
/// run without filling it again: the outputs of both runs.
fn input_filled_once(keep: bool) -> (Vec<f32>, Vec<f32>) {
    let mut cpu = Backend::cpu().unwrap();
    let ctx = Context::new(8, 8);
    let x = ctx.new_tensor(types::F32, &[64]).as_input();
    let x = if keep { x.as_output() } else { x };
    let y = ctx.scale(x, 2.0).as_output();
    let mut graph = ctx.new_graph(8);
    graph.expand(y);
    ctx.alloc_graph(&graph, ComputeAllocator::new(cpu.buffer_type()))
        .unwrap();
    x.set_f32(&[1.0; 64]);
    cpu.compute(&graph).unwrap();
    let first = y.to_f32();
    cpu.compute(&graph).unwrap();
    (first, y.to_f32())
}

/// #836: an input's memory is its own only until its last consumer ran (here `y`, computed in
/// place over it), so a graph run again must have its inputs filled again; a bench that filled
/// its input once read a stale token 0 on every later run.
#[test]
fn an_input_is_reused_after_its_last_consumer() {
    let (first, second) = input_filled_once(false);
    assert_eq!(first, [2.0; 64]);
    assert_eq!(
        second, [4.0; 64],
        "the second run read the first run's output"
    );
}

/// An input that is an output too keeps the value set once.
#[test]
fn an_input_marked_output_keeps_its_value() {
    let (first, second) = input_filled_once(true);
    assert_eq!(first, [2.0; 64]);
    assert_eq!(second, [2.0; 64]);
}

/// #836: a tensor's slot gets back the very tensor a search by its name finds.
#[test]
fn tensor_at_a_slot_is_the_named_tensor() {
    let ctx = Context::new(8, 0);
    let slots: Vec<_> = (0..8)
        .map(|i| {
            ctx.new_tensor(types::F32, &[4, i + 2])
                .named(&format!("blk.{i}.w"))
                .slot()
        })
        .collect();
    for (i, &slot) in slots.iter().enumerate() {
        let named = ctx.tensor(&format!("blk.{i}.w")).unwrap();
        // SAFETY: every slot is a tensor of `ctx`.
        let at = unsafe { ctx.tensor_at(slot) };
        assert_eq!(at.raw(), named.raw());
        assert_eq!(at.name(), format!("blk.{i}.w"));
        assert_eq!(at.shape(), [4, i as i64 + 2]);
    }
}

/// Tensors enough for a context's metadata to come from the pool, at a size no other test's
/// contexts have.
fn pooled_tensors() -> usize {
    // SAFETY: a size query.
    super::META_POOL_MIN / unsafe { ffi::ggml_tensor_overhead() } + 7
}

/// `x * 3 + 1` over 64 ones, built and run in `ctx`.
fn run_in(ctx: &Context) -> Vec<f32> {
    let mut cpu = Backend::cpu().unwrap();
    let x = ctx.new_tensor(types::F32, &[64]).as_input();
    let y = ctx.scale(x, 3.0);
    let one = ctx.new_tensor(types::F32, &[64]).as_input();
    let z = ctx.add(y, one).as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(z);
    ctx.alloc_graph(&graph, ComputeAllocator::new(cpu.buffer_type()))
        .unwrap();
    x.set_f32(&[1.0; 64]);
    one.set_f32(&[1.0; 64]);
    cpu.compute(&graph).unwrap();
    z.to_f32()
}

/// #836: a big context's metadata memory (a step graph's, ~7 MB) serves the next context of
/// its size once it is dropped, and nothing of what it held shows there.
#[test]
fn a_big_contexts_metadata_memory_serves_the_next_one() {
    let n = pooled_tensors();
    // SAFETY: a live context's metadata address.
    let base = |ctx: &Context| unsafe { ffi::ggml_get_mem_buffer(ctx.raw()) as usize };
    let first = Context::new(n, 16);
    assert!(first.meta.is_some(), "a pooled context");
    for i in 0..100 {
        first
            .new_tensor(types::F32, &[8, i + 1])
            .named(&format!("stale.{i}"));
    }
    assert_eq!(run_in(&first), [4.0; 64]);
    let at = base(&first);
    drop(first);
    let second = Context::new(n, 16);
    assert_eq!(base(&second), at, "the dropped context's memory");
    assert!(second.tensor("stale.0").is_none());
    assert!(second.tensor("stale.99").is_none());
    assert_eq!(run_in(&second), [4.0; 64]);
    let t = second.new_tensor(types::F32, &[5, 6]).named("fresh");
    assert_eq!(second.tensor("fresh").unwrap().raw(), t.raw());
}

/// Small contexts keep ggml's own metadata memory; the pool keeps at most
/// [`super::META_POOL_KEEP`] buffers.
#[test]
fn small_contexts_and_a_full_pool() {
    let small = Context::new(8, 8);
    assert!(small.meta.is_none());
    assert_eq!(run_in(&small), [4.0; 64]);
    let n = pooled_tensors() + 3;
    let many: Vec<Context> = (0..super::META_POOL_KEEP + 2)
        .map(|_| Context::new(n, 0))
        .collect();
    drop(many);
    assert!(super::meta_pool_len() <= super::META_POOL_KEEP);
}

/// #836: with its names indexed, a context finds what ggml's search finds: each name's first
/// tensor, nothing for a name it lacks, and (searching again) a tensor made after the index.
#[test]
fn indexed_names_find_what_the_search_finds() {
    let ctx = Context::new(16, 0);
    let ts: Vec<_> = (0..6)
        .map(|i| {
            ctx.new_tensor(types::F32, &[4, i + 1])
                .named(&format!("cache_k-{i}"))
        })
        .collect();
    let again = ctx.new_tensor(types::F32, &[9]).named("cache_k-2");
    ctx.index_names();
    for (i, t) in ts.iter().enumerate() {
        assert_eq!(ctx.tensor(&format!("cache_k-{i}")).unwrap().raw(), t.raw());
    }
    assert_ne!(again.raw(), ts[2].raw());
    assert!(ctx.tensor("cache_v-0").is_none());
    assert!(ctx.tensor("cache_k-").is_none());
    assert!(ctx.tensor("cache_k-1\0x").is_none());
    let late = ctx.new_tensor(types::F32, &[3]).named("late");
    assert_eq!(ctx.tensor("late").unwrap().raw(), late.raw());
    assert_eq!(ctx.tensor("cache_k-5").unwrap().raw(), ts[5].raw());
}

/// #1176: an I32 tensor read through a host tensor (pinned memory on a GPU) gives what a plain
/// read gives.
#[test]
fn to_i32_via_reads_what_to_i32_reads() {
    let cpu = Backend::cpu().unwrap();
    let ctx = Context::new(2, 0);
    let t = ctx.new_tensor(types::I32, &[5]);
    let via = ctx.new_tensor(types::F32, &[8]);
    ctx.alloc_tensors(cpu.buffer_type()).unwrap();
    t.set_i32(&[3, -1, 7, 0, 151_000]);
    assert_eq!(t.to_i32_via(via), [3, -1, 7, 0, 151_000]);
    assert_eq!(t.to_i32_via(via), t.to_i32());
}
