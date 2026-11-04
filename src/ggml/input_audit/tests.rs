use super::{Finding, Kind, scan, summary};
use crate::ggml::{Backend, ComputeAllocator, Context, Tensor, types};

/// `y = x * 2` then `z = y * 3` (each computed in place over its source), `x` an input built
/// by `make`, allocated on the CPU; the findings with `listed` (by `x`) refilled.
fn findings(make: impl Fn(Tensor<'_>) -> Tensor<'_>, listed: bool) -> Vec<Finding> {
    let cpu = Backend::cpu().unwrap();
    let ctx = Context::new(8, 8);
    let x = make(ctx.new_tensor(types::F32, &[64]).named("x").as_input());
    let y = ctx.scale(x, 2.0);
    let z = ctx.scale(y, 3.0).as_output();
    let mut graph = ctx.new_graph(8);
    graph.expand(z);
    ctx.alloc_graph(&graph, ComputeAllocator::new(cpu.buffer_type()))
        .unwrap();
    let refilled = if listed { vec![x.raw()] } else { Vec::new() };
    // SAFETY: an allocated graph of live tensors.
    unsafe { scan(graph.raw(), &refilled) }
}

/// #1028's case: an input filled once, then the graph run again, reads the node placed over it.
#[test]
fn an_input_set_once_with_a_node_over_it_is_reported() {
    let f = findings(|x| x, false);
    assert_eq!(f.len(), 1, "{f:?}");
    assert_eq!(f[0].name, "x");
    assert_eq!(f[0].kind, Kind::NotRefilled);
    assert_eq!(f[0].bytes, 256);
    assert!(f[0].over[0].starts_with("SCALE"), "{:?}", f[0].over);
    assert!(f[0].over[0].contains("at +0 B, 256 B"), "{:?}", f[0].over);
}

#[test]
fn a_refilled_input_is_listed_as_such() {
    let f = findings(|x| x, true);
    assert_eq!(f.len(), 1, "{f:?}");
    assert_eq!(f[0].kind, Kind::Refilled);
}

#[test]
fn an_input_marked_output_has_nothing_over_it() {
    assert!(findings(|x| x.as_output(), false).is_empty());
}

#[test]
fn a_scratch_input_is_listed_as_scratch() {
    let f = findings(|x| x.named("attn_scratch-0"), false);
    assert_eq!(f.len(), 1, "{f:?}");
    assert_eq!(f[0].kind, Kind::Scratch);
}

/// A leaf without the input flag in the compute buffer is checked too (the allocator gave it
/// memory, so the same holds).
#[test]
fn a_compute_buffer_leaf_counts_as_an_input() {
    let cpu = Backend::cpu().unwrap();
    let ctx = Context::new(8, 8);
    let w = ctx.new_tensor(types::F32, &[64]).named("w");
    let y = ctx.scale(w, 2.0);
    let z = ctx.scale(y, 3.0).as_output();
    let mut graph = ctx.new_graph(8);
    graph.expand(z);
    ctx.alloc_graph(&graph, ComputeAllocator::new(cpu.buffer_type()))
        .unwrap();
    // SAFETY: an allocated graph of live tensors.
    let f = unsafe { scan(graph.raw(), &[]) };
    assert_eq!(f.len(), 1, "{f:?}");
    assert_eq!(f[0].name, "w");
    assert_eq!(f[0].kind, Kind::NotRefilled);
}

#[test]
fn the_summary_counts_and_names_by_kind() {
    let f = |name: &str, kind| Finding {
        name: name.into(),
        bytes: 4,
        kind,
        over: vec!["SCALE".into()],
    };
    let all = [
        f("kq_mask-0", Kind::Refilled),
        f("kq_mask-1", Kind::Refilled),
        f("tokens", Kind::Refilled),
        f("attn_scratch-0", Kind::Scratch),
        f("ids", Kind::NotRefilled),
    ];
    assert_eq!(
        summary(10, &all),
        "graph input audit: 10 nodes, inputs overlapped by a node: refilled 3 [kq_mask, \
         tokens]; scratch 1 [attn_scratch]; NOT refilled 1 [ids]"
    );
    assert_eq!(
        summary(3, &[]),
        "graph input audit: 3 nodes, inputs overlapped by a node: none"
    );
}

/// `ctx.fill(ctx.new_tensor(..), c)`: the new tensor is only the FILL's shape, its data is
/// never read, so a node over it is no stale read.
#[test]
fn a_fill_shape_source_is_unread() {
    let cpu = Backend::cpu().unwrap();
    let ctx = Context::new(8, 8);
    let w = ctx.new_tensor(types::F32, &[64]).named("w");
    let y = ctx.scale(ctx.fill(w, 1.0), 2.0);
    let z = ctx.scale(y, 3.0).as_output();
    let mut graph = ctx.new_graph(8);
    graph.expand(z);
    ctx.alloc_graph(&graph, ComputeAllocator::new(cpu.buffer_type()))
        .unwrap();
    // SAFETY: an allocated graph of live tensors.
    let f = unsafe { scan(graph.raw(), &[]) };
    assert!(f.iter().all(|f| f.kind == Kind::Unread), "{f:?}");
}

/// Memory outside the compute buffers is not the allocator's: a node placed there on purpose
/// is not reported.
#[test]
fn an_input_outside_the_compute_buffer_is_not_checked() {
    let cpu = Backend::cpu().unwrap();
    let pool = Context::new(1, 0);
    let s = pool.new_tensor(types::F32, &[64]).named("state");
    pool.alloc_tensors(cpu.buffer_type()).unwrap();
    let ctx = Context::new(8, 8);
    let y = ctx.scale(s, 2.0);
    // SAFETY: `y` is an unallocated op result and `s` holds exactly its bytes.
    unsafe { y.place_in(s, 0) };
    let z = ctx.scale(y, 3.0).as_output();
    let mut graph = ctx.new_graph(8);
    graph.expand(z);
    ctx.alloc_graph(&graph, ComputeAllocator::new(cpu.buffer_type()))
        .unwrap();
    // SAFETY: an allocated graph of live tensors.
    assert!(unsafe { scan(graph.raw(), &[]) }.is_empty());
}
