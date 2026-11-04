//! Allocation dependencies: what they keep, and that they change no value.

use crate::ggml::{Backend, ComputeAllocator, Context, Graph, Tensor, ffi, types};

const E: i64 = 64;
const USED: i64 = 4;
const N: i64 = 2;

/// Whether `a` and `b` share memory (ggml-cuda's `nodes_overlap`, on allocated tensors).
fn overlap(a: Tensor<'_>, b: Tensor<'_>) -> bool {
    // SAFETY: allocated tensors of a live graph.
    unsafe {
        let start = |t: Tensor<'_>| (*t.raw()).data as usize;
        let (a0, b0) = (start(a), start(b));
        let (a1, b1) = (a0 + a.nbytes(), b0 + b.nbytes());
        (b0 <= a0 && a0 < b1) || (a0 <= b0 && b0 < a1)
    }
}

/// A MoE layer's weighted sum as ggml's chain (`mul` by the weights, a view per expert, adds)
/// after a temporary whose freed memory the routing weights then take: once the `mul` has read
/// the weights, their block and the rest of the temporary's are free together, as big as the
/// sum, which the allocator puts there. ggml-cuda would not fuse this chain.
/// `(experts, weights, sum, other output)`.
fn weighted_sum(ctx: &Context, pin: bool) -> (Graph<'_>, [Tensor<'_>; 4]) {
    // Inputs that stay allocated (outputs too), so no node takes their memory in place.
    let input = |shape: &[i64]| ctx.new_tensor(types::F32, shape).as_input().as_output();
    let (tmp_in, w_in, e_in) = (input(&[E * N]), input(&[1, USED, N]), input(&[E, USED, N]));
    let tmp = ctx.scale(tmp_in, 2.0);
    let side = ctx.sum_rows(tmp).as_output();
    let w = ctx.scale(w_in, 0.5);
    let experts = ctx.scale(e_in, 3.0);
    let weighted = ctx.mul(experts, w);
    let nb = weighted.nb();
    let mut sum = ctx.view_2d(weighted, [E, N], nb[2], 0);
    for e in 1..USED as usize {
        sum = ctx.add(sum, ctx.view_2d(weighted, [E, N], nb[2], e * nb[1]));
    }
    let mut graph = ctx.new_graph(64);
    graph.expand(side);
    if pin {
        assert_eq!(graph.keep_until(&[experts, w], sum), 1);
    }
    let out = ctx.scale(sum, 1.0).as_output();
    graph.expand(out);
    (graph, [experts, w, sum, out])
}

fn fill(t: Tensor<'_>, seed: u32) {
    let v: Vec<f32> = (0..t.elements())
        .map(|i| ((i as u32).wrapping_mul(2_654_435_761) ^ seed) as f32 / u32::MAX as f32 - 0.5)
        .collect();
    t.set_f32(&v);
}

/// Runs the case on the CPU: whether the sum overlaps the experts or their weights, and the
/// output's bits.
fn run(pin: bool) -> (bool, Vec<u32>) {
    let mut backend = Backend::cpu().unwrap();
    let ctx = Context::new(64, 64);
    let (graph, [experts, w, sum, out]) = weighted_sum(&ctx, pin);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let overlaps = overlap(sum, experts) || overlap(sum, w);
    // SAFETY: the graph's leaves and nodes; the inputs are the leaves the graph reads.
    let inputs: Vec<Tensor<'_>> = unsafe {
        let g = graph.raw();
        (0..ffi::ggml_graph_n_nodes(g))
            .flat_map(|i| (*ffi::ggml_graph_node(g, i)).src)
            .filter(|s| {
                !s.is_null()
                    && (**s).flags & ffi::ggml_tensor_flag_GGML_TENSOR_FLAG_INPUT as i32 != 0
            })
            .map(|s| ctx.wrap(s))
            .collect()
    };
    for (k, t) in inputs.iter().enumerate() {
        fill(*t, k as u32);
    }
    backend.compute(&graph).unwrap();
    (overlaps, out.to_f32().iter().map(|x| x.to_bits()).collect())
}

/// The layout ggml-cuda declines to fuse: the sum takes the weights' freed memory.
#[test]
fn unpinned_sum_takes_the_weights_memory() {
    let (overlaps, _) = run(false);
    assert!(overlaps, "the case no longer reproduces the overlap");
}

/// Kept until the sum, the experts and the weights cannot share its memory, so the fusion's
/// memory check passes whatever the rest of the graph is.
#[test]
fn pinned_sum_overlaps_none_of_its_inputs() {
    let (overlaps, _) = run(true);
    assert!(!overlaps);
}

/// The dependency node computes nothing: the same output bits.
#[test]
fn pinning_changes_no_value() {
    assert_eq!(run(false).1, run(true).1);
}

/// Leaves are not the allocator's to free: no dependency node for them alone.
#[test]
fn leaves_are_not_kept() {
    let ctx = Context::new(16, 16);
    let x = ctx.new_tensor(types::F32, &[8]);
    let y = ctx.scale(x, 2.0);
    let mut graph = ctx.new_graph(16);
    assert_eq!(graph.keep_until(&[x], y), 0);
    assert_eq!(graph.nodes(), 1);
    assert_eq!(
        graph.keep_until(&[y, y], ctx.scale(y, 3.0)),
        1,
        "one node, deduplicated"
    );
    assert_eq!(graph.nodes(), 3);
}

#[test]
fn srcs_and_mul() {
    let ctx = Context::new(16, 16);
    let a = ctx.new_tensor(types::F32, &[8]);
    let b = ctx.new_tensor(types::F32, &[8]);
    let m = ctx.mul(a, b);
    assert!(m.is_mul() && !a.is_mul());
    assert_eq!(
        m.srcs().iter().map(|t| t.raw()).collect::<Vec<_>>(),
        [a.raw(), b.raw()]
    );
    assert!(a.srcs().is_empty());
}

/// One pass of a graph whose nodes stay the same while an input it does not read grows (an MTP
/// take-in piece's attention mask over a longer context), allocated from `shared`: where the
/// mask ends and the buffer's size, in bytes from the buffer's start.
fn unread_mask(shared: &crate::ggml::SharedAllocator, n_kv: i64, unread: bool) -> (usize, usize) {
    let ctx = Context::new(8, 8);
    let x = ctx.new_tensor(types::F32, &[64]).as_input();
    let mask = ctx.new_tensor(types::F32, &[n_kv, 16]).as_input();
    let y = ctx.scale(x, 2.0).as_output();
    let mut graph = ctx.new_graph(8);
    graph.expand(y);
    if unread {
        assert_eq!(
            graph.expand_unread(&[x, mask]),
            1,
            "x is read: one node, for the mask"
        );
        assert_eq!(graph.nodes(), 2);
    } else {
        graph.expand(mask);
    }
    ctx.alloc_graph_shared(&graph, shared).unwrap();
    // SAFETY: an allocated tensor of a live graph, in a live buffer.
    unsafe {
        let buf = (*mask.raw()).buffer;
        let base = ffi::ggml_backend_buffer_get_base(buf) as usize;
        let end = (*mask.raw()).data as usize + mask.nbytes() - base;
        (end, ffi::ggml_backend_buffer_get_size(buf))
    }
}

/// #527: kept as a dependency node's source, an unread input that grew gets a new plan and
/// fits the buffer. (As a bare leaf the allocator kept the old plan: the mask ran past the
/// buffer and ggml aborted.)
#[test]
fn an_unread_input_that_grows_gets_room() {
    let shared = std::sync::Arc::new(std::sync::Mutex::new(
        ComputeAllocator::new(Backend::cpu().unwrap().buffer_type()).with_fresh_plans(false),
    ));
    for n_kv in [256, 512, 4096] {
        let (end, size) = unread_mask(&shared, n_kv, true);
        assert!(
            end <= size,
            "n_kv {n_kv}: the mask ends at {end}, the buffer at {size}"
        );
    }
}

/// The dependency node moves no tensor: the same layout as with the input a bare leaf.
#[test]
fn expand_unread_keeps_the_layout() {
    let alloc = || {
        std::sync::Arc::new(std::sync::Mutex::new(
            ComputeAllocator::new(Backend::cpu().unwrap().buffer_type()).with_fresh_plans(false),
        ))
    };
    assert_eq!(
        unread_mask(&alloc(), 256, true),
        unread_mask(&alloc(), 256, false)
    );
}
