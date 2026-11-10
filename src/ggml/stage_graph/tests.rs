use super::*;
use crate::ggml::{Context, types};

/// The order the plain path (`compute_with_external`) runs a graph in, as units: segments
/// between the external ops, the late input written just before its first reader's segment,
/// or before the whole graph when there are no external ops.
fn plain(n: i32, ext: &[i32], late: Option<i32>) -> Vec<Unit> {
    let mut out = Vec::new();
    if ext.is_empty() {
        if late.is_some() {
            out.push(Unit::Fill);
        }
        if n > 0 {
            out.push(Unit::Segment(0, n));
        }
        return out;
    }
    let mut late = late;
    let mut start = 0;
    let seg = |out: &mut Vec<Unit>, a: i32, b: i32| {
        if a < b {
            out.push(Unit::Segment(a, b));
        }
    };
    for i in ext.iter().copied().chain([n]) {
        if let Some(at) = late.filter(|&at| at <= i) {
            seg(&mut out, start, at);
            out.push(Unit::Fill);
            start = at;
            late = None;
        }
        seg(&mut out, start, i);
        if i < n {
            out.push(Unit::External(i));
        }
        start = i + 1;
    }
    out
}

#[test]
fn units_without_external_ops_are_the_whole_graph() {
    assert_eq!(units(5, &[], None), vec![Unit::Segment(0, 5)]);
    assert_eq!(
        units(5, &[], Some(3)),
        vec![Unit::Fill, Unit::Segment(0, 5)]
    );
    assert_eq!(units(0, &[], None), vec![]);
}

#[test]
fn units_split_around_external_ops() {
    use Unit::*;
    assert_eq!(
        units(6, &[2, 4], None),
        vec![
            Segment(0, 2),
            External(2),
            Segment(3, 4),
            External(4),
            Segment(5, 6)
        ]
    );
    // Adjacent ops and ops at the ends leave no empty segments.
    assert_eq!(
        units(4, &[0, 1, 3], None),
        vec![External(0), External(1), Segment(2, 3), External(3)]
    );
    // The late input goes right before its first reader's segment.
    assert_eq!(
        units(6, &[2, 4], Some(1)),
        vec![
            Segment(0, 1),
            Fill,
            Segment(1, 2),
            External(2),
            Segment(3, 4),
            External(4),
            Segment(5, 6)
        ]
    );
    assert_eq!(
        units(6, &[2, 4], Some(4)),
        vec![
            Segment(0, 2),
            External(2),
            Segment(3, 4),
            Fill,
            External(4),
            Segment(5, 6)
        ]
    );
    assert_eq!(
        units(6, &[2], Some(0)),
        vec![Fill, Segment(0, 2), External(2), Segment(3, 6)]
    );
}

/// Every layout of up to 7 nodes, external ops and late reader: the units are the plain
/// path's segments in its order, so ggml computes the same views.
#[test]
fn units_match_the_plain_path() {
    for n in 0..=7i32 {
        for mask in 0u32..1 << n {
            let ext: Vec<i32> = (0..n).filter(|i| mask >> i & 1 == 1).collect();
            for late in [None].into_iter().chain((0..n.max(1)).map(Some)) {
                assert_eq!(
                    units(n, &ext, late),
                    plain(n, &ext, late),
                    "n {n} ext {ext:?} late {late:?}"
                );
            }
        }
    }
}

#[test]
fn pieces_group_capturable_runs() {
    use Unit::*;
    let u = [
        Segment(0, 2),
        External(2),
        Segment(3, 4),
        Fill,
        External(4),
        Segment(5, 6),
    ];
    // All but the fill: two captures around it.
    assert_eq!(
        pieces(&u, |u| *u != Fill),
        vec![
            Piece::Captured(0..3),
            Piece::Direct(3),
            Piece::Captured(4..6)
        ]
    );
    // An external op the model layer did not allow splits the run.
    assert_eq!(
        pieces(&u, |u| !matches!(u, Fill | External(2))),
        vec![
            Piece::Captured(0..1),
            Piece::Direct(1),
            Piece::Captured(2..3),
            Piece::Direct(3),
            Piece::Captured(4..6)
        ]
    );
    // A lone external op is not captured; a lone segment is (it is ggml's own graph).
    assert_eq!(
        pieces(&u, |u| matches!(u, External(4))),
        (0..6).map(Piece::Direct).collect::<Vec<_>>()
    );
    assert_eq!(
        pieces(&[Segment(0, 3)], |_| true),
        vec![Piece::Captured(0..1)]
    );
    assert_eq!(pieces(&[], |_| true), vec![]);
}

/// Each unit is in exactly one piece, in order.
#[test]
fn pieces_cover_every_unit_once() {
    use Unit::*;
    let u: Vec<Unit> = (0..9)
        .map(|i| match i % 3 {
            0 => Segment(i, i + 1),
            1 => External(i),
            _ => Fill,
        })
        .collect();
    for mask in 0u32..1 << u.len() {
        let p = pieces(&u, |x| {
            let i = u.iter().position(|y| y == x).unwrap();
            mask >> i & 1 == 1
        });
        let order: Vec<usize> = p
            .iter()
            .flat_map(|p| match p {
                Piece::Captured(r) => r.clone().collect::<Vec<_>>(),
                Piece::Direct(i) => vec![*i],
            })
            .collect();
        assert_eq!(order, (0..u.len()).collect::<Vec<_>>(), "mask {mask:b}");
    }
}

/// `x * s + x`, `n` floats.
fn graph_words(ctx: &Context, s: f32, n: i64) -> Vec<u64> {
    let x = ctx.new_tensor(types::F32, &[n]);
    let y = ctx.add(ctx.scale(x, s), x);
    let mut graph = ctx.new_graph(8);
    graph.expand(y);
    let mut out = Vec::new();
    // SAFETY: a live graph of this context.
    unsafe { props(graph.raw(), &mut out) };
    out
}

#[test]
fn props_see_parameters_and_shapes() {
    let ctx = Context::new(16, 32);
    let a = graph_words(&ctx, 2.0, 8);
    // Two nodes, each with its words.
    assert_eq!(a[0], 2);
    let c = graph_words(&ctx, 3.0, 8);
    let d = graph_words(&ctx, 2.0, 16);
    assert_eq!(a.len(), c.len());
    assert_eq!(a.len(), d.len());
    assert_ne!(a, c, "a parameter");
    assert_ne!(a, d, "a shape");
}

#[test]
fn props_of_the_same_graph_repeat() {
    let ctx = Context::new(8, 32);
    let x = ctx.new_tensor(types::F32, &[4]);
    let mut graph = ctx.new_graph(8);
    graph.expand(ctx.scale(ctx.add(x, x), 2.0));
    let (mut a, mut b) = (Vec::new(), Vec::new());
    // SAFETY: a live graph of this context.
    unsafe {
        props(graph.raw(), &mut a);
        props(graph.raw(), &mut b);
    }
    assert_eq!(a, b);
    // A changed parameter changes the words.
    // SAFETY: the scale node's first parameter is its factor (ggml_scale_impl).
    unsafe {
        let node = ffi::ggml_graph_node(graph.raw(), 1);
        (*node).op_params[0] = 3.0f32.to_bits() as i32;
        b.clear();
        props(graph.raw(), &mut b);
    }
    assert_ne!(a, b);
}

#[test]
fn props_same_is_comparing_fresh_words() {
    for nodes in [false, true] {
        let ctx = Context::new(8, 32);
        let x = ctx.new_tensor(types::F32, &[4]);
        let mut graph = ctx.new_graph(8);
        graph.expand(ctx.scale(ctx.add(x, x), 2.0));
        let g = graph.raw();
        // This run's words built whole (`props` for the full form), then the tail: what
        // `compute` compared before #849.
        let fresh = |tail: u64| {
            let mut w = Vec::new();
            // SAFETY: a live graph of this context.
            unsafe { each_node(g, nodes, |x| w.extend_from_slice(x)) };
            w.push(tail);
            w
        };
        let mut full = Vec::new();
        // SAFETY: a live graph of this context.
        unsafe { props(g, &mut full) };
        // `props` is the full form; the shorter one is tagged second.
        assert_eq!(nodes, fresh(0)[1] == u64::MAX);
        if !nodes {
            assert_eq!(full[..], fresh(0)[..full.len()]);
        }
        let mut last = Vec::new();
        // SAFETY (all `props_same` calls): a live graph of this context.
        let same = |last: &mut Vec<u64>, tail| unsafe { props_same(g, last, tail, nodes) };
        assert!(!same(&mut last, 7), "the first run");
        assert_eq!(last, fresh(7));
        assert!(same(&mut last, 7));
        assert!(same(&mut last, 7));
        assert_eq!(last, fresh(7));
        // Another tail, then a changed parameter (the scale node's factor, ggml_scale_impl).
        assert!(!same(&mut last, 8));
        assert_eq!(last, fresh(8));
        // SAFETY: node 1 is the scale node of this live graph.
        unsafe { (*ffi::ggml_graph_node(g, 1)).op_params[0] = 3.0f32.to_bits() as i32 };
        assert!(!same(&mut last, 8));
        assert_eq!(last, fresh(8));
        assert!(same(&mut last, 8));
        // Longer or shorter words than the last run's are a change, and are replaced.
        let mut longer = fresh(8);
        longer.push(1);
        assert!(!same(&mut longer, 8));
        assert_eq!(longer, fresh(8));
        let mut shorter = fresh(8);
        shorter.truncate(shorter.len() - 1);
        assert!(!same(&mut shorter, 8));
        assert_eq!(shorter, fresh(8));
        // The other form is a change.
        let mut other = Vec::new();
        // SAFETY: a live graph of this context.
        unsafe { each_node(g, !nodes, |x| other.extend_from_slice(x)) };
        other.push(8);
        assert!(!same(&mut other, 8));
        assert_eq!(other, fresh(8));
    }
}

#[test]
fn props_shorter_form_sees_what_the_full_one_does() {
    // A change the full form sees through a source's words: the input's data, a computed
    // source's shape, and a node pointed at another source.
    let ctx = Context::new(16, 32);
    let x = ctx.new_tensor(types::F32, &[4]);
    let y = ctx.new_tensor(types::F32, &[4]);
    let mut graph = ctx.new_graph(8);
    graph.expand(ctx.scale(ctx.add(x, y), 2.0));
    let g = graph.raw();
    // SAFETY: a live graph of this context; node 0 is the add, node 1 the scale.
    unsafe {
        assert!(srcs_are_nodes(g));
        let mut last = Vec::new();
        props_same(g, &mut last, 0, true);
        assert!(props_same(g, &mut last, 0, true));
        let add = ffi::ggml_graph_node(g, 0);
        let scale = ffi::ggml_graph_node(g, 1);
        let xr = x.raw();
        let data = (*xr).data;
        (*xr).data = 64usize as *mut std::ffi::c_void;
        assert!(!props_same(g, &mut last, 0, true), "an input's data");
        (*xr).data = data;
        assert!(!props_same(g, &mut last, 0, true));
        assert!(props_same(g, &mut last, 0, true));
        (*add).nb[1] += 4;
        assert!(
            !props_same(g, &mut last, 0, true),
            "a computed source's strides"
        );
        (*add).nb[1] -= 4;
        assert!(!props_same(g, &mut last, 0, true));
        assert!(props_same(g, &mut last, 0, true));
        (*scale).src[0] = y.raw();
        assert!(!props_same(g, &mut last, 0, true), "another source");
        (*scale).src[0] = add;
    }
}

#[test]
fn srcs_are_nodes_only_when_every_computed_source_is_a_node() {
    let ctx = Context::new(16, 32);
    let x = ctx.new_tensor(types::F32, &[4]);
    let a = ctx.add(x, x);
    let mut graph = ctx.new_graph(8);
    graph.expand(ctx.scale(a, 2.0));
    // A computed source outside the graph: `b` is in no graph's node list.
    let b = ctx.add(x, x);
    // SAFETY: a live graph of this context; node 1 is the scale.
    unsafe {
        assert!(srcs_are_nodes(graph.raw()));
        let scale = ffi::ggml_graph_node(graph.raw(), 1);
        (*scale).src[0] = b.raw();
        assert!(!srcs_are_nodes(graph.raw()));
    }
}

#[test]
fn ggml_capturable_follows_ggml_cuda() {
    let ctx = Context::new(16, 64);
    let x = ctx.new_tensor(types::F32, &[32, 4]);
    // SAFETY: tensors of a live context.
    unsafe {
        for mmq_ids in [false, true] {
            assert!(ggml_capturable(ctx.add(x, x).raw(), mmq_ids));
            let mean = ffi::ggml_mean(ctx.raw(), x.raw());
            assert!(!ggml_capturable(mean, mmq_ids));
        }
        // Quantized experts times F32 rows of up to 4 tokens: mmvq, no synchronization.
        let ids = ctx.new_tensor(types::I32, &[2, 4]);
        let q = ctx.new_tensor(types::Q8_0, &[32, 16, 8]);
        let b = ctx.new_tensor(types::F32, &[32, 2, 4]);
        assert!(ggml_capturable(ctx.mul_mat_id(q, b, ids).raw(), false));
        assert!(ggml_capturable(ctx.mul_mat_id(q, b, ids).raw(), true));
        let ids5 = ctx.new_tensor(types::I32, &[2, 5]);
        let b5 = ctx.new_tensor(types::F32, &[32, 2, 5]);
        assert!(!ggml_capturable(ctx.mul_mat_id(q, b5, ids5).raw(), false));
        let f = ctx.new_tensor(types::F16, &[32, 16, 8]);
        assert!(!ggml_capturable(ctx.mul_mat_id(f, b, ids).raw(), false));
        assert!(!ggml_capturable(ctx.mul_mat_id(f, b, ids).raw(), true));
    }
}

/// Where the device takes MMQ for expert batches (#827): an MMQ type's `mul_mat_id` is
/// captured up to [`MMQ_IDS_MAX_TOKENS`] - 1 tokens, others as before.
#[test]
fn mmq_expert_batches_are_capturable_where_the_device_takes_mmq() {
    let ctx = Context::new(64, 64);
    let rows = |n: i64| {
        (
            ctx.new_tensor(types::F32, &[256, 2, n]),
            ctx.new_tensor(types::I32, &[2, n]),
        )
    };
    let q4k = ctx.new_tensor(types::Q4_K, &[256, 16, 8]);
    let q8 = ctx.new_tensor(types::Q8_0, &[256, 16, 8]);
    let f16 = ctx.new_tensor(types::F16, &[256, 16, 8]);
    // SAFETY: tensors of a live context.
    unsafe {
        for n in [5, 9, 36, MMQ_IDS_MAX_TOKENS - 1] {
            let (b, ids) = rows(n);
            for w in [q4k, q8] {
                let node = ctx.mul_mat_id(w, b, ids).raw();
                assert!(ggml_capturable(node, true), "{n} tokens");
                assert!(!ggml_capturable(node, false), "{n} tokens");
            }
            assert!(!ggml_capturable(ctx.mul_mat_id(f16, b, ids).raw(), true));
        }
        // From MMQ_DP4A_MAX_BATCH_SIZE on, ggml-cuda's pick follows the architectures it was
        // compiled for: left out.
        let (b, ids) = rows(MMQ_IDS_MAX_TOKENS);
        assert!(!ggml_capturable(ctx.mul_mat_id(q4k, b, ids).raw(), true));
    }
    assert!(mmq_type(types::Q4_K) && mmq_type(types::Q8_0) && mmq_type(types::Q5_1));
    assert!(!mmq_type(types::F16) && !mmq_type(types::F32) && !mmq_type(types::BF16));
}

/// A verify step's MoE segment becomes part of the capture around it where the device takes
/// MMQ, and stays direct elsewhere.
#[test]
fn plan_captures_mmq_expert_segments_with_mmq_ids() {
    let ctx = Context::new(64, 64);
    let q = ctx.new_tensor(types::Q8_0, &[256, 16, 8]);
    let b = ctx.new_tensor(types::F32, &[256, 2, 9]);
    let ids = ctx.new_tensor(types::I32, &[2, 9]);
    let mut g = ctx.new_graph(16);
    g.expand(ctx.scale(ctx.mul_mat_id(q, ctx.scale(b, 2.0), ids), 0.5));
    // SAFETY: a live graph of this context.
    unsafe {
        let n = ffi::ggml_graph_n_nodes(g.raw());
        let u = units(n, &[], None);
        assert_eq!(plan(g.raw(), &u, true), vec![Piece::Captured(0..1)]);
        assert_eq!(plan(g.raw(), &u, false), vec![Piece::Direct(0)]);
    }
}

/// A big capture becomes a head of at least 1/HEAD_DIV of its nodes and the rest; small
/// captures and direct units stay as they are (#836).
#[test]
fn split_heads_cut_big_captures_after_a_short_head() {
    use Unit::*;
    // A fill, then 100 segments of 8 nodes with an external op after each (900 nodes).
    let mut u = vec![Fill];
    for k in 0..100 {
        let i = 1 + 9 * k;
        u.push(Segment(i, i + 8));
        u.push(External(i + 8));
    }
    let p = pieces(&u, |u| *u != Fill);
    assert_eq!(p, vec![Piece::Direct(0), Piece::Captured(1..201)]);
    let s = split_heads(p, &u);
    // ceil(900 / 32) = 29 nodes: four segments and three external ops (8 + 1 + 8 + 1 + 8 + 1 + 8).
    let want = (900 + HEAD_DIV - 1) / HEAD_DIV;
    assert_eq!(
        s,
        vec![
            Piece::Direct(0),
            Piece::Captured(1..8),
            Piece::Captured(8..201)
        ]
    );
    let head: i32 = u[1..8].iter().map(unit_nodes).sum();
    assert!(head >= want && head - unit_nodes(&u[7]) < want, "{head}");

    // Under SPLIT_MIN_NODES: whole.
    let small: Vec<Unit> = (0..10).map(|k| Segment(10 * k, 10 * k + 10)).collect();
    let p = pieces(&small, |_| true);
    assert_eq!(split_heads(p.clone(), &small), p);

    // One unit holding everything cannot be cut (segments are ggml's to fuse).
    let one = [Segment(0, SPLIT_MIN_NODES * 2)];
    let p = pieces(&one, |_| true);
    assert_eq!(split_heads(p.clone(), &one), p);

    // A first unit big enough to be the head alone.
    let two = [Segment(0, 600), External(600)];
    assert_eq!(
        split_heads(pieces(&two, |_| true), &two),
        vec![Piece::Captured(0..1), Piece::Captured(1..2)]
    );
}

/// The split keeps every unit in exactly one piece, in order, whatever the mask.
#[test]
fn split_heads_cover_every_unit_once() {
    use Unit::*;
    let u: Vec<Unit> = (0..12)
        .map(|i| match i % 4 {
            0 => Segment(100 * i, 100 * i + 99),
            1 => External(100 * i + 99),
            2 => Fill,
            _ => Segment(100 * i, 100 * i + 100),
        })
        .collect();
    for mask in 0u32..1 << u.len() {
        let p = split_heads(
            pieces(&u, |x| {
                let i = u.iter().position(|y| y == x).unwrap();
                mask >> i & 1 == 1
            }),
            &u,
        );
        let order: Vec<usize> = p
            .iter()
            .flat_map(|p| match p {
                Piece::Captured(r) => {
                    assert!(!r.is_empty(), "mask {mask:b}");
                    r.clone().collect::<Vec<_>>()
                }
                Piece::Direct(i) => vec![*i],
            })
            .collect();
        assert_eq!(order, (0..u.len()).collect::<Vec<_>>(), "mask {mask:b}");
    }
}

/// A segment is captured only if ggml-cuda could capture every node of it.
#[test]
fn plan_keeps_uncapturable_segments_direct() {
    let ctx = Context::new(16, 64);
    let x = ctx.new_tensor(types::F32, &[32, 4]);
    let mut plain = ctx.new_graph(16);
    plain.expand(ctx.scale(ctx.add(x, x), 2.0));
    let mut with_mean = ctx.new_graph(16);
    // SAFETY: tensors of a live context.
    with_mean.expand(ctx.scale(
        ctx.wrap(unsafe { ffi::ggml_mean(ctx.raw(), ctx.add(x, x).raw()) }),
        2.0,
    ));
    // SAFETY: live graphs of this context.
    unsafe {
        let n = ffi::ggml_graph_n_nodes(plain.raw());
        let u = units(n, &[], Some(0));
        assert_eq!(
            plan(plain.raw(), &u, false),
            vec![Piece::Direct(0), Piece::Captured(1..2)]
        );
        let n = ffi::ggml_graph_n_nodes(with_mean.raw());
        assert_eq!(
            plan(with_mean.raw(), &units(n, &[], None), false),
            vec![Piece::Direct(0)]
        );
    }
}

/// Early captures (#827): only a small step's graph, with the setting on, on a device whose
/// small steps capture whole, and never a graph whose words changed after a single run.
#[test]
fn capture_early_only_for_small_steps_of_steady_graphs() {
    // New to its entry, or its last words ran twice or more: early.
    assert!(capture_early(true, true, true, true, 0));
    assert!(capture_early(true, true, true, false, 1));
    assert!(capture_early(true, true, true, false, 7));
    // Its words changed after one run (they may change every run): as before.
    assert!(!capture_early(true, true, true, false, 0));
    // Turned off, a prompt step, or an older GPU (the AC922's V100s): as before.
    assert!(!capture_early(false, true, true, true, 3));
    assert!(!capture_early(true, false, true, true, 3));
    assert!(!capture_early(true, true, false, true, 3));
}

/// A step's mark comes back when its scope ends, nested or not.
#[test]
fn early_scope_restores_the_mark() {
    let before = SMALL_STEP.load(Ordering::Relaxed);
    {
        let _a = EarlyScope::enter(true);
        assert!(SMALL_STEP.load(Ordering::Relaxed));
        {
            let _b = EarlyScope::enter(false);
            assert!(!SMALL_STEP.load(Ordering::Relaxed));
        }
        assert!(SMALL_STEP.load(Ordering::Relaxed));
    }
    assert_eq!(SMALL_STEP.load(Ordering::Relaxed), before);
}

/// Captures on reuse (#836): a marked graph is captured behind its second run, where an early
/// capture would apply, only while it has no captures, none was refused and no device
/// condition is to hold it.
#[test]
fn capture_on_reuse_only_for_marked_graphs_run_again() {
    let c = capture_on_reuse;
    // Marked, run again unchanged (or later, its captures lost): behind this run.
    assert!(c(true, true, true, true, 1, false, false, false));
    assert!(c(true, true, true, true, 5, false, false, false));
    // Its first run (or changed): directly, not captured.
    assert!(!c(true, true, true, true, 0, false, false, false));
    // Not marked: as before (captured behind its first run, or at its second).
    assert!(!c(true, true, true, false, 1, false, false, false));
    // Captured already, a capture refused, or behind a device condition: as before.
    assert!(!c(true, true, true, true, 1, true, false, false));
    assert!(!c(true, true, true, true, 1, false, true, false));
    assert!(!c(true, true, true, true, 1, false, false, true));
    // Early captures off, a prompt step, or an older GPU: as before.
    assert!(!c(false, true, true, true, 1, false, false, false));
    assert!(!c(true, false, true, true, 1, false, false, false));
    assert!(!c(true, true, false, true, 1, false, false, false));
}

/// A marked graph is never captured early; the mark comes back when its scope ends, and the
/// setting turns it off.
#[test]
fn reuse_scope_holds_early_captures_back() {
    let before = REUSE_STEP.load(Ordering::Relaxed);
    {
        let _a = ReuseScope::enter(true);
        assert!(reuse_marked());
        {
            let _b = ReuseScope::enter(false);
            assert!(!reuse_marked());
        }
        assert!(reuse_marked());
        set_reuse_capture(false);
        assert!(!reuse_marked(), "turned off");
        set_reuse_capture(true);
    }
    assert_eq!(REUSE_STEP.load(Ordering::Relaxed), before);
}

/// A CPU stand-in for an external op's kernel (the graphs below are only planned).
unsafe fn no_kernel(_: &ffi::ggml_tensor, _: usize, _: usize) {}

/// Named as a W4A4 op, which a capture holds in small steps only.
static SMALL_ONLY: ExternalOp = ExternalOp {
    name: "nvfp4_a4_mm",
    cpu: no_kernel,
    cuda: None,
};

static OTHER_OP: ExternalOp = ExternalOp {
    name: "stage_graph_test_op",
    cpu: no_kernel,
    cuda: None,
};

/// #1176: a prompt step's graph holding a W4A4 op runs all its units directly; a small step's,
/// or a graph without one, plans its pieces as before.
#[test]
fn prompt_graphs_with_small_only_ops_run_directly() {
    set_direct_prompt_ops(crate::model::core::stage_ops::direct_in_prompt);
    let ctx = Context::new(16, 64);
    let x = ctx.new_tensor(types::F32, &[32, 4]);
    let mut a4 = ctx.new_graph(16);
    a4.expand(ctx.scale(
        ctx.external(&SMALL_ONLY, types::F32, [32, 4, 1, 1], &[x]),
        2.0,
    ));
    let mut other = ctx.new_graph(16);
    other.expand(ctx.scale(
        ctx.external(&OTHER_OP, types::F32, [32, 4, 1, 1], &[x]),
        2.0,
    ));
    // SAFETY: live graphs of this context.
    unsafe {
        let ext = |g: *mut ffi::ggml_cgraph| -> Vec<i32> {
            (0..ffi::ggml_graph_n_nodes(g))
                .filter(|&i| op_of(ffi::ggml_graph_node(g, i)).is_some())
                .collect()
        };
        let (a4_ext, other_ext) = (ext(a4.raw()), ext(other.raw()));
        assert_eq!(a4_ext.len(), 1);
        assert!(direct_in(false, a4.raw(), &a4_ext));
        assert!(!direct_in(true, a4.raw(), &a4_ext));
        assert!(!direct_in(false, other.raw(), &other_ext));
        assert!(!direct_in(false, a4.raw(), &[]));
    }
}
