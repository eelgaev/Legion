use super::{Span, covered, peak, spans};
use crate::ggml::{Backend, ComputeAllocator, Context, types};

fn span(first: usize, last: usize, start: usize, end: usize) -> Span {
    Span {
        first,
        last,
        start,
        end,
    }
}

#[test]
fn no_spans_no_peak() {
    assert_eq!(peak(&[]), None);
    assert_eq!(covered(std::iter::empty()), 0);
}

/// The peak is the span ending highest; alive then are the spans made by its node and not yet
/// read for the last time, biggest first.
#[test]
fn peak_lists_what_is_alive_at_the_highest_end() {
    let spans = [
        span(0, 5, 0, 100),   // an input read until node 5
        span(1, 2, 100, 300), // dead after node 2
        span(3, 4, 100, 250), // reuses the dead one's memory
        span(4, 6, 250, 400), // ends highest, made by node 4
        span(5, 6, 100, 110), // made after the peak
    ];
    let (top, live) = peak(&spans).unwrap();
    assert_eq!(top, 3);
    assert_eq!(live, vec![2, 3, 0]);
    assert_eq!(covered(live.iter().map(|&i| spans[i])), 400);
}

/// Of spans ending equally high, the first; spans of the same size keep their order.
#[test]
fn ties_take_the_first() {
    let spans = [span(1, 2, 64, 128), span(2, 3, 64, 128), span(0, 3, 0, 64)];
    let (top, live) = peak(&spans).unwrap();
    assert_eq!(top, 0);
    assert_eq!(live, vec![0, 2]);
}

/// An output stays alive to the graph's end; a tensor no node reads is alive at its own node.
#[test]
fn outputs_and_unread_tensors() {
    let spans = [span(0, 9, 0, 32), span(7, 7, 32, 96)];
    let (top, live) = peak(&spans).unwrap();
    assert_eq!(top, 1);
    assert_eq!(live, vec![1, 0]);
}

/// Bytes shared by spans (an in-place node and its source) count once.
#[test]
fn covered_counts_overlaps_once() {
    let spans = [
        span(0, 1, 0, 100),
        span(1, 2, 0, 100),
        span(1, 2, 50, 150),
        span(1, 2, 200, 300),
        span(1, 2, 300, 300),
    ];
    assert_eq!(covered(spans.into_iter()), 250);
}

/// A tensor stays alive to the graph's end where ggml never frees it: one an unread output
/// view shares (not itself an output), and an unread node; a tensor read is alive to its last
/// reader.
#[test]
fn spans_keep_what_ggml_never_frees() {
    let buft = Backend::cpu().unwrap().buffer_type();
    let ctx = Context::new(64, 64);
    let x = ctx.new_tensor(types::F32, &[64]).as_input();
    let a = ctx.scale(x, 2.0); // node 0
    let view = ctx.view_1d(a, 32, 0).as_output(); // node 1: the last read of `a`
    let b = ctx.scale(x, 3.0); // node 2
    let c = ctx.add(b, b); // node 3: no node reads it
    let mut graph = ctx.new_graph(16);
    graph.expand(view);
    graph.expand(c);
    ctx.alloc_graph(&graph, ComputeAllocator::new(buft))
        .unwrap();
    // SAFETY: an allocated graph and its node.
    let (buffers, spans) = unsafe { spans(graph.raw(), &[a.raw()]) };
    assert_eq!(buffers.len(), 1);
    let last = |t: crate::ggml::Tensor<'_>| {
        (spans.iter())
            .find(|s| s.1 == t.raw())
            .map(|s| (s.2.first, s.2.last))
    };
    assert_eq!(last(a), Some((0, 4)), "the output view keeps it");
    assert_eq!(last(b), Some((2, 3)), "read last by node 3");
    assert_eq!(last(c), Some((3, 4)), "never read: never freed");
}
