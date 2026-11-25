use super::{Span, covered, peak};

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
