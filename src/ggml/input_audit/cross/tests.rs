use super::{Span, Spans, note_run, register_spans, spans, summary_line, unregister};
use crate::ggml::{Backend, ComputeAllocator, Context, types};

/// A span of `len` bytes at `at` in a band of fake addresses of its own per test (the registry
/// is shared by the tests running at the same time; bands far from any real allocation).
fn span(band: usize, at: usize, len: usize) -> Span {
    let start = (band << 40) + (1 << 60) + at;
    Span {
        name: "x".into(),
        start,
        end: start + len,
    }
}

fn graph(band: usize, held: &[(usize, usize)], writes: &[(usize, usize)]) -> Spans {
    Spans {
        held: held.iter().map(|&(a, l)| span(band, a, l)).collect(),
        writes: writes.iter().map(|&(a, l)| span(band, a, l)).collect(),
    }
}

/// Graph A holds a set-once input that graph B, kept in the same memory, writes over: the
/// pair is reported when B is kept, and A's run after a run of B is a hit, once per B run.
#[test]
fn a_set_once_input_written_by_another_graphs_run_is_a_hit() {
    let (a, b) = (u64::MAX - 10, u64::MAX - 11);
    assert!(register_spans(a, 5, graph(1, &[(0, 64)], &[(64, 64)])).is_empty());
    let pairs = register_spans(b, 7, graph(1, &[], &[(32, 128)]));
    assert_eq!(pairs.len(), 1, "{pairs:?}");
    assert!(pairs[0].contains("[x]"), "{}", pairs[0]);
    assert!(pairs[0].contains("5 nodes"), "{}", pairs[0]);
    assert!(pairs[0].contains("7 nodes"), "{}", pairs[0]);
    let sb = summary_line(b).unwrap();
    assert!(sb.contains("0 set-once input(s) []"), "{sb}");
    assert!(sb.contains("shares memory with 1 graph(s)"), "{sb}");
    assert!(sb.contains("0 of them"), "{sb}");
    let sa = summary_line(a).unwrap();
    assert!(sa.contains("1 set-once input(s) [x]"), "{sa}");
    assert!(sa.contains("1 of them"), "{sa}");
    assert!(note_run(a).is_empty(), "B has not run yet");
    assert!(note_run(b).is_empty(), "B holds nothing");
    let hits = note_run(a);
    assert_eq!(hits.len(), 1, "{hits:?}");
    assert!(hits[0].contains("STALE"), "{}", hits[0]);
    assert!(note_run(a).is_empty(), "B has not run again");
    unregister(a);
    unregister(b);
}

/// A run of the other graph before the first run of a graph kept after it is a hit too (its
/// set-once inputs were set when it was kept).
#[test]
fn a_run_between_keeping_and_the_first_run_is_a_hit() {
    let (a, b) = (u64::MAX - 20, u64::MAX - 21);
    register_spans(b, 3, graph(2, &[], &[(0, 256)]));
    assert_eq!(register_spans(a, 4, graph(2, &[(16, 16)], &[])).len(), 1);
    assert!(note_run(b).is_empty());
    assert_eq!(note_run(a).len(), 1);
    unregister(a);
    unregister(b);
}

#[test]
fn graphs_in_separate_memory_are_no_pair() {
    let (a, b) = (u64::MAX - 30, u64::MAX - 31);
    register_spans(a, 5, graph(3, &[(0, 64)], &[(64, 64)]));
    assert!(register_spans(b, 5, graph(4, &[(0, 64)], &[(0, 256)])).is_empty());
    note_run(b);
    assert!(note_run(a).is_empty());
    unregister(a);
    unregister(b);
}

/// A dropped graph is forgotten: it pairs with nothing kept after it.
#[test]
fn an_unregistered_graph_is_forgotten() {
    let (a, b, c) = (u64::MAX - 40, u64::MAX - 41, u64::MAX - 42);
    register_spans(a, 5, graph(5, &[(0, 64)], &[]));
    assert_eq!(register_spans(b, 5, graph(5, &[], &[(0, 64)])).len(), 1);
    note_run(b);
    unregister(b);
    assert!(note_run(a).is_empty(), "b is gone");
    unregister(a);
    assert!(register_spans(c, 5, graph(5, &[], &[(0, 64)])).is_empty());
    unregister(c);
}

/// `y = x * 2`, `z = y * 3`, on the CPU: the spans of the allocated graph, `x` listed or not.
fn cpu_spans(listed: bool, name: &str) -> Spans {
    let cpu = Backend::cpu().unwrap();
    let ctx = Context::new(8, 8);
    let x = ctx.new_tensor(types::F32, &[64]).named(name).as_input();
    let y = ctx.scale(x, 2.0);
    let z = ctx.scale(y, 3.0).as_output();
    let mut graph = ctx.new_graph(8);
    graph.expand(z);
    ctx.alloc_graph(&graph, ComputeAllocator::new(cpu.buffer_type()))
        .unwrap();
    let refilled = if listed { vec![x.raw()] } else { Vec::new() };
    // SAFETY: an allocated graph of live tensors.
    unsafe { spans(graph.raw(), &refilled) }
}

#[test]
fn an_input_set_once_is_held_and_the_nodes_are_writes() {
    let s = cpu_spans(false, "x");
    assert_eq!(s.held.len(), 1, "{s:?}");
    assert_eq!(s.held[0].name, "x");
    assert_eq!(s.held[0].end - s.held[0].start, 256);
    assert!(!s.writes.is_empty(), "{s:?}");
    assert!(s.writes.iter().all(|w| w.name != "x"), "{s:?}");
}

/// A refilled input is no set-once input: its caller writes it, so it is a write.
#[test]
fn a_refilled_input_is_a_write_not_held() {
    let s = cpu_spans(true, "x");
    assert!(s.held.is_empty(), "{s:?}");
    assert!(s.writes.iter().any(|w| w.name == "x"), "{s:?}");
}

#[test]
fn a_scratch_input_is_not_held() {
    assert!(cpu_spans(false, "attn_scratch-0").held.is_empty());
}
