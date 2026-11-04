use super::*;
use crate::ggml::{Backend, KeptParts, types};

/// A take-in shape of `n_tok` rows for one sequence.
fn shape(n_tok: usize) -> Vec<GroupShape> {
    vec![GroupShape {
        slot: 0,
        n_seqs: 1,
        n_tok,
        n_kv: 0,
        table: 4,
        ring: 0,
        logits: vec![1],
        last_only: true,
        verify: false,
        n_pool: 0,
        n_new: 0,
        pick_all: false,
        kv_slots: Vec::new(),
        par: false,
        pair: false,
        img: 0,
        captures: Vec::new(),
    }]
}

/// Builds `y = 2 x + 1` over `n` floats (the input first, then the output) and keeps it.
fn build(graphs: &mut MtpGraphs, backend: &Backend, n: usize) {
    graphs.evict_for_one();
    let ctx = Context::new(8, 8);
    let parts = {
        let x = ctx.new_tensor(types::F32, &[n as i64]).as_input();
        let y = ctx.scale_bias(x, 2.0, 1.0).as_output();
        let mut graph = ctx.new_graph(8);
        graph.expand(y);
        graphs.alloc(&ctx, &graph, backend.buffer_type()).unwrap();
        KeptParts::new(&graph, &[x, y])
    };
    // SAFETY: the graph reads no tensor outside its own context.
    let kept = unsafe { KeptGraph::new(ctx, parts) };
    graphs.push(shape(n), kept);
}

/// Runs the kept graph of `n` floats on `x = base, base + 1, ...` and checks its output.
fn run(graphs: &mut MtpGraphs, backend: &mut Backend, n: usize, base: f32) {
    assert!(graphs.touch(&shape(n)), "{n} rows not kept");
    let kept = graphs.get(&shape(n)).unwrap();
    let x: Vec<f32> = (0..n).map(|i| base + i as f32).collect();
    kept.tensor(0).set_f32(&x);
    backend.compute_kept(kept).unwrap();
    let want: Vec<f32> = x.iter().map(|v| 2.0 * v + 1.0).collect();
    assert_eq!(kept.tensor(1).to_f32(), want, "{n} rows");
}

/// Shapes the accepted count alternates between keep their graphs, in one buffer: running one
/// after another (each setting its inputs) gives every one its own right result.
#[test]
fn alternating_shapes_reuse_their_graphs_in_one_buffer() {
    let mut backend = Backend::cpu().unwrap();
    let mut graphs = MtpGraphs::new(MTP_KEPT);
    // The largest first, so the buffer never grows after it.
    for n in [64, 3, 2, 4] {
        build(&mut graphs, &backend, n);
    }
    assert_eq!((graphs.len(), graphs.builds()), (4, 4));
    let size = graphs.buffer_size();
    assert!(size >= 64 * 4, "{size} bytes");
    for (i, n) in [3, 2, 4, 2, 3, 64, 3].into_iter().enumerate() {
        run(&mut graphs, &mut backend, n, i as f32 * 10.0);
    }
    assert_eq!(graphs.builds(), 4, "no rebuilds");
    assert_eq!(graphs.buffer_size(), size, "one buffer, not one per graph");
}

/// Past the cap the least recently used shape goes.
#[test]
fn evicts_the_least_recently_used() {
    let mut backend = Backend::cpu().unwrap();
    let mut graphs = MtpGraphs::new(2);
    build(&mut graphs, &backend, 8);
    build(&mut graphs, &backend, 4);
    run(&mut graphs, &mut backend, 8, 1.0);
    build(&mut graphs, &backend, 2);
    assert!(graphs.get(&shape(8)).is_some());
    assert!(graphs.get(&shape(4)).is_none());
    assert!(graphs.get(&shape(2)).is_some());
    assert_eq!(graphs.len(), 2);
}

/// A graph bigger than the buffer moves it: the graphs kept before point into the old one and
/// go; the new one runs.
#[test]
fn a_growing_buffer_drops_the_older_graphs() {
    let mut backend = Backend::cpu().unwrap();
    let mut graphs = MtpGraphs::new(MTP_KEPT);
    build(&mut graphs, &backend, 2);
    build(&mut graphs, &backend, 1);
    let small = graphs.buffer_size();
    build(&mut graphs, &backend, 4096);
    assert!(graphs.buffer_size() > small);
    assert_eq!(graphs.len(), 1);
    run(&mut graphs, &mut backend, 4096, 0.5);
    graphs.clear();
    assert_eq!((graphs.len(), graphs.buffer_size()), (0, 0));
}

/// What `take` hands out (#466) drops on another thread, and the emptied graphs build and run
/// again from a buffer of their own.
#[test]
fn taken_graphs_drop_elsewhere_and_the_kind_starts_over() {
    let mut backend = Backend::cpu().unwrap();
    let mut graphs = MtpGraphs::new(MTP_KEPT);
    for n in [64, 3] {
        build(&mut graphs, &backend, n);
    }
    let taken = graphs.take();
    assert_eq!((graphs.len(), graphs.buffer_size()), (0, 0));
    assert!(!graphs.touch(&shape(3)), "a taken shape still kept");
    std::thread::spawn(move || drop(taken)).join().unwrap();
    build(&mut graphs, &backend, 3);
    run(&mut graphs, &mut backend, 3, 5.0);
    assert_eq!((graphs.len(), graphs.builds()), (1, 3));
}

/// The buffer kept through an expert-cache lend (#491): taken out with the graphs dropped, it is
/// what the next graphs allocate from once given back, without growing when they fit.
#[test]
fn a_kept_buffer_is_where_the_next_graphs_allocate() {
    let mut backend = Backend::cpu().unwrap();
    let mut graphs = MtpGraphs::new(MTP_KEPT);
    build(&mut graphs, &backend, 64);
    let size = graphs.buffer_size();
    let spare = graphs.take_buffer().expect("a buffer");
    assert_eq!(spare.buffer_size(), size);
    assert_eq!((graphs.len(), graphs.buffer_size()), (0, 0));
    assert!(graphs.take_buffer().is_none(), "nothing left to take");
    graphs.start_with(spare);
    assert_eq!(graphs.buffer_size(), size);
    for n in [3, 2] {
        build(&mut graphs, &backend, n);
        run(&mut graphs, &mut backend, n, 7.0);
    }
    assert_eq!(
        graphs.buffer_size(),
        size,
        "a graph that fit grew the buffer"
    );
    assert_eq!(graphs.len(), 2);
}

/// #836: only a take-in of several sequences waits for its graph's reuse to be captured.
#[test]
fn only_take_ins_of_several_sequences_wait_for_reuse() {
    // One sequence's take-in, and one sequence's draft pass: captured behind the first run.
    assert!(!take_in_waits_for_reuse(&shape(5)));
    assert!(!take_in_waits_for_reuse(&shape(1)));
    // Four sequences' take-in, each its own rows (one group each).
    let four: Vec<GroupShape> = [2, 3, 5, 9].into_iter().flat_map(shape).collect();
    assert!(take_in_waits_for_reuse(&four));
    // Four sequences' draft pass (one row each): its shape follows the drafting count only.
    let drafts: Vec<GroupShape> = (0..4).flat_map(|_| shape(1)).collect();
    assert!(!take_in_waits_for_reuse(&drafts));
    // One group of two sequences taking in the same rows.
    let mut two = shape(4);
    two[0].n_seqs = 2;
    assert!(take_in_waits_for_reuse(&two));
}
