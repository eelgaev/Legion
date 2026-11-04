//! The recorder op on the CPU: a step's router ids (a strided view) land in the file as
//! records that [`crate::tp_net::ep_place::Recorded`] reads back.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::tp_net::ep_place::Recorded;

#[test]
fn records_a_steps_ids() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("routes.bin");
    let at = sink(&path).unwrap();
    let mut b = Backend::cpu().unwrap();
    let ctx = Context::new(8, 16);
    // Four ids per token as the router's argsort leaves them; the first two are the top-k.
    let all = ctx.new_tensor(types::I32, &[4, 3]).as_input();
    let sel = ctx.view_2d(all, [2, 3], all.nb()[1], 0);
    let mut g = ctx.new_graph(16);
    for il in [5, 7] {
        g.expand(record_to(&ctx, at, il, 8, sel));
    }
    ctx.alloc_graph(&g, ComputeAllocator::new(b.buffer_type()))
        .unwrap();
    all.set_i32(&[3, 1, 0, 0, 6, 2, 0, 0, 7, 4, 0, 0]);
    b.compute(&g).unwrap();
    b.compute(&g).unwrap();
    let r = Recorded::parse(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(r.n_expert, 8);
    assert_eq!(r.layers.keys().copied().collect::<Vec<_>>(), [5, 7]);
    for rows in r.layers.values() {
        assert_eq!(rows.k, 2);
        assert_eq!(rows.ids, [3, 1, 6, 2, 7, 4, 3, 1, 6, 2, 7, 4]);
    }
}
