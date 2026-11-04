use super::*;

#[test]
fn nodes_rebuild_the_ring_layout_of_the_prefetch() {
    // Maya-S: 4096-wide input, 288 experts, an expert's gate, up and down about 7 MiB.
    let l = RingLayout::new(4, (7 << 20) + 100, 4096, 288);
    let p = [1, 4, l.slot as i32, 4096, 288, 0, 0, 0];
    assert_eq!(layout_of(&p), l);
}

#[test]
fn cpu_cached_is_the_cached_experts() {
    use crate::ggml::{Backend, ComputeAllocator};
    let mut b = Backend::cpu().unwrap();
    let ctx = Context::new(4, 32);
    let cached = ctx.new_tensor(types::I32, &[6]);
    ctx.alloc_tensors(b.buffer_type()).unwrap();
    cached.set_i32(&[1, 0, 0, 1, 0, 1]);
    let out = ctx.external_with(&CACHED, types::I32, [6, 1, 1, 1], &[cached], &[0; 8]);
    let mut g = ctx.new_graph(8);
    g.expand(out);
    ctx.alloc_graph(&g, ComputeAllocator::new(b.buffer_type()))
        .unwrap();
    b.compute(&g).unwrap();
    assert_eq!(out.to_i32(), [1, 0, 0, 1, 0, 1]);
}
