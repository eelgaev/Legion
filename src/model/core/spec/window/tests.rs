use super::*;
use crate::ggml::{Backend, ComputeAllocator};

/// The op over a `[len, rows, n]` pool into a `[w, rows, n]` state on the CPU: each slot keeping
/// `k > 0` gets its window from `k` on, the others keep what they held.
#[test]
fn cpu_window_copies_each_slot_from_its_own_keep() {
    let (len, w, rows) = (7i64, 3i64, 5i64);
    let keep = [2i32, 0, 4, 1];
    let n = keep.len() as i64;
    let ctx = Context::new(8, 8);
    let src = ctx.new_tensor(types::F32, &[len, rows, n]).as_input();
    let dst = ctx.new_tensor(types::F32, &[w, rows, n]).as_input();
    let keep_t = ctx.new_tensor(types::I32, &[n]).as_input();
    let mut graph = ctx.new_graph(8);
    graph.expand(window(&ctx, src, dst, keep_t).as_output());
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(3);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let s: Vec<f32> = (0..src.elements()).map(|i| i as f32).collect();
    let d0: Vec<f32> = (0..dst.elements()).map(|i| -(i as f32) - 1.0).collect();
    src.set_f32(&s);
    dst.set_f32(&d0);
    keep_t.set_i32(&keep);
    backend.compute(&graph).unwrap();
    let out = dst.to_f32();
    let (len, w, rows) = (len as usize, w as usize, rows as usize);
    for (slot, &k) in keep.iter().enumerate() {
        for r in 0..rows {
            for j in 0..w {
                let at = (slot * rows + r) * w + j;
                let want = if k == 0 {
                    d0[at]
                } else {
                    s[(slot * rows + r) * len + k as usize + j]
                };
                assert_eq!(out[at], want, "slot {slot} row {r} element {j}");
            }
        }
    }
}

/// The op takes F32 rows of one width and slot count and one I32 keep count per slot.
#[test]
fn window_shapes() {
    let ctx = Context::new(8, 0);
    let src = ctx.new_tensor(types::F32, &[7, 5, 4]);
    let dst = ctx.new_tensor(types::F32, &[3, 5, 4]);
    let keep = ctx.new_tensor(types::I32, &[4]);
    assert!(fits(src, dst, keep));
    assert!(!fits(src, dst, ctx.new_tensor(types::I32, &[3])));
    assert!(!fits(src, dst, ctx.new_tensor(types::F32, &[4])));
    assert!(!fits(src, ctx.new_tensor(types::F32, &[3, 6, 4]), keep));
    assert!(!fits(ctx.new_tensor(types::F32, &[2, 5, 4]), dst, keep));
}
