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

/// The one-launch op over three layers' pools (tensors of another context, as the pools are)
/// lands the bytes one [`window`] per layer lands, on the CPU.
#[test]
fn cpu_windows_copy_every_layer_as_one_window_each() {
    let (len, w, rows, layers) = (7i64, 3i64, 5i64, 3usize);
    let keep = [2i32, 0, 4, 1];
    let n = keep.len() as i64;
    let backend_ty = Backend::cpu().unwrap();
    let pools = Context::new(2 * 2 * layers + 1, 0);
    let mk = |ne: &[i64]| pools.new_tensor(types::F32, ne);
    let one: Vec<_> = (0..layers)
        .map(|_| (mk(&[len, rows, n]), mk(&[w, rows, n])))
        .collect();
    let all: Vec<_> = (0..layers)
        .map(|_| (mk(&[len, rows, n]), mk(&[w, rows, n])))
        .collect();
    let longer = mk(&[len + 1, rows, n]);
    pools.alloc_tensors(backend_ty.buffer_type()).unwrap();
    for (l, (&(s1, d1), &(s2, d2))) in one.iter().zip(&all).enumerate() {
        let s: Vec<f32> = (0..s1.elements()).map(|i| (i + 1000 * l) as f32).collect();
        let d: Vec<f32> = (0..d1.elements()).map(|i| -(i as f32) - 1.0).collect();
        s1.set_f32(&s);
        s2.set_f32(&s);
        d1.set_f32(&d);
        d2.set_f32(&d);
    }
    let ctx = Context::new(64, 64);
    let keep_t = ctx.new_tensor(types::I32, &[n]).as_input().as_output();
    let mut graph = ctx.new_graph(64);
    for &(s, d) in &one {
        graph.expand(window(&ctx, s, d, keep_t).as_output());
    }
    assert!(fits_all(&all, keep_t));
    let (node, table) = windows(&ctx, &all, keep_t);
    graph.expand(node.as_output());
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(3);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    fill_table(table, &all);
    keep_t.set_i32(&keep);
    backend.compute(&graph).unwrap();
    for (l, (&(_, d1), &(_, d2))) in one.iter().zip(&all).enumerate() {
        assert_eq!(d2.to_f32(), d1.to_f32(), "layer {l}");
    }
    // Pairs of another shape do not go in one launch.
    assert!(fits(longer, all[1].1, keep_t));
    assert!(!fits_all(&[all[0], (longer, all[1].1)], keep_t));
    assert!(!fits_all(&[], keep_t));
}
