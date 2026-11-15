//! A graph reading another context's rows in place.

use crate::ggml::{Backend, ComputeAllocator, Context, types};

/// Rows of a tensor allocated in one context, gathered by a graph of another context through
/// [`Context::over_rows`]: the same floats as picking them on the host.
#[test]
fn a_graph_gathers_rows_of_another_context() {
    let mut backend = Backend::cpu().unwrap();
    let (width, rows) = (6usize, 5usize);
    let src_ctx = Context::new(1, 0);
    let src = src_ctx.new_tensor(types::F32, &[width as i64, rows as i64]);
    src_ctx.alloc_tensors(backend.buffer_type()).unwrap();
    let vals: Vec<f32> = (0..width * rows).map(|i| i as f32 * 0.25 - 3.0).collect();
    src.set_f32(&vals);
    let at = src.rows_at().expect("an allocated F32 matrix");
    assert_eq!((at.width, at.rows, at.device), (width, rows, None));

    let ctx = Context::new(8, 8);
    // SAFETY: `src_ctx` (and its buffer) outlives the graph, which only reads it.
    let over = unsafe { ctx.over_rows(&at) };
    let ids = ctx.new_tensor(types::I32, &[3]).as_input();
    let out = ctx.get_rows(over, ids).as_output();
    let mut graph = ctx.new_graph(8);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    ids.set_i32(&[4, 1, 2]);
    backend.compute(&graph).unwrap();
    let want: Vec<f32> = [4, 1, 2]
        .iter()
        .flat_map(|&r| vals[r * width..(r + 1) * width].to_vec())
        .collect();
    assert_eq!(out.to_f32(), want);
    // The source is untouched.
    assert_eq!(src.to_f32(), vals);
}

/// Only allocated, contiguous F32 matrices have rows to hand on.
#[test]
fn rows_at_needs_an_allocated_f32_matrix() {
    let backend = Backend::cpu().unwrap();
    let ctx = Context::new(4, 0);
    let f16 = ctx.new_tensor(types::F16, &[4, 2]);
    let cube = ctx.new_tensor(types::F32, &[4, 2, 2]);
    let unallocated = Context::new(1, 0);
    let none = unallocated.new_tensor(types::F32, &[4, 2]);
    ctx.alloc_tensors(backend.buffer_type()).unwrap();
    assert!(f16.rows_at().is_none());
    assert!(cube.rows_at().is_none());
    assert!(none.rows_at().is_none());
}

/// A range of one matrix's rows copied into a range of another's ([`Backend::copy_rows`]):
/// those floats move, the rest of both stay.
#[test]
fn copy_rows_moves_a_slice_of_rows() {
    let backend = Backend::cpu().unwrap();
    let width = 4usize;
    let ctx = Context::new(2, 0);
    let src = ctx.new_tensor(types::F32, &[width as i64, 5]);
    let dst = ctx.new_tensor(types::F32, &[width as i64, 3]);
    ctx.alloc_tensors(backend.buffer_type()).unwrap();
    let vals: Vec<f32> = (0..width * 5).map(|i| i as f32 + 0.5).collect();
    src.set_f32(&vals);
    dst.set_f32(&vec![-1.0; width * 3]);
    let (s, d) = (src.rows_at().unwrap(), dst.rows_at().unwrap());
    assert_eq!(s.slice(2, 2).nbytes(), 2 * width * 4);
    // SAFETY: both tensors stay allocated; the CPU copy is done when it returns.
    unsafe { backend.copy_rows(&s.slice(2, 2), &d.slice(1, 2)) };
    let mut want = vec![-1.0; width];
    want.extend_from_slice(&vals[2 * width..4 * width]);
    assert_eq!(dst.to_f32(), want);
    assert_eq!(src.to_f32(), vals);
}

/// A slice past the rows is a bug.
#[test]
#[should_panic(expected = "rows 4+2 of 5 rows")]
fn a_slice_past_the_rows_panics() {
    let backend = Backend::cpu().unwrap();
    let ctx = Context::new(1, 0);
    let t = ctx.new_tensor(types::F32, &[3, 5]);
    ctx.alloc_tensors(backend.buffer_type()).unwrap();
    let _ = t.rows_at().unwrap().slice(4, 2);
}
