//! The token shares, BF16 rounding, and the reduce-scatter / all-gather (and the BF16
//! all-reduce) over loopback connections, as host calls and as graph ops on the CPU.

use std::sync::Arc;
use std::time::Duration;

use super::super::comm::{bf16_bits, decode, encode};
use super::super::tests::comms;
use super::super::{Comm, Wire, register};
use super::*;
use crate::ggml::{Backend, ComputeAllocator, Context, types};

#[test]
fn shares_cover_the_columns_in_rank_order() {
    for cols in [0, 1, 2, 3, 7, 4096, 4097] {
        for n in 1..=4 {
            let mut next = 0;
            for r in 0..n {
                let (a, c) = share(cols, r, n);
                assert_eq!(a, next, "{cols} cols, {n} ranks, rank {r}");
                assert!(c.abs_diff(cols / n) <= 1);
                next = a + c;
            }
            assert_eq!(next, cols);
        }
    }
    assert_eq!(share(4097, 0, 2), (0, 2048));
    assert_eq!(share(4097, 1, 2), (2048, 2049));
}

#[test]
fn bf16_rounds_to_nearest_even() {
    let f = |b: u32| f32::from_bits(b);
    assert_eq!(bf16_bits(1.0), 0x3f80);
    assert_eq!(bf16_bits(-2.5), 0xc020);
    // Halfway: to the even neighbour, down then up.
    assert_eq!(bf16_bits(f(0x3f80_8000)), 0x3f80);
    assert_eq!(bf16_bits(f(0x3f81_8000)), 0x3f82);
    // Above halfway: up.
    assert_eq!(bf16_bits(f(0x3f80_8001)), 0x3f81);
    // The largest finite F32 overflows to infinity, as CUDA's rounding does.
    assert_eq!(bf16_bits(f32::MAX), 0x7f80);
    assert_eq!(bf16_bits(f32::INFINITY), 0x7f80);
    assert!(f32::from_bits((bf16_bits(f32::NAN) as u32) << 16).is_nan());
    let x = [1.0f32, -3.0e-5, 65504.0, 7.1];
    let b = encode(&x, Wire::Bf16);
    assert_eq!(b.len(), 8);
    for (i, &v) in x.iter().enumerate() {
        assert!((decode(&b, i, Wire::Bf16) - v).abs() <= v.abs() / 128.0);
        assert_eq!(decode(&encode(&x, Wire::F32), i, Wire::F32), v);
    }
}

/// Rank `r`'s values for exchange `k`: their sum depends on the order of the adds.
fn values(r: usize, k: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = ((i * 7 + k * 13 + r * 31) % 97) as f32;
            if r.is_multiple_of(2) {
                x * 1e7
            } else {
                x * 0.3 + 1e-3
            }
        })
        .collect()
}

/// As the wire carries it.
fn wired(x: &[f32], wire: Wire) -> Vec<f32> {
    let b = encode(x, wire);
    (0..x.len()).map(|i| decode(&b, i, wire)).collect()
}

/// Rank-order sum of every rank's (wired) values.
fn want_sum(n: usize, k: usize, len: usize, wire: Wire) -> Vec<f32> {
    let parts: Vec<Vec<f32>> = (0..n).map(|r| wired(&values(r, k, len), wire)).collect();
    (0..len)
        .map(|i| {
            let mut acc = parts[0][i];
            for p in &parts[1..] {
                acc += p[i];
            }
            acc
        })
        .collect()
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// Runs `f(rank, comm)` on every rank of `n` at once.
fn on_ranks<T: Send + 'static>(
    n: usize,
    wire: Wire,
    f: impl Fn(usize, Arc<Comm>) -> T + Send + Sync + 'static,
) -> Vec<T> {
    let f = Arc::new(f);
    let threads: Vec<_> = comms(n, Duration::from_secs(20))
        .into_iter()
        .enumerate()
        .map(|(r, c)| {
            c.set_collectives(wire, 1);
            if wire == Wire::Bf16 {
                // Every exchange, however small.
                c.set_bf16_min(0);
            }
            let f = f.clone();
            std::thread::spawn(move || f(r, c))
        })
        .collect();
    threads.into_iter().map(|t| t.join().unwrap()).collect()
}

/// Row lengths and column counts, from fewer columns than ranks' worth to uneven shares.
const SHAPES: [(usize, usize); 4] = [(3, 5), (1, 3), (16, 64), (7, 101)];

fn collectives_match(n: usize, wire: Wire) {
    let got = on_ranks(n, wire, move |r, c| {
        let mut out = Vec::new();
        for (k, &(row, cols)) in SHAPES.iter().enumerate() {
            let len = row * cols;
            let (a, m) = share(cols, r, n);
            let mut rs = vec![0f32; m * row];
            c.reduce_scatter_host(&values(r, k, len), &mut rs, row)
                .unwrap();
            let mut ag = vec![0f32; len];
            let mine = values(r + 10, k, len)[a * row..(a + m) * row].to_vec();
            c.all_gather_host(&mine, &mut ag, row).unwrap();
            let mut ar = vec![0f32; len];
            c.all_reduce_host(&values(r, k, len), &mut ar).unwrap();
            out.push((rs, ag, ar));
        }
        assert_eq!(c.all_reduces(), 3 * SHAPES.len() as u32);
        out
    });
    for (k, &(row, cols)) in SHAPES.iter().enumerate() {
        let len = row * cols;
        let sum = want_sum(n, k, len, wire);
        // The gathered tensor: each rank's own columns of its values `r + 10`.
        let gathered: Vec<f32> = (0..n)
            .flat_map(|r| {
                let (a, m) = share(cols, r, n);
                wired(&values(r + 10, k, len)[a * row..(a + m) * row], wire)
            })
            .collect();
        for (r, res) in got.iter().enumerate() {
            let (a, m) = share(cols, r, n);
            let (rs, ag, ar) = &res[k];
            assert_eq!(
                bits(rs),
                bits(&sum[a * row..(a + m) * row]),
                "rs {k} rank {r}"
            );
            assert_eq!(bits(ag), bits(&gathered), "ag {k} rank {r}");
            assert_eq!(bits(ar), bits(&sum), "ar {k} rank {r}");
        }
    }
}

#[test]
fn two_ranks_scatter_and_gather_f32() {
    collectives_match(2, Wire::F32);
}

#[test]
fn three_ranks_scatter_and_gather_bf16() {
    collectives_match(3, Wire::Bf16);
}

#[test]
fn bf16_wire_rounds_every_share() {
    // A value BF16 rounds: the sum of two ranks' 1 + 2^-10 is 2, not 2 + 2^-9.
    let got = on_ranks(2, Wire::Bf16, |_, c| {
        let mut out = [0f32; 1];
        c.all_reduce_host(&[1.0 + 1.0 / 1024.0], &mut out).unwrap();
        out[0]
    });
    assert_eq!(got, vec![2.0, 2.0]);
}

#[test]
fn bf16_wire_leaves_small_sums_f32() {
    // `--tp-wire bf16` from 4 values: 3 values travel in F32, 4 in BF16.
    let x = 1.0 + 1.0 / 1024.0;
    let got = on_ranks(2, Wire::F32, move |_, c| {
        c.set_collectives(Wire::Bf16, 0);
        let standard = c.wire(super::super::BF16_MIN_VALUES - 1);
        c.set_bf16_min(4);
        let (mut small, mut big) = ([0f32; 3], [0f32; 4]);
        c.all_reduce_host(&[x; 3], &mut small).unwrap();
        c.all_reduce_host(&[x; 4], &mut big).unwrap();
        (standard, small, big)
    });
    for (standard, small, big) in got {
        assert_eq!(standard, Wire::F32);
        assert_eq!(small, [2.0 * x; 3]);
        assert_eq!(big, [2.0; 4]);
    }
}

#[test]
fn the_graph_ops_scatter_and_gather_on_the_cpu() {
    // [4, 5] partials: rank 0 gets columns 0..2 of the sum, rank 1 columns 2..5; the
    // gather of them gives every rank the whole sum.
    let (row, cols) = (4usize, 5usize);
    let got = on_ranks(2, Wire::F32, move |r, c| {
        let handle = register(c);
        let mut backend = Backend::cpu().unwrap();
        backend.set_threads(3);
        let ctx = Context::new(16, 16);
        let x = ctx
            .new_tensor(types::F32, &[row as i64, cols as i64])
            .as_input();
        let part = reduce_scatter(&ctx, handle.id(), x);
        let y = all_gather(&ctx, handle.id(), ctx.scale(part, 2.0), cols as i64).as_output();
        let mut graph = ctx.new_graph(16);
        graph.expand(y);
        ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
            .unwrap();
        x.set_f32(&values(r, 0, row * cols));
        backend.compute(&graph).unwrap();
        assert!(handle.comm().error().is_none());
        y.to_f32()
    });
    let want: Vec<f32> = want_sum(2, 0, row * cols, Wire::F32)
        .iter()
        .map(|v| v * 2.0)
        .collect();
    assert_eq!(bits(&got[0]), bits(&want));
    assert_eq!(bits(&got[1]), bits(&want));
}

#[test]
fn off_unless_set_and_big_enough() {
    let got = on_ranks(2, Wire::F32, |_, c| {
        let handle = register(c.clone());
        let id = handle.id();
        c.set_collectives(Wire::F32, 0);
        let off = active(id, 4096);
        c.set_collectives(Wire::F32, 1024);
        (off, active(id, 1023), active(id, 1024))
    });
    assert_eq!(got[0], (None, None, Some((0, 2))), "rank 0");
    assert_eq!(got[1], (None, None, Some((1, 2))));
}

#[test]
fn only_a_two_rank_bf16_sum_is_deferred_once_and_the_cpu_still_sums() {
    // #1176: a reader may take over a two-rank BF16 reduce-scatter's sum once; the CPU op
    // writes the sum anyway, so the flag changes nothing there.
    let (row, cols) = (4usize, 5usize);
    let got = on_ranks(2, Wire::Bf16, move |r, c| {
        let handle = register(c);
        let mut backend = Backend::cpu().unwrap();
        backend.set_threads(3);
        let ctx = Context::new(16, 16);
        let x = ctx
            .new_tensor(types::F32, &[row as i64, cols as i64])
            .as_input();
        let part = reduce_scatter(&ctx, handle.id(), x);
        let other = ctx.scale(part, 2.0);
        // A traced intermediate keeps its sum.
        let traced = reduce_scatter(&ctx, handle.id(), x).as_output();
        let flags = (deferrable(part), deferrable(other), deferrable(traced));
        defer_sum(part);
        let after = deferrable(part);
        let y = all_gather(&ctx, handle.id(), part, cols as i64).as_output();
        let mut graph = ctx.new_graph(16);
        graph.expand(y);
        ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
            .unwrap();
        x.set_f32(&values(r, 0, row * cols));
        backend.compute(&graph).unwrap();
        assert!(handle.comm().error().is_none());
        (flags, after, y.to_f32())
    });
    // The gather rounds every column to BF16 too.
    let want = wired(&want_sum(2, 0, row * cols, Wire::Bf16), Wire::Bf16);
    for (r, (flags, after, y)) in got.into_iter().enumerate() {
        assert_eq!(flags, (true, false, false), "rank {r}");
        assert!(!after, "rank {r}");
        assert_eq!(bits(&y), bits(&want), "rank {r}");
    }
}

#[test]
fn an_f32_wire_sum_is_not_deferred() {
    let got = on_ranks(2, Wire::F32, |_, c| {
        let handle = register(c);
        let ctx = Context::new(16, 16);
        let x = ctx.new_tensor(types::F32, &[4, 6]).as_input();
        deferrable(reduce_scatter(&ctx, handle.id(), x))
    });
    assert_eq!(got, [false, false]);
}

#[test]
fn a_reduce_scatter_takes_over_only_the_tail_it_reads() {
    let got = on_ranks(2, Wire::F32, |_, c| {
        let handle = register(c);
        let ctx = Context::new(16, 16);
        let x = ctx.new_tensor(types::F32, &[4, 6]).as_input();
        let w = ctx.new_tensor(types::F32, &[1, 6]).as_input();
        // External nodes standing in for the tail and for another node.
        let tail = all_gather(&ctx, handle.id(), x, 6);
        let other = all_gather(&ctx, handle.id(), x, 6);
        let rs = reduce_scatter(&ctx, handle.id(), tail);
        let ins = [x.raw(), w.raw()];
        // SAFETY: nodes of `ctx`, never run.
        unsafe {
            let not_rs = take_tail(other.raw(), tail.raw(), &ins);
            let not_its_source = take_tail(rs.raw(), other.raw(), &ins);
            let untouched =
                external_params(&*other.raw())[crate::model::core::node_fuse::P_BY_EXCHANGE] == 0
                    && (*rs.raw()).src[1].is_null();
            let took = take_tail(rs.raw(), tail.raw(), &ins);
            let again = take_tail(rs.raw(), tail.raw(), &ins);
            let t = &*rs.raw();
            let srcs = t.src[1] == x.raw() && t.src[2] == w.raw() && t.src[3].is_null();
            let by = external_params(&*tail.raw())[crate::model::core::node_fuse::P_BY_EXCHANGE];
            (not_rs, not_its_source, untouched, took, again, srcs, by)
        }
    });
    assert_eq!(got, [(false, false, true, true, false, true, 1); 2]);
}

#[test]
fn a_deferred_sum_read_after_another_exchange_is_written_after_all() {
    // #1176: the exchange after a deferred reduce-scatter frees its buffers and its reader
    // reads the partials' addresses from the communicator's slot, so the reader must come
    // first; otherwise `check_order` makes the node write its sum as before.
    let got = on_ranks(2, Wire::Bf16, |_, c| {
        let handle = register(c);
        let ctx = Context::new(32, 32);
        let x = ctx.new_tensor(types::F32, &[4, 6]).as_input();
        // Read right after (a view in between reads nothing).
        let good = reduce_scatter(&ctx, handle.id(), x);
        defer_sum(good);
        let read = ctx.scale(ctx.view_2d(good, [4, 3], 16, 0), 2.0).as_output();
        // Another exchange before the reader.
        let late = reduce_scatter(&ctx, handle.id(), x);
        defer_sum(late);
        let other = all_gather(&ctx, handle.id(), x, 12).as_output();
        let after = ctx.scale(late, 2.0).as_output();
        // No reader at all.
        let unread = reduce_scatter(&ctx, handle.id(), x).as_output();
        // SAFETY: a node of `ctx`, never run.
        unsafe { crate::ggml::set_external_param(unread.raw(), P_DEFERRED, 1) };
        let mut graph = ctx.new_graph(32);
        for t in [read, late, other, after, unread] {
            graph.expand(t);
        }
        // SAFETY: nodes of `ctx`, never run.
        let changed = unsafe { check_order(graph.raw()) };
        // SAFETY: as above.
        let flags = unsafe { [good, late, unread].map(|t| deferred(&*t.raw())) };
        (changed, flags)
    });
    assert_eq!(got, [(2, [true, false, false]); 2]);
}

#[test]
fn a_bf16_gather_holds_the_f32_gathers_values() {
    // #1176: with BF16 on the wire, the BF16 gather's values (and their widening) are the F32
    // gather's exactly; with F32 on the wire there is no BF16 gather.
    let (row, cols) = (4usize, 7usize);
    let got = on_ranks(2, Wire::Bf16, move |r, c| {
        let handle = register(c.clone());
        let mut backend = Backend::cpu().unwrap();
        backend.set_threads(3);
        let ctx = Context::new(16, 16);
        let (a, m) = share(cols, r, 2);
        let x = ctx
            .new_tensor(types::F32, &[row as i64, m as i64])
            .as_input();
        let g = all_gather_bf16(&ctx, handle.id(), x, cols as i64)
            .expect("BF16 on the wire")
            .as_output();
        let wide = ctx.cast(g, types::F32).as_output();
        let f = all_gather(&ctx, handle.id(), x, cols as i64).as_output();
        let mut graph = ctx.new_graph(16);
        for t in [g, wide, f] {
            graph.expand(t);
        }
        ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
            .unwrap();
        x.set_f32(&values(r, 0, row * cols)[a * row..(a + m) * row]);
        backend.compute(&graph).unwrap();
        assert!(handle.comm().error().is_none());
        let g16: Vec<u16> = g
            .bytes()
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        c.set_bf16_min(usize::MAX);
        let f32_wire = all_gather_bf16(&ctx, handle.id(), x, cols as i64).is_none();
        (g.ty(), g16, wide.to_f32(), f.to_f32(), f32_wire)
    });
    let want: Vec<f32> = (0..2)
        .flat_map(|r| {
            let (a, m) = share(cols, r, 2);
            wired(
                &values(r, 0, row * cols)[a * row..(a + m) * row],
                Wire::Bf16,
            )
        })
        .collect();
    let want16: Vec<u16> = want.iter().map(|&v| bf16_bits(v)).collect();
    for (r, (ty, g16, wide, f, f32_wire)) in got.into_iter().enumerate() {
        assert_eq!(ty, types::BF16, "rank {r}");
        assert_eq!(g16, want16, "rank {r}");
        assert_eq!(bits(&wide), bits(&want), "rank {r}");
        assert_eq!(bits(&f), bits(&want), "rank {r}");
        assert!(f32_wire, "rank {r}");
    }
}
