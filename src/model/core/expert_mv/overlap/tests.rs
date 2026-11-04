use super::*;
use crate::ggml::{Backend, ComputeAllocator};

/// The split pair of ops computes what the SwiGLU op followed by the down op computes, to the
/// bit, cached and missed experts alike (on the CPU, which ignores the streams and the peer):
/// the joining down op reads only the down stack, not the SwiGLU op's scratch it holds. In
/// every mode: a side stream, the peer, both.
#[test]
fn split_pair_matches_glu_then_down() {
    for mode in [MODE_SIDE, MODE_PEER, MODE_SIDE | MODE_PEER] {
        split_pair_matches(mode);
    }
}

fn split_pair_matches(mode: i32) {
    let (k, rows, n_expert, n_used, n_tok) = (256usize, 64usize, 6usize, 3usize, 2usize);
    let mk = |seed: usize, k: usize, rows: usize, ty| -> Vec<u8> {
        let wf: Vec<f32> = (0..k * rows * n_expert)
            .map(|i| (((i + seed) * 37 % 101) as f32 - 50.0) / 400.0)
            .collect();
        crate::ggml::quantize(ty, &wf, k)
    };
    let (gq, uq) = (mk(0, k, rows, types::Q4_K), mk(7, k, rows, types::Q4_K));
    let dq = mk(3, rows, k, types::Q5_1);
    let xs: Vec<f32> = (0..k * n_tok)
        .map(|i| ((i * 13 % 29) as f32 - 14.0) / 7.3)
        .collect();
    let ids = [5i32, 0, 2, 2, 4, 1];
    let run = |split: bool| {
        let mut b = Backend::cpu().unwrap();
        b.set_threads(2);
        let weights = Context::new(3, 0);
        let shape = [k as i64, rows as i64, n_expert as i64];
        let g = weights.new_tensor(types::Q4_K, &shape);
        let u = weights.new_tensor(types::Q4_K, &shape);
        let d = weights.new_tensor(types::Q5_1, &[rows as i64, k as i64, n_expert as i64]);
        weights.alloc_tensors(b.buffer_type()).unwrap();
        g.set_bytes(&gq);
        u.set_bytes(&uq);
        d.set_bytes(&dq);
        let ctx = Context::new(16, 32);
        let x = ctx.new_tensor(types::F32, &[k as i64, 1, n_tok as i64]);
        let it = ctx.new_tensor(types::I32, &[n_used as i64, n_tok as i64]);
        ctx.alloc_tensors(b.buffer_type()).unwrap();
        x.set_f32(&xs);
        it.set_i32(&ids);
        // Experts 0..3 cached, 3..6 missed.
        let out = if split {
            glu_down_node(&ctx, [g, u, d], x, it, (3, 0.0), mode)
        } else {
            let act = super::super::glu_tiered(&ctx, g, u, x, it, 3);
            super::super::mul_mat_id_tiered(&ctx, d, act, it, 3)
        };
        let mut gr = ctx.new_graph(32);
        gr.expand(out);
        ctx.alloc_graph(&gr, ComputeAllocator::new(b.buffer_type()))
            .unwrap();
        b.compute(&gr).unwrap();
        assert_eq!(out.shape(), vec![k as i64, n_used as i64, n_tok as i64]);
        out.to_f32()
    };
    let (split, plain) = (run(true), run(false));
    assert!(plain.iter().any(|&v| v != 0.0));
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&split), bits(&plain));
}

/// The peer alone stages outside the graph, so its pair of ops lays out the step graph's memory
/// as the plain ops do, node for node (#522): the switch cannot move which layers ggml-cuda
/// fuses the MoE weighted sum in (#472). The side stream's staging node adds a node.
#[test]
fn peer_alone_keeps_the_plain_layout() {
    let (k, rows, n_expert, n_used, n_tok) = (256i64, 64i64, 6i64, 3i64, 2i64);
    let layout = |split: Option<i32>| {
        let b = Backend::cpu().unwrap();
        let weights = Context::new(9, 0);
        let stacks: Vec<_> = (0..3)
            .map(|_| {
                [
                    weights.new_tensor(types::Q4_K, &[k, rows, n_expert]),
                    weights.new_tensor(types::Q4_K, &[k, rows, n_expert]),
                    weights.new_tensor(types::Q5_1, &[rows, k, n_expert]),
                ]
            })
            .collect();
        weights.alloc_tensors(b.buffer_type()).unwrap();
        let ctx = Context::new(64, 0);
        let x = ctx.new_tensor(types::F32, &[k, 1, n_tok]).as_input();
        let it = ctx.new_tensor(types::I32, &[n_used, n_tok]).as_input();
        // Three layers, each reading the last one's output summed over its experts.
        let mut h = x;
        for &[g, u, d] in &stacks {
            let out = match split {
                Some(mode) => glu_down_node(&ctx, [g, u, d], h, it, (3, 0.0), mode),
                None => {
                    let act = super::super::glu_tiered(&ctx, g, u, h, it, 3);
                    super::super::mul_mat_id_tiered(&ctx, d, act, it, 3)
                }
            };
            let sum = ctx.sum_rows(ctx.cont(ctx.permute(out, [1, 0, 2, 3])));
            h = ctx.add(ctx.scale(h, 0.5), ctx.reshape_3d(sum, k, 1, n_tok));
        }
        let mut gr = ctx.new_graph(64);
        gr.expand(h);
        ctx.alloc_graph(&gr, ComputeAllocator::new(b.buffer_type()))
            .unwrap();
        let g = gr.raw();
        // SAFETY: an allocated graph; every node has its buffer and data.
        unsafe {
            (0..ffi::ggml_graph_n_nodes(g))
                .map(|i| {
                    let t = ffi::ggml_graph_node(g, i);
                    (*t).data as usize - ffi::ggml_backend_buffer_get_base((*t).buffer) as usize
                })
                .collect::<Vec<_>>()
        }
    };
    let plain = layout(None);
    assert_eq!(layout(Some(MODE_PEER)), plain);
    assert_eq!(layout(Some(MODE_SIDE)).len(), plain.len() + 3);
}

/// Off by default; the CTA count falls back to [`CTAS`].
#[test]
fn off_by_default_and_ctas_default() {
    assert!(!enabled());
    assert_eq!(ctas(), CTAS as i32);
}

/// Forced off beats the automatic choice; forcing it off changes no graph, so this test may run
/// beside others, and it puts the state back.
#[test]
fn forced_off_beats_automatic() {
    set_enabled(false);
    set_auto(true);
    assert!(!enabled());
    set_auto(false);
    FORCED.store(0, Ordering::Relaxed);
    assert!(!enabled());
}
