use super::super::{Operands, glu_tiered_clamped, mul_mat_id_tiered};
use super::*;
use crate::ggml::{Backend, ComputeAllocator};

/// The staged-ahead ops compute what the plain ones do (on the CPU, which reads every expert in
/// place), and their nodes give the kernel the slot table and offsets.
#[test]
fn staged_ahead_ops_match_the_plain_ones_and_carry_the_ring() {
    let (k, rows, n_expert, n_used, n_tok) = (256usize, 32usize, 6usize, 3usize, 2usize);
    let mk = |seed: usize, ty, k: usize, rows: usize| -> Vec<u8> {
        let wf: Vec<f32> = (0..k * rows * n_expert)
            .map(|i| (((i + seed) * 37 % 101) as f32 - 50.0) / 400.0)
            .collect();
        crate::ggml::quantize(ty, &wf, k)
    };
    let (gq, uq) = (mk(0, types::Q4_K, k, rows), mk(7, types::Q4_K, k, rows));
    // The down projection: `rows` in, `k` out.
    let dq = mk(3, types::Q8_0, rows, k);
    let xs: Vec<f32> = (0..k * n_tok)
        .map(|i| ((i * 13 % 29) as f32 - 14.0) / 7.3)
        .collect();
    let ids = [5i32, 0, 2, 2, 4, -1];
    let run = |staged: bool| {
        let mut b = Backend::cpu().unwrap();
        b.set_threads(2);
        let weights = Context::new(4, 0);
        let shape = [k as i64, rows as i64, n_expert as i64];
        let g = weights.new_tensor(types::Q4_K, &shape);
        let u = weights.new_tensor(types::Q4_K, &shape);
        let d = weights.new_tensor(types::Q8_0, &[rows as i64, k as i64, n_expert as i64]);
        let ring = weights.new_tensor(types::I8, &[4096]);
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
        let a = Ahead {
            ring,
            after: it,
            table: 1024,
            slots: 2048,
            slot: 512,
            off: [0, 128, 256],
        };
        let (act, out) = if staged {
            let act = glu_clamped(&ctx, [g, u], x, it, (2, 0.5), &a);
            (act, down(&ctx, d, act, it, 2, &a))
        } else {
            let act = glu_tiered_clamped(&ctx, g, u, x, it, 2, 0.5);
            (act, mul_mat_id_tiered(&ctx, d, act, it, 2))
        };
        if staged {
            // SAFETY: nodes of `glu_clamped` and `down` with allocated sources.
            let (o, od) = unsafe { (Operands::of(&*act.raw()), Operands::of(&*out.raw())) };
            let base = unsafe { (*ring.raw()).data as *const u8 };
            let s = o.ahead.expect("staged");
            assert_eq!(s.table, unsafe { base.add(1024) });
            assert_eq!((s.hdr, s.slot, s.w, s.w2), (1024, 512, 0, 128));
            assert!(!o.w2.is_null(), "the SwiGLU op keeps its up projection");
            let s = od.ahead.expect("staged");
            assert_eq!((s.hdr, s.slot, s.w, s.w2), (1024, 512, 256, 0));
            assert!(od.w2.is_null(), "the down op's fifth source is the ring");
        }
        let mut gr = ctx.new_graph(64);
        gr.expand(out);
        ctx.alloc_graph(&gr, ComputeAllocator::new(b.buffer_type()))
            .unwrap();
        b.compute(&gr).unwrap();
        out.to_f32()
    };
    let (plain, staged) = (run(false), run(true));
    assert!(plain.iter().any(|&v| v != 0.0));
    assert_eq!(plain, staged);
}
