use super::*;
use crate::ggml::{Backend, Context, types};

/// A graph over a few weights: a matmul, the same weight again, a view of a head's first rows,
/// embedding lookups and an expert stack. Counted bytes are the bytes of what each op reads.
#[test]
fn counts_what_each_op_reads() {
    let b = Backend::cpu().unwrap();
    let weights = Context::new(8, 0);
    let w = weights.new_tensor(types::Q8_0, &[64, 32]);
    let head = weights.new_tensor(types::F32, &[64, 100]);
    let emb = weights.new_tensor(types::F16, &[64, 50]);
    let exps = weights.new_tensor(types::Q8_0, &[64, 16, 8]);
    weights.alloc_tensors(b.buffer_type()).unwrap();
    let ws: Vec<usize> = [w, head, emb, exps]
        .iter()
        .map(|t| t.raw() as usize)
        .collect();

    let ctx = Context::new(32, 64);
    let x = ctx.new_tensor(types::F32, &[64, 3]);
    let tok = ctx.new_tensor(types::I32, &[3]);
    let ids = ctx.new_tensor(types::I32, &[2, 3]);
    let a = ctx.mul_mat(w, x);
    let a2 = ctx.mul_mat(w, ctx.scale(x, 2.0));
    let part = ctx.view_2d(head, [64, 40], head.nb()[1], 0);
    let logits = ctx.mul_mat(part, x);
    let e = ctx.get_rows(emb, tok);
    let x3 = ctx.reshape_3d(x, 64, 1, 3);
    let moe = ctx.mul_mat_id(exps, x3, ids);
    let mut g = ctx.new_graph(64);
    for t in [a, a2, logits, e, moe] {
        g.expand(t);
    }
    let is_weight = |t: *const ffi::ggml_tensor| ws.contains(&(t as usize));
    let got = unsafe { count(g.raw(), &is_weight) };
    let q8 = crate::ggml::row_size(types::Q8_0, 64) as u64;
    let want_dense = 32 * q8 // w, once though read twice
        + 40 * 64 * 4 // the head's first 40 rows
        + 3 * 64 * 2; // three embedding rows
    assert_eq!(got.dense, want_dense);
    let per_expert = 16 * crate::ggml::row_size(types::Q8_0, 64) as u64;
    assert_eq!(
        got.experts,
        vec![ExpertRead {
            bytes_per_expert: per_expert,
            n_expert: 8,
            pairs: 6
        }]
    );
    assert_eq!(got.upper(), want_dense + 6 * per_expert);
}

/// Weights that share a data address (a `--tp` meta buffer gives every weight the same
/// placeholder) still count once each; two views of one weight at different offsets count
/// both, the same view twice once.
#[test]
fn weights_count_by_tensor_not_address() {
    let weights = Context::new(8, 0);
    // Not allocated: every data pointer is null, like the meta buffer's shared placeholder.
    let w1 = weights.new_tensor(types::F32, &[64, 32]);
    let w2 = weights.new_tensor(types::F32, &[64, 32]);
    let exps1 = weights.new_tensor(types::F32, &[64, 16, 8]);
    let exps2 = weights.new_tensor(types::F32, &[64, 16, 8]);
    let ws: Vec<usize> = [w1, w2, exps1, exps2]
        .iter()
        .map(|t| t.raw() as usize)
        .collect();

    let ctx = Context::new(32, 64);
    let x = ctx.new_tensor(types::F32, &[64, 3]);
    let ids = ctx.new_tensor(types::I32, &[2, 3]);
    let x3 = ctx.reshape_3d(x, 64, 1, 3);
    let lo = ctx.view_2d(w1, [64, 16], w1.nb()[1], 0);
    let hi = ctx.view_2d(w1, [64, 16], w1.nb()[1], 16 * w1.nb()[1]);
    let lo2 = ctx.view_2d(w1, [64, 16], w1.nb()[1], 0);
    let mut g = ctx.new_graph(64);
    for t in [
        ctx.mul_mat(w2, x),
        ctx.mul_mat(lo, x),
        ctx.mul_mat(hi, x),
        ctx.mul_mat(lo2, x),
        ctx.mul_mat_id(exps1, x3, ids),
        ctx.mul_mat_id(exps2, x3, ids),
    ] {
        g.expand(t);
    }
    let is_weight = |t: *const ffi::ggml_tensor| ws.contains(&(t as usize));
    let got = unsafe { count(g.raw(), &is_weight) };
    // w2 whole, w1's two halves once each.
    assert_eq!(got.dense, 2 * 64 * 32 * 4);
    assert_eq!(got.experts.len(), 2);
}

/// Inputs and caches are not weights: a graph over them counts nothing.
#[test]
fn non_weights_count_nothing() {
    let ctx = Context::new(8, 16);
    let a = ctx.new_tensor(types::F32, &[64, 32]);
    let x = ctx.new_tensor(types::F32, &[64, 3]);
    let y = ctx.mul_mat(a, x);
    let mut g = ctx.new_graph(16);
    g.expand(y);
    let got = unsafe { count(g.raw(), &|_| false) };
    assert_eq!(got, GraphBytes::default());
}

/// More choices than experts: at most every expert once.
#[test]
fn expert_reads_cap_at_the_stack() {
    let r = ExpertRead {
        bytes_per_expert: 10,
        n_expert: 4,
        pairs: 9,
    };
    assert_eq!(r.upper(), 40);
}

/// Device memory, host memory and the link are told apart; staged copies cross the link but
/// are read again from VRAM, so they don't count twice in the total.
#[test]
fn phase_splits_device_host_and_link() {
    let mut p = Phase::default();
    p.add(
        &GraphBytes {
            dense: 100,
            experts: vec![],
        },
        0.5,
    );
    p.add_experts(&ExpertTraffic {
        device: 40,
        host: 10,
        staged: 30,
        host_passes: 1,
    });
    p.kv = 20;
    p.kv_host = 5;
    assert_eq!(p.device(), 160);
    assert_eq!(p.host(), 15);
    assert_eq!(p.link(), 45);
    assert_eq!(p.total(), 175);
    assert!((p.gb_s() - 175.0 / 0.5 / 1e9).abs() < 1e-18);
    assert!((p.device_gb_s() - 160.0 / 0.5 / 1e9).abs() < 1e-18);
    let mut twice = p;
    twice.merge(&p);
    assert_eq!(twice.steps, 2);
    assert_eq!(twice.total(), 350);
    assert_eq!(twice.since(&p), p);
}

#[test]
fn totals_by_kind_and_stage() {
    let mut t = Totals::default();
    t.phase_mut(Kind::of_step(true)).dense = 1;
    t.phase_mut(Kind::of_step(false)).dense = 2;
    t.phase_mut(Kind::Draft).dense = 4;
    assert_eq!((t.prefill.dense, t.decode.dense, t.draft.dense), (1, 2, 4));
    assert_eq!(t.decoding().dense, 6);
    t.stage_mut(1, Kind::Decode).unwrap().kv = 7;
    // Stages past the last counted fold into it.
    t.stage_mut(MAX_STAGES + 3, Kind::Prefill).unwrap().kv = 9;
    assert!(t.stage_mut(0, Kind::Draft).is_none());
    assert_eq!(t.stages[1].decode.kv, 7);
    assert_eq!(t.stages[MAX_STAGES - 1].prefill.kv, 9);
    assert_eq!(t.since(&t), Totals::default());
}
