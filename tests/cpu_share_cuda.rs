//! The hybrid expert share (#268) on a GPU: one MoE layer's experts tiered (a few in VRAM
//! slots, the rest in host memory), the host computing its share of the host experts through
//! the exchange buffer and stream flags. The GPU's pairs must equal the layer computed by the
//! GPU alone bit for bit, the host's the one-thread host reference bit for bit (and the GPU's
//! result within quantization noise).
//!
//! `cargo test --release --features cuda --test cpu_share_cuda -- --ignored --nocapture`

#![cfg(feature = "cuda")]

use legion::ggml::{ComputeAllocator, Context, DeviceKind, device_by_kind, types};
use legion::model::core::expert_mv;
use legion::residency::cpu_share::{
    CpuShareConfig, DevicePlan, HostExperts, format_of, ops, pool, select, setup, threshold,
};
use legion::residency::cuda::{caps, host_arena, tiered};

#[test]
#[ignore = "needs a CUDA GPU that can read host memory"]
fn host_share_matches_the_gpu() {
    let (n_embd, n_ff, n_expert, used) = (512i64, 256i64, 64i64, 8i64);
    let kinds = [
        (types::Q4_K, n_embd, n_ff),
        (types::Q4_K, n_embd, n_ff),
        (types::Q5_1, n_ff, n_embd),
    ];
    let dev = device_by_kind(DeviceKind::Gpu).expect("no GPU");
    let mut backend = dev.init().unwrap();
    expert_mv::set_enabled(true);
    let node = caps(0).map_or(0, |c| c.host_numa_id.max(0) as u32);
    let arena = host_arena(node, &[0]).expect("host arena");
    let mut seed = 4242u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % 2000) as f32 / 1000.0 - 1.0
    };
    let cached: Vec<usize> = (0..16).map(|i| i * 3).collect();
    let n_slots = cached.len();
    let host_ctx = Context::new(3, 0);
    let mut host = Vec::new();
    let mut data = Vec::new();
    for (i, &(ty, k, m)) in kinds.iter().enumerate() {
        let w: Vec<f32> = (0..k * m * n_expert).map(|_| 0.1 * rnd()).collect();
        data.push(legion::ggml::quantize(ty, &w, k as usize));
        host.push(
            host_ctx
                .new_tensor(ty, &[k, m, n_expert])
                .named(&format!("h{i}")),
        );
    }
    host_ctx.alloc_tensors(arena).unwrap();
    let mut tiers = Vec::new();
    for i in 0..3 {
        host[i].set_bytes(&data[i]);
        let tc = tiered(0, host[i], n_slots, None, &format!("t{i}")).unwrap();
        let t = tc.tensor(&format!("t{i}")).unwrap();
        let per = host[i].nb()[2];
        for (s, &e) in cached.iter().enumerate() {
            t.set_bytes_at(s * per, &data[i][e * per..(e + 1) * per]);
        }
        tiers.push(tc);
    }
    let w: Vec<_> = (0..3)
        .map(|i| tiers[i].tensor(&format!("t{i}")).unwrap())
        .collect();
    let layer = 5usize;
    let he = HostExperts {
        base: [0, 1, 2].map(|i| host[i].host_bytes().expect("host memory").as_ptr() as usize),
        expert_bytes: [host[0].nb()[2], host[1].nb()[2], host[2].nb()[2]],
        up_gate: format_of(types::Q4_K).unwrap(),
        down: format_of(types::Q5_1).unwrap(),
        n_embd: n_embd as usize,
        n_ff: n_ff as usize,
        n_expert: n_expert as usize,
    };
    let mut layers = vec![None; layer + 1];
    layers[layer] = Some(he);
    let cfg = CpuShareConfig {
        share: 0.5,
        workers: 4,
        smt: 2,
    };
    setup(
        &cfg,
        vec![DevicePlan {
            device: 0,
            node,
            layers,
            n_embd: n_embd as usize,
        }],
    )
    .unwrap();
    let thr = threshold(cfg.share);
    let table: Vec<i32> = (0..n_expert as usize)
        .map(|e| match cached.iter().position(|&c| c == e) {
            Some(s) => s as i32,
            None => (n_slots + e) as i32,
        })
        .collect();
    for n in [1i64, 3] {
        let ctx = Context::new(32, 32);
        let h = ctx.new_tensor(types::F32, &[n_embd, n]).as_input();
        let ids = ctx.new_tensor(types::I32, &[used, n]).as_input();
        let x = ctx.reshape_3d(h, n_embd, 1, n);
        // The GPU alone.
        let act = expert_mv::glu_tiered(&ctx, w[1], w[0], x, ids, n_slots);
        let plain = expert_mv::mul_mat_id_tiered(&ctx, w[2], act, ids, n_slots).as_output();
        // With the host's share, run twice in a row (two layers' worth of sequence numbers).
        let mut shared_outs = Vec::new();
        for _ in 0..2 {
            let ws = [w[1], w[0], w[2]];
            let out = ops::glu_down(&ctx, ws, x, ids, n_slots, n_expert as usize, layer, thr)
                .expect("the share takes the step");
            shared_outs.push(out.as_output());
        }
        let mut g = ctx.new_graph(64);
        g.expand(plain);
        for &o in &shared_outs {
            g.expand(o);
        }
        ctx.alloc_graph(&g, ComputeAllocator::new(dev.buffer_type()))
            .unwrap();
        let xs: Vec<f32> = (0..n_embd * n).map(|_| rnd()).collect();
        h.set_f32(&xs);
        let picks: Vec<i32> = (0..n * used)
            .map(|i| table[((i / used * 11 + (i % used) * 7) % n_expert) as usize])
            .collect();
        ids.set_i32(&picks);
        backend.compute(&g).unwrap();
        let a = plain.to_f32();
        let host = select(&picks, n_slots as i32, n_expert as i32, thr, layer as u32);
        assert!(
            host.iter().any(|&h| h),
            "the case gives the host nothing to do"
        );
        assert!(
            !host.iter().all(|&h| h),
            "the case gives the GPU nothing to do"
        );
        // The host's rows as the pool computes them, on one thread.
        let host_ids: Vec<i32> = picks
            .iter()
            .zip(&host)
            .map(|(&id, &h)| if h { id - n_slots as i32 } else { -1 })
            .collect();
        let mut want = vec![0f32; a.len()];
        // SAFETY: the host experts live in `host_ctx`.
        unsafe { pool::reference(&he, used as usize, &host_ids, &xs, &mut want) };
        let scale = a.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
        let row = n_embd as usize;
        for (r, o) in shared_outs.iter().enumerate() {
            let b = o.to_f32();
            let mut diff = 0f32;
            for (p, &h) in host.iter().enumerate() {
                let (ra, rb) = (&a[p * row..][..row], &b[p * row..][..row]);
                if h {
                    let rw = &want[p * row..][..row];
                    assert!(
                        rb.iter().zip(rw).all(|(x, y)| x.to_bits() == y.to_bits()),
                        "{n} tokens, run {r}, pair {p}: not the host's rows"
                    );
                    diff = ra
                        .iter()
                        .zip(rb)
                        .fold(diff, |m, (x, y)| m.max((x - y).abs()));
                } else {
                    assert!(
                        ra.iter().zip(rb).all(|(x, y)| x.to_bits() == y.to_bits()),
                        "{n} tokens, run {r}, pair {p}: the GPU's rows changed"
                    );
                }
            }
            let n_host = host.iter().filter(|&&h| h).count();
            println!("{n} tokens, run {r}: {n_host} host pairs, max |diff| {diff} of {scale}");
            assert!(
                diff <= 0.02 * scale,
                "{n} tokens, run {r}: {diff} of {scale}"
            );
        }
    }
    ops::clear();
}
