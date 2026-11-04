//! A KV pool with a host tier (#352, [`crate::model::core::KvTier`]): the paged kernels read
//! blocks wherever they live, so attention over a tiered pool equals attention over the same
//! values in device memory, bit for bit, for decode and prefill, with sequences whose blocks
//! sit in both tiers. Prefill through staging slots (whole host blocks copied to VRAM once per
//! step) gives the same bits when one piece holds them all, and the same values to rounding
//! when the step runs in pieces.

use super::tests::{f16_bytes, hparams, noise};
use super::*;
use crate::ggml::{Backend, ComputeAllocator, DeviceKind, device_by_kind};
use crate::model::core::{KV_BLOCK, KvPool, KvTier};

/// The paged op over `kv` for one batch: `n_tok` tokens per sequence after `n_past`.
#[allow(clippy::too_many_arguments)]
fn run(
    backend: &mut Backend,
    kv: &KvPool,
    d: usize,
    n_head: usize,
    n_head_kv: usize,
    n_tok: usize,
    n_past: &[usize],
    tables: &[Vec<u32>],
    qv: &[f32],
) -> Vec<f32> {
    let (kp, vp) = kv.pools(0);
    let ns = n_past.len();
    let width = tables.iter().map(Vec::len).max().unwrap();
    let ctx = Context::new(64, 64);
    let q = ctx
        .new_tensor(
            types::F32,
            &[d as i64, n_head as i64, n_tok as i64, ns as i64],
        )
        .as_input();
    let tab = ctx
        .new_tensor(types::I32, &[width as i64, ns as i64])
        .as_input();
    let past = ctx.new_tensor(types::I32, &[ns as i64]).as_input();
    let tier = kv.paged_tier(0);
    let scratch_ne = match tier {
        Some(t) if n_tok > 1 => [
            tier_scratch_len(n_head, n_tok, ns, width, t.stage_blocks) as i64,
            1,
            1,
            1,
        ],
        _ => scratch_shape(d as i64, n_head as i64, n_head_kv as i64, n_tok, ns),
    };
    let scratch = ctx.new_tensor(types::F32, &scratch_ne).as_input();
    let scale = 1.0 / (d as f32).sqrt();
    let out = attention_tiered(&ctx, q, kp, vp, tab, past, scratch, scale, 0, 0, tier).as_output();
    let mut graph = ctx.new_graph(64);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    q.set_f32(qv);
    let mut t = Vec::new();
    for (s, tb) in tables.iter().enumerate() {
        t.extend(tb.iter().map(|&x| x as i32));
        t.resize((s + 1) * width, 0);
    }
    tab.set_i32(&t);
    past.set_i32(&n_past.iter().map(|&p| p as i32).collect::<Vec<_>>());
    backend.compute(&graph).unwrap();
    out.to_f32()
}

#[test]
#[ignore = "needs a GPU whose driver maps host NUMA memory"]
fn tiered_pool_reads_like_device_memory() {
    let mut backend = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let caps = crate::residency::cuda::caps(0).expect("CUDA0");
    if !caps.host_numa_vmm || caps.host_numa_id < 0 {
        eprintln!("skipped: no host NUMA memory through cuMemCreate on this driver");
        return;
    }
    let (d, n_head, n_head_kv) = (256, 16, 8);
    let hp = hparams(d as i64, n_head as i64, n_head_kv as i64);
    // 12 blocks, the first 4 in device memory and 8 in the host tier.
    let n_blocks = 12;
    let tier = KvTier {
        device: 0,
        node: caps.host_numa_id as u32,
        hbm_blocks: 4,
        stage_blocks: 0,
        split: None,
        alias: None,
    };
    let buft = backend.buffer_type();
    let plain = KvPool::new(&hp, buft, n_blocks).unwrap();
    let tiered = KvPool::new_tiered(
        &hp,
        buft,
        n_blocks,
        &[0],
        None,
        Default::default(),
        Some(tier),
    )
    .unwrap();
    assert_eq!(tiered.hbm_blocks(), 4);
    for kv in [&plain, &tiered] {
        let (kp, vp) = kv.pools(0);
        kp.set_bytes(&f16_bytes(&noise(kp.elements(), 31)));
        vp.set_bytes(&f16_bytes(&noise(vp.elements(), 32)));
    }
    // Sequences over blocks of both tiers, out of order.
    let tables = vec![vec![9u32, 1, 6, 3, 11], vec![0, 7], vec![10, 2, 8, 5]];
    for (n_tok, n_past) in [(1usize, vec![300usize, 70, 200]), (37, vec![190, 3, 120])] {
        assert!(
            tables
                .iter()
                .zip(&n_past)
                .all(|(t, &p)| t.len() * KV_BLOCK >= p + n_tok)
        );
        let qv: Vec<f32> = noise(d * n_head * n_tok * n_past.len(), 33)
            .iter()
            .map(|x| 0.5 * x)
            .collect();
        let want = run(
            &mut backend,
            &plain,
            d,
            n_head,
            n_head_kv,
            n_tok,
            &n_past,
            &tables,
            &qv,
        );
        let got = run(
            &mut backend,
            &tiered,
            d,
            n_head,
            n_head_kv,
            n_tok,
            &n_past,
            &tables,
            &qv,
        );
        assert_eq!(got, want, "{n_tok} tokens");
    }
}

/// Prefill over a tiered pool through staging slots: one piece (every host block of the step
/// staged at once) matches the device pool bit for bit; pieces of 3 and of 1 host block (the
/// rows merged across launches) match it to the F16 rounding of the weights. On the device's default prefill
/// kernel and on the m8n8k4 one (sm_70's, run emulated past it).
#[test]
#[ignore = "needs a GPU whose driver maps host NUMA memory"]
fn staged_prefill_reads_like_device_memory() {
    let mut backend = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let caps = crate::residency::cuda::caps(0).expect("CUDA0");
    if !caps.host_numa_vmm || caps.host_numa_id < 0 {
        eprintln!("skipped: no host NUMA memory through cuMemCreate on this driver");
        return;
    }
    let (d, n_head, n_head_kv) = (256, 16, 8);
    let hp = hparams(d as i64, n_head as i64, n_head_kv as i64);
    let n_blocks = 12;
    let buft = backend.buffer_type();
    let plain = KvPool::new(&hp, buft, n_blocks).unwrap();
    let tiered = |stage_blocks| {
        let tier = KvTier {
            device: 0,
            node: caps.host_numa_id as u32,
            hbm_blocks: 4,
            stage_blocks,
            split: None,
            alias: None,
        };
        KvPool::new_tiered(
            &hp,
            buft,
            n_blocks,
            &[0],
            None,
            Default::default(),
            Some(tier),
        )
        .unwrap()
    };
    let pools = [tiered(8), tiered(3), tiered(1)];
    for kv in std::iter::once(&plain).chain(&pools) {
        let (kp, vp) = kv.pools(0);
        kp.set_bytes(&f16_bytes(&noise(kp.elements(), 41)));
        vp.set_bytes(&f16_bytes(&noise(vp.elements(), 42)));
    }
    assert_eq!(pools[0].stage_blocks(), 8);
    // Host blocks (4..) in every sequence but the second, out of order; 37 tokens of 2 query
    // heads per kv head are two query tiles, which stage.
    let tables = vec![vec![9u32, 1, 6, 3, 11], vec![0, 7], vec![10, 2, 8, 5]];
    let (n_tok, n_past) = (37usize, vec![190usize, 3, 120]);
    let qv: Vec<f32> = noise(d * n_head * n_tok * n_past.len(), 43)
        .iter()
        .map(|x| 0.5 * x)
        .collect();
    for m884 in [false, true] {
        super::set_m884_prefill_here(m884);
        let launches = super::m884_launches();
        let go = |backend: &mut Backend, kv: &KvPool| {
            run(
                backend, kv, d, n_head, n_head_kv, n_tok, &n_past, &tables, &qv,
            )
        };
        let want = go(&mut backend, &plain);
        assert_eq!(go(&mut backend, &pools[0]), want, "one piece, m884 {m884}");
        for kv in &pools[1..] {
            let got = go(&mut backend, kv);
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            // P is rounded to F16 against each piece's own running max.
            eprintln!(
                "{} staging blocks, m884 {m884}: worst {worst}",
                kv.stage_blocks()
            );
            assert!(
                worst < 2e-3,
                "{} staging blocks, m884 {m884}: off by {worst}",
                kv.stage_blocks()
            );
        }
        assert_eq!(super::m884_launches() > launches, m884);
    }
    super::set_m884_prefill_here(false);
}

/// `--kv-vmm` (#814): a pool in a reserved virtual range (each cache tensor on a granule of its
/// own, in VRAM allocations of at most 32 MiB) reads like one in ggml's CUDA buffer, bit for
/// bit, for decode and prefill.
#[test]
#[ignore = "needs a GPU with virtual memory management"]
fn vmm_pool_reads_like_device_memory() {
    let mut backend = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let vmm = crate::residency::cuda::kv_vmm_type(0).expect("CUDA0 with VMM");
    let (d, n_head, n_head_kv) = (256, 16, 8);
    let hp = hparams(d as i64, n_head as i64, n_head_kv as i64);
    // 256 KiB per block of K or V: 40 MiB per tensor, two allocations.
    let n_blocks = 160;
    let plain = KvPool::new(&hp, backend.buffer_type(), n_blocks).unwrap();
    let mapped = KvPool::new(&hp, vmm, n_blocks).unwrap();
    let gran = crate::residency::cuda::caps(0).unwrap().device_granularity;
    for kv in [&plain, &mapped] {
        let (kp, vp) = kv.pools(0);
        kp.set_bytes(&f16_bytes(&noise(kp.elements(), 41)));
        vp.set_bytes(&f16_bytes(&noise(vp.elements(), 42)));
    }
    let (kp, vp) = mapped.pools(0);
    for t in [kp, vp] {
        // SAFETY: reading the address of a live tensor.
        let at = unsafe { (*t.raw()).data as usize };
        assert!(at.is_multiple_of(gran), "a cache tensor off its granule");
    }
    // Blocks in both allocations (the second from block 128 on), out of order.
    let tables = vec![
        vec![150u32, 1, 130, 3, 159],
        vec![0, 127],
        vec![128, 2, 140, 5],
    ];
    for (n_tok, n_past) in [(1usize, vec![300usize, 70, 200]), (37, vec![190, 3, 120])] {
        let qv: Vec<f32> = noise(d * n_head * n_tok * n_past.len(), 43)
            .iter()
            .map(|x| 0.5 * x)
            .collect();
        let want = run(
            &mut backend,
            &plain,
            d,
            n_head,
            n_head_kv,
            n_tok,
            &n_past,
            &tables,
            &qv,
        );
        let got = run(
            &mut backend,
            &mapped,
            d,
            n_head,
            n_head_kv,
            n_tok,
            &n_past,
            &tables,
            &qv,
        );
        assert_eq!(got, want, "{n_tok} tokens");
    }
}
