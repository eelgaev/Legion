use super::*;
use crate::engine::lm::report::{HostMemory, NgramPlacement};

const MIB: u64 = 1 << 20;

fn gpu(device: &str, cuda: usize, total: u64, free: u64) -> Gpu {
    Gpu {
        device: device.into(),
        name: "GPU".into(),
        cuda: Some(cuda),
        total,
        free,
    }
}

#[test]
fn shares_split_evenly_with_the_remainder_first() {
    assert_eq!(share(10, 1, 0), 10);
    assert_eq!(share(10, 0, 0), 10);
    assert_eq!(
        (0..3).map(|i| share(10, 3, i)).collect::<Vec<_>>(),
        [4, 3, 3]
    );
    assert_eq!((0..4).map(|i| share(8, 4, i)).sum::<u64>(), 8);
}

#[test]
fn spill_tokens_follow_the_budget() {
    // No lend: nothing spills.
    assert_eq!(spill_tokens(0, 0, 0), 0);
    assert_eq!(spill_tokens(1000, 0, 5 * MIB), 0);
    // A budget covering the tail keeps all of it; a larger one no more.
    assert_eq!(spill_tokens(229_376, 2880 * MIB, 2880 * MIB), 229_376);
    assert_eq!(spill_tokens(229_376, 2880 * MIB, 9000 * MIB), 229_376);
    // Half the bytes, half the tokens; none without a budget.
    assert_eq!(spill_tokens(1000, 100 * MIB, 50 * MIB), 500);
    assert_eq!(spill_tokens(1000, 100 * MIB, 0), 0);
}

#[test]
fn other_is_what_the_named_parts_leave_never_negative() {
    let g = GpuPlan {
        total: 16_000 * MIB,
        free: 1_000 * MIB,
        weights: 6_000 * MIB,
        kv_vram: 2_000 * MIB,
        kv_lent: 1_000 * MIB,
        expert_cache: 4_000 * MIB,
        compute: 1_500 * MIB,
        compute_allocated: 500 * MIB,
        ..Default::default()
    };
    assert_eq!(g.used(), 15_000 * MIB);
    // Only the allocated compute buffers count: the rest of their room is free memory.
    assert_eq!(g.other(), 1_500 * MIB);
    let over = GpuPlan {
        weights: 20_000 * MIB,
        ..g.clone()
    };
    assert_eq!(over.other(), 0);
    // A driver reporting more free than total.
    let odd = GpuPlan {
        free: 17_000 * MIB,
        ..g
    };
    assert_eq!(odd.used(), 0);
    assert_eq!(odd.other(), 0);
}

fn stage() -> StageMemory {
    StageMemory {
        device: "CUDA0".into(),
        weights: 6_000 * MIB,
        kv_tokens: 131_072,
        kv_bytes: 1_800 * MIB,
        kv_staging_bytes: 0,
        states_bytes: 200 * MIB,
        expert_slot_bytes: 5_000 * MIB,
        ..Default::default()
    }
}

#[test]
fn one_gpu_with_a_lent_tail() {
    let f = StageFacts {
        kv_lent: 1_440 * MIB,
        kv_vram_tokens: 32_768,
        compute_allocated: 300 * MIB,
        compute_planned: 900 * MIB,
    };
    let g = stage_gpus(
        &stage(),
        &f,
        &[gpu("CUDA0", 0, 16_000 * MIB, 600 * MIB)],
        &[(0, 5_000 * MIB)],
    );
    assert_eq!(g.len(), 1);
    let g = &g[0];
    assert_eq!(g.weights, 6_000 * MIB);
    // The pool and states, less the lent tail.
    assert_eq!(g.kv_vram, 560 * MIB);
    assert_eq!(g.kv_vram_tokens, 32_768);
    assert_eq!(g.kv_lent, 1_440 * MIB);
    // The slots less the lent tail they run on.
    assert_eq!(g.expert_cache, 3_560 * MIB);
    assert_eq!(g.compute, 900 * MIB);
    assert_eq!(g.compute_allocated, 300 * MIB);
    // 15,400 used = 6,000 + 560 + 1,440 + 3,560 + 300 + 3,540.
    assert_eq!(g.other(), 3_540 * MIB);
}

#[test]
fn one_gpu_without_a_lend() {
    let f = StageFacts {
        kv_vram_tokens: 131_072,
        compute_allocated: 1_200 * MIB,
        compute_planned: 900 * MIB,
        ..Default::default()
    };
    let g = &stage_gpus(
        &stage(),
        &f,
        &[gpu("CUDA0", 0, 16_000 * MIB, 600 * MIB)],
        &[],
    )[0];
    assert_eq!(g.kv_vram, 2_000 * MIB);
    assert_eq!(g.kv_lent, 0);
    // The stage's own slot bytes when the cache lists none by device.
    assert_eq!(g.expert_cache, 5_000 * MIB);
    // More allocated than planned: the allocated.
    assert_eq!(g.compute, 1_200 * MIB);
}

#[test]
fn tensor_parallel_stage_splits_over_its_gpus() {
    let f = StageFacts {
        kv_vram_tokens: 131_072,
        compute_planned: 1_000 * MIB,
        ..Default::default()
    };
    let gpus = [
        gpu("CUDA0", 0, 16_000 * MIB, 1_000 * MIB),
        gpu("CUDA1", 1, 16_000 * MIB, 2_000 * MIB),
    ];
    // Expert slots by device where the cache lists them.
    let g = stage_gpus(&stage(), &f, &gpus, &[(0, 3_000 * MIB), (1, 2_000 * MIB)]);
    assert_eq!(g.len(), 2);
    assert_eq!((g[0].weights, g[1].weights), (3_000 * MIB, 3_000 * MIB));
    assert_eq!((g[0].kv_vram, g[1].kv_vram), (1_000 * MIB, 1_000 * MIB));
    assert_eq!(
        (g[0].expert_cache, g[1].expert_cache),
        (3_000 * MIB, 2_000 * MIB)
    );
    assert_eq!((g[0].compute, g[1].compute), (500 * MIB, 500 * MIB));
    assert_eq!((g[0].free, g[1].free), (1_000 * MIB, 2_000 * MIB));
    assert_eq!(g[1].device, "CUDA1");
    // Else evenly.
    let g = stage_gpus(&stage(), &f, &gpus, &[]);
    assert_eq!(
        (g[0].expert_cache, g[1].expert_cache),
        (2_500 * MIB, 2_500 * MIB)
    );
}

#[test]
fn host_line_from_the_load_report() {
    let host = |kind, bytes| HostMemory {
        node: None,
        kind,
        bytes,
        tokens: None,
    };
    let mut r = LoadReport {
        host: vec![
            host(HostUse::ExpertArena, 40_000 * MIB),
            host(HostUse::MappedExperts, 1_000 * MIB),
            host(HostUse::SharedWeights, 700 * MIB),
            host(HostUse::HostWeights, 300 * MIB),
            host(HostUse::KvHostTier, 8_000 * MIB),
            host(HostUse::SavedStates, 600 * MIB),
        ],
        ngram: Some(NgramPlacement {
            bytes: 27_000 * MIB,
            io: "ram".into(),
            ..Default::default()
        }),
        ..Default::default()
    };
    let h = host_from_report(&r);
    assert_eq!(h.expert_host, 41_000 * MIB);
    assert_eq!(h.host_weights, 1_000 * MIB);
    assert_eq!(h.kv_host_tier, 8_000 * MIB);
    assert_eq!(h.saved_states, 600 * MIB);
    assert_eq!(h.ngram_table, 27_000 * MIB);
    // Left to the caller.
    assert_eq!(
        (h.kv_spill_max, h.compute_spill_max, h.available),
        (0, 0, 0)
    );
    // A direct-read table counts its row cache; a mapped one nothing.
    r.ngram = Some(NgramPlacement {
        bytes: 27_000 * MIB,
        io: "direct".into(),
        ram_cache_bytes: 2_000 * MIB,
        ..Default::default()
    });
    assert_eq!(host_from_report(&r).ngram_table, 2_000 * MIB);
    r.ngram.as_mut().unwrap().io = "mmap".into();
    assert_eq!(host_from_report(&r).ngram_table, 0);
    assert_eq!(
        host_from_report(&LoadReport::default()),
        HostPlan::default()
    );
}
