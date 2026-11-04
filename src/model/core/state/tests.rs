//! Slot copies between state pools: the host-link path (a pool in pinned host memory, #441)
//! against the graph path.

use super::*;
use crate::ggml::Backend;

/// Two layers, the first a delta-net (recurrent) layer.
pub(super) fn hparams() -> Hparams {
    Hparams {
        n_layer: 2,
        n_embd: 32,
        n_head: 2,
        n_head_kv: 1,
        head_dim: 16,
        n_rot: 16,
        rope_sections: [0; 4],
        rope_freq_base: 1e4,
        n_ctx_train: 4096,
        rms_eps: 1e-6,
        moe: false,
        n_expert: 0,
        n_expert_used: 0,
        full_attention_interval: 2,
        ssm_d_conv: 4,
        ssm_d_state: 8,
        ssm_n_group: 2,
        ssm_dt_rank: 2,
        ssm_d_inner: 16,
        n_vocab: 16,
        arch: crate::model::core::ArchRef::new(crate::model::qwen35::Qwen35),
    }
}

/// Fills every state tensor of `pool` with bytes that differ per tensor and per position.
pub(super) fn fill(pool: &StatePool, seed: u8) {
    for (i, t) in pool.tensors().enumerate() {
        let b: Vec<u8> = (0..t.nbytes())
            .map(|j| (j as u8).wrapping_mul(31).wrapping_add(seed ^ i as u8))
            .collect();
        t.set_bytes(&b);
    }
}

fn contents(pool: &StatePool) -> Vec<Vec<u8>> {
    pool.tensors().map(|t| t.bytes()).collect()
}

#[test]
fn host_link_copies_match_the_graph_copies() {
    let hp = hparams();
    assert!(hp.is_recurrent(0) && !hp.is_recurrent(1));
    let mut backend = Backend::cpu().unwrap();
    let buft = backend.buffer_type();
    let src = StatePool::new(&hp, buft, 2).unwrap();
    fill(&src, 7);
    let by_graph = StatePool::new(&hp, buft, 3).unwrap();
    let mut by_link = StatePool::new(&hp, buft, 3).unwrap();
    fill(&by_graph, 99);
    fill(&by_link, 99);
    // Stands for a pool in device memory: copies between it and `src` take the host link.
    by_link.host = false;
    StatePool::copy_slots(
        &mut backend,
        &[(&src, 1, &by_graph, 2), (&src, 0, &by_graph, 0)],
    )
    .unwrap();
    StatePool::copy_slots(
        &mut backend,
        &[(&src, 1, &by_link, 2), (&src, 0, &by_link, 0)],
    )
    .unwrap();
    assert_eq!(contents(&by_graph), contents(&by_link));
    // And back: device to host.
    let back = StatePool::new(&hp, buft, 2).unwrap();
    StatePool::copy_slots(
        &mut backend,
        &[(&by_link, 2, &back, 1), (&by_link, 0, &back, 0)],
    )
    .unwrap();
    assert_eq!(contents(&back), contents(&src));
}

#[test]
fn copies_between_host_pools_stay_on_the_graph() {
    let hp = hparams();
    let mut backend = Backend::cpu().unwrap();
    let buft = backend.buffer_type();
    let a = StatePool::new(&hp, buft, 2).unwrap();
    let b = StatePool::new(&hp, buft, 2).unwrap();
    assert!(a.is_host() && b.is_host());
    fill(&a, 3);
    StatePool::copy_slots(&mut backend, &[(&a, 0, &b, 0), (&a, 1, &b, 1)]).unwrap();
    assert_eq!(contents(&a), contents(&b));
    assert!(a.bytes() > 0 && a.bytes() == b.bytes());
}

#[test]
fn clear_slot_zeroes_that_slot_only() {
    let hp = hparams();
    let backend = Backend::cpu().unwrap();
    let pool = StatePool::new(&hp, backend.buffer_type(), 3).unwrap();
    fill(&pool, 5);
    let before = contents(&pool);
    pool.clear_slot(1);
    let ranges = pool.slot_ranges(1);
    assert_eq!(ranges.len(), pool.tensors().count());
    for ((t, off, n), (old, new)) in ranges.iter().zip(before.iter().zip(contents(&pool))) {
        assert_eq!(*n * 3, t.nbytes());
        assert_eq!(*off, *n);
        assert!(new[*off..*off + *n].iter().all(|&b| b == 0));
        assert_eq!(new[..*off], old[..*off]);
        assert_eq!(new[*off + *n..], old[*off + *n..]);
    }
}
