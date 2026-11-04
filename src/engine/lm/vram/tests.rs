use super::*;
use crate::ggml::Backend;
use crate::gguf::GgufFile;
use crate::model::core::{Hparams, KV_BLOCK, KvPool, KvTypes};
use crate::model::qwen4;

const MIB: usize = 1 << 20;
const GIB: usize = 1 << 30;

/// #433 as seen on an AC922 (Qwen3.8-Flash-Next over two V100s, `--experts host`, MTP k=2, 16
/// sequences, 131,072-token context), the last stage: 12,960 MiB free, 1,698 MiB of states and
/// the 2 GiB headroom. Its layers take 15,360 bytes per token and the MTP head's own pool 2,048
/// more, which the fit did not count: the pool took 1.2 GiB of the headroom, 0.8 GiB stayed
/// free and a 10K-token prompt step ran out of memory.
#[test]
fn the_mtp_heads_pool_stays_out_of_the_headroom() {
    let room = 12_960 * MIB - 2 * GIB - 1_698 * MIB;
    let per = 15_360 * KV_BLOCK;
    let mtp = 2_048 * KV_BLOCK;
    let (blocks, _) = kv_fit(room, per, CacheShare::None);
    assert!(
        blocks * (per + mtp) > room + 1_000 * MIB,
        "the bug: the headroom is eaten"
    );
    let (blocks, full) = kv_fit(room, per + mtp, CacheShare::None);
    assert!(blocks * (per + mtp) <= room);
    assert_eq!(blocks, full);
}

/// What the fit counts for the MTP head is what its pool allocates, and so for a pool of any
/// one layer (qwen4exp: K, V and, on sparse-attention layers, the indexer keys).
#[test]
fn mtp_block_bytes_match_the_heads_pool() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    qwen4::tiny::write(&path, None, 1).unwrap();
    let hp = Hparams::from_gguf(&GgufFile::open(&path).unwrap()).unwrap();
    let backend = Backend::cpu().unwrap();
    let types = KvTypes::default();
    let layers: Vec<usize> = (0..hp.n_layer)
        .filter(|&il| hp.owns_kv(il))
        .chain([hp.n_layer])
        .collect();
    assert!(layers.iter().any(|&il| hp.arch.indexed(il)));
    for il in layers {
        let pool = KvPool::new_ring(&hp, backend.buffer_type(), 3, &[il], None, types).unwrap();
        assert_eq!(
            KvPool::layer_block_bytes(&hp, il, types),
            pool.bytes_per_block(),
            "layer {il}"
        );
    }
}

/// `--expert-cache-gib 6.8` on an AC922's two V100s: the KV pool took all but the headroom, and
/// the cache's first slots did not fit (cuMemCreate out of memory). The cache's bytes now come
/// out of the KV pool's room; the automatic cache's half and no cache are as before.
#[test]
fn a_given_cache_size_comes_out_of_the_kv_room() {
    let room = 9 * GIB;
    let per = 1 << 20;
    let cache = (6.8 * GIB as f64) as u64;
    let (blocks, full) = kv_fit(room, per, CacheShare::Bytes(cache));
    assert_eq!(full, 9 * 1024);
    assert_eq!(blocks, (room - cache as usize) / per);
    assert_eq!(kv_fit(room, per, CacheShare::Half), (9 * 512, 9 * 1024));
    assert_eq!(kv_fit(room, per, CacheShare::None), (9 * 1024, 9 * 1024));
    // More than the room: no blocks next to it (the load then fails on the cache's size).
    assert_eq!(
        kv_fit(room, per, CacheShare::Bytes(20 << 30)),
        (0, 9 * 1024)
    );
}

/// The automatic size is all that the step buffers leave; a given size up to that is taken as
/// is, and past it the load fails with the GPU, the MiB needed and the MiB free.
#[test]
fn a_cache_size_past_the_free_vram_fails_with_the_numbers() {
    let free = 9 * GIB;
    let keep = 1_700 * MIB;
    let rest = (free - keep) as u64;
    assert_eq!(cache_budget(None, free, keep, "CUDA0"), Ok(rest));
    assert_eq!(cache_budget(Some(rest), free, keep, "CUDA0"), Ok(rest));
    assert_eq!(
        cache_budget(Some(1 << 30), free, keep, "CUDA0"),
        Ok(1 << 30)
    );
    let asked = (14.8 * GIB as f64) as u64;
    let err = cache_budget(Some(asked), free, keep, "CUDA1").unwrap_err();
    assert!(
        err.contains("--expert-cache-gib 14.80 does not fit on CUDA1"),
        "{err}"
    );
    assert!(err.contains("15156 MiB for the cache"), "{err}");
    assert!(
        err.contains("1700 MiB for the step compute buffers"),
        "{err}"
    );
    assert!(err.contains("9216 MiB are free"), "{err}");
    assert!(err.contains("at most 7.34"), "{err}");
    // Less free than the buffers need: the automatic cache is empty.
    assert_eq!(cache_budget(None, keep / 2, keep, "CUDA0"), Ok(0));
}

/// Host experts: the kept graphs get what the load left less the reserve, never "no budget".
#[test]
fn host_experts_budget_is_the_free_vram_less_the_reserve() {
    assert_eq!(free_budget(2 * GIB, GIB), GIB);
    assert_eq!(free_budget(GIB / 2, GIB), 1);
}

/// #827: with `--spec-gate-cost-model off` the MTP head drafts at a full batch, so its passes
/// there get room beside the expert cache; with the gate on (the default), without MTP or
/// without speculation the cache sizing is unchanged.
#[test]
fn mtp_full_batch_only_with_the_gate_off() {
    assert_eq!(mtp_full_batch(true, false, 16), Some(16));
    assert_eq!(mtp_full_batch(true, false, 0), Some(1));
    assert_eq!(mtp_full_batch(true, true, 16), None);
    assert_eq!(mtp_full_batch(false, false, 16), None);
    assert_eq!(mtp_full_batch(false, true, 16), None);
}

/// #814: the MTP gate ends at 8 sequences by default only on a V100 layer split of 2 GPUs
/// with the expert cache and `--spec-max-batch` at most 8, and never with `--spec-fixed`,
/// another drafter, other GPUs, other splits, tensor parallelism or experts outside the cache.
#[test]
fn mtp_gate_ends_at_8_only_on_a_two_v100_split_with_the_cache() {
    let until = |mtp, fixed, volta, cache, stages, tp, max_batch| {
        mtp_gate_until_default(mtp, fixed, volta, cache, stages, tp, max_batch)
    };
    assert_eq!(until(true, false, true, true, 2, false, 8), Some(8));
    assert_eq!(until(true, false, true, true, 2, false, 4), Some(8));
    assert_eq!(until(true, false, true, true, 2, false, 9), None);
    assert_eq!(until(false, false, true, true, 2, false, 8), None);
    assert_eq!(until(true, true, true, true, 2, false, 8), None);
    assert_eq!(until(true, false, false, true, 2, false, 8), None);
    assert_eq!(until(true, false, true, false, 2, false, 8), None);
    assert_eq!(until(true, false, true, true, 1, false, 8), None);
    assert_eq!(until(true, false, true, true, 4, false, 8), None);
    assert_eq!(until(true, false, true, true, 2, true, 8), None);
}

/// #827: the LM head's rows per step. Without speculation, every sequence's last token; with
/// the gate on, drafts only below `--spec-max-batch` sequences; with it off, every sequence
/// drafts `spec_k`.
#[test]
fn head_rows_count_the_drafting_sequences() {
    assert_eq!(max_head_rows(16, 0, false, 8, 0), 16);
    // the AC922 defaults (k 2, gate on at 8): 16 + 7 x 2.
    assert_eq!(max_head_rows(16, 2, false, 8, 0), 30);
    // Gate off: 16 x (k + 1).
    assert_eq!(max_head_rows(16, 2, true, 8, 0), 48);
    assert_eq!(max_head_rows(16, 3, true, 8, 0), 64);
    // Fewer sequences than the gate: all of them draft.
    assert_eq!(max_head_rows(4, 3, false, 8, 0), 16);
    assert_eq!(max_head_rows(16, 3, false, 0, 0), 16);
}

/// `--spec-max-rows 63`: gate off at k 3, 16 sequences draft 2 each (48 rows) and 12 keep 3 (48),
/// within 63 rows, so no step reaches ggml's cuBLAS head; 64 sequences take a row each.
#[test]
fn max_rows_bounds_the_head_rows() {
    assert_eq!(max_head_rows(16, 3, true, 8, 63), 63);
    assert!(max_head_rows(16, 3, true, 8, 63) < CUBLAS_MIN_ROWS);
    assert_eq!(max_head_rows(16, 2, true, 8, 63), 48);
    assert_eq!(max_head_rows(4, 3, true, 8, 63), 16);
    assert_eq!(max_head_rows(64, 3, true, 8, 63), 64);
}

/// #827 on an AC922 (4 V100s, Qwen3.8-Flash-Next, head 248,320 x 2,560 Q8_0): gate off at k 3 and
/// 16 sequences, the 64-row verify step converts the head to F16 in ggml's pool (1,212 MiB with
/// the input). At 48 rows (k 2), with the defaults, on other GPUs and with an F16 head nothing
/// is kept back.
#[test]
fn the_heads_f16_copy_is_kept_free_on_a_v100_from_64_rows() {
    let (elems, k) = (248_320 * 2_560, 2_560);
    let b = head_pool_bytes(true, true, elems, k, 64);
    assert_eq!(b, 2 * elems + 2 * 64 * k);
    assert_eq!(b >> 20, 1_212);
    assert_eq!(head_pool_bytes(true, true, elems, k, 48), 0);
    assert_eq!(head_pool_bytes(true, true, elems, k, 30), 0);
    assert_eq!(head_pool_bytes(false, true, elems, k, 64), 0);
    assert_eq!(head_pool_bytes(true, false, elems, k, 64), 0);
}
