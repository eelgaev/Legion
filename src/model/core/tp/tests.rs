use super::*;

/// Gemma 4 31B's two layer kinds: sliding (256-wide heads, 16 KV heads) and global (512-wide,
/// 4 KV heads), 32 query heads, FFN 21504.
#[test]
fn gemma4_31b_splits_by_heads_and_rows() {
    let layers = vec![(256, 16, 32), (512, 4, 32)];
    let s = |name: &str, ne: [i64; 4]| split_of(&layers, 2, name, ne);
    let ax = |axis, granularity| Split::Axis { axis, granularity };
    assert_eq!(s("blk.0.attn_q.weight", [5376, 8192, 1, 1]), ax(1, 512));
    assert_eq!(s("blk.0.attn_k.weight", [5376, 4096, 1, 1]), ax(1, 256));
    assert_eq!(s("blk.0.attn_v.weight", [5376, 4096, 1, 1]), ax(1, 256));
    assert_eq!(
        s("blk.0.attn_output.weight", [8192, 5376, 1, 1]),
        ax(0, 512)
    );
    assert_eq!(s("blk.1.attn_q.weight", [5376, 16384, 1, 1]), ax(1, 4096));
    assert_eq!(s("blk.1.attn_k.weight", [5376, 2048, 1, 1]), ax(1, 512));
    assert_eq!(
        s("blk.1.attn_output.weight", [16384, 5376, 1, 1]),
        ax(0, 4096)
    );
    assert_eq!(s("blk.1.ffn_up.weight", [5376, 21504, 1, 1]), ax(1, 256));
    assert_eq!(s("blk.1.ffn_down.weight", [21504, 5376, 1, 1]), ax(0, 256));
    assert_eq!(s("cache_k-1", [2048, 4096, 1, 1]), ax(0, 512));
    assert_eq!(s("cache_v-0", [4096, 4096, 1, 1]), ax(0, 256));
    for m in [
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.layer_output_scale.weight",
        "token_embd.weight",
        "output_norm.weight",
        "rope_freqs.weight",
        "kv_blocks-0",
    ] {
        assert_eq!(s(m, [5376, 1, 1, 1]), Split::Mirrored, "{m}");
    }
    // A tensor not shaped as the layer's heads stays whole.
    assert_eq!(
        s("blk.0.attn_q.weight", [5376, 1000, 1, 1]),
        Split::Mirrored
    );
    // An MTP head's global layer under a sliding layer's number (blk.3 of a 4-layer assistant
    // reading the target's global cache): split by its own width.
    assert_eq!(s("blk.0.attn_q.weight", [1024, 16384, 1, 1]), ax(1, 4096));
    assert_eq!(
        s("blk.0.attn_output.weight", [16384, 1024, 1, 1]),
        ax(0, 4096)
    );
    // Over 8 devices the global layers' 4 KV heads don't divide: their attention stays whole,
    // the sliding layers' 16 split.
    let s8 = |name: &str, ne: [i64; 4]| split_of(&layers, 8, name, ne);
    assert_eq!(
        s8("blk.1.attn_q.weight", [5376, 16384, 1, 1]),
        Split::Mirrored
    );
    assert_eq!(s8("cache_k-1", [2048, 4096, 1, 1]), Split::Mirrored);
    assert_eq!(s8("blk.0.attn_k.weight", [5376, 4096, 1, 1]), ax(1, 256));
}

/// A layer with uneven KV heads whose query width equals another layer's (which splits): its
/// own queries and output stay whole, as its cache, and do not take the other layer's split.
/// A tensor not shaped as its layer's queries (an MTP head's) still finds a split by width.
#[test]
fn uneven_layers_keep_their_queries_whole() {
    // Layer 0: 6 KV heads of 256 (uneven over 4); layer 1: 4 KV heads of 512 (even). Both
    // 8192-wide queries.
    let layers = vec![(256, 6, 32), (512, 4, 16)];
    let s = |name: &str, ne: [i64; 4]| split_of(&layers, 4, name, ne);
    let ax = |axis, granularity| Split::Axis { axis, granularity };
    assert_eq!(
        s("blk.0.attn_q.weight", [5376, 8192, 1, 1]),
        Split::Mirrored
    );
    assert_eq!(
        s("blk.0.attn_output.weight", [8192, 5376, 1, 1]),
        Split::Mirrored
    );
    assert_eq!(
        s("blk.0.attn_k.weight", [5376, 1536, 1, 1]),
        Split::Mirrored
    );
    assert_eq!(s("cache_k-0", [1536, 4096, 1, 1]), Split::Mirrored);
    assert_eq!(s("blk.1.attn_q.weight", [5376, 8192, 1, 1]), ax(1, 2048));
    assert_eq!(s("cache_k-1", [2048, 4096, 1, 1]), ax(0, 512));
    // An MTP layer numbered 0 whose queries are shaped as layer 1's (not layer 0's, 4096
    // wide here) reads layer 1's cache: layer 1's split.
    let mtp = |name: &str, ne: [i64; 4]| split_of(&[(256, 4, 16), (512, 4, 16)], 4, name, ne);
    assert_eq!(mtp("blk.0.attn_q.weight", [1024, 8192, 1, 1]), ax(1, 2048));
}

/// Layers whose KV heads do not divide among the devices (kept whole: `--tp` rejects them,
/// as its cache fit assumes every cache split): Gemma 4 31B's global layers over 3 or 8.
#[test]
fn uneven_kv_heads_are_listed() {
    let layers = vec![(256, 16, 32), (512, 4, 32), (256, 16, 32)];
    assert!(uneven(&layers, 2).is_empty());
    assert!(uneven(&layers, 4).is_empty());
    assert_eq!(uneven(&layers, 3), vec![0, 1, 2]);
    assert_eq!(uneven(&layers, 8), vec![1]);
}

/// Gemma 4's compact global caches (#551) split by kv heads in granules of `head_dim + r`;
/// every other tensor (plain caches, a compact layer's weights) is left to [`split_of`].
#[test]
fn compact_caches_split_by_granules() {
    let layers = vec![(256, 16, 32), (512, 4, 32), (512, 3, 32)];
    let c = |name: &str, ne0: i64| compact_split(&layers, 2, name, [ne0, 1024, 1, 1]);
    let ax = |axis, granularity| Split::Axis { axis, granularity };
    assert_eq!(c("cache_k-1", 640 * 4), Some(ax(0, 640)));
    assert_eq!(c("cache_k-2", 640 * 3), Some(Split::Mirrored));
    assert_eq!(c("cache_k-1", 512 * 4), None);
    assert_eq!(c("cache_k-0", 256 * 16), None);
    assert_eq!(c("cache_v-1", 512 * 4), None);
    assert_eq!(c("blk.1.attn_k.weight", 640 * 4), None);
    assert_eq!(c("cache_k-7", 640 * 4), None);
    assert_eq!(
        split_of(&layers, 2, "cache_k-1", [2048, 1024, 1, 1]),
        ax(0, 512)
    );
}
