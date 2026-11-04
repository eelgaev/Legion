use super::*;

fn rates() -> KvRates {
    KvRates {
        full: 100,
        sliding: 10,
        window: 128,
        sparse: 0,
        pool_key: 0,
        sparse_keys: None,
    }
}

/// Full layers read every position, sliding ones the window and the step's tokens.
#[test]
fn dense_and_sliding_layers() {
    let r = rates();
    let blocks: Vec<u32> = (0..64).collect();
    assert_eq!(
        r.seq(999, 1, &blocks, true, None),
        (100 * 1000 + 10 * 129, 0)
    );
    // Short contexts read what there is.
    assert_eq!(r.seq(10, 5, &blocks, true, None), (100 * 15 + 10 * 15, 0));
    // A sequence with no token this step reads nothing.
    assert_eq!(r.seq(10, 0, &blocks, true, None), (0, 0));
}

/// The host tier's share is the share of the positions in its blocks; rings stay in VRAM.
#[test]
fn host_tier_share() {
    let r = rates();
    // Two blocks of four in the tier (ids from 10 on).
    let blocks = [0u32, 11, 1, 12];
    let ctx = 4 * KV_BLOCK;
    let (dev, host) = r.seq(ctx - 1, 1, &blocks, true, Some(10));
    assert_eq!(host, 100 * ctx as u64 / 2);
    assert_eq!(dev + host, 100 * ctx as u64 + 10 * 129);
    // Only the blocks the context covers count.
    let (_, host) = r.seq(KV_BLOCK - 1, 1, &blocks, true, Some(10));
    assert_eq!(host, 0);
}

/// Sparse layers read each token's selection (at most) and every pool's key when paged, the
/// whole span when gathered.
#[test]
fn sparse_layers() {
    let r = KvRates {
        full: 0,
        sliding: 0,
        window: 0,
        sparse: 50,
        pool_key: 4,
        sparse_keys: Some((256, 64)),
    };
    let blocks: Vec<u32> = (0..2000).collect();
    let ctx = 100_000;
    assert_eq!(
        r.seq(ctx - 3, 3, &blocks, true, None),
        (50 * 3 * (256 + 64) + 4 * (ctx / 64) as u64, 0)
    );
    assert_eq!(
        r.seq(ctx - 3, 3, &blocks, false, None),
        (50 * ctx as u64 + 4 * (ctx / 64) as u64, 0)
    );
    // A prompt chunk's selections cover the context at most.
    assert_eq!(
        r.seq(ctx - 4096, 4096, &blocks, true, None).0,
        50 * ctx as u64 + 4 * (ctx / 64) as u64
    );
}
