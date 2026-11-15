use super::*;
use crate::engine::saved_states::Kind;
use crate::kv::BlockTable;

const MIB: usize = 1 << 20;

/// Two cache tensors of 1 KiB and 2 KiB rows whose blocks 8.. are lent, in 1 MiB pieces:
/// 16 blocks a piece for the first, 8 for the second.
fn tails() -> Vec<Tail> {
    let lent = |row: usize, n: usize| Tail {
        row,
        pieces: (0..n)
            .map(|i| (8 * KV_BLOCK * row + i * MIB, MIB))
            .collect(),
    };
    vec![lent(1024, 4), lent(2048, 8)]
}

#[test]
fn only_the_pieces_holding_blocks_in_use_move() {
    let t = tails();
    // Nothing held: nothing kept, as a lend without host memory.
    assert_eq!(
        host_plan(&t, &[]),
        HostPlan {
            offsets: vec![vec![], vec![]],
            bytes: 0
        }
    );
    // Block 8 (the first lent one) and block 30.
    let p = host_plan(&t, &[8, 30]);
    let (a, b) = (8 * KV_BLOCK * 1024, 8 * KV_BLOCK * 2048);
    // First tensor: blocks 8..24 in piece 0, 24..40 in piece 1.
    assert_eq!(p.offsets[0], vec![a, a + MIB]);
    // Second: blocks 8..16 in piece 0, 24..32 in piece 2.
    assert_eq!(p.offsets[1], vec![b, b + 2 * MIB]);
    assert_eq!(p.bytes, 4 * MIB);
}

/// A pool of 8 fast blocks and 56 past them, with a radix cache and saved states, as the
/// worker holds them.
struct Held {
    pool: BlockPool,
    radix: RadixCache,
    saved: SavedStates,
    /// Warm sequences: (stamp, table).
    warm: Vec<(u64, BlockTable)>,
    evictions: usize,
    /// The kept blocks, all in use: what a chat caches next lands past them.
    _busy: BlockTable,
}

impl Held {
    fn new() -> Self {
        let mut pool = BlockPool::with_fast(64, KV_BLOCK as u32, 8);
        let mut busy = BlockTable::new();
        busy.append(&mut pool, 8 * KV_BLOCK as u32).unwrap();
        Self {
            pool,
            radix: RadixCache::new(KV_BLOCK as u32),
            saved: SavedStates::new(4),
            warm: Vec::new(),
            evictions: 0,
            _busy: busy,
        }
    }

    /// A finished chat of `blocks` full blocks and a partly filled one: its full blocks in
    /// the radix cache, its state with the tail block in the saved states.
    fn chat(&mut self, first_token: u32, blocks: usize) -> Vec<u32> {
        let tokens: Vec<u32> = (0..(blocks * KV_BLOCK + 3) as u32)
            .map(|i| first_token + i)
            .collect();
        let mut table = BlockTable::new();
        table.append(&mut self.pool, tokens.len() as u32).unwrap();
        let ids = table.blocks().to_vec();
        self.radix.insert(&mut self.pool, &tokens, &ids);
        let tail = ids[blocks];
        self.pool.incref(tail);
        self.saved.store(&tokens, Kind::Anchor, Some(tail)).unwrap();
        table.release(&mut self.pool);
        ids.iter().map(|b| b.0).collect()
    }

    fn evict(&mut self) -> bool {
        self.evictions += 1;
        let warm = &mut self.warm;
        evict_tail_holder(
            &mut self.pool,
            Some(&mut self.radix),
            &mut self.saved,
            8,
            |pool| {
                let Some(i) = (0..warm.len()).min_by_key(|&i| warm[i].0) else {
                    return false;
                };
                warm.swap_remove(i).1.release(pool);
                true
            },
        )
    }
}

/// One tensor of 1 KiB rows, blocks 8..64 lent in 64 KiB pieces (one block each): a
/// piece's bytes are a block's.
fn block_tails() -> Vec<Tail> {
    let row = 1024;
    let bytes = KV_BLOCK * row;
    vec![Tail {
        row,
        pieces: (8..64).map(|b| (b * bytes, bytes)).collect(),
    }]
}

#[test]
fn chats_that_fit_the_budget_stay_whole() {
    let tails = block_tails();
    let block = KV_BLOCK * 1024;
    let mut h = Held::new();
    let a = h.chat(1000, 5);
    let b = h.chat(5000, 7);
    // 6 + 8 blocks held past the kept ones.
    assert_eq!(h.pool.slow_in_use(), 14);
    let plan = fit_host(
        14 * block,
        &mut h,
        |h| host_plan(&tails, &h.pool.slow_used()),
        Held::evict,
    )
    .unwrap();
    assert_eq!(h.evictions, 0);
    assert_eq!(plan.bytes, 14 * block);
    // Every block of both chats is kept, the tails included.
    let kept: Vec<usize> = plan.offsets[0].iter().map(|o| o / block).collect();
    let mut all: Vec<usize> = a.iter().chain(&b).map(|&x| x as usize).collect();
    all.sort_unstable();
    assert_eq!(kept, all);
    // The prefixes still match their blocks: a resume finds them.
    let m = h
        .radix
        .match_prefix(&(5000..5000 + 7 * KV_BLOCK as u32).collect::<Vec<_>>());
    assert_eq!(m.blocks.iter().map(|x| x.0).collect::<Vec<_>>(), b[..7]);
    h.radix.unlock(&m);
}

#[test]
fn the_least_recently_used_chat_goes_first_when_the_budget_is_short() {
    let tails = block_tails();
    let block = KV_BLOCK * 1024;
    let mut h = Held::new();
    let a = h.chat(1000, 5);
    let b = h.chat(5000, 7);
    // `a` is resumed again: `b` is now the least recently used.
    let m = h
        .radix
        .match_prefix(&(1000..1000 + 5 * KV_BLOCK as u32).collect::<Vec<_>>());
    h.radix.unlock(&m);
    // Room for `a`'s 6 blocks and a few more, not for `b`'s 8.
    let plan = fit_host(
        10 * block,
        &mut h,
        |h| host_plan(&tails, &h.pool.slow_used()),
        Held::evict,
    )
    .unwrap();
    assert!(plan.bytes <= 10 * block);
    // `a` stays whole (its full blocks; its tail block too until the saved tails go).
    let m = h
        .radix
        .match_prefix(&(1000..1000 + 5 * KV_BLOCK as u32).collect::<Vec<_>>());
    assert_eq!(m.blocks.iter().map(|x| x.0).collect::<Vec<_>>(), a[..5]);
    h.radix.unlock(&m);
    let kept: Vec<u32> = plan.offsets[0].iter().map(|o| (o / block) as u32).collect();
    assert!(a[..5].iter().all(|x| kept.contains(x)));
    // `b`'s prefix is gone.
    let m = h
        .radix
        .match_prefix(&(5000..5000 + 7 * KV_BLOCK as u32).collect::<Vec<_>>());
    assert!(m.blocks.is_empty());
    h.radix.unlock(&m);
    assert!(b[..7].iter().all(|x| !kept.contains(x)));
}

#[test]
fn warm_sequences_and_saved_tails_go_after_the_radix_entries() {
    let tails = block_tails();
    let block = KV_BLOCK * 1024;
    let mut h = Held::new();
    h.chat(1000, 2);
    // Two warm sequences of 3 blocks each past the kept ones; the first finished first.
    for stamp in [1, 2] {
        let mut t = BlockTable::new();
        t.append(&mut h.pool, 3 * KV_BLOCK as u32).unwrap();
        h.warm.push((stamp, t));
    }
    assert_eq!(h.pool.slow_in_use(), 3 + 6);
    // Room for one warm sequence: the radix entry, the saved tail and the older warm
    // sequence go, in that order.
    let plan = fit_host(
        3 * block,
        &mut h,
        |h| host_plan(&tails, &h.pool.slow_used()),
        Held::evict,
    )
    .unwrap();
    assert_eq!(plan.bytes, 3 * block);
    assert_eq!(h.evictions, 3);
    assert_eq!(h.warm.len(), 1);
    assert_eq!(h.warm[0].0, 2);
    assert_eq!(h.radix.cached_blocks(), 0);
}

#[test]
fn a_budget_nothing_fits_in_falls_back_to_evicting_everything() {
    let tails = block_tails();
    let block = KV_BLOCK * 1024;
    let mut h = Held::new();
    h.chat(1000, 3);
    // A block held by something no eviction reaches (a sequence still running).
    let mut running = BlockTable::new();
    running.append(&mut h.pool, KV_BLOCK as u32).unwrap();
    assert!(running.blocks()[0].0 >= 8);
    let plan = fit_host(
        0,
        &mut h,
        |h| host_plan(&tails, &h.pool.slow_used()),
        Held::evict,
    );
    assert_eq!(plan, None);
    // Every cached holder went: the shrink then evicts as without host memory.
    assert_eq!(h.radix.cached_blocks(), 0);
    assert_eq!(h.pool.slow_in_use(), 1);
    // With the running block's piece within the budget, it is kept.
    let plan = fit_host(
        block,
        &mut h,
        |h| host_plan(&tails, &h.pool.slow_used()),
        Held::evict,
    )
    .unwrap();
    assert_eq!(
        plan.offsets[0],
        vec![running.blocks()[0].0 as usize * block]
    );
    running.release(&mut h.pool);
}

#[test]
fn a_zero_budget_with_nothing_held_lends_as_before() {
    let mut h = Held::new();
    let plan = fit_host(
        0,
        &mut h,
        |h| host_plan(&block_tails(), &h.pool.slow_used()),
        Held::evict,
    )
    .unwrap();
    assert_eq!(plan.bytes, 0);
    assert_eq!(h.evictions, 0);
}

#[test]
fn the_host_tier_lends_the_tail_with_no_flag_only_on_its_one_gpu() {
    assert!(tier_default(true, None, true, 1, false, 0));
    // No bounded host tier (an AC922, a Spark): as before.
    assert!(!tier_default(false, None, true, 1, false, 0));
    // --kv-vmm and --no-kv-vmm decide for themselves.
    assert!(!tier_default(true, Some(false), true, 1, false, 0));
    assert!(!tier_default(true, Some(true), true, 1, false, 0));
    // No VRAM expert cache, a layer split, tensor parallelism, a KV host tier.
    assert!(!tier_default(true, None, false, 1, false, 0));
    assert!(!tier_default(true, None, true, 2, false, 0));
    assert!(!tier_default(true, None, true, 1, true, 0));
    assert!(!tier_default(true, None, true, 1, false, 4096));
}

#[test]
fn the_host_tiers_mapped_tail_waits_for_a_cached_chats_next_turn() {
    let s = std::time::Duration::from_secs;
    assert!(tier_holds(true, true, s(0)));
    assert!(tier_holds(true, true, TIER_HOLD - s(1)));
    // Idle long enough: lent again (the chat moves to host memory).
    assert!(!tier_holds(true, true, TIER_HOLD));
    // Nothing cached on it: lent again at once.
    assert!(!tier_holds(true, false, s(0)));
    // --kv-vmm as a flag, an AC922: as before.
    assert!(!tier_holds(false, true, s(0)));
}

#[test]
fn the_host_tier_counts_the_blocks_its_sequences_will_still_take() {
    assert_eq!(tier_blocks_ahead(std::iter::empty()), 0);
    // A new 7.9K prompt with a 600-token reply (34 blocks, none held yet) beside a chat that
    // holds 3 of its 5: 36 to come, more than 32 kept blocks hold.
    assert_eq!(tier_blocks_ahead([(34, 0), (5, 3)].into_iter()), 36);
    // A sequence that holds more than it was admitted for (a resumed prefix) counts as none.
    assert_eq!(tier_blocks_ahead([(2, 4), (1, 0)].into_iter()), 1);
}

#[test]
fn a_run_of_long_chats_turns_the_next_loads_lend_off_and_short_ones_on_again() {
    let dir = tempfile::tempdir().unwrap();
    let heat = dir.path().join("legion/expert-heat/m.gguf.1.heat");
    let file = tier_long_file(&heat);
    assert_eq!(
        file,
        dir.path().join("legion/expert-heat/m.gguf.1.long-chats")
    );
    let run = |long: u64, short: u64, file: Option<&std::path::Path>| {
        let mut u = TierUse {
            file: file.map(Into::into),
            total: 160,
            keep: 32,
            ..Default::default()
        };
        // Idle turns count nothing; 33 blocks in use is past the 32 kept, 32 is not.
        u.note(0, 0);
        (0..long).for_each(|_| u.note(1, 160 - 33));
        (0..short).for_each(|_| u.note(2, 160 - 32));
        assert_eq!(
            (u.long, u.steps),
            if file.is_some() {
                (long, long + short)
            } else {
                (0, 0)
            }
        );
    };
    // Too few steps to tell: nothing written.
    run(TIER_USE_STEPS - 1, 0, Some(&file));
    assert!(!file.exists());
    // Mostly long: noted (its directory made).
    run(TIER_USE_STEPS, TIER_USE_STEPS - 1, Some(&file));
    assert!(file.exists());
    // Too few steps: it stays. Half or fewer long: removed, and removing nothing is fine.
    run(0, 3, Some(&file));
    assert!(file.exists());
    run(TIER_USE_STEPS, TIER_USE_STEPS, Some(&file));
    assert!(!file.exists());
    run(0, TIER_USE_STEPS, Some(&file));
    // No bounded host tier: nothing counted or written.
    run(TIER_USE_STEPS, 0, None);
    assert!(!file.exists());
    assert_eq!(tier_chats_long(1, 2), None);
    assert_eq!(tier_chats_long(TIER_USE_STEPS, TIER_USE_STEPS), Some(true));
}

#[test]
fn the_default_host_budget_is_an_eighth_of_available_memory_at_most_the_tail() {
    let gib = 1usize << 30;
    // 574 GB free, a 119 GiB tail: an eighth of the memory.
    assert_eq!(default_host_budget(119 * gib, 560 * gib as u64), 70 * gib);
    // A small tail fits whole.
    assert_eq!(default_host_budget(3 * gib, 560 * gib as u64), 3 * gib);
    // Unreadable meminfo (0): nothing.
    assert_eq!(default_host_budget(3 * gib, 0), 0);
}
