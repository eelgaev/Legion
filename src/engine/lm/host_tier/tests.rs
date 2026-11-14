use super::*;

const GIB: u64 = 1 << 30;

fn node(block_bytes: u64, free_gib: u64, total_gib: u64) -> Node {
    Node {
        node: 0,
        block_bytes,
        free: free_gib * GIB,
        cache: 0,
        total: total_gib * GIB,
        planned: 0,
    }
}

#[test]
fn no_tier_when_a_context_fits() {
    assert_eq!(
        auto_host_blocks(100, 64, 400, 100, &[node(1 << 20, 100, 200)]),
        Ok(0)
    );
}

#[test]
fn the_rest_of_the_pool_goes_to_the_host() {
    // 4 sequences of 4096 blocks wanted, 1200 on the device, 1 MiB per block, a node with
    // 200 GiB free of 256: the margin is 25.6 GiB, so ~178 GiB of room: all 15,184 fit.
    let n = node(1 << 20, 200, 256);
    assert_eq!(
        auto_host_blocks(1200, 4096, 4 * 4096, 1200, &[n]),
        Ok(4 * 4096 - 1200)
    );
}

#[test]
fn capped_by_the_tightest_node() {
    // 50 GiB free of 64 on one node: margin 16 GiB leaves 34 GiB, but at most half the node
    // (32 GiB = 32,768 blocks of 1 MiB) is pinned; the other node has more.
    let tight = node(1 << 20, 50, 64);
    let loose = Node {
        node: 8,
        ..node(1 << 20, 400, 512)
    };
    assert_eq!(
        auto_host_blocks(1000, 4096, 100_000, 1000, &[loose, tight]),
        Ok(32 * 1024)
    );
}

#[test]
fn an_error_when_not_even_a_context_fits() {
    let small = node(1 << 30, 20, 32);
    let err = auto_host_blocks(1000, 4096, 4096, 1000, &[small]).unwrap_err();
    assert!(err.contains("does not fit in GPU memory"), "{err}");
    assert!(err.contains("NUMA node(s) 0"), "{err}");
    assert!(
        err.contains("node 0: 20.0 GiB free of 32.0 GiB, 0.0 GiB of it page cache"),
        "{err}"
    );
}

#[test]
fn the_error_says_how_much_is_page_cache() {
    let n = Node {
        cache: 12 * GIB,
        ..node(1 << 30, 20, 32)
    };
    let err = auto_host_blocks(1000, 4096, 4096, 1000, &[n]).unwrap_err();
    assert!(err.contains("12.0 GiB of it page cache"), "{err}");
}

fn mem(free_kb: u64, cache_kb: u64, total_kb: u64) -> Option<NodeMemory> {
    Some(NodeMemory {
        free: free_kb * 1024,
        cache: cache_kb * 1024,
        total: total_kb * 1024,
    })
}

#[test]
fn meminfo_parses() {
    let text = "Node 0 MemTotal:       263443456 kB\nNode 0 MemFree:        120000000 kB\n\
                Node 0 MemUsed:        143443456 kB\n";
    // No file lines: only MemFree.
    assert_eq!(parse_meminfo(text), mem(120_000_000, 0, 263_443_456));
    assert_eq!(parse_meminfo("nothing"), None);
}

#[test]
fn proc_meminfo_parses() {
    // The whole host's file (`/proc/meminfo`): a GPU with no NUMA node.
    let text = "MemTotal:       1000000 kB\nMemFree:        5000 kB\nMemAvailable:   9000 kB\n\
                Active(file):   300 kB\nInactive(file): 200 kB\nDirty:          40 kB\n\
                HugePages_Total:       0\n";
    assert_eq!(parse_meminfo(text), mem(5460, 460, 1_000_000));
}

#[test]
fn meminfo_counts_clean_page_cache() {
    let text = "Node 0 MemTotal:       263443456 kB\nNode 0 MemFree:        1000 kB\n\
                Node 0 Active(file):   300 kB\nNode 0 Inactive(file): 200 kB\n\
                Node 0 Dirty:          40 kB\nNode 0 Writeback:      10 kB\n";
    assert_eq!(parse_meminfo(text), mem(1450, 450, 263_443_456));
}

#[test]
fn meminfo_counts_unmapped_page_cache() {
    // The AC922 case of #541: 97 GiB free, 152 GB of file pages, little of it mapped.
    let text = "Node 0 MemTotal:       263443456 kB\nNode 0 MemFree:        101711872 kB\n\
                Node 0 Active(file):   50000000 kB\nNode 0 Inactive(file): 98000000 kB\n\
                Node 0 Mapped:         1000000 kB\n";
    assert_eq!(
        parse_meminfo(text),
        mem(101_711_872 + 147_000_000, 147_000_000, 263_443_456)
    );
}

#[test]
fn meminfo_mapped_past_the_file_pages_counts_no_cache() {
    let text = "Node 0 MemTotal:       1000000 kB\nNode 0 MemFree:        5000 kB\n\
                Node 0 Active(file):   300 kB\nNode 0 Inactive(file): 200 kB\n\
                Node 0 Mapped:         900 kB\n";
    assert_eq!(parse_meminfo(text), mem(5000, 0, 1_000_000));
}

#[test]
fn meminfo_free_never_passes_the_total() {
    let text = "Node 0 MemTotal:       1000 kB\nNode 0 MemFree:        800 kB\n\
                Node 0 Active(file):   300 kB\nNode 0 Inactive(file): 200 kB\n";
    assert_eq!(parse_meminfo(text), mem(1000, 200, 1000));
}

/// A whole per-node meminfo as Linux 6.x writes it on a two-socket POWER9 (AC922): node 0 is
/// mostly page cache of the model files (the #541 case), node 8 almost empty. Fields the sum
/// does not read (`Active:`, `FilePages:`, `Shmem:`, `HugePages_*` without a unit) must not
/// match the ones it does.
const POWER9_NODE0: &str = "\
Node 0 MemTotal:       263443456 kB
Node 0 MemFree:        101711872 kB
Node 0 MemUsed:        161731584 kB
Node 0 SwapCached:             0 kB
Node 0 Active:          60000000 kB
Node 0 Inactive:        99000000 kB
Node 0 Active(anon):     3000000 kB
Node 0 Inactive(anon):   1600000 kB
Node 0 Active(file):    57000000 kB
Node 0 Inactive(file):  97400000 kB
Node 0 Unevictable:         2048 kB
Node 0 Mlocked:             2048 kB
Node 0 Dirty:               1200 kB
Node 0 Writeback:              0 kB
Node 0 FilePages:      154600000 kB
Node 0 Mapped:            900000 kB
Node 0 AnonPages:        4600000 kB
Node 0 Shmem:             200000 kB
Node 0 KernelStack:        30000 kB
Node 0 PageTables:         40000 kB
Node 0 SecPageTables:          0 kB
Node 0 NFS_Unstable:           0 kB
Node 0 Bounce:                 0 kB
Node 0 WritebackTmp:           0 kB
Node 0 KReclaimable:     1500000 kB
Node 0 Slab:             2500000 kB
Node 0 SReclaimable:     1500000 kB
Node 0 SUnreclaim:       1000000 kB
Node 0 AnonHugePages:          0 kB
Node 0 ShmemHugePages:         0 kB
Node 0 ShmemPmdMapped:         0 kB
Node 0 FileHugePages:          0 kB
Node 0 FilePmdMapped:          0 kB
Node 0 HugePages_Total:     0
Node 0 HugePages_Free:      0
Node 0 HugePages_Surp:      0
";

const POWER9_NODE8: &str = "\
Node 8 MemTotal:       263602048 kB
Node 8 MemFree:        250000000 kB
Node 8 MemUsed:         13602048 kB
Node 8 Active:           2000000 kB
Node 8 Inactive:         3000000 kB
Node 8 Active(anon):     1600000 kB
Node 8 Inactive(anon):   1900000 kB
Node 8 Active(file):      400000 kB
Node 8 Inactive(file):   1100000 kB
Node 8 Dirty:                100 kB
Node 8 Writeback:              0 kB
Node 8 FilePages:        1500000 kB
Node 8 Mapped:            300000 kB
Node 8 AnonPages:        3500000 kB
Node 8 HugePages_Total:     0
Node 8 HugePages_Free:      0
";

/// GB10 (DGX Spark): one node, its 128 GB shared with the GPU. A model loaded by mmap is mapped
/// file cache, which the kernel cannot drop while it is mapped, so it is not counted.
const GB10_NODE0: &str = "\
Node 0 MemTotal:       128452612 kB
Node 0 MemFree:         20000000 kB
Node 0 MemUsed:        108452612 kB
Node 0 Active:          38000000 kB
Node 0 Inactive:        64000000 kB
Node 0 Active(anon):     8000000 kB
Node 0 Inactive(anon):   4000000 kB
Node 0 Active(file):    30000000 kB
Node 0 Inactive(file):  60000000 kB
Node 0 Dirty:              20000 kB
Node 0 Writeback:              0 kB
Node 0 FilePages:       90500000 kB
Node 0 Mapped:          75000000 kB
Node 0 AnonPages:       12000000 kB
Node 0 Shmem:             500000 kB
Node 0 HugePages_Total:     0
";

#[test]
fn meminfo_power9_two_nodes() {
    // Node 0: 57,000,000 + 97,400,000 file kB less 900,000 mapped and 1,200 dirty.
    let cache0 = 57_000_000 + 97_400_000 - 900_000 - 1_200;
    assert_eq!(
        parse_meminfo(POWER9_NODE0),
        mem(101_711_872 + cache0, cache0, 263_443_456)
    );
    // Node 8: little page cache, so about its MemFree, as before #541.
    let cache8 = 400_000 + 1_100_000 - 300_000 - 100;
    assert_eq!(
        parse_meminfo(POWER9_NODE8),
        mem(250_000_000 + cache8, cache8, 263_602_048)
    );
}

#[test]
fn meminfo_power9_page_cache_lets_the_context_fit() {
    // The #541 abort: a context needing ~98 GiB of 1 MiB host blocks on node 0, which has
    // 97 GiB free (72 GiB past the margin). With half its 146 GiB of page cache counted, node
    // 0 has room (a 125.6 GiB budget, half the node) and node 8 caps nothing.
    let as_node = |id: u32, text: &str| {
        let m = parse_meminfo(text).unwrap();
        Node {
            node: id,
            block_bytes: 1 << 20,
            free: m.free,
            cache: m.cache,
            total: m.total,
            planned: 0,
        }
    };
    let ctx_blocks = 120_000;
    let device = 20_000;
    let only_free = Node {
        free: 101_711_872 * 1024,
        cache: 0,
        ..as_node(0, POWER9_NODE0)
    };
    assert!(auto_host_blocks(device, ctx_blocks, ctx_blocks, device, &[only_free]).is_err());
    let nodes = [as_node(0, POWER9_NODE0), as_node(8, POWER9_NODE8)];
    assert_eq!(
        auto_host_blocks(device, ctx_blocks, ctx_blocks, device, &nodes),
        Ok(ctx_blocks - device)
    );
}

#[test]
fn meminfo_gb10_mapped_model_is_not_free() {
    // 90,000,000 kB of file pages, 75,000,000 of them the mapped model, 20,000 dirty.
    let cache = 30_000_000 + 60_000_000 - 75_000_000 - 20_000;
    assert_eq!(
        parse_meminfo(GB10_NODE0),
        mem(20_000_000 + cache, cache, 128_452_612)
    );
}

#[test]
fn the_margin_applies_to_free_plus_cache() {
    // 40 GiB free + 100 GiB page cache of 256: half the cache counts, 90 GiB, less a margin
    // of 25.6 GiB: ~64.4 GiB of 1 MiB blocks (under the 128 GiB cap).
    let n = Node {
        cache: 100 * GIB,
        ..node(1 << 20, 140, 256)
    };
    let margin = (256.0 * GIB as f64 * 0.10) as u64;
    let cap = ((90 * GIB - margin) >> 20) as usize;
    assert_eq!(auto_host_blocks(1000, 4096, 1_000_000, 1000, &[n]), Ok(cap));
}

#[test]
fn speed_note_per_link() {
    let nvlink = speed_note(TierLink::Nvlink);
    assert!(nvlink.starts_with("host blocks are read over the GPU's NVLink to the CPU (~68.5 GB/s measured on an AC922)"), "{nvlink}");
    let pcie = speed_note(TierLink::Pcie);
    assert!(
        pcie.starts_with("host blocks are read over PCIe (~25 GB/s"),
        "{pcie}"
    );
    // A GH200 never gets the AC922's figure.
    let c2c = speed_note(TierLink::C2c);
    assert!(
        c2c.contains("NVLink-C2C") && c2c.contains("~375 GB/s"),
        "{c2c}"
    );
    assert!(!c2c.contains("68.5"), "{c2c}");
}

fn gib(x: f64) -> u64 {
    (x * GIB as f64) as u64
}

fn as_gib(b: u64) -> f64 {
    b as f64 / GIB as f64
}

/// Node 0 of the 2026-11-14 hang (Qwen3.8-27B, `--tp 2 --tp-per-gpu`, 16 x 262,144 tokens, both
/// GPUs on node 0 under `--numa` bind 0): 240.0 GiB "free" of 255.5, 182.2 GiB of it page
/// cache.
fn hang_node(block_bytes: u64) -> Node {
    Node {
        node: 0,
        block_bytes,
        free: gib(240.0),
        cache: gib(182.2),
        total: gib(255.5),
        planned: 0,
    }
}

/// 34,816 bytes per token per rank (the 27B's KV with its MTP head's): one host block.
const HANG_RANK_BLOCK: u64 = 34_816 * KV_BLOCK as u64;

#[test]
fn page_cache_counts_half() {
    // 100 GiB "free" that is all page cache, of 256: 50 GiB counted, less 25.6 GiB margin.
    let n = Node {
        cache: 100 * GIB,
        ..node(1, 100, 256)
    };
    let margin = (256.0 * GIB as f64 * 0.10) as u64;
    assert_eq!(node_budget(&n), 50 * GIB - margin);
    // Page cache past "free" (a racy read) counts as no more than "free".
    let odd = Node {
        cache: 300 * GIB,
        ..node(1, 100, 256)
    };
    assert_eq!(node_budget(&odd), 50 * GIB - margin);
}

#[test]
fn at_most_half_a_node_is_pinned() {
    // An empty node: 250 GiB free of 256, the margin leaves 224.4, the cap 128.
    assert_eq!(node_budget(&node(1, 250, 256)), 128 * GIB);
    assert_eq!(node_cap(256 * GIB), 128 * GIB);
}

#[test]
fn the_hang_budget() {
    // Before: 240.0 - 25.55 = 214.45 GiB for the pair, 114 GiB pinned per rank, 228 GiB of
    // the node. Now: 57.8 GiB of real free + half of 182.2 GiB page cache - 25.55 GiB margin =
    // 123.35 GiB, under the 127.75 GiB cap.
    let budget = as_gib(node_budget(&hang_node(HANG_RANK_BLOCK)));
    assert!((budget - 123.35).abs() < 0.01, "{budget}");
}

#[test]
fn two_ranks_on_one_node_split_the_budget() {
    // The hang's pair: both GPUs on node 0 (a bind, or both GPUs of one socket), so each
    // rank's host block counts both ranks' bytes and each rank sizes half the node's budget.
    let ctx = 262_144 / KV_BLOCK;
    let want = 16 * ctx;
    let device = 600;
    let pair = [hang_node(2 * HANG_RANK_BLOCK)];
    let per_rank = auto_host_blocks(device, ctx, want, device, &pair).unwrap();
    let alone = auto_host_blocks(device, ctx, want, device, &[hang_node(HANG_RANK_BLOCK)]).unwrap();
    assert_eq!(per_rank, alone / 2);
    // ~1.90M tokens and ~61.7 GiB per rank, the pair's 123.35 GiB under half the node.
    let tokens = per_rank * KV_BLOCK;
    assert!((1_895_000..1_905_000).contains(&tokens), "{tokens}");
    let both = 2 * per_rank as u64 * HANG_RANK_BLOCK;
    assert!(both <= node_cap(gib(255.5)), "{}", as_gib(both));
    assert!((as_gib(both) - 123.35).abs() < 0.01, "{}", as_gib(both));
    // Every context still fits: the cut is the pool, never one sequence.
    assert!(per_rank >= ctx - device);
}

#[test]
fn ranks_on_two_nodes_each_take_their_node() {
    // No bind and the pair's GPUs on two sockets: each rank's tier is on its own node, with
    // only its own bytes there, and sizes from that node's whole budget.
    let ctx = 262_144 / KV_BLOCK;
    let n8 = Node {
        node: 8,
        ..hang_node(HANG_RANK_BLOCK)
    };
    let r0 = auto_host_blocks(600, ctx, 16 * ctx, 600, &[hang_node(HANG_RANK_BLOCK)]).unwrap();
    let r1 = auto_host_blocks(600, ctx, 16 * ctx, 600, &[n8]).unwrap();
    assert_eq!(r0, r1);
    let shared = auto_host_blocks(600, ctx, 16 * ctx, 600, &[hang_node(2 * HANG_RANK_BLOCK)]);
    assert_eq!(shared.unwrap(), r0 / 2);
}

#[test]
fn other_loads_shrink_the_cap() {
    // A replica loaded before planned 100 GiB on the node: 28 GiB of the 128 GiB cap is left,
    // whatever the node reads free now.
    let n = Node {
        planned: 100 * GIB,
        ..node(1 << 20, 250, 256)
    };
    assert_eq!(node_budget(&n), 28 * GIB);
    let full = Node {
        planned: 200 * GIB,
        ..node(1 << 20, 250, 256)
    };
    assert_eq!(node_budget(&full), 0);
    // Not even a context then: a clear refusal, not a hang.
    let err = auto_host_blocks(1000, 4096, 4096, 1000, &[full]).unwrap_err();
    assert!(err.contains("200.0 GiB planned by other loads"), "{err}");
    assert!(err.contains("pinning at most 50% of a node"), "{err}");
}

fn nm(free_gib: u64, total_gib: u64) -> NodeMemory {
    NodeMemory {
        free: free_gib * GIB,
        cache: 0,
        total: total_gib * GIB,
    }
}

#[test]
fn a_group_sizes_from_its_first_look() {
    // The pair's second rank reads the node after the first pinned: it must size from the
    // same figures, or the ranks disagree (the 2026-11-03 log: 237.5 vs 232.4 GiB).
    let mut l = Ledger::new();
    assert_eq!(l.memory(1, 0, || Some(nm(237, 256))), Some(nm(237, 256)));
    assert_eq!(l.memory(1, 0, || Some(nm(232, 256))), Some(nm(237, 256)));
    // Another group (a later replica) reads anew.
    assert_eq!(l.memory(2, 0, || Some(nm(100, 256))), Some(nm(100, 256)));
    // An unreadable node is not remembered.
    assert_eq!(l.memory(3, 8, || None), None);
    assert_eq!(l.memory(3, 8, || Some(nm(9, 16))), Some(nm(9, 16)));
}

#[test]
fn the_ledger_sums_every_load_per_node() {
    let mut l = Ledger::new();
    let r0 = Owner { group: 1, rank: 0 };
    let r1 = Owner { group: 1, rank: 1 };
    let replica = Owner { group: 2, rank: 0 };
    l.memory(1, 0, || Some(nm(240, 256)));
    l.memory(2, 0, || Some(nm(120, 256)));
    l.memory(2, 8, || Some(nm(250, 256)));
    l.plan(r0, &[(0, 60 * GIB)]);
    l.plan(r1, &[(0, 60 * GIB)]);
    // The pair's own ranks are not "other loads" to each other: their split is in the block
    // bytes.
    assert_eq!(l.others(1, 0), 0);
    assert_eq!(l.others(2, 0), 120 * GIB);
    l.plan(replica, &[(0, 4 * GIB), (8, 50 * GIB)]);
    assert_eq!(l.others(1, 0), 4 * GIB);
    assert_eq!(l.others(1, 8), 50 * GIB);
    assert_eq!(
        l.per_node(),
        vec![
            NodePlan {
                node: 0,
                planned: 124 * GIB,
                cap: 128 * GIB
            },
            NodePlan {
                node: 8,
                planned: 50 * GIB,
                cap: 128 * GIB
            },
        ]
    );
    // A replan replaces the owner's plan.
    l.plan(replica, &[(8, 10 * GIB)]);
    assert_eq!(l.others(1, 0), 0);
    assert_eq!(l.others(1, 8), 10 * GIB);
}

#[test]
fn a_released_load_leaves_the_plan() {
    let mut l = Ledger::new();
    let r0 = Owner { group: 1, rank: 0 };
    let r1 = Owner { group: 1, rank: 1 };
    l.memory(1, 0, || Some(nm(240, 256)));
    l.plan(r0, &[(0, 60 * GIB)]);
    l.plan(r1, &[(0, 60 * GIB)]);
    l.release(r0);
    assert_eq!(l.others(2, 0), 60 * GIB);
    // The group's look stays while a rank of it plans.
    assert_eq!(l.memory(1, 0, || Some(nm(1, 256))), Some(nm(240, 256)));
    l.release(r1);
    assert_eq!(l.others(2, 0), 0);
    assert!(l.per_node().is_empty());
    assert_eq!(l.memory(1, 0, || Some(nm(1, 256))), Some(nm(1, 256)));
}

#[test]
fn the_plan_note() {
    let note = plan_note(&[
        NodePlan {
            node: 0,
            planned: gib(61.7),
            cap: 128 * GIB,
        },
        NodePlan {
            node: 8,
            planned: 0,
            cap: 128 * GIB,
        },
    ]);
    assert_eq!(
        note,
        "node 0: 61.7 GiB of a 128.0 GiB cap; node 8: 0.0 GiB of a 128.0 GiB cap"
    );
}
