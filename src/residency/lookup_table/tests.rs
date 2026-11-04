use super::*;

/// A table of `n_rows` rows of 90 bytes (IQ4_NL's size: rows straddle page boundaries) at an
/// unaligned offset of a file, with bytes no two rows share.
fn table(dir: &Path, n_rows: u64) -> (TableFile, Vec<u8>) {
    let rb = 90usize;
    let offset = 12_345u64;
    let mut bytes = vec![0xA5u8; offset as usize];
    let rows: Vec<u8> = (0..n_rows as usize * rb)
        .map(|i| ((i / rb) * 131 + (i % rb) * 7) as u8 ^ ((i / rb) >> 8) as u8)
        .collect();
    bytes.extend_from_slice(&rows);
    bytes.extend_from_slice(&[0x5Au8; 777]);
    let path = dir.join("table.bin");
    std::fs::write(&path, &bytes).unwrap();
    (
        TableFile {
            path,
            offset,
            n_rows,
            row_bytes: rb,
        },
        rows,
    )
}

fn want(rows: &[u8], picks: &[u32]) -> Vec<u8> {
    picks
        .iter()
        .flat_map(|&r| rows[r as usize * 90..][..90].to_vec())
        .collect()
}

/// Every tier returns the file's rows, bit for bit: repeats, rows that straddle pages, the
/// last row of the file, rows served from the direct tier's cache after a first read.
#[test]
fn every_tier_returns_the_rows_of_the_file() {
    // Under the build directory: tmpfs refuses O_DIRECT (that falls back, still exact).
    let base = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"));
    std::fs::create_dir_all(&base).unwrap();
    let dir = tempfile::tempdir_in(&base).unwrap();
    let n_rows = 20_000u64;
    let (file, rows) = table(dir.path(), n_rows);
    let picks: Vec<u32> = (0..3000u32)
        .map(|i| i.wrapping_mul(2_654_435_761) % n_rows as u32)
        .chain([0, 45, 45, n_rows as u32 - 1, 9999, 0])
        .collect();
    let expect = want(&rows, &picks);
    let tiers = [
        (TableIo::Mmap, None),
        (TableIo::Ram, None),
        (TableIo::Direct, Some(0)),
        (TableIo::Direct, Some(64 * 1024)),
        (TableIo::Direct, Some(16 << 20)),
    ];
    for (io, cache) in tiers {
        let t = LookupTable::open(
            file.clone(),
            &TableOptions {
                io,
                ram_cache_bytes: cache,
                inflight: 8,
                aio_max: 0,
                keep_awake_ms: 0,
                node: None,
                unified: false,
            },
            None,
        )
        .unwrap();
        assert_eq!(t.choice().io, io);
        for round in 0..2 {
            // Two batches in flight at once, collected in reverse order.
            let (a, b) = picks.split_at(1000);
            let (ta, tb) = (t.issue(a), t.issue(b));
            let mut got = t.collect(tb).unwrap();
            got.splice(0..0, t.collect(ta).unwrap());
            assert!(got == expect, "{io:?} cache {cache:?} round {round}");
        }
        let s = t.stats();
        if io == TableIo::Direct {
            assert_eq!(s.rows, 2 * picks.len() as u64);
            if cache == Some(16 << 20) {
                // The second round comes from the cache.
                assert!(s.cache_hits >= picks.len() as u64, "{s:?}");
            }
            if cache == Some(0) {
                assert_eq!(s.cache_hits, 0);
            }
        }
    }
}

/// Rows prefetched into the direct tier's cache are served from it, and exactly.
#[test]
fn prefetched_rows_come_from_the_cache() {
    let base = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"));
    std::fs::create_dir_all(&base).unwrap();
    let dir = tempfile::tempdir_in(&base).unwrap();
    let (file, rows) = table(dir.path(), 5000);
    let t = LookupTable::open(
        file,
        &TableOptions {
            io: TableIo::Direct,
            ram_cache_bytes: Some(4 << 20),
            inflight: 4,
            aio_max: 0,
            keep_awake_ms: 0,
            node: None,
            unified: false,
        },
        None,
    )
    .unwrap();
    let picks: Vec<u32> = (0..500u32).map(|i| i * 7 % 5000).collect();
    t.prefetch(&picks);
    let got = t.fetch(&picks).unwrap();
    assert!(got == want(&rows, &picks));
    let s = t.stats();
    // The prefetch read them (counted once as rows asked for), the fetch found all in the cache.
    assert_eq!(s.cache_hits, picks.len() as u64, "{s:?}");
}

#[test]
fn auto_takes_ram_only_when_the_table_fits_and_may_be_locked() {
    let gib = 1u64 << 30;
    let big = MemInfo {
        total: 574 * gib,
        available: 400 * gib,
    };
    let pc = MemInfo {
        total: 64 * gib,
        available: 40 * gib,
    };
    let table = 27 * gib;
    assert_eq!(auto_choice(table, &big, true).0, TableIo::Ram);
    assert_eq!(auto_choice(table, &big, false).0, TableIo::Direct);
    // 27 GiB + an 8 GiB margin fit in 40 GiB available; with 30 they do not.
    assert_eq!(auto_choice(table, &pc, true).0, TableIo::Ram);
    let gaming = MemInfo {
        total: 64 * gib,
        available: 30 * gib,
    };
    assert_eq!(auto_choice(table, &gaming, true).0, TableIo::Direct);
    // GB10: experts fill most of its 128 GB.
    let gb10 = MemInfo {
        total: 128 * gib,
        available: 20 * gib,
    };
    assert_eq!(auto_choice(table, &gb10, true).0, TableIo::Direct);
    assert!(default_cache(&gb10) <= 4 * gib && default_cache(&gb10) >= 64 << 20);
}

#[test]
fn reads_meminfo() {
    let m = MemInfo::parse("MemTotal:       1024 kB\nMemFree: 1 kB\nMemAvailable:     512 kB\n");
    assert_eq!(
        m,
        MemInfo {
            total: 1 << 20,
            available: 512 << 10
        }
    );
}

#[test]
fn row_cache_is_bounded_and_exact() {
    let mut c = RowCache::new(90 * 64, 90);
    for r in 0..1000u32 {
        c.insert(r, &[r as u8; 90]);
    }
    let held: Vec<u32> = (0..1000).filter(|&r| c.find(r).is_some()).collect();
    assert!(held.len() <= 64);
    for r in held {
        assert_eq!(c.find(r).unwrap(), &[r as u8; 90][..]);
    }
    assert!(RowCache::new(0, 90).find(3).is_none());
}

#[test]
fn auto_reads_direct_where_the_gpu_shares_the_memory() {
    // O_DIRECT: not on tmpfs, so under the target directory as above.
    let base = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"));
    std::fs::create_dir_all(&base).unwrap();
    let dir = tempfile::tempdir_in(&base).unwrap();
    let (file, _) = table(dir.path(), 64);
    let t = LookupTable::open(
        file,
        &TableOptions {
            io: TableIo::Auto,
            ram_cache_bytes: None,
            inflight: 2,
            aio_max: 0,
            keep_awake_ms: 0,
            node: None,
            unified: true,
        },
        None,
    )
    .unwrap();
    assert_eq!(t.choice().io, TableIo::Direct);
    assert!(
        t.choice().reason.contains("shares"),
        "{}",
        t.choice().reason
    );
    // The row cache stays small: the OS keeps the rest of the shared memory.
    assert!(t.choice().ram_cache_bytes <= UNIFIED_CACHE);
    assert!(t.choice().ram_cache_bytes > 0);
}

/// Small batches go out as AIO (where the file takes `O_DIRECT`); with several in flight at
/// once, collected out of order, past the AIO slots (the rest on the I/O threads, some batches
/// split between both) and beside big batches on the threads, every row is the file's.
#[test]
fn small_batches_through_aio_return_the_rows_of_the_file() {
    let base = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"));
    std::fs::create_dir_all(&base).unwrap();
    let dir = tempfile::tempdir_in(&base).unwrap();
    let n_rows = 40_000u64;
    let (file, rows) = table(dir.path(), n_rows);
    for cache in [Some(0), Some(16 << 20)] {
        let t = LookupTable::open(
            file.clone(),
            &TableOptions {
                io: TableIo::Direct,
                ram_cache_bytes: cache,
                inflight: 4,
                aio_max: DEFAULT_AIO_MAX,
                keep_awake_ms: 0,
                node: None,
                unified: false,
            },
            None,
        )
        .unwrap();
        assert!(matches!(t.tier, Tier::Direct(_)));
        let mut seed = 7u32;
        let mut next = |n: usize| -> Vec<u32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    seed % n_rows as u32
                })
                .collect()
        };
        for round in 0..3 {
            // 16-row batches (a token's), 100-row ones (past the slots when several are out),
            // a 3000-row one (the threads), and prefetches taken by the next issue.
            let batches: Vec<Vec<u32>> = (0..24)
                .map(|i| match i % 6 {
                    5 => next(100),
                    _ if i == 7 => next(3000),
                    _ => next(16),
                })
                .collect();
            t.prefetch(&next(16));
            let tickets: Vec<_> = batches.iter().map(|b| t.issue(b)).collect();
            let mut got: Vec<(usize, Vec<u8>)> = tickets
                .into_iter()
                .enumerate()
                .rev()
                .map(|(i, tk)| (i, t.collect(tk).unwrap()))
                .collect();
            got.reverse();
            for (i, bytes) in got {
                assert!(
                    bytes == want(&rows, &batches[i]),
                    "cache {cache:?} round {round} batch {i}"
                );
            }
        }
        let p = next(16);
        t.prefetch(&p);
        assert!(t.fetch(&p).unwrap() == want(&rows, &p), "cache {cache:?}");
    }
}

/// Batches collected on other threads than the one that issued them (AIO completions are
/// reaped by whichever thread collects).
#[test]
fn aio_batches_collect_on_any_thread() {
    let base = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"));
    std::fs::create_dir_all(&base).unwrap();
    let dir = tempfile::tempdir_in(&base).unwrap();
    let n_rows = 30_000u64;
    let (file, rows) = table(dir.path(), n_rows);
    let t = LookupTable::open(
        file,
        &TableOptions {
            io: TableIo::Direct,
            ram_cache_bytes: Some(0),
            inflight: 2,
            aio_max: DEFAULT_AIO_MAX,
            keep_awake_ms: 0,
            node: None,
            unified: false,
        },
        None,
    )
    .unwrap();
    std::thread::scope(|s| {
        for k in 0..4u32 {
            let (t, rows) = (&t, &rows);
            s.spawn(move || {
                for i in 0..50u32 {
                    let picks: Vec<u32> = (0..16)
                        .map(|j| (k * 7919 + i * 104_729 + j * 1_299_709) % n_rows as u32)
                        .collect();
                    assert!(t.fetch(&picks).unwrap() == want(rows, &picks));
                }
            });
        }
    });
}

/// A table in two ranges of one file with a gap between them (Engram's per-layer tables): rows
/// numbered on from the first range to the second, every tier returns the file's rows, bit for
/// bit, the rows on both sides of the seam included.
#[test]
fn a_table_in_parts_returns_the_rows_of_each_part() {
    let base = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"));
    std::fs::create_dir_all(&base).unwrap();
    let dir = tempfile::tempdir_in(&base).unwrap();
    let (rb, n0, n1) = (90usize, 3001u64, 4999u64);
    let (off0, gap) = (777u64, 4321u64);
    let off1 = off0 + n0 * rb as u64 + gap;
    let len = (off1 + n1 * rb as u64 + 55) as usize;
    let bytes: Vec<u8> = (0..len)
        .map(|i| (i as u64).wrapping_mul(2_654_435_761).rotate_right(13) as u8)
        .collect();
    let path = dir.path().join("parts.bin");
    std::fs::write(&path, &bytes).unwrap();
    let part = |offset, n_rows| TableFile {
        path: path.clone(),
        offset,
        n_rows,
        row_bytes: rb,
    };
    let row = |r: u64| {
        let at = if r < n0 {
            off0 + r * rb as u64
        } else {
            off1 + (r - n0) * rb as u64
        } as usize;
        bytes[at..at + rb].to_vec()
    };
    let n = n0 + n1;
    let picks: Vec<u32> = (0..2000u32)
        .map(|i| i.wrapping_mul(2_654_435_761) % n as u32)
        .chain([0, n0 as u32 - 1, n0 as u32, n0 as u32 + 1, n as u32 - 1, 0])
        .collect();
    let expect: Vec<u8> = picks.iter().flat_map(|&r| row(r as u64)).collect();
    for (io, cache, aio_max) in [
        (TableIo::Mmap, None, 0),
        (TableIo::Ram, None, 0),
        (TableIo::Direct, Some(0), 0),
        (TableIo::Direct, Some(16 << 20), 0),
        (TableIo::Direct, Some(16 << 20), DEFAULT_AIO_MAX),
    ] {
        let t = LookupTable::open_parts(
            vec![part(off0, n0), part(off1, n1)],
            &TableOptions {
                io,
                ram_cache_bytes: cache,
                inflight: 4,
                aio_max,
                keep_awake_ms: 0,
                node: None,
                unified: false,
            },
            None,
        )
        .unwrap();
        assert_eq!(t.n_rows(), n);
        for round in 0..2 {
            assert!(t.fetch(&picks).unwrap() == expect, "{io:?} round {round}");
            // Small batches (a decode step's), each its own read.
            for chunk in picks.chunks(16).take(8) {
                let want: Vec<u8> = chunk.iter().flat_map(|&r| row(r as u64)).collect();
                assert!(t.fetch(chunk).unwrap() == want, "{io:?} round {round}");
            }
        }
    }
}

#[test]
fn table_parts_must_share_the_file_and_row_size() {
    let part = |path: &str, row_bytes| TableFile {
        path: PathBuf::from(path),
        offset: 0,
        n_rows: 4,
        row_bytes,
    };
    let opts = TableOptions::default();
    let err = |parts| match LookupTable::open_parts(parts, &opts, None) {
        Err(e) => e.kind(),
        Ok(_) => panic!("opened"),
    };
    assert_eq!(err(vec![]), io::ErrorKind::InvalidInput);
    assert_eq!(
        err(vec![part("a", 90), part("b", 90)]),
        io::ErrorKind::InvalidInput
    );
    assert_eq!(
        err(vec![part("a", 90), part("a", 144)]),
        io::ErrorKind::InvalidInput
    );
}

/// The direct tier without `--ngram-inflight` keeps [`DEFAULT_INFLIGHT`] reads in flight: deep
/// enough for a prompt's first chunk, whose thousands of page reads wait in its first step
/// (#815), and the rows still exact.
#[test]
fn direct_defaults_to_a_deep_read_queue() {
    const { assert!(DEFAULT_INFLIGHT >= 256) };
    let base = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"));
    std::fs::create_dir_all(&base).unwrap();
    let dir = tempfile::tempdir_in(&base).unwrap();
    let (file, rows) = table(dir.path(), 5000);
    let t = LookupTable::open(
        file,
        &TableOptions {
            io: TableIo::Direct,
            ram_cache_bytes: Some(0),
            aio_max: 0,
            keep_awake_ms: 0,
            ..TableOptions::default()
        },
        None,
    )
    .unwrap();
    match &t.tier {
        Tier::Direct(d) => assert_eq!(d.workers.len(), DEFAULT_INFLIGHT),
        Tier::Mapped { .. } => panic!("direct reads fell back to a map"),
    }
    let picks: Vec<u32> = (0..2000u32).map(|i| i * 13 % 5000).collect();
    assert!(t.fetch(&picks).unwrap() == want(&rows, &picks));
}

/// `keep_awake_ms` (#835): while fetches come, the direct tier reads a page whenever no read went
/// out for that long; a second after the last fetch it stops, and the next fetch starts it again.
/// The rows stay the file's. 0 reads nothing.
#[test]
fn keep_awake_reads_only_while_fetches_come() {
    use std::time::{Duration, Instant};
    let base = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"));
    std::fs::create_dir_all(&base).unwrap();
    let dir = tempfile::tempdir_in(&base).unwrap();
    let (file, rows) = table(dir.path(), 5000);
    let open = |keep_awake_ms| {
        LookupTable::open(
            file.clone(),
            &TableOptions {
                io: TableIo::Direct,
                ram_cache_bytes: Some(0),
                keep_awake_ms,
                ..TableOptions::default()
            },
            None,
        )
        .unwrap()
    };
    let picks = [3u32, 45, 4999, 1234];
    let off = open(0);
    let on = open(5);
    assert_eq!(on.stats().keep_awake_reads, 0, "no fetch yet");
    let fetched = Instant::now();
    assert!(off.fetch(&picks).unwrap() == want(&rows, &picks));
    assert!(on.fetch(&picks).unwrap() == want(&rows, &picks));
    std::thread::sleep(Duration::from_millis(200));
    let early = on.stats().keep_awake_reads;
    assert!(early >= 3, "{early} keep-awake reads in 200 ms at 5 ms");
    // Past the window: stopped.
    std::thread::sleep(
        (fetched + keep_awake::WINDOW + Duration::from_millis(300)) - Instant::now(),
    );
    let stopped = on.stats().keep_awake_reads;
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(
        on.stats().keep_awake_reads,
        stopped,
        "reads after the window"
    );
    // A fetch starts it again.
    assert!(on.fetch(&picks).unwrap() == want(&rows, &picks));
    std::thread::sleep(Duration::from_millis(100));
    assert!(
        on.stats().keep_awake_reads > stopped,
        "no reads after a new fetch"
    );
    assert_eq!(off.stats().keep_awake_reads, 0);
}
