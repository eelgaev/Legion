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
                ram: Default::default(),
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
            ram: Default::default(),
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
    for spec in [CacheSpec::plain(90 * 64), CacheSpec::budget(97 * 64, false)] {
        let mut c = RowCache::new(spec, 90);
        for r in 0..1000u32 {
            c.insert(r, &[r as u8; 90], false);
        }
        let held: Vec<u32> = (0..1000).filter(|&r| c.slot(r).is_some()).collect();
        assert!(held.len() <= 64, "{spec:?}");
        for r in held {
            assert_eq!(c.get(r, true).unwrap(), &[r as u8; 90][..]);
        }
    }
    assert!(
        RowCache::new(CacheSpec::plain(0), 90)
            .get(3, true)
            .is_none()
    );
    assert!(
        RowCache::new(CacheSpec::budget(0, false), 90)
            .get(3, true)
            .is_none()
    );
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
            ram: Default::default(),
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
                ram: Default::default(),
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
            ram: Default::default(),
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
                ram: Default::default(),
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

/// `--ngram-ram` sizes (#1209): binary units, no suffix GiB, decimals, a share of available
/// memory; 0 and garbage refused.
#[test]
fn parses_ram_budgets() {
    let gib = 1u64 << 30;
    let p = parse_ram_budget;
    assert_eq!(p("auto"), Ok(RamBudget::Auto));
    assert_eq!(p("ALL"), Ok(RamBudget::All));
    for (s, b) in [
        ("8G", 8 * gib),
        ("8g", 8 * gib),
        ("8GiB", 8 * gib),
        ("8", 8 * gib),
        ("1.5", 3 * gib / 2),
        ("512M", 512 << 20),
        ("64K", 64 << 10),
        ("64kib", 64 << 10),
        ("2T", 2 << 40),
        ("0.5t", 1 << 39),
        (" 4G ", 4 * gib),
    ] {
        assert_eq!(p(s), Ok(RamBudget::Bytes(b)), "{s}");
    }
    assert_eq!(p("30%"), Ok(RamBudget::Percent(30.0)));
    assert_eq!(p("12.5%"), Ok(RamBudget::Percent(12.5)));
    for bad in [
        "", "0", "0G", "0%", "-1G", "101%", "G", "8X", "8iB", "eight", "inf", "NaN", "1e400",
        "1e-12",
    ] {
        assert!(p(bad).is_err(), "{bad:?}");
    }
    assert_eq!(RamBudget::Percent(25.0).bytes(40 * gib), Some(10 * gib));
    assert_eq!(RamBudget::Bytes(5).bytes(0), Some(5));
    assert_eq!(RamBudget::Auto.bytes(gib), None);
    assert_eq!(RamBudget::All.bytes(gib), None);
}

/// The tier `--ngram-io` and `--ngram-ram` pick together, and the pairs refused.
#[test]
fn ngram_ram_picks_the_tier() {
    use super::{RamBudget as B, TableIo as T};
    let gib = 1u64 << 30;
    let mem = MemInfo {
        total: 128 * gib,
        available: 40 * gib,
    };
    let table = 27 * gib;
    let r = |io, ram| resolve_ram(io, ram, table, &mem);
    // auto: as without the flag.
    for io in [T::Auto, T::Ram, T::Mmap, T::Direct] {
        assert_eq!(r(io, B::Auto), Ok((io, None)));
    }
    // all: the ram tier.
    assert_eq!(r(T::Auto, B::All), Ok((T::Ram, None)));
    assert_eq!(r(T::Ram, B::All), Ok((T::Ram, None)));
    // A size below the table: direct with exactly that budget.
    assert_eq!(
        r(T::Auto, B::Bytes(8 * gib)),
        Ok((T::Direct, Some(8 * gib)))
    );
    assert_eq!(
        r(T::Direct, B::Bytes(2 * gib)),
        Ok((T::Direct, Some(2 * gib)))
    );
    assert_eq!(r(T::Auto, B::Percent(10.0)), Ok((T::Direct, Some(4 * gib))));
    // A size the table fits in: the ram tier, unless direct is asked for.
    assert_eq!(r(T::Auto, B::Bytes(table)), Ok((T::Ram, None)));
    assert_eq!(r(T::Auto, B::Bytes(32 * gib)), Ok((T::Ram, None)));
    assert_eq!(r(T::Ram, B::Bytes(32 * gib)), Ok((T::Ram, None)));
    assert_eq!(
        r(T::Direct, B::Bytes(32 * gib)),
        Ok((T::Direct, Some(32 * gib)))
    );
    // Refused: mmap with a size or all, direct with all, ram with less than the table, and a
    // share of no memory.
    for (io, ram) in [
        (T::Mmap, B::Bytes(2 * gib)),
        (T::Mmap, B::All),
        (T::Mmap, B::Percent(10.0)),
        (T::Direct, B::All),
        (T::Ram, B::Bytes(8 * gib)),
    ] {
        assert!(r(io, ram).is_err(), "{io:?} {ram:?}");
    }
    let none = MemInfo::default();
    assert!(resolve_ram(T::Auto, B::Percent(10.0), table, &none).is_err());
    // What does not need the table's size is caught at parse time.
    assert!(ram_conflict(T::Mmap, B::Bytes(1)).is_some());
    assert!(ram_conflict(T::Direct, B::All).is_some());
    assert!(ram_conflict(T::Ram, B::Bytes(1)).is_none());
    assert!(ram_conflict(T::Mmap, B::Auto).is_none());
}

/// A one-set RAM budget's cache of `rb`-byte rows.
fn one_set(rb: usize) -> RowCache {
    let c = RowCache::new(
        CacheSpec::budget(WAYS as u64 * (rb as u64 + HOT_WAY_BYTES) + 1, false),
        rb,
    );
    assert_eq!((c.sets, c.hot_max()), (1, HOT_WAYS as u64));
    c
}

/// A fetch's ask of `row` from `c`: true on a hit (its bytes checked), else read and cached.
fn ask(c: &mut RowCache, row: u32) -> bool {
    let bytes: Vec<u8> = (0..c.rb).map(|i| (row as usize * 31 + i) as u8).collect();
    match c.get(row, true) {
        Some(b) => {
            assert_eq!(b, &bytes[..], "row {row}");
            true
        }
        None => {
            c.insert(row, &bytes, false);
            false
        }
    }
}

/// The hot set: a row hit twice joins it while it has room; a row hit more takes the place of
/// the hot row hit least, which stays cached as a used one; rows read once replace by CLOCK
/// and never push out a hot row. A prefetched row's first ask is the one it was read for.
#[test]
fn hot_set_keeps_the_rows_hit_most() {
    let mut c = one_set(8);
    // Rows 1-4: read, then hit twice: hot. Row 4 a third time.
    for r in 1..=4 {
        for _ in 0..3 {
            ask(&mut c, r);
        }
    }
    ask(&mut c, 4);
    assert_eq!(c.pinned, 4);
    // Row 9 hit twice ties rows 1-3: no place. Hit three times, it takes row 1's.
    for _ in 0..3 {
        ask(&mut c, 9);
    }
    assert_eq!(c.pinned, 4);
    assert!(c.meta[c.slot(9).unwrap()] & PINNED == 0);
    ask(&mut c, 9);
    assert_eq!(c.pinned, 4);
    assert!(c.meta[c.slot(9).unwrap()] & PINNED != 0);
    assert!(c.meta[c.slot(1).unwrap()] & PINNED == 0);
    // A scan of rows read once.
    for r in 100..200 {
        ask(&mut c, r);
    }
    for r in [2, 3, 4, 9] {
        assert!(c.slot(r).is_some(), "hot row {r} evicted");
    }
    assert!(
        c.slot(1).is_none(),
        "row 1 lost its hot place and was read no more"
    );
    assert!(c.pinned <= c.hot_max());

    let mut c = one_set(8);
    let bytes = [7u8; 8];
    c.insert(7, &bytes, true);
    assert!(c.get(7, false).is_some(), "another prefetch: not an ask");
    assert!(c.get(7, true).is_some(), "the ask it was read for");
    assert!(c.get(7, true).is_some());
    assert_eq!(c.pinned, 0);
    assert!(c.get(7, true).is_some());
    assert_eq!(c.pinned, 1);
}

/// A synthetic trace: hashed row ids (no row stands for a token) asked a few times in a row
/// between scans of rows read once, four times the cache. With a RAM budget the hot set holds
/// them through every scan; round-robin over as many rows (the old policy) loses them.
#[test]
fn eviction_keeps_hot_rows_on_a_synthetic_trace() {
    let rb = 16usize;
    let sets = 16usize;
    let bytes = sets as u64 * (WAYS as u64 * (rb as u64 + HOT_WAY_BYTES) + 1);
    let run = |old: bool| {
        let mut c = RowCache::new(CacheSpec::budget(bytes, old), rb);
        assert_eq!(c.sets, sets);
        // Two hot rows per set, the hot ways' room.
        let mut per_set = vec![0; sets];
        let hot: Vec<u32> = (1..100_000u32)
            // Distinct: an odd multiply and a rotation are one to one.
            .map(|i| i.wrapping_mul(0x9E37_79B1).rotate_left(7))
            .filter(|&r| r < 1 << 31)
            .filter(|&r| {
                let s = c.set_of(r);
                per_set[s] < 2 && {
                    per_set[s] += 1;
                    true
                }
            })
            .take(2 * sets)
            .collect();
        assert_eq!(hot.len(), 2 * sets);
        let mut scan = 1u32 << 31;
        let (mut first, mut asked) = (0usize, 0usize);
        for round in 0..20 {
            for rep in 0..3 {
                for &r in &hot {
                    let hit = ask(&mut c, r);
                    if rep == 0 && round > 0 {
                        first += hit as usize;
                        asked += 1;
                    }
                }
            }
            for _ in 0..4 * sets * WAYS {
                ask(&mut c, scan);
                scan += 1;
            }
        }
        (first as f64 / asked as f64, c.pinned, c.hot_max())
    };
    let (hot, pinned, max) = run(false);
    assert_eq!(hot, 1.0, "hot rows evicted by a scan");
    assert_eq!((pinned, max), (2 * sets as u64, (HOT_WAYS * sets) as u64));
    let (old, pinned, max) = run(true);
    assert!(old < 0.1, "round-robin kept {old} of the hot rows");
    assert_eq!((pinned, max), (0, 0));
}

/// The same through the table, over an Engram-shaped one: several parts of one file (one per
/// head, with gaps), each row id a hash of an n-gram into its part. Every row comes back
/// exact; a budget keeps the hot rows through scans where round-robin over as many rows does
/// not, and the hot set is what the stats and the load choice say.
#[test]
fn hot_set_on_a_multi_part_table_with_hashed_rows() {
    let base = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"));
    std::fs::create_dir_all(&base).unwrap();
    let dir = tempfile::tempdir_in(&base).unwrap();
    let rb = 90usize;
    let heads = [2003u64, 3001, 4999];
    let mut offsets = Vec::new();
    let mut at = 1234u64;
    for n in heads {
        offsets.push(at);
        at += n * rb as u64 + 777;
    }
    let bytes: Vec<u8> = (0..at as usize)
        .map(|i| (i as u64).wrapping_mul(2_654_435_761).rotate_right(11) as u8)
        .collect();
    let path = dir.path().join("engram.bin");
    std::fs::write(&path, &bytes).unwrap();
    let parts: Vec<TableFile> = heads
        .iter()
        .zip(&offsets)
        .map(|(&n_rows, &offset)| TableFile {
            path: path.clone(),
            offset,
            n_rows,
            row_bytes: rb,
        })
        .collect();
    let firsts: Vec<u64> = heads
        .iter()
        .scan(0, |s, &n| {
            let f = *s;
            *s += n;
            Some(f)
        })
        .collect();
    let n: u64 = heads.iter().sum();
    let row = |r: u32| {
        let h = firsts.partition_point(|&f| f <= r as u64) - 1;
        let at = (offsets[h] + (r as u64 - firsts[h]) * rb as u64) as usize;
        bytes[at..at + rb].to_vec()
    };
    // 40 hot bigrams, a row in each head: the multiply-xor hash n-gram tables use.
    let mults = [23_703_573_157_769u64, 20_109_073_645_365];
    let mut hot: Vec<u32> = (0..40u64)
        .flat_map(|g| {
            let (a, b) = (g * 7 + 3, g * 13 + 5);
            let key = a.wrapping_mul(mults[0]) ^ b.wrapping_mul(mults[1]);
            (0..heads.len()).map(move |h| (h, key))
        })
        .map(|(h, key)| {
            let x = key.wrapping_mul(0x9E37_79B9_7F4A_7C15 + 2 * h as u64) >> 20;
            (firsts[h] + x % heads[h]) as u32
        })
        .collect();
    hot.sort_unstable();
    hot.dedup();
    let cold: Vec<u32> = (0..n as u32)
        .map(|i| (i as u64 * 7919 % n) as u32)
        .filter(|r| hot.binary_search(r).is_err())
        .collect();
    let sets = 128u64;
    let run = |budget: bool| {
        let opts = TableOptions {
            io: if budget {
                TableIo::Auto
            } else {
                TableIo::Direct
            },
            ram: if budget {
                RamBudget::Bytes(sets * (WAYS as u64 * (rb as u64 + HOT_WAY_BYTES) + 1))
            } else {
                RamBudget::Auto
            },
            // Round-robin over as many rows.
            ram_cache_bytes: (!budget).then_some(sets * WAYS as u64 * rb as u64),
            inflight: 4,
            aio_max: 0,
            keep_awake_ms: 0,
            node: None,
            unified: false,
        };
        let t = LookupTable::open_parts(parts.clone(), &opts, None).unwrap();
        assert_eq!(t.choice().io, TableIo::Direct);
        let want_hot = if budget { sets * HOT_WAYS as u64 } else { 0 };
        assert_eq!(t.choice().hot_rows, want_hot);
        assert_eq!(t.choice().hot_bytes, want_hot * rb as u64);
        let fetch = |rows: &[u32]| {
            let want: Vec<u8> = rows.iter().flat_map(|&r| row(r)).collect();
            assert!(t.fetch(rows).unwrap() == want, "rows differ");
        };
        let mut cold_at = 0;
        let (mut first, mut asked) = (0u64, 0u64);
        for round in 0..10 {
            for rep in 0..3 {
                let before = t.stats().cache_hits;
                fetch(&hot);
                if rep == 0 && round > 0 {
                    first += t.stats().cache_hits - before;
                    asked += hot.len() as u64;
                }
            }
            // A scan of four times the cache, in decode-step batches.
            let scan: Vec<u32> = (0..4 * sets as usize * WAYS)
                .map(|k| cold[(cold_at + k) % cold.len()])
                .collect();
            cold_at += scan.len();
            for chunk in scan.chunks(48) {
                fetch(chunk);
            }
        }
        (first as f64 / asked as f64, t.stats().hot_rows)
    };
    let (kept, pinned) = run(true);
    assert!(kept >= 0.9, "the budget kept {kept} of the hot rows");
    assert!(pinned >= (hot.len() as f64 * 0.9) as u64 && pinned <= hot.len() as u64);
    let (old, pinned) = run(false);
    assert!(old < 0.2, "round-robin kept {old} of the hot rows");
    assert_eq!(pinned, 0);
}
