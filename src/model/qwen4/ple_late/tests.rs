//! Late rows read from the fill on give the bytes of reads issued at the write.

use std::path::PathBuf;

use super::*;
use crate::residency::lookup_table::{TableFile, TableIo, TableOptions};

/// A direct-read table of `n_rows` rows of 90 bytes (rows straddle pages) and its rows' bytes.
fn direct_table(dir: &std::path::Path, n_rows: u64) -> (LookupTable, Vec<u8>) {
    let rb = 90usize;
    let rows: Vec<u8> = (0..n_rows as usize * rb)
        .map(|i| ((i / rb) * 131 + (i % rb) * 7) as u8 ^ ((i / rb) >> 8) as u8)
        .collect();
    let path = dir.join("ngram.bin");
    std::fs::write(&path, &rows).unwrap();
    let file = TableFile {
        path,
        offset: 0,
        n_rows,
        row_bytes: rb,
    };
    let opts = TableOptions {
        io: TableIo::Direct,
        ram: Default::default(),
        ram_cache_bytes: Some(0),
        inflight: 4,
        aio_max: 0,
        keep_awake_ms: 0,
        node: None,
        unified: false,
    };
    (LookupTable::open(file, &opts, None).unwrap(), rows)
}

#[test]
fn early_reads_give_the_rows_of_late_ones() {
    // Under the build directory: tmpfs refuses O_DIRECT (that falls back, still exact).
    let base = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"));
    std::fs::create_dir_all(&base).unwrap();
    let dir = tempfile::tempdir_in(&base).unwrap();
    let (table, bytes) = direct_table(dir.path(), 3000);
    let rows: Vec<i32> = [7, 2999, 45, 45, 0, 1234].to_vec();
    let want: Vec<u8> = rows
        .iter()
        .flat_map(|&r| bytes[r as usize * 90..][..90].to_vec())
        .collect();
    for early in [true, false] {
        set_ple_early_reads(early);
        let late = LateRows::new(rows.clone(), &table);
        set_ple_early_reads(true);
        assert_eq!(late.issued(), early);
        assert_eq!(late.rows, rows);
        assert!(late.bytes(&table).unwrap() == want, "early {early}");
    }
    // Issued reads left uncollected (a failed step's) do not disturb later ones.
    drop(LateRows::new(rows.clone(), &table));
    assert!(LateRows::new(rows, &table).bytes(&table).unwrap() == want);
}

#[test]
fn row_ids_are_the_rows() {
    assert_eq!(ids(&[0, 5, 1 << 20]), vec![0, 5, 1 << 20]);
}
