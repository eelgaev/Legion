use std::collections::HashMap;
use std::path::PathBuf;

use super::*;
use crate::ggml::Backend;
use crate::model::core::{Logits, SeqInput};
use crate::model::deepseek41::tiny;
use crate::residency::lookup_table::{TableIo, TableOptions};

/// A directory under the build directory: tmpfs refuses O_DIRECT (that falls back, still
/// exact, but would not test the direct reads).
fn build_dir() -> tempfile::TempDir {
    let base = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"));
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

fn tiny_gguf(dir: &tempfile::TempDir, tables: tiny::Tables) -> GgufFile {
    let path = dir.path().join("tiny.gguf");
    tiny::write_with(&path, 5, tables).unwrap();
    GgufFile::open(&path).unwrap()
}

fn backend() -> Backend {
    let mut b = Backend::cpu().unwrap();
    b.set_threads(2);
    b
}

/// Every io mode the table can take, with the direct tier's cache off, small and large, and
/// its small batches through the I/O threads or Linux AIO.
fn modes() -> Vec<TableOptions> {
    let opts = |io, ram_cache_bytes, aio_max| TableOptions {
        io,
        ram_cache_bytes,
        inflight: 4,
        aio_max,
        keep_awake_ms: 0,
        node: None,
        unified: false,
    };
    vec![
        opts(TableIo::Mmap, None, 0),
        opts(TableIo::Ram, None, 0),
        opts(TableIo::Direct, Some(0), 0),
        opts(TableIo::Direct, Some(4096), 0),
        opts(TableIo::Direct, Some(1 << 20), 0),
        opts(TableIo::Direct, Some(1 << 20), 64),
    ]
}

fn open(g: &GgufFile, opts: &TableOptions) -> LookupTable {
    let hp = Hparams::from_gguf(g).unwrap();
    let (parts, _) = lookup_table(g, &hp).unwrap();
    let t = LookupTable::open_parts(parts, opts, None).unwrap();
    assert_eq!(t.choice().io, opts.io);
    t
}

/// The model without its Engram tables in the weights, reading them through `table`.
fn without_tables(g: &GgufFile, b: &Backend, table: LookupTable) -> Model {
    let hp = Hparams::from_gguf(g).unwrap();
    let (_, skip) = lookup_table(g, &hp).unwrap();
    let opts = crate::load_io::Options {
        skip: &skip,
        ..crate::load_io::Options::with_threads(2)
    };
    let mut m = Model::load_with(g, |_| b.buffer_type(), &opts).unwrap();
    for name in &skip {
        assert!(m.w(name).is_err(), "{name} left out of the weights");
    }
    m.set_ngram_table(table);
    m
}

fn seq(tokens: &[u32]) -> SeqInput<'_> {
    SeqInput {
        tokens,
        n_past: 0,
        blocks: &[],
        ring: 0,
        logits: Logits::None,
        media: None,
    }
}

#[test]
fn the_tables_are_one_lookup_table_in_the_order_of_the_layers() {
    for tables in [tiny::Tables::Separate, tiny::Tables::Inline] {
        let dir = tempfile::tempdir().unwrap();
        let g = tiny_gguf(&dir, tables);
        let hp = Hparams::from_gguf(&g).unwrap();
        let (parts, names) = lookup_table(&g, &hp).unwrap();
        assert_eq!(
            names,
            ["blk.1.engram_embd.weight", "blk.3.engram_embd.weight"]
        );
        let file = match tables {
            tiny::Tables::Separate => dir.path().join(tiny::ENGRAM_FILE),
            _ => dir.path().join("tiny.gguf"),
        };
        for (li, (p, name)) in parts.iter().zip(&names).enumerate() {
            let t = g.tensor(name).unwrap();
            assert_eq!(p.path, file);
            assert_eq!(p.offset, t.offset);
            assert_eq!(p.n_rows, tiny::PRIMES[li].iter().sum::<i64>() as u64);
            assert_eq!(p.row_bytes, 4 * tiny::ENGRAM_DIM as usize);
        }
        let e = params(&hp).unwrap().engram.as_ref().unwrap();
        assert_eq!(bases(e), [0, parts[0].n_rows]);
    }
}

/// The rows of a batch of sequences (several groups, sequences with and without history)
/// from the lookup table in every io mode are the weights' rows, bit for bit.
#[test]
fn rows_from_the_table_are_the_weights_rows_in_every_io_mode() {
    let dir = build_dir();
    let g = tiny_gguf(&dir, tiny::Tables::Separate);
    let b = backend();
    let m = Model::load(&g, |_| b.buffer_type()).unwrap();
    let e = params(&m.hp).unwrap().engram.as_ref().unwrap();
    let toks: Vec<Vec<u32>> = vec![
        (0..9).map(|i| (i * 37 + 11) % 250).collect(),
        (0..9).map(|i| (i * 53 + 3) % 250).collect(),
        vec![7],
        vec![249],
        vec![0],
    ];
    let groups = |round: usize| {
        let n = if round == 0 { 9 } else { 3 };
        vec![
            Group {
                slot: 0,
                seqs: vec![seq(&toks[0][..n]), seq(&toks[1][..n])],
                verify: false,
            },
            Group {
                slot: 2,
                seqs: vec![seq(&toks[2]), seq(&toks[3]), seq(&toks[4])],
                verify: false,
            },
        ]
    };
    // Each slot's predecessors, as the state pool keeps them: [`NO_TOKEN`] before the start.
    let history = || {
        let mut prev: HashMap<usize, Vec<u32>> = HashMap::new();
        move |_: &Group<'_>, slot: usize, tokens: &[u32]| {
            let p = prev
                .entry(slot)
                .or_insert_with(|| vec![NO_TOKEN; e.ngram - 1]);
            let before = p.clone();
            p.extend_from_slice(tokens);
            p.drain(..p.len() - (e.ngram - 1));
            before
        }
    };
    // Two steps each: the second hashes with the first one's tokens.
    let mut h = history();
    let want: Vec<Vec<f32>> = (0..2)
        .map(|r| super::super::engram_rows(&m, e, &groups(r), &mut h))
        .collect();
    let n_tok = 2 * 9 + 3;
    assert_eq!(
        want[0].len(),
        n_tok * e.layers.len() * e.cols() * e.dim as usize
    );
    assert!(want[0].iter().any(|&v| v != 0.0));
    for opts in modes() {
        let table = open(&g, &opts);
        let mut h = history();
        for (r, want) in want.iter().enumerate() {
            let got = rows_from_table(&m, e, &table, &groups(r), &mut h);
            assert!(
                got.iter()
                    .map(|v| v.to_bits())
                    .eq(want.iter().map(|v| v.to_bits())),
                "{:?}, step {r}",
                opts.io
            );
        }
    }
}

/// The model loaded without its Engram tables and reading them through the lookup table, in
/// every io mode and with the tables inline or in their own GGUF, gives the logits of the
/// model reading them from its weights, bit for bit: a prompt, then token by token.
#[test]
fn logits_through_the_table_are_the_weights_logits_in_every_io_mode() {
    let mut b = backend();
    let prompt: Vec<u32> = (0..24).map(|i| (i * 37 + 11) % 250).collect();
    let run = |m: &Model, b: &mut Backend| {
        let mut st = m.new_state(b.buffer_type(), 64).unwrap();
        let mut alloc = None;
        let mut out = m
            .forward(b, &mut alloc, &mut st, &prompt[..16], true)
            .unwrap();
        for &t in &prompt[16..] {
            out.extend(m.forward(b, &mut alloc, &mut st, &[t], true).unwrap());
        }
        out
    };
    for tables in [tiny::Tables::Separate, tiny::Tables::Inline] {
        let dir = build_dir();
        let g = tiny_gguf(&dir, tables);
        let m = Model::load(&g, |_| b.buffer_type()).unwrap();
        let want = run(&m, &mut b);
        assert!(want.iter().all(|v| v.is_finite()));
        for opts in modes() {
            let m = without_tables(&g, &b, open(&g, &opts));
            let got = run(&m, &mut b);
            assert!(
                got.iter()
                    .map(|v| v.to_bits())
                    .eq(want.iter().map(|v| v.to_bits())),
                "{:?} ({tables:?})",
                opts.io
            );
        }
    }
}

/// A prompt chunk's Engram rows prefetched while the one before it computes are in the direct
/// tier's cache when its step reads them.
#[test]
fn a_prefetched_chunk_reads_its_rows_from_the_cache() {
    let dir = build_dir();
    let g = tiny_gguf(&dir, tiny::Tables::Separate);
    let mut b = backend();
    let opts = TableOptions {
        io: TableIo::Direct,
        ram_cache_bytes: Some(1 << 20),
        inflight: 4,
        aio_max: 0,
        keep_awake_ms: 0,
        node: None,
        unified: false,
    };
    let m = without_tables(&g, &b, open(&g, &opts));
    let prompt: Vec<u32> = (0..20).map(|i| (i * 37 + 11) % 250).collect();
    let mut st = m.new_state(b.buffer_type(), 64).unwrap();
    let mut alloc = None;
    m.forward(&mut b, &mut alloc, &mut st, &prompt[..10], true)
        .unwrap();
    let table = m.ngram.as_ref().unwrap();
    let before = table.stats();
    m.prefetch_engram(&prompt[..10], &prompt[10..]);
    let prefetched = table.stats();
    m.forward(&mut b, &mut alloc, &mut st, &prompt[10..], true)
        .unwrap();
    let after = table.stats();
    let e = params(&m.hp).unwrap().engram.as_ref().unwrap();
    let rows = (10 * e.layers.len() * e.cols()) as u64;
    assert_eq!(after.rows - prefetched.rows, rows, "{after:?}");
    assert_eq!(after.cache_hits - prefetched.cache_hits, rows, "{after:?}");
    assert!(prefetched.reads > before.reads, "{prefetched:?}");
    assert_eq!(after.reads, prefetched.reads, "{after:?}");
}

#[test]
fn other_models_have_no_engram_table() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("qwen.gguf");
    crate::model::qwen35::tiny::write(&path, &Default::default()).unwrap();
    let g = GgufFile::open(&path).unwrap();
    let hp = Hparams::from_gguf(&g).unwrap();
    assert!(lookup_table(&g, &hp).is_none());
}
