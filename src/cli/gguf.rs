use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};

use clap::Subcommand;
use legion::gguf::{GgufFile, TensorInfo, Value};
use legion::residency::lookup_table::{TableIo, TableOptions};
use serde_json::json;

#[derive(Subcommand)]
pub enum GgufCommand {
    /// Print a GGUF file's header and metadata.
    Inspect {
        path: PathBuf,
        /// Also list every tensor: name, type, shape, size, offset.
        #[arg(long)]
        tensors: bool,
        /// Machine-readable output.
        #[arg(long)]
        json: bool,
        /// With `--json`, print long arrays in full instead of summarising them.
        #[arg(long)]
        full: bool,
        /// Print `tokenizer.chat_template` verbatim and nothing else.
        #[arg(long, conflicts_with_all = ["json", "tensors", "full"])]
        chat_template: bool,
    },
    /// Compare two GGUF models: metadata and tensors.
    Diff { a: PathBuf, b: PathBuf },
    /// Check a DeepSeek-V4.1-Flash GGUF (or safetensors checkpoint directory) without loading
    /// it: every tensor the graph reads (name, shape, type) and the Engram tables, its own
    /// file's included. CPU only, no mapping.
    Check {
        path: PathBuf,
        /// Tokens hashed through each Engram layer whose table rows are read and dequantized.
        #[arg(long, default_value_t = 64)]
        samples: usize,
        /// Also read the sampled rows through the Engram lookup table `serve` opens, in this
        /// io mode (`--ngram-io`'s), and check they are the file's bytes. `ram` takes the whole
        /// tables (~110 GB) into memory.
        #[arg(long, value_enum)]
        engram_io: Option<TableIo>,
    },
}

pub fn run(cmd: GgufCommand) -> anyhow::Result<()> {
    match cmd {
        GgufCommand::Inspect {
            path,
            tensors,
            json,
            full,
            chat_template,
        } => {
            if chat_template {
                print_chat_template(&path)
            } else {
                inspect(&path, tensors, json, full)
            }
        }
        GgufCommand::Diff { a, b } => diff(&a, &b),
        GgufCommand::Check {
            path,
            samples,
            engram_io,
        } => check(&path, samples, engram_io),
    }
}

fn check(path: &Path, samples: usize, engram_io: Option<TableIo>) -> anyhow::Result<()> {
    let g = GgufFile::open(path)?;
    let files: Vec<_> = g.shard_paths().collect();
    println!("{} files, {} tensors:", files.len(), g.tensor_count());
    for f in &files {
        println!("  {}", f.display());
    }
    let table = engram_io.map(|io| TableOptions {
        io,
        ..Default::default()
    });
    let r = legion::model::deepseek41::check::check_with(&g, samples, table.as_ref())?;
    let types: Vec<_> = r.types.iter().map(|(t, n)| format!("{t} {n}")).collect();
    println!("types: {}", types.join(", "));
    println!("{} tensors the graph reads checked", r.checked);
    let list = |what: &str, v: &[String]| {
        if !v.is_empty() {
            println!("{what} ({}):", v.len());
            for s in v {
                println!("  {s}");
            }
        }
    };
    list("missing", &r.missing);
    list("wrong shape", &r.shapes);
    list("experts the fused expert op does not take", &r.experts);
    list(
        "dense FP8 weights with block scales, which no matmul takes yet",
        &r.block_fp8,
    );
    list("Engram problems", &r.engram);
    list("Engram", &r.engram_notes);
    list(
        "dense weights the CUDA kernels do not take (CPU only)",
        &r.cuda_types,
    );
    list("tensors the graph does not read", &r.unread);
    if r.ok() {
        println!("ok");
        Ok(())
    } else {
        anyhow::bail!("{}: does not load", path.display())
    }
}

/// Arrays longer than this are summarised in JSON unless `--full` is given.
const SUMMARIZE_OVER: usize = 64;
/// How many leading elements a summary keeps.
const SUMMARY_HEAD: usize = 8;

/// Metadata keys the text listing leaves out because the tokenizer section summarises them.
const SUMMARIZED_KEYS: [&str; 5] = [
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.scores",
    "tokenizer.chat_template",
];

/// The special tokens shown with their ids and strings: (label, id key).
const SPECIAL_TOKENS: [(&str, &str); 4] = [
    ("bos", "tokenizer.ggml.bos_token_id"),
    ("eos", "tokenizer.ggml.eos_token_id"),
    ("pad", "tokenizer.ggml.padding_token_id"),
    ("unk", "tokenizer.ggml.unknown_token_id"),
];

/// Writes the chat template to stdout exactly as stored (no added newline).
fn print_chat_template(path: &Path) -> anyhow::Result<()> {
    let f = GgufFile::open(path)?;
    let template = f
        .get_str("tokenizer.chat_template")
        .ok_or_else(|| anyhow::anyhow!("{} has no tokenizer.chat_template", path.display()))?;
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(template.as_bytes())?;
    stdout.flush()?;
    Ok(())
}

/// The id -> string lookup for a special token, `None` if the id is unset or out of range.
fn token_string(tokens: Option<&Value>, id: u64) -> Option<String> {
    let items = tokens?.as_array()?;
    items
        .get(usize::try_from(id).ok()?)?
        .as_str()
        .map(String::from)
}

/// The tokenizer summary shared by the text and JSON output.
fn tokenizer_json(f: &GgufFile) -> serde_json::Value {
    let tokens = f.get("tokenizer.ggml.tokens");
    let len_of = |v: &Option<Value>| v.as_ref().and_then(|v| v.as_array()).map(<[Value]>::len);
    let mut special = serde_json::Map::new();
    for (label, key) in SPECIAL_TOKENS {
        if let Some(id) = f.get_u64(key) {
            special.insert(
                label.to_string(),
                json!({ "id": id, "token": token_string(tokens.as_ref(), id) }),
            );
        }
    }
    json!({
        "model": f.get_str("tokenizer.ggml.model"),
        "pre": f.get_str("tokenizer.ggml.pre"),
        "vocab_size": len_of(&tokens),
        "merges": len_of(&f.get("tokenizer.ggml.merges")),
        "special_tokens": special,
        "add_bos_token": f.get("tokenizer.ggml.add_bos_token").and_then(|v| v.as_bool()),
        "add_eos_token": f.get("tokenizer.ggml.add_eos_token").and_then(|v| v.as_bool()),
        "has_chat_template": f.get("tokenizer.chat_template").is_some(),
    })
}

fn print_tokenizer(f: &GgufFile) {
    let t = tokenizer_json(f);
    let show = |v: &serde_json::Value| match v {
        serde_json::Value::Null => "-".to_string(),
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    println!("tokenizer:");
    for (label, key) in [
        ("model", "model"),
        ("pre", "pre"),
        ("vocab size", "vocab_size"),
        ("merges", "merges"),
    ] {
        println!("  {label:<12} {}", show(&t[key]));
    }
    for (label, _) in SPECIAL_TOKENS {
        let s = &t["special_tokens"][label];
        if s.is_null() {
            println!("  {label:<12} -");
        } else {
            let token = s["token"]
                .as_str()
                .map_or("?".to_string(), |s| format!("{s:?}"));
            println!("  {label:<12} {} {token}", s["id"]);
        }
    }
    println!("  {:<12} {}", "add_bos", show(&t["add_bos_token"]));
    println!("  {:<12} {}", "add_eos", show(&t["add_eos_token"]));
    println!(
        "  {:<12} {}",
        "chat template",
        if t["has_chat_template"] == true {
            "yes (see --chat-template)"
        } else {
            "no"
        }
    );
}

fn inspect(path: &Path, list: bool, as_json: bool, full: bool) -> anyhow::Result<()> {
    let f = GgufFile::open(path)?;

    if as_json {
        let mut metadata = serde_json::Map::new();
        for (key, value) in f.metadata() {
            metadata.insert(key, meta_json(&value, full)?);
        }
        let mut out = json!({
            "file": path.display().to_string(),
            "version": f.version(),
            "architecture": f.architecture(),
            "alignment": f.alignment(),
            "data_offset": f.data_offset(),
            "shards": f.shard_paths().map(|p| p.display().to_string()).collect::<Vec<_>>(),
            "tokenizer": tokenizer_json(&f),
            "metadata": metadata,
        });
        if list {
            let tensors: Vec<_> = f.tensors().iter().map(tensor_json).collect();
            out["tensors"] = serde_json::to_value(tensors)?;
        }
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    let gib = f.tensor_bytes() as f64 / (1u64 << 30) as f64;
    println!("file:          {}", path.display());
    println!("version:       {}", f.version());
    println!(
        "architecture:  {}",
        f.architecture().as_deref().unwrap_or("?")
    );
    if f.shard_count() > 1 {
        println!("shards:        {}", f.shard_count());
    }
    println!("metadata keys: {}", f.kv_count());
    println!("tensors:       {} ({gib:.2} GiB)", f.tensor_count());
    println!(
        "data offset:   {} (alignment {}{})",
        f.data_offset(),
        f.alignment(),
        if f.shard_count() > 1 {
            ", first shard"
        } else {
            ""
        }
    );
    println!();
    print_tokenizer(&f);
    println!();
    for (key, value) in f.metadata() {
        if SUMMARIZED_KEYS.contains(&key.as_str()) {
            continue;
        }
        println!("{key:<48} {:<14} {value}", value.type_name());
    }
    if list {
        println!();
        print!(
            "{:<64} {:<8} {:<24} {:>10} {:>12}",
            "name", "type", "shape", "MiB", "offset"
        );
        let split = f.shard_count() > 1;
        println!("{}", if split { " shard" } else { "" });
        for t in f.tensors() {
            let mib = t.size as f64 / (1u64 << 20) as f64;
            print!(
                "{:<64} {:<8} {:<24} {:>10.4} {:>12}",
                t.name,
                t.type_name(),
                format!("{:?}", t.shape),
                mib,
                t.offset
            );
            if split {
                print!(" {:>5}", t.file_index + 1);
            }
            println!();
        }
    }
    Ok(())
}

/// Prints the metadata and tensor differences between two GGUF files, newest
/// formatting as `inspect`. Exits with status 1 when anything differs.
fn diff(a_path: &Path, b_path: &Path) -> anyhow::Result<()> {
    let a = GgufFile::open(a_path)?;
    let b = GgufFile::open(b_path)?;

    let am: BTreeMap<_, _> = a.metadata().into_iter().collect();
    let bm: BTreeMap<_, _> = b.metadata().into_iter().collect();
    println!("metadata:");
    let mut n_meta = 0;
    for key in am.keys().chain(bm.keys()).collect::<BTreeSet<_>>() {
        let (va, vb) = (am.get(key), bm.get(key));
        if va == vb {
            continue;
        }
        n_meta += 1;
        match (va, vb) {
            (Some(va), Some(vb)) => println!("~ {key}: {va} -> {vb}"),
            (Some(va), None) => println!("- {key} = {va}"),
            (None, Some(vb)) => println!("+ {key} = {vb}"),
            (None, None) => unreachable!("key comes from one of the maps"),
        }
    }

    let at: BTreeMap<_, _> = a
        .tensors()
        .into_iter()
        .map(|t| (t.name.clone(), t))
        .collect();
    let bt: BTreeMap<_, _> = b
        .tensors()
        .into_iter()
        .map(|t| (t.name.clone(), t))
        .collect();
    println!();
    println!("tensors:");
    let mut n_tensors = 0;
    for name in at.keys().chain(bt.keys()).collect::<BTreeSet<_>>() {
        match (at.get(name), bt.get(name)) {
            (Some(ta), Some(tb)) => {
                // Only type and shape are compared; offset, shard and size are layout detail.
                if ta.ggml_type != tb.ggml_type || ta.shape != tb.shape {
                    n_tensors += 1;
                    println!(
                        "~ {name}: {} {:?} -> {} {:?}",
                        ta.type_name(),
                        ta.shape,
                        tb.type_name(),
                        tb.shape
                    );
                }
            }
            (Some(ta), None) => {
                n_tensors += 1;
                println!("- {name} {} {:?}", ta.type_name(), ta.shape);
            }
            (None, Some(tb)) => {
                n_tensors += 1;
                println!("+ {name} {} {:?}", tb.type_name(), tb.shape);
            }
            (None, None) => unreachable!("name comes from one of the maps"),
        }
    }

    println!();
    println!("{n_meta} metadata and {n_tensors} tensor differences");
    if n_meta + n_tensors > 0 {
        std::process::exit(1);
    }
    Ok(())
}

/// A metadata value as JSON. Arrays longer than [`SUMMARIZE_OVER`] become
/// `{"type": ..., "len": ..., "head": [...]}` so a 150k-entry vocab stays readable.
fn meta_json(value: &Value, full: bool) -> anyhow::Result<serde_json::Value> {
    match value {
        Value::Array(items) if !full && items.len() > SUMMARIZE_OVER => Ok(json!({
            "type": value.type_name(),
            "len": items.len(),
            "head": items.iter().take(SUMMARY_HEAD).collect::<Vec<_>>(),
        })),
        other => Ok(serde_json::to_value(other)?),
    }
}

fn tensor_json(t: &TensorInfo) -> serde_json::Value {
    json!({
        "name": t.name,
        "type": t.type_name(),
        "shape": t.shape,
        "bytes": t.size,
        "offset": t.offset,
        "shard": t.file_index,
    })
}
