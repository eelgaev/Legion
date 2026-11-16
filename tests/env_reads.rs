//! Feature switches are `legion serve` flags, not environment variables (#421): the code reads a
//! `LEGION_*` variable only where this file allows it (test and debug hooks, listed in
//! docs/debug-env.md), and passes a name it did not spell out only in
//! the files listed for that. A new switch goes into `src/cli/serve.rs` (or
//! `src/cli/serve/switches.rs`) as a clap argument instead.
//!
//! `LEGION_DEBUG_*` variables are a class of their own: A/B switches that put back the code
//! before a change, for measurements. Any file may read one by its literal name, as long as
//! docs/debug-env.md has a table row for that exact name; a row whose variable nothing reads
//! any more fails too.

use std::path::{Path, PathBuf};

/// The prefix of the A/B switches documented row by row in docs/debug-env.md.
const DEBUG_PREFIX: &str = "LEGION_DEBUG_";

/// `(file, variable)`: the other `LEGION_*` reads that may stay.
const ALLOWED: &[(&str, &str)] = &[
    // Debug and development hooks (docs/debug-env.md).
    ("src/cpu/bench.rs", "LEGION_CPU_BENCH"),
    ("src/ggml/meta.rs", "LEGION_META_DEBUG"),
    ("src/ggml/op_profile.rs", "LEGION_OP_PROFILE"),
    ("src/hf/deepseek41/hash/tests.rs", "LEGION_TEST_DS41_HF"),
    ("src/model/moe_ffn/tests.rs", "LEGION_MOE_SPEED_TYPES"),
    ("src/model/qwen4/qsa_attn/tests.rs", "LEGION_TINY"),
    ("src/residency/experts.rs", "LEGION_EXPERT_COUNTS"),
    ("src/residency/ssd_experts.rs", "LEGION_SSD_EXPERT_TRACE"),
];

/// Files that may read a variable whose name is not a literal at the call.
const INDIRECT: &[&str] = &[
    // GGML_CUDA_ENABLE_UNIFIED_MEMORY, through a constant.
    "src/engine/lm.rs",
    // Opt-in GPU measurements' LEGION_TEST_* inputs.
    "src/engine/lm/tests/hc_inject_kl.rs",
    "src/engine/lm/tests/hc_norm_fma_kl.rs",
    "src/engine/lm/tests/kv_quant_kl.rs",
    "src/engine/lm/tests/pair_out_kl.rs",
];

/// The calls that read the environment, in Rust and in the CUDA/C sources.
const READS: &[&str] = &["env::var(", "env::var_os(", "getenv("];

fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            sources(&path, out);
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| matches!(e, "rs" | "cu" | "cuh" | "c" | "h"))
        {
            out.push(path);
        }
    }
}

/// Every environment read of `text`: the variable's name when the call spells it out as a
/// literal, `None` otherwise. Comment lines are skipped.
fn reads(text: &str) -> Vec<Option<String>> {
    let mut found = Vec::new();
    for line in text.lines() {
        let code = line.trim_start();
        if code.starts_with("//") {
            continue;
        }
        for call in READS {
            let mut rest = line;
            while let Some(at) = rest.find(call) {
                // `std::env::set_var(` and the like are not reads; `getenv(` must stand alone.
                let before = rest[..at].chars().last();
                rest = &rest[at + call.len()..];
                if *call == "getenv(" && before.is_some_and(|c| c.is_alphanumeric() || c == '_') {
                    continue;
                }
                let arg = rest.trim_start();
                found.push(
                    arg.strip_prefix('"')
                        .and_then(|a| a.split('"').next())
                        .map(str::to_string),
                );
            }
        }
    }
    found
}

/// The `LEGION_DEBUG_*` names of `doc`'s table rows: those spelled in a row's first cell.
fn debug_rows(doc: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in doc.lines() {
        let Some(cell) = line
            .trim_start()
            .strip_prefix('|')
            .and_then(|l| l.split('|').next())
        else {
            continue;
        };
        let mut rest = cell;
        while let Some(at) = rest.find(DEBUG_PREFIX) {
            rest = &rest[at..];
            let end = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(rest.len());
            names.push(rest[..end].to_string());
            rest = &rest[end..];
        }
    }
    names
}

#[test]
fn legion_variables_are_read_only_where_allowed() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let doc = std::fs::read_to_string(root.join("docs/debug-env.md")).unwrap();
    let rows = debug_rows(&doc);
    let mut files = Vec::new();
    sources(&root.join("src"), &mut files);
    files.sort();
    let mut bad = Vec::new();
    let mut seen = Vec::new();
    let mut seen_debug = Vec::new();
    for path in &files {
        let rel = path
            .strip_prefix(root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let text = std::fs::read_to_string(path).unwrap();
        for read in reads(&text) {
            match read {
                Some(name) if name.starts_with(DEBUG_PREFIX) => {
                    if rows.contains(&name) {
                        seen_debug.push(name);
                    } else {
                        bad.push(format!(
                            "{rel}: reads {name}, which has no row in docs/debug-env.md"
                        ));
                    }
                }
                Some(name) if name.starts_with("LEGION_") => {
                    if ALLOWED.contains(&(rel.as_str(), name.as_str())) {
                        seen.push((rel.clone(), name));
                    } else {
                        bad.push(format!("{rel}: reads {name}"));
                    }
                }
                Some(_) => {}
                None if INDIRECT.contains(&rel.as_str()) => {}
                None => bad.push(format!("{rel}: reads a variable named elsewhere")),
            }
        }
    }
    assert!(
        bad.is_empty(),
        "environment reads outside the allow-list (make a feature switch a `legion serve` \
         flag; a LEGION_DEBUG_* A/B switch needs a row in docs/debug-env.md; another test or \
         debug hook goes into tests/env_reads.rs and docs/debug-env.md):\n{}",
        bad.join("\n")
    );
    // The list follows the code: an entry nothing reads any more goes.
    for (file, name) in ALLOWED {
        assert!(
            seen.iter().any(|(f, n)| f == file && n == name),
            "{file} no longer reads {name}: drop it from ALLOWED"
        );
    }
    // So do the doc's rows.
    for name in &rows {
        assert!(
            seen_debug.contains(name),
            "nothing reads {name} any more: drop its row from docs/debug-env.md"
        );
    }
}

#[test]
fn debug_hooks_are_documented() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let doc = std::fs::read_to_string(root.join("docs/debug-env.md")).unwrap();
    for (_, name) in ALLOWED {
        assert!(doc.contains(name), "docs/debug-env.md does not list {name}");
    }
}

#[test]
fn the_doc_rows_parse() {
    let doc = "\
Text naming `LEGION_DEBUG_PROSE=1` outside a table.

| Variable | Effect |
|---|---|
| `LEGION_DEBUG_A=1` | Effect, see `LEGION_DEBUG_IN_EFFECT`. |
| `LEGION_DEBUG_B=<n>[,<m>]` | Effect. |
| `LEGION_DEBUG_C` | Effect. |
| `LEGION_OP_PROFILE=1` | Not a debug switch. |
";
    assert_eq!(
        debug_rows(doc),
        ["LEGION_DEBUG_A", "LEGION_DEBUG_B", "LEGION_DEBUG_C"]
    );
}

#[test]
fn the_scan_finds_reads() {
    let text = r#"
        let a = std::env::var("LEGION_A");
        let b = std::env::var_os( "LEGION_B" );
        // std::env::var("LEGION_COMMENT")
        let c = std::env::var(NAME);
        unsafe { std::env::set_var("LEGION_SET", "1") };
        const char * d = getenv("LEGION_D");
        my_getenv("LEGION_NOT");
    "#;
    assert_eq!(
        reads(text),
        [
            Some("LEGION_A".to_string()),
            Some("LEGION_B".to_string()),
            None,
            Some("LEGION_D".to_string()),
        ]
    );
}
