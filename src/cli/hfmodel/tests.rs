//! Tests of `--hfmodel`'s resolution: offline, against tiny Hugging Face caches in temp dirs.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use clap::Parser as _;

use super::super::Cli;
use super::*;

/// Creates a repo's cache directory (`models--<org>--<name>` with empty `refs/` and
/// `snapshots/`), returning it.
fn repo(root: &Path, spec: &str) -> PathBuf {
    let dir = root.join(format!("models--{}", spec.replace('/', "--")));
    std::fs::create_dir_all(dir.join("snapshots")).unwrap();
    std::fs::create_dir_all(dir.join("refs")).unwrap();
    dir
}

/// Creates a snapshot of a repo holding the named files, returning its directory.
fn snapshot(root: &Path, spec: &str, commit: &str, files: &[&str]) -> PathBuf {
    let dir = repo(root, spec).join("snapshots").join(commit);
    std::fs::create_dir_all(&dir).unwrap();
    for name in files {
        std::fs::write(dir.join(name), []).unwrap();
    }
    dir
}

/// Records a revision in `refs/`, as `huggingface-cli download` does.
fn ref_to(root: &Path, spec: &str, revision: &str, commit: &str) {
    std::fs::write(
        repo(root, spec).join("refs").join(revision),
        format!("{commit}\n"),
    )
    .unwrap();
}

fn root_for(vars: &[(&str, &str)]) -> PathBuf {
    let get = |name: &str| {
        vars.iter()
            .find(|(var, _)| *var == name)
            .map(|(_, value)| OsString::from(*value))
    };
    cache_root_from(
        &CacheEnv {
            hf_hub_cache: get("HF_HUB_CACHE"),
            huggingface_hub_cache: get("HUGGINGFACE_HUB_CACHE"),
            hf_home: get("HF_HOME"),
        },
        Path::new("/home/u"),
    )
}

#[test]
fn cache_root_precedence() {
    // No variable at all: the default under the home directory.
    assert_eq!(root_for(&[]), Path::new("/home/u/.cache/huggingface/hub"));
    assert_eq!(root_for(&[("HF_HOME", "/hf")]), Path::new("/hf/hub"));
    assert_eq!(
        root_for(&[("HUGGINGFACE_HUB_CACHE", "/hub")]),
        Path::new("/hub")
    );
    assert_eq!(root_for(&[("HF_HUB_CACHE", "/cache")]), Path::new("/cache"));
    // The higher-precedence variables win; an empty value does not count.
    assert_eq!(
        root_for(&[
            ("HF_HUB_CACHE", "/cache"),
            ("HUGGINGFACE_HUB_CACHE", "/hub")
        ]),
        Path::new("/cache")
    );
    assert_eq!(
        root_for(&[("HUGGINGFACE_HUB_CACHE", "/hub"), ("HF_HOME", "/hf")]),
        Path::new("/hub")
    );
    assert_eq!(
        root_for(&[("HF_HUB_CACHE", ""), ("HF_HOME", "/hf")]),
        Path::new("/hf/hub")
    );
}

#[test]
fn resolves_through_refs() {
    let cache = tempfile::tempdir().unwrap();
    let hash = "0123456789abcdef0123456789abcdef01234567";
    let snap = snapshot(
        cache.path(),
        "nvidia/X",
        hash,
        &["m.gguf", "mmproj-F16.gguf"],
    );
    ref_to(cache.path(), "nvidia/X", "main", hash);
    // The default revision, the same one named explicitly, and a second branch.
    assert_eq!(
        resolve_in(cache.path(), "nvidia/X").unwrap(),
        snap.join("m.gguf")
    );
    assert_eq!(
        resolve_in(cache.path(), "nvidia/X@main").unwrap(),
        snap.join("m.gguf")
    );
    ref_to(cache.path(), "nvidia/X", "v2", hash);
    assert_eq!(
        resolve_in(cache.path(), "nvidia/X@v2").unwrap(),
        snap.join("m.gguf")
    );
}

#[test]
fn follows_the_cache_symlinks() {
    // The cache's own layout: blobs/ holds the files, the snapshot links to them.
    let cache = tempfile::tempdir().unwrap();
    let r = repo(cache.path(), "org/gguf");
    std::fs::create_dir_all(r.join("blobs")).unwrap();
    std::fs::write(r.join("blobs").join("sha256-m"), []).unwrap();
    let snap = r.join("snapshots").join("hash1");
    std::fs::create_dir_all(&snap).unwrap();
    std::os::unix::fs::symlink(r.join("blobs").join("sha256-m"), snap.join("m.gguf")).unwrap();
    ref_to(cache.path(), "org/gguf", "main", "hash1");
    assert_eq!(
        resolve_in(cache.path(), "org/gguf").unwrap(),
        snap.join("m.gguf")
    );
}

#[test]
fn revision_by_hash_prefix() {
    let cache = tempfile::tempdir().unwrap();
    let hash = "fc694b5400000000000000000000000000000000";
    let snap = snapshot(cache.path(), "org/X", hash, &["model.gguf"]);
    // A full commit hash needs no refs entry; so does a prefix of one.
    assert_eq!(
        resolve_in(
            cache.path(),
            "org/X@fc694b5400000000000000000000000000000000"
        )
        .unwrap(),
        snap.join("model.gguf")
    );
    assert_eq!(
        resolve_in(cache.path(), "org/X@fc694b54").unwrap(),
        snap.join("model.gguf")
    );
    // Several snapshots sharing the prefix are ambiguous; a longer prefix is not.
    let other_hash = "fc694b54ffffffffffffffffffffffffffffffff";
    let other = snapshot(cache.path(), "org/X", other_hash, &["model.gguf"]);
    let err = resolve_in(cache.path(), "org/X@fc694b54")
        .unwrap_err()
        .to_string();
    assert!(err.contains("matches 2 snapshots"), "{err}");
    assert!(err.contains("full commit hash"), "{err}");
    assert_eq!(
        resolve_in(cache.path(), "org/X@fc694b54ffff").unwrap(),
        other.join("model.gguf")
    );
}

#[test]
fn missing_repo_error_names_the_cache_and_the_fetch_command() {
    let cache = tempfile::tempdir().unwrap();
    let err = resolve_in(cache.path(), "nvidia/Missing")
        .unwrap_err()
        .to_string();
    assert!(err.contains("not in the Hugging Face cache"), "{err}");
    assert!(err.contains("models--nvidia--Missing"), "{err}");
    assert!(
        err.contains("huggingface-cli download nvidia/Missing"),
        "{err}"
    );
    assert!(err.contains("hf download nvidia/Missing"), "{err}");
    // With an explicit revision, the fetch command names it.
    let err = resolve_in(cache.path(), "nvidia/Missing@dev")
        .unwrap_err()
        .to_string();
    assert!(err.contains("--revision dev"), "{err}");
}

#[test]
fn revision_not_found_lists_what_the_cache_has() {
    let cache = tempfile::tempdir().unwrap();
    snapshot(cache.path(), "org/X", "abc", &["m.gguf"]);
    ref_to(cache.path(), "org/X", "main", "abc");
    let err = resolve_in(cache.path(), "org/X@dev")
        .unwrap_err()
        .to_string();
    assert!(err.contains("revision dev is not in the cache"), "{err}");
    assert!(err.contains("revisions in the cache: main"), "{err}");
    // Without any refs, the message says so.
    let cache = tempfile::tempdir().unwrap();
    snapshot(cache.path(), "org/Y", "abc", &["m.gguf"]);
    let err = resolve_in(cache.path(), "org/Y").unwrap_err().to_string();
    assert!(err.contains("revisions in the cache: none"), "{err}");
}

#[test]
fn refs_to_a_missing_snapshot_is_an_error() {
    let cache = tempfile::tempdir().unwrap();
    ref_to(cache.path(), "org/X", "main", "deadbeef");
    let err = resolve_in(cache.path(), "org/X").unwrap_err().to_string();
    assert!(err.contains("deadbeef"), "{err}");
    assert!(err.contains("no snapshot"), "{err}");
}

#[test]
fn empty_snapshot_reports_an_incomplete_download() {
    let cache = tempfile::tempdir().unwrap();
    let dir = repo(cache.path(), "org/X");
    std::fs::create_dir_all(dir.join("snapshots").join("abc")).unwrap();
    ref_to(cache.path(), "org/X", "main", "abc");
    std::fs::create_dir_all(dir.join("blobs")).unwrap();
    std::fs::write(dir.join("blobs").join("sha.incomplete"), []).unwrap();
    let err = resolve_in(cache.path(), "org/X").unwrap_err().to_string();
    assert!(err.contains(".incomplete"), "{err}");
    assert!(err.contains("did not finish"), "{err}");
    // An empty snapshot without .incomplete blobs points at the layout instead.
    std::fs::remove_dir_all(dir.join("blobs")).unwrap();
    let err = resolve_in(cache.path(), "org/X").unwrap_err().to_string();
    assert!(err.contains("no model files"), "{err}");
    assert!(!err.contains(".incomplete"), "{err}");
}

#[test]
fn safetensors_snapshot_is_returned_as_a_directory() {
    let cache = tempfile::tempdir().unwrap();
    let snap = snapshot(
        cache.path(),
        "org/HF",
        "h1",
        &[
            "config.json",
            "model.safetensors.index.json",
            "m-00001-of-00002.safetensors",
        ],
    );
    ref_to(cache.path(), "org/HF", "main", "h1");
    assert_eq!(resolve_in(cache.path(), "org/HF").unwrap(), snap);
}

#[test]
fn split_shards_pick_the_first_shard() {
    let cache = tempfile::tempdir().unwrap();
    let snap = snapshot(
        cache.path(),
        "org/S",
        "h1",
        &["M-Q8_0-00001-of-00002.gguf", "M-Q8_0-00002-of-00002.gguf"],
    );
    ref_to(cache.path(), "org/S", "main", "h1");
    assert_eq!(
        resolve_in(cache.path(), "org/S").unwrap(),
        snap.join("M-Q8_0-00001-of-00002.gguf")
    );
}

#[test]
fn several_quantizations_ask_for_model() {
    let cache = tempfile::tempdir().unwrap();
    snapshot(
        cache.path(),
        "org/Q",
        "h1",
        &[
            "Qwen-Q8_0-00001-of-00002.gguf",
            "Qwen-Q8_0-00002-of-00002.gguf",
            "Qwen-IQ2_XS-00001-of-00002.gguf",
        ],
    );
    ref_to(cache.path(), "org/Q", "main", "h1");
    let err = resolve_in(cache.path(), "org/Q").unwrap_err().to_string();
    assert!(err.contains("several GGUF models"), "{err}");
    assert!(err.contains("Qwen-Q8_0-00001-of-00002.gguf"), "{err}");
    assert!(err.contains("Qwen-IQ2_XS-00001-of-00002.gguf"), "{err}");
    assert!(err.contains("--model"), "{err}");
}

#[test]
fn spec_must_be_org_slash_name() {
    let cache = tempfile::tempdir().unwrap();
    for spec in [
        "no-slash",
        "",
        "org/",
        "/name",
        "org/name@",
        "org/name@..",
        "a/../b",
    ] {
        let err = resolve_in(cache.path(), spec).unwrap_err().to_string();
        assert!(err.contains("--hfmodel expects"), "{spec}: {err}");
    }
}

#[test]
fn model_and_hfmodel_are_mutually_exclusive() {
    let err =
        match Cli::try_parse_from(["legion", "serve", "--model", "m.gguf", "--hfmodel", "org/n"]) {
            Ok(_) => panic!("--model and --hfmodel must not parse together"),
            Err(err) => err.to_string(),
        };
    assert!(err.contains("cannot be used with"), "{err}");
    // The alias parses on its own.
    Cli::try_parse_from(["legion", "serve", "--hf-model", "org/n@dev"]).unwrap();
}

#[test]
fn partial_safetensors_download_is_reported_before_loading() {
    let cache = tempfile::tempdir().unwrap();
    let snap = snapshot(
        cache.path(),
        "org/P",
        "h1",
        &["config.json", "m-00001-of-00002.safetensors"],
    );
    std::fs::write(
        snap.join("model.safetensors.index.json"),
        r#"{"weight_map": {"a": "m-00001-of-00002.safetensors",
                           "b": "m-00002-of-00002.safetensors"}}"#,
    )
    .unwrap();
    ref_to(cache.path(), "org/P", "main", "h1");
    let blobs = repo(cache.path(), "org/P").join("blobs");
    std::fs::create_dir_all(&blobs).unwrap();
    std::fs::write(blobs.join("sha.incomplete"), []).unwrap();
    let err = resolve_in(cache.path(), "org/P").unwrap_err().to_string();
    assert!(err.contains("did not finish"), "{err}");
    assert!(err.contains("m-00002-of-00002.safetensors"), "{err}");
    assert!(err.contains("1 .incomplete blobs"), "{err}");
    assert!(err.contains("huggingface-cli download org/P"), "{err}");
    // Once the shard is there, the checkpoint directory resolves.
    std::fs::write(snap.join("m-00002-of-00002.safetensors"), []).unwrap();
    assert_eq!(resolve_in(cache.path(), "org/P").unwrap(), snap);
}

#[test]
fn partial_split_gguf_download_is_reported_before_loading() {
    let cache = tempfile::tempdir().unwrap();
    let snap = snapshot(
        cache.path(),
        "org/G",
        "h1",
        &["M-Q4_K-00001-of-00003.gguf", "M-Q4_K-00002-of-00003.gguf"],
    );
    ref_to(cache.path(), "org/G", "main", "h1");
    let err = resolve_in(cache.path(), "org/G").unwrap_err().to_string();
    assert!(err.contains("did not finish"), "{err}");
    assert!(err.contains("M-Q4_K-00003-of-00003.gguf"), "{err}");
    std::fs::write(snap.join("M-Q4_K-00003-of-00003.gguf"), []).unwrap();
    assert_eq!(
        resolve_in(cache.path(), "org/G").unwrap(),
        snap.join("M-Q4_K-00001-of-00003.gguf")
    );
}

#[test]
fn only_later_shards_count_as_an_incomplete_download() {
    let cache = tempfile::tempdir().unwrap();
    snapshot(cache.path(), "org/L", "h1", &["M-00002-of-00002.gguf"]);
    ref_to(cache.path(), "org/L", "main", "h1");
    let blobs = repo(cache.path(), "org/L").join("blobs");
    std::fs::create_dir_all(&blobs).unwrap();
    std::fs::write(blobs.join("sha.incomplete"), []).unwrap();
    let err = resolve_in(cache.path(), "org/L").unwrap_err().to_string();
    assert!(err.contains("did not finish"), "{err}");
}

#[test]
fn gguf_in_subdirectories_are_listed() {
    // The per-quantization layout of many GGUF repos: one directory per quantization.
    let cache = tempfile::tempdir().unwrap();
    let snap = snapshot(cache.path(), "org/U", "h1", &["README.md"]);
    for (dir, file) in [
        ("Q4_K_M", "U-Q4_K_M-00001-of-00002.gguf"),
        ("Q4_K_M", "U-Q4_K_M-00002-of-00002.gguf"),
        ("Q8_0", "U-Q8_0-00001-of-00003.gguf"),
    ] {
        std::fs::create_dir_all(snap.join(dir)).unwrap();
        std::fs::write(snap.join(dir).join(file), []).unwrap();
    }
    ref_to(cache.path(), "org/U", "main", "h1");
    let err = resolve_in(cache.path(), "org/U").unwrap_err().to_string();
    assert!(err.contains("subdirectories"), "{err}");
    assert!(err.contains("--model <path>"), "{err}");
    assert!(err.contains("U-Q4_K_M-00001-of-00002.gguf"), "{err}");
    assert!(err.contains("U-Q8_0-00001-of-00003.gguf"), "{err}");
    assert!(!err.contains("00002-of-00002"), "{err}");
}

#[test]
fn repo_id_is_the_default_model_id() {
    assert_eq!(repo_id("nvidia/X@abc").as_deref(), Some("nvidia/X"));
    assert_eq!(repo_id("nvidia/X").as_deref(), Some("nvidia/X"));
    assert_eq!(repo_id("no-slash"), None);
}
