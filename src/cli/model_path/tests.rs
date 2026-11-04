//! Tests of `--model` directory resolution.

use super::*;

fn touch(dir: &Path, name: &str) {
    std::fs::write(dir.join(name), []).unwrap();
}

#[test]
fn file_path_passes_through() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("m.gguf");
    touch(dir.path(), "m.gguf");
    assert_eq!(resolve_model_path(&file).unwrap(), file);
    // a missing path is returned unchanged too; the loader reports it
    let missing = dir.path().join("nope.gguf");
    assert_eq!(resolve_model_path(&missing).unwrap(), missing);
}

#[test]
fn picks_the_model_and_skips_mmproj() {
    let dir = tempfile::tempdir().unwrap();
    touch(dir.path(), "m.gguf");
    touch(dir.path(), "mmproj-F16.gguf");
    assert_eq!(
        resolve_model_path(dir.path()).unwrap(),
        dir.path().join("m.gguf")
    );
}

#[test]
fn split_shards_resolve_to_the_first() {
    let dir = tempfile::tempdir().unwrap();
    for n in ["00001", "00002", "00003"] {
        touch(dir.path(), &format!("x-{n}-of-00003.gguf"));
    }
    assert_eq!(
        resolve_model_path(dir.path()).unwrap(),
        dir.path().join("x-00001-of-00003.gguf")
    );
}

#[test]
fn empty_dir_reports_no_model() {
    let dir = tempfile::tempdir().unwrap();
    let err = resolve_model_path(dir.path()).unwrap_err().to_string();
    assert!(err.contains("no .gguf model file"), "{err}");
}

#[test]
fn several_models_are_listed() {
    let dir = tempfile::tempdir().unwrap();
    touch(dir.path(), "b.gguf");
    touch(dir.path(), "a.gguf");
    let err = resolve_model_path(dir.path()).unwrap_err().to_string();
    assert!(err.contains("a.gguf, b.gguf"), "{err}");
}

#[test]
fn safetensors_checkpoint_dirs_are_kept() {
    let dir = tempfile::tempdir().unwrap();
    touch(dir.path(), "config.json");
    touch(dir.path(), "model.safetensors.index.json");
    touch(dir.path(), "mmproj-F16.gguf");
    assert_eq!(resolve_model_path(dir.path()).unwrap(), dir.path());
}

#[test]
fn non_model_files_are_ignored() {
    let dir = tempfile::tempdir().unwrap();
    touch(dir.path(), "README.GGUF");
    touch(dir.path(), "notes.txt");
    let err = resolve_model_path(dir.path()).unwrap_err().to_string();
    assert!(err.contains("no .gguf model file"), "{err}");
}

#[test]
fn symlinked_gguf_files_are_followed() {
    // A Hugging Face cache snapshot: blobs/ holds the files, the snapshot links to them.
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("blobs")).unwrap();
    std::fs::write(dir.path().join("blobs").join("sha256-m"), []).unwrap();
    std::fs::create_dir_all(dir.path().join("snapshots").join("abc")).unwrap();
    let snap = dir.path().join("snapshots").join("abc");
    std::os::unix::fs::symlink(
        dir.path().join("blobs").join("sha256-m"),
        snap.join("m.gguf"),
    )
    .unwrap();
    assert_eq!(resolve_model_path(&snap).unwrap(), snap.join("m.gguf"));
}
