use std::path::{Path, PathBuf};

use crate::ggml::types;
use crate::gguf::{GgufError, GgufFile, GgufWriter, Value};

/// Writes a GGUF of architecture `arch` with F32 tensors `names` (8 values each) and the
/// string keys `keys`.
fn write(path: &Path, arch: &str, keys: &[(&str, &str)], names: &[&str]) {
    let mut g = GgufWriter::new();
    g.set("general.architecture", &Value::String(arch.into()))
        .unwrap();
    for (k, v) in keys {
        g.set(k, &Value::String((*v).into())).unwrap();
    }
    for (i, name) in names.iter().enumerate() {
        let data: Vec<u8> = (0..32).map(|b| (b + 3 * i) as u8).collect();
        g.add_tensor(name, types::F32, &[8], &data).unwrap();
    }
    g.write(path).unwrap();
}

const FILE: &str = "tiny-engram.gguf";
const TABLES: [&str; 2] = ["blk.1.engram_embd.weight", "blk.3.engram_embd.weight"];

/// A main file naming [`FILE`] (with `source`, if any) and the tensors `main`.
fn model(dir: &Path, source: Option<&str>, main: &[&str]) -> PathBuf {
    let path = dir.join("model.gguf");
    let mut keys = vec![("ds.engram.file", FILE)];
    keys.extend(source.map(|s| ("ds.engram.source", s)));
    write(&path, "ds", &keys, main);
    path
}

fn companion(dir: &Path, source: Option<&str>, names: &[&str]) {
    let keys: Vec<_> = source
        .map(|s| ("ds.engram.source", s))
        .into_iter()
        .collect();
    write(&dir.join(FILE), "ds-engram", &keys, names);
}

fn names(g: &GgufFile) -> Vec<String> {
    g.tensors().into_iter().map(|t| t.name).collect()
}

#[test]
fn attaches_the_named_file_after_the_model() {
    let dir = tempfile::tempdir().unwrap();
    let path = model(dir.path(), Some("abc"), &["token_embd.weight"]);
    companion(dir.path(), Some("abc"), &TABLES);
    let g = GgufFile::open(&path).unwrap();
    assert_eq!(
        names(&g),
        ["token_embd.weight", TABLES[0], TABLES[1]].map(String::from)
    );
    assert_eq!(g.shard_count(), 2);
    let paths: Vec<_> = g.shard_paths().map(Path::to_path_buf).collect();
    assert_eq!(paths, [path.clone(), dir.path().join(FILE)]);
    let t = g.tensor(TABLES[1]).unwrap();
    assert_eq!(t.file_index, 1);
    // Metadata still comes from the model.
    assert_eq!(g.architecture().as_deref(), Some("ds"));
    assert_eq!(g.tensor_count(), 3);
}

#[test]
fn a_missing_companion_opens_the_model_alone() {
    let dir = tempfile::tempdir().unwrap();
    let path = model(dir.path(), None, &["token_embd.weight"]);
    let g = GgufFile::open(&path).unwrap();
    assert_eq!(names(&g), ["token_embd.weight"]);
    assert_eq!(g.shard_count(), 1);
    let expected = crate::gguf::companion_path(&g, "engram.file").unwrap();
    assert_eq!(expected, Some(dir.path().join(FILE)));
}

#[test]
fn a_companion_already_in_the_model_is_left_out() {
    let dir = tempfile::tempdir().unwrap();
    let path = model(
        dir.path(),
        None,
        &["token_embd.weight", TABLES[0], TABLES[1]],
    );
    companion(dir.path(), None, &TABLES);
    let g = GgufFile::open(&path).unwrap();
    assert_eq!(g.shard_count(), 1);
    assert_eq!(g.tensor_count(), 3);
}

#[test]
fn a_partly_shared_companion_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = model(dir.path(), None, &["token_embd.weight", TABLES[0]]);
    companion(dir.path(), None, &TABLES);
    let err = GgufFile::open(&path).err().expect("an error");
    assert!(matches!(err, GgufError::Companion { .. }), "{err}");
    assert!(err.to_string().contains(TABLES[0]), "{err}");
}

#[test]
fn another_checkpoints_tables_are_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = model(dir.path(), Some("abc"), &["token_embd.weight"]);
    companion(dir.path(), Some("xyz"), &TABLES);
    let err = GgufFile::open(&path).err().expect("an error");
    let msg = err.to_string();
    assert!(msg.contains("xyz") && msg.contains("abc"), "{msg}");
}

#[test]
fn one_side_without_a_source_is_accepted() {
    let dir = tempfile::tempdir().unwrap();
    let path = model(dir.path(), Some("abc"), &["token_embd.weight"]);
    companion(dir.path(), None, &TABLES);
    assert_eq!(GgufFile::open(&path).unwrap().shard_count(), 2);
}

#[test]
fn only_plain_file_names_are_followed() {
    let dir = tempfile::tempdir().unwrap();
    for name in ["../x.gguf", "/x.gguf", "sub/x.gguf", ".."] {
        let path = dir.path().join("model.gguf");
        write(&path, "ds", &[("ds.engram.file", name)], &["a"]);
        let err = GgufFile::open(&path).err().expect("an error");
        assert!(err.to_string().contains("plain file name"), "{name}: {err}");
    }
}

#[test]
fn follows_a_split_model() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..2u16 {
        let mut g = GgufWriter::new();
        if i == 0 {
            g.set("general.architecture", &Value::String("ds".into()))
                .unwrap();
            g.set("ds.engram.file", &Value::String(FILE.into()))
                .unwrap();
        }
        g.set("split.no", &Value::U16(i)).unwrap();
        g.set("split.count", &Value::U16(2)).unwrap();
        g.set("split.tensors.count", &Value::I32(2)).unwrap();
        g.add_tensor(&format!("s{i}"), types::F32, &[8], &[0; 32])
            .unwrap();
        g.write(dir.path().join(format!("m-{:05}-of-00002.gguf", i + 1)))
            .unwrap();
    }
    companion(dir.path(), None, &TABLES);
    // From any shard: the companion comes after both.
    let g = GgufFile::open(dir.path().join("m-00002-of-00002.gguf")).unwrap();
    assert_eq!(g.shard_count(), 3);
    assert_eq!(g.tensor(TABLES[0]).unwrap().file_index, 2);
    assert_eq!(
        names(&g),
        ["s0", "s1", TABLES[0], TABLES[1]].map(String::from)
    );
}
