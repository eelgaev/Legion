use std::path::{Path, PathBuf};

use legion::ggml::{Backend, types};
use legion::gguf::{GgufError, GgufFile, GgufWriter, Value};
use legion::load_io::Options;
use legion::weights::Weights;

const SHARDS: u64 = 3;

fn bytes_of(shard: usize, n: usize) -> Vec<u8> {
    (0..n * 4).map(|i| (i * 7 + shard * 31 + 5) as u8).collect()
}

/// Writes `model-0000i-of-00003.gguf`; shard `i` holds tensors `s{i}.a` (F32 x 16) and
/// `s{i}.b` (F32 x 8 * (i + 1)). Returns the shard paths and the expected bytes per tensor.
fn write_split(
    dir: &Path,
    declared_tensors: Option<u64>,
) -> (Vec<PathBuf>, Vec<(String, Vec<u8>)>) {
    let mut paths = Vec::new();
    let mut expected = Vec::new();
    for i in 0..SHARDS as usize {
        let path = dir.join(format!("model-{:05}-of-{:05}.gguf", i + 1, SHARDS));
        let mut g = GgufWriter::new();
        if i == 0 {
            g.set("general.architecture", &Value::String("test".into()))
                .unwrap();
        }
        g.set("split.no", &Value::U16(i as u16)).unwrap();
        g.set("split.count", &Value::U16(SHARDS as u16)).unwrap();
        g.set(
            "split.tensors.count",
            &Value::I32(declared_tensors.unwrap_or(2 * SHARDS) as i32),
        )
        .unwrap();
        for (name, n) in [(format!("s{i}.a"), 16), (format!("s{i}.b"), 8 * (i + 1))] {
            let data = bytes_of(i + n, n);
            g.add_tensor(&name, types::F32, &[n as i64], &data).unwrap();
            expected.push((name, data));
        }
        g.write(&path).unwrap();
        paths.push(path);
    }
    (paths, expected)
}

fn check_weights(gguf: &GgufFile, expected: &[(String, Vec<u8>)], opts: &Options<'_>) {
    let backend = Backend::cpu().unwrap();
    let weights = Weights::load_with(gguf, |_| backend.buffer_type(), opts).unwrap();
    assert_eq!(weights.len(), expected.len());
    for (name, data) in expected {
        assert_eq!(&weights.get(name).unwrap().bytes(), data, "{name}");
    }
}

#[test]
fn opens_from_any_shard_and_lists_all_tensors() {
    let dir = tempfile::tempdir().unwrap();
    let (paths, expected) = write_split(dir.path(), None);
    for path in [&paths[0], &paths[1], &paths[2]] {
        let f = GgufFile::open(path).unwrap();
        assert_eq!(f.shard_count(), 3);
        assert_eq!(f.path(), paths[0].as_path());
        assert_eq!(f.shard_paths().collect::<Vec<_>>(), paths);
        assert_eq!(f.architecture().as_deref(), Some("test"));
        assert_eq!(f.tensor_count(), expected.len());
        let infos = f.tensors();
        for (info, (name, data)) in infos.iter().zip(&expected) {
            assert_eq!(&info.name, name);
            assert_eq!(info.size as usize, data.len());
            assert_eq!(info.file_index, name[1..2].parse::<usize>().unwrap());
        }
        assert_eq!(f.tensor("s2.b").unwrap().file_index, 2);
        assert!(f.tensor("nope").is_none());
    }
}

#[test]
fn loads_every_tensor_across_shards_copy_and_mmap() {
    let dir = tempfile::tempdir().unwrap();
    let (paths, expected) = write_split(dir.path(), None);
    for path in [&paths[1], &paths[0]] {
        let f = GgufFile::open(path).unwrap();
        for mmap in [false, true] {
            let mut opts = Options::with_threads(2);
            opts.mmap = mmap;
            check_weights(&f, &expected, &opts);
        }
    }
}

#[test]
fn mmap_maps_tensors_from_every_shard() {
    let dir = tempfile::tempdir().unwrap();
    let (paths, expected) = write_split(dir.path(), None);
    let f = GgufFile::open(&paths[2]).unwrap();
    let backend = Backend::cpu().unwrap();
    let w = Weights::load_with(&f, |_| backend.buffer_type(), &Options::with_threads(2)).unwrap();
    let stats = w.stats();
    assert_eq!(
        stats.mapped + stats.mmap_fallback + stats.copied,
        expected.len()
    );
    assert!(stats.mapped > 0);
}

#[test]
fn single_file_is_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("one.gguf");
    let mut g = GgufWriter::new();
    g.add_tensor("t", types::F32, &[4], &[0u8; 16]).unwrap();
    g.write(&path).unwrap();
    let f = GgufFile::open(&path).unwrap();
    assert_eq!(f.shard_count(), 1);
    assert_eq!(f.tensors()[0].file_index, 0);
}

#[test]
fn missing_shard_names_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let (paths, _) = write_split(dir.path(), None);
    std::fs::remove_file(&paths[1]).unwrap();
    for path in [&paths[0], &paths[2]] {
        let err = GgufFile::open(path).err().unwrap();
        assert!(
            err.to_string().contains("model-00002-of-00003.gguf"),
            "{err}"
        );
    }
}

#[test]
fn tensor_count_mismatch_errors() {
    let dir = tempfile::tempdir().unwrap();
    let (paths, _) = write_split(dir.path(), Some(7));
    let err = GgufFile::open(&paths[1]).err().unwrap();
    assert!(matches!(err, GgufError::Split { .. }), "{err}");
    assert!(err.to_string().contains("split.tensors.count"), "{err}");
}

#[test]
fn duplicate_tensor_names_error() {
    let dir = tempfile::tempdir().unwrap();
    let (paths, _) = write_split(dir.path(), None);
    // Rewrite shard 3 with a tensor name that shard 1 already has.
    let mut g = GgufWriter::new();
    g.set("split.no", &Value::U16(2)).unwrap();
    g.set("split.count", &Value::U16(3)).unwrap();
    g.set("split.tensors.count", &Value::I32(6)).unwrap();
    g.add_tensor("s0.a", types::F32, &[16], &[0u8; 64]).unwrap();
    g.add_tensor("s2.b", types::F32, &[24], &[0u8; 96]).unwrap();
    g.write(&paths[2]).unwrap();
    let err = GgufFile::open(&paths[0]).err().unwrap();
    assert!(err.to_string().contains("duplicate tensor"), "{err}");
}

#[test]
fn wrong_split_no_errors() {
    let dir = tempfile::tempdir().unwrap();
    let (paths, _) = write_split(dir.path(), None);
    // Swap shards 2 and 3 on disk: the names say 2-of-3 but split.no says 2.
    let tmp = dir.path().join("swap");
    std::fs::rename(&paths[1], &tmp).unwrap();
    std::fs::rename(&paths[2], &paths[1]).unwrap();
    std::fs::rename(&tmp, &paths[2]).unwrap();
    let err = GgufFile::open(&paths[0]).err().unwrap();
    assert!(err.to_string().contains("split.no"), "{err}");
}

#[test]
fn badly_named_split_file_errors() {
    let dir = tempfile::tempdir().unwrap();
    let (paths, _) = write_split(dir.path(), None);
    let odd = dir.path().join("odd.gguf");
    std::fs::copy(&paths[0], &odd).unwrap();
    let err = GgufFile::open(&odd).err().unwrap();
    assert!(err.to_string().contains("-00001-of-00003.gguf"), "{err}");
}

#[test]
fn inspect_shows_the_whole_model() {
    let dir = tempfile::tempdir().unwrap();
    let (paths, _) = write_split(dir.path(), None);
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_legion"))
        .args(["gguf", "inspect", "--tensors"])
        .arg(&paths[1])
        .output()
        .unwrap();
    assert!(out.status.success(), "{out:?}");
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("shards:        3"), "{text}");
    assert!(text.contains("tensors:       6 "), "{text}");
    for name in ["s0.a", "s1.b", "s2.b"] {
        assert!(text.contains(name), "{text}");
    }
}
