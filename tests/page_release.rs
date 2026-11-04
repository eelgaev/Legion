//! Dropping a loaded model's file pages (#836): the bytes read afterwards are the same.

use legion::safetensors::{Checkpoint, SafetensorsFile};

/// A one-tensor `model.safetensors` image: `w`, U8, `bytes.len()` long.
fn image(bytes: &[u8]) -> Vec<u8> {
    let header = serde_json::json!({
        "w": {"dtype": "U8", "shape": [bytes.len()], "data_offsets": [0, bytes.len()]},
    });
    let header = serde_json::to_vec(&header).unwrap();
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header);
    out.extend_from_slice(bytes);
    out
}

fn pattern(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i * 7 + i / 4096) as u8).collect()
}

#[test]
fn released_pages_read_back_the_same_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let want = pattern(3 << 16);
    let path = dir.path().join("model.safetensors");
    std::fs::write(&path, image(&want)).unwrap();

    let file = SafetensorsFile::open(&path).unwrap();
    let before = file.data("w").unwrap();
    assert_eq!(before, &want[..]);
    file.release_pages();
    // The same slice, faulted back in from the file.
    assert_eq!(before, &want[..]);
    assert_eq!(file.data("w").unwrap(), &want[..]);

    let ckpt = Checkpoint::open(dir.path()).unwrap();
    assert_eq!(ckpt.data("w").unwrap(), &want[..]);
    ckpt.release_pages();
    ckpt.release_pages();
    assert_eq!(ckpt.data("w").unwrap(), &want[..]);
}

#[test]
fn a_gguf_drops_its_page_cache_and_still_loads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.gguf");
    let mut w = legion::gguf::GgufWriter::new();
    w.set(
        "general.architecture",
        &legion::gguf::Value::String("test".into()),
    )
    .unwrap();
    w.write(&path).unwrap();
    let g = legion::gguf::GgufFile::open(&path).unwrap();
    assert!(g.checkpoint().is_none());
    g.release_page_cache();
    assert_eq!(g.architecture(), Some("test".to_string()));
}
