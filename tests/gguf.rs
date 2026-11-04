use legion::ggml::ffi;
use legion::gguf::{GgufFile, GgufWriter, Value};

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

#[test]
fn round_trip_metadata_and_tensors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");

    let mut w = GgufWriter::new();
    w.set("general.architecture", &Value::String("qwen3next".into()))
        .unwrap()
        .set("qwen3next.block_count", &Value::U32(2))
        .unwrap()
        .set("qwen3next.rope.freq_base", &Value::F32(1e6))
        .unwrap()
        .set("tokenizer.ggml.add_bos_token", &Value::Bool(false))
        .unwrap()
        .set(
            "tokenizer.ggml.tokens",
            &Value::Array(vec![Value::String("a".into()), Value::String("é".into())]),
        )
        .unwrap()
        .set(
            "tokenizer.ggml.token_type",
            &Value::Array(vec![Value::I32(1), Value::I32(3)]),
        )
        .unwrap();
    let a: Vec<f32> = (0..6).map(|i| i as f32).collect();
    w.add_tensor(
        "blk.0.attn_q.weight",
        ffi::ggml_type_GGML_TYPE_F32,
        &[3, 2],
        &f32_bytes(&a),
    )
    .unwrap();
    // q8_0: blocks of 32 values = 2-byte scale + 32 int8.
    w.add_tensor(
        "blk.0.ffn_up.weight",
        ffi::ggml_type_GGML_TYPE_Q8_0,
        &[64],
        &[0u8; 68],
    )
    .unwrap();
    w.write(&path).unwrap();

    let f = GgufFile::open(&path).unwrap();
    assert_eq!(f.version(), 3);
    assert_eq!(f.architecture().as_deref(), Some("qwen3next"));
    assert_eq!(f.get_u64("qwen3next.block_count"), Some(2));
    assert_eq!(f.get("qwen3next.rope.freq_base"), Some(Value::F32(1e6)));
    assert_eq!(
        f.get("tokenizer.ggml.add_bos_token"),
        Some(Value::Bool(false))
    );
    assert_eq!(
        f.get("tokenizer.ggml.tokens"),
        Some(Value::Array(vec![
            Value::String("a".into()),
            Value::String("é".into())
        ]))
    );
    assert_eq!(
        f.get("tokenizer.ggml.token_type").unwrap().type_name(),
        "array[i32]"
    );
    assert_eq!(f.get("missing"), None);
    assert_eq!(f.kv_count(), f.metadata().len());

    let ts = f.tensors();
    assert_eq!(ts.len(), 2);
    assert_eq!(ts[0].name, "blk.0.attn_q.weight");
    assert_eq!(ts[0].type_name(), "f32");
    assert_eq!(ts[0].shape, vec![3, 2]);
    assert_eq!(ts[0].size, 24);
    assert_eq!(ts[1].type_name(), "q8_0");
    assert_eq!(ts[1].size, 68);
    assert_eq!(ts[1].elements(), 64);
    assert_eq!(f.tensor_bytes(), 92);

    // The data offset is where ggml wrote it: read the f32 tensor back from the file.
    let raw = std::fs::read(&path).unwrap();
    let t = f.tensor("blk.0.attn_q.weight").unwrap();
    assert_eq!(t.offset % f.alignment() as u64, 0);
    let got = &raw[t.offset as usize..(t.offset + t.size) as usize];
    assert_eq!(got, f32_bytes(&a).as_slice());
}

#[test]
fn wrong_tensor_size_is_an_error() {
    let mut w = GgufWriter::new();
    assert!(
        w.add_tensor("t", ffi::ggml_type_GGML_TYPE_F32, &[4], &[0u8; 3])
            .is_err()
    );
}

#[test]
fn open_rejects_non_gguf() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("junk.gguf");
    std::fs::write(&path, b"not a gguf file at all").unwrap();
    assert!(GgufFile::open(&path).is_err());
    assert!(GgufFile::open(dir.path().join("missing.gguf")).is_err());
}
