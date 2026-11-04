//! The tied LM head's device copy (a multi-GPU layout loads the token embeddings a second time
//! as `output.weight` on the last GPU): the copy holds the same bytes, the graph takes it for
//! the head, and the logits are bit-identical to the head read from the embeddings.

use legion::ggml::Backend;
use legion::gguf::GgufFile;
use legion::load_io::Options;
use legion::model::core::Model;
use legion::model::gemma4::tiny;
use legion::weights::alias_info;

const COPY: (&str, &str) = ("output.weight", "token_embd.weight");

fn cpu() -> Backend {
    let mut b = Backend::cpu().unwrap();
    b.set_threads(4);
    b.set_use_ref(true);
    b
}

#[test]
fn alias_takes_the_source_range_only_when_the_name_is_free() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 3).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let src = gguf.tensor(COPY.1).unwrap();
    let a = alias_info(&gguf, COPY.0, COPY.1).expect("a tied head");
    assert_eq!(a.name, COPY.0);
    assert_eq!(
        (a.ggml_type, &a.shape, a.file_index, a.offset, a.size),
        (
            src.ggml_type,
            &src.shape,
            src.file_index,
            src.offset,
            src.size
        )
    );
    // A name the file has already, or a missing source: nothing.
    assert!(alias_info(&gguf, COPY.1, "output_norm.weight").is_none());
    assert!(alias_info(&gguf, COPY.0, "no_such.weight").is_none());
}

#[test]
fn tied_head_copy_gives_identical_logits() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 4).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let mut backend = cpu();
    let plain = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    assert!(plain.weights().get(COPY.0).is_none());
    let aliases = [COPY];
    let mut opts = Options::configured();
    opts.aliases = &aliases;
    let copied = Model::load_with(&gguf, |_| backend.buffer_type(), &opts).unwrap();
    let head = copied.weights().get(COPY.0).expect("the copy");
    let embd = copied.weights().get(COPY.1).unwrap();
    assert_eq!(head.to_f32(), embd.to_f32());
    let prompt: Vec<u32> = "tied heads".bytes().map(u32::from).collect();
    assert_eq!(
        plain.prompt_logits_all(&mut backend, &prompt).unwrap(),
        copied.prompt_logits_all(&mut backend, &prompt).unwrap()
    );
}

/// No copy of a source whose matmuls take hints by its name: a `.scale` companion, or a place
/// in the FP8 or activation-precision lists. Under another name the copy would run without
/// them.
#[test]
fn no_alias_for_a_source_with_precision_hints() {
    use legion::ggml::types;
    use legion::gguf::{GgufWriter, Value};
    let dir = tempfile::tempdir().unwrap();
    let write = |name: &str, scale: bool, list: Option<&str>| {
        let path = dir.path().join(name);
        let mut w = GgufWriter::new();
        let f32s = |n: usize| -> Vec<u8> { (0..n).flat_map(|_| 0.5f32.to_ne_bytes()).collect() };
        w.add_tensor(COPY.1, types::F32, &[4, 8], &f32s(32))
            .unwrap();
        if scale {
            w.add_tensor("token_embd.scale", types::F32, &[1], &f32s(1))
                .unwrap();
        }
        if let Some(key) = list {
            w.set(key, &Value::Array(vec![Value::String(COPY.1.into())]))
                .unwrap();
        }
        w.write(&path).unwrap();
        GgufFile::open(&path).unwrap()
    };
    assert!(alias_info(&write("plain.gguf", false, None), COPY.0, COPY.1).is_some());
    assert!(alias_info(&write("scale.gguf", true, None), COPY.0, COPY.1).is_none());
    for key in ["general.tensor_extra.name", legion::hf::nvfp4::KEY_FP8_E4M3] {
        let g = write(&format!("{}.gguf", key.len()), false, Some(key));
        assert!(alias_info(&g, COPY.0, COPY.1).is_none(), "{key}");
    }
}
