use super::*;
use crate::model::deepseek41::tiny;

/// The tiny model with its tables in their own GGUF: every tensor the graph reads is there in
/// its shape, nothing else is, and the hashed rows come from the tables file.
#[test]
fn tiny_with_separate_tables_checks_clean() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write_with(&path, 5, tiny::Tables::Separate).unwrap();
    let g = GgufFile::open(&path).unwrap();
    let r = check(&g, 16).unwrap();
    assert!(r.ok(), "{r:?}");
    assert_eq!(r.checked, g.tensor_count());
    assert!(r.unread.is_empty(), "{:?}", r.unread);
    assert!(r.cuda_types.is_empty(), "{:?}", r.cuda_types);
    let notes = r.engram_notes.join("\n");
    let tables = dir.path().join(tiny::ENGRAM_FILE);
    assert!(notes.contains(&tables.display().to_string()), "{notes}");
    // 16 tokens, two layers, (ngram - 1) * heads rows each.
    let rows = 16 * 2 * ((tiny::NGRAM - 1) * tiny::ENGRAM_HEADS) as usize;
    assert!(
        notes.contains(&format!("{rows} rows of 16 tokens")),
        "{notes}"
    );
    assert_eq!(r.types.get("q8_0"), Some(&(3 * tiny::N_LAYER)));
}

#[test]
fn other_architectures_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.gguf");
    let mut w = crate::gguf::GgufWriter::new();
    w.set(
        "general.architecture",
        &crate::gguf::Value::String("llama".into()),
    )
    .unwrap();
    w.write(&path).unwrap();
    let err = check(&GgufFile::open(&path).unwrap(), 1).unwrap_err();
    assert!(err.to_string().contains("deepseek41"), "{err}");
}

/// Expert stacks as the fused op takes them: IQ2_XS among IQ2_XXS / IQ1_M (the GSQ-RCO mix)
/// passes, IQ4_NL is reported by layer.
#[test]
fn expert_types_are_checked_per_layer() {
    let info = |name: String, ty, shape: [u64; 3]| TensorInfo {
        name,
        ggml_type: ty,
        shape: shape.to_vec(),
        file_index: 0,
        offset: 0,
        size: 0,
    };
    let mut all = BTreeMap::new();
    for (il, gu, down) in [
        (0, types::IQ2_XS, types::IQ1_M),
        (1, types::IQ2_XXS, types::IQ2_XS),
        (2, types::IQ4_NL, types::IQ2_XXS),
    ] {
        for (s, ty, shape) in [
            ("gate", gu, [512, 256, 8]),
            ("up", gu, [512, 256, 8]),
            ("down", down, [256, 512, 8]),
        ] {
            let name = format!("blk.{il}.ffn_{s}_exps.weight");
            all.insert(name.clone(), info(name, ty, shape));
        }
    }
    let mut r = Report::default();
    check_experts(&all, 3, &mut r);
    assert_eq!(r.experts.len(), 1, "{:?}", r.experts);
    assert!(r.experts[0].starts_with("layer 2:"), "{:?}", r.experts);
}

/// With an io mode the sampled rows are read again through the Engram lookup table the engine
/// opens, and are the file's bytes.
#[test]
fn sampled_rows_read_through_the_lookup_table_are_the_files() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write_with(&path, 5, tiny::Tables::Separate).unwrap();
    let g = GgufFile::open(&path).unwrap();
    for io in [
        crate::residency::lookup_table::TableIo::Mmap,
        crate::residency::lookup_table::TableIo::Direct,
    ] {
        let opts = TableOptions {
            io,
            ..Default::default()
        };
        let r = check_with(&g, 16, Some(&opts)).unwrap();
        assert!(r.ok(), "{r:?}");
        let notes = r.engram_notes.join("\n");
        let rows = 16 * 2 * ((tiny::NGRAM - 1) * tiny::ENGRAM_HEADS) as usize;
        assert!(notes.contains("lookup table over 2 parts"), "{notes}");
        assert!(
            notes.contains(&format!(
                "{rows} rows read through the lookup table: the file's bytes"
            )),
            "{notes}"
        );
    }
}
