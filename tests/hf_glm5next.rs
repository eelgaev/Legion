//! A GLM-5.3-Flash checkpoint (`glm5_next`: ModelOpt NVFP4 routed experts and leading dense
//! MLPs, everything else BF16) opened as a synthetic `glm5-next` GGUF (`src/hf/glm5next.rs`)
//! against a GGUF of the same model (ignored; run it on a machine that has both, CPU only):
//!
//! ```text
//! LEGION_TEST_GLM5_HF=<snapshot dir> LEGION_TEST_GLM5_GGUF=<first .gguf shard> \
//!   cargo test --release --test hf_glm5next -- --ignored --nocapture
//! ```
//!
//! The GGUF may hold more layers than the checkpoint, and its MTP block: only the
//! checkpoint's layers are compared, and the MTP keys and tensors are left out.

use std::io::{Read, Seek, SeekFrom};

use legion::ggml::{self, types};
use legion::gguf::{GgufFile, Recipe, TensorInfo, Value};

fn read_tensor(g: &GgufFile, t: &TensorInfo, limit: Option<u64>) -> Vec<u8> {
    let len = limit.map_or(t.size, |l| l.min(t.size));
    let read = |file: usize, offset: u64, len: u64| {
        let path = g.shard_paths().nth(file).unwrap().to_path_buf();
        let mut f = std::fs::File::open(path).unwrap();
        f.seek(SeekFrom::Start(offset)).unwrap();
        let mut v = vec![0u8; len as usize];
        f.read_exact(&mut v).unwrap();
        v
    };
    let mut v = match g.recipe(&t.name) {
        Some(Recipe::Computed(f)) => f().unwrap(),
        Some(Recipe::Gather(parts)) => parts
            .iter()
            .flat_map(|p| read(p.file, p.offset, p.len))
            .collect(),
        None => read(t.file_index, t.offset, len),
    };
    v.truncate(len as usize);
    v
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect()
}

/// `t` as f32 (its first `rows` rows when given), times its per-tensor or per-expert `.scale`.
fn values(g: &GgufFile, t: &TensorInfo, rows: Option<u64>) -> Vec<f32> {
    let row = ggml::row_size(t.ggml_type, t.shape[0] as usize) as u64;
    let bytes = read_tensor(g, t, rows.map(|r| r * row));
    let n = bytes.len() / row as usize * t.shape[0] as usize;
    let mut v = if t.ggml_type == types::F32 {
        f32s(&bytes)
    } else {
        ggml::dequantize(t.ggml_type, &bytes, n)
    };
    let scale = t.name.strip_suffix(".weight").map(|b| format!("{b}.scale"));
    if let Some(s) = scale.and_then(|s| g.tensor(&s)) {
        let s = f32s(&read_tensor(g, &s, None));
        let per = if s.len() == 1 {
            v.len()
        } else {
            (t.shape[0] * t.shape[1]) as usize
        };
        for (i, x) in v.iter_mut().enumerate() {
            *x *= s[i / per];
        }
    }
    v
}

/// `|a - b| / |b|` over all values.
fn rel_rmse(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        num += ((x - y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

#[test]
#[ignore = "needs the NVFP4 checkpoint (LEGION_TEST_GLM5_HF) and a GGUF (LEGION_TEST_GLM5_GGUF)"]
fn real_checkpoint_matches_the_gguf() {
    let (Ok(hf), Ok(gguf)) = (
        std::env::var("LEGION_TEST_GLM5_HF"),
        std::env::var("LEGION_TEST_GLM5_GGUF"),
    ) else {
        eprintln!("skipping: LEGION_TEST_GLM5_HF / LEGION_TEST_GLM5_GGUF not set");
        return;
    };
    let ours = GgufFile::open(&hf).unwrap();
    let want = GgufFile::open(&gguf).unwrap();
    let mut problems = Vec::new();
    let key = |k: &str| ours.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as usize;
    // The decoder's layers (the checkpoint's MTP block, when it has one, after them).
    let n_layer = key("glm5-next.block_count") - key("glm5-next.nextn_predict_layers");

    // Metadata: every architecture key of the GGUF (per-layer arrays cut to the checkpoint's
    // layers), numbers compared as numbers.
    let num = |v: &Value, cut: bool| -> Vec<f64> {
        match v {
            Value::Array(a) => a
                .iter()
                .filter_map(Value::as_f64)
                .take(if cut { n_layer } else { usize::MAX })
                .collect(),
            v => v.as_f64().into_iter().collect(),
        }
    };
    let skip = ["glm5-next.block_count", "glm5-next.nextn_predict_layers"];
    for (key, value) in want.metadata() {
        if !key.starts_with("glm5-next.") && key != "tokenizer.ggml.pre" || skip.contains(&&*key) {
            continue;
        }
        match ours.get(&key) {
            None => problems.push(format!("missing key {key}")),
            Some(v) if num(&v, true) != num(&value, true) || v.as_str() != value.as_str() => {
                problems.push(format!("key {key}: {v} vs GGUF {value}"))
            }
            Some(_) => {}
        }
    }
    for key in ours.keys() {
        if key.starts_with("glm5-next.") && want.get(&key).is_none() {
            problems.push(format!("extra key {key}"));
        }
    }
    let n_tokens = |g: &GgufFile| {
        g.get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array().map(<[Value]>::len))
    };
    if n_tokens(&ours) != n_tokens(&want) {
        problems.push(format!(
            "tokens: {:?} vs GGUF {:?}",
            n_tokens(&ours),
            n_tokens(&want)
        ));
    }

    // Tensor names and shapes of the checkpoint's layers and the globals.
    let layer = |name: &str| -> Option<usize> {
        name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
    };
    let mut theirs: std::collections::BTreeMap<String, TensorInfo> = want
        .tensors()
        .into_iter()
        .filter(|t| layer(&t.name).is_none_or(|il| il < n_layer))
        .map(|t| (t.name.clone(), t))
        .collect();
    let mut mine = std::collections::BTreeMap::new();
    for t in ours.tensors() {
        if layer(&t.name).is_some_and(|il| il >= n_layer) {
            continue;
        }
        match theirs.remove(&t.name) {
            None if t.name.ends_with(".scale") || t.name.ends_with(".input_scale") => {}
            None => problems.push(format!("extra tensor {}", t.name)),
            Some(r) => {
                if r.shape != t.shape {
                    problems.push(format!(
                        "tensor {}: {:?} vs GGUF {:?}",
                        t.name, t.shape, r.shape
                    ));
                }
                mine.insert(t.name.clone(), (t, r));
            }
        }
    }
    problems.extend(theirs.keys().map(|n| format!("missing tensor {n}")));

    // Values (the big ones by their first rows).
    for (name, (a, b)) in &mine {
        let big = a.shape.iter().product::<u64>() > (64 << 20);
        let rows = big.then(|| {
            if a.shape.len() > 2 {
                16 * a.shape[1]
            } else {
                8192
            }
        });
        let (x, y) = (values(&ours, a, rows), values(&want, b, rows));
        let e = rel_rmse(&x, &y);
        // Tensors the GGUF keeps in F32 (norms, routers, ...) come from the same BF16 values,
        // but for the hyper-connection vectors: F32 in the checkpoint the GGUFs were made
        // from, BF16 in the NVFP4 release (so they agree to BF16 rounding).
        let limit = match b.ggml_type {
            types::F32 if name.contains(".hc_") => 4e-3,
            types::F32 => 1e-6,
            _ => 0.2,
        };
        eprintln!(
            "{name:40} {:>6} vs {:>6}  rel rmse {e:.5}{}",
            a.type_name(),
            b.type_name(),
            if e > limit { "  <-- over" } else { "" },
        );
        if e.is_nan() || e > limit {
            problems.push(format!("tensor {name}: rel rmse {e:.5} > {limit}"));
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}
