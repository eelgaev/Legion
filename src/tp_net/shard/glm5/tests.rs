//! The shares of a tiny glm5-next model: metadata, shapes and bytes; the NVFP4 expert cuts
//! against the whole stack's repack.

use super::*;
use crate::ggml::types;

/// Every byte of tensor `name` as the loader reads it from `g`.
fn bytes(g: &GgufFile, name: &str) -> Vec<u8> {
    use std::os::unix::fs::FileExt;
    let t = g.tensor(name).unwrap();
    let paths: Vec<PathBuf> = g.shard_paths().map(|p| p.to_path_buf()).collect();
    let read = |file: usize, offset: u64, len: u64| {
        let mut b = vec![0u8; len as usize];
        std::fs::File::open(&paths[file])
            .unwrap()
            .read_exact_at(&mut b, offset)
            .unwrap();
        b
    };
    let b = match g.recipe(name) {
        Some(Recipe::Computed(f)) => f().unwrap(),
        Some(Recipe::Gather(parts)) => parts
            .iter()
            .flat_map(|p| read(p.file, p.offset, p.len))
            .collect(),
        None => read(t.file_index, t.offset, t.size),
    };
    assert_eq!(b.len() as u64, t.size, "{name}");
    b
}

/// F32 values `0, 1, 2, ..` as bytes.
fn ramp(n: usize) -> Vec<u8> {
    (0..n).flat_map(|i| (i as f32).to_le_bytes()).collect()
}

/// Items `start..start + len` of the outermost axis (`per` bytes each) of `whole`.
fn outer(whole: &[u8], per: usize, start: usize, len: usize) -> Vec<u8> {
    whole[start * per..(start + len) * per].to_vec()
}

/// Columns `start..start + len` (`elem` bytes each) of every `ne0`-wide row of `whole`.
fn cols(whole: &[u8], ne0: usize, elem: usize, start: usize, len: usize) -> Vec<u8> {
    whole
        .chunks_exact(ne0 * elem)
        .flat_map(|r| r[start * elem..(start + len) * elem].to_vec())
        .collect()
}

/// Rows `start..start + len` of every `ne1`-row matrix (rows of `row` bytes) of `whole`.
fn rows(whole: &[u8], row: usize, ne1: usize, start: usize, len: usize) -> Vec<u8> {
    whole
        .chunks_exact(row * ne1)
        .flat_map(|m| m[start * row..(start + len) * row].to_vec())
        .collect()
}

const E: i64 = 16;
const N_HEAD: i64 = 4;
const HEAD_K: i64 = 8;
const HEAD_V: i64 = 6;
const KDA_HEAD: i64 = 4;
const N_KDA: i64 = 4;
const N_FF: i64 = 8;
const N_EXP: i64 = 3;

/// Tensors of a two-layer glm5-next model (layer 0 KDA with a dense FFN, layer 1 MLA with
/// routed and shared experts), F32 ramps: name and shape.
fn tensors() -> Vec<(&'static str, Vec<i64>)> {
    let kda = KDA_HEAD * N_KDA;
    vec![
        ("token_embd.weight", vec![E, 10]),
        ("output.weight", vec![E, 10]),
        ("output_norm.weight", vec![E]),
        ("blk.0.attn_norm.weight", vec![E]),
        ("blk.0.hc_attn_fn.weight", vec![4 * E, 24]),
        ("blk.0.attn_q.weight", vec![E, kda]),
        ("blk.0.attn_k.weight", vec![E, kda]),
        ("blk.0.attn_v.weight", vec![E, kda]),
        ("blk.0.ssm_conv1d_q.weight", vec![4, 1, kda]),
        ("blk.0.ssm_f_a.weight", vec![E, KDA_HEAD]),
        ("blk.0.ssm_f_b.weight", vec![KDA_HEAD, kda]),
        ("blk.0.ssm_beta.weight", vec![E, N_KDA]),
        ("blk.0.ssm_a", vec![N_KDA]),
        ("blk.0.ssm_dt.bias", vec![kda]),
        ("blk.0.ssm_norm.weight", vec![KDA_HEAD]),
        ("blk.0.attn_output.weight", vec![kda, E]),
        ("blk.0.ffn_gate.weight", vec![E, N_FF]),
        ("blk.0.ffn_up.weight", vec![E, N_FF]),
        ("blk.0.ffn_down.weight", vec![N_FF, E]),
        ("blk.1.attn_q_a.weight", vec![E, 8]),
        ("blk.1.attn_q_b.weight", vec![8, N_HEAD * HEAD_K]),
        ("blk.1.attn_kv_a_mqa.weight", vec![E, 6]),
        ("blk.1.attn_k_b.weight", vec![HEAD_K, 6, N_HEAD]),
        ("blk.1.attn_v_b.weight", vec![6, HEAD_V, N_HEAD]),
        ("blk.1.indexer.attn_k.weight", vec![E, 4]),
        ("blk.1.attn_output.weight", vec![N_HEAD * HEAD_V, E]),
        ("blk.1.ffn_gate_inp.weight", vec![E, N_EXP]),
        ("blk.1.exp_probs_b.bias", vec![N_EXP]),
        ("blk.1.ffn_gate_exps.weight", vec![E, N_FF, N_EXP]),
        ("blk.1.ffn_up_exps.weight", vec![E, N_FF, N_EXP]),
        ("blk.1.ffn_down_exps.weight", vec![N_FF, E, N_EXP]),
        ("blk.1.ffn_gate_shexp.weight", vec![E, N_FF]),
        ("blk.1.ffn_up_shexp.weight", vec![E, N_FF]),
        ("blk.1.ffn_down_shexp.weight", vec![N_FF, E]),
    ]
}

fn tiny(path: &std::path::Path) -> GgufFile {
    let mut g = GgufWriter::new();
    let u = |v: i64| Value::U32(v as u32);
    for (k, v) in [
        ("general.architecture", Value::String(ARCH.into())),
        ("glm5-next.block_count", u(2)),
        (
            "glm5-next.attention.head_count_kv",
            Value::Array(vec![Value::I32(0), Value::I32(1)]),
        ),
        ("glm5-next.attention.head_count", u(N_HEAD)),
        ("glm5-next.attention.key_length_mla", u(HEAD_K)),
        ("glm5-next.attention.value_length_mla", u(HEAD_V)),
        ("glm5-next.kda.head_dim", u(KDA_HEAD)),
        ("glm5-next.feed_forward_length", u(N_FF)),
        ("glm5-next.expert_feed_forward_length", u(N_FF)),
        ("glm5-next.expert_shared_feed_forward_length", u(N_FF)),
    ] {
        g.set(k, &v).unwrap();
    }
    for (name, shape) in tensors() {
        let n: i64 = shape.iter().product();
        g.add_tensor(name, types::F32, &shape, &ramp(n as usize))
            .unwrap();
    }
    g.write(path).unwrap();
    GgufFile::open(path).unwrap()
}

#[test]
fn shares_halve_heads_kda_heads_and_ffn_rows() {
    let dir = tempfile::tempdir().unwrap();
    let whole = tiny(&dir.path().join("glm.gguf"));
    let f = 4; // F32 bytes
    for rank in 0..2usize {
        let share = view(&whole, rank, 2).unwrap();
        let key = |k: &str| share.get_u64(&format!("{ARCH}.{k}"));
        assert_eq!(key("attention.head_count"), Some(N_HEAD as u64 / 2));
        assert_eq!(key("feed_forward_length"), Some(N_FF as u64 / 2));
        assert_eq!(key("expert_feed_forward_length"), Some(N_FF as u64 / 2));
        assert_eq!(
            key("expert_shared_feed_forward_length"),
            Some(N_FF as u64 / 2)
        );
        assert_eq!(key("block_count"), Some(2));
        let r = rank;
        let kda = (KDA_HEAD * N_KDA) as usize;
        let half_kda = kda / 2;
        let ff = N_FF as usize / 2;
        let e = E as usize;
        for (name, _) in tensors() {
            let w = bytes(&whole, name);
            let t = whole.tensor(name).unwrap();
            let want = match name {
                "blk.0.attn_q.weight" | "blk.0.attn_k.weight" | "blk.0.attn_v.weight" => {
                    outer(&w, e * f, r * half_kda, half_kda)
                }
                "blk.0.ssm_conv1d_q.weight" => outer(&w, 4 * f, r * half_kda, half_kda),
                "blk.0.ssm_f_b.weight" => outer(&w, KDA_HEAD as usize * f, r * half_kda, half_kda),
                "blk.0.ssm_dt.bias" => outer(&w, f, r * half_kda, half_kda),
                "blk.0.ssm_beta.weight" => outer(&w, e * f, r * 2, 2),
                "blk.0.ssm_a" => outer(&w, f, r * 2, 2),
                "blk.0.attn_output.weight" => cols(&w, kda, f, r * half_kda, half_kda),
                "blk.0.ffn_gate.weight" | "blk.0.ffn_up.weight" => outer(&w, e * f, r * ff, ff),
                "blk.0.ffn_down.weight" => cols(&w, N_FF as usize, f, r * ff, ff),
                "blk.1.attn_q_b.weight" => {
                    outer(&w, 8 * f, r * 2 * HEAD_K as usize, 2 * HEAD_K as usize)
                }
                "blk.1.attn_k_b.weight" => outer(&w, HEAD_K as usize * 6 * f, r * 2, 2),
                "blk.1.attn_v_b.weight" => outer(&w, 6 * HEAD_V as usize * f, r * 2, 2),
                "blk.1.attn_output.weight" => cols(
                    &w,
                    (N_HEAD * HEAD_V) as usize,
                    f,
                    r * 2 * HEAD_V as usize,
                    2 * HEAD_V as usize,
                ),
                "blk.1.ffn_gate_exps.weight" | "blk.1.ffn_up_exps.weight" => {
                    rows(&w, e * f, N_FF as usize, r * ff, ff)
                }
                "blk.1.ffn_down_exps.weight" => cols(&w, N_FF as usize, f, r * ff, ff),
                "blk.1.ffn_gate_shexp.weight" | "blk.1.ffn_up_shexp.weight" => {
                    outer(&w, e * f, r * ff, ff)
                }
                "blk.1.ffn_down_shexp.weight" => cols(&w, N_FF as usize, f, r * ff, ff),
                // Mirrored: the same bytes and shape.
                _ => {
                    assert_eq!(share.tensor(name).unwrap().shape, t.shape, "{name}");
                    w.clone()
                }
            };
            assert_eq!(bytes(&share, name), want, "rank {rank}: {name}");
        }
        assert_eq!(
            share.tensor("blk.1.attn_k_b.weight").unwrap().shape,
            vec![HEAD_K as u64, 6, 2]
        );
        assert_eq!(
            share.tensor("blk.1.ffn_down_exps.weight").unwrap().shape,
            vec![ff as u64, e as u64, N_EXP as u64]
        );
    }
}

#[test]
fn heads_that_do_not_divide_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let whole = tiny(&dir.path().join("glm.gguf"));
    let e = view(&whole, 0, 3).err().unwrap();
    assert!(e.contains("do not divide among 3 ranks"), "{e}");
}

#[test]
fn layer_tensors_it_does_not_know_are_refused() {
    let d = Dims {
        n_head: 4,
        head_k: 8,
        head_v: 8,
        kda_head: 4,
        mla: vec![true],
    };
    let t = TensorInfo {
        name: "blk.0.new_proj.weight".into(),
        ggml_type: types::F32,
        shape: vec![16, 16],
        file_index: 0,
        offset: 0,
        size: 16 * 16 * 4,
    };
    let e = piece(&d, &t, &|_| None, &|_, _| Some(types::F32), 2, 0)
        .err()
        .unwrap();
    assert!(e.contains("cannot split"), "{e}");
}

/// Two experts of `rows` x `k` ModelOpt NVFP4 codes and scales (pseudo-random) in one file.
fn stack(dir: &std::path::Path, rows: usize, k: usize) -> (Arc<Nvfp4Stack>, Vec<PathBuf>, Vec<u8>) {
    let path = dir.join("experts.bin");
    let mut data = Vec::new();
    let mut experts = Vec::new();
    let mut x = 0x9e37_79b9u32;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x as u8
    };
    let mut whole = Vec::new();
    for _ in 0..2 {
        let w: Vec<u8> = (0..rows * k / 2).map(|_| next()).collect();
        // E4M3 scales without the sign bit's surprises: the repack clears it anyway.
        let s: Vec<u8> = (0..rows * k / 16).map(|_| next() & 0x7f).collect();
        whole.extend(crate::hf::nvfp4::repack_nvfp4(&w, &s, rows, k));
        let at = data.len() as u64;
        data.extend_from_slice(&w);
        let sat = data.len() as u64;
        data.extend_from_slice(&s);
        experts.push(crate::gguf::Nvfp4Source {
            weight: FileRange {
                file: 0,
                offset: at,
                len: w.len() as u64,
            },
            scale: FileRange {
                file: 0,
                offset: sat,
                len: s.len() as u64,
            },
            scale2: 1.0,
        });
    }
    std::fs::write(&path, &data).unwrap();
    (Arc::new(Nvfp4Stack { rows, k, experts }), vec![path], whole)
}

#[test]
fn nvfp4_expert_cuts_are_the_whole_repack_cut() {
    let dir = tempfile::tempdir().unwrap();
    let (rows_n, k) = (6, 256);
    let (st, paths, whole) = stack(dir.path(), rows_n, k);
    let row = k / 64 * 36;
    let got = nvfp4_rows(st.clone(), &paths, 2, 3)().unwrap();
    assert_eq!(got, rows(&whole, row, rows_n, 2, 3));
    let got = nvfp4_cols(st.clone(), &paths, 64, 128)().unwrap();
    // 36 bytes per 64-element block.
    assert_eq!(got, cols(&whole, k / 64, 36, 1, 2));
    assert!(nvfp4_cols(st, &paths, 32, 64)().is_err());
}
