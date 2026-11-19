//! The shares of a tiny dense qwen35 (and its MTP head): metadata, shapes and bytes.

use super::*;
use crate::ggml;
use crate::gguf::Recipe;
use crate::model::qwen35::tiny::{self, TinyConfig};

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

/// Rows `ranges` (row width `row` bytes) of `whole`, back to back.
fn rows_of(whole: &[u8], row: usize, ranges: &[(usize, usize)]) -> Vec<u8> {
    ranges
        .iter()
        .flat_map(|&(s, l)| whole[s * row..(s + l) * row].to_vec())
        .collect()
}

/// Columns `ranges` of every row (`row` bytes, F32) of `whole`, back to back.
fn cols_of(whole: &[u8], row: usize, ranges: &[(usize, usize)]) -> Vec<u8> {
    let mut out = Vec::new();
    for r in whole.chunks_exact(row) {
        for &(s, l) in ranges {
            out.extend_from_slice(&r[s * 4..(s + l) * 4]);
        }
    }
    out
}

fn tiny(dir: &std::path::Path) -> (GgufFile, GgufFile) {
    let (path, mtp) = (dir.join("q35.gguf"), dir.join("mtp.gguf"));
    let cfg = TinyConfig::default();
    tiny::write(&path, &cfg).unwrap();
    tiny::write_mtp_head(&mtp, &cfg).unwrap();
    (
        GgufFile::open(&path).unwrap(),
        GgufFile::open(&mtp).unwrap(),
    )
}

#[test]
fn shares_halve_heads_delta_net_groups_and_ffn_rows() {
    let dir = tempfile::tempdir().unwrap();
    let (whole, _) = tiny(dir.path());
    let hp = Hparams::from_gguf(&whole).unwrap();
    let d = Dims::of_arch(&whole, "qwen35").unwrap();
    for rank in 0..2 {
        let share = view(&whole, rank, 2).unwrap();
        let sp = Hparams::from_gguf(&share).unwrap();
        assert_eq!(sp.n_head * 2, hp.n_head);
        assert_eq!(sp.n_head_kv * 2, hp.n_head_kv);
        assert_eq!(sp.ssm_n_group * 2, hp.ssm_n_group);
        assert_eq!(sp.ssm_dt_rank * 2, hp.ssm_dt_rank);
        assert_eq!(sp.ssm_d_inner * 2, hp.ssm_d_inner);
        assert_eq!(sp.ssm_d_state, hp.ssm_d_state);
        assert_eq!((sp.n_embd, sp.n_layer), (hp.n_embd, hp.n_layer));
        assert_eq!(full_dims(&sp, 2), d);
        let ff = whole.tensor("blk.0.ffn_down.weight").unwrap().shape[0];
        assert_eq!(share.get_u64("qwen35.feed_forward_length"), Some(ff / 2));
        for name in [
            "token_embd.weight",
            "output.weight",
            "output_norm.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ssm_norm.weight",
            "blk.3.attn_q_norm.weight",
        ] {
            assert_eq!(bytes(&share, name), bytes(&whole, name), "{name}");
        }
        let f32_rows = |name: &str, ranges: &[(usize, usize)]| {
            let t = whole.tensor(name).unwrap();
            let row = ggml::row_size(t.ggml_type, t.shape[0] as usize);
            assert_eq!(
                bytes(&share, name),
                rows_of(&bytes(&whole, name), row, ranges),
                "{name} rank {rank}"
            );
        };
        let f32_cols = |name: &str, ranges: &[(usize, usize)]| {
            let t = whole.tensor(name).unwrap();
            let row = ggml::row_size(t.ggml_type, t.shape[0] as usize);
            assert_eq!(
                bytes(&share, name),
                cols_of(&bytes(&whole, name), row, ranges),
                "{name} rank {rank}"
            );
        };
        let ffr = ff as usize / 2;
        for il in 0..hp.n_layer {
            let t = |n: &str| format!("blk.{il}.{n}");
            f32_rows(&t("ffn_gate.weight"), &[(rank * ffr, ffr)]);
            f32_rows(&t("ffn_up.weight"), &[(rank * ffr, ffr)]);
            f32_cols(&t("ffn_down.weight"), &[(rank * ffr, ffr)]);
        }
        // Attention (layer 3): this rank's gated query heads, kv heads and their columns.
        let (hd, qh, kvh) = (d.head_dim as usize, 2, 1);
        f32_rows("blk.3.attn_q.weight", &[(rank * qh * 2 * hd, qh * 2 * hd)]);
        f32_rows("blk.3.attn_k.weight", &[(rank * kvh * hd, kvh * hd)]);
        f32_rows("blk.3.attn_v.weight", &[(rank * kvh * hd, kvh * hd)]);
        f32_cols("blk.3.attn_output.weight", &[(rank * qh * hd, qh * hd)]);
        // DeltaNet (layer 0): value head `h` reads key head `h % n_k`.
        let (hk, hv, nk, nv) = (
            d.head_k as usize,
            d.head_v as usize,
            d.n_k as usize,
            d.n_v as usize,
        );
        let kq = nk / 2;
        let vheads: Vec<usize> = (0..nv).filter(|h| (h % nk) / kq == rank).collect();
        let v = |unit: usize, at: usize| -> Vec<(usize, usize)> {
            vheads.iter().map(|&h| (at + h * unit, unit)).collect()
        };
        let mut qkv = vec![
            (rank * kq * hk, kq * hk),
            (nk * hk + rank * kq * hk, kq * hk),
        ];
        qkv.extend(v(hv, 2 * nk * hk));
        f32_rows("blk.0.attn_qkv.weight", &qkv);
        f32_rows("blk.0.ssm_conv1d.weight", &qkv);
        f32_rows("blk.0.attn_gate.weight", &v(hv, 0));
        f32_rows("blk.0.ssm_alpha.weight", &v(1, 0));
        f32_rows("blk.0.ssm_beta.weight", &v(1, 0));
        f32_cols("blk.0.ssm_out.weight", &v(hv, 0));
        assert_eq!(
            bytes(&share, "blk.0.ssm_a"),
            cols_of(&bytes(&whole, "blk.0.ssm_a"), nv * 4, &v(1, 0))
        );
    }
}

#[test]
fn the_mtp_head_is_cut_like_the_target() {
    let dir = tempfile::tempdir().unwrap();
    let (whole, head) = tiny(dir.path());
    let il = Hparams::from_gguf(&whole).unwrap().n_layer;
    for rank in 0..2 {
        let sp = Hparams::from_gguf(&view(&whole, rank, 2).unwrap()).unwrap();
        let share = head_view(&head, &sp, rank, 2).unwrap();
        let t = |n: &str| format!("blk.{il}.{n}");
        let shape = |g: &GgufFile, n: &str| g.tensor(&t(n)).unwrap().shape.clone();
        for (name, axis) in [
            ("attn_q.weight", 1),
            ("attn_k.weight", 1),
            ("attn_v.weight", 1),
            ("attn_output.weight", 0),
            ("ffn_gate.weight", 1),
            ("ffn_up.weight", 1),
            ("ffn_down.weight", 0),
        ] {
            let (a, b) = (shape(&head, name), shape(&share, name));
            assert_eq!(a[axis], b[axis] * 2, "{name}");
            assert_eq!(a[1 - axis], b[1 - axis], "{name}");
        }
        for name in [
            "nextn.eh_proj.weight",
            "nextn.enorm.weight",
            "nextn.shared_head_norm.weight",
            "attn_q_norm.weight",
        ] {
            assert_eq!(bytes(&share, &t(name)), bytes(&head, &t(name)), "{name}");
        }
    }
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// `whole` as a checkpoint with channel-wise FP8 weights opens: plus a computed F32 per-row
/// scale `<name>.scale` (values 0.5, 1.5, ...: row `i` holds `i + 0.5`) for each weight of `of`.
fn with_row_scales(whole: &GgufFile, of: &[&str]) -> GgufFile {
    let mut meta = GgufWriter::new();
    for (k, v) in whole.metadata() {
        meta.set(&k, &v).unwrap();
    }
    let paths: Vec<PathBuf> = whole.shard_paths().map(|p| p.to_path_buf()).collect();
    let mut tensors = whole.tensors();
    let mut recipes = HashMap::new();
    for w in of {
        let rows = whole.tensor(w).unwrap().shape[1];
        let v: Vec<u8> = (0..rows)
            .flat_map(|i| (0.5 + i as f32).to_le_bytes())
            .collect();
        let name = format!("{}.scale", w.strip_suffix(".weight").unwrap());
        tensors.push(TensorInfo {
            name: name.clone(),
            ggml_type: crate::ggml::types::F32,
            shape: vec![rows],
            file_index: 0,
            offset: 0,
            size: 4 * rows,
        });
        recipes.insert(
            name,
            Recipe::Computed(std::sync::Arc::new(move || Ok(v.clone()))),
        );
    }
    GgufFile::synthetic(meta, whole.path().to_path_buf(), paths, tensors, recipes).unwrap()
}

#[test]
fn per_row_scales_are_cut_with_their_weights_rows() {
    let dir = tempfile::tempdir().unwrap();
    let (whole, _) = tiny(dir.path());
    let d = Dims::of_arch(&whole, "qwen35").unwrap();
    let down = |il: usize| {
        whole
            .tensor(&format!("blk.{il}.ffn_down.weight"))
            .map(|t| t.ggml_type)
    };
    // Cut by rows (queries, keys, DeltaNet q/k/v and gate, FFN gate) and by columns (the
    // attention, DeltaNet and FFN outputs, whose rows are the embedding).
    let weights = [
        "blk.3.attn_q.weight",
        "blk.3.attn_k.weight",
        "blk.3.attn_output.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ssm_out.weight",
        "blk.1.ffn_gate.weight",
        "blk.1.ffn_down.weight",
    ];
    let synth = with_row_scales(&whole, &weights);
    let (mut by_rows, mut whole_scales) = (0, 0);
    for rank in 0..2 {
        let plain = view(&whole, rank, 2).unwrap();
        let share = view(&synth, rank, 2).unwrap();
        for w in weights {
            assert_eq!(bytes(&share, w), bytes(&plain, w), "{w} rank {rank}");
            let s = format!("{}.scale", w.strip_suffix(".weight").unwrap());
            let t = whole.tensor(w).unwrap();
            let want: Vec<f32> = match piece(&d, &t, &down, 2, rank).unwrap() {
                Piece::Rows(r) => {
                    by_rows += 1;
                    r.iter()
                        .flat_map(|&(a, l)| (a..a + l).map(|i| 0.5 + i as f32))
                        .collect()
                }
                _ => {
                    whole_scales += 1;
                    (0..t.shape[1]).map(|i| 0.5 + i as f32).collect()
                }
            };
            let got = f32s(&bytes(&share, &s));
            assert_eq!(got.len() as u64, share.tensor(w).unwrap().shape[1], "{s}");
            assert_eq!(got, want, "{s} rank {rank}");
        }
    }
    assert_eq!((by_rows, whole_scales), (10, 6));
}

#[test]
fn a_per_row_scale_must_match_its_weights_rows() {
    let t = |name: &str, shape: Vec<u64>| TensorInfo {
        name: name.into(),
        ggml_type: crate::ggml::types::F32,
        shape,
        file_index: 0,
        offset: 0,
        size: 0,
    };
    let w = t("blk.0.attn_q.weight", vec![64, 32]);
    let rows = Piece::Rows(vec![(0, 8), (16, 8)]);
    assert_eq!(
        scale_piece(&t("blk.0.attn_q.scale", vec![32]), &w, rows.clone()),
        Ok(Piece::Cols(vec![(0, 8), (16, 8)]))
    );
    assert_eq!(
        scale_piece(
            &t("blk.0.attn_q.scale", vec![32]),
            &w,
            Piece::Cols(vec![(0, 32)])
        ),
        Ok(Piece::Mirrored)
    );
    assert!(scale_piece(&t("blk.0.attn_q.scale", vec![64]), &w, rows.clone()).is_err());
    assert!(scale_piece(&t("blk.0.attn_q.scale", vec![32, 1]), &w, rows).is_err());
}

#[test]
fn shares_refuse_other_architectures_and_odd_splits() {
    let dir = tempfile::tempdir().unwrap();
    let (whole, _) = tiny(dir.path());
    // The tiny model's 4 query and 2 kv heads do not divide among 3 ranks.
    assert!(view(&whole, 0, 3).is_err());
    let gemma = dir.path().join("g.gguf");
    crate::model::gemma4::tiny::write_dense(&gemma, 5).unwrap();
    let Err(e) = view(&GgufFile::open(&gemma).unwrap(), 0, 2) else {
        panic!("a qwen35 share of a Gemma 4 model");
    };
    assert!(e.contains("not qwen35"), "{e}");
}
