//! The shares of a tiny qwen4exp (and its MTP head): metadata, shapes and bytes.

use super::*;
use crate::model::core::Hparams;

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

/// Rows `ranges` (row width `row` bytes) of every `ne1`-row matrix of `whole`, back to back.
fn rows_of(whole: &[u8], row: usize, ne1: usize, ranges: &[(usize, usize)]) -> Vec<u8> {
    let mut out = Vec::new();
    for m in whole.chunks_exact(row * ne1) {
        for &(s, l) in ranges {
            out.extend_from_slice(&m[s * row..(s + l) * row]);
        }
    }
    out
}

fn tiny(dir: &std::path::Path) -> (GgufFile, GgufFile) {
    let (path, mtp) = (dir.join("q4.gguf"), dir.join("mtp.gguf"));
    crate::model::qwen4::tiny::write_tp(&path, Some(&mtp), 3).unwrap();
    (
        GgufFile::open(&path).unwrap(),
        GgufFile::open(&mtp).unwrap(),
    )
}

#[test]
fn shares_halve_heads_delta_net_groups_and_expert_rows() {
    let dir = tempfile::tempdir().unwrap();
    let (whole, _) = tiny(dir.path());
    let hp = Hparams::from_gguf(&whole).unwrap();
    let d = Dims::of(&whole).unwrap();
    for rank in 0..2 {
        let share = view(&whole, rank, 2).unwrap();
        let sp = Hparams::from_gguf(&share).unwrap();
        assert_eq!(sp.n_head * 2, hp.n_head);
        assert_eq!(sp.n_head_kv * 2, hp.n_head_kv);
        assert_eq!(sp.ssm_n_group * 2, hp.ssm_n_group);
        assert_eq!(sp.ssm_dt_rank * 2, hp.ssm_dt_rank);
        assert_eq!(sp.ssm_d_inner * 2, hp.ssm_d_inner);
        assert_eq!(sp.ssm_d_state, hp.ssm_d_state);
        assert_eq!(sp.n_embd, hp.n_embd);
        assert_eq!(sp.n_expert, hp.n_expert);
        let ff = whole.tensor("blk.0.ffn_down_exps.weight").unwrap().shape[0];
        assert_eq!(
            share.get_u64("qwen4exp.expert_feed_forward_length"),
            Some(ff / 2)
        );
        // Mirrored: the same bytes.
        for name in [
            "token_embd.weight",
            "per_layer_token_embd.weight",
            "blk.0.hc_attn_down.weight",
            "blk.0.ssm_norm.weight",
            "blk.0.ffn_gate_inp.weight",
            "blk.1.ple_key.weight",
            "blk.3.indexer.q_proj.weight",
            "blk.3.attn_q_norm.weight",
        ] {
            assert_eq!(bytes(&share, name), bytes(&whole, name), "{name}");
        }
        // The n-gram table stays where the reader finds it.
        let (a, b) = (
            whole.tensor("per_layer_token_embd.weight").unwrap(),
            share.tensor("per_layer_token_embd.weight").unwrap(),
        );
        assert_eq!((a.offset, a.file_index), (b.offset, b.file_index));

        let (hk, hv, nk, nv) = (
            d.head_k as usize,
            d.head_v as usize,
            d.n_k as usize,
            d.n_v as usize,
        );
        let kq = nk / 2;
        // Value heads: those whose key head (`h % n_k`) is this rank's.
        let vheads: Vec<usize> = (0..nv).filter(|h| (h % nk) / kq == rank).collect();
        let vrows = |unit: usize| -> Vec<(usize, usize)> {
            vheads.iter().map(|&h| (h * unit, unit)).collect()
        };
        let check_rows = |name: &str, ranges: &[(usize, usize)]| {
            let t = whole.tensor(name).unwrap();
            let row = ggml::row_size(t.ggml_type, t.shape[0] as usize);
            let want = rows_of(&bytes(&whole, name), row, t.shape[1] as usize, ranges);
            assert_eq!(bytes(&share, name), want, "{name} rank {rank}");
        };
        let check_cols = |name: &str, ranges: &[(usize, usize)]| {
            let t = whole.tensor(name).unwrap();
            let ty = t.ggml_type;
            let row = ggml::row_size(ty, t.shape[0] as usize);
            let mut want = Vec::new();
            for r in bytes(&whole, name).chunks_exact(row) {
                for &(s, l) in ranges {
                    let (a, n) = (ggml::row_size(ty, s), ggml::row_size(ty, l));
                    want.extend_from_slice(&r[a..a + n]);
                }
            }
            assert_eq!(bytes(&share, name), want, "{name} rank {rank}");
        };
        // DeltaNet (layer 0): q and k of this rank's key heads, v of its value heads.
        let mut qkv = vec![
            (rank * kq * hk, kq * hk),
            (nk * hk + rank * kq * hk, kq * hk),
        ];
        qkv.extend(vrows(hv).into_iter().map(|(s, l)| (2 * nk * hk + s, l)));
        check_rows("blk.0.attn_qkv.weight", &qkv);
        check_rows("blk.0.ssm_conv1d.weight", &qkv);
        check_rows("blk.0.attn_gate.weight", &vrows(hv));
        check_rows("blk.0.ssm_alpha.weight", &vrows(1));
        check_rows("blk.0.ssm_beta.weight", &vrows(1));
        check_cols("blk.0.ssm_a", &vrows(1));
        check_cols("blk.0.ssm_dt.bias", &vrows(1));
        check_cols("blk.0.ssm_out.weight", &vrows(hv));
        // Attention (layer 3): gated query heads, kv heads, output columns.
        let (dh, qh, kvh) = (
            d.head_dim as usize,
            d.n_head as usize / 2,
            d.n_head_kv as usize / 2,
        );
        check_rows("blk.3.attn_q.weight", &[(rank * qh * 2 * dh, qh * 2 * dh)]);
        check_rows("blk.3.attn_k.weight", &[(rank * kvh * dh, kvh * dh)]);
        check_rows("blk.3.attn_v.weight", &[(rank * kvh * dh, kvh * dh)]);
        check_cols("blk.3.attn_output.weight", &[(rank * qh * dh, qh * dh)]);
        // MoE: every expert's rows, the shared expert's rows.
        let half = ff as usize / 2;
        for il in [0, 3] {
            let t = |s: &str| format!("blk.{il}.{s}");
            check_rows(&t("ffn_gate_exps.weight"), &[(rank * half, half)]);
            check_rows(&t("ffn_up_exps.weight"), &[(rank * half, half)]);
            check_cols(&t("ffn_down_exps.weight"), &[(rank * half, half)]);
            check_rows(&t("ffn_gate_shexp.weight"), &[(rank * half, half)]);
            check_cols(&t("ffn_down_shexp.weight"), &[(rank * half, half)]);
        }
    }
}

#[test]
fn the_mtp_head_is_cut_like_the_target() {
    let dir = tempfile::tempdir().unwrap();
    let (_, mtp) = tiny(dir.path());
    let d = Dims::of(&mtp).unwrap();
    for rank in 0..2 {
        let share = view(&mtp, rank, 2).unwrap();
        let s = Dims::of(&share).unwrap();
        assert_eq!(s.n_head * 2, d.n_head);
        assert_eq!(s.n_head_kv * 2, d.n_head_kv);
        let il = crate::model::qwen4::tiny::N_LAYER;
        let q = share.tensor(&format!("blk.{il}.attn_q.weight")).unwrap();
        assert_eq!(q.shape[1], d.n_head * d.head_dim);
        assert_eq!(
            bytes(&share, &format!("blk.{il}.nextn.eh_proj.weight")),
            bytes(&mtp, &format!("blk.{il}.nextn.eh_proj.weight"))
        );
    }
}

#[test]
fn expert_ranges_cover_the_experts_once() {
    assert_eq!(expert_range(512, 2, 0), Ok((0, 256)));
    assert_eq!(expert_range(512, 2, 1), Ok((256, 256)));
    assert_eq!(expert_range(16, 4, 3), Ok((12, 4)));
    assert_eq!(expert_range(16, 1, 0), Ok((0, 16)));
    for (e, n, r) in [(16, 3, 0), (0, 2, 0), (16, 2, 2)] {
        assert!(expert_range(e, n, r).is_err(), "{e} {n} {r}");
    }
}

/// `--tp-moe-ep`: whole experts, half of them per rank; everything else as with rows.
#[test]
fn whole_experts_take_half_the_stack() {
    let dir = tempfile::tempdir().unwrap();
    let (whole, mtp) = tiny(dir.path());
    for rank in 0..2 {
        let share = view_with(&whole, rank, 2, Experts::Whole).unwrap();
        let rows = view(&whole, rank, 2).unwrap();
        let ff = whole.get_u64("qwen4exp.expert_feed_forward_length");
        assert_eq!(share.get_u64("qwen4exp.expert_feed_forward_length"), ff);
        // The router still picks among all of them.
        let n_expert = whole.get_u64("qwen4exp.expert_count");
        assert_eq!(share.get_u64("qwen4exp.expert_count"), n_expert);
        for il in [0, 3] {
            for k in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
                let name = format!("blk.{il}.{k}.weight");
                let (w, s) = (whole.tensor(&name).unwrap(), share.tensor(&name).unwrap());
                assert_eq!(s.shape[..2], w.shape[..2], "{name}");
                assert_eq!(s.shape[2] * 2, w.shape[2], "{name}");
                let b = bytes(&whole, &name);
                let half = b.len() / 2;
                assert_eq!(bytes(&share, &name), b[rank * half..][..half], "{name}");
            }
            // The router whole, the shared expert split by rows as before.
            for k in ["ffn_gate_inp", "ffn_gate_shexp", "ffn_down_shexp"] {
                let name = format!("blk.{il}.{k}.weight");
                assert_eq!(bytes(&share, &name), bytes(&rows, &name), "{name}");
            }
        }
        let il = crate::model::qwen4::tiny::N_LAYER;
        let head = view_with(&mtp, rank, 2, Experts::Whole).unwrap();
        let name = format!("blk.{il}.ffn_up_exps.weight");
        let b = bytes(&mtp, &name);
        let half = b.len() / 2;
        assert_eq!(bytes(&head, &name), b[rank * half..][..half], "{name}");
    }
}

/// `--tp-moe-ep-place` (#803): each rank's experts of a placed layer, gathered in ascending
/// order, their bytes as shipped; unnamed layers keep the contiguous halves.
#[test]
fn placed_experts_gather_each_ranks_experts() {
    let dir = tempfile::tempdir().unwrap();
    let (whole, mtp) = tiny(dir.path());
    let n_expert = whole.get_u64("qwen4exp.expert_count").unwrap() as usize;
    let head_il = crate::model::qwen4::tiny::N_LAYER;
    // Layer 3: the even experts on rank 0, the odd ones on rank 1; the head: halves swapped.
    let odd: Vec<u8> = (0..n_expert).map(|e| (e % 2) as u8).collect();
    let swapped: Vec<u8> = (0..n_expert).map(|e| u8::from(e < n_expert / 2)).collect();
    let place = Arc::new(Placement {
        n_expert,
        ranks: 2,
        layers: [(3, odd.clone()), (head_il, swapped.clone())].into(),
    });
    place.check().unwrap();
    let experts = || Experts::Placed(place.clone());
    for rank in 0..2 {
        let share = view_with(&whole, rank, 2, experts()).unwrap();
        let head = view_with(&mtp, rank, 2, experts()).unwrap();
        let pick = |file: &GgufFile, name: &str, owner: &[u8]| -> Vec<u8> {
            let b = bytes(file, name);
            let per = b.len() / n_expert;
            (0..n_expert)
                .filter(|&e| owner[e] as usize == rank)
                .flat_map(|e| b[e * per..(e + 1) * per].to_vec())
                .collect()
        };
        for k in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
            let name = format!("blk.3.{k}.weight");
            assert_eq!(share.tensor(&name).unwrap().shape[2] as usize, n_expert / 2);
            assert_eq!(bytes(&share, &name), pick(&whole, &name, &odd), "{name}");
            let name = format!("blk.0.{k}.weight");
            let b = bytes(&whole, &name);
            let half = b.len() / 2;
            assert_eq!(bytes(&share, &name), b[rank * half..][..half], "{name}");
            let name = format!("blk.{head_il}.{k}.weight");
            assert_eq!(bytes(&head, &name), pick(&mtp, &name, &swapped), "{name}");
        }
    }
    // A table for another model's experts is refused.
    let other = Arc::new(Placement {
        n_expert: n_expert * 2,
        ranks: 2,
        layers: Default::default(),
    });
    assert!(view_with(&whole, 0, 2, Experts::Placed(other)).is_err());
}

#[test]
fn shares_refuse_what_does_not_divide() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("q4.gguf");
    // The plain tiny model's experts have 32 rows: 16 per rank is half a Q8_0 block.
    crate::model::qwen4::tiny::write(&path, None, 3).unwrap();
    let whole = GgufFile::open(&path).unwrap();
    let e = view(&whole, 0, 2).err().unwrap();
    assert!(e.contains("whole blocks"), "{e}");
    // Three ranks: two kv heads do not divide.
    let path = dir.path().join("q4tp.gguf");
    crate::model::qwen4::tiny::write_tp(&path, None, 3).unwrap();
    let whole = GgufFile::open(&path).unwrap();
    let e = view(&whole, 0, 3).err().unwrap();
    assert!(e.contains("do not divide"), "{e}");
}

#[test]
fn value_heads_tile_over_the_key_heads() {
    // The released shape: 16 key heads, 48 value heads (value head h reads key head h % 16).
    let d = Dims {
        n_head: 24,
        n_head_kv: 2,
        head_dim: 256,
        n_k: 16,
        head_k: 128,
        n_v: 48,
        head_v: 128,
    };
    assert_eq!(value_heads(&d, 2, 0), [(0, 8), (16, 8), (32, 8)]);
    assert_eq!(value_heads(&d, 2, 1), [(8, 8), (24, 8), (40, 8)]);
    // Local value head j reads local key head j % 8 in either share.
    for rank in 0..2 {
        let local: Vec<u64> = value_heads(&d, 2, rank)
            .into_iter()
            .flat_map(|(s, l)| s..s + l)
            .collect();
        for (j, h) in local.iter().enumerate() {
            assert_eq!(h % 16 - rank as u64 * 8, j as u64 % 8);
        }
    }
}

/// `--tp-moe-ep` on a safetensors checkpoint's stacked NVFP4 experts (#1150): each rank reads
/// and repacks only its own experts, and gets the bytes the whole stack's recipe would give
/// for them. The whole stack's recipe is never run.
#[test]
fn nvfp4_stacks_repack_only_this_ranks_experts() {
    use crate::gguf::Nvfp4Source;
    use crate::hf::nvfp4::repack_nvfp4;
    let dir = tempfile::tempdir().unwrap();
    let (whole, _) = tiny(dir.path());
    let n_expert = whole.get_u64("qwen4exp.expert_count").unwrap() as usize;
    let (rows, k) = (3usize, 128usize);
    let (wb, sb) = (rows * k / 2, rows * k / 16);
    // Every expert's packed codes, then its E4M3 sub-block scales (sign bits set on some),
    // back to back in a file of their own.
    let mut seed = 0x9e37_79b9_u32;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 24) as u8
    };
    let data: Vec<u8> = (0..n_expert * (wb + sb)).map(|_| next()).collect();
    let file = dir.path().join("experts.bin");
    std::fs::write(&file, &data).unwrap();
    let mut paths: Vec<PathBuf> = whole.shard_paths().map(|p| p.to_path_buf()).collect();
    let fi = paths.len();
    paths.push(file);
    let experts: Vec<Nvfp4Source> = (0..n_expert)
        .map(|e| {
            let at = (e * (wb + sb)) as u64;
            Nvfp4Source {
                weight: FileRange {
                    file: fi,
                    offset: at,
                    len: wb as u64,
                },
                scale: FileRange {
                    file: fi,
                    offset: at + wb as u64,
                    len: sb as u64,
                },
                scale2: 1.0,
            }
        })
        .collect();
    let repacked = |e: usize| {
        let at = e * (wb + sb);
        repack_nvfp4(&data[at..at + wb], &data[at + wb..at + wb + sb], rows, k)
    };
    // blk.0's up stack replaced by the NVFP4 one; its recipe fails if it is ever run.
    let name = "blk.0.ffn_up_exps.weight";
    let per = (rows * k / 64 * 36) as u64;
    let mut meta = crate::gguf::GgufWriter::new();
    for (key, v) in whole.metadata() {
        meta.set(&key, &v).unwrap();
    }
    let mut tensors = whole.tensors();
    let t = tensors.iter_mut().find(|t| t.name == name).unwrap();
    t.ggml_type = crate::ggml::types::NVFP4;
    t.shape = vec![k as u64, rows as u64, n_expert as u64];
    t.size = per * n_expert as u64;
    let fail: Compute = Arc::new(|| Err("the whole stack was read".to_string()));
    let recipes = HashMap::from([(name.to_string(), Recipe::Computed(fail))]);
    let mut synth =
        GgufFile::synthetic(meta, whole.path().to_path_buf(), paths, tensors, recipes).unwrap();
    synth.set_nvfp4_stacks(HashMap::from([(
        name.to_string(),
        Arc::new(Nvfp4Stack { rows, k, experts }),
    )]));
    for rank in 0..2 {
        let share = view_with(&synth, rank, 2, Experts::Whole).unwrap();
        let s = share.tensor(name).unwrap();
        assert_eq!(s.shape, [k as u64, rows as u64, n_expert as u64 / 2]);
        let half = n_expert / 2;
        let want: Vec<u8> = (rank * half..(rank + 1) * half)
            .flat_map(&repacked)
            .collect();
        assert_eq!(bytes(&share, name), want, "rank {rank}");
    }
    // Placed: the odd experts on rank 1, in ascending order.
    let odd: Vec<u8> = (0..n_expert).map(|e| (e % 2) as u8).collect();
    let place = Arc::new(Placement {
        n_expert,
        ranks: 2,
        layers: [(0, odd)].into(),
    });
    place.check().unwrap();
    let share = view_with(&synth, 1, 2, Experts::Placed(place)).unwrap();
    let want: Vec<u8> = (1..n_expert).step_by(2).flat_map(&repacked).collect();
    assert_eq!(bytes(&share, name), want);
}
