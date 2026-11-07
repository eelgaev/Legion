//! Nemotron-H's heterogeneous ("puzzle") MoE blocks and latent MoE (Nemotron 3.5 Bolt
//! 74B-A9B) on the tiny random models of [`super::tiny`], on the CPU with reference kernels.

use super::tiny::{self, Latent, PUZZLE_EXPERTS, PUZZLE_LATENT, PUZZLE_PATTERN};
use super::*;
use crate::ggml::Backend;
use crate::model::core::Model;

fn cpu() -> Backend {
    let mut b = Backend::cpu().unwrap();
    b.set_threads(4);
    b.set_use_ref(true);
    b
}

fn puzzle(dir: &std::path::Path, name: &str, latent: Latent) -> GgufFile {
    let path = dir.join(name);
    tiny::write_puzzle(&path, 3, latent).unwrap();
    GgufFile::open(&path).unwrap()
}

#[test]
fn puzzle_layers_keep_their_own_expert_counts() {
    let dir = tempfile::tempdir().unwrap();
    let g = puzzle(dir.path(), "p.gguf", Latent::Random);
    let hp = Hparams::from_gguf(&g).unwrap();
    let nm = params(&hp).expect("Nemotron-H hyperparameters");
    assert_eq!(hp.n_layer, PUZZLE_PATTERN.len());
    let mut moe = PUZZLE_EXPERTS.iter();
    for (il, c) in PUZZLE_PATTERN.chars().enumerate() {
        match c {
            'E' => {
                let &(n, used, _) = moe.next().unwrap();
                assert_eq!(nm.experts(il), (n, used), "layer {il}");
                assert_eq!(nm.blocks[il], Block::Ffn);
            }
            _ => assert_eq!(nm.experts(il), (0, 0), "layer {il}"),
        }
    }
    // The global counts are the largest: `expert_count` and the most any layer picks.
    assert_eq!(hp.n_expert, 12);
    assert_eq!(hp.n_expert_used, 3);
    assert_eq!(nm.moe_latent, PUZZLE_LATENT);
    assert!(hp.moe && nm.weights_norm && nm.weights_scale == 5.0);
}

#[test]
fn uniform_moe_reads_one_count_for_every_layer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.gguf");
    tiny::write(&path, 1).unwrap();
    let hp = Hparams::from_gguf(&GgufFile::open(&path).unwrap()).unwrap();
    let nm = params(&hp).unwrap();
    assert_eq!((hp.n_expert, hp.n_expert_used), (8, 2));
    assert_eq!(nm.moe_latent, 0);
    // Every MoE layer and the MTP head (block n_layer) have the model's counts.
    for il in 0..=hp.n_layer {
        let want = if il == hp.n_layer || nm.blocks[il] == Block::Ffn {
            (8, 2)
        } else {
            (0, 0)
        };
        assert_eq!(nm.experts(il), want, "layer {il}");
    }
}

/// Identity latent projections give the model without them bit for bit: the experts read the
/// projected input and their weighted sum is projected back, with the router and the shared
/// expert on the unprojected input.
#[test]
fn identity_latent_projections_change_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let plain = puzzle(dir.path(), "plain.gguf", Latent::None);
    let eye = puzzle(dir.path(), "eye.gguf", Latent::Identity);
    let mut backend = cpu();
    let a = Model::load(&plain, |_| backend.buffer_type()).unwrap();
    let b = Model::load(&eye, |_| backend.buffer_type()).unwrap();
    assert!(b.w("blk.1.ffn_latent_down.weight").is_ok());
    let ids: Vec<u32> = "latent experts, two sizes".bytes().map(u32::from).collect();
    let la = a.prompt_logits_all(&mut backend, &ids).unwrap();
    let lb = b.prompt_logits_all(&mut backend, &ids).unwrap();
    assert!(la.iter().all(|x| x.is_finite()));
    assert_eq!(la, lb);
}

/// The latent model's decode steps (one token each, carrying the Mamba-2 states) give the
/// logits of one pass over the whole text bit for bit.
#[test]
fn latent_puzzle_decode_matches_full_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let g = puzzle(dir.path(), "p.gguf", Latent::Random);
    let mut backend = cpu();
    let model = Model::load(&g, |_| backend.buffer_type()).unwrap();
    let n_vocab = model.hp.n_vocab as usize;
    let prompt: Vec<u32> = "puzzle blocks".bytes().map(u32::from).collect();
    let mut state = model.new_state(backend.buffer_type(), 256).unwrap();
    let mut alloc = None;
    let mut rows = vec![
        model
            .forward(&mut backend, &mut alloc, &mut state, &prompt, false)
            .unwrap(),
    ];
    let mut ids = prompt.clone();
    for _ in 0..8 {
        let next = crate::engine::sampling::argmax(rows.last().unwrap()) as u32;
        ids.push(next);
        rows.push(
            model
                .forward(&mut backend, &mut alloc, &mut state, &[next], false)
                .unwrap(),
        );
    }
    let full = model.prompt_logits_all(&mut backend, &ids).unwrap();
    for (i, row) in rows.iter().enumerate() {
        let p = prompt.len() - 1 + i;
        assert!(row.iter().all(|x| x.is_finite()), "position {p}");
        assert_eq!(
            row.as_slice(),
            &full[p * n_vocab..(p + 1) * n_vocab],
            "position {p}"
        );
    }
}
