//! The expert split: the renumbering, and the router and experts the loader reads
//! through it.

use super::*;
use crate::ggml::{DeviceKind, device_by_kind, types};
use crate::gguf::GgufWriter;
use crate::load_io::Options;
use crate::model::glm5next::par::{ParPlan, expert_bounds, mirror_name};
use crate::weights::Weights;

#[test]
fn interleave_gives_each_gpu_every_nth_expert() {
    let order = expert_order(ExpertSplit::Interleave, 288, 4).unwrap();
    let bounds = expert_bounds(288, 4);
    for d in 0..4 {
        let mine = &order[bounds[d] as usize..bounds[d + 1] as usize];
        let want: Vec<u32> = (d as u32..288).step_by(4).collect();
        assert_eq!(mine, &want[..], "GPU {d}");
    }
    assert_eq!(expert_order(ExpertSplit::Contiguous, 288, 4), None);
    assert_eq!(expert_order(ExpertSplit::Interleave, 288, 1), None);
}

/// The default keeps the file's order: greedy output as before #677.
#[test]
fn the_default_split_keeps_the_file_order() {
    assert_eq!(ExpertSplit::default(), ExpertSplit::Contiguous);
    assert_eq!(expert_order(ExpertSplit::default(), 288, 4), None);
}

#[test]
fn every_order_is_a_permutation() {
    for n_expert in [1usize, 7, 10, 64, 256, 288] {
        for n_dev in 2..=4 {
            let order = expert_order(ExpertSplit::Interleave, n_expert, n_dev).unwrap();
            let mut seen = order.clone();
            seen.sort_unstable();
            assert_eq!(seen, (0..n_expert as u32).collect::<Vec<_>>());
            let inv = inverse(&order);
            for (i, &e) in order.iter().enumerate() {
                assert_eq!(inv[e as usize], i as u32);
            }
        }
    }
}

#[test]
fn a_permuted_tensor_gathers_its_expert_slices() {
    let t = TensorInfo {
        name: "blk.1.ffn_up_exps.weight".into(),
        ggml_type: types::Q8_0,
        shape: vec![32, 2, 4],
        file_index: 2,
        offset: 96,
        size: 4 * 68,
    };
    let Recipe::Gather(parts) = permuted(&t, &[0, 2, 1, 3]).unwrap() else {
        panic!("a gather");
    };
    let at: Vec<(usize, u64, u64)> = parts.iter().map(|p| (p.file, p.offset, p.len)).collect();
    assert_eq!(
        at,
        vec![
            (2, 96, 68),
            (2, 96 + 2 * 68, 68),
            (2, 96 + 68, 68),
            (2, 96 + 3 * 68, 68)
        ]
    );
    // Not the experts last, or not whole slices.
    assert!(permuted(&t, &[0, 1, 2]).is_err());
    let odd = TensorInfo {
        size: 4 * 68 + 1,
        ..t.clone()
    };
    assert!(permuted(&odd, &[0, 1, 2, 3]).is_err());
}

const N_EXPERT: usize = 8;
const D: i64 = 4;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Expert `e`'s values of a `[k, rows, N_EXPERT]` tensor: `base + e` everywhere.
fn expert_values(k: usize, rows: usize, base: f32) -> Vec<f32> {
    (0..N_EXPERT)
        .flat_map(|e| std::iter::repeat_n(base + e as f32, k * rows))
        .collect()
}

/// One dense layer (0) and one MoE layer (1) of the routed tensors, each value naming its
/// expert, plus `extra` per-expert tensors of layer 1.
fn write_model(path: &std::path::Path, extra: &[&str]) {
    let mut w = GgufWriter::new();
    let n = N_EXPERT as i64;
    let add = |w: &mut GgufWriter, name: &str, shape: &[i64], v: &[f32]| {
        w.add_tensor(name, types::F32, shape, &f32_bytes(v))
            .unwrap();
    };
    add(&mut w, "blk.0.ffn_up.weight", &[D, D], &[1.0; 16]);
    add(
        &mut w,
        "blk.1.ffn_gate_inp.weight",
        &[D, n],
        &expert_values(4, 1, 100.0),
    );
    add(
        &mut w,
        "blk.1.exp_probs_b.bias",
        &[n],
        &expert_values(1, 1, 200.0),
    );
    for (k, kind) in ["up", "gate", "down"].into_iter().enumerate() {
        let base = 300.0 + 100.0 * k as f32;
        let name = format!("blk.1.ffn_{kind}_exps.weight");
        add(&mut w, &name, &[D, 2, n], &expert_values(4, 2, base));
    }
    for name in extra {
        add(&mut w, name, &[n], &expert_values(1, 1, 900.0));
    }
    w.write(path).unwrap();
}

#[test]
fn check_layer_refuses_per_expert_tensors_it_does_not_know() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.gguf");
    write_model(&path, &[]);
    let g = GgufFile::open(&path).unwrap();
    assert!(check_layer(&g, 1).is_ok());
    assert!(check_layer(&g, 0).is_ok());
    for extra in ["blk.1.ffn_up_exps.scale", "blk.1.ffn_gate_inp.bias"] {
        write_model(&path, &[extra]);
        let g = GgufFile::open(&path).unwrap();
        assert!(check_layer(&g, 1).is_err(), "{extra}");
    }
}

#[test]
fn the_loader_reads_router_and_experts_renumbered() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.gguf");
    write_model(&path, &[]);
    let g = GgufFile::open(&path).unwrap();
    let order = expert_order(ExpertSplit::Interleave, N_EXPERT, 2).unwrap();
    assert_eq!(order, vec![0, 2, 4, 6, 1, 3, 5, 7]);
    // The recipes `ParPlan::tensors` adds for layer 1 (owner GPU 0): its routed tensors and
    // the router's mirror on GPU 1.
    let mut recipes = std::collections::HashMap::new();
    let mut extra = Vec::new();
    for w in ROUTED {
        let name = format!("blk.1.{w}");
        let t = g.tensor(&name).unwrap();
        let r = permuted(&t, &order).unwrap();
        if !w.contains("_exps") {
            let m = mirror_name(&name, 1);
            extra.push(TensorInfo {
                name: m.clone(),
                ..t.clone()
            });
            recipes.insert(m, r.clone());
        }
        recipes.insert(name, r);
    }
    let opts = Options {
        extra: &extra,
        extra_recipes: Some(&recipes),
        ..Options::with_threads(2)
    };
    let cpu = device_by_kind(DeviceKind::Cpu).unwrap().buffer_type();
    let weights = Weights::load_with(&g, |_| cpu, &opts).unwrap();
    let values = |name: &str| weights.get(name).unwrap().to_f32();
    let renumbered = |k: usize, rows: usize, base: f32| -> Vec<f32> {
        order
            .iter()
            .flat_map(|&e| std::iter::repeat_n(base + e as f32, k * rows))
            .collect()
    };
    let gate = renumbered(4, 1, 100.0);
    assert_eq!(values("blk.1.ffn_gate_inp.weight"), gate);
    assert_eq!(values(&mirror_name("blk.1.ffn_gate_inp.weight", 1)), gate);
    let bias = renumbered(1, 1, 200.0);
    assert_eq!(values("blk.1.exp_probs_b.bias"), bias);
    assert_eq!(values(&mirror_name("blk.1.exp_probs_b.bias", 1)), bias);
    for (k, kind) in ["up", "gate", "down"].into_iter().enumerate() {
        let base = 300.0 + 100.0 * k as f32;
        assert_eq!(
            values(&format!("blk.1.ffn_{kind}_exps.weight")),
            renumbered(4, 2, base),
            "{kind}"
        );
    }
    // The dense layer as the file has it.
    assert_eq!(values("blk.0.ffn_up.weight"), vec![1.0; 16]);
}

#[test]
fn a_plan_split_contiguous_keeps_the_file_order() {
    let plan = ParPlan::new(4, vec![0, 1, 2, 3], 288);
    assert_eq!(plan.order, None);
    let inter = plan.clone().split_experts(ExpertSplit::Interleave);
    assert_eq!(inter.bounds, plan.bounds);
    assert_eq!(inter.order.as_ref().map(|o| o[72]), Some(1));
    assert_eq!(plan.split_experts(ExpertSplit::Contiguous).order, None);
}
