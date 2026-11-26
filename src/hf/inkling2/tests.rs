//! The Inkling-2 checkpoint's reshaped tensors on a fake checkpoint (tiny dims, the release's
//! dtypes): the converter's transforms (llama.cpp's `Inkling2Model.modify_tensors`, the
//! loop model's message queries, the expert de-interleave) byte for byte.

use std::path::Path;
use std::sync::Arc;

use serde_json::json;

use super::*;
use crate::gguf::Recipe;
use crate::safetensors::Checkpoint;

/// A one-file safetensors checkpoint of `(name, dtype, shape, bytes)`.
fn write_checkpoint(dir: &Path, tensors: &[(String, &str, Vec<usize>, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        header.insert(
            name.clone(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [data.len(), data.len() + bytes.len()]}),
        );
        data.extend_from_slice(bytes);
    }
    let mut h = serde_json::to_vec(&header).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(data);
    std::fs::write(dir.join("model.safetensors"), out).unwrap();
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// `n` values `base + i` (exact in BF16 for small `n`), as BF16 bytes.
fn bf16s(n: usize, base: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|i| (((base + i as f32).to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn computed(b: &Builder, name: &str) -> Vec<u8> {
    match b.recipes.get(name) {
        Some(Recipe::Computed(f)) => f().unwrap(),
        r => panic!("{name}: {r:?}"),
    }
}

fn shape_of(b: &Builder, name: &str) -> (crate::ggml::ffi::ggml_type, Vec<u64>) {
    let t = b.tensors.iter().find(|t| t.name == name).unwrap();
    (t.ggml_type, t.shape.clone())
}

#[test]
fn w13_rows_are_taken_gate_first() {
    let rows = 6;
    let order: Vec<usize> = (0..rows).map(|j| src_row(j, rows, true)).collect();
    assert_eq!(order, [0, 2, 4, 1, 3, 5]);
    assert_eq!(
        (0..rows)
            .map(|j| src_row(j, rows, false))
            .collect::<Vec<_>>(),
        [0, 1, 2, 3, 4, 5]
    );
}

const K: usize = 4;
const C: usize = 3;
const NR: usize = 2;
const EXT: usize = 5;
const E: usize = 4;
const CORE: usize = 2;

#[test]
fn convs_relative_bias_and_message_queries_follow_the_converter() {
    let dir = tempfile::tempdir().unwrap();
    let q: Vec<f32> = (0..2 * CORE * E).map(|i| i as f32).collect();
    let kv: Vec<f32> = (0..2 * CORE * E).map(|i| 100.0 + i as f32).collect();
    let t = vec![
        ("sw".to_string(), "BF16", vec![K, C], bf16s(K * C, 1.0)),
        ("sc".to_string(), "BF16", vec![C, 1, K], bf16s(C * K, 1.0)),
        (
            "rel".to_string(),
            "BF16",
            vec![NR, EXT],
            bf16s(NR * EXT, 1.0),
        ),
        (
            format!("{PREFIX}middle_cycle_cross_pass_queries"),
            "F32",
            vec![2 * CORE, E],
            f32s(&q),
        ),
        (
            format!("{PREFIX}middle_cycle_cross_pass_kv_queries"),
            "F32",
            vec![2 * CORE, E],
            f32s(&kv),
        ),
    ];
    write_checkpoint(dir.path(), &t);
    let mut b = Builder::new(Arc::new(Checkpoint::open(dir.path()).unwrap()));

    // `static_w` [K, C] newest tap first -> ggml [K, C] oldest first: `flip(0).T`.
    static_taps(&mut b, "static", "sw", K as u64).unwrap();
    assert_eq!(shape_of(&b, "static").1, [K as u64, C as u64]);
    let got = floats(&computed(&b, "static"));
    for ch in 0..C {
        for j in 0..K {
            // source value at [K - 1 - j, ch] is 1 + (K - 1 - j) * C + ch
            assert_eq!(got[ch * K + j], (1 + (K - 1 - j) * C + ch) as f32);
        }
    }
    // A depthwise conv's taps [C, 1, K] keep their order.
    conv_taps(&mut b, "conv", "sc", K as u64).unwrap();
    assert_eq!(shape_of(&b, "conv"), (types::F32, vec![K as u64, C as u64]));
    let got = floats(&computed(&b, "conv"));
    assert_eq!(got, (0..C * K).map(|i| 1.0 + i as f32).collect::<Vec<_>>());
    // The relative-bias projection [NR, EXT] transposed: ggml [NR, EXT], NR innermost.
    rel_table(&mut b, "relb", "rel").unwrap();
    assert_eq!(shape_of(&b, "relb").1, [NR as u64, EXT as u64]);
    let got = floats(&computed(&b, "relb"));
    for r in 0..NR {
        for d in 0..EXT {
            assert_eq!(got[d * NR + r], (1 + r * EXT + d) as f32);
        }
    }
    // Core block i: [q 2i, kv 2i, kv 2i + 1, q 2i + 1].
    for i in 0..CORE {
        let name = format!("msg{i}");
        msg_query(&mut b, &name, i, E).unwrap();
        assert_eq!(shape_of(&b, &name), (types::F32, vec![E as u64, 4]));
        let got = floats(&computed(&b, &name));
        let row = |v: &[f32], r: usize| v[r * E..][..E].to_vec();
        let want = [
            row(&q, 2 * i),
            row(&kv, 2 * i),
            row(&kv, 2 * i + 1),
            row(&q, 2 * i + 1),
        ]
        .concat();
        assert_eq!(got, want);
    }
}

const NE: usize = 2;
const ROWS: usize = 4;
const KK: usize = 64;

#[test]
fn nvfp4_experts_repack_per_row_with_their_expert_scale() {
    let dir = tempfile::tempdir().unwrap();
    // Every row its own codes and scales, so a row out of place shows.
    let w: Vec<u8> = (0..NE * ROWS * KK / 2)
        .map(|i| (i * 7 % 256) as u8)
        .collect();
    let s: Vec<u8> = (0..NE * ROWS * KK / 16)
        .map(|i| (0x30 + i % 40) as u8)
        .collect();
    let l = "model.llm.layers.0.mlp.experts.";
    let t = vec![
        (
            format!("{l}w13_weight"),
            "U8",
            vec![NE, ROWS, KK / 2],
            w.clone(),
        ),
        (
            format!("{l}w13_weight_scale"),
            "F8_E4M3",
            vec![NE, ROWS, KK / 16],
            s.clone(),
        ),
        (
            format!("{l}w13_weight_amax"),
            "F32",
            vec![NE],
            f32s(&[3.0, 12.0]),
        ),
        (
            format!("{l}w13_weight_outer_scale_denominator"),
            "F32",
            vec![1],
            f32s(&[6.0]),
        ),
    ];
    write_checkpoint(dir.path(), &t);
    let mut b = Builder::new(Arc::new(Checkpoint::open(dir.path()).unwrap()));
    let name = "blk.0.ffn_gate_up_exps.weight";
    experts(&mut b, name, &format!("{l}w13_weight"), NE, ROWS, KK).unwrap();
    assert_eq!(
        shape_of(&b, name),
        (types::NVFP4, vec![KK as u64, ROWS as u64, NE as u64])
    );
    let got = computed(&b, name);
    let row_out = KK / 64 * 36;
    for e in 0..NE {
        for j in 0..ROWS {
            let r = e * ROWS + src_row(j, ROWS, true);
            let want = super::super::nvfp4::repack_nvfp4(
                &w[r * KK / 2..][..KK / 2],
                &s[r * KK / 16..][..KK / 16],
                1,
                KK,
            );
            assert_eq!(
                &got[(e * ROWS + j) * row_out..][..row_out],
                &want[..],
                "{e} {j}"
            );
        }
    }
    let scale = "blk.0.ffn_gate_up_exps.scale";
    assert_eq!(shape_of(&b, scale), (types::F32, vec![NE as u64]));
    assert_eq!(floats(&computed(&b, scale)), [0.5, 2.0]);
}

#[test]
fn mxfp8_experts_keep_their_bytes_and_power_of_two_scales() {
    let dir = tempfile::tempdir().unwrap();
    let w: Vec<u8> = (0..NE * ROWS * KK).map(|i| (i % 251) as u8).collect();
    let s: Vec<u8> = (0..NE * ROWS * KK / 32).map(|i| (120 + i) as u8).collect();
    let l = "model.llm.layers.20.mlp.experts.";
    let t = vec![
        (
            format!("{l}w13_weight"),
            "F8_E4M3",
            vec![NE, ROWS, KK],
            w.clone(),
        ),
        (
            format!("{l}w13_weight_scale"),
            "U8",
            vec![NE, ROWS, KK / 32],
            s.clone(),
        ),
        (
            format!("{l}w2_weight"),
            "F8_E4M3",
            vec![NE, ROWS, KK],
            w.clone(),
        ),
        (
            format!("{l}w2_weight_scale"),
            "U8",
            vec![NE, ROWS, KK / 32],
            s.clone(),
        ),
    ];
    write_checkpoint(dir.path(), &t);
    let mut b = Builder::new(Arc::new(Checkpoint::open(dir.path()).unwrap()));
    for (which, gate_up) in [("w13", true), ("w2", false)] {
        let name = format!("blk.20.{which}.weight");
        experts(&mut b, &name, &format!("{l}{which}_weight"), NE, ROWS, KK).unwrap();
        assert_eq!(
            shape_of(&b, &name),
            (types::I8, vec![KK as u64, ROWS as u64, NE as u64])
        );
        let got = computed(&b, &name);
        let sc = floats(&computed(&b, &format!("blk.20.{which}.scale")));
        assert_eq!(
            shape_of(&b, &format!("blk.20.{which}.scale")),
            (types::F32, vec![KK as u64 / 32, ROWS as u64, NE as u64])
        );
        for e in 0..NE {
            for j in 0..ROWS {
                let r = e * ROWS + src_row(j, ROWS, gate_up);
                let o = e * ROWS + j;
                assert_eq!(&got[o * KK..][..KK], &w[r * KK..][..KK]);
                for blk in 0..KK / 32 {
                    let e8 = s[r * KK / 32 + blk] as i32;
                    assert_eq!(sc[o * KK / 32 + blk], 2f32.powi(e8 - 127));
                }
            }
        }
    }
}

/// Malformed taps and outer scales are errors, not panics or wrong scales.
#[test]
fn bad_conv_taps_and_outer_scales_are_refused() {
    let l = "model.llm.layers.0.mlp.experts.";
    let fixture = |amax_dtype: &str, denoms: &[f32]| {
        let dir = tempfile::tempdir().unwrap();
        let t = vec![
            ("scalar".to_string(), "BF16", vec![], bf16s(1, 1.0)),
            (
                format!("{l}w13_weight"),
                "U8",
                vec![NE, ROWS, KK / 2],
                vec![0u8; NE * ROWS * KK / 2],
            ),
            (
                format!("{l}w13_weight_scale"),
                "F8_E4M3",
                vec![NE, ROWS, KK / 16],
                vec![0x38u8; NE * ROWS * KK / 16],
            ),
            (
                format!("{l}w13_weight_amax"),
                amax_dtype,
                vec![NE],
                f32s(&[3.0, 12.0]),
            ),
            (
                format!("{l}w13_weight_outer_scale_denominator"),
                "F32",
                vec![denoms.len()],
                f32s(denoms),
            ),
        ];
        write_checkpoint(dir.path(), &t);
        let b = Builder::new(Arc::new(Checkpoint::open(dir.path()).unwrap()));
        (dir, b)
    };
    let name = "blk.0.ffn_gate_up_exps.weight";
    let scale = "blk.0.ffn_gate_up_exps.scale";

    // A scalar named as conv taps.
    let (_dir, mut b) = fixture("F32", &[6.0]);
    let e = conv_taps(&mut b, "conv", "scalar", 4).unwrap_err();
    assert!(e.contains("conv taps"), "{e}");
    // A broadcast denominator (all one value) is fine.
    let (_dir, mut b) = fixture("F32", &[6.0, 6.0]);
    experts(&mut b, name, &format!("{l}w13_weight"), NE, ROWS, KK).unwrap();
    assert_eq!(floats(&computed(&b, scale)), [0.5, 2.0]);
    // Differing denominators.
    let (_dir, mut b) = fixture("F32", &[6.0, 7.0]);
    experts(&mut b, name, &format!("{l}w13_weight"), NE, ROWS, KK).unwrap();
    let e = match b.recipes.get(scale) {
        Some(Recipe::Computed(f)) => f().unwrap_err(),
        r => panic!("{scale}: {r:?}"),
    };
    assert!(e.contains("differing denominators"), "{e}");
    // An amax of another 4-byte dtype.
    let (_dir, mut b) = fixture("I32", &[6.0]);
    let e = experts(&mut b, name, &format!("{l}w13_weight"), NE, ROWS, KK).unwrap_err();
    assert!(e.contains("want F32"), "{e}");
}
