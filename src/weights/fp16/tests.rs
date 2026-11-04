//! Which tensors the FP16 load mode picks, and its conversion with the range guard.

use super::*;

fn info(name: &str, ty: ffi::ggml_type, shape: &[u64]) -> TensorInfo {
    TensorInfo {
        name: name.into(),
        ggml_type: ty,
        shape: shape.to_vec(),
        file_index: 0,
        offset: 0,
        size: 0,
    }
}

#[test]
fn picks_bf16_and_with_gates_the_f32_gates() {
    let off = Fp16Load::default();
    let load = Fp16Load {
        mode: WeightsFp16::Load,
        gates: false,
        hc_inject_f32: false,
    };
    let gates = Fp16Load {
        gates: true,
        ..load
    };
    let bf16 = info("blk.3.indexer.q_proj.weight", types::BF16, &[2560, 512]);
    let router = info("blk.3.ffn_gate_inp.weight", types::F32, &[2560, 512]);
    let shexp = info("blk.0.ffn_gate_inp_shexp.weight", types::F32, &[2560]);
    let alpha = info("blk.0.ssm_alpha.weight", types::F32, &[2560, 48]);
    let inject = info("blk.0.hc_attn_inject.weight", types::F32, &[10240, 4]);
    let norm = info("blk.0.attn_norm.weight", types::F32, &[2560]);
    let q8 = info("blk.0.attn_qkv.weight", types::Q8_0, &[2560, 10240]);
    for t in [&bf16, &router, &shexp, &alpha, &inject, &norm, &q8] {
        assert!(!off.wants(t), "{}", t.name);
    }
    assert!(load.wants(&bf16));
    for t in [&router, &shexp, &alpha, &inject, &norm, &q8] {
        assert!(!load.wants(t), "{}", t.name);
    }
    for t in [&bf16, &router, &shexp, &alpha] {
        assert!(gates.wants(t), "{}", t.name);
    }
    for t in [&inject, &norm, &q8] {
        assert!(!gates.wants(t), "{}", t.name);
    }
}

#[test]
fn converts_in_range_and_keeps_the_rest() {
    let v = [0.0f32, -1.5, 3.140625, 65504.0, -65504.0, 1e-7];
    let h = to_f16(&v).expect("in range");
    let back: Vec<f32> = h
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| {
            let x = u16::from_ne_bytes(*b);
            // SAFETY: a plain value conversion.
            unsafe { ffi::ggml_fp16_to_fp32(x) }
        })
        .collect();
    assert_eq!(&back[..5], &v[..5]);
    // a subnormal: within half of its 2^-24 step
    assert!(
        (back[5] - 1e-7).abs() <= 2.0f32.powi(-25),
        "subnormal kept: {}",
        back[5]
    );
    assert!(to_f16(&[1.0, 70000.0]).is_none());
    assert!(to_f16(&[f32::NAN]).is_none());
    // BF16 values with 8 mantissa bits are exact in F16's range.
    let bf: Vec<u8> = [0x3FC0u16, 0xC2F7]
        .iter()
        .flat_map(|x| x.to_ne_bytes())
        .collect();
    assert_eq!(as_f32(types::BF16, &bf), vec![1.5, -123.5]);
}

#[test]
fn hc_inject_f32_picks_only_bf16_inject_weights() {
    let inject = Fp16Load {
        hc_inject_f32: true,
        ..Fp16Load::default()
    };
    let attn = info("blk.3.hc_attn_inject.weight", types::BF16, &[10240, 4]);
    let ffn = info("blk.3.hc_ffn_inject.weight", types::BF16, &[10240, 4]);
    let f32_inject = info("blk.3.hc_attn_inject.weight", types::F32, &[10240, 4]);
    let other = info("blk.3.hc_attn_down.weight", types::BF16, &[10240, 320]);
    assert_eq!(inject.target(&attn), Some(types::F32));
    assert_eq!(inject.target(&ffn), Some(types::F32));
    assert_eq!(inject.target(&f32_inject), None);
    // Off mode: nothing else.
    assert_eq!(inject.target(&other), None);
    assert_eq!(Fp16Load::default().target(&attn), None);
    // With the FP16 load mode: inject to F32, the other BF16 tensors to F16.
    let both = Fp16Load {
        mode: WeightsFp16::Load,
        ..inject
    };
    assert_eq!(both.target(&attn), Some(types::F32));
    assert_eq!(both.target(&other), Some(types::F16));
}

#[test]
fn f16_range_and_inject_names() {
    assert!(fits_f16(&[0.0, -1.5, 65504.0, -65504.0, 1e-9]));
    assert!(!fits_f16(&[1.0, 65536.0]));
    assert!(!fits_f16(&[f32::NAN]));
    assert!(!fits_f16(&[f32::NEG_INFINITY]));
    assert!(is_hc_inject("blk.48.hc_attn_inject.weight"));
    assert!(is_hc_inject("blk.0.hc_ffn_inject.weight"));
    assert!(!is_hc_inject("blk.48.hc_attn_down.weight"));
    assert!(!is_hc_inject("blk.48.ffn_gate_inp.weight"));
}
