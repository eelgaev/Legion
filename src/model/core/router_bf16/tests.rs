//! The routers' BF16 copies: the exactness check, which routers get one, and the lookup by the
//! router they were made from.

use super::*;
use crate::ggml::Backend;

fn le(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn bf16_copy_takes_exact_values_only() {
    // ~2.99e38 with BF16's mantissa (3.0e38 itself has low bits set: 0x7f61_b1e6).
    let big = f32::from_bits(0x7f61_0000);
    let exact = [0.0f32, -0.0, 1.0, -2.5, 0.046875, f32::INFINITY, big];
    let copy = bf16_copy(&le(&exact)).expect("exact in BF16");
    assert_eq!(copy.len(), exact.len() * 2);
    for (v, b) in exact.iter().zip(copy.as_chunks::<2>().0) {
        let back = f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16);
        assert_eq!(back.to_bits(), v.to_bits(), "{v}");
    }
    // One value with a low bit set: no copy.
    let mut vals = exact.to_vec();
    vals.push(f32::from_bits(1.0f32.to_bits() | 1));
    assert!(bf16_copy(&le(&vals)).is_none());
    assert!(bf16_copy(&le(&[0.1])).is_none());
    // Not whole values.
    assert!(bf16_copy(&[0, 0, 0x80]).is_none());
    assert_eq!(bf16_copy(&[]), Some(Vec::new()));
}

#[test]
fn candidates_are_plain_f32_matrices_in_gpu_memory() {
    let ctx = Context::new(8, 0);
    assert!(candidate(ctx.new_tensor(types::F32, &[2560, 512])));
    assert!(!candidate(ctx.new_tensor(types::BF16, &[2560, 512])));
    assert!(!candidate(ctx.new_tensor(types::F32, &[2560, 512, 2])));
    let host = Context::new(8, 0);
    let w = host.new_tensor(types::F32, &[64, 32]);
    host.alloc_tensors(Backend::cpu().unwrap().buffer_type())
        .unwrap();
    assert!(!candidate(w), "host memory: no op reads a copy there");
}

#[test]
fn switch_defaults_on() {
    assert!(on());
    set_router_bf16(false);
    assert!(!on());
    set_router_bf16(true);
}
