use super::*;

#[test]
fn every_kernel_of_this_host_matches_scalar() {
    let cases = check().expect("bit-exact");
    let simd = supported_isas().len() - 1;
    assert_eq!(cases > 0, simd > 0, "{cases} cases for {simd} SIMD ISAs");
}

#[test]
fn random_experts_have_the_shape_bytes() {
    let shape = ExpertShape {
        n_embd: 512,
        n_ff: 64,
        up_gate: QuantFormat::Q4K,
        down: QuantFormat::Q8_0,
    };
    let [up, gate, down] = random_experts(shape, 6);
    assert_eq!(up.len(), 6 * shape.gate_bytes());
    assert_eq!(gate.len(), 6 * shape.gate_bytes());
    assert_eq!(down.len(), 6 * shape.down_bytes());
    assert_eq!(shape.bytes(), 2 * 64 * 2 * 144 + 512 * 2 * 34);
    // Four distinct experts, repeated.
    let g = shape.gate_bytes();
    assert_eq!(gate[..g], gate[4 * g..5 * g]);
    assert_ne!(gate[..g], gate[g..2 * g]);
}

#[test]
fn qwen_expert_is_three_megabytes() {
    let s = ExpertShape::QWEN_FLASH_NEXT;
    assert_eq!(s.gate_bytes(), 640 * 10 * 144);
    assert_eq!(s.down_bytes(), 2560 * 20 * 24);
    assert_eq!(s.bytes(), 3_072_000);
}
