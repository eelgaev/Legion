//! The BF16 expert matmul: which weights it takes, the CPU reference against F64 sums, and on
//! CUDA (ignored test) the same values as the CPU to rounding.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};

/// Deterministic values in [-1, 1).
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

#[test]
fn takes_bf16_expert_stacks() {
    let ctx = Context::new(64, 0);
    assert!(takes(ctx.new_tensor(types::BF16, &[256, 64, 4])));
    assert!(
        !takes(ctx.new_tensor(types::BF16, &[256, 64])),
        "one expert"
    );
    assert!(
        !takes(ctx.new_tensor(types::BF16, &[12, 64, 4])),
        "rows not a multiple of 8"
    );
    assert!(!takes(ctx.new_tensor(types::F16, &[256, 64, 4])));
    assert!(!takes(ctx.new_tensor(types::F32, &[256, 64, 4])));
    let w = ctx.new_tensor(types::BF16, &[256, 64, 4]);
    assert!(!takes(ctx.transpose(w)), "rows not contiguous");
}

/// One case: `n_expert` experts of `[k, m]`, `n_used` slots of `n_tok` tokens, `x` shared by
/// the slots or not, and the slots' expert ids (`-1`: none). Returns the op's result and the
/// F64 sums of the BF16 weights.
fn case(
    backend: &mut Backend,
    (k, m, n_expert): (i64, i64, i64),
    n_used: i64,
    n_tok: i64,
    shared_x: bool,
    ids: &[i32],
) -> (Vec<f32>, Vec<f64>) {
    let ctx = Context::new(16, 16);
    let w = ctx.new_tensor(types::BF16, &[k, m, n_expert]).as_input();
    let x_n1 = if shared_x { 1 } else { n_used };
    let x = ctx.new_tensor(types::F32, &[k, x_n1, n_tok]).as_input();
    let id = ctx.new_tensor(types::I32, &[n_used, n_tok]).as_input();
    let out = mul_mat_id(&ctx, w, x, id).as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let wb: Vec<u16> = noise((k * m * n_expert) as usize, 3)
        .iter()
        .map(|&v| (v.to_bits() >> 16) as u16)
        .collect();
    let xv = noise((k * x_n1 * n_tok) as usize, 9);
    w.set_bytes(&wb.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    x.set_f32(&xv);
    id.set_i32(ids);
    backend.compute(&graph).unwrap();
    let (k, m) = (k as usize, m as usize);
    let mut want = Vec::new();
    for t in 0..n_tok as usize {
        for u in 0..n_used as usize {
            let e = ids[t * n_used as usize + u];
            let xi = t * x_n1 as usize + if shared_x { 0 } else { u };
            for r in 0..m {
                if e < 0 || e as i64 >= n_expert {
                    want.push(0.0);
                    continue;
                }
                let e = e as usize;
                want.push(
                    (0..k)
                        .map(|c| {
                            let q = f32::from_bits((wb[(e * m + r) * k + c] as u32) << 16);
                            f64::from(q) * f64::from(xv[xi * k + c])
                        })
                        .sum(),
                );
            }
        }
    }
    (out.to_f32(), want)
}

fn check(got: &[f32], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(
            (f64::from(*a) - b).abs() <= 1e-4 * (1.0 + b.abs()),
            "{what}: row {i}: {a} vs {b}"
        );
    }
}

#[test]
fn cpu_matches_f64_sums() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    // Decode: 2 slots of one token, x shared (the MoE's gate and up).
    let (got, want) = case(&mut backend, (256, 64, 4), 2, 1, true, &[3, 1]);
    check(&got, &want, "shared x");
    // The down projection: x per slot, 3 tokens, an empty slot.
    let ids = [0, 2, 1, -1, 3, 3];
    let (got, want) = case(&mut backend, (64, 256, 4), 2, 3, false, &ids);
    check(&got, &want, "x per slot");
    assert!(got[3 * 256..4 * 256].iter().all(|&v| v == 0.0), "no expert");
    // An out-of-range id is 0 too.
    let (got, _) = case(&mut backend, (128, 32, 2), 1, 1, true, &[7]);
    assert!(got.iter().all(|&v| v == 0.0));
}

/// On a GPU, the kernel gives the CPU reference's values to rounding, at GLM-5.3-Flash's MTP
/// expert shapes (gate/up 4096 -> 2048, down 2048 -> 4096; 16 of the 288 experts).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_matches_cpu() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(8);
    let ids: Vec<i32> = (0..24).map(|i| i * 7 % 16).collect();
    for (shape, shared) in [((4096, 2048, 16), true), ((2048, 4096, 16), false)] {
        let (g, want) = case(&mut gpu, shape, 8, 3, shared, &ids);
        let (c, _) = case(&mut cpu, shape, 8, 3, shared, &ids);
        check(&g, &want, "cuda");
        check(&c, &want, "cpu");
    }
}
