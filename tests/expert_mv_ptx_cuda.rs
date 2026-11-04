//! The decode expert matvec's PTX kernels (`--expert-mv-ptx`, #703) on a GPU: the nvcc tile
//! kernel's bits for Qwen3.8-Flash-Next's gate/up Q4_K and down Q5_1 experts, with all or half
//! of the experts in cache slots, at 1-4 and 8 tokens a step (on the compute stream and on its
//! side stream), and close to a plain f32 reference.
//!
//! `cargo test --release --features cuda --test expert_mv_ptx_cuda -- --ignored --nocapture`
//! (`legion expert-mv-ptx-bench` runs the same from a release binary, with timings).

#![cfg(feature = "cuda")]

use legion::ggml::backend_by_name;
use legion::model::core::expert_mv::ptx::bench::{self, Config};

#[test]
#[ignore = "needs a CUDA GPU (sm_70 or newer)"]
fn ptx_kernels_give_the_tile_kernels_bits() {
    let mut gpu = backend_by_name("CUDA0").unwrap();
    let cfg = Config {
        n_expert: 32,
        layers: 2,
        sets: 2,
        rounds: 1,
        // 8: experts with more users than a round of the `_users` kernels takes.
        toks: vec![1, 2, 3, 4, 8],
        ..Config::default()
    };
    for r in bench::run(&mut gpu, &cfg).unwrap() {
        assert_eq!(r.differ, [0, 0, 0, 0, 0], "{r:?}");
        // Q8_1 activations: about 1% of the largest value.
        assert!(r.ref_err < 0.05, "{r:?}");
    }
}
