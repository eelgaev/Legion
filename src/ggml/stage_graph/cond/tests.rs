use std::mem::{offset_of, size_of};

use super::*;

#[test]
fn node_params_have_cudas_layout() {
    // cudaGraphNodeParams: type, reserved0[3], a 29-word union, reserved2.
    assert_eq!(size_of::<GraphNodeParams>(), 256);
    assert_eq!(offset_of!(GraphNodeParams, cond), 16);
    assert_eq!(offset_of!(GraphNodeParams, reserved2), 248);
    // cudaConditionalNodeParams: handle, type, size, phGraph_out.
    assert_eq!(offset_of!(ConditionalParams, ty), 8);
    assert_eq!(offset_of!(ConditionalParams, size), 12);
    assert_eq!(offset_of!(ConditionalParams, graphs_out), 16);
    // cudaKernelNodeParams: func, gridDim, blockDim, sharedMemBytes, kernelParams, extra.
    assert_eq!(offset_of!(KernelNodeParams, grid), 8);
    assert_eq!(offset_of!(KernelNodeParams, block), 20);
    assert_eq!(offset_of!(KernelNodeParams, shared), 32);
    assert_eq!(offset_of!(KernelNodeParams, params), 40);
    assert_eq!(size_of::<KernelNodeParams>(), 56);
}

#[test]
fn arg_blocks_match_the_kernels() {
    // legion_mtp_decide (src/cuda/mtp_cond.cu).
    assert_eq!(offset_of!(DecideArgs, alive), 8);
    assert_eq!(offset_of!(DecideArgs, p_min), 16);
    assert_eq!(offset_of!(DecideArgs, rows), 20);
    assert_eq!(offset_of!(DecideArgs, reset), 24);
    assert_eq!(size_of::<DecideArgs>(), 32);
    // legion_mtp_cond.
    assert_eq!(offset_of!(CondArgs, alive), 8);
    assert_eq!(size_of::<CondArgs>(), 16);
}

#[test]
fn scope_sets_and_restores_the_mark() {
    assert!(!current());
    {
        let _outer = CondScope::enter(true);
        assert!(current());
        {
            let _inner = CondScope::enter(false);
            assert!(!current());
        }
        assert!(current());
    }
    assert!(!current());
}
