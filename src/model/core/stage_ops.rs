//! Which external ops a stage graph may capture (`legion serve --stage-graphs`, #817; see
//! [`crate::ggml::stage_graph`]): those whose CUDA launch enqueues only kernels on the compute
//! stream, chosen from the node alone (its sources, shape and parameters) and settings fixed
//! at startup. Each launcher listed here was read for that: no host waits, no side streams or
//! events, no host-side state that changes between steps, no allocation. An op not listed runs
//! between the captures as before, so leaving one out costs speed, never bits.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::ggml::{ExternalOp, ffi};

#[cfg(test)]
mod tests;

/// External ops whose launch is a pure function of the node (see the module doc). The expert
/// matvecs are checked per node ([`super::expert_mv::capture_safe`]).
pub(crate) const PURE: &[&str] = &[
    // qwen4 (Qwen3.8-Flash-Next): the hyper-connection mixers, per-layer embeddings, QSA.
    // The reader of a deferred reduce-scatter's partials too (#1176): its kernel reads their
    // addresses from the communicator's device slot (`crate::tp_net::seq_par::sum_slot`, fixed
    // for the communicator's life), which the exchange writes on the stream before it.
    "hc_post_norm",
    "hc_norm",
    "qwen4_hc_mix",
    "hc_up_mix",
    "hc_up_lob",
    // The prompt steps' down matmul rebuilding xn from BF16 streams (#1183): one kernel, its
    // tile chosen from the node's token count and the GPU's SM count.
    "hc_down_bf16",
    // ggml's gated hc_pre (#836) and the V100's F16-writing mix: one kernel on the compute
    // stream, picked by the node's shape, parameters and pointers' alignment.
    "hc_mix16",
    // The mixer chain's scale+SiLU, combine and grouped norm (#836): one kernel each.
    "hc_lo_act",
    "hc_combine",
    "hc_rms_mul",
    "ple_gate",
    "ple_conv",
    "qsa_pool_mask",
    "qsa_select",
    "qsa_select_all",
    "qsa_scores",
    "qsa_scores_exact",
    "qsa_topk",
    "qsa_attn",
    // The shared blocks: gated DeltaNet, the fused norms and gates, the MoE router and tail.
    "gdn",
    "conv_silu",
    "delta_prep",
    "gated_norm",
    "rms_norm_bf16",
    "gate_bf16",
    "moe_combine",
    "shexp_gate",
    "router_topk",
    "router_weights",
    // The router's matmul and selection in one kernel (#836): its completion counter is a
    // device global the kernel sets back to 0 itself, so a replay finds it as a launch does.
    // Its logits and ids nodes launch nothing.
    "router_mv",
    "router_mv_logits",
    "router_mv_ids",
    // A DeltaNet layer's BF16 alpha and beta of 1-8 tokens in one launch (#689).
    "mvf_bf16",
    "moe_tail",
    "moe_sum_gate",
    "ids_remap",
    "gate_q8",
    "shared_q8",
    // Q8_0 projections of 9-255 tokens sharing one quantization (#836): ggml's quantization (if
    // the op's node before it in the graph read another input) and ggml's MMQ on a scratch
    // buffer kept per stream. Whether it quantizes follows from the graph, which a capture
    // keeps, and the MMQ takes ggml's pool as ggml's own captured matmuls do.
    "shared_q8_small",
    // Q4_K / Q5_1 expert matmuls of 9-63 tokens with ggml's MMQ bits (#836): its scratch is a
    // buffer kept per stream (allocated at the first launch), two kernels on the compute
    // stream, picked by the shapes.
    "mmq_ids",
    // The BF16 decode matmuls and the NVFP4 matmuls (dense, expert, SwiGLU): kernels picked by
    // the shapes and by settings fixed at startup, an expert op's plan kept in its own node
    // (#836). Left out, each one cut a decode step's capture in two (~2 us a cut on a GB10).
    "dense_mv",
    // Q8_0 matvecs of one input quantized once (#819): the quantization and the matvec (with
    // or without the SwiGLU), kernels picked by the shapes.
    "q8_group",
    "q8_quantize",
    // One-column Q8_0 projections of 6,144 values on a V100 (#814): q8_head's kernel (the one
    // q8_group's row matvecs launch), its variant fixed at startup.
    "q8_head_out",
    // Q8_0 matmuls of 9-255 tokens with ggml's MMQ bits and no fixup launch (#836): two
    // kernels on the compute stream, scratch kept per stream from its first launch.
    "q8_mmq",
    "nvfp4_mm",
    "nvfp4_mm_id",
    "nvfp4_swiglu",
    "nvfp4_swiglu_id",
    // NVFP4 expert matvecs of 2-8 tokens reading each routed expert once (#836), alone or the
    // SwiGLU pair: two kernels on the compute stream, the activations' Q8_1 blocks in a
    // device-global scratch, the grid from the device's occupancy (fixed per kernel).
    "nvfp4_once",
    "nvfp4_once_glu",
    // Paged attention and the KV writes.
    "paged_attn",
    "norm_rope",
    "pool_key",
    "kv_append",
    "idx_append",
    // Selections on thread-block clusters (compute capability 9.0+): the QSA top-k and the
    // heads' greedy argmax, a kernel picked by the row length (#827).
    "cluster_topk",
    "cluster_argmax",
];

/// Pure ops captured in small steps only ([`crate::ggml::stage_graph::small_step`]: decode,
/// verify and draft passes): the W4A4 NVFP4 ops (`super::super::nvfp4_a4`, #1202). Each one
/// quantizes the activations into the node's own scratch, plans the experts on the device (as
/// `nvfp4_mm_id`), then runs the GEMM, matvec or combine, the kernels and tiles picked by the
/// shapes, the weights' alignment and settings fixed at startup; the TMA tile's tensor maps
/// are encoded on the host from the node's pointers and passed as kernel parameters. They take
/// batches over 64 tokens under `--nvfp4-a4 prefill` (a 16-user verify step) and from 9 under
/// `experts-decode`; left out, each one cut such a step's capture in two. A prompt step keeps
/// them between its captures: its graph seldom runs twice, and a capture of it whole would
/// hold its first kernel back until the whole step is recorded. Their launchers must stay
/// free of host waits and allocations (see `nvfp4_a4.cu`).
pub(crate) const SMALL: &[&str] = &[
    "nvfp4_a4_mm",
    "nvfp4_a4_mm_multi",
    "nvfp4_a4_mm_id",
    "nvfp4_a4_mm_id_weighted",
    "nvfp4_a4_swiglu",
    "nvfp4_a4_swiglu_id",
    "nvfp4_a4_quant",
];

/// Dense matmuls captured in small steps only, as [`SMALL`] (#689): `nvfp4_mvq` (5-8 columns
/// on compute capability 12.x: ggml's quantize_q8_1 and the NVFP4 matvec, two kernels picked
/// by the column count, the ring by the shapes and the device's attributes, read once) and
/// `fp8_mm` (the FP8 W8A16 matmuls: one kernel picked by the shapes, the node's parameters and
/// the device's compute capability; its shared memory opt-in is set once, at the first launch,
/// which a step runs directly before it is captured). Neither allocates, waits or touches
/// another stream. Left out, each one cut a verify step's capture: on Qwen3.8-27B NVFP4 with
/// DFlash on a DGX Spark (GB10) a verify step ran as ~380 captures of 1-7 kernels around
/// them, a 2-3 µs gap at each cut. Prompt steps keep them between their captures, as before.
pub(crate) const SMALL_MM: &[&str] = &["nvfp4_mvq", "fp8_mm"];

/// Pure ops captured only while [`set_commit_capture`] is on: a verify commit's conv window
/// copy (`legion_spec_window_launch`, one kernel picked by the node's shape). Left out, it runs
/// between captures, and since a commit graph alternates it with single `gdn` launches, no
/// piece of the commit is captured at all.
pub(crate) const COMMIT: &[&str] = &["spec_window"];

/// See [`set_commit_capture`].
static COMMIT_CAPTURE: AtomicBool = AtomicBool::new(false);

/// Whether stage graphs from now on may capture a verify commit whole (#814): each stage's
/// commit graph (18-19 kernels on Qwen3.8-Flash-Next over 4 V100s, ~180 µs of host launches)
/// becomes one capture, its kept counts a device input, so one capture serves every commit.
/// The same kernels with the same arguments on the same stream, so the same bits. Off by
/// default (`legion serve --commit-capture` turns it on): flat on 4x V100 so far.
pub fn set_commit_capture(on: bool) {
    COMMIT_CAPTURE.store(on, Ordering::Relaxed);
}

/// Whether a stage graph may capture `node`'s launch of `op` ([`crate::ggml::stage_graph::SafeOp`]).
pub fn capture_safe(op: &ExternalOp, node: &ffi::ggml_tensor) -> bool {
    match op.name {
        // SAFETY: the node of an expert matvec op, which only its builders make.
        "expert_mv" | "expert_mv_glu" => unsafe { super::expert_mv::capture_safe(node) },
        name if COMMIT.contains(&name) => COMMIT_CAPTURE.load(Ordering::Relaxed),
        name => listed(name, crate::ggml::stage_graph::small_step()),
    }
}

/// Whether a capture may hold an op `name` of [`PURE`], [`SMALL`] or [`SMALL_MM`], in a small
/// step (`small`) or a prompt step.
fn listed(name: &str, small: bool) -> bool {
    if SMALL.contains(&name) || SMALL_MM.contains(&name) {
        small
    } else {
        PURE.contains(&name)
    }
}
