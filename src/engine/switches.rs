//! Kernel and load switches of `legion serve` / `legion run` (#421): opt-in kernels, paths kept
//! for comparisons and loader tuning that the model code reads from process-wide state while it
//! builds graphs. They used to be `LEGION_*` environment variables read deep in that code; the
//! flags now fill a [`Switches`], and [`LmEngine::load`](super::lm::LmEngine::load) applies it
//! once, before the model loads.
//!
//! Every field is an `Option`: `None` leaves the built-in default (or what a test set through
//! the module's own setter) alone, so a config without flags changes nothing.

use crate::model::moe_ffn::PairOut;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Switches {
    /// Gated DeltaNet layers' output norm and gate as one op ([`crate::model::core::set_gated_norm`]).
    pub gated_norm: Option<bool>,
    /// MoE layers' weighted expert sum as one op ([`crate::model::core::set_moe_combine`]).
    pub moe_combine: Option<bool>,
    /// ggml-cuda's MoE fusions kept from depending on the memory layout
    /// ([`crate::model::core::set_stable_fusions`]).
    pub stable_fusions: Option<bool>,
    /// Gated DeltaNet layers' prep op ([`crate::model::core::set_delta_prep`]).
    pub delta_prep: Option<bool>,
    /// Gated DeltaNet prompt steps' recurrence on Legion's kernel
    /// ([`crate::model::core::set_gdn_scan`]).
    pub gdn_scan: Option<bool>,
    /// Gated DeltaNet prompt steps whose head counts ggml-cuda keeps on its per-token kernel (a
    /// tensor parallel rank's share) on its chunked kernel ([`crate::model::core::set_gdn_mma`]).
    pub gdn_mma: Option<bool>,
    /// Gated DeltaNet verify and commit steps' delta rule as Legion's op
    /// ([`crate::model::core::set_gdn`]).
    pub gdn: Option<bool>,
    /// Gated DeltaNet one-token steps' delta rule as Legion's op
    /// ([`crate::model::core::set_gdn_decode`]).
    pub gdn_decode: Option<bool>,
    /// Gated DeltaNet replaying verify steps' inputs staged through shared memory
    /// ([`crate::model::core::set_gdn_replay_staged`]).
    pub gdn_replay_staged: Option<bool>,
    /// Gated DeltaNet replaying verify steps skipping their padding rows
    /// ([`crate::model::core::set_gdn_pad_skip`]).
    pub gdn_pad_skip: Option<bool>,
    /// Gated DeltaNet replay commits as one graph for every slot
    /// ([`crate::model::core::set_commit_slots`]).
    pub gdn_commit_slots: Option<bool>,
    /// Gated DeltaNet commits deferred into the next verify step
    /// ([`crate::model::core::set_commit_in_verify`]).
    pub gdn_commit_in_verify: Option<bool>,
    /// Gated DeltaNet verify steps that may leave their states where the packed rows have room
    /// ([`crate::model::core::set_state_defer`]).
    pub gdn_state_defer: Option<bool>,
    /// The packed rows such pools add per slot
    /// ([`crate::model::core::set_state_defer_rows`]).
    pub gdn_state_defer_rows: Option<u8>,
    /// The fewest sequences a verify step of such pools leaves the states in
    /// ([`crate::model::core::set_state_defer_min_seqs`]).
    pub gdn_state_defer_min_seqs: Option<u8>,
    /// Attention layers' fused norm + RoPE op ([`crate::model::core::set_norm_rope`]).
    pub norm_rope: Option<bool>,
    /// Attention layers' K/V and indexer cache appends and pool-key op
    /// ([`crate::model::core::set_kv_append`]).
    pub kv_append: Option<bool>,
    /// Gemma 4's layers sharing one positions view, so ggml-cuda fuses the heads' norm and RoPE
    /// ([`crate::model::gemma4::set_head_rope_fused`]).
    pub head_rope_fused: Option<bool>,
    /// Small steps read weight pairs that lie back to back as one matmul
    /// ([`crate::model::core::set_joined_matmuls`]).
    pub joined_matmuls: Option<bool>,
    /// Fewest tokens of a prompt step that reads BF16 weight pairs as one GEMM; 0: none
    /// ([`crate::model::core::set_joined_prompt_min_tokens`]).
    pub joined_prompt_min_tokens: Option<u32>,
    /// Those prompt steps read a DeltaNet layer's z with alpha and beta as one GEMM
    /// ([`crate::model::core::set_joined_prompt_z`]).
    pub joined_prompt_z: Option<bool>,
    /// Those prompt steps read a full-attention layer's query + gate, key and value as one GEMM
    /// ([`crate::model::core::set_joined_prompt_qkv`]).
    pub joined_prompt_qkv: Option<bool>,
    /// BF16 matmuls of decode steps on Legion's dense matvec op
    /// ([`crate::model::dense_mv::set_dense_mv`]).
    pub dense_mv: Option<bool>,
    /// A Q8_0 MTP draft LM head of 9-16 sequences on Legion's Q8_0 matvec op
    /// ([`crate::model::core::set_q8_mv`]).
    pub q8_mv: Option<bool>,
    /// A one-column Q8_0 MTP draft LM head on Legion's one-column Q8_0 matvec op
    /// ([`crate::model::core::set_q8_head`]).
    pub q8_head: Option<bool>,
    /// The MTP draft pick (argmax and its probability) from that op's partials
    /// ([`crate::model::core::q8_head::pick::set_pick`]).
    pub q8_head_pick: Option<bool>,
    /// q8_group's one-column matvecs on the draft head's kernel
    /// ([`crate::model::core::q8_head::set_row_mv`]).
    pub q8_row_mv: Option<bool>,
    /// A Q4_K MTP draft LM head of 1-8 columns on a GB10 on Legion's Q4_K head matvec op
    /// ([`crate::model::core::set_q4k_head`]).
    pub q4k_head: Option<bool>,
    /// One-column Q8_0 projections of 6,144 values on the draft head's kernel
    /// ([`crate::model::core::q8_head::set_out_mv`]).
    pub q8_out_mv: Option<bool>,
    /// The MTP draft pick from that op's partials
    /// ([`crate::model::core::q4k_head::set_pick`]).
    pub q4k_head_pick: Option<bool>,
    /// That pick at 2-7 columns ([`crate::model::core::q4k_head::set_pick_cols`]).
    pub q4k_head_pick_cols: Option<bool>,
    /// Q8_0 matvecs of one input on Legion's op that quantizes it once
    /// ([`crate::model::core::set_q8_group`]).
    pub q8_group: Option<bool>,
    /// Which parts of `q8_group` run ([`crate::model::core::q8_group::set_parts`]).
    pub q8_group_parts: Option<u8>,
    /// The small BF16 matmuls that the dense matvec op reads faster than ggml, on the op
    /// ([`crate::model::dense_mv::set_small`]); `None` follows the stage graphs.
    pub dense_mv_small: Option<bool>,
    /// Decode-sized top-k and argmax on thread-block clusters
    /// ([`crate::model::cluster_select::set_cluster_select`]).
    pub cluster_select: Option<bool>,
    /// F32 matmuls of 4-16 tokens on Legion's dense matvec op
    /// ([`crate::model::dense_mv::set_f32`]).
    pub dense_mv_f32: Option<bool>,
    /// A GDN layer's alpha and beta of 9-16 tokens in its `qkv` and `z` dense op, in cuBLAS's
    /// order ([`crate::model::dense_mv::set_chain`]).
    pub dense_mv_ab: Option<bool>,
    /// Dense matvec launches of few row tiles on thread-block clusters
    /// ([`crate::model::dense_mv::set_split`]; off by default).
    pub dense_mv_split: Option<bool>,
    /// Dense BF16 matmuls of 17-48 tokens in cuBLAS's wmma order: all of them, or none
    /// ([`crate::model::dense_mv::set_wide_chain`]); `None` keeps the shapes of cuBLAS's bits
    /// ([`crate::model::dense_mv::wide_chain_bitwise`]).
    pub dense_mv_wide_chain: Option<bool>,
    /// MoE routers of 1-3 tokens as one op ([`crate::model::core::set_router_mv`]).
    pub router_mv: Option<bool>,
    /// The router's top-k and weights as one CUDA launch
    /// ([`crate::model::core::set_router_join`]).
    pub router_join: Option<bool>,
    /// MoE routers read from their BF16 copies where exact
    /// ([`crate::model::core::set_router_bf16`]).
    pub router_bf16: Option<bool>,
    /// Late n-gram rows' reads issued by the step's fill
    /// ([`crate::model::qwen4::set_ple_early_reads`]).
    pub ple_early_reads: Option<bool>,
    /// NVFP4 experts' gate, up and SwiGLU of up to 8 tokens as ggml-cuda's fused matvec chain
    /// ([`crate::model::core::set_mmvq_glu`]; off by default).
    pub nvfp4_mmvq_glu: Option<bool>,
    /// W8A16 FP8 matmuls of 5-9 tokens on the GEMV
    /// ([`crate::model::fp8::set_small_batch_gemv`]).
    pub fp8_small_batch_gemv: Option<bool>,
    /// W8A16 FP8 matmuls of 5-64 tokens on the sliced-K GEMM
    /// ([`crate::model::fp8::set_sliced_k`]).
    pub fp8_sliced_k: Option<bool>,
    /// Dense NVFP4 matmuls of 33-64 tokens on the mv kernel
    /// ([`crate::model::nvfp4_mm::set_wide_mv`]).
    pub nvfp4_wide_mv: Option<bool>,
    /// The routed experts' down mv as a programmatic dependent launch of the grouped SwiGLU mv
    /// ([`crate::model::nvfp4_mm::set_expert_pdl`]).
    pub nvfp4_expert_pdl: Option<bool>,
    /// The routed experts' decode mvs prefetching whole row tiles into L2
    /// ([`crate::model::nvfp4_mm::set_expert_tile_prefetch`]).
    pub nvfp4_expert_tile_prefetch: Option<bool>,
    /// The most pairs per routed expert on the grouped mv kernel, more on the grouped GEMM
    /// ([`crate::model::nvfp4_mm::set_expert_mv_max`]).
    pub nvfp4_expert_mv_max: Option<i64>,
    /// The grouped mv kernel's widest token tile
    /// ([`crate::model::nvfp4_mm::set_expert_mv_tile`]).
    pub nvfp4_expert_mv_tile: Option<i32>,
    /// How the mv/GEMM cutoff applies: per step, per expert, the down projection only
    /// ([`crate::model::nvfp4_mm::set_expert_mv_split`]).
    pub nvfp4_expert_mv_split: Option<crate::model::nvfp4_mm::MvSplit>,
    /// Verify steps routing their padding rows' experts as their sequences' first rows
    /// ([`crate::model::core::pad_rows::set_pad_remap`]).
    pub moe_pad_remap: Option<bool>,
    /// The padding-row remap over an expert cache's tiered experts too
    /// ([`crate::model::core::pad_rows::set_tiered`]).
    pub moe_pad_remap_tiered: Option<bool>,
    /// Which uses of the tiered remap take it, a diagnostic
    /// ([`crate::model::core::pad_rows::set_probe`]).
    pub moe_pad_remap_probe: Option<crate::model::core::pad_rows::Probe>,
    /// The expert matvec over an expert cache skipping verify steps' dead padding pairs
    /// ([`crate::model::core::pad_rows::set_skip_tiered`]).
    pub moe_pad_skip_tiered: Option<bool>,
    /// The skip's row map attached but unused, a diagnostic
    /// ([`crate::model::core::pad_rows::set_skip_layout_only`]).
    pub moe_pad_skip_layout_only: Option<bool>,
    /// Verify steps skipping their padding rows' routed expert work on the decode mvs
    /// ([`crate::model::nvfp4_mm::set_pad_skip`]).
    pub moe_pad_skip: Option<bool>,
    /// `--tp-moe-ep` NVFP4 experts skipping the other ranks' pairs in the grouped kernels
    /// ([`crate::model::core::tp_ep::set_skip`]).
    pub tp_moe_ep_skip: Option<bool>,
    /// Paged attention of 2-16-token groups split over the key blocks
    /// ([`crate::model::core::paged::set_verify_split`]).
    pub paged_verify_split: Option<bool>,
    /// sm_70 / sm_75: 2-16-token groups' rows split over the key blocks, the WMMA kernel's bits
    /// ([`crate::model::core::paged::set_verify_split_wmma`]).
    pub paged_verify_split_wmma: Option<bool>,
    /// MoE layers' shared-expert gate, scaling and sum as one op
    /// ([`crate::model::core::set_shexp_gate`]).
    pub shexp_gate: Option<bool>,
    /// Qwen3.8-Flash-Next's decode-sized steps on fewer, fused ops
    /// ([`crate::model::core::set_decode_fuse`]).
    pub decode_fuse: Option<bool>,
    /// Small ggml node chains as Legion ops after allocation, with ggml's bits
    /// ([`crate::model::core::set_node_fuse`]; on by default).
    pub node_fuse: Option<bool>,
    /// How the fused MoE op keeps per-pair outputs ([`PairOut::set_default`]).
    pub moe_pair_out: Option<PairOut>,
    /// The fused MoE op's GEMMs on sm_80+ staging weight rows through shared memory with
    /// cp.async ([`crate::model::moe_ffn::set_raw_staging`]; on by default).
    pub moe_raw_staging: Option<bool>,
    /// The fused MoE op's down GEMM making the per-token sums itself on the GB10
    /// ([`crate::model::moe_ffn::set_down_sum`]; off by default).
    pub moe_down_sum: Option<bool>,
    /// How the decode expert matvec reads VRAM experts
    /// ([`crate::model::core::expert_mv::set_vram_kernel`]).
    pub expert_mv_vram: Option<usize>,
    /// The decode expert matvec reads VRAM experts in place with each lane's loads up front
    /// ([`crate::model::core::expert_mv::set_prefetch`]).
    pub expert_mv_prefetch: Option<bool>,
    /// The tile kernel's link worker CTAs
    /// ([`crate::model::core::expert_mv::set_link_ctas`]).
    pub expert_mv_link_ctas: Option<usize>,
    /// VRAM experts of the formats it takes through the PTX kernels
    /// ([`crate::model::core::expert_mv::ptx::set_enabled`]).
    pub expert_mv_ptx: Option<bool>,
    /// The PTX kernels on a side stream beside the tile kernel
    /// ([`crate::model::core::expert_mv::ptx::set_side`]).
    pub expert_mv_ptx_side: Option<bool>,
    /// Experts in GPU memory take the SwiGLU matvec in verify steps
    /// ([`crate::model::core::expert_mv::set_device`]).
    pub expert_mv_device: Option<bool>,
    /// Most tokens of a step the expert matvec takes
    /// ([`crate::model::core::expert_mv::set_max_tokens`]).
    pub expert_mv_max_tokens: Option<usize>,
    /// Gate and up in one expert matvec ([`crate::model::core::expert_mv::set_glu`]).
    pub expert_mv_glu: Option<bool>,
    /// The tile kernel's dot products ([`crate::model::core::expert_mv::set_dot`]).
    pub expert_mv_dot: Option<crate::model::core::expert_mv::Dot>,
    /// Tiles per CTA of the tile kernel, pipelined
    /// ([`crate::model::core::expert_mv::set_pipe_tiles`]).
    pub expert_mv_pipe: Option<usize>,
    /// The Q4_K gate/up kernel's per-unit activations
    /// ([`crate::model::core::expert_mv::set_q4k_units`]).
    pub expert_mv_q4k_units: Option<bool>,
    /// Q4_K gate/up activations held in registers over the piped kernel's tiles
    /// ([`crate::model::core::expert_mv::set_q4k_hold`]).
    pub expert_mv_q4k_hold: Option<bool>,
    /// The piped tiles balanced over the resident CTAs
    /// ([`crate::model::core::expert_mv::set_balance`]).
    pub expert_mv_balance: Option<bool>,
    /// The piped Q5_1 down on tiles of half the rows
    /// ([`crate::model::core::expert_mv::set_half_tiles`]).
    pub expert_mv_half_tiles: Option<bool>,
    /// A tiered layer's missed experts on a side stream
    /// ([`crate::model::core::expert_mv::overlap::set_enabled`]).
    pub expert_miss_overlap: Option<bool>,
    /// The running GPU's idle NVLink peer fetches a share of a tiered layer's missed experts
    /// ([`crate::model::core::expert_mv::peer::set_enabled`]); `None`: the automatic choice
    /// ([`crate::model::core::expert_mv::peer::auto`]).
    pub expert_peer_fetch: Option<bool>,
    /// The peer's share of a layer's missed experts
    /// ([`crate::model::core::expert_mv::peer::set_share`]).
    pub expert_peer_share: Option<crate::model::core::expert_mv::peer::Share>,
    /// The peer's copy CTAs ([`crate::model::core::expert_mv::peer::set_ctas`]).
    pub expert_peer_ctas: Option<usize>,
    /// qwen4exp's indexer scores on Volta's tensor cores
    /// ([`crate::model::qwen4::qsa_attn::set_scores_tc`]).
    pub qsa_scores_tc: Option<bool>,
    /// qwen4exp's Volta prompt kernel accumulating in chains
    /// ([`crate::model::qwen4::qsa_attn::set_attn_chain`]).
    pub qsa_attn_chain: Option<bool>,
    /// qwen4exp's indexer early exit: steps whose tokens see at most its budget of pools select
    /// them all unscored ([`crate::model::qwen4::qsa_attn::set_pick_all`]).
    pub qsa_pick_all: Option<bool>,
    /// qwen4exp's small-step hyper-connection mixer op ([`crate::model::qwen4::hc::set_mix_op`]).
    pub hc_mix: Option<bool>,
    /// qwen4exp's small-step hyper-connection mixer op on BF16 mixer weights
    /// ([`crate::model::qwen4::hc::set_mix_bf16`]).
    pub hc_mix_bf16: Option<bool>,
    /// Which mixer weights the V100 Q8_0 kernels read into L2 ahead of their reader
    /// ([`crate::model::qwen4::hc::set_mix_l2_prefetch`]).
    pub hc_l2_prefetch: Option<crate::model::qwen4::hc::L2Prefetch>,
    /// qwen4exp's V100 mixer norm with the stream in registers
    /// ([`crate::model::qwen4::hc::set_norm_regs`]).
    pub hc_norm_regs: Option<bool>,
    /// qwen4exp's V100 mixer down kernel split over twice the threads at 1-2 tokens
    /// ([`crate::model::qwen4::hc::set_down_split`]).
    pub hc_down_split: Option<bool>,
    /// qwen4exp's V100 mixer down kernel with its address math out of the loop
    /// ([`crate::model::qwen4::hc::set_down_hoist`]).
    pub hc_down_hoist: Option<bool>,
    /// qwen4exp's V100 mixer up kernel with its rows staged in shared memory
    /// ([`crate::model::qwen4::hc::set_up_stage`]).
    pub hc_up_stage: Option<bool>,
    /// qwen4exp's fused combine-and-norm ([`crate::model::qwen4::set_fused_post_norm`]).
    pub hc_post_norm: Option<bool>,
    /// qwen4exp's MoE tails folded into the next combine-and-norm before the graph is planned
    /// ([`crate::model::qwen4::set_moe_fold`]; on by default on an integrated GPU).
    pub hc_moe_fold: Option<bool>,
    /// qwen4exp's hyper-connection glue fused around the MoE op on a V100 or an sm_80+ GPU
    /// ([`crate::model::qwen4::set_hc_fuse`]).
    pub hc_fuse: Option<bool>,
    /// qwen4exp's combine-and-norm also writing the normed streams as F16 for the mixer's down
    /// projection on a V100 ([`crate::model::qwen4::set_xn16`]).
    pub hc_xn16: Option<bool>,
    /// qwen4exp's fused combine-and-norm quantizing the down matmul's input for ggml's MMQ
    /// ([`crate::model::core::set_mmq_preq`]).
    pub hc_quant: Option<bool>,
    /// Q4_K / Q5_1 expert matmuls of 9-63 tokens on Legion's kernel with MMQ's bits
    /// ([`crate::model::core::set_mmq_ids`]).
    pub mmq_ids: Option<bool>,
    /// NVFP4 expert matmuls of 2-8 tokens reading each routed expert once, with ggml's MMVQ bits
    /// ([`crate::model::core::set_nvfp4_once`]).
    pub nvfp4_once: Option<bool>,
    /// NVFP4 dense matmuls of 2-8 tokens on Legion's tensor-core op with ggml's MMVQ bits
    /// ([`crate::model::core::set_nvfp4_mvq`]).
    pub nvfp4_mvq: Option<bool>,
    /// Q8_0 matmuls of 9-255 tokens on Legion's two-launch op with MMQ's bits
    /// ([`crate::model::q8_mmq::set_q8_mmq`]).
    pub q8_mmq: Option<bool>,
    /// The op's matmul on the streaming kernel with MMQ's bits, 8 warps a CTA
    /// ([`crate::model::q8_mmq::set_q8_mmq_fast`]).
    pub q8_mmq_stream: Option<bool>,
    /// How the op adds up split tiles' pieces, all with MMQ's bits
    /// ([`crate::model::q8_mmq::set_q8_mmq_fixup`]).
    pub q8_mmq_fixup: Option<crate::model::q8_mmq::Fixup>,
    /// qwen4exp's mixer up matmul and gated mix as one op ([`crate::model::qwen4::set_up_mix`]).
    pub hc_up_mix: Option<bool>,
    /// The same op for BF16 mixer up weights ([`crate::model::qwen4::set_up_mix_bf16`], on by
    /// default).
    pub hc_up_mix_bf16: Option<bool>,
    /// qwen4exp's combine-and-norm writing the streams' norm scales instead of the normed
    /// streams, rebuilt by the up-mix op ([`crate::model::qwen4::set_up_mix_rebuild`]).
    pub hc_xn_rebuild: Option<bool>,
    /// qwen4exp's residual streams kept in BF16 between the GB10 prompt mixers
    /// ([`crate::model::qwen4::set_streams_bf16`], opt-in, moves output).
    pub hc_streams_bf16: Option<bool>,
    /// qwen4exp's mixer gated mix also writing the block input as F16 for its Q8_0 projections
    /// on a V100 ([`crate::model::qwen4::set_mix16`]).
    pub hc_mix16: Option<bool>,
    /// qwen4exp's sm_80+ BF16 mixers with `xn` rebuilt in the gated mix instead of written by
    /// the norm ([`crate::model::qwen4::set_mix_rebuild_bf16`], opt-in).
    pub hc_mix_rebuild_bf16: Option<bool>,
    /// qwen4exp's mixer scale+SiLU, combine and grouped norm on a GPU as Legion's kernels
    /// instead of ggml-cuda's ops ([`crate::model::qwen4::set_hc_chain`], on by default).
    pub hc_chain: Option<bool>,
    /// qwen4exp's mixer gated mix on a GPU as Legion's kernel instead of ggml-cuda's hc_pre
    /// ([`crate::model::qwen4::set_hc_pre`]).
    pub hc_pre: Option<bool>,
    /// A sigmoid-gated output projection's gate and input quantization as one op
    /// ([`crate::model::core::set_gate_q8`]).
    pub gate_q8: Option<bool>,
    /// A gated DeltaNet layer's output RMS norm inside that op
    /// ([`crate::model::core::set_gate_norm`]).
    pub gate_norm: Option<bool>,
    /// One quantization of a layer input for the Q8_0 projections that read it
    /// ([`crate::model::core::set_shared_q8`]).
    pub shared_quant: Option<bool>,
    /// The same in steps of 9-255 tokens, through a scratch buffer
    /// ([`crate::model::core::set_shared_q8_small`]).
    pub shared_quant_small: Option<bool>,
    /// A gated shared expert added in the fused MoE op's sum
    /// ([`crate::model::moe_ffn::set_shexp_sum`]).
    pub shexp_sum: Option<bool>,
    /// qwen4exp's separate hyper-connection norm rounded as the fused one on sm_7x
    /// ([`crate::model::qwen4::set_norm_fma`]).
    pub hc_norm_fma: Option<bool>,
    /// qwen4exp's per-layer embedding as Legion's two ops in prompt steps
    /// ([`crate::model::qwen4::set_fused_ple`]; off by default).
    pub ple: Option<bool>,
    /// The expert cache's work after a step overlapping GPU work (the next MTP pass, the next
    /// stage)
    /// ([`crate::model::core::Executor::set_after_overlap`]; on by default).
    pub after_overlap: Option<bool>,
    /// A decode or verify step's and an MTP draft pass's small input copies as one kernel
    /// launch per stage ([`crate::ggml::pinned_io::pack::set_input_pack`]; on by default).
    pub input_pack: Option<bool>,
    /// A decode or verify step's next stage queued while the stage before it computes
    /// ([`crate::model::core::Executor::set_launch_ahead`]; on by default).
    pub launch_ahead: Option<bool>,
    /// A layer split's verify commit queued beside the first MTP draft pass
    /// ([`crate::model::core::Executor::set_commit_beside_draft`]; on by default on compute
    /// capability 7.x).
    pub commit_beside_draft: Option<bool>,
    /// Of such a held commit, the last stage's queued behind the second draft pass
    /// ([`crate::model::core::Executor::set_commit_last_late`]; off by default,
    /// `--commit-last-late`).
    pub commit_last_late: Option<bool>,
    /// Each stage's verify commit captured as one stage graph
    /// ([`crate::model::core::stage_ops::set_commit_capture`]; off by default,
    /// `--commit-capture`).
    pub commit_capture: Option<bool>,
    /// MTP draft passes queue the next pass's inputs while they compute
    /// ([`crate::model::core::Executor::set_mtp_ahead`]; off by default).
    pub mtp_ahead: Option<bool>,
    /// The most rows of an MTP draft pass queued before the host has the last pass's picks
    /// ([`crate::model::core::Executor::set_mtp_pipe_rows`]; 0, off, by default).
    pub mtp_pipe_rows: Option<usize>,
    /// The fewest rows of an MTP draft pass whose routed experts take the fused expert op
    /// ([`crate::model::core::mtp_moe::set_rows`]; 0, off, by default).
    pub mtp_moe_fused_rows: Option<usize>,
    /// With the expert cache on discrete GPUs: keep prefix caching's saved states in VRAM
    /// instead of pinned host memory (#441; read by [`LmEngine::load`](super::lm::LmEngine::load)).
    pub saved_states_vram: Option<bool>,
    /// GLM-5.3-Flash: routed experts in GPU memory through the expert matvec in small steps
    /// ([`crate::model::glm5next::set_expert_mv_device`]).
    pub glm_expert_mv: Option<bool>,
    /// GLM-5.3-Flash: the most tokens of a verify step over tiered experts the expert matvec
    /// takes ([`crate::model::glm5next::set_verify_mv_tokens`]).
    pub glm_verify_mv_tokens: Option<usize>,
    /// GLM-5.3-Flash: plain decode steps over tiered experts take the verify limit too
    /// ([`crate::model::glm5next::set_decode_mv`]; off by default).
    pub glm_decode_mv: Option<bool>,
    /// GLM-5.3-Flash: the GB10-tuned linear kernels
    /// ([`crate::model::glm5next::ops::set_gb10_linear`]; `None`: the engine's choice, on for
    /// integrated GPUs, V100s and Ampere GPUs).
    pub glm_gb10_linear: Option<bool>,
    /// GLM-5.3-Flash: prompt MLA attention on tensor cores
    /// ([`crate::model::glm5next::ops::set_mla_tc`]; off by default).
    pub glm_mla_tc: Option<bool>,
    /// GLM-5.3-Flash: prompt expert GEMMs on int8 tensor cores
    /// ([`crate::model::glm5next::ops::set_expert_q8`]; off by default).
    pub glm_expert_q8: Option<bool>,
    /// Loader worker threads ([`crate::load_io::configure`]).
    pub load_threads: Option<usize>,
    /// Every loader worker writes device tensors ([`crate::load_io::configure`]).
    pub load_parallel_device: Option<bool>,
    /// Map plain-CPU tensors from the file ([`crate::load_io::configure`]).
    pub mmap: Option<bool>,
    /// The loader's page cache handling ([`crate::load_io::configure`]).
    pub load_page_cache: Option<crate::load_io::PageCache>,
    /// CUDA graphs as stage graphs, external ops included ([`crate::ggml::stage_graph`];
    /// `legion serve` turns ggml's own graphs off for them before CUDA starts).
    pub stage_graphs: Option<bool>,
    /// Big stage-graph captures launched as a short head and the rest
    /// ([`crate::ggml::stage_graph::set_split`]; on by default).
    pub stage_graph_split: Option<bool>,
    /// Small steps' new stage graphs captured behind their first run
    /// ([`crate::ggml::stage_graph::set_early`]; on by default).
    pub stage_graph_early: Option<bool>,
    /// MTP take-ins of several sequences captured behind their second run
    /// ([`crate::ggml::stage_graph::set_reuse_capture`]; on by default).
    pub stage_graph_reuse_capture: Option<bool>,
    /// Unchanged kept stage graphs skip their words check
    /// ([`crate::ggml::stage_graph::kept_words::set_enabled`]; on by default).
    pub stage_graph_words_skip: Option<bool>,
    /// Every run of a kept graph keeps its stage-graph captures with the graph
    /// ([`crate::ggml::stage_graph::kept_words::set_hold_all`]; off by default).
    pub stage_graph_hold_kept: Option<bool>,
    /// The stage graphs' periodic counts in the log ([`crate::ggml::stage_graph::stats`]).
    pub stage_graph_stats: Option<bool>,
    /// One-stage prompt steps build the next chunk's graph while they compute
    /// ([`crate::model::core::set_prompt_build_ahead`]; on by default).
    pub prompt_build_ahead: Option<bool>,
    /// The load allocates an integrated GPU's kept prompt graphs' compute buffers
    /// ([`crate::model::core::set_prompt_buffer_reserve`]; on by default).
    pub prompt_buffer_reserve: Option<bool>,
    /// Long prompt steps' stats line in the log
    /// ([`crate::model::core::set_prompt_step_stats`]; off by default).
    pub prompt_step_stats: Option<bool>,
    /// The kept graphs' input check in the log ([`crate::ggml::input_audit`]; off by default).
    pub audit_graph_inputs: Option<bool>,
    /// Sets of stage-graph captures each device keeps by their words
    /// ([`crate::ggml::stage_graph::set_cache_sets`]; 64 by default, 0 off).
    pub stage_graph_cache: Option<usize>,
    /// Spare executable graphs each device keeps from dropped stage-graph captures, updated
    /// by new captures of the same topology ([`crate::ggml::stage_graph::set_spare_execs`]; 8
    /// by default, 0 off).
    pub stage_graph_spare_execs: Option<usize>,
    /// Kept stage-graph captures unlaunched this many ms are taken over by new captures of
    /// the same topology ([`crate::ggml::stage_graph::set_exec_steal_ms`]; 1000 by default,
    /// 0 never).
    pub stage_graph_exec_steal_ms: Option<u64>,
    /// Graph shapes the MTP head keeps per kind of pass
    /// ([`crate::model::core::set_mtp_kept_graphs`]; 16 by default).
    pub mtp_kept_graphs: Option<usize>,
    /// Verify shapes each stage keeps apart from prompt shapes
    /// ([`crate::model::core::set_verify_kept_graphs`]; 0 by default: shared).
    pub verify_kept_graphs: Option<usize>,
    /// MiB each stage's kept verify graphs may hold before the least recently run go
    /// ([`crate::model::core::Executor::set_verify_budget`]; `None`: an eighth of the free
    /// memory after the load, at most 2 GiB; 0: no limit).
    pub verify_kept_mib: Option<usize>,
    /// MTP take-in passes order their sequences by rows
    /// ([`crate::engine::lm::set_mtp_take_in_sorted`]; on by default).
    pub mtp_take_in_sort: Option<bool>,
    /// qwen4exp's decode and verify pool padding in steps
    /// ([`crate::model::qwen4::set_step_pool_steps`]; 0, the default: off).
    pub qsa_pool_pad_steps: Option<usize>,
    /// The GPU stages' KV pools in reserved virtual ranges, each cache tensor backed by VRAM
    /// allocations of its own (`crate::residency::kv_vmm`; None: on for a V100 layer split of 2
    /// GPUs with the expert cache, off elsewhere).
    pub kv_vmm: Option<bool>,
    /// `--kv-vmm-keep`: the KV tokens of each GPU's pool that keep VRAM at load, the rest lent
    /// to the expert cache until needed (`engine::lm::kv_vmm::KvLend`; None: 32768 where
    /// `kv_vmm` is on by default, none elsewhere).
    pub kv_vmm_keep: Option<usize>,
}

/// Turns stage graphs on or off, with the model layer's list of the external ops they may
/// capture. On only with ggml's own CUDA graphs off (`GGML_CUDA_DISABLE_GRAPHS`): a stage
/// graph would capture ggml's graph launches otherwise, so without it they stay off.
fn apply_stage_graphs(on: bool) {
    use crate::ggml::stage_graph;
    stage_graph::set_safe_ops(crate::model::core::stage_ops::capture_safe);
    stage_graph::set_direct_prompt_ops(crate::model::core::stage_ops::direct_in_prompt);
    let ggml_off = std::env::var_os("GGML_CUDA_DISABLE_GRAPHS").is_some();
    if on && !ggml_off {
        tracing::warn!("--stage-graphs needs GGML_CUDA_DISABLE_GRAPHS set before CUDA starts; off");
    }
    stage_graph::set_enabled(on && ggml_off);
}

impl Switches {
    /// Sets the process-wide state for what is `Some`.
    pub fn apply(&self) {
        use crate::model::core::{self, expert_mv};
        use crate::model::qwen4::{self, qsa_attn};
        let set = |v: Option<bool>, f: fn(bool)| v.into_iter().for_each(f);
        set(self.gated_norm, core::set_gated_norm);
        set(self.moe_combine, core::set_moe_combine);
        set(self.stable_fusions, core::set_stable_fusions);
        set(self.delta_prep, core::set_delta_prep);
        set(self.gdn_scan, core::set_gdn_scan);
        set(self.gdn_mma, core::set_gdn_mma);
        set(self.gdn, core::set_gdn);
        set(self.gdn_decode, core::set_gdn_decode);
        set(self.gdn_replay_staged, core::set_gdn_replay_staged);
        set(self.gdn_pad_skip, core::set_gdn_pad_skip);
        set(self.gdn_commit_slots, core::set_commit_slots);
        set(self.gdn_commit_in_verify, core::set_commit_in_verify);
        if self.gdn_state_defer.is_some() {
            core::set_state_defer(self.gdn_state_defer);
        }
        if let Some(rows) = self.gdn_state_defer_rows {
            core::set_state_defer_rows(rows.into());
        }
        if let Some(n) = self.gdn_state_defer_min_seqs {
            core::set_state_defer_min_seqs(n.into());
        }
        set(self.norm_rope, core::set_norm_rope);
        set(self.kv_append, core::set_kv_append);
        set(
            self.head_rope_fused,
            crate::model::gemma4::set_head_rope_fused,
        );
        set(self.joined_matmuls, core::set_joined_matmuls);
        if let Some(n) = self.joined_prompt_min_tokens {
            core::set_joined_prompt_min_tokens(n);
        }
        set(self.joined_prompt_z, core::set_joined_prompt_z);
        set(self.joined_prompt_qkv, core::set_joined_prompt_qkv);
        set(
            self.input_pack,
            crate::ggml::pinned_io::pack::set_input_pack,
        );
        set(self.dense_mv, crate::model::dense_mv::set_dense_mv);
        set(self.q8_mv, core::set_q8_mv);
        set(self.q8_head, core::set_q8_head);
        set(self.q8_head_pick, core::q8_head::pick::set_pick);
        set(self.q8_row_mv, core::q8_head::set_row_mv);
        set(self.q4k_head, core::set_q4k_head);
        set(self.q8_out_mv, core::q8_head::set_out_mv);
        set(self.q4k_head_pick, core::q4k_head::set_pick);
        set(self.q4k_head_pick_cols, core::q4k_head::set_pick_cols);
        set(self.q8_group, core::set_q8_group);
        if let Some(m) = self.q8_group_parts {
            core::q8_group::set_parts(m);
        }
        set(self.dense_mv_small, crate::model::dense_mv::set_small);
        set(
            self.cluster_select,
            crate::model::cluster_select::set_cluster_select,
        );
        set(self.dense_mv_f32, crate::model::dense_mv::set_f32);
        set(self.dense_mv_ab, crate::model::dense_mv::set_chain);
        set(self.dense_mv_split, crate::model::dense_mv::set_split);
        set(
            self.dense_mv_wide_chain,
            crate::model::dense_mv::set_wide_chain,
        );
        set(self.router_mv, core::set_router_mv);
        set(self.router_join, core::set_router_join);
        set(self.router_bf16, core::set_router_bf16);
        set(
            self.ple_early_reads,
            crate::model::qwen4::set_ple_early_reads,
        );
        set(self.nvfp4_mmvq_glu, core::set_mmvq_glu);
        set(
            self.fp8_small_batch_gemv,
            crate::model::fp8::set_small_batch_gemv,
        );
        set(self.fp8_sliced_k, crate::model::fp8::set_sliced_k);
        set(self.nvfp4_wide_mv, crate::model::nvfp4_mm::set_wide_mv);
        set(
            self.nvfp4_expert_pdl,
            crate::model::nvfp4_mm::set_expert_pdl,
        );
        set(
            self.nvfp4_expert_tile_prefetch,
            crate::model::nvfp4_mm::set_expert_tile_prefetch,
        );
        if let Some(n) = self.nvfp4_expert_mv_max {
            crate::model::nvfp4_mm::set_expert_mv_max(n);
        }
        if let Some(nt) = self.nvfp4_expert_mv_tile {
            crate::model::nvfp4_mm::set_expert_mv_tile(nt);
        }
        if let Some(mode) = self.nvfp4_expert_mv_split {
            crate::model::nvfp4_mm::set_expert_mv_split(mode);
        }
        set(
            self.moe_pad_remap,
            crate::model::core::pad_rows::set_pad_remap,
        );
        set(self.moe_pad_skip, crate::model::nvfp4_mm::set_pad_skip);
        set(self.tp_moe_ep_skip, crate::model::core::tp_ep::set_skip);
        set(
            self.moe_pad_remap_tiered,
            crate::model::core::pad_rows::set_tiered,
        );
        if let Some(p) = self.moe_pad_remap_probe {
            crate::model::core::pad_rows::set_probe(p);
        }
        set(
            self.moe_pad_skip_tiered,
            crate::model::core::pad_rows::set_skip_tiered,
        );
        set(
            self.moe_pad_skip_layout_only,
            crate::model::core::pad_rows::set_skip_layout_only,
        );
        set(
            self.paged_verify_split,
            crate::model::core::paged::set_verify_split,
        );
        set(
            self.paged_verify_split_wmma,
            crate::model::core::paged::set_verify_split_wmma,
        );
        set(self.shexp_gate, core::set_shexp_gate);
        set(self.decode_fuse, core::set_decode_fuse);
        set(self.node_fuse, core::set_node_fuse);
        crate::ggml::post_alloc::set_rewrite(core::node_fuse::rewrite);
        crate::ggml::post_alloc::set_relayout(core::node_fuse::fold::RELAYOUT);
        if let Some(p) = self.moe_pair_out {
            PairOut::set_default(p);
        }
        set(self.moe_raw_staging, crate::model::moe_ffn::set_raw_staging);
        set(self.moe_down_sum, crate::model::moe_ffn::set_down_sum);
        if let Some(n) = self.mtp_moe_fused_rows {
            core::mtp_moe::set_rows(n);
        }
        if let Some(m) = self.expert_mv_vram {
            expert_mv::set_vram_kernel(m);
        }
        set(self.expert_mv_prefetch, expert_mv::set_prefetch);
        if let Some(n) = self.expert_mv_link_ctas {
            expert_mv::set_link_ctas(n);
        }
        set(self.expert_mv_ptx, expert_mv::ptx::set_enabled);
        set(self.expert_mv_ptx_side, expert_mv::ptx::set_side);
        set(self.expert_mv_device, expert_mv::set_device);
        if let Some(n) = self.expert_mv_max_tokens {
            expert_mv::set_max_tokens(n);
        }
        set(self.expert_mv_glu, expert_mv::set_glu);
        if let Some(d) = self.expert_mv_dot {
            expert_mv::set_dot(d);
        }
        if let Some(n) = self.expert_mv_pipe {
            expert_mv::set_pipe_tiles(Some(n));
        }
        set(self.expert_mv_q4k_units, expert_mv::set_q4k_units);
        set(self.expert_mv_q4k_hold, expert_mv::set_q4k_hold);
        set(self.expert_mv_balance, expert_mv::set_balance);
        set(self.expert_mv_half_tiles, expert_mv::set_half_tiles);
        set(self.expert_miss_overlap, expert_mv::overlap::set_enabled);
        set(self.expert_peer_fetch, expert_mv::peer::set_enabled);
        if let Some(sh) = self.expert_peer_share {
            expert_mv::peer::set_share(sh.take, sh.every);
        }
        if let Some(n) = self.expert_peer_ctas {
            expert_mv::peer::set_ctas(n);
        }
        set(self.qsa_scores_tc, qsa_attn::set_scores_tc);
        set(self.qsa_attn_chain, qsa_attn::set_attn_chain);
        set(self.qsa_pick_all, qsa_attn::set_pick_all);
        set(self.hc_mix, qwen4::hc::set_mix_op);
        set(self.hc_mix_bf16, qwen4::hc::set_mix_bf16);
        if let Some(m) = self.hc_l2_prefetch {
            qwen4::hc::set_mix_l2_prefetch(m);
        }
        set(self.hc_norm_regs, qwen4::hc::set_norm_regs);
        set(self.hc_down_split, qwen4::hc::set_down_split);
        set(self.hc_down_hoist, qwen4::hc::set_down_hoist);
        set(self.hc_up_stage, qwen4::hc::set_up_stage);
        set(self.hc_post_norm, qwen4::set_fused_post_norm);
        set(self.hc_moe_fold, qwen4::set_moe_fold);
        set(self.hc_fuse, qwen4::set_hc_fuse);
        set(self.hc_xn16, qwen4::set_xn16);
        set(self.hc_quant, core::set_mmq_preq);
        set(self.mmq_ids, core::set_mmq_ids);
        set(self.nvfp4_once, core::set_nvfp4_once);
        set(self.nvfp4_mvq, core::set_nvfp4_mvq);
        set(self.q8_mmq, crate::model::q8_mmq::set_q8_mmq);
        set(self.q8_mmq_stream, |on| {
            crate::model::q8_mmq::set_q8_mmq_fast(if on { 8 } else { 0 })
        });
        if let Some(f) = self.q8_mmq_fixup {
            crate::model::q8_mmq::set_q8_mmq_fixup(f);
        }
        set(self.hc_up_mix, qwen4::set_up_mix);
        set(self.hc_up_mix_bf16, qwen4::set_up_mix_bf16);
        set(self.hc_xn_rebuild, qwen4::set_up_mix_rebuild);
        set(self.hc_streams_bf16, qwen4::set_streams_bf16);
        set(self.hc_mix16, qwen4::set_mix16);
        set(self.hc_mix_rebuild_bf16, qwen4::set_mix_rebuild_bf16);
        set(self.hc_chain, qwen4::set_hc_chain);
        set(self.hc_pre, qwen4::set_hc_pre);
        set(self.gate_q8, core::set_gate_q8);
        set(self.gate_norm, core::set_gate_norm);
        set(self.shared_quant, core::set_shared_q8);
        set(self.shared_quant_small, core::set_shared_q8_small);
        set(self.shexp_sum, crate::model::moe_ffn::set_shexp_sum);
        set(self.hc_norm_fma, qwen4::set_norm_fma);
        set(self.ple, qwen4::set_fused_ple);
        use crate::model::glm5next;
        set(self.glm_expert_mv, glm5next::set_expert_mv_device);
        if let Some(n) = self.glm_verify_mv_tokens {
            glm5next::set_verify_mv_tokens(n);
        }
        set(self.glm_decode_mv, glm5next::set_decode_mv);
        set(self.glm_gb10_linear, glm5next::ops::set_gb10_linear);
        set(self.glm_mla_tc, glm5next::ops::set_mla_tc);
        set(self.glm_expert_q8, glm5next::ops::set_expert_q8);
        if let Some(on) = self.stage_graphs {
            apply_stage_graphs(on);
        }
        set(self.stage_graph_split, crate::ggml::stage_graph::set_split);
        set(self.stage_graph_early, crate::ggml::stage_graph::set_early);
        set(
            self.stage_graph_reuse_capture,
            crate::ggml::stage_graph::set_reuse_capture,
        );
        set(
            self.stage_graph_words_skip,
            crate::ggml::stage_graph::kept_words::set_enabled,
        );
        set(
            self.stage_graph_hold_kept,
            crate::ggml::stage_graph::kept_words::set_hold_all,
        );
        set(
            self.stage_graph_stats,
            crate::ggml::stage_graph::stats::set_logging,
        );
        set(
            self.prompt_build_ahead,
            crate::model::core::set_prompt_build_ahead,
        );
        set(
            self.prompt_buffer_reserve,
            crate::model::core::set_prompt_buffer_reserve,
        );
        set(
            self.prompt_step_stats,
            crate::model::core::set_prompt_step_stats,
        );
        set(
            self.audit_graph_inputs,
            crate::ggml::input_audit::set_enabled,
        );
        if let Some(n) = self.stage_graph_cache {
            crate::ggml::stage_graph::set_cache_sets(n);
        }
        if let Some(n) = self.stage_graph_spare_execs {
            crate::ggml::stage_graph::set_spare_execs(n);
        }
        if let Some(ms) = self.stage_graph_exec_steal_ms {
            crate::ggml::stage_graph::set_exec_steal_ms(ms);
        }
        if let Some(n) = self.mtp_kept_graphs {
            crate::model::core::set_mtp_kept_graphs(n);
        }
        if let Some(n) = self.verify_kept_graphs {
            crate::model::core::set_verify_kept_graphs(n);
        }
        if let Some(on) = self.mtp_take_in_sort {
            crate::engine::lm::set_mtp_take_in_sorted(on);
        }
        if let Some(n) = self.qsa_pool_pad_steps {
            qwen4::set_step_pool_steps(n);
        }
        crate::load_io::configure(
            self.load_threads,
            self.load_parallel_device,
            self.mmap,
            self.load_page_cache,
        );
    }
}
