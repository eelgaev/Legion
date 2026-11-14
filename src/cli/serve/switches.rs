//! Kernel and load switches of `legion serve` / `legion run` (#421). The flags are the only way
//! in: the `LEGION_*` environment variables they replaced are no longer read (#737).

use legion::engine::switches::Switches;
use legion::model::core::expert_mv::Dot;
use legion::model::core::pad_rows::Probe;
use legion::model::moe_ffn::PairOut;
use legion::model::nvfp4_mm::MvSplit;
use legion::model::qwen4::hc::L2Prefetch;

/// `--help`'s heading for the switches (per argument: a struct-wide heading would carry on to
/// the flags after the flattened struct).
const HEADING: &str = "Kernel and load switches";

#[derive(clap::Args)]
pub(in crate::cli) struct SwitchArgs {
    /// CUDA, gated DeltaNet layers (Qwen3.5 / 3.6, Qwen3.8-Flash-Next): the output norm and its
    /// gate as one op instead of ggml's chain of nodes (the same rows in the tests). Off by
    /// default until measured. Replaces LEGION_GATED_NORM=1.
    #[arg(long, help_heading = HEADING)]
    gated_norm: bool,
    /// CUDA, MoE layers: the weighted sum of the routed experts' outputs as one op, rounded as
    /// ggml-cuda's fused reduction rounds it whatever the memory layout (ggml's graph rounds
    /// twice per expert when its tensors overlap, so outputs could move with unrelated
    /// allocations, #410). Off by default. Replaces LEGION_MOE_COMBINE=1.
    #[arg(long, help_heading = HEADING)]
    moe_combine: bool,
    /// CUDA, MoE layers: keep the inputs of ggml-cuda's fused weighted expert sum and top-k
    /// router allocated until their outputs, as llama.cpp's scheduler does, so ggml-cuda fuses
    /// them in every layer whatever the memory layout. Without it a graph change elsewhere
    /// (--kv-append, --ple-ops, --shexp-gate, ...) can flip some layers' sums between two
    /// roundings and move greedy text (#472). No extra kernels and no measured speed cost
    /// (prompt chunks take the fused MoE op and build no chain); the step-buffer sizing grows
    /// ~10 MiB per GPU on Qwen3.8-Flash-Next. Greedy text moves once from the default (the
    /// layers that ran unfused now run fused; the same bits as --moe-combine) and then no longer
    /// follows the layout. Off by default, so the default's text stays as it was.
    #[arg(long, help_heading = HEADING)]
    stable_fusions: bool,
    /// CUDA, gated DeltaNet layers: ggml's chain for the recurrent prep (l2 norms, g, beta, the
    /// replay stores) instead of Legion's one op (the default: the same bits, about 11 kernels
    /// fewer per layer). For comparisons. Replaces LEGION_DELTA_PREP=0.
    #[arg(long, help_heading = HEADING)]
    delta_prep_ggml: bool,
    /// CUDA, gated DeltaNet layers, prompt steps on compute capability 7.x GPUs (V100): ggml's
    /// gated_delta_net kernel instead of Legion's gdn_scan (the default there: ggml's bits, about
    /// 1.5x faster per call; other GPUs always run ggml's). For comparisons.
    #[arg(long, help_heading = HEADING)]
    gdn_scan_ggml: bool,
    /// CUDA sm_80+, gated DeltaNet layers, prompt groups of 64+ tokens whose head counts
    /// ggml-cuda keeps on its per-token kernel (it takes its chunked tensor-core kernel only for
    /// 16 key heads and 16/32/48/64 value heads): run a chunked kernel on them anyway (Legion's
    /// port of ggml-cuda's, the same bits). A tensor parallel rank's share is such a layer (--tp-rank:
    /// Qwen3.8-Flash-Next keeps 8 key and 24 value heads on each of 2 ranks; one GPU's 16/48
    /// already runs the chunked kernel). The rounding becomes the chunked kernel's, so tensor
    /// parallel output moves. Decode and verify steps are unchanged. Off by default; for tensor
    /// parallel prompts on two DGX Sparks (#781: plain TP 8K prefill 2,433/2,482 -> 2,517/2,577
    /// tok/s, 32K 2,493/2,503 -> 2,590/2,600; TP+MTP 32K 2,432/2,447 -> 2,510/2,513).
    #[arg(long, help_heading = HEADING)]
    gdn_mma: bool,
    /// CUDA, gated DeltaNet layers, prompt groups of 64+ tokens that run a chunked kernel with
    /// one CTA per head: ggml-cuda's chunked kernel for the groups it runs itself, and Legion's
    /// port of it for the groups --gdn-mma brings, instead of Legion's staged one (the default:
    /// the same arithmetic and bits on 16 warps that each keep an 8-column block of the state in
    /// registers, the next chunk's inputs copied into shared memory while the current one
    /// computes; on a DGX Spark (GB10) a Qwen3.8-Flash-Next layer of 4K prompt tokens takes 1.33
    /// ms instead of 2.41). The default takes ggml-cuda's own groups on compute capability 12.0
    /// and 12.1 only. For comparisons.
    #[arg(long, help_heading = HEADING)]
    gdn_prompt_ggml: bool,
    /// CUDA, gated DeltaNet layers in MTP verify and commit steps: ggml's gated_delta_net
    /// instead of Legion's op (the default: the same bits; the verify step skips the final
    /// states it never reads, the commit writes them in place instead of computing the rows and
    /// copying the states, and the token loop is unrolled for 1-8 tokens). For comparisons.
    #[arg(long, help_heading = HEADING)]
    gdn_ggml: bool,
    /// CUDA, compute capability 9.0+ (GB10, Hopper, RTX 50), gated DeltaNet layers in decode
    /// steps (one token per sequence): ggml's gated_delta_net instead of Legion's op (the
    /// default there: the same bits, the recurrent states read and written in 16-byte vectors
    /// through shared memory, which is what these steps spend their time on). For comparisons.
    #[arg(long, help_heading = HEADING)]
    gdn_decode_ggml: bool,
    /// CUDA, compute capability 9.0+ (GB10, Hopper, RTX 50), gated DeltaNet layers in MTP verify
    /// steps that run the deferred commit first (--gdn-commit-in-verify, the default): the
    /// tokens' k, q, v, g and beta in shared memory, filled 8 tokens at a time (5 CTAs per SM),
    /// instead of each lane holding its tokens' q and k rows in registers (4 CTAs per SM). The
    /// same bits; about 3% faster in these steps on GB10 but flat end to end (#836).
    #[arg(long, help_heading = HEADING)]
    gdn_replay_smem: bool,
    /// CUDA, compute capability 9.0+, gated DeltaNet layers in verify steps that run the deferred
    /// commit first, at one user and 8 tokens: the per-lane kernel instead of the default one for
    /// that shape (each CTA issues all its state, input and row-map loads at once and stages the
    /// tokens through shared memory, instead of a chain of round trips; the same bits, #689). For
    /// comparisons.
    #[arg(long, help_heading = HEADING)]
    no_gdn_replay_few: bool,
    /// CUDA, gated DeltaNet layers in MTP verify steps that run the deferred commit first
    /// (qwen4exp, with the MoE row map --no-moe-pad-remap leaves on): run the delta rule of a
    /// sequence's padding rows too, and store their inputs for the next commit, instead of the
    /// default (padding rows get zero rows and no stores; nothing reads either, and the real
    /// rows are the same bits). For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_gdn_pad_skip: bool,
    /// CUDA, gated DeltaNet layers, MTP replay commits: a commit graph per set of runs of slots
    /// keeping the same count, with a delta-rule launch and a conv window copy per run and
    /// layer, instead of the default one graph for every slot (one launch of each per layer,
    /// each slot's kept count a graph input; the same kernel code per slot, so the same
    /// states). For comparisons.
    #[arg(long, help_heading = HEADING)]
    gdn_commit_runs: bool,
    /// CUDA, gated DeltaNet layers, MTP on one stage (one GPU, no split over machines): each
    /// commit's delta rule runs inside the next verify step's (#836 part 14). The commit leaves
    /// the kept tokens' inputs where the verify step stored them (packed per token, no more
    /// memory than the default's, part 38); the next verify step's kernel advances each state by
    /// them, writes it, and computes its rows from it, so a round reads and writes each state
    /// once less. The same kernel code per state, so the same bits; any other use of a waiting
    /// state runs its commit first. On by default; this flag is still accepted.
    #[arg(long, help_heading = HEADING, conflicts_with = "no_gdn_commit_in_verify")]
    gdn_commit_in_verify: bool,
    /// Runs each MTP commit's delta rule in the commit (see --gdn-commit-in-verify), with the
    /// replay inputs in the pool's own tensors: the same bits. For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_gdn_commit_in_verify: bool,
    /// CUDA, gated DeltaNet layers, MTP verify steps that run the deferred commit
    /// (--gdn-commit-in-verify): a step whose tokens fit the packed replay rows after the kept
    /// ones advances the state by those in registers only, leaves the slot's state as it is and
    /// stores its own tokens after them, so the next step replays both runs from that state
    /// (token by token as before: the same bits) and a state write of 3.1 MB per sequence and
    /// layer (Qwen3.8-Flash-Next) is saved. Off by default: the kept tokens are replayed twice,
    /// which on GB10 costs more than the write (decode 1-4% slower, #836).
    #[arg(long, help_heading = HEADING, conflicts_with = "no_gdn_state_defer")]
    gdn_state_defer: bool,
    /// Writes every verify step's states (see --gdn-state-defer, the default): the same bits.
    #[arg(long, help_heading = HEADING)]
    no_gdn_state_defer: bool,
    /// With --gdn-state-defer: packed replay rows per sequence slot beyond the verify width, so
    /// that the kept tokens of more rounds wait there before a step writes the state (0-32,
    /// default 8; each row takes ~1.5 MiB per slot on Qwen3.8-Flash-Next). 0 leaves the states
    /// only where a step is shorter than the widest one.
    #[arg(long, help_heading = HEADING, value_name = "ROWS",
          value_parser = clap::value_parser!(u8).range(0..=32))]
    gdn_state_defer_rows: Option<u8>,
    /// With --gdn-state-defer: only MTP verify steps of at least this many sequences leave the
    /// states; narrower steps write them as without the flag (the same bits either way). The
    /// second replay of the kept tokens costs little where a step's state traffic is the
    /// bottleneck (many sequences) but more than the write saves at one or a few (#858).
    /// 0-255, default 0 (every step may leave them).
    #[arg(long, help_heading = HEADING, value_name = "N")]
    gdn_state_defer_min_seqs: Option<u8>,
    /// CUDA, attention layers: Legion's one-launch norm + RoPE for the q and k heads (and
    /// Qwen3.8-Flash-Next's indexer queries) instead of ggml's rms_norm, mul and rope_multi.
    /// The same bits, but no faster on V100 (the fused external launch splits ggml's CUDA
    /// graph). Off by default.
    #[arg(long, help_heading = HEADING)]
    norm_rope: bool,
    /// CUDA, attention layers with F16 caches in VRAM: Legion's one-launch writers for the
    /// step's K and V cache rows (norm, rope, two set_rows), the indexer rows (fill, concat,
    /// set_rows) and Qwen3.8-Flash-Next's indexer pool keys (a nine-op chain) instead of
    /// ggml's nodes. Each kernel writes ggml's bits, but the graph's buffer layout moves and
    /// ggml-cuda picks some fusions by layout, so greedy output differs from the default.
    /// Decode about +2-5% on V100. Off by default.
    #[arg(long, help_heading = HEADING)]
    kv_append: bool,
    /// CUDA, Gemma 4: one positions view shared by every layer's RoPE, so ggml-cuda runs the
    /// q and k heads' norm, weight and RoPE as its one fused kernel instead of two. Prompts ~4%
    /// faster at 8K on a DGX Spark (GB10). The fused kernel rounds differently, so greedy
    /// output differs from the default. Off by default.
    #[arg(long, help_heading = HEADING)]
    head_rope_fused: bool,
    /// CUDA, steps of up to 3 tokens: two matmuls per layer each (DeltaNet alpha and beta; with
    /// --shexp-gate also the MoE router and the shared expert's gate) instead of one over both
    /// weights, which lie back to back in memory (the default: the same bits, one launch fewer
    /// per layer). For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_joined_matmuls: bool,
    /// CUDA GPUs with BF16 tensor cores, prompt steps of at least this many tokens: BF16 weight
    /// pairs that lie back to back in memory (the NVFP4 checkpoints' DeltaNet alpha and beta,
    /// a shared expert's gate and up) as one cuBLAS GEMM instead of two (one launch fewer, a
    /// taller GEMM that fills the GPU better; the same bits). Default 1024; 0 keeps two
    /// matmuls (for comparisons).
    #[arg(long, help_heading = HEADING, value_name = "TOKENS")]
    joined_prompt_min_tokens: Option<u32>,
    /// Those prompt steps (--joined-prompt-min-tokens): a DeltaNet layer's z gate projection on
    /// a GEMM of its own instead of one GEMM with alpha and beta, which an NVFP4 checkpoint
    /// stores right after it in BF16 (the default: one launch fewer per layer, z read strided
    /// in place by the gated norm; the same bits). For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_joined_prompt_z: bool,
    /// Those prompt steps (--joined-prompt-min-tokens): a full-attention layer's query + gate,
    /// key and value projections as three GEMMs instead of one, which they take where the
    /// checkpoint stores them back to back in BF16, as the NVFP4 ones do (the default: two
    /// launches fewer per layer, the short key and value GEMMs inside a tall one, their rows
    /// read strided in place; the same bits). For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_joined_prompt_qkv: bool,
    /// CUDA, compute capability 9.0+ (GB10, Hopper, RTX 50): BF16 matmuls of 2-16 tokens on
    /// ggml's kernel instead of Legion's dense matmul op (the default for weights of 16 MiB or
    /// more: ggml's bits, its weights read on the read-only path that ggml's sm_90+ builds give
    /// up). For comparisons.
    #[arg(long, help_heading = HEADING)]
    dense_mv_ggml: bool,
    /// Compute capability 12.x GPUs (GB10; no effect elsewhere), MTP draft passes of 9-16
    /// sequences with a Q8_0 draft LM head: the head on Legion's Q8_0 matvec op instead of
    /// ggml's mul_mat (MMQ). The same bits, but slower so far (~175 GB/s against ~191 on a
    /// GB10); for comparisons and further work.
    #[arg(long, help_heading = HEADING)]
    draft_head_q8_mv: bool,
    /// Compute capability 7.x GPUs (V100; no effect elsewhere), MTP draft passes of one
    /// sequence with a Q8_0 draft LM head: keep the head on ggml's mul_mat (quantize + mmvq)
    /// instead of Legion's one-column Q8_0 matvec op (the default there: ggml's bits, the head
    /// read with one warp per row and 16-byte loads). For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_draft_head_mv: bool,
    /// Where that op runs the head (default on V100): take the draft token and its probability
    /// with ggml's split argmax, soft_max and get_rows over the logits instead of from the
    /// head's own per-CTA partials plus one sum pass (the default: the same token and
    /// probability bits, ~16 µs less per draft pass). For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_draft_head_pick: bool,
    /// Compute capability 7.x GPUs (V100; no effect elsewhere), steps of one token: keep the
    /// one-column Q8_0 matvecs of the q8_group op (a DeltaNet layer's qkv and gate, an
    /// attention layer's Q, K and V, rows of 1,024 or 2,560 values) on its copy of ggml's mmvq
    /// (a 4-warp CTA a row) instead of the draft head's kernel (the default there: one warp a
    /// row, 16-byte loads, the same bits). For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_q8_group_row_mv: bool,
    /// GB10 (compute capability 12.1; no effect elsewhere), MTP draft passes of 1-8 sequences
    /// with a Q4_K draft LM head (`--spec-draft-lm-head q4`, the auto choice for qwen4exp's
    /// NVFP4 checkpoint there): keep the head on ggml's mul_mat (quantize + mmvq) instead of
    /// Legion's Q4_K head matvec op (the default there: ggml's bits, the head read with one warp
    /// per row and 16-byte loads). For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_draft_head_q4k_mv: bool,
    /// Compute capability 7.x GPUs (V100; no effect elsewhere), steps of one token: run the
    /// one-column Q8_0 projections of 6,144 values a row (an attention layer's output and a
    /// DeltaNet layer's ssm_out on Qwen3.8-Flash-Next) on the draft head's kernel (one warp a
    /// row, the input quantized once per CTA in shared memory, ggml's bits) instead of ggml's
    /// mul_mat. Off by default: 11-34% slower than ggml's mmvq on a V100 (#814). For research.
    #[arg(long, help_heading = HEADING)]
    q8_out_row_mv: bool,
    /// Where that op runs the head for 1-7 drafting sequences (default on GB10): take the draft
    /// tokens and their probabilities with the split argmax, ggml's soft_max and get_rows over
    /// the 65,536 logits instead of from the head's own per-CTA partials plus one sum pass (the
    /// default: the same token and probability bits). For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_draft_head_q4k_pick: bool,
    /// Keep that pick for one drafting sequence only: at 2-7 sequences take the split argmax,
    /// ggml's soft_max and get_rows (its cooperative kernel loops over the rows one after
    /// another) instead of every column's pick from the head's partials plus one sum pass, all
    /// columns at once (the default: the same token and probability bits). For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_draft_head_q4k_pick_cols: bool,
    /// Verify steps of 49-128 rows (7-16 sequences drafting 8 tokens) with a BF16 LM head
    /// (Gemma 4 31B on a GB10): keep the head on ggml's matmul (cuBLAS), soft-cap and argmax
    /// instead of Legion's head op (the default where it gives cuBLAS's bits: the matmul, the
    /// soft-cap and every row's argmax in one pass over the weights, the logits written once).
    /// For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_head_mm: bool,
    /// CUDA, compute capability 7.x-8.x (V100 to RTX 40), steps of 1-8 tokens inside stage
    /// graphs, with this flag an RTX 50 (12.0) too, without the mixer's parts (unmeasured there),
    /// and with this flag a GB10 at 2-8 tokens (without the mixer's parts; ggml's bits
    /// but flat decode there, #311 and #994, and at a verify of 8 tokens or fewer the DeltaNet
    /// pair's one launch moves the greedy text there): Q8_0 projections of one input (an
    /// attention layer's Q, K and V, a shared expert's gate and up with its SwiGLU, which also
    /// quantizes its output for the down projection) on Legion's op that quantizes the input
    /// once and runs them in one launch, a DeltaNet layer's qkv and gate as one launch each
    /// without the quantize launch, and qwen4exp's hyper-connection mixer writing that
    /// quantized input itself (which the routed experts' gate and up then read too), instead of
    /// ggml's quantize + matvec launch pair for each. On by default for qwen4exp
    /// (Qwen3.8-Flash-Next), with ggml's greedy text; this flag takes the op on every
    /// architecture (the DeltaNet pair as one launch where no mixer wrote the quantized input,
    /// which can move the text through ggml-cuda's layout-dependent MoE fusions).
    #[arg(long, help_heading = HEADING, conflicts_with = "no_q8_group")]
    q8_group: bool,
    /// Keeps ggml's quantize + matvec launch pair for every Q8_0 projection (see --q8-group).
    #[arg(long, help_heading = HEADING)]
    no_q8_group: bool,
    /// Diagnostics for --q8-group (#819): which of its parts run, a bit mask (1 DeltaNet qkv and
    /// gate, 2 DeltaNet beta and alpha in the same launch, 4 attention Q, K and V, 8 the shared
    /// expert, 16 the hyper-connection mixer's Q8_1 blocks, 32 the routed experts' gate and up
    /// reading them, 64 the shared expert's SwiGLU writing its Q8_1 blocks for the down
    /// projection), in place of the default's (127, on qwen4exp) or --q8-group's (all).
    #[arg(long, hide = true, value_name = "MASK")]
    q8_group_parts: Option<u8>,
    /// CUDA, compute capability 9.0+: the small BF16 matmuls of 2-16 tokens that Legion's dense
    /// matmul op reads faster than ggml's kernel (a shared expert's gate and up as one launch,
    /// the hyper-connection mixers' down projections; ~4-5 us a layer each on a GB10) on the op
    /// too: ggml's bits either way. On by default where the stage graphs are (see
    /// --stage-graphs), which capture the op with the rest: ~2.5 ms less GPU time per
    /// 16-sequence decode step of Qwen3.8-Flash-Next NVFP4 (#836). This flag turns it on without
    /// them too, where each op splits ggml's CUDA graph (gain within noise, #860).
    #[arg(long, help_heading = HEADING, conflicts_with = "no_dense_mv_small")]
    dense_mv_small: bool,
    /// Keep the small BF16 matmuls of --dense-mv-small on ggml's kernel even with the stage
    /// graphs on: the same bits, a little slower decode. For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_dense_mv_small: bool,
    /// CUDA: decode-sized selections on one CTA per row. By default, on compute capability 9.0+
    /// (GB10, Hopper, RTX 50), the QSA indexer's top-k (up to 4 rows of 8,192+ scores or 8 rows
    /// of 16,384+) and the greedy argmax over the vocabulary (up to 16 rows) run on a
    /// thread-block cluster of 8 CTAs a row; on older GPUs the argmax runs split over up to 64
    /// CTAs a row and joined in a second launch. The same outputs either way, ties included.
    /// For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_cluster_select: bool,
    /// CUDA, compute capability 9.0+ (GB10, Hopper, RTX 50): F32 matmuls of 4-16 tokens (the
    /// MoE router of a decode step) on Legion's dense matmul op instead of ggml's
    /// mul_mat_f<float>: ggml's bits (TF32 products in ggml's order) with the weights' loads in
    /// flight. GB10, Qwen3.8-Flash-Next at 16 sequences: 1.73 ms against 1.87 per step, decode
    /// tok/s within noise, so off by default. For tuning.
    #[arg(long, help_heading = HEADING)]
    dense_mv_f32: bool,
    /// Keep a GDN layer's alpha and beta projections (BF16, 48 rows each: NVFP4 checkpoints of
    /// Qwen3.8-Flash-Next) on ggml's cuBLAS GEMM in decode steps of 9-16 tokens. By default
    /// (CUDA, compute capability 9.0+) they ride along in the layer's qkv+z dense op, computed
    /// in cuBLAS's order: the same bits, without the two ~18 us latency-bound cuBLAS launches
    /// per layer (#836). For comparisons, and on Hopper (9.x), whose cuBLAS the order was not
    /// matched against.
    #[arg(long, help_heading = HEADING)]
    no_dense_mv_ab: bool,
    /// CUDA, compute capability 9.0+: a dense BF16 matmul of 2-16 tokens with fewer row tiles
    /// than the GPU has SMs (on a GB10, a hyper-connection mixer's down matrix: 10-20 tiles)
    /// spreads each tile's warps over a thread-block cluster of 2-8 CTAs instead of one CTA a
    /// tile: the same bits, the weights read on more SMs (#836). GB10, Qwen3.8-Flash-Next with
    /// MTP at one sequence: 35.2 against 37.6 us a launch, decode tok/s within noise, so off by
    /// default. For tuning.
    #[arg(long, help_heading = HEADING)]
    dense_mv_split: bool,
    /// CUDA, compute capability 9.0+: run every dense BF16 matmul (16 MiB and up) of 17-48
    /// tokens (MTP verify steps of 2-5 sequences, short prompt steps) on Legion's wide-chain
    /// kernel, in the order of cuBLAS's wmma kernels without split-K, instead of ggml-cuda's
    /// cuBLAS GEMM (#836). By default only the shapes and step sizes where cuBLAS gives the same
    /// bits take it (checked on a GB10; nowhere else); cuBLAS takes split-K kernels for the
    /// others, whose bits differ, so this flag can change greedy output. For tuning.
    #[arg(long, help_heading = HEADING, conflicts_with = "no_dense_mv_wide_chain")]
    dense_mv_wide_chain: bool,
    /// Keep every dense BF16 matmul of 17-48 tokens on ggml-cuda's cuBLAS GEMM. By default, on a
    /// GB10, those of the shapes and step sizes where cuBLAS runs without split-K take Legion's
    /// wide-chain kernel (see --dense-mv-wide-chain): the same bits, read faster (#836). For
    /// comparisons.
    #[arg(long, help_heading = HEADING)]
    no_dense_mv_wide_chain: bool,
    /// Keep a MoE layer's router in steps of 1-3 tokens (decode at 1-3 sequences, MTP verify
    /// steps at one sequence, draft passes) as three launches: ggml's F32 matmul, then the top-k
    /// and the weights ops. By default (CUDA, compute capability 9.0+) one kernel computes the
    /// logits as ggml's mul_mat_vec_f does, and its last CTA selects the experts as the two ops
    /// do: the same bits, two launches fewer per MoE layer, and on a GB10 the router's weight
    /// rows load while the kernel ahead still runs (#836). For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_router_mv: bool,
    /// Keep a DeltaNet layer's BF16 alpha and beta gate projections in steps of 8 tokens (a
    /// one-user verify step) as two ggml matmuls. By default (CUDA, compute capability 9.0+,
    /// stage graphs on) one launch computes both, each row as ggml's mul_mat_vec_f does: the
    /// same bits, one launch fewer per DeltaNet layer, and each thread's weight loads in flight
    /// at once (#689). For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_mvf_ab: bool,
    /// Run a MoE layer's expert selection as two CUDA launches, the top-k and then the
    /// weights. By default (CUDA, routers outside the one-op router above and without
    /// `--nvfp4-once`'s routing plan or `--decode-fuse`) one launch picks the experts and makes
    /// their weights from the logits it picked: the same bits, one launch fewer per MoE layer
    /// (#814). For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_router_join: bool,
    /// Keep reading the MoE routers' F32 weights in decode and verify steps (1-16 tokens). By
    /// default (CUDA, compute capability 9.0+) a router whose F32 values are all exact in BF16
    /// (a BF16 checkpoint's router, widened to F32 by the loader or the GGUF converter) gets a
    /// BF16 copy, which the 1-3 token router kernel and the 4-16 token F32 matmul read instead,
    /// widening each value back to its F32 bits: the same bits from half the bytes (2.6 instead
    /// of 5.2 MB per layer for a 512 x 2560 router; ~126 MB more GPU memory for a 49-router
    /// model). Routers that are not exact in BF16 keep the F32 reads either way (#836). For
    /// comparisons, or to save that memory.
    #[arg(long, help_heading = HEADING)]
    no_router_bf16: bool,
    /// Qwen3.8-Flash-Next with the n-gram table read from the SSD (direct reads) and a verify
    /// step that leaves its n-gram rows to its compute: issue the rows' reads only when the
    /// step writes them, after its first layers are launched. By default the step's input fill
    /// issues them, so they run through the step's launch and its first layers; the write then
    /// collects them. The same bytes either way (#836). For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_ple_early_reads: bool,
    /// CUDA, NVFP4 routed experts in GPU memory: build gate, up and their SwiGLU in decode and
    /// verify steps (up to 8 tokens) as the chain ggml-cuda fuses into one matvec launch over
    /// one quantized input, instead of two matvecs, a SiLU and a multiply. The same bits, about
    /// 140 launches fewer per step on a 48-layer model, but on a GB10 the fused kernel is
    /// slower than the two it replaces, so C=1 decode is flat (#836). Off by default.
    #[arg(long, help_heading = HEADING)]
    nvfp4_mmvq_glu: bool,
    /// CUDA, compute capability 8.0+, FP8 W8A16 weights (compressed-tensors' channel-wise FP8,
    /// e.g. unsloth's Qwen3.8-27B-NVFP4): matmuls of 5-9 tokens (a speculative verify step, a
    /// decode step of 5-9 sequences, a short prompt chunk) on the GEMV that reads each weight
    /// once for all tokens instead of the tensor-core GEMM. Every token then gets the bits it
    /// gets alone in a one-token decode step, so greedy text can differ from the default's where
    /// such steps occur. Faster verify steps with --spec mtp at depth 4+. Off by default.
    #[arg(long, help_heading = HEADING)]
    fp8_small_batch_gemv: bool,
    /// CUDA, compute capability 8.0+, FP8 W8A16 weights (as --fp8-small-batch-gemv): matmuls of
    /// 5-64 tokens (speculative verify steps, e.g. MTP or DFlash at 4-8 users; decode steps of
    /// that many sequences; short prompt chunks) on a sliced-K tensor-core GEMM (after NInfer:
    /// 16-32 weight rows per CTA, K split over its warps) that streams the weights near the GEMV's
    /// bandwidth instead of the prefill-shaped GEMMs. With --fp8-small-batch-gemv too, that
    /// keeps 5-9 tokens and this takes 10-64. The sums round in another order, so greedy text
    /// can differ from the default's where such steps occur. Off by default.
    #[arg(long, help_heading = HEADING)]
    fp8_sliced_k: bool,
    /// CUDA, compute capability 9.0 only (H100, H200, GH200) with a binary built for sm_90a
    /// (CMAKE_CUDA_ARCHITECTURES=90a): W8A8 FP8 prompt matmuls (more than 64 tokens) on the
    /// Hopper GEMM (warpgroup MMA fed by the TMA, a warp-specialized pipeline of 128 x 128 tiles,
    /// each 128-deep FP8 sum promoted to F32) instead of the mma.sync GEMM. UNTESTED ON HARDWARE:
    /// written and compiled without a Hopper GPU, never run or measured. The sums round in
    /// another order, so greedy text can differ from the default's. Elsewhere (other GPUs, a
    /// binary without sm_90a) it does nothing. Off by default.
    #[arg(long, help_heading = HEADING)]
    hopper_gemm: bool,
    /// CUDA, compute capability 9.0 only (H100, H200, GH200) with a binary built for sm_90a
    /// (CMAKE_CUDA_ARCHITECTURES=90a): paged attention prefill (prompt chunks and verify steps
    /// of more than one token per sequence, F16 KV cache, heads of 128 or 256) on a
    /// FlashAttention-3 style kernel (warpgroup MMA, K and V blocks loaded by the TMA, a producer
    /// warpgroup and two ping-ponged consumer warpgroups) instead of the mma.sync kernel. Ops it
    /// does not cover (quantized or compact caches, a host KV tier, other head sizes) keep the
    /// default kernels. UNTESTED ON HARDWARE: written and compiled without a Hopper GPU, never
    /// run or measured. The sums round in another order, so greedy text can differ from the
    /// default's. Elsewhere (other GPUs, a binary without sm_90a) it does nothing. Off by
    /// default.
    #[arg(long, help_heading = HEADING)]
    hopper_attn: bool,
    /// CUDA, compute capability 12.x (DGX Spark's GB10, RTX 50), paged attention with heads of
    /// 256 (Qwen3.5+ and Gemma 4's sliding layers) over an F16 cache: keep every step on the
    /// kernel that takes 8 query columns per warp. By default a step goes to the one that takes
    /// 16 query rows per warp with Q in shared memory when its grid costs less there, costed as
    /// waves of CTAs over the SMs; steps too small to fill the GPU (most verify steps, short
    /// chunks) stay on the old kernel. The same bits either way. For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_paged_prefill_qm: bool,
    /// CUDA, compute capability 9.0 only (H100, H200, GH200) with a binary built for sm_90
    /// or sm_90a: programmatic dependent launch (each kernel set up while the one ahead of it
    /// still runs, then waiting for it on the GPU) for Legion's decode kernels that take it on
    /// compute capability 12.x (the MoE router, the GLM and Qwen4 decode kernels), as on a DGX
    /// Spark. Each kernel still waits for the one ahead before it reads that one's output, so
    /// the results keep their bits. UNTESTED ON HARDWARE:
    /// never run or measured on a Hopper GPU. Elsewhere it does nothing. Off by default.
    #[arg(long, help_heading = HEADING)]
    hopper_pdl: bool,
    /// CUDA, compute capability 8.0+, dense NVFP4 weights on Legion's W4A16 kernels
    /// (--nvfp4-mm auto or legion; e.g. the MLPs of unsloth's Qwen3.8-27B-NVFP4): matmuls and
    /// fused SwiGLUs of 33-64 tokens (MTP or DFlash verify steps at 4-8 users, short prompt
    /// chunks) on the decode mv kernel, 64 tokens per CTA, instead of the GEMM, whose tiles read
    /// the weights at about half the bandwidth there. Each token then gets the bits the mv
    /// kernel gives it at 1-32 tokens, so greedy text can differ from the default's where such
    /// steps occur. Off by default.
    #[arg(long, help_heading = HEADING)]
    nvfp4_wide_mv: bool,
    /// CUDA, GB10 (sm_90+ code), NVFP4 routed experts at decode batches: launch the down mv as
    /// a plain kernel after the grouped SwiGLU mv instead of a programmatic dependent launch
    /// (the default), which finds its expert tile, copies its first weight stages and
    /// prefetches while the SwiGLU mv's last CTAs still run, and waits for them only before it
    /// reads their output. The same bits either way. For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_nvfp4_expert_pdl: bool,
    /// CUDA, NVFP4 routed experts at decode batches (one token per expert tile): prefetch each
    /// weight row into L2 a few stages ahead instead of whole row tiles (the default: each
    /// expert's rows fetched in long contiguous runs; GB10, Qwen3.8-Flash-Next at 16 sequences
    /// ~3% faster expert FFNs). The same bits either way. For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_nvfp4_expert_tile_prefetch: bool,
    /// CUDA, NVFP4 routed experts on Legion's W4A16 kernels: matmuls of up to this many (token,
    /// expert) pairs per used expert (if the routing were even) take the grouped mv kernel,
    /// more the grouped GEMM. Default 32: prompt steps of up to ~1,600 tokens on
    /// Qwen3.8-Flash-Next (10 of 512 experts per token) are on the mv kernel. A lower value puts
    /// shorter prompt steps on the GEMM, whose sums round in another order, so greedy text can
    /// differ from the default's where such steps occur (decode and verify steps, about one
    /// pair per expert, stay on the mv kernel at any value above 1). For A/B runs.
    #[arg(long, help_heading = HEADING, value_name = "PAIRS")]
    nvfp4_expert_mv_max: Option<u8>,
    /// CUDA, NVFP4 routed experts on Legion's grouped mv kernel (decode steps and short prompt
    /// steps): the widest token tile, 8 x this many tokens of one expert per CTA (1, 2 or 4;
    /// default 4). Experts with more pairs take more tiles, each reading the expert's weights
    /// again; narrower tiles do fewer multiplies on the columns an expert with few pairs leaves
    /// empty. A token's sums do not depend on its tile, so the output is the same bits at every
    /// value. For A/B runs.
    #[arg(long, help_heading = HEADING, value_name = "NT", value_parser = mv_tile)]
    nvfp4_expert_mv_tile: Option<u8>,
    /// CUDA, NVFP4 routed experts on Legion's W4A16 kernels: how --nvfp4-expert-mv-max's cutoff
    /// applies. `step` (default): a step's pairs per used expert (if the routing were even)
    /// put all its experts on one kernel. `expert`: prompt steps of 256+ tokens with up to 32
    /// pairs per expert run the experts with more pairs than the cutoff on the grouped GEMM and
    /// the others on the grouped mv kernel (two launches on one plan of the pairs). `down`: the
    /// cutoff for the down projections only; the fused gate/up keeps the default's kernel.
    /// `expert-down`: `expert` for the down projections only. Every mode but `step` can move
    /// greedy text where an expert moves to the GEMM (each expert's outputs are the bits of its
    /// own kernel); decode and verify steps keep the default's kernels. For A/B runs.
    #[arg(long, value_enum, help_heading = HEADING, value_name = "MODE")]
    nvfp4_expert_mv_split: Option<MvSplit>,
    /// CUDA, NVFP4 routed experts on Legion's W4A16 kernels, prompt steps of more than 64
    /// tokens: the down projection stores each (token, expert) row as BF16 instead of F32, and
    /// the weighted sum reads it back (on qwen4exp the hyper-connection norm the MoE tail folds
    /// into reads the BF16 rows itself). Half the bytes of those rows written and read: on a
    /// DGX Spark (GB10), Qwen3.8-Flash-Next NVFP4 prefill is a few percent faster. Each row is
    /// rounded to BF16 before the sum, so output moves (opt-in); decode and verify steps, and
    /// steps that batch a prompt with them, keep F32 rows.
    #[arg(long, help_heading = HEADING)]
    nvfp4_expert_rows_bf16: bool,
    /// qwen4exp speculative verify steps with NVFP4 routed experts on Legion's kernels: read
    /// the experts of the padding rows too (the rows after a sequence's drafts that fill every
    /// sequence up to the step's shape; nothing reads their output), instead of routing their
    /// expert work as the sequence's first row (the default: the experts of the real rows only,
    /// and the same output, since every real row's sums are the same instructions on the same
    /// values). For comparisons. Also turns off `--moe-pad-remap-tiered`.
    #[arg(long, help_heading = HEADING)]
    no_moe_pad_remap: bool,
    /// qwen4exp speculative verify steps with GGUF experts behind an expert cache on Legion's
    /// expert matvec (an AC922): route the padding rows' expert work as their sequences' first
    /// rows there too, so the cache looks up, logs and learns only the real rows' experts and
    /// the matvec reads only theirs. Off by default: on 4x V100 with MTP k=2 it cut the decode
    /// expert bytes by ~1.4% with no measurable speed change, and greedy output is not
    /// byte-identical to the default. For experiments with more drafts per step.
    #[arg(long, help_heading = HEADING)]
    moe_pad_remap_tiered: bool,
    /// A diagnostic of --moe-pad-remap-tiered: which uses of the expert cache's lookup take the
    /// remapped ids. both: the remap as it is; kernels: the expert matvec reads them, the cache
    /// logs and learns the router's own ids; log: the other way round. Each probe adds one table
    /// lookup per MoE layer. Only with --moe-pad-remap-tiered.
    #[arg(long, value_enum, help_heading = HEADING)]
    moe_pad_remap_probe: Option<Probe>,
    /// qwen4exp speculative verify steps with GGUF experts behind an expert cache on Legion's
    /// expert matvec (an AC922): hand the matvec the step's row map, so it skips the padding
    /// rows' pairs of experts that no real row chose (zero results; experts only padding rows
    /// chose are never read) and computes every expert a real row chose as before, for all its
    /// pairs, with the router's ids and the cache's lookup unchanged. Off by default: with MTP
    /// k=2 on 4x V100 it changed no speed, and the greedy text differed from the default's
    /// (the extra source moves ggml's graph layout; see --moe-pad-skip-layout-only).
    #[arg(long, help_heading = HEADING)]
    moe_pad_skip_tiered: bool,
    /// A diagnostic of --moe-pad-skip-tiered: the expert ops keep the row map as a source (the
    /// same graph and layout) but skip nothing.
    #[arg(long, help_heading = HEADING)]
    moe_pad_skip_layout_only: bool,
    /// qwen4exp speculative verify steps with NVFP4 routed experts on Legion's decode mv
    /// kernels: route the padding rows' expert work as their sequences' first rows (as
    /// --no-moe-pad-remap leaves it on) instead of skipping it (the default: padding rows get
    /// zeros from the routed experts and take no room in an expert's tiles, so an expert of a
    /// first row that many padding rows used to share is read once; the real rows' output is the
    /// same bits). For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_moe_pad_skip: bool,
    /// --tp-moe-ep with an NVFP4 checkpoint's experts (per-expert scales) on CUDA: compute the
    /// other ranks' (token, expert) pairs on this rank's first expert with a zero router
    /// weight, as before, instead of skipping them in Legion's grouped NVFP4 kernels (the
    /// default: each rank computes only its own experts' pairs, about half the expert work at
    /// prompt steps on two ranks, and the others' rows are zeros; the same output bits). For
    /// comparisons.
    #[arg(long, help_heading = HEADING)]
    no_tp_moe_ep_skip: bool,
    /// CUDA, compute capability 8.0+ (GB10, Ampere and newer; ignored with a note elsewhere):
    /// the paged attention of groups of 2-16 tokens per sequence (speculative verify steps, MTP
    /// or DFlash, at 1-4 users; short prompt chunks) split over the key blocks as one-token
    /// decode steps are (flash-decoding: up to 32 splits per sequence, ~512 CTAs, then a combine
    /// kernel), instead of one CTA per kv head walking the whole context. Long contexts gain
    /// most (GB10, Qwen3.8-27B at 64K: a verify step's attention ran on 4 CTAs of 48 SMs). The
    /// rows' sums are taken in another order, so greedy text can differ from the default's.
    /// Not used over a pool with a host tier. Off by default.
    #[arg(long, help_heading = HEADING)]
    paged_verify_split: bool,
    /// CUDA, compute capability 7.0 / 7.5 (V100, Turing): split the paged attention of groups
    /// of 2-16 tokens per sequence (speculative verify steps, short prompt chunks) over the key
    /// blocks across the GPU, at up to 32 CTAs of the WMMA kernel (one per query tile and kv
    /// head, each walking the whole context twice). The attention's output bits are the same,
    /// up to 11x faster at a 64K context, but slower at a short one with several sequences
    /// (more launches). With --spec, cheaper verify steps change how much the adaptive draft
    /// length drafts, and with it which steps compute each token, so greedy text can differ
    /// from the default's. Off by default.
    #[arg(long, help_heading = HEADING)]
    paged_verify_split_wmma: bool,
    /// CUDA, MoE layers: the shared expert's gate (sigmoid, scaling and sum) as one op instead of
    /// ggml's three, and on steps of up to 3 tokens the router's and the gate's logits from one
    /// matmul (one matmul and two kernels fewer per layer). Each op gives ggml's bits, but the
    /// graph's memory layout moves, and ggml-cuda's fused MoE weighted sum rounds by layout
    /// (#410): speculative (MTP) outputs can move unless --moe-combine is on too. Off by default.
    #[arg(long, help_heading = HEADING)]
    shexp_gate: bool,
    /// CUDA, Qwen3.8-Flash-Next, steps of up to 64 tokens (decode and MTP verify): fewer, larger
    /// kernels in place of ggml's small ops. Each MoE layer's tail (the router weights, the
    /// weighted sum of the routed experts and the gated shared expert) runs as one kernel, the
    /// router's and the shared expert's gate logits come from one matmul (up to 3 tokens), and
    /// each gated DeltaNet layer's output norm and gate run as one op (about 7 kernels fewer
    /// per layer). The sums round as with --moe-combine --shexp-gate --gated-norm (ggml-cuda's
    /// fused rounding in every layer), so greedy text can differ from the default's. Prompt
    /// steps and other models are unchanged. Off by default.
    #[arg(long, help_heading = HEADING)]
    decode_fuse: bool,
    /// CUDA: ggml's own small kernels instead of Legion's ops taking over their chains after
    /// the graph is allocated (the default: the same bits, ggml's arithmetic). The ops: each
    /// MoE layer's routed-expert weighted sum where ggml-cuda runs it unfused (a mul and an add
    /// per expert), with the shared expert's sigmoid gate; the gate alone behind the sums
    /// ggml-cuda fuses; the expert cache's id remap (cont and get_rows). About 10 kernels fewer
    /// per MoE layer and step. For comparisons.
    #[arg(long, help_heading = HEADING)]
    node_fuse_ggml: bool,
    /// CUDA fused MoE op (--moe-kernel): how each (token, expert) output is kept between the
    /// down GEMM and the per-token sum. `f32` (default); `f16` / `bf16` halve the op's largest
    /// memory stream (GB10, Qwen3.8-Flash-Next: prefill +2-2.5%) but move the output (greedy
    /// chats diverge after a few hundred characters). Replaces LEGION_MOE_PAIR_OUT.
    #[arg(long, value_enum, help_heading = HEADING)]
    moe_pair_out: Option<PairOut>,
    /// CUDA fused MoE op (--moe-kernel) on compute capability 8.0+: the GEMM that loads each K
    /// step's weight bytes straight from global memory instead of copying row blocks into
    /// shared memory with cp.async first (the default for Q4_K, Q5_1, IQ2_S and IQ2_XXS weights,
    /// and on the GB10 the fragment kernels for Q4_K gate/up and Q5_1 down). The same output
    /// bits. For comparisons. Replaces LEGION_MOE_RAW=0.
    #[arg(long, help_heading = HEADING)]
    no_moe_raw_staging: bool,
    /// CUDA fused MoE op (--moe-kernel) on the GB10 (compute capability 12.x), prompt steps whose
    /// down GEMM runs the fragment kernel (Q5_1 / Q8_0 down, F32 pair outputs, no --hc-fuse on
    /// the layer): the down GEMM makes the per-token sums itself instead of the separate sum
    /// kernel (#826). The block that finishes a token's last pair sums the token's pairs in the
    /// same order, so the same output bits. Off by default: on Qwen3.8-Flash-Next prefill moved
    /// within noise (8K -0.2%, 32K +2.2% in one session).
    #[arg(long, help_heading = HEADING)]
    moe_down_sum: bool,
    /// Decode expert matvec (--expert-mv legion): how experts in VRAM cache slots are read: 0
    /// (default) through tiles as host experts are, 1 by a kernel reading them in place as
    /// ggml's mmvq does, 2 that kernel for gate/up only. 1 and 2 are not byte-identical to 0,
    /// and greedy output may then differ from run to run (the sums' order follows the cache's
    /// residency). For tuning. Replaces LEGION_EXPERT_MV_VRAM.
    #[arg(long, help_heading = HEADING, value_name = "MODE", value_parser = clap::value_parser!(u8).range(0..=2))]
    expert_mv_vram: Option<u8>,
    /// Decode expert matvec over an expert cache (--experts cache): experts in VRAM (cache
    /// slots, and with --expert-miss-overlap / --expert-peer-fetch the staged misses) are read
    /// in place with all of a lane's weight loads issued before its first dot product, instead
    /// of through the kernel's shared-memory tile (adapted from Strata-AC922's S28 decode
    /// kernels); experts several tokens of a step share, Q8_0 ones, and steps of more than 2
    /// tokens (speculative verify) keep the tile. The same lanes and summation order, so the
    /// same output bits. Off by default.
    #[arg(long, help_heading = HEADING)]
    expert_mv_prefetch: bool,
    /// Decode expert matvec over an expert cache (--experts cache): how many CTAs of the tile
    /// kernel read the cache misses that are read in place over the host link (0, the default:
    /// off; 64 was tried). They are dispatched ahead of the other CTAs, so the link reads start
    /// first from a few SM slots, and the cached experts take the rest of the GPU meanwhile.
    /// Experimental (#710): on 2 V100s with MTP, 64 decoded greedy chats about 5% slower, and
    /// their text differed from the default's (each tile is computed by the same code; MTP's
    /// adaptive depth follows timing, the likely cause).
    #[arg(long, help_heading = HEADING, value_name = "CTAS", value_parser = clap::value_parser!(u16))]
    expert_mv_link_ctas: Option<u16>,
    /// Decode expert matvec over an expert cache (--experts cache): experts in VRAM cache slots
    /// of Q4_K gate/up (SwiGLU op) and Q5_1 down (Qwen3.8-Flash-Next) are read in place by
    /// hand-written sm_70 PTX kernels (src/cuda/ptx, JIT-loaded at the first launch) instead of
    /// the nvcc tile kernel. The same lanes and summation order, so the same output bits; other
    /// formats, host and staged experts keep the tile kernel. Off by default; a pilot (#703)
    /// for A/B runs.
    #[arg(long, help_heading = HEADING)]
    expert_mv_ptx: bool,
    /// With --expert-mv-ptx: the PTX kernel runs on a side stream beside the tile kernel's host
    /// and staged experts instead of after them (#710). The same output bits; two more events
    /// per op. Off by default.
    #[arg(long, help_heading = HEADING, requires = "expert_mv_ptx")]
    expert_mv_ptx_side: bool,
    /// Decode expert matvec: experts in GPU memory (not the expert cache's) take the SwiGLU op
    /// in verify steps of 2-8 tokens, one launch for gate and up with each expert read once
    /// (ggml's bits on Q4_K per layer). Not byte-identical end to end: greedy text can differ
    /// from the first multi-token verify step, as the new graph layout can flip ggml-cuda's
    /// layout-dependent fusions (#410, #472). About 2.7% shorter verify step at a fixed MTP k=2
    /// on a GB10, a tie at k=4; plain decoding is unchanged (see docs/tuning-integrated-gpus.md).
    /// Off by default. Replaces LEGION_EXPERT_MV_DEVICE=1.
    #[arg(long, help_heading = HEADING)]
    expert_mv_device: bool,
    /// Decode expert matvec: the most tokens of a step it takes (default 8; it computes up to
    /// 64 exactly). 64 halved short prompts' time to the first token on an AC922 but slowed
    /// 16-sequence decode batches. For A/B runs. Replaces LEGION_EXPERT_MV_MAX_TOKENS.
    #[arg(long, help_heading = HEADING, value_name = "TOKENS", value_parser = clap::value_parser!(u8).range(1..=64))]
    expert_mv_max_tokens: Option<u8>,
    /// Decode expert matvec: gate and up as two ops instead of the SwiGLU op (the default). For
    /// comparisons. Replaces LEGION_EXPERT_MV_GLU=0.
    #[arg(long, help_heading = HEADING)]
    expert_mv_split_glu: bool,
    /// Decode expert matvec (--expert-mv legion), tile kernel: how a tile's rows are dotted with
    /// the users of its expert. `auto` (default): each weight unit decoded once for all users
    /// where that was faster on V100 (Q4_K/Q5_K gate+up always, Q5_1 from 3 tokens a step,
    /// Q8_0 at 3-4), else one ggml vec_dot per user; `per-user` always the latter (the kernel
    /// before #288); `once` always the former; `sub16` that with 16 lanes per 32-lane row, two
    /// rows per warp (slower on V100). All give the same bits. For comparisons.
    #[arg(long, value_enum, help_heading = HEADING, value_name = "MODE")]
    expert_mv_dot: Option<Dot>,
    /// Decode expert matvec (--expert-mv legion), tile kernel: each CTA takes this many
    /// consecutive tiles of its expert and loads the next one while it computes the current one
    /// (software pipelined), so HBM keeps streaming while an SM's CTAs compute. 0: one tile per
    /// CTA, loaded and then computed (the kernel before #816). Default: chosen per op and step
    /// size (Qwen3.8-Flash-Next's gate/up Q4_K and down Q5_1 at 1-4 tokens a step, its MTP
    /// layer's Q8_0 at 1-3; 0 for other types and over 4 tokens). Every tile is computed by the same code from the same
    /// shared-memory layout, so the same output bits at any value. Ignored with
    /// --expert-mv-prefetch. For tuning and comparisons.
    #[arg(long, help_heading = HEADING, value_name = "TILES", value_parser = clap::value_parser!(u16).range(0..=1024))]
    expert_mv_pipe: Option<u16>,
    /// Decode expert matvec (--expert-mv legion), Q4_K gate/up at 3-4 tokens a step (MTP verify
    /// steps; Qwen3.8-Flash-Next): the activation quantizer also writes each weight unit's
    /// activation ints, scales and quant sums side by side, and the tile kernel reads them in two
    /// 16-byte loads per sequence and unit instead of six (#816). Measured on a V100: the op
    /// 13-20% faster at 4 tokens, unchanged at 3. The same bits per op; the bigger activation
    /// scratch moves the step graph's memory layout, which can move greedy text through
    /// ggml-cuda's layout-dependent fusions (identical with --deterministic --stable-fusions).
    /// Off by default.
    #[arg(long, help_heading = HEADING)]
    expert_mv_q4k_units: bool,
    /// Turns off the default activation hold of the pipelined decode expert matvec for Q4_K
    /// gate/up at 3-4 tokens a step (Qwen3.8-Flash-Next's MTP verify steps), where each lane
    /// keeps the activations of its first weight units in registers over the CTA's tiles instead
    /// of reloading them for every tile (a kernel compiled for 3 CTAs per SM). On a V100 the hold
    /// makes that op 9% faster at 3 tokens and 11% at 4. The same output bits on or off; for
    /// comparisons. --expert-mv-q4k-units replaces the hold when given.
    #[arg(long, help_heading = HEADING)]
    no_expert_mv_q4k_hold: bool,
    /// Turns off the default work balance of the pipelined decode expert matvec (#814) for the
    /// Q4_K gate/up at one token: the step's tiles spread evenly over as many CTAs as the GPU holds
    /// at once, each CTA going through its share in order with the next tile's loads in flight,
    /// instead of one CTA per tile (~12% faster on a V100). The same output bits on or off; for
    /// comparisons.
    #[arg(long, help_heading = HEADING)]
    no_expert_mv_balance: bool,
    /// Turns off the default half tiles of the pipelined decode expert matvec (#814) for the Q5_1
    /// down projection at 1-3 tokens a step: tiles of half the rows in twice as many CTAs, each CTA
    /// with the same chain of --expert-mv-pipe tiles at 2-3 tokens and 8 half tiles at one token
    /// (on a V100 ~11% faster at 1 token, ~19% at 2, ~3% at 3). The same output bits on or off;
    /// for comparisons.
    #[arg(long, help_heading = HEADING)]
    no_expert_mv_half_tiles: bool,
    /// Decode expert matvec over an expert cache (--experts cache): each MoE layer's experts
    /// missing from the cache are copied over the host link into a VRAM staging buffer by a
    /// copy kernel on a second CUDA stream while the cached experts are computed, then computed
    /// from VRAM (adapted from Strata-AC922). The same output bits for the same cache contents.
    /// The buffer (16 experts' bytes, ~50 MB on Qwen3.8-Flash-Next) is part of each 1-8 token
    /// step's compute buffer, inside the 1 GiB the cache leaves free. Off by default (no decode
    /// gain on V100 over NVLink2), except on GLM-5.3-Flash where --expert-peer-fetch is on by
    /// its automatic choice (with the peer, C=4 decode +10%).
    #[arg(long, help_heading = HEADING, conflicts_with = "no_expert_miss_overlap")]
    expert_miss_overlap: bool,
    /// Turns the automatic --expert-miss-overlap off (the same output bits either way). For
    /// comparisons.
    #[arg(long, help_heading = HEADING)]
    no_expert_miss_overlap: bool,
    /// Decode over an expert cache (--experts cache) on a layer split over GPUs paired by NVLink
    /// on one host NUMA node (an AC922's 0-1, 2-3): while one GPU runs its layers its idle peer
    /// copies half of each MoE layer's missed experts from host memory into the running GPU's
    /// VRAM over its own host link and the pair's NVLink, so the busy GPU's link carries half
    /// the bytes (adapted from Strata-AC922). With --expert-miss-overlap the running GPU's side
    /// stream copies the other half; without it the running GPU reads them in place. The same
    /// output bits for the same cache contents and the same step graph layout, so the same
    /// greedy text; prompt steps and steps where both GPUs work run as without it. Takes a
    /// ~25 MB VRAM buffer per GPU (with --expert-miss-overlap ~50 MB of each step's compute
    /// buffer instead) and a stream on each peer. Automatic by default: on where it was measured
    /// to help, a layer split over exactly the two GPUs of one NVLink pair on an AC922 (POWER9 +
    /// V100, ATS; decode C=1/4/8 +11/15/13%, prefill and C=16 a tie), and for GLM-5.3-Flash on
    /// any split whose GPUs all pair up (4 GPUs: decode +13%, C=4 +34% with
    /// --expert-miss-overlap); off elsewhere (Qwen on 4 GPUs: no gain). This flag forces it on
    /// elsewhere.
    #[arg(long, help_heading = HEADING, conflicts_with = "no_expert_peer_fetch")]
    expert_peer_fetch: bool,
    /// Turns the automatic --expert-peer-fetch off (the same output bits either way). For
    /// comparisons.
    #[arg(long, help_heading = HEADING)]
    no_expert_peer_fetch: bool,
    /// --expert-peer-fetch without --expert-miss-overlap: the idle peer stages TAKE of every
    /// EVERY of a layer's missed experts (in the order of their first use, at most TAKE/EVERY of
    /// 16 a layer), the running GPU reads the rest in place over its own host link. Default 1/2
    /// (Strata-AC922's every other one). Every pair is computed by the same code from the same
    /// bytes, so every share gives the same output bits. For tuning.
    #[arg(long, help_heading = HEADING, value_name = "TAKE/EVERY")]
    expert_peer_share: Option<legion::model::core::expert_mv::peer::Share>,
    /// --expert-peer-fetch: CTAs of the peer's copy kernel (default 24). The same output bits at
    /// any value. For tuning.
    #[arg(long, help_heading = HEADING, value_name = "CTAS", value_parser = clap::value_parser!(u16).range(1..=1024))]
    expert_peer_ctas: Option<u16>,
    /// qwen4exp prompt steps on Volta (every CUDA device compute capability 7.x; ignored
    /// otherwise, with a note): the sparse-attention indexer's pool scores on the tensor cores,
    /// FP32 split into F16 high and low parts (adapted from Strata-AC922). FP32-level, not
    /// ggml's bits: the pools selected may shift, so opt-in until measured. Replaces
    /// LEGION_QSA_SCORES_TC=1.
    #[arg(long, help_heading = HEADING)]
    qsa_scores_tc: bool,
    /// qwen4exp prompt steps, Volta sparse-attention kernel (sm_70/75; sm_80+ GPUs run the
    /// mma.m16n8k16 kernel, which has none, and say so once): the hi and lo m8n8k4 products
    /// accumulate in chains across K steps (adapted from Strata-AC922), 17% faster at 32K. Opt-in:
    /// it changes the sums' rounding. Replaces LEGION_QSA_ATTN_CHAIN=1.
    #[arg(long, help_heading = HEADING)]
    qsa_attn_chain: bool,
    /// qwen4exp, CUDA paged sparse attention: score every visible pool and take the indexer's
    /// top_k even in steps where no token sees more pools than it picks (contexts up to its
    /// top_k positions, 2048 on Qwen3.8-Flash-Next), instead of selecting them all with no
    /// indexer query, scores or top_k (the default: the same selection). For comparisons.
    #[arg(long, help_heading = HEADING)]
    qsa_always_score: bool,
    /// qwen4exp, CUDA: steps of up to 8 tokens build the hyper-connection mixer from ggml's
    /// nodes instead of Legion's one op (the default; the two agree to rounding). For
    /// comparisons. Replaces LEGION_QWEN4_HC=0.
    #[arg(long, help_heading = HEADING)]
    hc_mix_ggml: bool,
    /// qwen4exp, CUDA: steps of up to 8 tokens also run the hyper-connection mixer as Legion's
    /// one op when its down and up matrices are BF16 (Qwen3.8-Flash-Next-NVFP4 ships them so),
    /// instead of ggml's dozen nodes (the default). The op sums in another order than ggml's
    /// matvecs, so greedy output changes by rounding. On a GB10 the op loads its weights before
    /// waiting on the previous kernel: C=1 decode 19.1 -> 19.7 tok/s (+3.1%), C=4 per request
    /// 13.4 -> 13.6, prefill unchanged (#748). Q8_0 mixers (the GGUF) take the op either way.
    #[arg(long, help_heading = HEADING)]
    hc_mix_bf16: bool,
    /// qwen4exp, CUDA, the hyper-connection mixer op's V100 Q8_0 kernels: which weights the
    /// kernel before their reader fetches into L2 while it runs, so the reader finds them there
    /// instead of in DRAM. `down`: extra CTAs of the norm read the down matrix (3.5 MB on
    /// Qwen3.8-Flash-Next); `up`: extra CTAs of the down kernel read the up matrix; `both`;
    /// `off` (default): no reads ahead. The reads only warm L2, so every mode gives the same
    /// bits. No effect on the GB10 or BF16 mixers.
    #[arg(long, value_enum, help_heading = HEADING, value_name = "MODE")]
    hc_l2_prefetch: Option<L2Prefetch>,
    /// qwen4exp, CUDA, compute capability 7.x (V100): turns off the default norm of the
    /// hyper-connection mixer op that loads gamma and the stream once, into registers, before
    /// the RMS (#814; ~2.3 us less per mixer, 6-7% of the mixer at 1-4 tokens). The same output
    /// bits on or off; for comparisons. No effect on other GPUs.
    #[arg(long, help_heading = HEADING)]
    no_hc_norm_regs: bool,
    /// qwen4exp, CUDA, compute capability 7.x (V100): turns off the default down kernel of the
    /// hyper-connection mixer op that splits each down row over twice the threads at 1-2 tokens
    /// a step (#814; 3-5% of the mixer). The same output bits on or off; for comparisons. No
    /// effect on other GPUs or at 3+ tokens.
    #[arg(long, help_heading = HEADING)]
    no_hc_down_split: bool,
    /// qwen4exp, CUDA, compute capability 7.x (V100): turns off the default down kernel of the
    /// hyper-connection mixer op with its per-unit address math out of the loop (#814; 3-8% of
    /// the mixer at 1-4 tokens, and it takes --no-hc-down-split's kernel's place: with this flag
    /// that kernel runs at 1-2 tokens again). The same output bits on or off; for comparisons. No
    /// effect on other GPUs.
    #[arg(long, help_heading = HEADING)]
    no_hc_down_hoist: bool,
    /// qwen4exp, CUDA, compute capability 7.x (V100): turns off the default up kernel of the
    /// hyper-connection mixer op that stages each warp's up rows in shared memory with 16-byte
    /// loads (#814; 2-3% of the mixer at 1-4 tokens). The same output bits on or off; for
    /// comparisons. No effect on other GPUs or above 4 tokens.
    #[arg(long, help_heading = HEADING)]
    no_hc_up_stage: bool,
    /// qwen4exp, CUDA: ggml's hyper-connection combine and a separate norm instead of Legion's
    /// combine-and-norm op (the default). For comparisons. Replaces LEGION_QWEN4_HC_POST_NORM=0.
    #[arg(long, help_heading = HEADING)]
    hc_post_norm_ggml: bool,
    /// qwen4exp prompt steps on an sm_80+ GPU with BF16 mixers (the NVFP4 checkpoints), CUDA:
    /// the next hyper-connection combine-and-norm makes each MoE layer's tail (the routed
    /// experts' weighted sum and the shared expert's gate) from the tail's inputs, rounded as
    /// the tail's kernels round it, and the tail's kernels go (#836: the same bits, the layer's
    /// output no longer written and read back). The default on an integrated GPU such as the
    /// GB10 (prompt steps about 1% faster there); this turns it on elsewhere. Prompt graphs are
    /// planned twice. No effect with --node-fuse-ggml or on steps under 256 tokens.
    #[arg(long, help_heading = HEADING, conflicts_with = "no_hc_moe_fold")]
    hc_moe_fold: bool,
    /// Keeps each MoE layer's tail as its own kernels (see --hc-moe-fold; the default except on
    /// an integrated GPU such as the GB10). For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_hc_moe_fold: bool,
    /// qwen4exp prompt steps where --hc-moe-fold folds the MoE tails (the default on a GB10):
    /// each layer's FFN-side hyper-connection combine-and-norm writes the new residual streams
    /// (4 x n_embd F32 values a token) for that mixer's up-mix and the next layer's norm to
    /// read, and not, as by default, both of them making those streams per value from the
    /// streams before it and the attention output, with the norm's own arithmetic (#1183: the
    /// same bits, ~84 MB less memory traffic per layer at 4K tokens; the attention mixer's
    /// streams then stay allocated through the MoE layer). For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_hc_lazy_streams: bool,
    /// Tensor parallelism with --tp-seq-par and --tp-wire bf16 (two ranks' prompt steps): the
    /// hyper-connection mixer writes the block input it makes as F32 and the all-gather rounds
    /// it into its send buffer, and not, as by default, the all-gather running that mixer
    /// with its values rounded straight into the send buffer (#1176: the same bits, one pass
    /// over the block input less before each gather). For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_tp_gather_mix: bool,
    /// Tensor parallelism with --tp-seq-par and --tp-wire bf16: the F32 copy of a gathered
    /// BF16 block input (for its F32 readers, the MoE router) is made by a copy after the
    /// all-gather, and not, as by default, by the all-gather itself, this rank's share while
    /// the other rank's is still on the wire (#1176: the same values). For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_tp_gather_widen: bool,
    /// qwen4exp prompt steps on a V100 (compute capability 7.0) or an sm_80+ GPU such as the
    /// GB10 (#826; no effect elsewhere): the layer's gated shared expert is added in the fused
    /// MoE op's per-token sum, and the next hyper-connection combine-and-norm makes that sum
    /// from the op's per-expert rows itself instead of the op writing it and the norm reading
    /// it back (#774). The same bits per op as without; the graph's layout changes, which can
    /// move greedy output through ggml-cuda's layout-dependent fusions (identical with
    /// --deterministic --stable-fusions; on a GB10 the default text was unchanged). Only steps
    /// the fused MoE op takes (64+ tokens); decode steps keep the default graph. On a GB10 the
    /// fused kernel saves ~2% of the two kernels' time, within prefill noise.
    #[arg(long, help_heading = HEADING)]
    hc_fuse: bool,
    /// qwen4exp prompt steps of 64+ tokens on a V100 (compute capability 7.0; no effect
    /// elsewhere): the hyper-connection mixer's down projection reads xn as F32 and its Q8_0
    /// weight, and ggml-cuda converts both to F16 inside the matmul, instead of the
    /// combine-and-norm op also writing xn and the weight as F16 for it (the default: the same
    /// GEMM on the same F16 values, without a separate conversion of xn per mixer). With BF16
    /// mixer weights (the NVFP4 checkpoints) on sm_80+ GPUs, prompt steps of 128+ tokens: ggml-cuda
    /// converts xn to BF16 inside the matmul instead of the op also writing it as BF16. The same
    /// bits either way. For comparisons.
    #[arg(long, help_heading = HEADING)]
    hc_xn_f32: bool,
    /// qwen4exp prompt steps of 256+ tokens on compute capability 12.x GPUs (GB10; no effect
    /// elsewhere): let
    /// ggml-cuda quantize the hyper-connection down matmul's input itself instead of the
    /// combine-and-norm op writing it quantized for ggml's MMQ as well (the default: the same
    /// bytes, so the same output, minus ggml's quantization pass). For comparisons.
    #[arg(long, help_heading = HEADING)]
    hc_quant_ggml: bool,
    /// Q4_K and Q5_1 expert matmuls of 9-63 tokens (multi-sequence decode, MTP verify) on
    /// compute capability 12.x GPUs (GB10) and V100s (no effect elsewhere): run them on Legion's
    /// expert-grid kernel instead of ggml's MMQ, also over a VRAM expert cache. The same bits
    /// as MMQ on that GPU (`legion mmq-ids-bench` checks them). Off by default.
    #[arg(long, help_heading = HEADING)]
    expert_mmq: bool,
    /// NVFP4 expert FFNs of 2-8 tokens (MTP verify steps) on compute capability 12.x (GB10;
    /// RTX 50, whose bits want a `legion nvfp4-once-bench` run first; no effect elsewhere):
    /// run Legion's ops that read each routed expert's weights once per step for all its tokens
    /// instead of ggml's MMVQ (once per token): gate and up with their per-expert scales and
    /// the SwiGLU in one op, then the down matmul. MMVQ's bits, with the scales too. Each
    /// expert's work picks its kernel body on the GPU from how many of the step's tokens chose
    /// it. Faster than ggml when a step's tokens share their experts (0.78-0.91x per layer at
    /// 3-8 tokens on 10 experts); with spread routing 0.90-0.99x at 5-8 tokens but up to 2%
    /// slower at 2-4 tokens, the size of a C=1 verify step (`legion nvfp4-once-bench`), so off
    /// by default.
    #[arg(long, help_heading = HEADING)]
    nvfp4_once: bool,
    /// The default since #1133 round 2; accepted for older command lines (see --no-nvfp4-mvq).
    #[arg(long, hide = true, conflicts_with = "no_nvfp4_mvq")]
    nvfp4_mvq: bool,
    /// NVFP4 dense matmuls of 5-8 tokens (verify steps, decode of 5-8 users) on compute capability 12.x
    /// (GB10, RTX 50; no effect elsewhere): keep them on ggml's MMVQ instead of Legion's op
    /// (the default: the sub-block sums on tensor cores, then ggml's float order, so the same
    /// bits as MMVQ; `legion nvfp4-mvq-bench` checks them and times both). The same output
    /// either way; for comparisons.
    #[arg(long, help_heading = HEADING)]
    no_nvfp4_mvq: bool,
    /// The default since #858; accepted for older command lines (see --q8-mmq-ggml).
    #[arg(long, hide = true, conflicts_with = "q8_mmq_ggml")]
    q8_mmq: bool,
    /// Q8_0 matmuls of 9-255 tokens (multi-sequence decode, MTP verify) on compute capability
    /// 12.x GPUs (GB10; no effect elsewhere) inside stage graphs: all on ggml's MMQ (quantize,
    /// tiles, stream-k fixup: three launches), instead of the shapes Legion's op runs faster
    /// (the default since #858: ggml's own tile code on ggml's grid, the split tiles' pieces
    /// added in ggml's fixup order, the input quantized once for the matmuls that read it;
    /// today a 10240-value weight of 320 rows at 9-16 tokens, `legion q8-mmq-check`) on the
    /// op. The same output either way; for comparisons.
    #[arg(long, help_heading = HEADING)]
    q8_mmq_ggml: bool,
    /// Q8_0 matmuls on Legion's op (see --q8-mmq-ggml): run the op's matmul on Legion's
    /// streaming kernel (weights copied by cp.async through a two-stage shared-memory ring, 8
    /// warps a CTA) instead of ggml's MMQ tile code, on ggml's grid and split, so the output is
    /// the same. Shapes it does not take (k not a multiple of 256, row strides not a multiple
    /// of 8 blocks) keep ggml's tile code. Slower than ggml's MMQ on the GB10 so far (`legion
    /// q8-mmq-check`), so off by default; for experiments.
    #[arg(long, help_heading = HEADING)]
    q8_mmq_stream: bool,
    /// Q8_0 matmuls on Legion's op (see --q8-mmq-ggml): how the op adds up the pieces of tiles
    /// split over several CTAs. `last`: the CTA that finishes a tile's last piece adds them
    /// inside the matmul kernel (two launches); `launch`: a fixup launch of its own after the
    /// matmul, ggml's stream-k fixup (three launches, the add-up spread over 4 CTAs per tile);
    /// `auto` (default): per weight shape and step size the faster of the two on the GB10
    /// (`legion q8-mmq-check --fixup`). All add the pieces in ggml's order, so the output is
    /// the same. For comparisons.
    #[arg(long, value_enum, help_heading = HEADING, value_name = "MODE")]
    q8_mmq_fixup: Option<legion::model::q8_mmq::Fixup>,
    /// Q4_K, Q5_K, Q6_K and Q8_0 prompt matmuls on Turing GPUs without tensor cores (GTX 16xx,
    /// MX450, MX550; no effect elsewhere): all on ggml's MMQ, whose int8 MMA these GPUs run
    /// without tensor cores, instead of Legion's dp4a op (the default: ggml's quantization,
    /// tile loads and per-output sums in ggml's order, taken where ggml runs every tile on one
    /// CTA). The same output either way; for comparisons.
    #[arg(long, help_heading = HEADING)]
    tu_mmq_ggml: bool,
    /// qwen4exp prompt steps on compute capability 12.x GPUs (GB10; no effect elsewhere): run
    /// the hyper-connection mixer's up matmul (ggml's MMQ, which writes the gate) and the gated
    /// mix (ggml's hc_pre, which reads it back; Legion's copy of it unless --hc-pre-ggml) as
    /// two ops instead of one kernel that
    /// keeps the gate in registers (the default where ggml's MMQ computes whole tiles: the same
    /// sums in the same order, so the same output). For comparisons.
    #[arg(long, help_heading = HEADING)]
    hc_up_mix_ggml: bool,
    /// qwen4exp prompt steps of 256+ tokens on compute capability 12.x GPUs (GB10) with BF16
    /// mixer weights (the NVFP4 checkpoints; no effect elsewhere): run the hyper-connection
    /// mixer's up matmul (ggml's cuBLAS BF16 GEMM, which writes the gate) and gated mix as two
    /// ops instead of one BF16 MMA kernel that keeps the gate in registers (the default: the
    /// GEMM's sums in its order and ggml's graph layout, so the same output; 8K/32K prefill
    /// +4-9% on one GB10, #836). For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_hc_up_mix_bf16: bool,
    /// qwen4exp prompt steps on compute capability 12.x GPUs (GB10) where the up-mix kernel
    /// runs, of 64+ tokens on a V100 (7.0) where the F16 mix of --hc-mix-f16 runs, and of 128+
    /// tokens on compute capability 8.0+ GPUs with BF16 mixer weights under --hc-mix-rebuild-bf16;
    /// no effect elsewhere: the combine-and-norm writes the normed streams (4 x n_embd floats a
    /// token) for that kernel (or ggml's hc_pre) to read, instead of each stream's norm scale with the kernel rebuilding the
    /// normed values from the streams (the default: the norm's own two multiplies, so the same
    /// output). For comparisons.
    #[arg(long, help_heading = HEADING)]
    hc_xn_written: bool,
    /// Opt-in, moves output. qwen4exp prompt steps of 256+ tokens on compute capability 12.x
    /// GPUs (GB10) with Q8_0 mixer weights (the GGUFs; no effect elsewhere or with
    /// --hc-xn-written / --hc-up-mix-ggml): keep the hyper-connection residual streams in BF16
    /// between the mixers instead of F32, so the combine-and-norm and the up-mix kernel move
    /// half the stream bytes. Each new stream value is rounded to BF16 (off by default).
    #[arg(long, help_heading = HEADING)]
    hc_streams_bf16: bool,
    /// Opt-in, moves output. qwen4exp passes where every sequence is in its prompt (steps of
    /// 256+ tokens), on compute capability 12.x GPUs (GB10, RTX 50, where the BF16 up-mix kernel
    /// runs) with BF16 mixer weights (the NVFP4 checkpoints; no effect elsewhere, with
    /// --hc-xn-written, or with --tp-seq-par splitting the step): keep the hyper-connection
    /// residual streams in BF16 between the mixers instead of F32, so the combine-and-norm (with
    /// the MoE output folded in) and the up-mix kernel move half the stream bytes. Math stays F32
    /// inside each kernel; each new stream value is rounded to BF16. In steps of 2048+ tokens
    /// the norm then skips writing its BF16 output and the mixer's down matmul rebuilds it from
    /// the streams (the same values and, on a GB10, cuBLAS's sums). Decode, verify and mixed
    /// steps keep F32 streams (off by default).
    #[arg(long, help_heading = HEADING)]
    hc_bf16: bool,
    /// qwen4exp prompt steps of 64+ tokens on a V100 (compute capability 7.0; no effect
    /// elsewhere): the hyper-connection mix as one op that also writes the block input as F16,
    /// which its Q8_0 projections (cuBLAS F16 GEMMs) then take with an F16 copy of their weight
    /// made by ggml's own converter, instead of ggml's gated hc_pre with ggml-cuda converting
    /// the input to F16 again inside each projection. The same GEMMs on the same F16 values
    /// (bit-exact under --deterministic --stable-fusions), but the smaller graph moves where
    /// ggml's allocator places tensors, which can flip ggml-cuda's MoE fusions in some layers
    /// and so change greedy output. Off by default (#774).
    #[arg(long, help_heading = HEADING)]
    hc_mix_f16: bool,
    /// Opt-in. qwen4exp prompt steps of 128+ tokens on compute capability 8.0+ GPUs with BF16
    /// mixer weights (the NVFP4 checkpoints; no effect elsewhere): the combine-and-norm writes
    /// each stream's norm scale and the BF16 normed streams for the down matmul, not the F32
    /// normed streams (40 KB a token), and the gated mix rebuilds them from the streams (the
    /// norm's own two multiplies). The same ops' bits (the same greedy output as the default
    /// under --deterministic --stable-fusions), but without --stable-fusions greedy output moved
    /// in serving tests, also with the norm's node in the default layout: the smaller graph
    /// moves ggml-cuda's layout-dependent MoE fusions (#836). Steps that the default BF16
    /// up-mix takes (256+ tokens on a GB10) run the up-mix instead.
    #[arg(long, help_heading = HEADING)]
    hc_mix_rebuild_bf16: bool,
    /// qwen4exp on one GPU: ggml-cuda's ops for the hyper-connection mixer's elementwise steps
    /// (the low-rank gate's scale and SiLU in every mixer; in decode and verify steps of BF16
    /// mixers without --hc-mix-bf16 and in the MTP head, the combine's scale, sigmoid, scale
    /// and hc_post and the grouped norm's rms_norm and mul) instead of Legion's kernels with
    /// ggml's expressions in ggml's order (the default: the same output, #836). For
    /// comparisons.
    #[arg(long, help_heading = HEADING)]
    hc_chain_ggml: bool,
    /// qwen4exp on a GPU, every step that runs the hyper-connection mixer's gated mix as its own
    /// op (prompt steps of the NVFP4 checkpoints and of runs without the up-mix kernel, decode
    /// steps of BF16 mixers without --hc-mix-bf16, the MTP head): ggml-cuda's hc_pre instead of
    /// Legion's kernel with ggml's expressions in ggml's order (the default: the same output).
    /// For comparisons.
    #[arg(long, help_heading = HEADING)]
    hc_pre_ggml: bool,
    /// Prompt steps of 256+ tokens on compute capability 12.x GPUs (GB10; no effect elsewhere)
    /// with Q8_0 output projections behind a sigmoid gate (gated attention's attn_output, a
    /// sigmoid-gated DeltaNet layer's ssm_out): ggml's gate (writing the gated values) and the
    /// matmul's own quantization pass (reading them back), instead of one op that gates and
    /// quantizes (the same arithmetic, so the same output). For comparisons.
    #[arg(long, help_heading = HEADING)]
    gate_quant_ggml: bool,
    /// Where that op runs for a sigmoid-gated DeltaNet layer's ssm_out (qwen4exp): ggml's output
    /// RMS norm and its weight (writing the normed values) before the op, instead of the op
    /// norming each 128-value row itself in ggml's order (the same output). For comparisons.
    #[arg(long, help_heading = HEADING)]
    gate_norm_ggml: bool,
    /// Prompt steps of 256+ tokens on compute capability 12.x GPUs (GB10; no effect elsewhere):
    /// every Q8_0 projection of a layer input that several read (a DeltaNet layer's attn_qkv and
    /// attn_gate, an attention layer's attn_q, attn_k and attn_v, the shared expert's gate and
    /// up) quantizes it on its own, as ggml's mul_mat does, instead of once for all of them with
    /// ggml's own quantization kernel (the same bytes, so the same output). For comparisons.
    #[arg(long, help_heading = HEADING)]
    shared_quant_ggml: bool,
    /// Steps of 9-255 tokens (multi-sequence decode, MTP verify) on compute capability 12.x GPUs
    /// (GB10; no effect elsewhere) inside stage graphs: every Q8_0 projection of a layer input
    /// that several read (a DeltaNet layer's attn_qkv and attn_gate, an attention layer's
    /// attn_q, attn_k and attn_v, the shared expert's gate and up) quantizes it on its own, as
    /// ggml's mul_mat does, instead of the first quantizing it once into a scratch buffer with
    /// ggml's kernel and each running ggml's MMQ on those blocks (the default since #858). The
    /// same bytes and kernels, so the same output either way; for comparisons.
    #[arg(long, help_heading = HEADING)]
    shared_quant_small_ggml: bool,
    /// Compute capability 12.x GPUs (GB10; no effect elsewhere), steps on the fused MoE op, with
    /// a sigmoid-gated shared expert (qwen4exp, Qwen3.5 MoE): ggml's sigmoid, mul and add after
    /// the op (each a pass over the step's hidden states), instead of the op adding the shared
    /// expert in its per-token sum with the same roundings (the same output). For comparisons.
    #[arg(long, help_heading = HEADING)]
    shexp_sum_ggml: bool,
    /// qwen4exp prompt steps on Volta and Turing (compute capability 7.x; no effect elsewhere):
    /// the separate hyper-connection norm (layer 0's mixer, the MTP head, and every layer with
    /// --hc-post-norm-ggml) adds each square with one FMA, as the fused combine-and-norm does
    /// there, so the two round alike and --hc-post-norm-ggml gives the fused op's bits. Off by
    /// default: nvcc adds every fourth square with a separate multiply and add, and taking the
    /// one rounding moves greedy output (#409).
    #[arg(long, help_heading = HEADING)]
    hc_norm_fma: bool,
    /// qwen4exp prompt steps, CUDA: the per-layer n-gram embedding after its two matmuls (gate,
    /// norms, causal conv, adds) as Legion's two ops instead of ggml's chain of about thirty
    /// nodes: ~10 ms less per 4,096-token chunk on the first pipeline stage of a V100. Each op
    /// gives the chain's bits, but the smaller graph can flip which MoE layers ggml-cuda fuses
    /// the weighted expert sum in (FMA adds), so greedy output may change. Off by default.
    #[arg(long, help_heading = HEADING)]
    ple_ops: bool,
    /// With an expert cache: the cache's work after a step or a draft pass (learning the
    /// routing, planning swaps, writing the slot tables) runs before the next pass, with the
    /// GPUs idle, instead of beside GPU work (the default: with MTP beside the next MTP pass,
    /// without MTP over several GPUs each stage's share beside the next stage's compute). For
    /// comparisons. Replaces LEGION_AFTER_OVERLAP=0.
    #[arg(long, help_heading = HEADING)]
    no_after_overlap: bool,
    /// CUDA: one cudaMemcpyAsync per input of a decode or verify step and of an MTP draft pass
    /// (tokens, positions, slot and block tables, n-gram rows, ...), instead of the default: the
    /// host writes them all into pinned memory and one kernel per stage copies them into their
    /// tensors ahead of the graph (#814; the same bytes, so the same output, with fewer host calls
    /// while the GPUs wait). For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_input_pack: bool,
    /// A decode or verify step over several GPUs (a layer split): each next stage's residual
    /// copy and graph are queued only after the host waited for the stage before it, instead
    /// of the default: queued while that stage computes, the next GPU waiting for it on an
    /// event, so the GPUs go from stage to stage without the host in between (#814; the same
    /// graphs on the same inputs in the same order, so the same output). Not with
    /// --expert-peer-fetch, prompt steps or pipelined micro-batches, which launch after the
    /// wait anyway. For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_stage_launch_ahead: bool,
    /// --spec mtp over several GPUs (a layer split): a verify step's commit (the gated
    /// DeltaNet states moved to the accepted tokens) queued on every stage before the first
    /// draft pass, instead of the default on compute capability 7.x GPUs: the first draft pass
    /// queued at once on the last GPU and the commits beside it, the other stages' while it
    /// computes and the last stage's behind its reads (#814; the same commits on the same
    /// states before each stage's next step, so the same output). For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_commit_beside_draft: bool,
    /// With the commit beside the first draft pass (see --no-commit-beside-draft): the last
    /// GPU's commit queued behind the second draft pass's reads (or before the next step when
    /// the round ends after one pass) instead of behind the first pass's, where it runs in
    /// front of the second pass on that GPU (#814; the same commit on the same states before
    /// that stage's next step, so the same output). Off by default: on 4x V100 it cut the
    /// draft time ~6% but the decode steps took that time back (flat end to end).
    #[arg(long, help_heading = HEADING)]
    commit_last_late: bool,
    /// Stage graphs: each stage's verify commit (per gated DeltaNet layer a conv window copy
    /// and the delta rule over the kept tokens) captured as one CUDA graph and replayed,
    /// instead of launched kernel by kernel from the host (#814; the kept counts are a device
    /// input, so one capture serves every commit; the same kernels with the same arguments,
    /// so the same output). Off by default: flat on 4x V100, where the commit's launches
    /// already run beside the first draft pass.
    #[arg(long, help_heading = HEADING)]
    commit_capture: bool,
    /// --spec mtp: each draft pass queues the next pass's inputs while it computes (its picks
    /// and output residual copied on the GPU, the rest from pinned memory, #827), wherever
    /// nothing else runs between a round's passes (no --experts cache, one machine, one pass
    /// within the gather budget), instead of the host building them after the pass with the
    /// GPU idle. Off by default; output is the same either way, at any number of parallel
    /// requests (#858). Measured slower on a GB10 with --mtp-pipe-rows 16 (2-8% at 1, 4 and
    /// 16 requests, #858); for experiments.
    #[arg(long, help_heading = HEADING)]
    mtp_ahead: bool,
    /// --spec mtp with --mtp-ahead: a draft pass of at most this many rows is queued while
    /// the pass before it still computes, before the host has read that pass's picks, so the
    /// GPU does not wait for the host between passes (#827). On CUDA 12.3+ the queued pass is
    /// a conditional graph node: a one-thread kernel checks that every row of the pass before
    /// reached --spec-p-min and the GPU skips the pass otherwise (where CUDA refuses that graph,
    /// the pass runs and the host drops it). Output is the same either way. Default 0 (off).
    #[arg(long, value_name = "ROWS", help_heading = HEADING)]
    mtp_pipe_rows: Option<usize>,
    /// --spec mtp, CUDA: MTP draft passes of at least ROWS output rows run the head's routed
    /// experts on the fused expert op (on a GB10 its fragment kernels, as the target's verify
    /// steps) instead of ggml's MMQ / MMVQ (#858). The fused op sums in FP16 on tensor cores
    /// where MMQ quantizes the activations to Q8_1, so drafts and acceptance can change, and
    /// greedy output with them. Default 0 (off); try 9 (the passes MMQ takes).
    #[arg(long, value_name = "ROWS", help_heading = HEADING)]
    mtp_moe_fused_rows: Option<usize>,
    /// --experts cache on discrete GPUs: keep prefix caching's saved recurrent states
    /// (--prefix-cache-states, ~55 MiB per state per GPU on Qwen3.8-Flash-Next over two GPUs) in
    /// VRAM instead of pinned host memory. Default: host memory, so that VRAM goes to expert
    /// cache slots; a save or a resume then copies one state over the host link (about a
    /// millisecond on NVLink, a few on PCIe). For comparisons.
    #[arg(long, help_heading = HEADING)]
    prefix_states_vram: bool,
    /// GLM-5.3-Flash, CUDA, routed experts all in GPU memory (no --experts cache; e.g. one
    /// GB10): steps of up to 8 tokens run them through Legion's expert matvec (as the expert
    /// cache's experts run on an AC922; each expert streamed once, the gate and up rows in one
    /// launch) instead of the fused op's GEMM tiles. GB10, Maya-S: the IQ2 experts' reads
    /// ~94 -> ~210 GB/s. Changes greedy output vs the default (Q8_1-quantized activations
    /// summed as ggml's mmvq does, against the fused op's F16 tiles). Off by default.
    #[arg(long, help_heading = HEADING)]
    glm_expert_mv: bool,
    /// GLM-5.3-Flash, CUDA, --experts cache: the most tokens of a speculative verify step (e.g.
    /// --spec mtp --spec-fixed at k = 2 over 4 sequences: 12 rows) whose routed experts run
    /// through Legion's expert matvec (default 64, all it computes exactly; never below
    /// --expert-mv-max-tokens). Bigger verify steps, and prompt steps past
    /// --expert-mv-max-tokens, run the fused MoE op, which reads the uncached experts in place
    /// over the host link, ~20x slower in such steps. The matvec gives a token the same bits at
    /// any step size. Set 8 for the behaviour before #569, for A/B runs.
    #[arg(long, help_heading = HEADING, value_name = "TOKENS", value_parser = clap::value_parser!(u8).range(1..=64))]
    glm_verify_mv_max_tokens: Option<u8>,
    /// GLM-5.3-Flash, CUDA, --experts cache: plain decode steps of up to
    /// --glm-verify-mv-max-tokens sequences (default 64) also run through the expert matvec,
    /// not only verify steps. Past --expert-mv-max-tokens (8) they otherwise take the fused MoE
    /// op, which reads the uncached experts in place: 13.1 against 36.5 tok/s at C=16 on an AC922
    /// (#659); Maya-S on an AC922 (4x V100) with --glm-parallel-decode, 16 sequences: 18.6
    /// against 75.0 (#1174). Each sequence then decodes to the bits it gets alone, which changes
    /// greedy output past 8 sequences. Off by default.
    #[arg(long, help_heading = HEADING)]
    glm_decode_mv: bool,
    /// GLM-5.3-Flash, CUDA: the plain dense-weight kernels instead of the GB10-tuned ones (the
    /// default on integrated GPUs such as the GB10, on V100s and on Ampere GPUs (compute
    /// capability 8.0 to 8.7); other GPUs run the plain ones): a Q6_K decode GEMV staging its
    /// weight rows through shared memory with 16-byte loads (~165 -> ~220 GB/s; not on V100s,
    /// where it is slower, nor on Ampere GPUs), a prompt GEMM on 128 x 128 tiles (~2x; on
    /// sm_80+ Q6_K weights of 1024+ rows on mma.sync, ~1.3x more, reading the input once rounded
    /// to F16 when 2+ share it), prompt MLA attention with 16 heads a block (each staged cache row
    /// read by 4x the heads), the prompt hyper-connection mix 4 tokens a block, the prompt
    /// KDA conv and scan 64 state rows a block, and on sm_80+ the routed experts' prompt GEMMs
    /// on IQ2_XXS gate/up and IQ2_S down weights building their fragments in registers (~1.2x).
    /// The same bits either way. For comparisons.
    #[arg(long, help_heading = HEADING, conflicts_with = "glm_gb10_linear")]
    glm_linear_plain: bool,
    /// GLM-5.3-Flash, CUDA: the GB10-tuned dense-weight kernels (see --glm-linear-plain) on any
    /// discrete GPU too. The same bits either way. For comparisons.
    #[arg(long, help_heading = HEADING)]
    glm_gb10_linear: bool,
    /// GLM-5.3-Flash, CUDA sm_80+ (e.g. the GB10), with the GB10-tuned kernels on: the MLA
    /// attention of prompt steps (64+ tokens) on tensor cores (mma.sync, F32 sums), the queries
    /// and the softmax weights rounded to F16. Several times faster than the default CUDA-core
    /// kernel on long prompts; changes greedy output vs the default (other roundings and sums).
    /// Decode steps are unchanged. Off by default.
    #[arg(long, help_heading = HEADING, conflicts_with = "glm_linear_plain")]
    glm_mla_tc: bool,
    /// GLM-5.3-Flash, CUDA sm_80+ (e.g. the GB10), with the GB10-tuned kernels on: the routed
    /// experts' prompt GEMMs (steps of 17+ tokens) on IQ2_XXS gate/up and IQ2_S or IQ3_XXS down
    /// weights on int8 tensor cores, MMQ-style: the expert inputs quantized to 8 bits with an F32 scale per
    /// 32 values. Faster prompt processing; changes output slightly vs the default (8-bit instead
    /// of F16 activations). Decode steps are unchanged. Off by default.
    #[arg(long, help_heading = HEADING, conflicts_with = "glm_linear_plain")]
    glm_expert_q8: bool,
    /// CUDA, one GPU per model part (not with --tp, --pair-tp or the GLM parallel decode): run
    /// each graph as one CUDA graph with its external ops' kernels inside, instead of ggml's
    /// CUDA graphs between them and the external ops launched one by one. The same kernels in
    /// the same order (the same bits), without the launch gaps between them. ggml's own CUDA
    /// graphs are turned off for it; ops that wait on the host or use other streams (the expert
    /// staging, the late n-gram rows) run between the captured parts as before. On by default
    /// for `legion serve` since #849 (Flash-Next, Gemma 4 and GLM validated bitwise on an AC922);
    /// this flag only states the default. `--no-stage-graphs` turns them off (#817).
    #[arg(long, help_heading = HEADING, conflicts_with = "no_stage_graphs")]
    stage_graphs: bool,
    /// Turn the stage graphs (see --stage-graphs, on by default) off: ggml's own CUDA graphs
    /// between the external ops, launched one by one, as before #849. The same bits, a few
    /// percent slower decode on Flash-Next; for comparisons and as a fallback.
    #[arg(long, help_heading = HEADING)]
    no_stage_graphs: bool,
    /// Launch every stage-graph capture whole (see --stage-graphs). By default a capture of
    /// 512 or more nodes is launched as two: a short head (1/32 of its nodes), then the rest
    /// while the head runs, because the GPU starts a big CUDA graph only some way into its
    /// launch (1-2 ms idle per Qwen3.8-Flash-Next decode step on a GB10, #836). The same
    /// kernels in the same order (the same bits); for comparisons.
    #[arg(long, help_heading = HEADING)]
    no_stage_graph_split: bool,
    /// Capture a new stage graph of a decode, verify or MTP draft step at its second run, as
    /// ggml does (see --stage-graphs), with the GPU waiting for the capture. By default it is
    /// captured right behind its first run while the GPU runs that, on GPUs of compute
    /// capability 7.5 or newer (10-16 ms of GPU idle per new Qwen3.8-Flash-Next verify-step
    /// shape on a GB10, at each request's start and as the context grows, #827). The same
    /// kernels in the same order (the same bits); for comparisons.
    #[arg(long, help_heading = HEADING)]
    no_stage_graph_early_capture: bool,
    /// Capture a new MTP take-in graph of several parallel requests right behind its first run,
    /// as the other small steps' graphs (see --no-stage-graph-early-capture). By default such a
    /// graph is captured only behind its second run: its rows follow each request's accepted
    /// draft count, so most of its shapes never run again, and at 4 parallel requests on a
    /// DGX Spark (GB10) nearly every verify round captured a new one (2.6 CUDA graph
    /// instantiations, ~1.5 ms of host time a round, #836). The same kernels in the same order
    /// (the same bits); for comparisons.
    #[arg(long, help_heading = HEADING)]
    no_stage_graph_capture_on_reuse: bool,
    /// Compare every stage-graph run's words (each node's op, shape, strides, parameters and
    /// data pointers, and its sources') with its last run's, as before #814. By default a kept
    /// step or MTP graph that ran unchanged skips that compare (~40-45 µs of host time per
    /// graph on a Qwen3.8-Flash-Next V100 stage, on the critical path at a step's first stage
    /// and each MTP draft pass): a kept graph is never written after its build, so its words
    /// stay the same; every 64th run in a row compares anyway. Such a run also reuses the late
    /// input's first reader found before and keeps its cache entry with the graph (no shared
    /// lock or lookup); this switch turns those off too. The same kernels in the same order
    /// (the same bits); for comparisons.
    #[arg(long, help_heading = HEADING)]
    no_stage_graph_words_skip: bool,
    /// Keep each kept step or MTP graph's stage-graph captures with the graph for as long as it
    /// is kept (#858). By default they go into the shared capture store after a graph's first
    /// runs and stay with the graph only once it runs unchanged; the store keeps 64 sets a GPU
    /// (--stage-graph-cache) and drops sets unused for 10 s, so with hundreds of kept graphs
    /// (the MTP take-in shapes at 4+ parallel requests) graphs that stayed kept were captured
    /// again (captures_lost in --stage-graph-stats). The same kernels in the same order (the
    /// same bits); costs the kept graphs' captures' memory. Needs the words skip (see
    /// --no-stage-graph-words-skip).
    #[arg(long, help_heading = HEADING)]
    stage_graph_hold_kept: bool,
    /// Log every 10 s how many stage-graph runs, captures (CUDA graph instantiations) and kept
    /// step graphs the server had, the captures' host time and each run's check time (words
    /// compare and cache lookup before its first launch, check_us_per_run), while any capture
    /// or build happened (#858), with step graphs apart from the MTP head's: runs that ran
    /// directly (direct_runs, direct_runs_mtp), the MTP share of captures_new and the build host
    /// time in a step, ahead of it and for the MTP head (build_ms, build_ahead_ms,
    /// mtp_build_ms). For finding shapes that keep changing (each change is a new
    /// graph and new captures); no effect on what runs. Off by default.
    #[arg(long, help_heading = HEADING)]
    stage_graph_stats: bool,
    /// Do not build a long prompt's next chunk graph while the current chunk computes (#836).
    /// By default a one-stage step that runs one prompt's chunk alone builds and plans the
    /// next chunk's graph between its launch and its wait, so the next chunk starts its
    /// kernels without the ~25-30 ms graph build (each chunk is a new shape: its
    /// sparse-attention pools grow). The same graph in the same compute buffer, so the same
    /// bits. Use only to compare against the old behaviour.
    #[arg(long, help_heading = HEADING)]
    no_prompt_build_ahead: bool,
    /// Do not allocate the kept prompt graphs' compute buffers at load (#836). By default, on
    /// an integrated GPU (a DGX Spark's GB10), the load measures a full prompt chunk at the
    /// end of the context and allocates one buffer that big for each kept prompt graph,
    /// writing it once, so the first long prompt after a start does not free and grow its
    /// buffers chunk by chunk (each growth waits for the GPU: up to ~1 s per chunk). Holds
    /// that memory from load (logged as "prompt compute buffers reserved"). The same plans
    /// in bigger buffers, so the same bits. Use to save the memory or to compare against
    /// the old behaviour.
    #[arg(long, help_heading = HEADING)]
    no_prompt_buffer_reserve: bool,
    /// Log one line per prompt step of 1,024 tokens or more (#836): its host phases (graph
    /// build and plan, inputs, launch, GPU wait, outputs, the next chunk's graph built ahead),
    /// whether it built its graph, and what grew meanwhile: direct reclaim, compaction stalls,
    /// major faults and huge pages from the kernel's memory counters, the process's page faults
    /// and CPU time, the n-gram table's rows and SSD reads. For finding why one prefill is
    /// slower than another of the same prompt; no effect on what runs. Off by default.
    #[arg(long, help_heading = HEADING)]
    prompt_step_stats: bool,
    /// Debug check (#836): after each kept graph is allocated, log the inputs that a node of
    /// the graph is placed over (ggml frees an input after its last consumer) and whether the
    /// runs fill each again, warning about any that is neither filled before every run nor an
    /// output (a value set once would be stale from the second run on). Across graphs kept in
    /// one compute buffer too: logs each pair where one writes over the other's set-once
    /// inputs, and warns (STALE) when such a graph ran between two runs of the other. Each
    /// distinct line once; no effect on what runs. Off by default; for audits.
    #[arg(long, hide = true)]
    audit_graph_inputs: bool,
    /// Sets of stage-graph captures each GPU keeps by their graph's contents (default 64; 0:
    /// off, as before #858). A step graph built again with the contents of one seen before
    /// (a kept graph dropped and rebuilt for the same shape in the same compute buffer, many
    /// times a second at 4+ parallel requests) then launches that one's captures instead of
    /// capturing again (~0.8 ms each on a V100). The same kernels with the same arguments (the
    /// same bits). The least recently used set goes beyond the limit, and any set unused for
    /// 10 s; lower it only to save the captures' memory.
    #[arg(long, help_heading = HEADING, value_name = "SETS", value_parser = clap::value_parser!(u16).range(0..=4096))]
    stage_graph_cache: Option<u16>,
    /// Spare CUDA executable graphs each GPU keeps from dropped stage-graph captures (default
    /// 8; 0: off, every capture instantiated as before #836). A new capture with the same
    /// topology (the same kernels in the same order, other sizes and addresses: a new MTP
    /// verify shape as the contexts grow) then updates a spare (cudaGraphExecUpdate) instead
    /// of instantiating its graph (~18 ms each for a Qwen3.8-Flash-Next verify step on a DGX
    /// Spark (GB10), ~11 of them per 256 tokens at 4 parallel requests). The updated graph runs
    /// the new capture's kernels with its arguments (the same bits); where CUDA refuses the
    /// update, the capture is instantiated. Only captures of 256+ CUDA graph nodes are kept;
    /// lower it to save their memory. While this or --stage-graph-exec-steal-ms is on, a big
    /// capture's device-to-device memcpy nodes run as copy kernels (the same bytes), which an
    /// update may move to other addresses (it refuses that for a 2-D memcpy).
    #[arg(long, help_heading = HEADING, value_name = "EXECS", value_parser = clap::value_parser!(u16).range(0..=256))]
    stage_graph_spare_execs: Option<u16>,
    /// A new big stage-graph capture with no spare of its topology (see
    /// --stage-graph-spare-execs) takes over the executable graph of a kept capture of the same
    /// topology that was not launched for this many milliseconds (default 1000; 0: never) and
    /// updates it, rather than instantiating its graph. The kept graphs of the verify shapes a
    /// context grew out of hold their captures until evicted, so early on there are no spares;
    /// a taken-over capture is captured again if its graph ever runs again. The same bits.
    #[arg(long, help_heading = HEADING, value_name = "MS", value_parser = clap::value_parser!(u32).range(0..=600_000))]
    stage_graph_exec_steal_ms: Option<u32>,
    /// Graph shapes the MTP head keeps per kind of pass (decode drafts, take-in), default 16
    /// (8 before #858; 256 for qwen4exp's MTP head on an integrated GPU such as the GB10, which
    /// drafts up to 8 tokens: ~100 take-in shapes at 4 parallel requests). With several sequences drafting together the take-in shapes follow each
    /// sequence's accepted count: every pass of a shape not kept builds its graph again and
    /// captures it (#858). The kept graphs of a kind share one compute buffer, so more costs
    /// host memory and their captures, not a bigger buffer. The same kernels (the same bits).
    #[arg(long, help_heading = HEADING, value_name = "GRAPHS", value_parser = clap::value_parser!(u16).range(1..=1024))]
    mtp_kept_graphs: Option<u16>,
    /// Verify-step graph shapes each pipeline stage keeps apart from prompt-chunk shapes
    /// (default 0: off, verify and prompt shapes share the stage's 3 kept mixed graphs; 32 for
    /// qwen4exp with its MTP head on an integrated GPU such as the GB10). With
    /// MTP at several sequences the verify shape follows the sequences' padded context, so a
    /// round of requests walks through several shapes and the next round walks them again;
    /// 3 shared slots rebuilt and re-captured each one (#858). With N, verify shapes evict only
    /// each other. Each kept verify graph has its own (small) compute buffer, so this costs
    /// N verify buffers of device or unified memory per stage. The same kernels (the same bits).
    #[arg(long, help_heading = HEADING, value_name = "GRAPHS", value_parser = clap::value_parser!(u16).range(0..=256))]
    verify_kept_graphs: Option<u16>,
    /// MiB of compute buffers each pipeline stage's kept verify graphs (--verify-kept-graphs)
    /// may hold; past it the least recently run verify graphs go, the others stay kept. Default
    /// (not given): an eighth of the memory the stage's device has free after the load, at most
    /// 2048 (about 30 det C=4 verify shapes on the GB10 take ~1.4 GiB; at 16 sequences each
    /// verify graph is several times bigger). 0: no limit, only the count. Applies only with
    /// --verify-kept-graphs above 0 (the default for qwen4exp + MTP on an integrated GPU). The
    /// same kernels (the same bits); lower it to save memory, raise it if C=16 rebuilds verify
    /// graphs often (--stage-graph-stats logs verify_mib and the drops).
    #[arg(long, help_heading = HEADING, value_name = "MIB", value_parser = clap::value_parser!(u32).range(0..=1048576))]
    verify_kept_mib: Option<u32>,
    /// Keep an MTP take-in pass's sequences in the batch's order instead of ordering them by
    /// their rows (the accepted drafts plus one). By default they go in order of rows, so the
    /// pass's graph shape depends on how many sequences take in how many rows, not on which:
    /// at 4 parallel requests 15 take-in shapes instead of 81, kept by the head rather than
    /// built and captured again for most passes (#858). Each sequence's rows are the same, but
    /// their place in the pass is not, and with many rows that moves bits: at 16 parallel
    /// requests the drafts, and so the output, differ from the default's (measured on a GB10;
    /// 1 and 4 requests match). For comparisons.
    #[arg(long, help_heading = HEADING)]
    no_mtp_take_in_sort: bool,
    /// Qwen3.8-Flash-Next (qsa attention): pad the pools a decode or verify step of several
    /// sequences lists so its graph shape lasts at least STEPS steps (e.g. 64: 1,024 pools at
    /// 16 sequences of 3 tokens) instead of changing every few steps and rebuilding and
    /// re-capturing the stage graphs (#858). The extra pools are masked, so outputs stay the
    /// same; they cost scoring time. Default 0 (off: pools padded to 64 as before).
    #[arg(long, help_heading = HEADING, value_name = "STEPS", value_parser = clap::value_parser!(u16).range(0..=1024))]
    qsa_pool_pad_steps: Option<u16>,
    /// CUDA, a layer split or one GPU (not --tp, not a KV host tier's stages): each GPU's KV
    /// pool in a reserved virtual range, every cache tensor on a mapping granule of its own and
    /// backed by VRAM allocations of at most 4 MiB each (CUDA virtual memory management),
    /// instead of one ggml CUDA buffer. The same blocks, read and written the same way (the
    /// same bits); each tensor's rounding to whole granules takes up to 2 MiB more VRAM, so the
    /// expert cache can lose a few slots. Needed by --kv-vmm-keep (#814). On by default, with
    /// --kv-vmm-keep 32768, on V100s (and GH200s, unmeasured) with the expert cache over a layer
    /// split of 2 GPUs; off elsewhere.
    #[arg(long, help_heading = HEADING)]
    kv_vmm: bool,
    /// Turn --kv-vmm (and so --kv-vmm-keep) off where it is on by default: every block of the
    /// KV pools in one ggml CUDA buffer, the expert cache sized beside all of them.
    #[arg(long, help_heading = HEADING, conflicts_with_all = ["kv_vmm", "kv_vmm_keep"])]
    no_kv_vmm: bool,
    /// The expert cache and a layer split whose long prompts lend the cache (an AC922 with 2
    /// GPUs; implies --kv-vmm): only the first TOKENS tokens of each GPU's KV pool keep VRAM
    /// at load; the VRAM of the rest goes to more expert-cache slots. When the sequences need
    /// more blocks than that, cached prefixes and warm sequences give theirs back first, then
    /// the expert cache is lent (as to a long prompt) and the rest of the pool is mapped
    /// again, until none of its blocks is in use (the cache is then restored). The same
    /// outputs; faster while the kept tokens hold the load (2 V100s with MTP: +17% cache
    /// slots, decode C=1/4/16 +3/10/7%), slower while the rest is needed. Also with
    /// --glm-parallel-decode, where every GPU keeps every layer's KV (an AC922 with 4 GPUs,
    /// 131072 context: 2.9 GiB a GPU back to the cache at 32768; a long prompt past the kept
    /// tokens maps the rest when it starts). Default 32768 where --kv-vmm is on by default;
    /// elsewhere off (every block keeps its VRAM).
    #[arg(long, help_heading = HEADING, value_name = "TOKENS", value_parser = clap::value_parser!(u32).range(1..))]
    kv_vmm_keep: Option<u32>,
    /// Threads that read the model files at load (1-8; default 4). Replaces
    /// LEGION_LOAD_THREADS.
    #[arg(long, help_heading = HEADING, value_name = "THREADS", value_parser = clap::value_parser!(u8).range(1..=8))]
    load_threads: Option<u8>,
    /// Let every load thread write GPU tensors (default: one per queue does). Unverified on a
    /// GPU, so opt-in. Replaces LEGION_LOAD_PARALLEL_DEVICE=1.
    #[arg(long, help_heading = HEADING)]
    load_parallel_device: bool,
    /// Read the tensors that stay in CPU memory at load instead of mapping them from the file
    /// (the default). Replaces LEGION_NO_MMAP=1.
    #[arg(long, help_heading = HEADING)]
    no_mmap: bool,
    /// The page cache of model data read into host memory at load: `auto` (default: data not
    /// cached yet keeps its cache while it fits in its NUMA node's free memory, else drops it
    /// once read, so it cannot spill onto other nodes, on an AC922 the GPUs' memory; data
    /// already cached stays, so a warm reload reads no disk), `keep` (never drop it) or `drop`
    /// (always drop it once read).
    #[arg(long, help_heading = HEADING, value_enum, value_name = "MODE")]
    load_page_cache: Option<legion::load_io::PageCache>,
}

impl SwitchArgs {
    /// Whether stage graphs are wanted: on unless `--no-stage-graphs` (#849). `legion serve`
    /// turns them on where they apply ([`super::ModelArgs::stage_graphs`]).
    pub(in crate::cli) fn stage_graphs(&self) -> bool {
        !self.no_stage_graphs
    }

    /// Whether `--stage-graphs` was given explicitly.
    pub(in crate::cli) fn stage_graphs_given(&self) -> bool {
        self.stage_graphs
    }

    /// The switches these flags ask for: `Some` for each flag given, `None` (the built-in
    /// default) for the rest. `legion serve` then sets the stage graphs' default itself.
    pub(in crate::cli) fn switches(&self) -> Switches {
        Switches {
            gated_norm: self.gated_norm.then_some(true),
            moe_combine: self.moe_combine.then_some(true),
            stable_fusions: self.stable_fusions.then_some(true),
            delta_prep: self.delta_prep_ggml.then_some(false),
            gdn_scan: self.gdn_scan_ggml.then_some(false),
            gdn_mma: self.gdn_mma.then_some(true),
            gdn_staged: self.gdn_prompt_ggml.then_some(false),
            gdn: self.gdn_ggml.then_some(false),
            gdn_decode: self.gdn_decode_ggml.then_some(false),
            gdn_replay_staged: self.gdn_replay_smem.then_some(true),
            gdn_replay_few: self.no_gdn_replay_few.then_some(false),
            gdn_pad_skip: self.no_gdn_pad_skip.then_some(false),
            gdn_commit_slots: self.gdn_commit_runs.then_some(false),
            gdn_commit_in_verify: if self.no_gdn_commit_in_verify {
                Some(false)
            } else {
                self.gdn_commit_in_verify.then_some(true)
            },
            gdn_state_defer: if self.no_gdn_state_defer {
                Some(false)
            } else {
                self.gdn_state_defer.then_some(true)
            },
            gdn_state_defer_rows: self.gdn_state_defer_rows,
            gdn_state_defer_min_seqs: self.gdn_state_defer_min_seqs,
            norm_rope: self.norm_rope.then_some(true),
            kv_append: self.kv_append.then_some(true),
            head_rope_fused: self.head_rope_fused.then_some(true),
            joined_matmuls: self.no_joined_matmuls.then_some(false),
            joined_prompt_min_tokens: self.joined_prompt_min_tokens,
            joined_prompt_z: self.no_joined_prompt_z.then_some(false),
            joined_prompt_qkv: self.no_joined_prompt_qkv.then_some(false),
            dense_mv: self.dense_mv_ggml.then_some(false),
            q8_mv: self.draft_head_q8_mv.then_some(true),
            q8_head: self.no_draft_head_mv.then_some(false),
            q8_head_pick: self.no_draft_head_pick.then_some(false),
            q8_row_mv: self.no_q8_group_row_mv.then_some(false),
            q4k_head: self.no_draft_head_q4k_mv.then_some(false),
            q8_out_mv: self.q8_out_row_mv.then_some(true),
            q4k_head_pick: self.no_draft_head_q4k_pick.then_some(false),
            q4k_head_pick_cols: self.no_draft_head_q4k_pick_cols.then_some(false),
            head_mm: self.no_head_mm.then_some(false),
            q8_group: if self.no_q8_group {
                Some(false)
            } else {
                self.q8_group.then_some(true)
            },
            q8_group_parts: self.q8_group_parts,
            dense_mv_small: if self.no_dense_mv_small {
                Some(false)
            } else {
                self.dense_mv_small.then_some(true)
            },
            cluster_select: self.no_cluster_select.then_some(false),
            dense_mv_f32: self.dense_mv_f32.then_some(true),
            dense_mv_ab: self.no_dense_mv_ab.then_some(false),
            dense_mv_split: self.dense_mv_split.then_some(true),
            dense_mv_wide_chain: if self.no_dense_mv_wide_chain {
                Some(false)
            } else {
                self.dense_mv_wide_chain.then_some(true)
            },
            router_mv: self.no_router_mv.then_some(false),
            mvf_ab: self.no_mvf_ab.then_some(false),
            router_join: self.no_router_join.then_some(false),
            router_bf16: self.no_router_bf16.then_some(false),
            ple_early_reads: self.no_ple_early_reads.then_some(false),
            nvfp4_mmvq_glu: self.nvfp4_mmvq_glu.then_some(true),
            fp8_small_batch_gemv: self.fp8_small_batch_gemv.then_some(true),
            fp8_sliced_k: self.fp8_sliced_k.then_some(true),
            hopper_gemm: self.hopper_gemm.then_some(true),
            hopper_attn: self.hopper_attn.then_some(true),
            paged_prefill_qm: self.no_paged_prefill_qm.then_some(false),
            hopper_pdl: self.hopper_pdl.then_some(true),
            nvfp4_wide_mv: self.nvfp4_wide_mv.then_some(true),
            nvfp4_expert_pdl: self.no_nvfp4_expert_pdl.then_some(false),
            nvfp4_expert_tile_prefetch: self.no_nvfp4_expert_tile_prefetch.then_some(false),
            nvfp4_expert_mv_max: self.nvfp4_expert_mv_max.map(i64::from),
            nvfp4_expert_mv_tile: self.nvfp4_expert_mv_tile.map(i32::from),
            nvfp4_expert_mv_split: self.nvfp4_expert_mv_split,
            nvfp4_expert_rows_bf16: self.nvfp4_expert_rows_bf16.then_some(true),
            moe_pad_remap: self.no_moe_pad_remap.then_some(false),
            moe_pad_remap_tiered: self.moe_pad_remap_tiered.then_some(true),
            moe_pad_remap_probe: self.moe_pad_remap_probe,
            moe_pad_skip_tiered: self.moe_pad_skip_tiered.then_some(true),
            moe_pad_skip_layout_only: self.moe_pad_skip_layout_only.then_some(true),
            moe_pad_skip: self.no_moe_pad_skip.then_some(false),
            tp_moe_ep_skip: self.no_tp_moe_ep_skip.then_some(false),
            paged_verify_split: self.paged_verify_split.then_some(true),
            paged_verify_split_wmma: self.paged_verify_split_wmma.then_some(true),
            shexp_gate: self.shexp_gate.then_some(true),
            decode_fuse: self.decode_fuse.then_some(true),
            node_fuse: self.node_fuse_ggml.then_some(false),
            moe_pair_out: self.moe_pair_out,
            moe_raw_staging: self.no_moe_raw_staging.then_some(false),
            moe_down_sum: self.moe_down_sum.then_some(true),
            expert_mv_vram: self.expert_mv_vram.map(usize::from),
            expert_mv_prefetch: self.expert_mv_prefetch.then_some(true),
            expert_mv_link_ctas: self.expert_mv_link_ctas.map(usize::from),
            expert_mv_ptx: self.expert_mv_ptx.then_some(true),
            expert_mv_ptx_side: self.expert_mv_ptx_side.then_some(true),
            expert_mv_device: self.expert_mv_device.then_some(true),
            expert_mv_max_tokens: self.expert_mv_max_tokens.map(usize::from),
            expert_mv_glu: self.expert_mv_split_glu.then_some(false),
            expert_mv_dot: self.expert_mv_dot,
            expert_mv_pipe: self.expert_mv_pipe.map(usize::from),
            expert_mv_q4k_units: self.expert_mv_q4k_units.then_some(true),
            expert_mv_q4k_hold: self.no_expert_mv_q4k_hold.then_some(false),
            expert_mv_balance: self.no_expert_mv_balance.then_some(false),
            expert_mv_half_tiles: self.no_expert_mv_half_tiles.then_some(false),
            expert_miss_overlap: if self.expert_miss_overlap {
                Some(true)
            } else {
                self.no_expert_miss_overlap.then_some(false)
            },
            expert_peer_fetch: if self.expert_peer_fetch {
                Some(true)
            } else {
                self.no_expert_peer_fetch.then_some(false)
            },
            expert_peer_share: self.expert_peer_share,
            expert_peer_ctas: self.expert_peer_ctas.map(usize::from),
            qsa_scores_tc: self.qsa_scores_tc.then_some(true),
            qsa_attn_chain: self.qsa_attn_chain.then_some(true),
            qsa_pick_all: self.qsa_always_score.then_some(false),
            hc_mix: self.hc_mix_ggml.then_some(false),
            hc_mix_bf16: self.hc_mix_bf16.then_some(true),
            hc_l2_prefetch: self.hc_l2_prefetch,
            hc_norm_regs: self.no_hc_norm_regs.then_some(false),
            hc_down_split: self.no_hc_down_split.then_some(false),
            hc_down_hoist: self.no_hc_down_hoist.then_some(false),
            hc_up_stage: self.no_hc_up_stage.then_some(false),
            hc_post_norm: self.hc_post_norm_ggml.then_some(false),
            hc_moe_fold: if self.hc_moe_fold {
                Some(true)
            } else {
                self.no_hc_moe_fold.then_some(false)
            },
            hc_lazy_streams: self.no_hc_lazy_streams.then_some(false),
            tp_gather_mix: self.no_tp_gather_mix.then_some(false),
            tp_gather_widen: self.no_tp_gather_widen.then_some(false),
            hc_fuse: self.hc_fuse.then_some(true),
            hc_xn16: self.hc_xn_f32.then_some(false),
            hc_quant: self.hc_quant_ggml.then_some(false),
            mmq_ids: self.expert_mmq.then_some(true),
            nvfp4_once: self.nvfp4_once.then_some(true),
            nvfp4_mvq: if self.no_nvfp4_mvq {
                Some(false)
            } else {
                self.nvfp4_mvq.then_some(true)
            },
            q8_mmq: if self.q8_mmq_ggml {
                Some(false)
            } else {
                self.q8_mmq.then_some(true)
            },
            q8_mmq_stream: self.q8_mmq_stream.then_some(true),
            q8_mmq_fixup: self.q8_mmq_fixup,
            tu_mmq: self.tu_mmq_ggml.then_some(false),
            hc_up_mix: self.hc_up_mix_ggml.then_some(false),
            hc_up_mix_bf16: self.no_hc_up_mix_bf16.then_some(false),
            hc_xn_rebuild: self.hc_xn_written.then_some(false),
            hc_streams_bf16: self.hc_streams_bf16.then_some(true),
            hc_bf16: self.hc_bf16.then_some(true),
            hc_mix16: self.hc_mix_f16.then_some(true),
            hc_mix_rebuild_bf16: self.hc_mix_rebuild_bf16.then_some(true),
            hc_chain: self.hc_chain_ggml.then_some(false),
            hc_pre: self.hc_pre_ggml.then_some(false),
            gate_q8: self.gate_quant_ggml.then_some(false),
            gate_norm: self.gate_norm_ggml.then_some(false),
            shared_quant: self.shared_quant_ggml.then_some(false),
            shared_quant_small: self.shared_quant_small_ggml.then_some(false),
            shexp_sum: self.shexp_sum_ggml.then_some(false),
            hc_norm_fma: self.hc_norm_fma.then_some(true),
            ple: self.ple_ops.then_some(true),
            after_overlap: self.no_after_overlap.then_some(false),
            input_pack: self.no_input_pack.then_some(false),
            launch_ahead: self.no_stage_launch_ahead.then_some(false),
            commit_beside_draft: self.no_commit_beside_draft.then_some(false),
            commit_last_late: self.commit_last_late.then_some(true),
            commit_capture: self.commit_capture.then_some(true),
            mtp_ahead: self.mtp_ahead.then_some(true),
            mtp_pipe_rows: self.mtp_pipe_rows,
            mtp_moe_fused_rows: self.mtp_moe_fused_rows,
            saved_states_vram: self.prefix_states_vram.then_some(true),
            load_threads: self.load_threads.map(usize::from),
            load_parallel_device: self.load_parallel_device.then_some(true),
            mmap: self.no_mmap.then_some(false),
            load_page_cache: self.load_page_cache,
            glm_expert_mv: self.glm_expert_mv.then_some(true),
            glm_verify_mv_tokens: self.glm_verify_mv_max_tokens.map(usize::from),
            glm_decode_mv: self.glm_decode_mv.then_some(true),
            glm_gb10_linear: if self.glm_linear_plain {
                Some(false)
            } else {
                self.glm_gb10_linear.then_some(true)
            },
            glm_mla_tc: self.glm_mla_tc.then_some(true),
            glm_expert_q8: self.glm_expert_q8.then_some(true),
            stage_graphs: if self.no_stage_graphs {
                Some(false)
            } else {
                self.stage_graphs.then_some(true)
            },
            stage_graph_split: self.no_stage_graph_split.then_some(false),
            stage_graph_early: self.no_stage_graph_early_capture.then_some(false),
            stage_graph_reuse_capture: self.no_stage_graph_capture_on_reuse.then_some(false),
            stage_graph_words_skip: self.no_stage_graph_words_skip.then_some(false),
            stage_graph_hold_kept: self.stage_graph_hold_kept.then_some(true),
            stage_graph_stats: self.stage_graph_stats.then_some(true),
            prompt_build_ahead: self.no_prompt_build_ahead.then_some(false),
            prompt_buffer_reserve: self.no_prompt_buffer_reserve.then_some(false),
            prompt_step_stats: self.prompt_step_stats.then_some(true),
            audit_graph_inputs: self.audit_graph_inputs.then_some(true),
            stage_graph_cache: self.stage_graph_cache.map(usize::from),
            stage_graph_spare_execs: self.stage_graph_spare_execs.map(usize::from),
            stage_graph_exec_steal_ms: self.stage_graph_exec_steal_ms.map(u64::from),
            mtp_kept_graphs: self.mtp_kept_graphs.map(usize::from),
            verify_kept_graphs: self.verify_kept_graphs.map(usize::from),
            verify_kept_mib: self.verify_kept_mib.map(|m| m as usize),
            mtp_take_in_sort: self.no_mtp_take_in_sort.then_some(false),
            qsa_pool_pad_steps: self.qsa_pool_pad_steps.map(usize::from),
            kv_vmm: if self.no_kv_vmm {
                Some(false)
            } else {
                (self.kv_vmm || self.kv_vmm_keep.is_some()).then_some(true)
            },
            kv_vmm_keep: self.kv_vmm_keep.map(|t| t as usize),
        }
    }
}

/// `--nvfp4-expert-mv-tile`: 1, 2 or 4 (the grouped mv kernel's compiled token tiles).
fn mv_tile(value: &str) -> Result<u8, String> {
    match value.parse::<u8>() {
        Ok(nt @ (1 | 2 | 4)) => Ok(nt),
        _ => Err(format!("1, 2 or 4 token tiles, not {value}")),
    }
}

#[cfg(test)]
mod tests;
