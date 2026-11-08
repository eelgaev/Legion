//! Tensor parallelism over machines (#733) for qwen4exp (Qwen3.8-Flash-Next) through the
//! engine on the CPU, ranks as threads over loopback: the tiny qwen4exp (hyper-connections,
//! the per-layer n-gram embedding, sparse attention, DeltaNet, a MoE with a shared expert)
//! over two ranks gives the greedy output of one (up to the rounding of the split sums, which
//! this model's greedy picks do not feel), batched and for a warm sequence, twice; also with
//! `--spec mtp` drafting with its MTP head split like the target.

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
use legion::engine::{Engine, FinishReason, GenerateRequest, Prompt, SamplingParams, TokenEvent};
use legion::model::core::MoeKernel;
use legion::model::qwen4::tiny;
use legion::spec::{SpecConfig, SpecKind};
use legion::tp_net::ep_place::Placement;
use legion::tp_net::{Overlap, TpNetConfig, Wire};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

const MAX_TOKENS: u32 = 40;

fn prompts() -> Vec<String> {
    vec![
        "The quick brown fox jumps over the lazy dog. ".repeat(3),
        "0123456789".repeat(5),
        "tensor parallel".into(),
    ]
}

fn config(path: &std::path::Path, tp_net: Option<TpNetConfig>) -> LmConfig {
    LmConfig {
        model_name: Some("tiny".into()),
        ctx_size: 512,
        threads: 2,
        max_seqs: 2,
        max_batch_tokens: 40,
        prefill_chunk: 16,
        use_ref: true,
        tp_net,
        ..LmConfig::new(path, DeviceChoice::Cpu)
    }
}

fn request(prompt: &str) -> GenerateRequest {
    GenerateRequest {
        model: "tiny".into(),
        prompt: Prompt::Text(prompt.into()),
        sampling: SamplingParams {
            temperature: 0.0,
            ..Default::default()
        },
        max_tokens: MAX_TOKENS,
        stop: vec![],
        ignore_eos: true,
        min_tokens: 0,
        thinking: None,
        template_kwargs: Default::default(),
        response_format: None,
        cancel: CancellationToken::new(),
        scheduling: Default::default(),
    }
}

/// Every prompt at once (more than `max_seqs`), then the first again (a warm sequence).
async fn run_all(engine: &LmEngine) -> Vec<(String, FinishReason, u32)> {
    let ps = prompts();
    let streams: Vec<_> = ps
        .iter()
        .map(|p| engine.generate(request(p)).unwrap())
        .collect();
    let mut replies = Vec::new();
    for mut s in streams {
        replies.push(collect(&mut s).await);
    }
    let mut s = engine.generate(request(&ps[0])).unwrap();
    replies.push(collect(&mut s).await);
    replies
}

async fn collect(
    s: &mut (impl futures::Stream<Item = TokenEvent> + Unpin),
) -> (String, FinishReason, u32) {
    let mut text = String::new();
    loop {
        match s.next().await.expect("stream ended without Finished") {
            TokenEvent::Text(t) => text += &t,
            TokenEvent::Finished { reason, usage, .. } => {
                return (text, reason, usage.completion_tokens);
            }
            TokenEvent::Error(e) => panic!("engine error: {e}"),
            _ => {}
        }
    }
}

fn free_ports(n: usize) -> Vec<String> {
    let ls: Vec<std::net::TcpListener> = (0..n)
        .map(|_| std::net::TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    ls.iter()
        .map(|l| l.local_addr().unwrap().to_string())
        .collect()
}

/// The output of one rank, then of two ranks (twice) must be the same, with `spec`.
async fn two_ranks_match_one(path: &std::path::Path, spec: SpecConfig) {
    two_ranks_match_one_with(path, spec, Wire::F32, 0, false, Overlap::Off).await;
}

/// [`two_ranks_match_one`] with `--tp-wire wire`, `--tp-seq-par seq_par`, (`chain`)
/// `--tp-mtp-chain` and `--tp-overlap`.
async fn two_ranks_match_one_with(
    path: &std::path::Path,
    spec: SpecConfig,
    wire: Wire,
    seq_par: usize,
    chain: bool,
    overlap: Overlap,
) {
    two_ranks_match_one_all(path, spec, wire, seq_par, chain, false, None, None, overlap).await;
}

/// [`two_ranks_match_one_with`] and (`local_rows`) `--tp-mtp-local-rows`; returns rank 0's
/// drafted and accepted draft counts.
async fn two_ranks_match_one_rows(
    path: &std::path::Path,
    spec: SpecConfig,
    wire: Wire,
    seq_par: usize,
    chain: bool,
    local_rows: bool,
) -> (u64, u64) {
    two_ranks_match_one_ep(path, spec, wire, seq_par, chain, local_rows, None, None).await
}

/// [`two_ranks_match_one_rows`], and with `ep` `--tp-moe-ep` with `--moe-kernel ep` (for
/// every run, one rank's too) and `place` its `--tp-moe-ep-place` placement.
#[allow(clippy::too_many_arguments)]
async fn two_ranks_match_one_ep(
    path: &std::path::Path,
    spec: SpecConfig,
    wire: Wire,
    seq_par: usize,
    chain: bool,
    local_rows: bool,
    ep: Option<MoeKernel>,
    place: Option<Arc<Placement>>,
) -> (u64, u64) {
    two_ranks_match_one_all(
        path,
        spec,
        wire,
        seq_par,
        chain,
        local_rows,
        ep,
        place,
        Overlap::Off,
    )
    .await
}

/// [`two_ranks_match_one_ep`] and `--tp-overlap overlap`.
#[allow(clippy::too_many_arguments)]
async fn two_ranks_match_one_all(
    path: &std::path::Path,
    spec: SpecConfig,
    wire: Wire,
    seq_par: usize,
    chain: bool,
    local_rows: bool,
    ep: Option<MoeKernel>,
    place: Option<Arc<Placement>>,
    overlap: Overlap,
) -> (u64, u64) {
    let config = |tp_net| LmConfig {
        spec: spec.clone(),
        moe_kernel: ep.unwrap_or_default(),
        ..config(path, tp_net)
    };
    let one = {
        let e = LmEngine::load(config(None)).unwrap();
        run_all(&e).await
    };
    assert!(
        one.iter().all(|r| r.2 == MAX_TOKENS),
        "generations too short"
    );

    let peers = free_ports(2);
    let cfg = |rank| TpNetConfig {
        rank,
        peers: peers.clone(),
        timeout: Duration::from_secs(60),
        wire,
        seq_par,
        mtp_chain: chain,
        mtp_local_rows: local_rows,
        moe_ep: ep.is_some(),
        overlap,
        moe_ep_place: place.clone(),
        ..Default::default()
    };
    // Rank 1 serves rank 0 until it disconnects; its load returns then.
    let follower = {
        let c = config(Some(cfg(1)));
        std::thread::spawn(move || LmEngine::load(c).err().map(|e| e.to_string()))
    };
    let lead = {
        let c = config(Some(cfg(0)));
        tokio::task::spawn_blocking(move || LmEngine::load(c))
            .await
            .unwrap()
            .unwrap()
    };
    for round in 0..2 {
        let two = run_all(&lead).await;
        for (i, (a, b)) in one.iter().zip(&two).enumerate() {
            assert_eq!(
                a,
                b,
                "{wire:?} wire, seq-par {seq_par}, chain {chain}, local rows {local_rows}, \
                 ep {ep:?}, placed {}, {overlap:?} overlap, round {round}, prompt {i}: two \
                 ranks differ from one",
                place.is_some()
            );
        }
    }
    let drafts = lead.stats().spec;
    if spec.enabled() {
        assert!(drafts.drafted > 0, "rank 0 drafted nothing");
    }
    drop(lead);
    let e = follower.join().unwrap().expect("rank 1 stops with rank 0");
    assert!(e.contains("rank 0 disconnected"), "{e}");
    (drafts.drafted, drafts.accepted)
}

#[tokio::test(flavor = "multi_thread")]
async fn two_ranks_give_the_output_of_one() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("q4.gguf");
    tiny::write_tp(&path, None, 11).unwrap();
    two_ranks_match_one(&path, SpecConfig::default()).await;
}

/// `--tp-seq-par 4` (#751): the prompt steps (16-token chunks, batches up to 40) run the
/// mixers on each rank's share of the tokens, with a reduce-scatter and an all-gather per sum;
/// the decode steps of fewer tokens keep the all-reduce. `--tp-wire bf16` too: this model's
/// sums are below `BF16_MIN_VALUES`, so they stay F32 and exact (the BF16 exchanges have
/// unit tests in `src/tp_net/seq_par/tests.rs`).
#[tokio::test(flavor = "multi_thread")]
async fn two_ranks_split_the_tokens_too() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("q4.gguf");
    tiny::write_tp(&path, None, 11).unwrap();
    two_ranks_match_one_with(
        &path,
        SpecConfig::default(),
        Wire::F32,
        4,
        false,
        Overlap::Off,
    )
    .await;
    two_ranks_match_one_with(
        &path,
        SpecConfig::default(),
        Wire::Bf16,
        4,
        false,
        Overlap::Off,
    )
    .await;
}

/// `--tp-overlap` (#758): the split layers in two micro-batches, the mixers only and the MoE
/// too, give the output of one rank.
#[tokio::test(flavor = "multi_thread")]
async fn two_ranks_split_the_tokens_in_micro_batches() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("q4.gguf");
    tiny::write_tp(&path, None, 11).unwrap();
    // The tiny prompt steps' micro-batches are a few tokens (#1176 leaves short steps whole).
    legion::tp_net::micro::set_min_cols(1);
    for overlap in [Overlap::Mixers, Overlap::Ffn] {
        for wire in [Wire::F32, Wire::Bf16] {
            two_ranks_match_one_with(&path, SpecConfig::default(), wire, 4, false, overlap).await;
        }
    }
}

/// `--spec mtp`: the MTP head's attention and MoE split like the target's layers.
#[tokio::test(flavor = "multi_thread")]
async fn two_ranks_draft_with_a_split_mtp_head() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("q4.gguf");
    let mtp = dir.path().join("mtp.gguf");
    tiny::write_tp(&path, Some(&mtp), 11).unwrap();
    let spec = SpecConfig {
        kind: SpecKind::Mtp,
        k: 3,
        fixed: true,
        p_min: 0.0,
        draft: Some(mtp),
        ..Default::default()
    };
    two_ranks_match_one(&path, spec).await;
}

/// `--tp-mtp-chain` (#771): rank 0 sends each draft round as one request and every rank runs
/// its further passes on its own, from the head's picks (the same bits on every rank). A
/// floor that cuts some rounds short and the adaptive depth (rounds of different caps) as well
/// as fixed rounds: the ranks must make the same passes (their sums would wait for each other
/// otherwise) and give one rank's output.
#[tokio::test(flavor = "multi_thread")]
async fn two_ranks_chain_the_draft_passes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("q4.gguf");
    let mtp = dir.path().join("mtp.gguf");
    tiny::write_tp(&path, Some(&mtp), 11).unwrap();
    // The tiny head gives its picks about 0.010 to 0.017.
    for (fixed, p_min) in [(true, 0.0), (true, 0.013), (false, 0.013)] {
        let spec = SpecConfig {
            kind: SpecKind::Mtp,
            k: 3,
            fixed,
            p_min,
            draft: Some(mtp.clone()),
            ..Default::default()
        };
        two_ranks_match_one_with(&path, spec, Wire::F32, 0, true, Overlap::Off).await;
    }
}

/// `--tp-mtp-local-rows` (#780): a prompt chunk's take-in pieces name rows of the step's
/// output and rank 1 reads its own copy (a debug build checks that its bits are rank 0's, so
/// a wrong range fails the request). The 16-token chunks go in an 8-token piece and the tail;
/// a chunk that leaves at most a KV block of its prompt is taken in by the next step (#477),
/// whose passes name the rows of the step before.
/// The output is one rank's, and the drafts are the ones sent rows give: the same drafted and
/// accepted counts, with chains too and with the tokens split (`--tp-seq-par`).
#[tokio::test(flavor = "multi_thread")]
async fn two_ranks_take_in_prompt_rows_from_their_own_copy() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("q4.gguf");
    let mtp = dir.path().join("mtp.gguf");
    tiny::write_tp(&path, Some(&mtp), 11).unwrap();
    let spec = SpecConfig {
        kind: SpecKind::Mtp,
        k: 3,
        fixed: true,
        p_min: 0.0,
        draft: Some(mtp),
        ..Default::default()
    };
    for (seq_par, chain) in [(0, false), (0, true), (4, false)] {
        let sent =
            two_ranks_match_one_rows(&path, spec.clone(), Wire::F32, seq_par, chain, false).await;
        let local =
            two_ranks_match_one_rows(&path, spec.clone(), Wire::F32, seq_par, chain, true).await;
        assert_eq!(
            sent, local,
            "seq-par {seq_par}, chain {chain}: drafts differ"
        );
    }
}

/// `--tp-moe-ep` (#798): each rank holds half the experts whole and runs only the pairs routed
/// to them; the all-reduce adds the halves. On ggml's `mul_mat_id` path (remapped ids, masked
/// weights) and on the fused op (its table skips the other half), plain and drafting with the
/// MTP head cut the same way, with the tokens split too.
#[tokio::test(flavor = "multi_thread")]
async fn two_ranks_split_the_experts() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("q4.gguf");
    let mtp = dir.path().join("mtp.gguf");
    tiny::write_tp(&path, Some(&mtp), 11).unwrap();
    let mtp_spec = SpecConfig {
        kind: SpecKind::Mtp,
        k: 3,
        fixed: true,
        p_min: 0.0,
        draft: Some(mtp),
        ..Default::default()
    };
    for kernel in [MoeKernel::Graph, MoeKernel::Fused] {
        // (spec, seq-par, --tp-mtp-chain and --tp-mtp-local-rows)
        for (spec, seq_par, mtp) in [
            (SpecConfig::default(), 0, false),
            (mtp_spec.clone(), 4, true),
        ] {
            let (wire, ep) = (Wire::F32, Some(kernel));
            two_ranks_match_one_ep(&path, spec, wire, seq_par, mtp, mtp, ep, None).await;
        }
    }
}

/// `--tp-moe-ep-place` (#803): the ranks hold the experts a placement gives them, scattered
/// and out of order on some layers and the MTP head (block `N_LAYER`), contiguous on the
/// others; still the output of one rank, on both expert paths.
#[tokio::test(flavor = "multi_thread")]
async fn two_ranks_split_the_experts_by_a_placement() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("q4.gguf");
    let mtp = dir.path().join("mtp.gguf");
    tiny::write_tp(&path, Some(&mtp), 11).unwrap();
    let mtp_spec = SpecConfig {
        kind: SpecKind::Mtp,
        k: 3,
        fixed: true,
        p_min: 0.0,
        draft: Some(mtp),
        ..Default::default()
    };
    let mut layers = std::collections::BTreeMap::new();
    layers.insert(1, vec![1, 0, 0, 1, 0, 1, 1, 0]);
    layers.insert(3, vec![1, 1, 1, 1, 0, 0, 0, 0]);
    layers.insert(tiny::N_LAYER, vec![0, 1, 0, 1, 0, 1, 0, 1]);
    let place = Placement {
        n_expert: 8,
        ranks: 2,
        layers,
    };
    place.check().unwrap();
    let place = Some(Arc::new(place));
    for kernel in [MoeKernel::Graph, MoeKernel::Fused] {
        for (spec, seq_par, mtp) in [
            (SpecConfig::default(), 0, false),
            (mtp_spec.clone(), 4, true),
        ] {
            let (wire, ep) = (Wire::F32, Some(kernel));
            two_ranks_match_one_ep(&path, spec, wire, seq_par, mtp, mtp, ep, place.clone()).await;
        }
    }
}

/// A model whose expert rows do not split in whole blocks is refused before connecting.
#[test]
fn an_uneven_model_is_refused_before_connecting() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("q4.gguf");
    tiny::write(&path, None, 11).unwrap();
    let c = config(
        &path,
        Some(TpNetConfig {
            rank: 1,
            peers: free_ports(2),
            timeout: Duration::from_secs(5),
            ..Default::default()
        }),
    );
    let e = LmEngine::load(c).err().expect("refused").to_string();
    assert!(e.contains("whole blocks"), "{e}");
}
