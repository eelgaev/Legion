//! Images under tensor parallelism over machines (#1195) for qwen4exp (Qwen3.8-Flash-Next)
//! through the engine on the CPU, ranks as threads over loopback: rank 0 alone loads the
//! vision tower and encodes, the other rank gets the images' rows and positions with the steps.
//! Two ranks give one rank's greedy answers (up to the rounding of the split sums, which the
//! tiny model's greedy picks do not feel) for two chats with images at once, a text chat, and
//! the first image chat again; in prompt chunks that cut the images, with the tokens split
//! over the ranks (`--tp-seq-par`), and with MTP drafts (`--tp-mtp-chain`,
//! `--tp-mtp-local-rows`), whose take-in reads the image rows too.

use std::path::Path;
use std::time::Duration;

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
use legion::engine::{
    Engine, GenerateRequest, ImageInput, Message, Prompt, Role, SamplingParams, TokenEvent,
};
use legion::model::qwen4::tiny;
use legion::spec::{SpecConfig, SpecKind};
use legion::tp_net::{TpNetConfig, Wire};
use legion::vision::tiny as tiny_mmproj;
use tokio_util::sync::CancellationToken;

/// The tiny model's width.
const E: u32 = 64;

const MAX_TOKENS: u32 = 24;

fn config(dir: &Path, spec: SpecConfig, tp_net: Option<TpNetConfig>) -> LmConfig {
    LmConfig {
        model_name: Some("tiny".into()),
        mmproj: Some(dir.join("mmproj.gguf")),
        ctx_size: 512,
        threads: 2,
        max_seqs: 2,
        max_batch_tokens: 40,
        prefill_chunk: 16,
        spec,
        use_ref: true,
        tp_net,
        ..LmConfig::new(dir.join("tiny.gguf"), DeviceChoice::Cpu)
    }
}

/// A 64 × 48 image: 8 × 6 merged patches of 8 × 8 pixels, 48 tokens.
fn image(seed: u8) -> ImageInput {
    ImageInput {
        bytes: tiny_mmproj::bmp(64, 48, |x, y| {
            [(x * 4) as u8 ^ seed, (y * 5) as u8, seed.wrapping_mul(31)]
        }),
        mime: "image/bmp".into(),
    }
}

fn request(images: Vec<ImageInput>) -> GenerateRequest {
    GenerateRequest {
        model: "tiny".into(),
        prompt: Prompt::Chat {
            system: None,
            messages: vec![Message {
                role: Some(Role::User),
                content: "what is in this picture, and what comes after it?".into(),
                images,
                ..Default::default()
            }],
            tools: vec![],
        },
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

/// (generated text, prompt tokens, completion tokens).
async fn collect(s: &mut (impl futures::Stream<Item = TokenEvent> + Unpin)) -> (String, u32, u32) {
    let mut text = String::new();
    loop {
        match s.next().await.expect("stream ended without Finished") {
            TokenEvent::Text(t) => text += &t,
            TokenEvent::Finished { usage, .. } => {
                return (text, usage.prompt_tokens, usage.completion_tokens);
            }
            TokenEvent::Error(e) => panic!("engine error: {e}"),
            _ => {}
        }
    }
}

/// Two image chats at once, a text chat, then the first image chat again.
async fn run_all(engine: &LmEngine) -> Vec<(String, u32, u32)> {
    let reqs = [vec![image(1)], vec![image(2)], vec![]];
    let streams: Vec<_> = (reqs.into_iter())
        .map(|i| engine.generate(request(i)).unwrap())
        .collect();
    let mut out = Vec::new();
    for mut s in streams {
        out.push(collect(&mut s).await);
    }
    let mut s = engine.generate(request(vec![image(1)])).unwrap();
    out.push(collect(&mut s).await);
    out
}

fn free_ports(n: usize) -> Vec<String> {
    let ls: Vec<std::net::TcpListener> = (0..n)
        .map(|_| std::net::TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    ls.iter()
        .map(|l| l.local_addr().unwrap().to_string())
        .collect()
}

/// One rank's answers, then two ranks' (twice), with `spec`, `--tp-seq-par seq_par` and (for
/// MTP) `--tp-mtp-chain` and `--tp-mtp-local-rows`.
async fn two_ranks_match_one(dir: &Path, spec: SpecConfig, seq_par: usize) {
    let one = {
        let e = LmEngine::load(config(dir, spec.clone(), None)).unwrap();
        assert!(e.models()[0].vision);
        run_all(&e).await
    };
    assert!(
        one.iter().all(|r| r.2 == MAX_TOKENS),
        "generations too short"
    );
    assert_ne!(one[0].0, one[1].0, "the image does not reach the answer");
    // `<|vision_start|>`, 48 image tokens, `<|vision_end|>`.
    assert_eq!(one[0].1, one[2].1 + 50);
    assert_eq!(one[0], one[3]);

    let peers = free_ports(2);
    let mtp = spec.enabled();
    let cfg = |rank| TpNetConfig {
        rank,
        peers: peers.clone(),
        timeout: Duration::from_secs(60),
        wire: Wire::F32,
        seq_par,
        mtp_chain: mtp,
        mtp_local_rows: mtp,
        ..Default::default()
    };
    // Rank 1 serves rank 0 until it disconnects; its load returns then. It is given the
    // projector too, as a serve command is: only rank 0 loads it.
    let follower = {
        let c = config(dir, spec.clone(), Some(cfg(1)));
        std::thread::spawn(move || LmEngine::load(c).err().map(|e| e.to_string()))
    };
    let lead = {
        let c = config(dir, spec.clone(), Some(cfg(0)));
        tokio::task::spawn_blocking(move || LmEngine::load(c))
            .await
            .unwrap()
            .unwrap()
    };
    assert!(lead.models()[0].vision);
    for round in 0..2 {
        let two = run_all(&lead).await;
        for (i, (a, b)) in one.iter().zip(&two).enumerate() {
            assert_eq!(
                a, b,
                "seq-par {seq_par}, mtp {mtp}, round {round}, request {i}: two ranks differ \
                 from one"
            );
        }
    }
    if mtp {
        assert!(lead.stats().spec.drafted > 0, "rank 0 drafted nothing");
    }
    drop(lead);
    let e = follower.join().unwrap().expect("rank 1 stops with rank 0");
    assert!(e.contains("rank 0 disconnected"), "{e}");
}

fn write(dir: &Path) {
    tiny::write_tp_vision(dir.join("tiny.gguf"), Some(&dir.join("mtp.gguf")), 11).unwrap();
    tiny_mmproj::write(dir.join("mmproj.gguf"), E, 3, false).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn two_ranks_answer_images_as_one() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path());
    two_ranks_match_one(dir.path(), SpecConfig::default(), 0).await;
    two_ranks_match_one(dir.path(), SpecConfig::default(), 4).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn two_ranks_draft_after_images_as_one() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path());
    let spec = SpecConfig {
        kind: SpecKind::Mtp,
        k: 3,
        fixed: true,
        p_min: 0.0,
        draft: Some(dir.path().join("mtp.gguf")),
        ..Default::default()
    };
    two_ranks_match_one(dir.path(), spec, 0).await;
}
