//! Images in prompts (#146): the vision tower's rows enter the language model in place of the
//! image tokens' embeddings, a chat with images runs end to end through the engine, and the
//! prefix cache tells images apart.

use std::os::unix::fs::FileExt;

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
use legion::engine::{
    Engine, EngineError, GenerateRequest, ImageInput, Message, Prompt, Role, SamplingParams,
    TokenEvent,
};
use legion::ggml::Backend;
use legion::gguf::GgufFile;
use legion::model::core::{Executor, Group, ImageSpan, KV_BLOCK, Logits, Media, Model, SeqInput};
use legion::model::qwen35::tiny::{self, TinyConfig};
use legion::vision::tiny as tiny_mmproj;
use tokio_util::sync::CancellationToken;

/// HF's content parts, as Qwen's template renders them (one placeholder per image part).
const TEMPLATE: &str = "{%- for m in messages %}<|im_start|>{{ m.role }}\n\
{%- if m.content is string %}{{ m.content }}\
{%- else %}{%- for p in m.content %}\
{%- if p.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>\
{%- else %}{{ p.text }}{%- endif %}{%- endfor %}{%- endif %}<|im_end|>\n\
{%- endfor %}{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}";

const VISION_TOKENS: &[&str] = &["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"];

fn tiny_cfg() -> TinyConfig {
    TinyConfig {
        user_tokens: VISION_TOKENS,
        chat_template: Some(TEMPLATE),
        ..Default::default()
    }
}

/// A 1 × 1 image whose row is token `t`'s embedding is token `t`: the same positions (a one-token
/// image advances them by one) and the same input row, so the same logits, bit for bit, at
/// every position.
#[test]
fn image_rows_replace_the_token_embeddings() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, &tiny_cfg()).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let embd = gguf.tensor("token_embd.weight").unwrap();
    let n_embd = embd.shape[0] as usize;
    let tokens: Vec<u32> = b"vision rows!".iter().map(|&b| b as u32).collect();
    let at = 4;
    let mut row = vec![0u8; n_embd * 4];
    std::fs::File::open(&path)
        .unwrap()
        .read_exact_at(
            &mut row,
            embd.offset + (tokens[at] as u64) * (n_embd as u64 * 4),
        )
        .unwrap();
    let row: Vec<f32> = row
        .chunks(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .collect();
    let media = Media::new(
        vec![ImageSpan {
            start: at,
            nx: 1,
            ny: 1,
            rows: row.into(),
        }],
        n_embd,
    );
    let mut with_image = tokens.clone();
    with_image[at] = 0;
    let logits = |tokens: &[u32], media: Option<&Media>| {
        let mut cpu = Backend::cpu().unwrap();
        cpu.set_threads(2);
        cpu.set_use_ref(true);
        let model = Model::load(&gguf, |_| cpu.buffer_type()).unwrap();
        let mut exec = Executor::new(model, cpu, 2, 1, 0).unwrap();
        let blocks = [0u32, 1];
        let g = Group {
            slot: 0,
            seqs: vec![SeqInput {
                tokens,
                n_past: 0,
                blocks: &blocks,
                ring: 0,
                logits: Logits::All,
                media,
            }],
            verify: false,
        };
        exec.step(std::slice::from_ref(&g)).unwrap().rows
    };
    let text = logits(&tokens, None);
    let image = logits(&with_image, Some(&media));
    assert_eq!(text.len(), tokens.len());
    for (i, (a, b)) in text.iter().zip(&image).enumerate() {
        assert!(
            a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "position {i}"
        );
    }
    assert!(tokens.len() < KV_BLOCK);
}

fn engine(dir: &std::path::Path, mmproj: bool) -> LmEngine {
    let path = dir.join("tiny.gguf");
    tiny::write(&path, &tiny_cfg()).unwrap();
    let mmproj = mmproj.then(|| {
        let p = dir.join("mmproj.gguf");
        tiny_mmproj::write(&p, tiny_cfg().n_embd, 3, false).unwrap();
        p
    });
    LmEngine::load(LmConfig {
        model_name: Some("tiny".into()),
        mmproj,
        ctx_size: 512,
        threads: 2,
        max_seqs: 2,
        use_ref: true,
        ..LmConfig::new(&path, DeviceChoice::Cpu)
    })
    .unwrap()
}

/// A 64 × 64 image: 8 × 8 merged patches of 8 × 8 pixels, 64 tokens.
fn image(seed: u8) -> ImageInput {
    ImageInput {
        bytes: tiny_mmproj::bmp(64, 64, |x, y| {
            [(x * 4) as u8 ^ seed, (y * 4) as u8, seed.wrapping_mul(31)]
        }),
        mime: "image/bmp".into(),
    }
}

/// The image's first token: after `<|im_start|>`, `user\n` (byte tokens) and `<|vision_start|>`
/// (a message's images come before its text).
const IMAGE_START: u32 = 7;
const TEXT: &str = "A question long enough for the prompt to fill a few KV blocks: ";

fn request(images: Vec<ImageInput>) -> GenerateRequest {
    GenerateRequest {
        model: "tiny".into(),
        prompt: Prompt::Chat {
            system: None,
            messages: vec![Message {
                role: Some(Role::User),
                content: format!("{TEXT}what is in it?"),
                images,
                ..Default::default()
            }],
            tools: vec![],
        },
        sampling: SamplingParams {
            temperature: 0.0,
            ..Default::default()
        },
        max_tokens: 6,
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

/// (generated text, prompt tokens, cached prompt tokens).
async fn run(engine: &LmEngine, req: GenerateRequest) -> (String, u32, u32) {
    let mut s = engine.generate(req).unwrap();
    let mut text = String::new();
    loop {
        match s.next().await.expect("stream ended without Finished") {
            TokenEvent::Text(t) => text += &t,
            TokenEvent::Finished { usage, .. } => {
                return (text, usage.prompt_tokens, usage.cached_tokens);
            }
            TokenEvent::Error(e) => panic!("engine error: {e}"),
            _ => {}
        }
    }
}

/// With `--mmproj`, the model advertises vision; an image is its placeholder's 64 tokens; the
/// same request again resumes from the prefix cache past the image and answers the same, and
/// the same text with another image resumes no further than the image's start (nothing: the
/// cache resumes whole blocks).
#[tokio::test(flavor = "multi_thread")]
async fn chats_with_images_run_and_the_prefix_cache_tells_images_apart() {
    let dir = tempfile::tempdir().unwrap();
    let engine = engine(dir.path(), true);
    assert!(engine.models()[0].vision);
    let text_only = engine.count_tokens(&request(vec![])).unwrap();
    let with_image = engine.count_tokens(&request(vec![image(1)])).unwrap();
    // `<|vision_start|>`, 64 image tokens, `<|vision_end|>`.
    assert_eq!(with_image, text_only + 66);

    let (first, prompt, cached) = run(&engine, request(vec![image(1)])).await;
    assert_eq!((prompt, cached), (with_image, 0));
    let (again, _, cached) = run(&engine, request(vec![image(1)])).await;
    assert_eq!(again, first);
    assert!(cached >= IMAGE_START + 64, "cached {cached} of {prompt}");
    let (_, _, cached) = run(&engine, request(vec![image(2)])).await;
    assert!(
        cached <= IMAGE_START,
        "another image resumed {cached} tokens"
    );
}

/// Without `--mmproj`, the model takes no images.
#[tokio::test(flavor = "multi_thread")]
async fn images_need_a_projector() {
    let dir = tempfile::tempdir().unwrap();
    let engine = engine(dir.path(), false);
    assert!(!engine.models()[0].vision);
    match engine.generate(request(vec![image(1)])) {
        Err(EngineError::InvalidRequest(e)) => assert!(e.contains("images"), "{e}"),
        Err(e) => panic!("{e}"),
        Ok(_) => panic!("accepted an image"),
    }
}
