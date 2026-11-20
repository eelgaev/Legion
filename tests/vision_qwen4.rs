//! Images in Qwen3.8-Flash-Next (`qwen4exp`) prompts (#1195): the vision rows enter the
//! hyper-connection streams in place of the image tokens' embeddings, a chat with an image runs
//! through the engine with the same greedy answer with and without MTP drafts, and an HF
//! checkpoint's vision tower (`model.visual.*`) is the mmproj llama.cpp's converter writes.

use std::os::unix::fs::FileExt;
use std::path::Path;

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
use legion::engine::{
    Engine, GenerateRequest, ImageInput, Message, Prompt, Role, SamplingParams, TokenEvent,
};
use legion::ggml::{Backend, types};
use legion::gguf::{GgufFile, TensorInfo};
use legion::model::core::{Executor, Group, ImageSpan, KV_BLOCK, Logits, Media, Model, SeqInput};
use legion::model::qwen4::tiny;
use legion::spec::{SpecConfig, SpecKind};
use legion::vision::Encoder;
use legion::vision::tiny as tiny_mmproj;
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// The tiny model's width.
const E: usize = 64;

/// Token `t`'s embedding row (the tiny model's table is Q8_0).
fn embedding(g: &GgufFile, path: &Path, t: u32) -> Vec<f32> {
    let embd = g.tensor("token_embd.weight").unwrap();
    assert_eq!(embd.shape[0] as usize, E);
    let row = legion::ggml::row_size(types::Q8_0, E);
    let mut bytes = vec![0u8; row];
    std::fs::File::open(path)
        .unwrap()
        .read_exact_at(&mut bytes, embd.offset + t as u64 * row as u64)
        .unwrap();
    legion::ggml::dequantize(types::Q8_0, &bytes, E)
}

/// A 1 × 1 image whose row is its token's embedding is that token: the same positions, the
/// same input row, the same n-gram rows (the placeholder stays in the token stream, as the
/// reference hashes `image_token_id` there), so the same logits, bit for bit, at every
/// position. Another row moves them from the image on, not before.
#[test]
fn image_rows_replace_the_token_embeddings() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, None, 7).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let tokens: Vec<u32> = b"images in the streams".iter().map(|&b| b as u32).collect();
    let at = 5;
    let media = |row: Vec<f32>| {
        Media::new(
            vec![ImageSpan {
                start: at,
                nx: 1,
                ny: 1,
                rows: row.into(),
            }],
            E,
        )
    };
    let logits = |media: Option<&Media>| {
        let mut cpu = Backend::cpu().unwrap();
        cpu.set_threads(2);
        cpu.set_use_ref(true);
        let model = Model::load(&gguf, |_| cpu.buffer_type()).unwrap();
        let mut exec = Executor::new(model, cpu, 2, 1, 0).unwrap();
        let blocks = [0u32, 1];
        let g = Group {
            slot: 0,
            seqs: vec![SeqInput {
                tokens: &tokens,
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
    let bits = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    let text = logits(None);
    assert_eq!(text.len(), tokens.len());
    let same = logits(Some(&media(embedding(&gguf, &path, tokens[at]))));
    for (i, (a, b)) in text.iter().zip(&same).enumerate() {
        assert!(bits(a, b), "position {i}");
    }
    let other = logits(Some(&media(vec![0.25; E])));
    for (i, (a, b)) in text.iter().zip(&other).enumerate() {
        assert_eq!(bits(a, b), i < at, "position {i}");
    }
    assert!(tokens.len() < KV_BLOCK);
}

fn engine(dir: &Path, spec: SpecConfig, max_batch_tokens: usize) -> LmEngine {
    LmEngine::load(LmConfig {
        model_name: Some("tiny".into()),
        mmproj: Some(dir.join("mmproj.gguf")),
        ctx_size: 512,
        threads: 2,
        max_seqs: 2,
        max_batch_tokens,
        spec,
        use_ref: true,
        ..LmConfig::new(dir.join("tiny.gguf"), DeviceChoice::Cpu)
    })
    .unwrap()
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
        max_tokens: 24,
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

/// (generated text, prompt tokens, drafted tokens).
async fn run(engine: &LmEngine, req: GenerateRequest) -> (String, u32, u64) {
    let mut s = engine.generate(req).unwrap();
    let mut text = String::new();
    loop {
        match s.next().await.expect("stream ended without Finished") {
            TokenEvent::Text(t) => text += &t,
            TokenEvent::Finished { usage, .. } => {
                let drafted = usage.spec.map_or(0, |s| s.drafted);
                return (text, usage.prompt_tokens, drafted);
            }
            TokenEvent::Error(e) => panic!("engine error: {e}"),
            _ => {}
        }
    }
}

/// With a projector the model advertises vision; a chat with an image answers the same in one
/// prompt step and in chunks that cut the image, and with the MTP head drafting (its take-in
/// embeds the image rows at the target's positions).
#[tokio::test(flavor = "multi_thread")]
async fn chats_with_images_run_with_and_without_mtp() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    tiny::write_vision(d.join("tiny.gguf"), Some(&d.join("mtp.gguf")), 7).unwrap();
    tiny_mmproj::write(d.join("mmproj.gguf"), E as u32, 3, false).unwrap();
    let plain = engine(d, SpecConfig::default(), 512);
    assert!(plain.models()[0].vision);
    let text_only = plain.count_tokens(&request(vec![])).unwrap();
    let with_image = plain.count_tokens(&request(vec![image(1)])).unwrap();
    // `<|vision_start|>`, 48 image tokens, `<|vision_end|>`.
    assert_eq!(with_image, text_only + 50);
    let (want, prompt, _) = run(&plain, request(vec![image(1)])).await;
    assert_eq!(prompt, with_image);
    let (other, _, _) = run(&plain, request(vec![image(2)])).await;
    assert_ne!(want, other, "the image does not reach the answer");
    drop(plain);

    let chunked = engine(d, SpecConfig::default(), 16);
    assert_eq!(run(&chunked, request(vec![image(1)])).await.0, want);
    drop(chunked);

    for k in [1, 3] {
        let mtp = SpecConfig {
            kind: SpecKind::Mtp,
            k,
            draft: Some(d.join("mtp.gguf")),
            p_min: 0.0,
            ..Default::default()
        };
        for batch in [512, 16] {
            let e = engine(d, mtp.clone(), batch);
            let (got, _, drafted) = run(&e, request(vec![image(1)])).await;
            assert_eq!(got, want, "k {k}, batch {batch}");
            assert!(drafted > 0, "k {k}, batch {batch}: nothing drafted");
        }
    }
}

/// The tiny projector's tensors as a checkpoint ships them (`model.visual.*`, row-major dims,
/// the two frames' patch kernels as one Conv3D kernel), with the configs the converter reads.
fn checkpoint_from(mmproj: &GgufFile, mmproj_path: &Path, dir: &Path) {
    let read = |t: &TensorInfo| {
        let mut v = vec![0u8; t.size as usize];
        std::fs::File::open(mmproj_path)
            .unwrap()
            .read_exact_at(&mut v, t.offset)
            .unwrap();
        v
    };
    let tensor = |name: &str| mmproj.tensor(name).unwrap();
    let hf_shape = |t: &TensorInfo| t.shape.iter().rev().map(|&d| d as usize).collect();
    let mut out: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    // [n, 3, 2, p, p]: per output channel and colour, frame 0's kernel, then frame 1's.
    let (k0, k1) = (
        tensor("v.patch_embd.weight"),
        tensor("v.patch_embd.weight.1"),
    );
    let plane = (k0.shape[0] * k0.shape[1]) as usize * 4;
    let (a, b) = (read(&k0), read(&k1));
    let conv: Vec<u8> = a
        .chunks(plane)
        .zip(b.chunks(plane))
        .flat_map(|(x, y)| x.iter().chain(y).copied())
        .collect();
    let [p, _, c, n] = k0.shape[..] else {
        panic!("{:?}", k0.shape)
    };
    out.push((
        "patch_embed.proj.weight".into(),
        vec![n as usize, c as usize, 2, p as usize, p as usize],
        conv,
    ));
    let mut renames = vec![
        ("v.patch_embd.bias", "patch_embed.proj.bias".to_string()),
        ("v.position_embd.weight", "pos_embed.weight".into()),
        ("v.post_ln.weight", "merger.norm.weight".into()),
        ("v.post_ln.bias", "merger.norm.bias".into()),
        ("mm.0.weight", "merger.linear_fc1.weight".into()),
        ("mm.0.bias", "merger.linear_fc1.bias".into()),
        ("mm.2.weight", "merger.linear_fc2.weight".into()),
        ("mm.2.bias", "merger.linear_fc2.bias".into()),
    ];
    let names = [
        ("ln1", "norm1"),
        ("ln2", "norm2"),
        ("attn_qkv", "attn.qkv"),
        ("attn_out", "attn.proj"),
        ("ffn_up", "mlp.linear_fc1"),
        ("ffn_down", "mlp.linear_fc2"),
    ];
    let mut owned = Vec::new();
    for il in 0..2 {
        for (from, to) in names {
            for wb in ["weight", "bias"] {
                owned.push((
                    format!("v.blk.{il}.{from}.{wb}"),
                    format!("blocks.{il}.{to}.{wb}"),
                ));
            }
        }
    }
    for (from, to) in &owned {
        renames.push((from.as_str(), to.clone()));
    }
    for (from, to) in renames {
        let t = tensor(from);
        out.push((to, hf_shape(&t), read(&t)));
    }
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, bytes) in &out {
        header.insert(
            format!("model.visual.{name}"),
            json!({"dtype": "F32", "shape": shape,
                   "data_offsets": [data.len(), data.len() + bytes.len()]}),
        );
        data.extend_from_slice(bytes);
    }
    let mut h = serde_json::to_vec(&header).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut file = (h.len() as u64).to_le_bytes().to_vec();
    file.extend(h);
    file.extend(data);
    std::fs::write(dir.join("model.safetensors"), file).unwrap();
    let e = tiny_mmproj::N_EMBD;
    let config = json!({
        "model_type": "qwen4_exp",
        "text_config": {"hidden_size": E, "rms_norm_eps": 1e-6},
        "vision_config": {
            "depth": 2, "hidden_size": e, "intermediate_size": 32, "num_heads": 2,
            "patch_size": tiny_mmproj::PATCH, "spatial_merge_size": 2, "temporal_patch_size": 2,
            "num_position_embeddings": 64, "out_hidden_size": E, "deepstack_visual_indexes": []
        }
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let pre = json!({"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]});
    std::fs::write(dir.join("preprocessor_config.json"), pre.to_string()).unwrap();
}

/// A checkpoint's tower opens as the mmproj llama.cpp's converter writes from it: the same
/// metadata the encoder reads and the same rows for an image, bit for bit.
#[test]
fn checkpoint_tower_is_the_converted_mmproj() {
    let dir = tempfile::tempdir().unwrap();
    let mmproj_path = dir.path().join("mmproj.gguf");
    tiny_mmproj::write(&mmproj_path, E as u32, 5, false).unwrap();
    let mmproj = GgufFile::open(&mmproj_path).unwrap();
    let ckpt = dir.path().join("ckpt");
    std::fs::create_dir(&ckpt).unwrap();
    checkpoint_from(&mmproj, &mmproj_path, &ckpt);
    let tower = legion::hf::open_vision(&ckpt)
        .unwrap()
        .expect("a vision tower");
    for t in mmproj.tensors() {
        let s = tower
            .tensor(&t.name)
            .unwrap_or_else(|| panic!("{}", t.name));
        assert_eq!(
            (s.shape.as_slice(), s.ggml_type),
            (t.shape.as_slice(), t.ggml_type)
        );
    }
    let encode = |g: &GgufFile, img: &ImageInput| {
        let mut cpu = Backend::cpu().unwrap();
        cpu.set_threads(2);
        let mut enc = Encoder::from_gguf(g, cpu).unwrap();
        enc.encode_bytes(&img.bytes).unwrap()
    };
    for img in [image(1), image(9)] {
        let (a, b) = (encode(&mmproj, &img), encode(&tower, &img));
        assert_eq!((a.nx, a.ny, a.n_embd), (b.nx, b.ny, b.n_embd));
        assert_eq!(a.n_embd, E);
        assert!(
            a.data
                .iter()
                .zip(&b.data)
                .all(|(x, y)| x.to_bits() == y.to_bits())
        );
    }
    // A dense Qwen3.5 / Qwen3.8-27B checkpoint (`qwen3_5`, #1218) ships the same tower.
    let config = ckpt.join("config.json");
    let mut c: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&config).unwrap()).unwrap();
    c["model_type"] = json!("qwen3_5");
    std::fs::write(&config, c.to_string()).unwrap();
    let dense = legion::hf::open_vision(&ckpt)
        .unwrap()
        .expect("a qwen3_5 vision tower");
    for t in mmproj.tensors() {
        let s = dense.tensor(&t.name).unwrap();
        assert_eq!(
            (s.shape.as_slice(), s.ggml_type),
            (t.shape.as_slice(), t.ggml_type)
        );
    }
    let (a, b) = (encode(&mmproj, &image(3)), encode(&dense, &image(3)));
    assert!(
        a.data
            .iter()
            .zip(&b.data)
            .all(|(x, y)| x.to_bits() == y.to_bits())
    );
    // Not a vision checkpoint: none.
    let text = dir.path().join("text");
    std::fs::create_dir(&text).unwrap();
    std::fs::copy(
        ckpt.join("model.safetensors"),
        text.join("model.safetensors"),
    )
    .unwrap();
    std::fs::write(
        text.join("config.json"),
        json!({"model_type": "qwen4_exp"}).to_string(),
    )
    .unwrap();
    assert!(legion::hf::open_vision(&text).unwrap().is_none());
}
