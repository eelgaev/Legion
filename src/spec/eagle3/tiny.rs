//! A tiny random EAGLE-3 checkpoint (safetensors + `config.json`) for tests that need the head
//! without a real one: for a target of width 64 and 259 tokens (see
//! [`crate::model::qwen35::tiny`]), one layer of 4 heads of 16, a draft vocabulary of every
//! second byte token. Its outputs are meaningless; what it computes is what a real head does.

use std::path::Path;

use serde_json::{Value, json};

use crate::engine::sampling::Rng;

/// Shapes and variants of the tiny head.
#[derive(Debug, Clone, PartialEq)]
pub struct TinyEagle3 {
    pub hidden: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    /// The target's tokens.
    pub vocab: usize,
    pub draft_vocab: usize,
    /// `eagle_aux_hidden_state_layer_ids`, if any.
    pub layer_ids: Option<Vec<usize>>,
    pub seed: u64,
    /// Ship `embed_tokens` (else the head shares the target's).
    pub embed: bool,
    /// Ship `d2t` and `t2d` (needs `draft_vocab` below `vocab`).
    pub map: bool,
    /// Matrices as BF16 instead of F32.
    pub bf16: bool,
    pub attention_bias: bool,
    pub qk_norm: bool,
    pub norm_before_residual: bool,
    pub norm_before_fc: bool,
    pub fc_norm: bool,
    pub norm_output: bool,
    pub llama3_rope: bool,
    /// Tensor names under `midlayer.` (else `layers.0.`).
    pub midlayer: bool,
}

impl Default for TinyEagle3 {
    fn default() -> Self {
        Self {
            hidden: 64,
            n_ff: 96,
            n_head: 4,
            n_kv_head: 2,
            head_dim: 16,
            vocab: 259,
            draft_vocab: 128,
            layer_ids: Some(vec![0, 1, 3]),
            seed: 7,
            embed: true,
            map: true,
            bf16: false,
            attention_bias: false,
            qk_norm: false,
            norm_before_residual: false,
            norm_before_fc: false,
            fc_norm: false,
            norm_output: false,
            llama3_rope: false,
            midlayer: true,
        }
    }
}

/// Draft id `i` is target token `2 i` (byte tokens 0, 2, 4, ...).
pub fn tiny_offsets(draft_vocab: usize) -> Vec<i64> {
    (0..draft_vocab as i64).collect()
}

struct Tensor {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn bf16_bytes(x: f32) -> [u8; 2] {
    // Round to nearest even.
    let b = x.to_bits();
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1));
    ((r >> 16) as u16).to_le_bytes()
}

/// Writes `config.json` and `model.safetensors` into the existing directory `dir`.
pub fn write(dir: impl AsRef<Path>, c: &TinyEagle3) -> std::io::Result<()> {
    let dir = dir.as_ref();
    let mut rng = Rng::new(c.seed);
    let mut tensors: Vec<Tensor> = Vec::new();
    let mut matrix = |name: &str, out: usize, inp: usize| {
        let a = 1.0 / (inp as f32).sqrt();
        let vals = (0..out * inp).map(|_| (rng.next_f64() as f32 * 2.0 - 1.0) * a);
        let (dtype, bytes) = if c.bf16 {
            ("BF16", vals.flat_map(bf16_bytes).collect())
        } else {
            ("F32", vals.flat_map(f32::to_le_bytes).collect())
        };
        tensors.push(Tensor {
            name: name.into(),
            dtype,
            shape: vec![out, inp],
            bytes,
        });
    };
    let prefix = if c.midlayer { "midlayer" } else { "layers.0" };
    let (h, nh, nkv, hd) = (c.hidden, c.n_head, c.n_kv_head, c.head_dim);
    matrix("fc.weight", h, 3 * h);
    matrix(&format!("{prefix}.self_attn.q_proj.weight"), nh * hd, 2 * h);
    matrix(
        &format!("{prefix}.self_attn.k_proj.weight"),
        nkv * hd,
        2 * h,
    );
    matrix(
        &format!("{prefix}.self_attn.v_proj.weight"),
        nkv * hd,
        2 * h,
    );
    matrix(&format!("{prefix}.self_attn.o_proj.weight"), h, nh * hd);
    matrix(&format!("{prefix}.mlp.gate_proj.weight"), c.n_ff, h);
    matrix(&format!("{prefix}.mlp.up_proj.weight"), c.n_ff, h);
    matrix(&format!("{prefix}.mlp.down_proj.weight"), h, c.n_ff);
    matrix("lm_head.weight", c.draft_vocab, h);
    if c.embed {
        matrix("embed_tokens.weight", c.vocab, h);
    }
    let mut rng = Rng::new(c.seed ^ 0x5eed);
    let mut vector = |name: String, n: usize, around: f32, spread: f32| {
        let bytes = (0..n)
            .flat_map(|_| (around + (rng.next_f64() as f32 * 2.0 - 1.0) * spread).to_le_bytes())
            .collect();
        tensors.push(Tensor {
            name,
            dtype: "F32",
            shape: vec![n],
            bytes,
        });
    };
    for n in ["input_layernorm", "hidden_norm", "post_attention_layernorm"] {
        vector(format!("{prefix}.{n}.weight"), h, 1.0, 0.1);
    }
    vector("norm.weight".into(), h, 1.0, 0.1);
    if c.attention_bias {
        vector(format!("{prefix}.self_attn.q_proj.bias"), nh * hd, 0.0, 0.2);
        vector(
            format!("{prefix}.self_attn.k_proj.bias"),
            nkv * hd,
            0.0,
            0.2,
        );
        vector(
            format!("{prefix}.self_attn.v_proj.bias"),
            nkv * hd,
            0.0,
            0.2,
        );
    }
    if c.qk_norm {
        vector(format!("{prefix}.self_attn.q_norm.weight"), hd, 1.0, 0.1);
        vector(format!("{prefix}.self_attn.k_norm.weight"), hd, 1.0, 0.1);
    }
    if c.norm_before_fc {
        vector("input_norm.weight".into(), 3 * h, 1.0, 0.1);
    }
    if c.fc_norm {
        for i in 0..3 {
            vector(format!("fc_norm.{i}.weight"), h, 1.0, 0.1);
        }
    }
    if c.map {
        assert!(c.draft_vocab < c.vocab);
        let offsets = tiny_offsets(c.draft_vocab);
        tensors.push(Tensor {
            name: "d2t".into(),
            dtype: "I64",
            shape: vec![c.draft_vocab],
            bytes: offsets.iter().flat_map(|o| o.to_le_bytes()).collect(),
        });
        let mut mask = vec![0u8; c.vocab];
        for (i, o) in offsets.iter().enumerate() {
            mask[(i as i64 + o) as usize] = 1;
        }
        tensors.push(Tensor {
            name: "t2d".into(),
            dtype: "BOOL",
            shape: vec![c.vocab],
            bytes: mask,
        });
    }
    write_safetensors(&dir.join("model.safetensors"), &tensors)?;

    let mut config = json!({
        "architectures": ["LlamaForCausalLMEagle3"],
        "num_hidden_layers": 1,
        "hidden_size": h,
        "intermediate_size": c.n_ff,
        "num_attention_heads": nh,
        "num_key_value_heads": nkv,
        "head_dim": hd,
        "vocab_size": c.vocab,
        "draft_vocab_size": c.draft_vocab,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 4096,
        "hidden_act": "silu",
        "attention_bias": c.attention_bias,
        "norm_before_residual": c.norm_before_residual,
        "norm_before_fc": c.norm_before_fc,
        "fc_norm": c.fc_norm,
        "norm_output": c.norm_output,
    });
    if let Some(ids) = &c.layer_ids {
        config["eagle_config"] = json!({"eagle_aux_hidden_state_layer_ids": ids});
    }
    if c.llama3_rope {
        config["rope_scaling"] = json!({
            "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 512,
        });
    }
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec_pretty(&config).expect("json"),
    )
}

fn write_safetensors(path: &Path, tensors: &[Tensor]) -> std::io::Result<()> {
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), json!({"format": "pt"}));
    let mut data = Vec::new();
    for t in tensors {
        let begin = data.len();
        data.extend_from_slice(&t.bytes);
        header.insert(
            t.name.clone(),
            json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [begin, data.len()]}),
        );
    }
    let header = serde_json::to_vec(&Value::Object(header)).expect("json");
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header);
    out.extend_from_slice(&data);
    std::fs::write(path, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::Checkpoint;

    #[test]
    fn written_checkpoint_reads_back() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), &TinyEagle3::default()).unwrap();
        let ck = Checkpoint::open(dir.path()).unwrap();
        assert!(ck.config().is_some());
        let names: Vec<String> = ck.tensors().into_iter().map(|t| t.name).collect();
        for n in [
            "fc.weight",
            "d2t",
            "t2d",
            "embed_tokens.weight",
            "lm_head.weight",
        ] {
            assert!(names.iter().any(|x| x == n), "{n} in {names:?}");
        }
    }

    #[test]
    fn bf16_rounds_to_nearest() {
        assert_eq!(bf16_bytes(1.0), 0x3f80u16.to_le_bytes());
        assert_eq!(bf16_bytes(-2.5), 0xc020u16.to_le_bytes());
    }
}
