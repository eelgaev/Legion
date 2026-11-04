//! Loading an EAGLE-3 checkpoint: the head's matrices in ggml backend memory, the token
//! embeddings kept on the host (rows are looked up per token and fed to the graph as input).

use std::collections::HashMap;
use std::os::unix::fs::FileExt;
use std::path::Path;

use crate::ggml::{self, Backend, Context, Tensor, ffi, types};
use crate::gguf::GgufFile;
use crate::safetensors::{Checkpoint, Dtype, TensorInfo};
use crate::spec::TargetInfo;

use super::config::{Eagle3Config, VocabMap};

/// Where the token embeddings come from.
enum Embed {
    /// `embed_tokens.weight` of the checkpoint.
    Checkpoint { name: String, ty: Dtype },
    /// The target's `token_embd.weight`, read row by row from its GGUF.
    Gguf {
        file: std::fs::File,
        offset: u64,
        ty: ffi::ggml_type,
        row_bytes: usize,
    },
}

pub struct Weights {
    pub cfg: Eagle3Config,
    pub vocab: VocabMap,
    ctx: Context,
    embed: Embed,
    /// Rows of the embedding table.
    embed_rows: usize,
    ckpt: Checkpoint,
}

/// A tensor to load: its name in the graph, its checkpoint name, its ggml shape.
struct Spec {
    name: String,
    src: String,
    ne: Vec<i64>,
    /// A matrix stays in its file type (F32, F16, BF16); the rest is converted to F32.
    matrix: bool,
}

fn float_type(d: &Dtype) -> Option<ffi::ggml_type> {
    match d {
        Dtype::F32 => Some(types::F32),
        Dtype::F16 => Some(types::F16),
        Dtype::BF16 => Some(types::BF16),
        _ => None,
    }
}

/// Converts a tensor's little-endian bytes to f32.
fn to_f32(d: &Dtype, bytes: &[u8]) -> Result<Vec<f32>, String> {
    Ok(match d {
        Dtype::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        Dtype::F64 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c) as f32)
            .collect(),
        Dtype::F16 | Dtype::BF16 => {
            let ty = float_type(d).unwrap();
            ggml::dequantize(ty, bytes, bytes.len() / 2)
        }
        other => return Err(format!("a {other} tensor where a float one is needed")),
    })
}

type Index = HashMap<String, TensorInfo>;

/// A checkpoint tensor by name, which may sit under `model.`.
fn find<'a>(index: &'a Index, name: &str) -> Option<(String, &'a TensorInfo)> {
    [name.to_string(), format!("model.{name}")]
        .into_iter()
        .find_map(|n| index.get(&n).map(|t| (n, t)))
}

impl Weights {
    pub fn load(draft: &Path, info: &TargetInfo, backend: &Backend) -> Result<Self, String> {
        let ckpt = Checkpoint::open(draft).map_err(|e| e.to_string())?;
        let json = ckpt
            .config()
            .ok_or_else(|| format!("{}: no config.json", draft.display()))?;
        let cfg = Eagle3Config::parse(json)?;
        if cfg.target_hidden != info.n_embd {
            return Err(format!(
                "the head reads {}-wide target hidden states, the target's are {} wide",
                cfg.target_hidden, info.n_embd
            ));
        }
        let index: Index = ckpt
            .tensors()
            .into_iter()
            .map(|t| (t.name.clone(), t))
            .collect();
        let find = |name: &str| find(&index, name);
        let layer = ["midlayer", "layers.0"]
            .into_iter()
            .find(|p| find(&format!("{p}.input_layernorm.weight")).is_some())
            .ok_or("the checkpoint has no decoder layer (`midlayer.*` or `layers.0.*`)")?;

        let (h, nh, nkv, hd) = (
            cfg.hidden as i64,
            cfg.n_head as i64,
            cfg.n_kv_head as i64,
            cfg.head_dim as i64,
        );
        let (ff, ht, na) = (cfg.n_ff as i64, cfg.target_hidden as i64, cfg.n_aux as i64);
        let mut plan = Vec::new();
        let mut add = |name: &str, src: String, ne: &[i64], matrix: bool| {
            plan.push(Spec {
                name: name.into(),
                src,
                ne: ne.to_vec(),
                matrix,
            });
        };
        let l = |n: &str| format!("{layer}.{n}.weight");
        add("fc", "fc.weight".into(), &[na * ht, h], true);
        add("wq", l("self_attn.q_proj"), &[2 * h, nh * hd], true);
        add("wk", l("self_attn.k_proj"), &[2 * h, nkv * hd], true);
        add("wv", l("self_attn.v_proj"), &[2 * h, nkv * hd], true);
        add("wo", l("self_attn.o_proj"), &[nh * hd, h], true);
        add("gate", l("mlp.gate_proj"), &[h, ff], true);
        add("up", l("mlp.up_proj"), &[h, ff], true);
        add("down", l("mlp.down_proj"), &[ff, h], true);
        add(
            "lm_head",
            "lm_head.weight".into(),
            &[h, cfg.draft_vocab as i64],
            true,
        );
        add("input_layernorm", l("input_layernorm"), &[h], false);
        add("hidden_norm", l("hidden_norm"), &[h], false);
        add(
            "post_attention_layernorm",
            l("post_attention_layernorm"),
            &[h],
            false,
        );
        add("norm", "norm.weight".into(), &[h], false);
        if cfg.norm_before_fc {
            add("input_norm", "input_norm.weight".into(), &[na * ht], false);
        }
        if cfg.fc_norm {
            for i in 0..cfg.n_aux {
                add(
                    &format!("fc_norm.{i}"),
                    format!("fc_norm.{i}.weight"),
                    &[ht],
                    false,
                );
            }
        }
        // Optional by presence: projection biases and Qwen3-style q/k norms.
        for (name, src, ne) in [
            ("bq", "self_attn.q_proj.bias", nh * hd),
            ("bk", "self_attn.k_proj.bias", nkv * hd),
            ("bv", "self_attn.v_proj.bias", nkv * hd),
            ("bo", "self_attn.o_proj.bias", h),
            ("q_norm", "self_attn.q_norm.weight", hd),
            ("k_norm", "self_attn.k_norm.weight", hd),
        ] {
            let src = format!("{layer}.{src}");
            if find(&src).is_some() {
                add(name, src, &[ne], false);
            }
        }

        let factors = cfg.rope_factors();
        let ctx = Context::new(plan.len() + 1, 0);
        let mut loaded = Vec::new();
        for spec in &plan {
            let (src, t) = find(&spec.src)
                .ok_or_else(|| format!("the checkpoint has no tensor `{}`", spec.src))?;
            let want: Vec<usize> = spec.ne.iter().rev().map(|&d| d as usize).collect();
            if t.shape != want {
                return Err(format!(
                    "`{src}` has shape {:?}, the config needs {want:?}",
                    t.shape
                ));
            }
            let ty = if spec.matrix {
                float_type(&t.dtype)
                    .ok_or_else(|| format!("`{src}` is {}, not F32, F16 or BF16", t.dtype))?
            } else {
                types::F32
            };
            let tensor = ctx.new_tensor(ty, &spec.ne).named(&spec.name);
            loaded.push((tensor, src, t.dtype.clone()));
        }
        let rope = factors.as_ref().map(|f| {
            ctx.new_tensor(types::F32, &[f.len() as i64])
                .named("rope_freqs")
        });
        ctx.alloc_tensors(backend.buffer_type())
            .map_err(|e| format!("cannot allocate the head's weights: {e}"))?;
        for (tensor, src, dtype) in &loaded {
            let bytes = ckpt.data(src).map_err(|e| e.to_string())?;
            if tensor.ty() == types::F32 && *dtype != Dtype::F32 {
                tensor.set_f32(&to_f32(dtype, bytes)?);
            } else {
                // The same type as in the file; checkpoints are little endian, as the host is.
                tensor.set_bytes(bytes);
            }
        }
        if let (Some(t), Some(f)) = (rope, &factors) {
            t.set_f32(f);
        }

        let (vocab, embed, embed_rows) = Self::read_vocab(&ckpt, &cfg, &index, info)?;
        Ok(Self {
            cfg,
            vocab,
            ctx,
            embed,
            embed_rows,
            ckpt,
        })
    }

    #[allow(clippy::type_complexity)]
    fn read_vocab(
        ckpt: &Checkpoint,
        cfg: &Eagle3Config,
        index: &Index,
        info: &TargetInfo,
    ) -> Result<(VocabMap, Embed, usize), String> {
        let find = |name: &str| find(index, name);
        let vocab = match find("d2t") {
            Some((name, t)) => {
                if t.shape != [cfg.draft_vocab] {
                    return Err(format!(
                        "`{name}` has shape {:?}, draft_vocab_size is {}",
                        t.shape, cfg.draft_vocab
                    ));
                }
                let bytes = ckpt.data(&name).map_err(|e| e.to_string())?;
                let offsets: Vec<i64> = match t.dtype {
                    Dtype::I64 => bytes
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .map(|c| i64::from_le_bytes(*c))
                        .collect(),
                    Dtype::I32 => bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| i32::from_le_bytes(*c) as i64)
                        .collect(),
                    ref d => return Err(format!("`{name}` is {d}, not I64 or I32")),
                };
                let mask = match find("t2d") {
                    Some((name, t)) => {
                        if t.shape != [cfg.vocab] {
                            return Err(format!(
                                "`{name}` has shape {:?}, vocab_size is {}",
                                t.shape, cfg.vocab
                            ));
                        }
                        let bytes = ckpt.data(&name).map_err(|e| e.to_string())?;
                        if !matches!(t.dtype, Dtype::Bool | Dtype::U8 | Dtype::I8) {
                            return Err(format!("`{name}` is {}, not BOOL", t.dtype));
                        }
                        Some(bytes.iter().map(|&b| b != 0).collect::<Vec<bool>>())
                    }
                    None => None,
                };
                VocabMap::from_offsets(&offsets, cfg.vocab, mask.as_deref())?
            }
            None if cfg.draft_vocab == cfg.vocab => VocabMap::identity(cfg.vocab),
            None => {
                return Err(format!(
                    "draft_vocab_size {} is below vocab_size {} and the checkpoint has no d2t",
                    cfg.draft_vocab, cfg.vocab
                ));
            }
        };
        if !vocab.fits(info.n_vocab) {
            return Err(format!(
                "the draft vocabulary has tokens outside the target's {} tokens",
                info.n_vocab
            ));
        }

        let (embed, rows) = match find("embed_tokens.weight") {
            Some((name, t)) => {
                if t.shape.len() != 2 || t.shape[1] != cfg.hidden {
                    return Err(format!(
                        "`{name}` has shape {:?}, the head is {} wide",
                        t.shape, cfg.hidden
                    ));
                }
                if float_type(&t.dtype).is_none() {
                    return Err(format!("`{name}` is {}, not F32, F16 or BF16", t.dtype));
                }
                (
                    Embed::Checkpoint {
                        name,
                        ty: t.dtype.clone(),
                    },
                    t.shape[0],
                )
            }
            None => Self::gguf_embed(cfg, info)?,
        };
        if rows < info.n_vocab {
            return Err(format!(
                "the embedding table has {rows} rows for a {}-token target",
                info.n_vocab
            ));
        }
        Ok((vocab, embed, rows))
    }

    /// No `embed_tokens` in the checkpoint: the head shares the target's token embeddings.
    fn gguf_embed(cfg: &Eagle3Config, info: &TargetInfo) -> Result<(Embed, usize), String> {
        if cfg.hidden != info.n_embd {
            return Err(format!(
                "the checkpoint has no embed_tokens and its width {} is not the target's {}",
                cfg.hidden, info.n_embd
            ));
        }
        let gguf = GgufFile::open(&info.model).map_err(|e| e.to_string())?;
        let t = gguf
            .tensor("token_embd.weight")
            .ok_or("the checkpoint has no embed_tokens and the target GGUF no token_embd.weight")?;
        if t.file_index != 0 || t.shape.len() != 2 || t.shape[0] as usize != cfg.hidden {
            return Err(format!(
                "the target's token_embd.weight (shape {:?}) cannot be shared",
                t.shape
            ));
        }
        let file = std::fs::File::open(gguf.path()).map_err(|e| e.to_string())?;
        Ok((
            Embed::Gguf {
                file,
                offset: t.offset,
                ty: t.ggml_type,
                row_bytes: ggml::row_size(t.ggml_type, cfg.hidden),
            },
            t.shape[1] as usize,
        ))
    }

    /// A weight tensor of the head (panics on an unknown name: the names are the loader's own).
    pub fn tensor(&self, name: &str) -> Tensor<'_> {
        self.ctx
            .tensor(name)
            .unwrap_or_else(|| panic!("no head tensor `{name}`"))
    }

    pub fn has(&self, name: &str) -> bool {
        self.ctx.tensor(name).is_some()
    }

    /// Appends the embedding of `token` to `out`.
    pub fn embed(&self, token: u32, out: &mut Vec<f32>) -> Result<(), String> {
        let t = token as usize;
        if t >= self.embed_rows {
            return Err(format!("token {token} is outside the embedding table"));
        }
        let h = self.cfg.hidden;
        match &self.embed {
            Embed::Checkpoint { name, ty } => {
                let data = self.ckpt.data(name).map_err(|e| e.to_string())?;
                let rb = h * ty.size().unwrap_or(4);
                out.extend(to_f32(ty, &data[t * rb..(t + 1) * rb])?);
            }
            Embed::Gguf {
                file,
                offset,
                ty,
                row_bytes,
            } => {
                let mut buf = vec![0u8; *row_bytes];
                file.read_exact_at(&mut buf, offset + (t * row_bytes) as u64)
                    .map_err(|e| format!("reading a token embedding: {e}"))?;
                if *ty == types::F32 {
                    out.extend(to_f32(&Dtype::F32, &buf)?);
                } else {
                    out.extend(ggml::dequantize(*ty, &buf, h));
                }
            }
        }
        Ok(())
    }
}
