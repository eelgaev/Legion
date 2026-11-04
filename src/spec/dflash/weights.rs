//! Loading a DFlash drafter's tensors into backend memory, from a GGUF or a Hugging Face
//! safetensors checkpoint, under the GGUF tensor names (llama.cpp's `dflash` architecture):
//! `fc.weight`, `enc.output_norm.weight`, `output_norm.weight`, `blk.N.{attn_norm, attn_q,
//! attn_k, attn_v, attn_output, attn_q_norm, attn_k_norm, ffn_norm, ffn_gate, ffn_up,
//! ffn_down}.weight`, and optionally `token_embd.weight`, `output.weight` and `d2t`.
//! A drafter without embeddings or output head uses the target's.
//!
//! A Hugging Face checkpoint is read as float tensors, except FP8 block-scaled matrices
//! (`quant_method: fp8`, such as `tcclaviger/Qwen3.8-27B-DFlash2-FP8`: an F8_E4M3 `weight` and a
//! `weight_scale_inv` factor per `weight_block_size` block). There is no ggml matmul for that
//! layout, so they are read as Q8_0, whose 32-element blocks hold the same 8-bit values within
//! one rounding step.

use std::os::unix::fs::FileExt;
use std::path::Path;

use crate::ggml::{BufferType, Context, Tensor, ffi, types};
use crate::gguf::{FileRange, GgufFile, Recipe};
use crate::safetensors::{Checkpoint, Dtype};
use crate::weights::Weights;

use super::config::{Config, ConfigError, Flavor};
use crate::spec::DraftWeights;

/// Named tensors in backend memory: one or more owners behind one lookup.
#[derive(Default)]
pub struct Store {
    gguf: Vec<Weights>,
    ctxs: Vec<Context>,
}

impl Store {
    pub fn get(&self, name: &str) -> Option<Tensor<'_>> {
        self.gguf
            .iter()
            .find_map(|w| w.get(name))
            .or_else(|| self.ctxs.iter().find_map(|c| c.tensor(name)))
    }

    /// The tensor `name` with exactly `shape` (ggml order, trailing 1s trimmed), or why not.
    pub fn require(&self, name: &str, shape: &[usize]) -> Result<Tensor<'_>, String> {
        let t = self
            .get(name)
            .ok_or_else(|| format!("the drafter has no tensor `{name}`"))?;
        let got: Vec<usize> = t.shape().iter().map(|&d| d as usize).collect();
        let mut want = shape.to_vec();
        while want.len() > 1 && want.last() == Some(&1) {
            want.pop();
        }
        if got != want {
            return Err(format!(
                "tensor `{name}` has shape {got:?}, expected {want:?}"
            ));
        }
        Ok(t)
    }
}

/// What a drafter checkpoint is, from its path.
pub enum Source {
    Gguf(GgufFile),
    Hf(Box<Checkpoint>),
}

impl Source {
    /// A `.gguf` file, or a directory (or its `model.safetensors`) with `config.json`.
    pub fn open(path: &Path) -> Result<Self, String> {
        let is_gguf = path.extension().is_some_and(|e| e == "gguf");
        if path.is_file() && is_gguf {
            return GgufFile::open(path)
                .map(Source::Gguf)
                .map_err(|e| format!("{}: {e}", path.display()));
        }
        let dir = if path.is_file() {
            path.parent().unwrap_or(Path::new("."))
        } else {
            path
        };
        Checkpoint::open(dir)
            .map(|c| Source::Hf(Box::new(c)))
            .map_err(|e| e.to_string())
    }

    pub fn config(&self, flavor: Flavor) -> Result<Config, ConfigError> {
        match self {
            Source::Gguf(g) => Config::from_gguf(g, flavor),
            Source::Hf(c) => Config::from_hf(
                c.config().ok_or_else(|| {
                    ConfigError::Invalid("the checkpoint directory has no config.json".into())
                })?,
                flavor,
            ),
        }
    }

    /// Names of all tensors, in the GGUF naming.
    pub fn tensor_names(&self) -> Vec<String> {
        match self {
            Source::Gguf(g) => g.tensors().into_iter().map(|t| t.name).collect(),
            Source::Hf(c) => c
                .tensors()
                .into_iter()
                .map(|t| hf_name(&t.name).unwrap_or(t.name))
                .collect(),
        }
    }

    /// Puts every tensor into `buft`, but the [`HOST_ONLY`] ones: a GGUF's go to host memory,
    /// a Hugging Face checkpoint's are left out (read them with [`Self::host_tensor`]). `weights`:
    /// a Hugging Face checkpoint's [`q8_matrix`] tensors as stored, as Q8_0 or as 4-bit copies
    /// (a GGUF's are read as stored).
    pub fn load(&self, buft: BufferType, weights: DraftWeights) -> Result<Store, String> {
        match self {
            Source::Gguf(g) => {
                let host = crate::ggml::device_by_kind(crate::ggml::DeviceKind::Cpu)
                    .ok_or("no CPU device")?
                    .buffer_type();
                let w = Weights::load(g, |t| {
                    if HOST_ONLY.contains(&t.name.as_str()) {
                        host
                    } else {
                        buft
                    }
                })
                .map_err(|e| e.to_string())?;
                Ok(Store {
                    gguf: vec![w],
                    ctxs: Vec::new(),
                })
            }
            Source::Hf(c) => load_hf(c, buft, weights),
        }
    }

    /// The tensor `name` (GGUF naming) as it is stored: type, shape (ggml order) and bytes.
    pub fn host_tensor(&self, name: &str) -> Result<Option<HostTensor>, String> {
        match self {
            Source::Gguf(g) => {
                let Some(t) = g.tensor(name) else {
                    return Ok(None);
                };
                let path = g.shard_paths().nth(t.file_index).ok_or("no shard")?;
                let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
                let mut bytes = vec![0u8; t.size as usize];
                file.read_exact_at(&mut bytes, t.offset)
                    .map_err(|e| format!("reading `{name}`: {e}"))?;
                Ok(Some(HostTensor {
                    ty: t.ggml_type,
                    shape: t.shape.iter().map(|&d| d as usize).collect(),
                    bytes,
                }))
            }
            Source::Hf(c) => {
                let Some(t) = c
                    .tensors()
                    .into_iter()
                    .find(|t| hf_name(&t.name).as_deref() == Some(name))
                else {
                    return Ok(None);
                };
                let ty = ggml_type(&t.dtype)
                    .ok_or_else(|| format!("tensor `{}` has type {}", t.name, t.dtype))?;
                let bytes = c.data(&t.name).map_err(|e| e.to_string())?.to_vec();
                Ok(Some(HostTensor {
                    ty,
                    shape: t.shape.iter().rev().copied().collect(),
                    bytes,
                }))
            }
        }
    }
}

/// Tensors the drafter reads on the host, a row at a time (DFlash 2's selector codebooks).
pub const HOST_ONLY: [&str; 2] = ["selector_predecessor.weight", "selector_successor.weight"];

/// A tensor's bytes in host memory, as stored.
pub struct HostTensor {
    pub ty: ffi::ggml_type,
    /// ggml order.
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

/// A Hugging Face tensor name in the GGUF naming (`None`: not a DFlash tensor).
pub fn hf_name(name: &str) -> Option<String> {
    let name = name.strip_prefix("model.").unwrap_or(name);
    let top = match name {
        "fc.weight" => Some("fc.weight"),
        "hidden_norm.weight" => Some("enc.output_norm.weight"),
        "norm.weight" => Some("output_norm.weight"),
        "embed_tokens.weight" => Some("token_embd.weight"),
        "lm_head.weight" => Some("output.weight"),
        "d2t" => Some("d2t"),
        // DSpark's heads.
        "markov_head.markov_w1.weight" => Some("markov_w1.weight"),
        "markov_head.markov_w2.weight" => Some("markov_w2.weight"),
        "confidence_head.proj.weight" => Some("conf_proj.weight"),
        "confidence_head.proj.bias" => Some("conf_proj.bias"),
        // DFlash 2's candidate selector (llama.cpp's names).
        "candidate_selector.hidden_projection.weight" => Some("selector_hidden.weight"),
        "candidate_selector.predecessor_codebook"
        | "candidate_selector.predecessor_codebook.weight" => Some("selector_predecessor.weight"),
        "candidate_selector.successor_codebook"
        | "candidate_selector.successor_codebook.weight" => Some("selector_successor.weight"),
        _ => None,
    };
    if let Some(t) = top {
        return Some(t.into());
    }
    let rest = name.strip_prefix("layers.")?;
    let (n, rest) = rest.split_once('.')?;
    n.parse::<usize>().ok()?;
    let leaf = match rest {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        // Attention sinks, under the names the architectures that have them use.
        "self_attn.sinks"
        | "self_attn.sinks.weight"
        | "self_attn.attention_sink_bias"
        | "self_attn.attention_sink_bias.weight"
        | "self_attn.learnable_sink_param"
        | "self_attn.learnable_sink_param.weight" => "attn_sinks.weight",
        // DFlash 2's convolutions (llama.cpp's names; the base kernels have no `.weight`).
        "attention_conv.base_kernel" => "attn_conv_base",
        "attention_conv.kernel_projection.weight" => "attn_conv_proj.weight",
        "mlp_conv.base_kernel" => "ffn_conv_base",
        "mlp_conv.kernel_projection.weight" => "ffn_conv_proj.weight",
        _ => return None,
    };
    Some(format!("blk.{n}.{leaf}"))
}

fn ggml_type(d: &Dtype) -> Option<ffi::ggml_type> {
    match d {
        Dtype::F32 => Some(types::F32),
        Dtype::F16 => Some(types::F16),
        Dtype::BF16 => Some(types::BF16),
        _ => None,
    }
}

/// Whether `--spec-draft-weights q8_0|q4` stores the tensor `name` (GGUF naming, ggml `shape`)
/// as Q8_0 or 4 bits: the matrices a draft pass multiplies activations by (the layers', `fc` and DFlash 2's
/// selector projection) with rows of whole Q8_0 blocks. Norms, convolution base kernels,
/// embeddings, the head and DSpark's heads stay as stored.
pub fn q8_matrix(name: &str, shape: &[i64]) -> bool {
    (name.starts_with("blk.") || name == "fc.weight" || name == "selector_hidden.weight")
        && name.ends_with(".weight")
        && shape.len() == 2
        && shape[1] > 1
        && shape[0] % 32 == 0
}

/// `bytes` (`ty`: F32, F16 or BF16, rows of `k`) as Q8_0, the rows split over threads.
pub(super) fn to_q8(ty: ffi::ggml_type, bytes: &[u8], k: usize) -> Vec<u8> {
    let row = crate::ggml::row_size(ty, k);
    let rows = bytes.len() / row;
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(16);
    let per = rows.div_ceil(threads).max(1);
    let parts: Vec<Vec<u8>> = std::thread::scope(|scope| {
        let hs: Vec<_> = bytes
            .chunks(per * row)
            .map(|c| {
                scope.spawn(move || {
                    let f = crate::ggml::dequantize(ty, c, c.len() / row * k);
                    crate::ggml::quantize(types::Q8_0, &f, k)
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("quantize thread"))
            .collect()
    });
    parts.concat()
}

/// An FP8 block-scaled matrix as Q8_0: `w` holds rows of `cols` E4M3 bytes, `scale` (F32, row
/// major) one factor per `block[0]` rows x `block[1]` columns, and element `(r, c)` is
/// `e4m3(w[r][c]) * scale[r / block[0]][c / block[1]]`. The rows are split over threads.
pub(super) fn fp8_blocks_to_q8(w: &[u8], scale: &[f32], cols: usize, block: [usize; 2]) -> Vec<u8> {
    let lut: [f32; 256] = std::array::from_fn(|b| crate::model::fp8::e4m3_to_f32(b as u8));
    let rows = w.len() / cols;
    let scale_cols = cols.div_ceil(block[1]);
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(16);
    let per = rows.div_ceil(threads).max(1);
    let parts: Vec<Vec<u8>> = std::thread::scope(|scope| {
        let lut = &lut;
        let hs: Vec<_> = (0..rows)
            .step_by(per)
            .map(|r0| {
                scope.spawn(move || {
                    let r1 = (r0 + per).min(rows);
                    let mut f = Vec::with_capacity((r1 - r0) * cols);
                    for r in r0..r1 {
                        let s = &scale[r / block[0] * scale_cols..][..scale_cols];
                        let row = &w[r * cols..][..cols];
                        f.extend(
                            row.iter()
                                .enumerate()
                                .map(|(c, &b)| lut[b as usize] * s[c / block[1]]),
                        );
                    }
                    crate::ggml::quantize(types::Q8_0, &f, cols)
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("quantize thread"))
            .collect()
    });
    parts.concat()
}

/// The `weight_block_size` of a `quantization_config` (`[rows, cols]`; 128 x 128 if unset).
fn fp8_block(config: Option<&serde_json::Value>) -> Result<[usize; 2], String> {
    let v = config
        .and_then(|c| c.get("quantization_config"))
        .and_then(|q| q.get("weight_block_size"));
    let Some(v) = v.filter(|v| !v.is_null()) else {
        return Ok([128, 128]);
    };
    let dims: Option<Vec<usize>> = v.as_array().map(|a| {
        a.iter()
            .filter_map(|d| d.as_u64().filter(|&d| d > 0).map(|d| d as usize))
            .collect()
    });
    match (v.as_array().map(Vec::len), dims.as_deref()) {
        (Some(2), Some(&[r, c])) => Ok([r, c]),
        _ => Err(format!(
            "quantization_config.weight_block_size {v} is not two positive integers"
        )),
    }
}

/// One-dimensional tensors (norm weights) and convolution base kernels are F32 in memory: ggml
/// multiplies activations by them. Matrices keep the checkpoint's float type, or with
/// [`DraftWeights::Q8_0`] / [`DraftWeights::Q4`] the [`q8_matrix`] ones become Q8_0 / Q4_K
/// (Q4_0 for rows not a multiple of 256 wide); FP8 block-scaled matrices become Q8_0 in every
/// case (see the module docs). The [`HOST_ONLY`] tensors are left out.
fn load_hf(c: &Checkpoint, buft: BufferType, weights: DraftWeights) -> Result<Store, String> {
    struct Item {
        name: String,
        src: String,
        dtype: Dtype,
        ty: ffi::ggml_type,
        shape: Vec<i64>,
        /// FP8: the `weight_scale_inv` tensor and its float type.
        fp8_scale: Option<(String, ffi::ggml_type)>,
    }
    let all = c.tensors();
    let mut block = None;
    let mut items = Vec::new();
    for t in &all {
        let Some(name) = hf_name(&t.name) else {
            continue;
        };
        if HOST_ONLY.contains(&name.as_str()) {
            continue;
        }
        if name == "d2t" {
            // Offsets from the draft row's own index; stored as I64 (see `load_hf` below).
            if t.dtype != Dtype::I64 {
                return Err("`d2t` is not int64".into());
            }
            items.push(Item {
                name,
                src: t.name.clone(),
                dtype: Dtype::I64,
                ty: ffi::ggml_type_GGML_TYPE_I64,
                shape: vec![t.numel() as i64],
                fp8_scale: None,
            });
            continue;
        }
        let shape: Vec<i64> = t.shape.iter().rev().map(|&d| d as i64).collect();
        if t.dtype == Dtype::F8E4M3 {
            if !q8_matrix(&name, &shape) {
                return Err(format!(
                    "tensor `{}` is F8_E4M3: FP8 is read for the layers' matrices, `fc` and the \
                     selector projection only",
                    t.name
                ));
            }
            let src = format!("{}_scale_inv", t.name);
            let s = all.iter().find(|s| s.name == src).ok_or_else(|| {
                format!(
                    "tensor `{}` is F8_E4M3 without a `{src}`: only block-scaled FP8 is read",
                    t.name
                )
            })?;
            let b = match block {
                Some(b) => b,
                None => *block.insert(fp8_block(c.config())?),
            };
            let want = [t.shape[0].div_ceil(b[0]), t.shape[1].div_ceil(b[1])];
            let sty = ggml_type(&s.dtype);
            if s.shape != want || sty.is_none() {
                return Err(format!(
                    "`{src}` is {} {:?}, expected a float [{}, {}] for {} x {} blocks",
                    s.dtype, s.shape, want[0], want[1], b[0], b[1]
                ));
            }
            items.push(Item {
                name,
                src: t.name.clone(),
                dtype: Dtype::F8E4M3,
                ty: types::Q8_0,
                shape,
                fp8_scale: Some((src, sty.expect("checked above"))),
            });
            continue;
        }
        // DFlash 2's convolution base kernels multiply activations too.
        let ty = if shape.len() == 1 || name.ends_with("_conv_base") {
            types::F32
        } else if weights != DraftWeights::Model && q8_matrix(&name, &shape) {
            match weights {
                DraftWeights::Q4 => crate::spec::q4_head::q4_type(shape[0] as usize)
                    .expect("q8_matrix rows are whole 32-value blocks"),
                _ => types::Q8_0,
            }
        } else {
            ggml_type(&t.dtype).ok_or_else(|| {
                format!(
                    "tensor `{}` is {}: only float checkpoints are read (quantized drafters \
                     come as GGUF)",
                    t.name, t.dtype
                )
            })?
        };
        if ggml_type(&t.dtype).is_none() {
            return Err(format!("tensor `{}` has type {}", t.name, t.dtype));
        }
        items.push(Item {
            name,
            src: t.name.clone(),
            dtype: t.dtype.clone(),
            ty,
            shape,
            fp8_scale: None,
        });
    }
    let ctx = Context::new(items.len().max(1), 0);
    for i in &items {
        ctx.new_tensor(i.ty, &i.shape).named(&i.name);
    }
    ctx.alloc_tensors(buft).map_err(|e| e.to_string())?;
    for i in &items {
        let bytes = c.data(&i.src).map_err(|e| e.to_string())?;
        let t = ctx.tensor(&i.name).expect("created above");
        if let Some((src, sty)) = &i.fp8_scale {
            let s = c.data(src).map_err(|e| e.to_string())?;
            let s: Vec<f32> = if *sty == types::F32 {
                s.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect()
            } else {
                let n = s.len() / crate::ggml::row_size(*sty, 1);
                crate::ggml::dequantize(*sty, s, n)
            };
            let b = block.expect("set with the item");
            t.set_bytes(&fp8_blocks_to_q8(bytes, &s, i.shape[0] as usize, b));
        } else if i.ty == types::F32 && i.dtype != Dtype::F32 {
            let ty = ggml_type(&i.dtype).expect("checked above");
            let n = i.shape.iter().product::<i64>() as usize;
            t.set_f32(&crate::ggml::dequantize(ty, bytes, n));
        } else if i.ty == types::Q8_0 {
            let ty = ggml_type(&i.dtype).expect("checked above");
            t.set_bytes(&to_q8(ty, bytes, i.shape[0] as usize));
        } else if i.ty == types::Q4_K || i.ty == types::Q4_0 {
            let ty = ggml_type(&i.dtype).expect("checked above");
            let (_, q) = crate::spec::q4_head::to_q4(ty, bytes, None, i.shape[0] as usize)?;
            t.set_bytes(&q);
        } else if i.name == "d2t" {
            // Offsets to absolute target ids (llama.cpp's converter does the same).
            let v: Vec<i64> = bytes
                .as_chunks::<8>()
                .0
                .iter()
                .enumerate()
                .map(|(n, b)| i64::from_le_bytes(*b) + n as i64)
                .collect();
            let raw: Vec<u8> = v.iter().flat_map(|x| x.to_ne_bytes()).collect();
            t.set_bytes(&raw);
        } else {
            t.set_bytes(bytes);
        }
    }
    Ok(Store {
        gguf: Vec::new(),
        ctxs: vec![ctx],
    })
}

/// The tensors of the GGUF at `path` named by `(name in the file, name here)`, copied as they
/// are into `buft`. Names the file does not have are skipped (see [`Store::get`]).
pub fn load_from_gguf(
    path: &Path,
    names: &[(&str, &str)],
    buft: BufferType,
) -> Result<Store, String> {
    let gguf = GgufFile::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let infos: Vec<_> = names
        .iter()
        .filter_map(|(src, dst)| {
            gguf.tensor(src).map(|mut i| {
                i.name = (*dst).to_string();
                (i, *src)
            })
        })
        .collect();
    let ctx = Context::new(infos.len().max(1), 0);
    for (i, _) in &infos {
        let shape: Vec<i64> = i.shape.iter().map(|&d| d as i64).collect();
        ctx.new_tensor(i.ggml_type, &shape).named(&i.name);
    }
    ctx.alloc_tensors(buft).map_err(|e| e.to_string())?;
    let shards: Vec<_> = gguf.shard_paths().collect();
    const CHUNK: usize = 64 << 20;
    for (i, src) in &infos {
        let t = ctx.tensor(&i.name).expect("created above");
        // A model synthesized from a Hugging Face checkpoint makes some tensors (repacked
        // NVFP4, gathered rows) instead of pointing at a file range.
        let parts = match gguf.recipe(src) {
            Some(Recipe::Computed(f)) => {
                let bytes = f().map_err(|e| format!("making `{src}`: {e}"))?;
                if bytes.len() != t.nbytes() {
                    return Err(format!(
                        "`{src}`: {} bytes made, {} expected",
                        bytes.len(),
                        t.nbytes()
                    ));
                }
                t.set_bytes(&bytes);
                continue;
            }
            Some(Recipe::Gather(parts)) => parts.clone(),
            None => vec![FileRange {
                file: i.file_index,
                offset: i.offset,
                len: i.size,
            }],
        };
        let mut done = 0usize;
        for p in parts {
            let file = std::fs::File::open(shards[p.file])
                .map_err(|e| format!("{}: {e}", shards[p.file].display()))?;
            let mut buf = vec![0u8; CHUNK.min(p.len as usize)];
            let mut read = 0usize;
            while read < p.len as usize {
                let n = buf.len().min(p.len as usize - read);
                file.read_exact_at(&mut buf[..n], p.offset + read as u64)
                    .map_err(|e| format!("reading `{}`: {e}", i.name))?;
                t.set_bytes_at(done, &buf[..n]);
                read += n;
                done += n;
            }
        }
    }
    Ok(Store {
        gguf: Vec::new(),
        ctxs: vec![ctx],
    })
}

impl Store {
    /// The tensors of `ctx`, allocated.
    pub fn from_context(ctx: Context) -> Self {
        Self {
            gguf: Vec::new(),
            ctxs: vec![ctx],
        }
    }

    /// Adds the tensors of `other` (names already here win).
    pub fn extend(&mut self, other: Store) {
        self.gguf.extend(other.gguf);
        self.ctxs.extend(other.ctxs);
    }
}
