//! Images in prompts (#146, `--mmproj`; #1195; #1217): the vision tower of a Qwen3.5 / Qwen3.6 /
//! Qwen3.8-Flash-Next or Gemma 4 mmproj, or the one an HF checkpoint ships (`model.visual.*`,
//! `model.vision_tower.*`), and how a request's images become its tokens and their input rows.
//!
//! Qwen's chat template renders one `<|vision_start|><|image_pad|><|vision_end|>` per image, in
//! the order the messages carry them. Each `<|image_pad|>` becomes `nx × ny` of itself, one per
//! merged patch. Gemma 4's renders one `<|image|>`, which becomes `<|image>`, `nx × ny` of
//! itself and `<image|>` (as llama.cpp's mtmd places Gemma 4's images). [`Vision::expand`] needs
//! only the preprocessed size, so token counting does not run the tower. [`Vision::encode`]
//! then runs it, giving the sequence's [`Media`] (the rows and positions the model reads) and
//! its prefix-cache key: the prompt with every image token replaced by an id from the image's
//! hash, above any vocabulary, so the prefix cache matches an image only against the same
//! image.
//!
//! The tower runs one image at a time on the language model's first GPU (Legion's kernels on a
//! stream of its own, so other sequences keep decoding; #1195), or on the CPU without one
//! (`LEGION_DEBUG_VISION_CPU=1` keeps it there, for A/B runs). An image sent again (the same
//! bytes, as a chat resends its images every turn) takes its rows from a small cache instead.

use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::ggml::Backend;
use crate::gguf::GgufFile;
use crate::model::core::{ImageLayout, ImageSpan, Media};
use crate::tokenizer::Tokenizer;
use crate::vision::{
    Encoder, EncoderConfig, ImageEmbedding, Preprocessed, VisionError, VisionHparams,
};

use super::super::{EngineError, ImageInput};
use super::LoadError;

/// The placeholder token each image's tokens repeat (Qwen-VL).
const IMAGE_PAD: &str = "<|image_pad|>";

/// Gemma 4's image placeholder (one per image in its template, repeated per image token), and
/// the tokens around an image's tokens (llama.cpp's mtmd, `PROJECTOR_TYPE_GEMMA4V`).
const GEMMA_IMAGE: &str = "<|image|>";
const GEMMA_IMAGE_OPEN: &str = "<|image>";
const GEMMA_IMAGE_CLOSE: &str = "<image|>";

/// Most CPU threads the tower takes, so other work on the host keeps some cores.
const MAX_THREADS: usize = 8;

/// Host memory the encoded images' rows may keep (a 1024 × 768 image's are 7.5 MiB at the
/// width of Qwen3.8-Flash-Next).
const ROW_CACHE_BYTES: usize = 256 << 20;

/// The language models whose positions are Qwen-VL's M-RoPE: Qwen3.5 / Qwen3.6 and
/// Qwen3.8-Flash-Next (`qwen4exp`, whose image tokens' n-gram rows hash `<|image_pad|>` as the
/// reference does: the placeholders stay in the token stream).
const ARCHS: [&str; 3] = ["qwen35", "qwen35moe", "qwen4exp"];

/// Gemma 4 (#1217): one position per image token, `<|image>` and `<image|>` around them.
const GEMMA_ARCH: &str = "gemma4";

pub(super) struct Vision {
    tower: Mutex<Tower>,
    hp: VisionHparams,
    marks: Marks,
    rows: Mutex<RowCache>,
}

/// How a model's prompts mark images.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Marks {
    /// The placeholder the template renders once per image, repeated once per image token.
    pad: u32,
    /// Tokens put before and after an image's tokens (the template's placeholder has none).
    open: Option<u32>,
    close: Option<u32>,
    layout: ImageLayout,
}

impl Marks {
    /// The marks of language models of architecture `arch`, from `tokenizer`.
    fn of(arch: &str, tokenizer: &Tokenizer) -> Result<Self, String> {
        let id =
            |t: &str| (tokenizer.token_id(t)).ok_or_else(|| format!("the tokenizer has no {t}"));
        Ok(if arch == GEMMA_ARCH {
            Marks {
                pad: id(GEMMA_IMAGE)?,
                open: Some(id(GEMMA_IMAGE_OPEN)?),
                close: Some(id(GEMMA_IMAGE_CLOSE)?),
                layout: ImageLayout::Sequential,
            }
        } else {
            Marks {
                pad: id(IMAGE_PAD)?,
                open: None,
                close: None,
                layout: ImageLayout::MRope,
            }
        })
    }
}

/// `tokens` with each placeholder of `marks` (one per image, in order) expanded to its image's
/// tokens, for images of `grids` (`nx × ny` tokens each), and where each image's tokens start.
fn expand_tokens(
    tokens: &[u32],
    marks: &Marks,
    grids: &[(usize, usize)],
) -> (Vec<u32>, Vec<usize>) {
    let n: usize = grids.iter().map(|&(x, y)| x * y + 2).sum();
    let mut out = Vec::with_capacity(tokens.len() + n);
    let mut starts = Vec::with_capacity(grids.len());
    let mut grids = grids.iter();
    for &t in tokens {
        if t != marks.pad {
            out.push(t);
            continue;
        }
        let &(nx, ny) = grids.next().expect("a grid per placeholder");
        out.extend(marks.open);
        starts.push(out.len());
        out.extend(std::iter::repeat_n(marks.pad, nx * ny));
        out.extend(marks.close);
    }
    (out, starts)
}

/// Where the tower runs.
enum Tower {
    Cpu(Encoder),
    #[cfg(feature = "cuda")]
    Gpu(crate::vision::GpuEncoder),
}

impl Tower {
    fn config(&self) -> &EncoderConfig {
        match self {
            Tower::Cpu(e) => e.config(),
            #[cfg(feature = "cuda")]
            Tower::Gpu(e) => e.config(),
        }
    }

    /// `p`'s rows; `alone`: the engine runs nothing else meanwhile, so a GPU encode may take
    /// the whole GPU ([`crate::vision::share`]).
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    fn encode(&mut self, p: &Preprocessed, alone: bool) -> Result<ImageEmbedding, VisionError> {
        match self {
            Tower::Cpu(e) => e.encode(p),
            #[cfg(feature = "cuda")]
            Tower::Gpu(e) => e.encode_with(p, alone),
        }
    }

    /// The CPU tower `enc`, or its weights on CUDA device `gpu` when the kernels take it.
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    fn place(enc: Encoder, gpu: Option<usize>) -> Self {
        #[cfg(feature = "cuda")]
        {
            let cpu_only = std::env::var_os("LEGION_DEBUG_VISION_CPU").is_some_and(|v| v == "1");
            if let Some(dev) = gpu.filter(|_| !cpu_only) {
                match crate::vision::GpuEncoder::new(&enc, dev) {
                    Ok(Some(g)) => return Tower::Gpu(g),
                    Ok(None) => tracing::info!(
                        device = dev,
                        "vision: the GPU tower does not take this device or tower; it runs on \
                         the CPU"
                    ),
                    Err(e) => tracing::warn!(
                        device = dev,
                        error = %e,
                        "vision: the tower did not load on the GPU; it runs on the CPU"
                    ),
                }
            }
        }
        Tower::Cpu(enc)
    }

    /// Device memory the largest image's scratch takes on the GPU (0 on the CPU).
    fn scratch_bytes(&self) -> usize {
        match self {
            Tower::Cpu(_) => 0,
            #[cfg(feature = "cuda")]
            Tower::Gpu(e) => e.scratch_bytes(),
        }
    }

    fn place_name(&self) -> String {
        match self {
            Tower::Cpu(_) => "the CPU".into(),
            #[cfg(feature = "cuda")]
            Tower::Gpu(e) => format!("GPU {}", e.device()),
        }
    }
}

/// Encoded images' rows by the image's hash and grid, least recently used first.
#[derive(Default)]
struct RowCache {
    entries: Vec<(u64, usize, usize, Arc<[f32]>)>,
    bytes: usize,
}

impl RowCache {
    fn get(&mut self, hash: u64, nx: usize, ny: usize) -> Option<Arc<[f32]>> {
        let i = self
            .entries
            .iter()
            .position(|e| (e.0, e.1, e.2) == (hash, nx, ny))?;
        let e = self.entries.remove(i);
        let rows = e.3.clone();
        self.entries.push(e);
        Some(rows)
    }

    fn put(&mut self, hash: u64, nx: usize, ny: usize, rows: Arc<[f32]>, limit: usize) {
        let size = rows.len() * 4;
        if size > limit || self.get(hash, nx, ny).is_some() {
            return;
        }
        while self.bytes + size > limit {
            let e = self.entries.remove(0);
            self.bytes -= e.3.len() * 4;
        }
        self.bytes += size;
        self.entries.push((hash, nx, ny, rows));
    }
}

/// An image of a prompt, placed and preprocessed, not yet encoded.
pub(super) struct Pending {
    start: usize,
    nx: usize,
    ny: usize,
    pixels: Preprocessed,
    hash: u64,
}

impl Vision {
    /// Loads the mmproj `path` (`--mmproj`) for a language model of architecture `arch` and
    /// width `n_embd`, onto CUDA device `gpu` when there is one.
    pub(super) fn load(
        path: &Path,
        arch: &str,
        n_embd: usize,
        tokenizer: &Tokenizer,
        threads: usize,
        gpu: Option<usize>,
    ) -> Result<Self, LoadError> {
        let err = |e: String| LoadError::Device(format!("--mmproj {}: {e}", path.display()));
        Self::check_arch(arch).map_err(err)?;
        let g = GgufFile::open(path).map_err(|e| err(e.to_string()))?;
        Self::from_gguf(&g, arch, n_embd, tokenizer, threads, gpu).map_err(err)
    }

    /// The vision tower the HF checkpoint `dir` ships (`model.visual.*`), if it has one, for a
    /// language model of architecture `arch` and width `n_embd`, onto CUDA device `gpu` when
    /// there is one.
    pub(super) fn from_checkpoint(
        dir: &Path,
        arch: &str,
        n_embd: usize,
        tokenizer: &Tokenizer,
        threads: usize,
        gpu: Option<usize>,
    ) -> Result<Option<Self>, String> {
        if Self::check_arch(arch).is_err() {
            return Ok(None);
        }
        let Some(g) = crate::hf::open_vision(dir).map_err(|e| e.to_string())? else {
            return Ok(None);
        };
        Self::from_gguf(&g, arch, n_embd, tokenizer, threads, gpu).map(Some)
    }

    fn check_arch(arch: &str) -> Result<(), String> {
        if ARCHS.contains(&arch) || arch == GEMMA_ARCH {
            return Ok(());
        }
        Err(format!(
            "images need a Qwen3.5 / Qwen3.6 / Qwen3.8-Flash-Next or Gemma 4 language model \
             ({}, {GEMMA_ARCH}), not {arch}",
            ARCHS.join(", ")
        ))
    }

    /// The tower `g` (an mmproj, or a checkpoint's tower opened as one) for a language model
    /// of architecture `arch`.
    fn from_gguf(
        g: &GgufFile,
        arch: &str,
        n_embd: usize,
        tokenizer: &Tokenizer,
        threads: usize,
        gpu: Option<usize>,
    ) -> Result<Self, String> {
        let marks = Marks::of(arch, tokenizer)?;
        let mut backend = Backend::cpu().map_err(|e| e.to_string())?;
        backend.set_threads(threads.clamp(1, MAX_THREADS));
        let encoder = Encoder::from_gguf(g, backend).map_err(|e| e.to_string())?;
        let gemma = encoder.hparams().projector == crate::vision::Projector::Gemma4v;
        if gemma != (arch == GEMMA_ARCH) {
            return Err(format!(
                "a {} vision tower for a {arch} language model",
                if gemma { "Gemma 4" } else { "Qwen-VL" }
            ));
        }
        let out = encoder.config().n_out;
        if out != n_embd {
            return Err(format!(
                "its rows are {out} wide, the language model's embeddings {n_embd}"
            ));
        }
        let hp = encoder.hparams().clone();
        // The GPU tower keeps its own copy of the weights: the CPU tower's are dropped then.
        let tower = Tower::place(encoder, gpu);
        tracing::info!(
            tower = %g.path().display(),
            layers = tower.config().n_layer,
            largest_image_scratch_mib = tower.scratch_bytes() >> 20,
            "vision: images accepted (the vision tower runs on {})",
            tower.place_name()
        );
        Ok(Self {
            tower: Mutex::new(tower),
            hp,
            marks,
            rows: Mutex::new(RowCache::default()),
        })
    }

    /// Device memory the GPU tower's largest image's scratch takes (0 on the CPU), for the
    /// language model's budgets on that GPU.
    pub(super) fn gpu_scratch_bytes(&self) -> usize {
        self.tower.lock().unwrap().scratch_bytes()
    }

    /// Expands each image placeholder of `tokens` (one per image, in order) to its image's
    /// tokens (with Gemma 4's `<|image>` and `<image|>` around them).
    pub(super) fn expand(
        &self,
        tokens: Vec<u32>,
        images: &[&ImageInput],
    ) -> Result<(Vec<u32>, Vec<Pending>), EngineError> {
        let pads = tokens.iter().filter(|&&t| t == self.marks.pad).count();
        if pads != images.len() {
            return Err(EngineError::InvalidRequest(format!(
                "{} images, but the prompt has {pads} image placeholders",
                images.len()
            )));
        }
        let hp = &self.hp;
        let merge = hp.spatial_merge_size as usize;
        let mut pending = Vec::with_capacity(images.len());
        for img in images {
            let pixels = crate::vision::preprocess(hp, &img.bytes)
                .map_err(|e| EngineError::InvalidRequest(format!("image: {e}")))?;
            let (nx, ny) = (
                pixels.grid_w as usize / merge,
                pixels.grid_h as usize / merge,
            );
            // Gemma 4's steps and rings hold an image of up to this many tokens whole (#1217):
            // a bigger one is refused, never cut (the resize keeps every image within it).
            // Qwen-VL's images read causally, so a step may cut them.
            if hp.projector == crate::vision::Projector::Gemma4v && nx * ny > hp.max_image_tokens()
            {
                return Err(EngineError::InvalidRequest(format!(
                    "image: {nx} x {ny} tokens, more than the {} one image may take",
                    hp.max_image_tokens()
                )));
            }
            let mut h = std::collections::hash_map::DefaultHasher::new();
            img.bytes.hash(&mut h);
            pending.push(Pending {
                start: 0,
                nx,
                ny,
                pixels,
                hash: h.finish(),
            });
        }
        let grids: Vec<_> = pending.iter().map(|p| (p.nx, p.ny)).collect();
        let (out, starts) = expand_tokens(&tokens, &self.marks, &grids);
        for (p, s) in pending.iter_mut().zip(starts) {
            p.start = s;
        }
        Ok((out, pending))
    }

    /// Runs the tower over `pending` (of `tokens`): the sequence's [`Media`] and prefix-cache key.
    /// `alone()`: whether the engine runs nothing else, asked as each image's encode starts
    /// (holding the tower).
    pub(super) fn encode(
        &self,
        tokens: &[u32],
        pending: Vec<Pending>,
        alone: impl Fn() -> bool,
    ) -> Result<(Arc<Media>, Vec<u32>), EngineError> {
        let mut key = tokens.to_vec();
        let mut spans = Vec::with_capacity(pending.len());
        let mut n_embd = 0;
        let reencode = reencode();
        for p in pending {
            let cached = if reencode {
                None
            } else {
                self.rows
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(p.hash, p.nx, p.ny)
            };
            let rows = match cached {
                Some(rows) => {
                    tracing::info!(tokens = p.nx * p.ny, "vision: image rows from the cache");
                    rows
                }
                None => {
                    let t = std::time::Instant::now();
                    let mut was_alone = false;
                    let e = blocking(|| {
                        let mut tower = self.tower.lock().unwrap_or_else(|e| e.into_inner());
                        // Asked once the tower is ours: an encode queued behind another one
                        // sees the engine as it is when its own encode starts.
                        was_alone = alone();
                        tower.encode(&p.pixels, was_alone)
                    })
                    .map_err(|e| EngineError::Internal(format!("vision encoder: {e}")))?;
                    assert_eq!((e.nx, e.ny), (p.nx, p.ny), "the tower's grid");
                    let ms = t.elapsed().as_millis() as u64;
                    if reencode {
                        tracing::info!(
                            tokens = e.n_tokens(),
                            ms,
                            alone = was_alone,
                            rows = %format!("{:016x}", rows_hash(&e.data)),
                            "vision: image encoded"
                        );
                    } else {
                        tracing::info!(
                            tokens = e.n_tokens(),
                            ms,
                            alone = was_alone,
                            "vision: image encoded"
                        );
                    }
                    let rows: Arc<[f32]> = e.data.into();
                    self.rows.lock().unwrap_or_else(|e| e.into_inner()).put(
                        p.hash,
                        p.nx,
                        p.ny,
                        rows.clone(),
                        ROW_CACHE_BYTES,
                    );
                    rows
                }
            };
            n_embd = rows.len() / (p.nx * p.ny);
            for (i, k) in key[p.start..p.start + p.nx * p.ny].iter_mut().enumerate() {
                *k = image_key(p.hash, i);
            }
            spans.push(ImageSpan {
                start: p.start,
                nx: p.nx,
                ny: p.ny,
                rows,
            });
        }
        let media = Media::with_layout(spans, n_embd, self.marks.layout);
        Ok((Arc::new(media), key))
    }
}

/// `LEGION_DEBUG_VISION_REENCODE=1` (A/B runs): every image is encoded, the row cache unused,
/// and the log names a hash of its rows' bits.
fn reencode() -> bool {
    std::env::var_os("LEGION_DEBUG_VISION_REENCODE").is_some_and(|v| v == "1")
}

/// FNV-1a over the bits of `rows`.
fn rows_hash(rows: &[f32]) -> u64 {
    rows.iter().fold(0xcbf2_9ce4_8422_2325, |h, v| {
        (h ^ u64::from(v.to_bits())).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// The prefix-cache key of token `i` of an image hashed `hash`: above `u32::MAX / 2`, which no
/// vocabulary reaches.
pub(super) fn image_key(hash: u64, i: usize) -> u32 {
    let x =
        (hash ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    0x8000_0000 | ((x >> 33) as u32)
}

/// Runs `f`, which takes seconds on the CPU, so that a multi-threaded Tokio runtime calling
/// it (`generate` is called from the server's handlers) moves its other tasks off this thread.
fn blocking<T>(f: impl FnOnce() -> T) -> T {
    match tokio::runtime::Handle::try_current() {
        Ok(h) if h.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(f)
        }
        _ => f(),
    }
}

#[cfg(test)]
mod tests;
