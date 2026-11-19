//! Images in prompts (#146, `--mmproj`; #1195): the vision tower of a Qwen3.5 / Qwen3.6 /
//! Qwen3.8-Flash-Next mmproj, or the one an HF checkpoint ships (`model.visual.*`), and how a
//! request's images become its tokens and their input rows.
//!
//! The chat template renders one `<|vision_start|><|image_pad|><|vision_end|>` per image, in the
//! order the messages carry them. Each `<|image_pad|>` becomes `nx × ny` of itself, one per
//! merged patch: [`Vision::expand`] needs only the preprocessed size, so token counting does not
//! run the tower. [`Vision::encode`] then runs it, giving the sequence's [`Media`] (the rows
//! and M-RoPE positions the model reads) and its prefix-cache key: the prompt with every image
//! token replaced by an id from the image's hash, above any vocabulary, so the prefix cache
//! matches an image only against the same image.
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
use crate::model::core::{ImageSpan, Media};
use crate::tokenizer::Tokenizer;
use crate::vision::{
    Encoder, EncoderConfig, ImageEmbedding, Preprocessed, VisionError, VisionHparams,
};

use super::super::{EngineError, ImageInput};
use super::LoadError;

/// The placeholder token each image's tokens repeat.
const IMAGE_PAD: &str = "<|image_pad|>";

/// Most CPU threads the tower takes, so other work on the host keeps some cores.
const MAX_THREADS: usize = 8;

/// Host memory the encoded images' rows may keep (a 1024 × 768 image's are 7.5 MiB at the
/// width of Qwen3.8-Flash-Next).
const ROW_CACHE_BYTES: usize = 256 << 20;

/// The language models whose positions are Qwen-VL's M-RoPE: Qwen3.5 / Qwen3.6 and
/// Qwen3.8-Flash-Next (`qwen4exp`, whose image tokens' n-gram rows hash `<|image_pad|>` as the
/// reference does: the placeholders stay in the token stream).
const ARCHS: [&str; 3] = ["qwen35", "qwen35moe", "qwen4exp"];

pub(super) struct Vision {
    tower: Mutex<Tower>,
    hp: VisionHparams,
    image_pad: u32,
    rows: Mutex<RowCache>,
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

    fn encode(&mut self, p: &Preprocessed) -> Result<ImageEmbedding, VisionError> {
        match self {
            Tower::Cpu(e) => e.encode(p),
            #[cfg(feature = "cuda")]
            Tower::Gpu(e) => e.encode(p),
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
        Self::from_gguf(&g, n_embd, tokenizer, threads, gpu).map_err(err)
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
        Self::from_gguf(&g, n_embd, tokenizer, threads, gpu).map(Some)
    }

    fn check_arch(arch: &str) -> Result<(), String> {
        if ARCHS.contains(&arch) {
            return Ok(());
        }
        Err(format!(
            "images need a Qwen3.5 / Qwen3.6 / Qwen3.8-Flash-Next language model ({}), not {arch}",
            ARCHS.join(", ")
        ))
    }

    /// The tower `g` (an mmproj, or a checkpoint's tower opened as one).
    fn from_gguf(
        g: &GgufFile,
        n_embd: usize,
        tokenizer: &Tokenizer,
        threads: usize,
        gpu: Option<usize>,
    ) -> Result<Self, String> {
        let image_pad = tokenizer
            .token_id(IMAGE_PAD)
            .ok_or_else(|| format!("the tokenizer has no {IMAGE_PAD}"))?;
        let mut backend = Backend::cpu().map_err(|e| e.to_string())?;
        backend.set_threads(threads.clamp(1, MAX_THREADS));
        let encoder = Encoder::from_gguf(g, backend).map_err(|e| e.to_string())?;
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
            image_pad,
            rows: Mutex::new(RowCache::default()),
        })
    }

    /// Device memory the GPU tower's largest image's scratch takes (0 on the CPU), for the
    /// language model's budgets on that GPU.
    pub(super) fn gpu_scratch_bytes(&self) -> usize {
        self.tower.lock().unwrap().scratch_bytes()
    }

    /// Expands each `<|image_pad|>` of `tokens` (one per image, in order) to its image's tokens.
    pub(super) fn expand(
        &self,
        tokens: Vec<u32>,
        images: &[&ImageInput],
    ) -> Result<(Vec<u32>, Vec<Pending>), EngineError> {
        let pads = tokens.iter().filter(|&&t| t == self.image_pad).count();
        if pads != images.len() {
            return Err(EngineError::InvalidRequest(format!(
                "{} images, but the prompt has {pads} image placeholders",
                images.len()
            )));
        }
        let hp = &self.hp;
        let merge = hp.spatial_merge_size as usize;
        let mut out = Vec::with_capacity(tokens.len());
        let mut pending = Vec::with_capacity(images.len());
        let mut images = images.iter();
        for t in tokens {
            if t != self.image_pad {
                out.push(t);
                continue;
            }
            let img = images.next().expect("as many images as placeholders");
            let pixels = crate::vision::preprocess(hp, &img.bytes)
                .map_err(|e| EngineError::InvalidRequest(format!("image: {e}")))?;
            let (nx, ny) = (
                pixels.grid_w as usize / merge,
                pixels.grid_h as usize / merge,
            );
            let mut h = std::collections::hash_map::DefaultHasher::new();
            img.bytes.hash(&mut h);
            pending.push(Pending {
                start: out.len(),
                nx,
                ny,
                pixels,
                hash: h.finish(),
            });
            out.extend(std::iter::repeat_n(self.image_pad, nx * ny));
        }
        Ok((out, pending))
    }

    /// Runs the tower over `pending` (of `tokens`): the sequence's [`Media`] and prefix-cache key.
    pub(super) fn encode(
        &self,
        tokens: &[u32],
        pending: Vec<Pending>,
    ) -> Result<(Arc<Media>, Vec<u32>), EngineError> {
        let mut key = tokens.to_vec();
        let mut spans = Vec::with_capacity(pending.len());
        let mut n_embd = 0;
        for p in pending {
            let cached = self
                .rows
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(p.hash, p.nx, p.ny);
            let rows = match cached {
                Some(rows) => {
                    tracing::info!(tokens = p.nx * p.ny, "vision: image rows from the cache");
                    rows
                }
                None => {
                    let t = std::time::Instant::now();
                    let e = blocking(|| {
                        self.tower
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .encode(&p.pixels)
                    })
                    .map_err(|e| EngineError::Internal(format!("vision encoder: {e}")))?;
                    assert_eq!((e.nx, e.ny), (p.nx, p.ny), "the tower's grid");
                    tracing::info!(
                        tokens = e.n_tokens(),
                        ms = t.elapsed().as_millis() as u64,
                        "vision: image encoded"
                    );
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
        Ok((Arc::new(Media::new(spans, n_embd)), key))
    }
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
