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
//! The tower runs on the CPU, one image at a time.

use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::ggml::Backend;
use crate::gguf::GgufFile;
use crate::model::core::{ImageSpan, Media};
use crate::tokenizer::Tokenizer;
use crate::vision::{Encoder, Preprocessed};

use super::super::{EngineError, ImageInput};
use super::LoadError;

/// The placeholder token each image's tokens repeat.
const IMAGE_PAD: &str = "<|image_pad|>";

/// Most CPU threads the tower takes, so other work on the host keeps some cores.
const MAX_THREADS: usize = 8;

/// The language models whose positions are Qwen-VL's M-RoPE: Qwen3.5 / Qwen3.6 and
/// Qwen3.8-Flash-Next (`qwen4exp`, whose image tokens' n-gram rows hash `<|image_pad|>` as the
/// reference does: the placeholders stay in the token stream).
const ARCHS: [&str; 3] = ["qwen35", "qwen35moe", "qwen4exp"];

pub(super) struct Vision {
    encoder: Mutex<Encoder>,
    image_pad: u32,
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
    /// width `n_embd`.
    pub(super) fn load(
        path: &Path,
        arch: &str,
        n_embd: usize,
        tokenizer: &Tokenizer,
        threads: usize,
    ) -> Result<Self, LoadError> {
        let err = |e: String| LoadError::Device(format!("--mmproj {}: {e}", path.display()));
        Self::check_arch(arch).map_err(err)?;
        let g = GgufFile::open(path).map_err(|e| err(e.to_string()))?;
        Self::from_gguf(&g, n_embd, tokenizer, threads).map_err(err)
    }

    /// The vision tower the HF checkpoint `dir` ships (`model.visual.*`), if it has one, for a
    /// language model of architecture `arch` and width `n_embd`.
    pub(super) fn from_checkpoint(
        dir: &Path,
        arch: &str,
        n_embd: usize,
        tokenizer: &Tokenizer,
        threads: usize,
    ) -> Result<Option<Self>, String> {
        if Self::check_arch(arch).is_err() {
            return Ok(None);
        }
        let Some(g) = crate::hf::open_vision(dir).map_err(|e| e.to_string())? else {
            return Ok(None);
        };
        Self::from_gguf(&g, n_embd, tokenizer, threads).map(Some)
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
        tracing::info!(
            tower = %g.path().display(),
            layers = encoder.config().n_layer,
            "vision: images accepted (the vision tower runs on the CPU)"
        );
        Ok(Self {
            encoder: Mutex::new(encoder),
            image_pad,
        })
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
        let hp = self
            .encoder
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .hparams()
            .clone();
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
            let pixels = crate::vision::preprocess(&hp, &img.bytes)
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
            let t = std::time::Instant::now();
            let e = blocking(|| {
                self.encoder
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
            n_embd = e.n_embd;
            for (i, k) in key[p.start..p.start + e.n_tokens()].iter_mut().enumerate() {
                *k = image_key(p.hash, i);
            }
            spans.push(ImageSpan {
                start: p.start,
                nx: e.nx,
                ny: e.ny,
                rows: e.data.into(),
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
