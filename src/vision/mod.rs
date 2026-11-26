//! Vision for Qwen3-VL, Gemma 4 and GLM-5.3-Flash mmproj files: image preprocessing
//! ([`preprocess`]) and the vision tower ([`Encoder`]).

mod decode;
mod gemma4v;
mod glm5v;
#[cfg(feature = "cuda")]
mod gpu;
mod preprocess;
mod qwen3vl;
pub mod share;
pub mod tiny;

#[cfg(feature = "cuda")]
pub use gpu::GpuEncoder;
pub use preprocess::{Preprocessed, preprocess};
pub use qwen3vl::{Encoder, EncoderConfig, ImageEmbedding};

use crate::gguf::GgufFile;

#[derive(Debug, thiserror::Error)]
pub enum VisionError {
    #[error("unsupported vision architecture/projector: {0}")]
    Unsupported(String),
    #[error("missing or invalid vision metadata: {0}")]
    Metadata(String),
    #[error("image dimensions are out of range")]
    Dimensions,
    #[error("invalid or unsupported image for stb_image decoding")]
    StbDecode,
    #[error(transparent)]
    Decode(#[from] image::ImageError),
    #[error("vision encoder: {0}")]
    Compute(String),
}

/// The tower an mmproj holds (`clip.projector_type`, or `clip.vision.projector_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Projector {
    /// Qwen3-VL's ViT and 2×2 merger (Qwen3.5 / Qwen3.6 / Qwen3.8-Flash-Next).
    Qwen3vl,
    /// Gemma 4's ViT, 3×3 average pool and projection (`gemma4v`).
    Gemma4v,
    /// GLM-5.3-Flash's ViT, 2×2 convolution and SwiGLU projector (`glm5v`).
    Glm5v,
}

/// Preprocessing parameters from a `clip` mmproj.
/// Resizing uses bicubic interpolation, as in mtmd: Qwen3-VL's keeps the aspect ratio with
/// centered black padding, Gemma 4's resizes to the aligned size itself, GLM-5.3-Flash's
/// places the image at the top left of an aligned black canvas.
#[derive(Debug, Clone)]
pub struct VisionHparams {
    pub projector: Projector,
    pub patch_size: u32,
    pub spatial_merge_size: u32,
    pub image_min_pixels: u32,
    pub image_max_pixels: u32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl VisionHparams {
    pub fn from_gguf(g: &GgufFile) -> Result<Self, VisionError> {
        let arch = g.architecture().unwrap_or_default();
        let projector = g
            .get_str("clip.vision.projector_type")
            .or_else(|| g.get_str("clip.projector_type"))
            .unwrap_or_default();
        let kind = match (arch.as_str(), projector.as_str()) {
            ("clip", "qwen3vl_merger") => Projector::Qwen3vl,
            ("clip", "gemma4v") => Projector::Gemma4v,
            ("clip", "glm5v") => Projector::Glm5v,
            _ => return Err(VisionError::Unsupported(format!("{arch}/{projector}"))),
        };
        let integer = |name: &str, default: Option<u32>| {
            let key = format!("clip.vision.{name}");
            match g.get(&key) {
                Some(value) => value.as_u64().and_then(|v| u32::try_from(v).ok()),
                None => default,
            }
            .filter(|&v| v > 0)
            .ok_or(VisionError::Metadata(key))
        };
        let array = |name: &str| -> Result<[f32; 3], VisionError> {
            let key = format!("clip.vision.{name}");
            let value = g
                .get(&key)
                .ok_or_else(|| VisionError::Metadata(key.clone()))?;
            let values = value
                .as_array()
                .filter(|v| v.len() >= 3)
                .ok_or_else(|| VisionError::Metadata(key.clone()))?;
            let mut result = [0.0; 3];
            for (out, value) in result.iter_mut().zip(values) {
                *out = value
                    .as_f64()
                    .ok_or_else(|| VisionError::Metadata(key.clone()))?
                    as f32;
            }
            Ok(result)
        };
        let patch_size = integer("patch_size", None)?;
        // Ported from llama.cpp's tools/mtmd/clip.cpp (MIT),
        // PROJECTOR_TYPE_QWEN3VL / PROJECTOR_TYPE_GEMMA4V and
        // clip_hparams::set_limit_image_tokens: Qwen3-VL merges 2×2 patches into one token and
        // takes 8 to 4096 of them, Gemma 4 pools 3×3 (`projector.scale_factor` when present)
        // and takes 70 to 1120. image_size is a nominal model size, not the dynamic resize
        // target. GLM-5.3-Flash's (GLM5V) limits are required keys of its mmproj (the
        // processor's 16 to 8000 tokens as pixels); the defaults here are those.
        let (spatial_merge_size, tokens) = match kind {
            Projector::Qwen3vl => (integer("spatial_merge_size", Some(2))?, (8, 4096)),
            Projector::Gemma4v => (integer("projector.scale_factor", Some(3))?, (70, 1120)),
            Projector::Glm5v => (integer("spatial_merge_size", Some(2))?, (16, 8000)),
        };
        let area = patch_size
            .checked_mul(spatial_merge_size)
            .and_then(|f| f.checked_mul(f))
            .ok_or(VisionError::Dimensions)?;
        let min_pixels = area.checked_mul(tokens.0).ok_or(VisionError::Dimensions)?;
        let max_pixels = area.checked_mul(tokens.1).ok_or(VisionError::Dimensions)?;
        let hp = Self {
            projector: kind,
            patch_size,
            spatial_merge_size,
            image_min_pixels: integer("image_min_pixels", Some(min_pixels))?,
            image_max_pixels: integer("image_max_pixels", Some(max_pixels))?,
            image_mean: array("image_mean")?,
            image_std: array("image_std")?,
        };
        hp.validate()?;
        Ok(hp)
    }

    /// The most tokens one image takes: the pixel budget over a token's pixels (the resize
    /// keeps every image within the budget, [`preprocess`], and each token is a merged patch).
    /// Gemma 4's: 1,120 by default.
    pub fn max_image_tokens(&self) -> usize {
        let f = self.patch_size as usize * self.spatial_merge_size as usize;
        self.image_max_pixels as usize / (f * f).max(1)
    }

    pub(super) fn validate(&self) -> Result<(), VisionError> {
        if self.patch_size == 0
            || self.spatial_merge_size == 0
            || self
                .patch_size
                .checked_mul(self.spatial_merge_size)
                .is_none_or(|v| v > 65536)
            || self.image_min_pixels == 0
            || self.image_max_pixels < self.image_min_pixels
            || self.image_max_pixels > i32::MAX as u32
        {
            return Err(VisionError::Metadata(
                "patch/merge size or pixel limits".into(),
            ));
        }
        if self.image_mean.iter().any(|v| !v.is_finite())
            || self.image_std.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(VisionError::Metadata("image_mean/image_std".into()));
        }
        Ok(())
    }
}
