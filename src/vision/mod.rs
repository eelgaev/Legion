//! Vision for Qwen3-VL mmproj files: image preprocessing ([`preprocess`]) and the vision tower
//! ([`Encoder`]).

mod decode;
mod preprocess;
mod qwen3vl;
pub mod tiny;

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

/// Preprocessing parameters from a `clip` / `qwen3vl_merger` mmproj.
/// Resizing uses bicubic interpolation with centered black padding, as in mtmd.
#[derive(Debug, Clone)]
pub struct VisionHparams {
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
        let projector = g.get_str("clip.projector_type").unwrap_or_default();
        if arch != "clip" || projector != "qwen3vl_merger" {
            return Err(VisionError::Unsupported(format!("{arch}/{projector}")));
        }
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
        let spatial_merge_size = integer("spatial_merge_size", Some(2))?;
        // Ported from llama.cpp's tools/mtmd/clip.cpp (MIT),
        // PROJECTOR_TYPE_QWEN3VL and clip_hparams::set_limit_image_tokens.
        // image_size is a nominal model size, not the dynamic resize target.
        let area = patch_size
            .checked_mul(spatial_merge_size)
            .and_then(|f| f.checked_mul(f))
            .ok_or(VisionError::Dimensions)?;
        let max_pixels = area.checked_mul(4096).ok_or(VisionError::Dimensions)?;
        let hp = Self {
            patch_size,
            spatial_merge_size,
            image_min_pixels: integer("image_min_pixels", Some(area * 8))?,
            image_max_pixels: integer("image_max_pixels", Some(max_pixels))?,
            image_mean: array("image_mean")?,
            image_std: array("image_std")?,
        };
        hp.validate()?;
        Ok(hp)
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
