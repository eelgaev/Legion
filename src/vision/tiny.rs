//! A tiny `qwen3vl_merger` mmproj with random weights, written as a GGUF, for tests that need
//! the vision tower's graph but no real projector: two blocks of width 16 (two heads), patches
//! of 4 pixels, an 8 × 8 learned position grid (resampled for any other grid), F32 weights.
//! Its outputs are meaningless; what it computes is the same as for a real projector.
//! Also [`bmp`], a 24-bit BMP of any size for test images (decoded by stb_image like a PNG).

use std::path::Path;

use crate::engine::sampling::Rng;
use crate::ggml::types;
use crate::gguf::{GgufError, GgufWriter, Value};

pub const N_EMBD: u32 = 16;
pub const PATCH: u32 = 4;
const N_HEAD: u32 = 2;
const N_FF: u32 = 32;
const N_LAYER: u32 = 2;
const POS_SIDE: u32 = 8;

/// Writes the tiny projector for a language model of width `n_out`; `deepstack` marks its first
/// block as a deepstack layer (which the encoder refuses).
pub fn write(
    path: impl AsRef<Path>,
    n_out: u32,
    seed: u64,
    deepstack: bool,
) -> Result<(), GgufError> {
    let mut w = GgufWriter::new();
    let u32v = Value::U32;
    let f3 = |v: f32| Value::Array(vec![Value::F32(v); 3]);
    w.set("general.architecture", &Value::String("clip".into()))?;
    w.set("general.type", &Value::String("mmproj".into()))?;
    w.set(
        "clip.projector_type",
        &Value::String("qwen3vl_merger".into()),
    )?;
    w.set("clip.has_vision_encoder", &Value::Bool(true))?;
    w.set("clip.use_gelu", &Value::Bool(true))?;
    for (k, v) in [
        ("projection_dim", u32v(n_out)),
        ("image_size", u32v(PATCH * POS_SIDE)),
        ("patch_size", u32v(PATCH)),
        ("embedding_length", u32v(N_EMBD)),
        ("feed_forward_length", u32v(N_FF)),
        ("block_count", u32v(N_LAYER)),
        ("attention.head_count", u32v(N_HEAD)),
        ("attention.layer_norm_epsilon", Value::F32(1e-6)),
        ("spatial_merge_size", u32v(2)),
        ("image_mean", f3(0.5)),
        ("image_std", f3(0.5)),
        (
            "is_deepstack_layers",
            Value::Array(
                (0..N_LAYER)
                    .map(|il| Value::Bool(deepstack && il == 0))
                    .collect(),
            ),
        ),
    ] {
        w.set(&format!("clip.vision.{k}"), &v)?;
    }
    let mut rng = Rng::new(seed);
    let mut add = |name: &str, shape: &[i64], a: f32| -> Result<(), GgufError> {
        let n: i64 = shape.iter().product();
        let data: Vec<u8> = (0..n)
            .flat_map(|_| ((rng.next_f64() as f32 * 2.0 - 1.0) * a).to_ne_bytes())
            .collect();
        w.add_tensor(name, types::F32, shape, &data).map(|_| ())
    };
    let (e, p, ff, o) = (N_EMBD as i64, PATCH as i64, N_FF as i64, n_out as i64);
    let scale = |fan_in: i64| 1.0 / (fan_in as f32).sqrt();
    add("v.patch_embd.weight", &[p, p, 3, e], scale(3 * p * p))?;
    add("v.patch_embd.weight.1", &[p, p, 3, e], scale(3 * p * p))?;
    add("v.patch_embd.bias", &[e], 0.1)?;
    add(
        "v.position_embd.weight",
        &[e, (POS_SIDE * POS_SIDE) as i64],
        0.5,
    )?;
    for il in 0..N_LAYER {
        let t = |n: &str| format!("v.blk.{il}.{n}");
        for ln in ["ln1", "ln2"] {
            add(&t(&format!("{ln}.weight")), &[e], 1.0)?;
            add(&t(&format!("{ln}.bias")), &[e], 0.1)?;
        }
        add(&t("attn_qkv.weight"), &[e, 3 * e], scale(e))?;
        add(&t("attn_qkv.bias"), &[3 * e], 0.1)?;
        add(&t("attn_out.weight"), &[e, e], scale(e))?;
        add(&t("attn_out.bias"), &[e], 0.1)?;
        add(&t("ffn_up.weight"), &[e, ff], scale(e))?;
        add(&t("ffn_up.bias"), &[ff], 0.1)?;
        add(&t("ffn_down.weight"), &[ff, e], scale(ff))?;
        add(&t("ffn_down.bias"), &[e], 0.1)?;
    }
    add("v.post_ln.weight", &[e], 1.0)?;
    add("v.post_ln.bias", &[e], 0.1)?;
    add("mm.0.weight", &[4 * e, 4 * e], scale(4 * e))?;
    add("mm.0.bias", &[4 * e], 0.1)?;
    add("mm.2.weight", &[4 * e, o], scale(4 * e))?;
    add("mm.2.bias", &[o], 0.1)?;
    w.write(path)
}

/// A `width × height` 24-bit BMP whose pixel `(x, y)` is `rgb(x, y)`.
pub fn bmp(width: u32, height: u32, rgb: impl Fn(u32, u32) -> [u8; 3]) -> Vec<u8> {
    let row = (3 * width).next_multiple_of(4);
    let size = 54 + row * height;
    let mut out = Vec::with_capacity(size as usize);
    out.extend_from_slice(b"BM");
    for v in [size, 0, 54, 40] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&(width as i32).to_le_bytes());
    out.extend_from_slice(&(height as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    for v in [0u32, row * height, 2835, 2835, 0, 0] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    // Bottom row first, BGR.
    for y in (0..height).rev() {
        let start = out.len();
        for x in 0..width {
            let [r, g, b] = rgb(x, y);
            out.extend_from_slice(&[b, g, r]);
        }
        out.resize(start + row as usize, 0);
    }
    out
}
