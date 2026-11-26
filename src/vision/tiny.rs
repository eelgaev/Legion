//! A tiny `qwen3vl_merger` mmproj with random weights, written as a GGUF, for tests that need
//! the vision tower's graph but no real projector: two blocks of width 16 (two heads), patches
//! of 4 pixels, an 8 × 8 learned position grid (resampled for any other grid), F32 weights.
//! Its outputs are meaningless; what it computes is the same as for a real projector.
//! [`write_gemma4v`] writes a tiny `gemma4v` one the same way ([`write_gemma4v_as`] also the
//! smaller towers' kind, with clipped linears). Also [`bmp`], a 24-bit BMP of any size for test images (decoded by stb_image like a PNG).

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

/// Rows of each of the tiny Gemma 4 tower's two position tables (patch grids up to this side).
pub const GEMMA_POS_ROWS: u32 = 128;

/// What a tiny Gemma 4 tower has beside its matrices: the 31B's standardization, or the smaller
/// towers' clipped linears (#1220).
#[derive(Clone, Copy, Debug)]
pub struct GemmaKind {
    pub standardize: bool,
    pub clipped: bool,
}

/// The 31B's tower: standardized, no clips.
pub const GEMMA_31B: GemmaKind = GemmaKind {
    standardize: true,
    clipped: false,
};

/// The E2B/E4B-sized towers (Nemotron 3.5 Bolt's): clipped linears, no standardization.
pub const GEMMA_E4B: GemmaKind = GemmaKind {
    standardize: false,
    clipped: true,
};

/// The tiny clipped tower's bounds of block `il`'s matrix `m`: `[input_min, input_max,
/// output_min, output_max]`, `None` where a scalar is left out (block 1's `attn_out` has only
/// `input_max` and `output_min`). Values exact in F16 that clamp a good share of the tiny
/// tower's activations; `q`, `k` and `v` share their input bounds, as `gate` and `up` do.
pub fn gemma_clip(il: u32, m: &str) -> [Option<f32>; 4] {
    let s = if il == 0 { 1.0 } else { 0.75 };
    let sym = |i: f32, o: f32| [Some(-i * s), Some(i * s * 0.875), Some(-o * s), Some(o * s)];
    match m {
        "attn_q" => sym(1.0, 0.5),
        "attn_k" => sym(1.0, 0.375),
        "attn_v" => sym(1.0, 0.25),
        "attn_out" if il == 1 => [None, Some(0.625), Some(-0.25), None],
        "attn_out" => sym(0.75, 0.375),
        "ffn_gate" => sym(1.25, 0.5),
        "ffn_up" => sym(1.25, 0.625),
        "ffn_down" => sym(0.125, 0.0625),
        _ => [None; 4],
    }
}

/// Writes a tiny `gemma4v` projector (Gemma 4's tower, #1217) for a language model of width
/// `n_out`: two blocks of width 16 (two heads of 8), patches of 4 pixels pooled 3 × 3, F32
/// weights, mean 0 and std 1 as Gemma's; the 31B's kind ([`GEMMA_31B`]).
pub fn write_gemma4v(path: impl AsRef<Path>, n_out: u32, seed: u64) -> Result<(), GgufError> {
    write_gemma4v_as(path, n_out, seed, GEMMA_31B)
}

/// [`write_gemma4v`] of `kind`: the same blocks for a seed whatever the kind (the clips draw
/// nothing), the clips' scalars F32 `[1]` as the converter writes them ([`gemma_clip`]).
pub fn write_gemma4v_as(
    path: impl AsRef<Path>,
    n_out: u32,
    seed: u64,
    kind: GemmaKind,
) -> Result<(), GgufError> {
    let none = |_: u32, _: &str| -> [Option<f32>; 4] { [None; 4] };
    let clip: &dyn Fn(u32, &str) -> [Option<f32>; 4] =
        if kind.clipped { &gemma_clip } else { &none };
    write_gemma4v_with(path, n_out, seed, kind.standardize, clip, &[])
}

/// [`write_gemma4v_as`] with the clips of `clip` (block, matrix -> its four scalars) and the
/// further F32 `[1]` tensors `extra` (name, value): towers the encoder refuses.
pub fn write_gemma4v_with(
    path: impl AsRef<Path>,
    n_out: u32,
    seed: u64,
    standardize: bool,
    clip: &dyn Fn(u32, &str) -> [Option<f32>; 4],
    extra: &[(&str, f32)],
) -> Result<(), GgufError> {
    let mut w = GgufWriter::new();
    let u32v = Value::U32;
    let f3 = |v: f32| Value::Array(vec![Value::F32(v); 3]);
    w.set("general.architecture", &Value::String("clip".into()))?;
    w.set("general.type", &Value::String("mmproj".into()))?;
    w.set("clip.has_vision_encoder", &Value::Bool(true))?;
    w.set("clip.use_gelu", &Value::Bool(true))?;
    w.set(
        "clip.vision.projector_type",
        &Value::String("gemma4v".into()),
    )?;
    for (k, v) in [
        ("projection_dim", u32v(n_out)),
        ("image_size", u32v(224)),
        ("patch_size", u32v(PATCH)),
        ("embedding_length", u32v(N_EMBD)),
        ("feed_forward_length", u32v(N_FF)),
        ("block_count", u32v(N_LAYER)),
        ("attention.head_count", u32v(N_HEAD)),
        ("attention.layer_norm_epsilon", Value::F32(1e-6)),
        ("image_mean", f3(0.0)),
        ("image_std", f3(1.0)),
    ] {
        w.set(&format!("clip.vision.{k}"), &v)?;
    }
    let mut rng = Rng::new(seed);
    let mut add = |name: &str, shape: &[i64], base: f32, a: f32| -> Result<(), GgufError> {
        let n: i64 = shape.iter().product();
        let data: Vec<u8> = (0..n)
            .flat_map(|_| (base + (rng.next_f64() as f32 * 2.0 - 1.0) * a).to_ne_bytes())
            .collect();
        w.add_tensor(name, types::F32, shape, &data).map(|_| ())
    };
    let (e, p, ff, o) = (N_EMBD as i64, PATCH as i64, N_FF as i64, n_out as i64);
    let d = e / N_HEAD as i64;
    let scale = |fan_in: i64| 1.0 / (fan_in as f32).sqrt();
    add("v.patch_embd.weight", &[p, p, 3, e], 0.0, scale(3 * p * p))?;
    add(
        "v.position_embd.weight",
        &[e, GEMMA_POS_ROWS as i64, 2],
        0.0,
        0.5,
    )?;
    for il in 0..N_LAYER {
        let t = |n: &str| format!("v.blk.{il}.{n}.weight");
        for ln in ["ln1", "ln2", "attn_post_norm", "ffn_post_norm"] {
            add(&t(ln), &[e], 1.0, 0.2)?;
        }
        for qk in ["attn_q_norm", "attn_k_norm"] {
            add(&t(qk), &[d], 1.0, 0.2)?;
        }
        for m in ["attn_q", "attn_k", "attn_v", "attn_out"] {
            add(&t(m), &[e, e], 0.0, scale(e))?;
        }
        add(&t("ffn_gate"), &[e, ff], 0.0, scale(e))?;
        add(&t("ffn_up"), &[e, ff], 0.0, scale(e))?;
        add(&t("ffn_down"), &[ff, e], 0.0, scale(ff))?;
    }
    if standardize {
        add("v.std_bias", &[e], 0.0, 0.5)?;
        add("v.std_scale", &[e], 1.0, 0.3)?;
    }
    add("mm.input_projection.weight", &[e, o], 0.0, scale(e))?;
    for il in 0..N_LAYER {
        for m in [
            "attn_q", "attn_k", "attn_v", "attn_out", "ffn_gate", "ffn_up", "ffn_down",
        ] {
            let names = ["input_min", "input_max", "output_min", "output_max"];
            for (s, v) in names.iter().zip(clip(il, m)) {
                if let Some(v) = v {
                    let name = format!("v.blk.{il}.{m}.{s}");
                    w.add_tensor(&name, types::F32, &[1], &v.to_ne_bytes())?;
                }
            }
        }
    }
    for (name, v) in extra {
        w.add_tensor(name, types::F32, &[1], &v.to_ne_bytes())?;
    }
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
