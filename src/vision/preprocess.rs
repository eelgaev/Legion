use super::{VisionError, VisionHparams};

/// A single image ready for the clip graph input, before patch embedding.
#[derive(Debug)]
pub struct Preprocessed {
    pub width: u32,
    pub height: u32,
    /// Patch grid before spatial merging: width/patch_size, height/patch_size.
    pub grid_w: u32,
    pub grid_h: u32,
    /// Planar RGB `[3][height][width]`, x contiguous. This matches `inp_raw`
    /// in llama.cpp's clip.cpp, not the interleaved intermediate clip_image_f32.
    /// Each value is `(byte / 255.0 - image_mean[channel]) / image_std[channel]`.
    pub pixels: Vec<f32>,
}

/// Decode PNG, JPEG, BMP, WebP or GIF (first frame), resize and normalize an image.
/// Alpha is discarded, as in mtmd's three-channel stb_image decode.
/// WebP uses the image crate and has no llama.cpp parity requirement.
pub fn preprocess(hp: &VisionHparams, bytes: &[u8]) -> Result<Preprocessed, VisionError> {
    hp.validate()?;
    let img = super::decode::decode(bytes)?;
    let (width, height) = smart_resize(
        img.width(),
        img.height(),
        hp.patch_size * hp.spatial_merge_size,
        hp.image_min_pixels,
        hp.image_max_pixels,
    )?;

    // Ported from llama.cpp's tools/mtmd/mtmd-image.cpp (MIT),
    // img_tool::resize with PAD_CEIL and mtmd_image_preprocessor_dyn_size.
    let scale = (width as f32 / img.width() as f32).min(height as f32 / img.height() as f32);
    let resized_w = ((img.width() as f32 * scale).ceil() as u32).min(width);
    let resized_h = ((img.height() as f32 * scale).ceil() as u32).min(height);
    let rgb = resize_bicubic(
        img.as_raw(),
        img.width() as usize,
        img.height() as usize,
        resized_w as usize,
        resized_h as usize,
    );
    let n = width as usize * height as usize;
    let mut pixels = vec![0.0; 3 * n];
    let offset_x = (width - resized_w) / 2;
    let offset_y = (height - resized_h) / 2;
    for channel in 0..3 {
        let plane = &mut pixels[channel * n..(channel + 1) * n];
        plane.fill((0.0 - hp.image_mean[channel]) / hp.image_std[channel]);
        for y in 0..resized_h as usize {
            for x in 0..resized_w as usize {
                let byte = rgb[3 * (y * resized_w as usize + x) + channel];
                plane[(y + offset_y as usize) * width as usize + x + offset_x as usize] =
                    (byte as f32 / 255.0 - hp.image_mean[channel]) / hp.image_std[channel];
            }
        }
    }
    Ok(Preprocessed {
        width,
        height,
        grid_w: width / hp.patch_size,
        grid_h: height / hp.patch_size,
        pixels,
    })
}

// Ported from img_tool::calc_size_preserved_ratio in
// llama.cpp's tools/mtmd/mtmd-image.cpp (MIT). Keep f32 arithmetic
// and round-half-away-from-zero, including the one-factor floor on thin images.
fn smart_resize(
    width: u32,
    height: u32,
    factor: u32,
    min: u32,
    max: u32,
) -> Result<(u32, u32), VisionError> {
    if width == 0 || height == 0 || factor == 0 || min == 0 || max < min {
        return Err(VisionError::Dimensions);
    }
    let f = factor as f32;
    let (w, h) = (width as f32, height as f32);
    let mut rw = (w / f).round().max(1.0) * f;
    let mut rh = (h / f).round().max(1.0) * f;
    if rw as f64 * rh as f64 > max as f64 {
        let beta = (h * w / max as f32).sqrt();
        rh = (h / beta / f).floor().max(1.0) * f;
        rw = (w / beta / f).floor().max(1.0) * f;
    } else if rw as f64 * (rh as f64) < min as f64 {
        let beta = (min as f32 / (h * w)).sqrt();
        rh = (h * beta / f).ceil() * f;
        rw = (w * beta / f).ceil() * f;
    }
    if !(1.0..=65536.0).contains(&rw) || !(1.0..=65536.0).contains(&rh) {
        return Err(VisionError::Dimensions);
    }
    Ok((rw as u32, rh as u32))
}

// Ported from img_tool::resize_pillow in
// llama.cpp's tools/mtmd/mtmd-image.cpp (MIT). Unlike image's floating
// point filters, mtmd rounds 22-bit fixed-point weights and clips to u8 after
// EACH separable pass. Changing either rounding step changes encoder inputs.
fn coefficients(input: usize, output: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    (0..output)
        .map(|out| {
            let center = (out as f64 + 0.5) * scale;
            let start = ((center - support + 0.5) as i64).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(input);
            let mut weights: Vec<f64> = (start..end)
                .map(|i| {
                    let x = ((i as f64 - center + 0.5) * (1.0 / filter_scale)).abs();
                    if x < 1.0 {
                        (1.5 * x - 2.5) * x * x + 1.0
                    } else if x < 2.0 {
                        (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5
                    } else {
                        0.0
                    }
                })
                .collect();
            let sum: f64 = weights.iter().sum();
            if sum != 0.0 {
                weights.iter_mut().for_each(|w| *w /= sum);
            }
            (
                start,
                weights
                    .into_iter()
                    .map(|w| (w * (1 << 22) as f64 + if w < 0.0 { -0.5 } else { 0.5 }) as i32)
                    .collect(),
            )
        })
        .collect()
}

fn resize_bicubic(src: &[u8], width: usize, height: usize, out_w: usize, out_h: usize) -> Vec<u8> {
    let horizontal = if width == out_w {
        src.to_vec()
    } else {
        let coeffs = coefficients(width, out_w);
        let mut dst = vec![0; out_w * height * 3];
        for y in 0..height {
            for (x, (start, weights)) in coeffs.iter().enumerate() {
                for c in 0..3 {
                    let mut acc = 1 << 21;
                    for (i, weight) in weights.iter().enumerate() {
                        acc += src[3 * (y * width + start + i) + c] as i32 * weight;
                    }
                    dst[3 * (y * out_w + x) + c] = (acc >> 22).clamp(0, 255) as u8;
                }
            }
        }
        dst
    };
    if height == out_h {
        return horizontal;
    }
    let coeffs = coefficients(height, out_h);
    let row = out_w * 3;
    let mut dst = vec![0; row * out_h];
    for (y, (start, weights)) in coeffs.iter().enumerate() {
        for x in 0..row {
            let mut acc = 1 << 21;
            for (i, weight) in weights.iter().enumerate() {
                acc += horizontal[(start + i) * row + x] as i32 * weight;
            }
            dst[y * row + x] = (acc >> 22).clamp(0, 255) as u8;
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_table() {
        // factor 32, area range [8192, 4194304]. Tiny square grows to
        // ceil(sqrt(8192)/32)*32=96; huge square shrinks to sqrt(max)=2048.
        for (w, h, expected) in [
            (1, 1, (96, 96)),
            (32, 32, (96, 96)),
            (64, 128, (64, 128)),
            (80, 112, (96, 128)),
            (10000, 10000, (2048, 2048)),
            (4096, 1024, (4096, 1024)),
            (8192, 2048, (4096, 1024)),
            // Aligned area is already in budget: no aspect-ratio rejection.
            (1, 4096, (32, 4096)),
            (4096, 1, (4096, 32)),
            // The first aligned area meets min; the second grows by
            // sqrt(8192/128)=8, then ceils to (32, 1024).
            (1, 256, (32, 256)),
            (1, 128, (32, 1024)),
        ] {
            assert_eq!(
                smart_resize(w, h, 32, 8192, 4194304).unwrap(),
                expected,
                "{w}x{h}"
            );
        }
        assert!(smart_resize(0, 10, 32, 8192, 4194304).is_err());
        assert!(smart_resize(u32::MAX, 1, 32, 8192, 4194304).is_err());
    }
}
