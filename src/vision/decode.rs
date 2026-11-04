use std::ffi::{c_int, c_void};
use std::io::Cursor;
use std::ptr::NonNull;

use super::VisionError;

unsafe extern "C" {
    fn stbi_load_from_memory(
        buffer: *const u8,
        len: c_int,
        x: *mut c_int,
        y: *mut c_int,
        channels_in_file: *mut c_int,
        desired_channels: c_int,
    ) -> *mut u8;
    fn stbi_image_free(data: *mut c_void);
}

struct StbImage(NonNull<u8>);

impl Drop for StbImage {
    fn drop(&mut self) {
        // SAFETY: this is the unique owner of an allocation returned by stb.
        unsafe { stbi_image_free(self.0.as_ptr().cast()) };
    }
}

pub(super) fn decode(bytes: &[u8]) -> Result<image::RgbImage, VisionError> {
    if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        // stb has no WebP support. This path has no llama.cpp parity requirement.
        return Ok(
            image::ImageReader::with_format(Cursor::new(bytes), image::ImageFormat::WebP)
                .decode()?
                .to_rgb8(),
        );
    }
    let len = c_int::try_from(bytes.len()).map_err(|_| VisionError::Dimensions)?;
    if len == 0 {
        return Err(VisionError::StbDecode);
    }
    let (mut width, mut height, mut channels) = (0, 0, 0);
    // Same call as llama.cpp's tools/mtmd/mtmd-helper.cpp (MIT):
    // force RGB, discard alpha and decode only the first GIF frame.
    // SAFETY: bytes holds len readable bytes, and each output points to a live
    // c_int. stb retains none of these pointers after returning.
    let ptr = unsafe {
        stbi_load_from_memory(
            bytes.as_ptr(),
            len,
            &mut width,
            &mut height,
            &mut channels,
            3,
        )
    };
    let data = StbImage(NonNull::new(ptr).ok_or(VisionError::StbDecode)?);
    let width = u32::try_from(width).map_err(|_| VisionError::Dimensions)?;
    let height = u32::try_from(height).map_err(|_| VisionError::Dimensions)?;
    let size = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(3))
        .filter(|&n| n > 0 && n <= isize::MAX as usize)
        .ok_or(VisionError::Dimensions)?;
    // SAFETY: a successful stb decode allocates width*height*desired_channels
    // initialized bytes. The allocation remains alive until after this copy.
    let pixels = unsafe { std::slice::from_raw_parts(data.0.as_ptr(), size) }.to_vec();
    image::RgbImage::from_raw(width, height, pixels).ok_or(VisionError::Dimensions)
}
