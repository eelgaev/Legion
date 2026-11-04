//! A buffer type that allocates more than it is asked for, for compute allocators that grow.
//!
//! A kept graph's compute buffer is sized to its graph. A long prompt's chunks each need a few
//! more bytes than the one before (the listed pools grow with the context), and ggml's graph
//! allocator frees and reallocates its buffer whenever the next graph needs more than it has:
//! on a GPU that is a `cudaFree` and a `cudaMalloc` of 1-2 GiB per new shape (~20 ms, and the
//! device synchronizes, so it cannot overlap the stage's compute). Giving the allocator a buffer
//! with headroom lets the next several shapes fit.
//!
//! The headroom only comes out of memory the device has to spare (#490). Padded allocators
//! serve the steps of a lent expert cache, which take most of the device: a padded 11.8K-token
//! step graph once left ggml-cuda's memory pool 148 MiB, and the pool's next allocation inside
//! the graph aborted the process (`cuMemCreate` out of memory). An allocation now pads only
//! while [`PAD_KEEPS_FREE`] stays free beside it; a graph that needs more grows the buffer as
//! before.

use std::collections::HashMap;
use std::ffi::c_char;
use std::sync::{Mutex, OnceLock};

use super::ffi;

#[cfg(test)]
mod tests;

/// Device memory padding leaves free: for what the backend allocates as graphs compute
/// (ggml-cuda's pool, cuBLAS workspaces) and the smaller compute buffers allocated after a big
/// one, as the expert cache keeps beside itself (`ResidencyConfig::cache_reserve`).
const PAD_KEEPS_FREE: usize = 1 << 30;

/// How much a padded type adds to what it is asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Pad {
    /// A quarter and 64 MiB ([`padded`]): a lent expert cache's steps, whose graphs grow by
    /// 8-58 MiB per chunk.
    Quarter,
    /// A sixty-fourth and 16 MiB ([`light`]): steps whose graphs grow by a few KiB per new KV
    /// length (#363: an integrated GPU's prompt chunks, ~0.04 MiB per shape).
    Light,
}

/// Bytes to allocate for a request of `size` with `free` bytes free on the device: a quarter of
/// the request and 64 MiB more, as far as [`PAD_KEEPS_FREE`] stays free.
fn padded(size: usize, free: usize) -> usize {
    pad_by(size, free, size / 4 + (64 << 20))
}

/// [`padded`] with [`Pad::Light`]'s headroom: a sixty-fourth of the request and 16 MiB more.
fn light(size: usize, free: usize) -> usize {
    pad_by(size, free, size / 64 + (16 << 20))
}

/// `size` and up to `extra` more, as far as [`PAD_KEEPS_FREE`] stays free.
fn pad_by(size: usize, free: usize, extra: usize) -> usize {
    let room = free.saturating_sub(size).saturating_sub(PAD_KEEPS_FREE);
    size + extra.min(room)
}

/// Free memory of the device `buft` allocates on; unbounded for a type without a device.
unsafe fn free_on(buft: ffi::ggml_backend_buffer_type_t) -> usize {
    // SAFETY: a live buffer type; its device, when it has one, lives as long.
    let dev = unsafe { ffi::ggml_backend_buft_get_device(buft) };
    if dev.is_null() {
        return usize::MAX;
    }
    // SAFETY: a registry device lives for the whole process.
    let dev = unsafe { super::Device::from_raw(std::ptr::NonNull::new_unchecked(dev)) };
    dev.memory().0
}

struct Padded {
    inner: ffi::ggml_backend_buffer_type_t,
    pad: Pad,
    buft: ffi::ggml_backend_buffer_type,
}

// SAFETY: buffer types are static descriptors, only read after registration.
unsafe impl Send for Padded {}
unsafe impl Sync for Padded {}

/// The inner type of a padded one (its `context`).
unsafe fn inner(buft: ffi::ggml_backend_buffer_type_t) -> ffi::ggml_backend_buffer_type_t {
    // SAFETY: `buft` is a `Padded::buft`, whose context points at its `Padded`.
    unsafe { (*((*buft).context as *const Padded)).inner }
}

/// The padding of a padded type.
unsafe fn pad_of(buft: ffi::ggml_backend_buffer_type_t) -> Pad {
    // SAFETY: as for `inner`.
    unsafe { (*((*buft).context as *const Padded)).pad }
}

unsafe extern "C" fn name(buft: ffi::ggml_backend_buffer_type_t) -> *const c_char {
    unsafe { ffi::ggml_backend_buft_name(inner(buft)) }
}

unsafe extern "C" fn alloc_buffer(
    buft: ffi::ggml_backend_buffer_type_t,
    size: usize,
) -> ffi::ggml_backend_buffer_t {
    // The buffer is the inner type's own, just bigger: the allocator sees its size.
    unsafe {
        let inner = inner(buft);
        let size = match pad_of(buft) {
            Pad::Quarter => padded(size, free_on(inner)),
            Pad::Light => light(size, free_on(inner)),
        };
        ffi::ggml_backend_buft_alloc_buffer(inner, size)
    }
}

unsafe extern "C" fn alignment(buft: ffi::ggml_backend_buffer_type_t) -> usize {
    unsafe { ffi::ggml_backend_buft_get_alignment(inner(buft)) }
}

unsafe extern "C" fn max_size(buft: ffi::ggml_backend_buffer_type_t) -> usize {
    unsafe { ffi::ggml_backend_buft_get_max_size(inner(buft)) }
}

unsafe extern "C" fn alloc_size(
    buft: ffi::ggml_backend_buffer_type_t,
    tensor: *const ffi::ggml_tensor,
) -> usize {
    unsafe { ffi::ggml_backend_buft_get_alloc_size(inner(buft), tensor) }
}

unsafe extern "C" fn is_host(buft: ffi::ggml_backend_buffer_type_t) -> bool {
    unsafe { ffi::ggml_backend_buft_is_host(inner(buft)) }
}

/// The type over `base` padded by `pad` (one per base type and padding, kept for the process).
pub(super) fn over(
    base: ffi::ggml_backend_buffer_type_t,
    pad: Pad,
) -> ffi::ggml_backend_buffer_type_t {
    static TYPES: OnceLock<Mutex<HashMap<(usize, Pad), usize>>> = OnceLock::new();
    let mut types = TYPES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();
    let at = types.entry((base as usize, pad)).or_insert_with(|| {
        let p: &'static mut Padded = Box::leak(Box::new(Padded {
            inner: base,
            pad,
            // SAFETY: a plain-data struct of function pointers and a pointer, all null for now.
            buft: unsafe { std::mem::zeroed() },
        }));
        p.buft.iface.get_name = Some(name);
        p.buft.iface.alloc_buffer = Some(alloc_buffer);
        p.buft.iface.get_alignment = Some(alignment);
        p.buft.iface.get_max_size = Some(max_size);
        p.buft.iface.get_alloc_size = Some(alloc_size);
        p.buft.iface.is_host = Some(is_host);
        // SAFETY: `base` is a live buffer type.
        p.buft.device = unsafe { (*base).device };
        let ctx = p as *const Padded as *mut std::ffi::c_void;
        p.buft.context = ctx;
        &mut p.buft as *mut ffi::ggml_backend_buffer_type as usize
    });
    *at as ffi::ggml_backend_buffer_type_t
}
