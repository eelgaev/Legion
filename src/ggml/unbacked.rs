//! A buffer type whose buffers are address space only, for plans that only lay tensors out
//! (#1183).
//!
//! A graph a relayout rewrites is planned twice: as built, so the rewrite reads off that
//! layout which sums ggml-cuda fuses, then as rewritten, the plan it runs in. ggml's buffers
//! only grow, so planning both in one allocator's buffer sized it for the bigger plan: a
//! 16,384-token Qwen3.8-Flash-Next prompt step's as built peaked at 6,982 MiB and its plan as
//! rewritten at 5,542. The plan as built only needs offsets. A second graph allocator over this
//! type ([`super::ComputeAllocator`]'s shadow) plans it into a buffer that reserves address
//! space and no memory: the tensors get addresses the rewrite can compare (no other buffer's,
//! the range is the process's), and the type's alignment, maximum and allocation sizes are its
//! base type's, so the offsets are the ones the base type's buffer would give them. Nothing may
//! read or write the tensors placed there: the range is mapped without access, and the buffers
//! have no tensor reads, writes or clears.

use std::collections::HashMap;
use std::ffi::{c_char, c_void};
use std::sync::{Mutex, OnceLock};

use super::ffi;

#[cfg(test)]
mod tests;

/// An unbacked type: its base type, and itself.
struct Unbacked {
    base: ffi::ggml_backend_buffer_type_t,
    buft: ffi::ggml_backend_buffer_type,
}

// SAFETY: buffer types are static descriptors, only read after registration.
unsafe impl Send for Unbacked {}
unsafe impl Sync for Unbacked {}

/// An unbacked buffer's address range.
struct Range {
    start: *mut c_void,
    len: usize,
}

/// The base type of an unbacked one (its `context`).
unsafe fn base(buft: ffi::ggml_backend_buffer_type_t) -> ffi::ggml_backend_buffer_type_t {
    // SAFETY: `buft` is an `Unbacked::buft`, whose context points at its `Unbacked`.
    unsafe { (*((*buft).context as *const Unbacked)).base }
}

unsafe extern "C" fn name(buft: ffi::ggml_backend_buffer_type_t) -> *const c_char {
    unsafe { ffi::ggml_backend_buft_name(base(buft)) }
}

/// Bytes of address space reserved for a buffer of `size`: at least one page.
fn reserved_len(size: usize, page: usize) -> usize {
    size.max(1).div_ceil(page).saturating_mul(page)
}

unsafe extern "C" fn alloc_buffer(
    buft: ffi::ggml_backend_buffer_type_t,
    size: usize,
) -> ffi::ggml_backend_buffer_t {
    // SAFETY: sysconf has no preconditions.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page = usize::try_from(page)
        .ok()
        .filter(|&p| p > 0)
        .unwrap_or(4096);
    let len = reserved_len(size, page);
    // SAFETY: a fresh private mapping without access and without memory reserved for it:
    // address space only, unmapped by `free_buffer`.
    let start = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
            -1,
            0,
        )
    };
    if start == libc::MAP_FAILED {
        return std::ptr::null_mut();
    }
    let iface = ffi::ggml_backend_buffer_i {
        free_buffer: Some(free_buffer),
        get_base: Some(get_base),
        init_tensor: None,
        memset_tensor: None,
        set_tensor: None,
        get_tensor: None,
        set_tensor_2d: None,
        get_tensor_2d: None,
        cpy_tensor: None,
        clear: None,
        reset: None,
    };
    let range = Box::new(Range { start, len });
    // SAFETY: ggml takes the interface and the boxed range (freed by `free_buffer`).
    unsafe { ffi::ggml_backend_buffer_init(buft, iface, Box::into_raw(range).cast(), size) }
}

unsafe extern "C" fn free_buffer(buffer: ffi::ggml_backend_buffer_t) {
    // SAFETY: the buffer's context is the range `alloc_buffer` boxed, freed once here.
    unsafe {
        let range = Box::from_raw((*buffer).context as *mut Range);
        libc::munmap(range.start, range.len);
    }
}

unsafe extern "C" fn get_base(buffer: ffi::ggml_backend_buffer_t) -> *mut c_void {
    // SAFETY: as in `free_buffer`; the range lives as long as the buffer.
    unsafe { (*((*buffer).context as *const Range)).start }
}

unsafe extern "C" fn alignment(buft: ffi::ggml_backend_buffer_type_t) -> usize {
    unsafe { ffi::ggml_backend_buft_get_alignment(base(buft)) }
}

unsafe extern "C" fn max_size(buft: ffi::ggml_backend_buffer_type_t) -> usize {
    unsafe { ffi::ggml_backend_buft_get_max_size(base(buft)) }
}

unsafe extern "C" fn alloc_size(
    buft: ffi::ggml_backend_buffer_type_t,
    tensor: *const ffi::ggml_tensor,
) -> usize {
    unsafe { ffi::ggml_backend_buft_get_alloc_size(base(buft), tensor) }
}

/// No host can read the buffers' tensors.
unsafe extern "C" fn is_host(_buft: ffi::ggml_backend_buffer_type_t) -> bool {
    false
}

/// The unbacked type over `base` (one per base type, kept for the process).
pub(super) fn over(base: ffi::ggml_backend_buffer_type_t) -> ffi::ggml_backend_buffer_type_t {
    static TYPES: OnceLock<Mutex<HashMap<usize, usize>>> = OnceLock::new();
    let mut types = TYPES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();
    let at = types.entry(base as usize).or_insert_with(|| {
        let u: &'static mut Unbacked = Box::leak(Box::new(Unbacked {
            base,
            // SAFETY: a plain-data struct of function pointers and a pointer, all null for now.
            buft: unsafe { std::mem::zeroed() },
        }));
        u.buft.iface.get_name = Some(name);
        u.buft.iface.alloc_buffer = Some(alloc_buffer);
        u.buft.iface.get_alignment = Some(alignment);
        u.buft.iface.get_max_size = Some(max_size);
        u.buft.iface.get_alloc_size = Some(alloc_size);
        u.buft.iface.is_host = Some(is_host);
        // SAFETY: `base` is a live buffer type.
        u.buft.device = unsafe { (*base).device };
        let ctx = u as *const Unbacked as *mut c_void;
        u.buft.context = ctx;
        &mut u.buft as *mut ffi::ggml_backend_buffer_type as usize
    });
    *at as ffi::ggml_backend_buffer_type_t
}
