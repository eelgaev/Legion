use super::{over, reserved_len};
use crate::ggml::{Backend, Context, ffi, types};

#[test]
fn reserves_whole_pages_and_at_least_one() {
    assert_eq!(reserved_len(0, 4096), 4096);
    assert_eq!(reserved_len(1, 4096), 4096);
    assert_eq!(reserved_len(4096, 4096), 4096);
    assert_eq!(reserved_len(4097, 4096), 8192);
    assert_eq!(reserved_len(usize::MAX, 4096), usize::MAX);
}

/// The type lays tensors out as its base type does: the same alignment, maximum and
/// allocation sizes, one type per base.
#[test]
fn sizes_are_the_base_types() {
    let cpu = Backend::cpu().unwrap().buffer_type().raw();
    let u = over(cpu);
    assert_eq!(u, over(cpu), "one type per base");
    let ctx = Context::new(4, 4);
    let t = ctx.new_tensor(types::F32, &[256, 3]).raw();
    // SAFETY: live buffer types and a valid tensor.
    unsafe {
        assert_eq!(
            ffi::ggml_backend_buft_get_alignment(u),
            ffi::ggml_backend_buft_get_alignment(cpu)
        );
        assert_eq!(
            ffi::ggml_backend_buft_get_max_size(u),
            ffi::ggml_backend_buft_get_max_size(cpu)
        );
        assert_eq!(
            ffi::ggml_backend_buft_get_alloc_size(u, t),
            ffi::ggml_backend_buft_get_alloc_size(cpu, t)
        );
        assert!(
            !ffi::ggml_backend_buft_is_host(u),
            "nothing behind it to read"
        );
    }
}

/// A buffer is as big as asked, at an address of its own, aligned as the base type aligns,
/// and takes tensors (their addresses only: nothing is written).
#[test]
fn buffers_take_tensors_at_addresses_of_their_own() {
    let u = over(Backend::cpu().unwrap().buffer_type().raw());
    let size = 3usize << 30;
    // SAFETY: a live buffer type; each buffer is freed once, after its last use.
    unsafe {
        let a = ffi::ggml_backend_buft_alloc_buffer(u, size);
        let b = ffi::ggml_backend_buft_alloc_buffer(u, size);
        assert!(!a.is_null() && !b.is_null(), "3 GiB of address space each");
        assert_eq!(ffi::ggml_backend_buffer_get_size(a), size);
        let (a0, b0) = (
            ffi::ggml_backend_buffer_get_base(a) as usize,
            ffi::ggml_backend_buffer_get_base(b) as usize,
        );
        assert!(a0 + size <= b0 || b0 + size <= a0, "apart");
        assert_eq!(a0 % ffi::ggml_backend_buft_get_alignment(u), 0);
        let ctx = Context::new(4, 4);
        let t = ctx.new_tensor(types::F32, &[1 << 20]).raw();
        let at = (a0 + size - (4 << 20)) as *mut std::ffi::c_void;
        assert_eq!(
            ffi::ggml_backend_tensor_alloc(a, t, at),
            ffi::ggml_status_GGML_STATUS_SUCCESS
        );
        assert_eq!((*t).data, at);
        assert_eq!((*t).buffer, a);
        (*t).data = std::ptr::null_mut();
        (*t).buffer = std::ptr::null_mut();
        ffi::ggml_backend_buffer_free(a);
        ffi::ggml_backend_buffer_free(b);
    }
}
