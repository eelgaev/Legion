//! Rows that stay where a graph wrote them, for another graph to read (#1181).
//!
//! A drafter that reads the target's hidden states (DFlash) used to get them on the host: every
//! verify step read all its captured rows back (128 rows of five 5,376-wide layers at 16
//! sequences of Gemma 4 31B, ~14 MB), the host picked the accepted ones, and the drafter's
//! take-in uploaded those again. [`RowsAt`] names a captured output where it is (its buffer and
//! address), so the drafter's graph gathers the rows it needs on the device instead. Its bytes
//! are the target graph's until that graph runs again: the reader uses them before then.
//! The MTP head's take-in of a prompt chunk copies them into its input on the device
//! ([`Backend::copy_rows`], #1183).

use super::backend::Backend;
use super::context::{Context, Tensor};
use super::ffi;

#[cfg(test)]
mod tests;

/// An allocated, contiguous F32 tensor `[width, rows]` of some graph, by where its bytes are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowsAt {
    /// Its `ggml_backend_buffer_t` and data pointer, as addresses (the struct is `Send`; only
    /// [`Context::over_rows`] turns them back into a tensor).
    buffer: usize,
    data: usize,
    /// Floats per row, and rows.
    pub width: usize,
    pub rows: usize,
    /// The CUDA device whose buffer holds it ([`Tensor::cuda_device`]); `None` for host memory.
    pub device: Option<usize>,
}

impl<'a> Tensor<'a> {
    /// Where this tensor's rows are, if it is an allocated, contiguous F32 matrix in a buffer
    /// that is not a meta buffer (a meta buffer's tensors are slices on several devices).
    pub fn rows_at(self) -> Option<RowsAt> {
        let [width, rows, d2, d3] = self.ne();
        if self.ty() != ffi::ggml_type_GGML_TYPE_F32
            || d2 != 1
            || d3 != 1
            || !self.is_contiguous()
            || self.is_meta()
        {
            return None;
        }
        // SAFETY: plain reads of a live tensor's metadata.
        let (buffer, data) = unsafe { ((*self.raw()).buffer, (*self.raw()).data) };
        if buffer.is_null() || data.is_null() {
            return None;
        }
        Some(RowsAt {
            buffer: buffer as usize,
            data: data as usize,
            width: width as usize,
            rows: rows as usize,
            device: self.cuda_device(),
        })
    }
}

impl RowsAt {
    /// `n` of its rows from row `first` on, where they are.
    pub fn slice(&self, first: usize, n: usize) -> RowsAt {
        assert!(
            first + n <= self.rows,
            "rows {first}+{n} of {} rows",
            self.rows
        );
        RowsAt {
            data: self.data + first * self.width * std::mem::size_of::<f32>(),
            rows: n,
            ..*self
        }
    }

    /// Its bytes.
    pub fn nbytes(&self) -> usize {
        self.width * self.rows * std::mem::size_of::<f32>()
    }
}

impl Backend {
    /// Queues a copy of the rows `src` names into the rows `dst` names (the same size) on this
    /// backend's stream, after the work queued there (#1183): device to device on CUDA, and
    /// the host waits for nothing. Where no CUDA stream queues it, ggml copies it (waiting for
    /// the backend first). The bytes are the same either way.
    ///
    /// # Safety
    /// Both buffers must hold those rows until the copy has run (the backend's next
    /// synchronize), and nothing may write `src` or read or write `dst` meanwhile, but the
    /// work queued on this backend after it.
    pub unsafe fn copy_rows(&self, src: &RowsAt, dst: &RowsAt) {
        assert_eq!(
            (src.width, src.rows),
            (dst.width, dst.rows),
            "copy_rows: shapes differ"
        );
        if src.rows == 0 {
            return;
        }
        #[cfg(feature = "cuda")]
        if let Some(stream) = self.cuda_launch_stream() {
            // SAFETY: two ranges of `nbytes` in live buffers (the caller's contract), copied in
            // order on the backend's stream.
            let e = unsafe {
                cuda::cudaMemcpyAsync(
                    dst.data as *mut std::ffi::c_void,
                    src.data as *const std::ffi::c_void,
                    src.nbytes(),
                    cuda::DEFAULT,
                    stream,
                )
            };
            assert_eq!(e, 0, "copy_rows: cudaMemcpyAsync failed (CUDA error {e})");
            return;
        }
        let ctx = Context::new(2, 0);
        // SAFETY: the caller keeps both buffers and their rows (see above).
        let (s, d) = unsafe { (ctx.over_rows(src), ctx.over_rows(dst)) };
        self.copy_to(self, s, d);
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::os::raw::{c_int, c_void};

    unsafe extern "C" {
        pub(super) fn cudaMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            n: usize,
            kind: c_int,
            s: *mut c_void,
        ) -> c_int;
    }

    /// `cudaMemcpyDefault`: the direction from the pointers (unified addressing).
    pub(super) const DEFAULT: c_int = 4;
}

impl Context {
    /// A tensor of this context over the bytes `at` names (F32 `[width, rows]`): a graph reads
    /// it as a leaf with its memory already set, which the compute allocator leaves alone.
    ///
    /// # Safety
    /// The buffer `at` was taken from must still be allocated and hold those rows for as long
    /// as any graph using the tensor runs, and nothing may write them meanwhile.
    pub unsafe fn over_rows(&self, at: &RowsAt) -> Tensor<'_> {
        let t = self.new_tensor(
            ffi::ggml_type_GGML_TYPE_F32,
            &[at.width as i64, at.rows as i64],
        );
        // SAFETY: a fresh tensor without memory; the caller keeps the buffer and its bytes
        // alive, and `at` came from a tensor of exactly this size there.
        unsafe {
            let status = ffi::ggml_backend_tensor_alloc(
                at.buffer as ffi::ggml_backend_buffer_t,
                t.raw(),
                at.data as *mut std::ffi::c_void,
            );
            assert_eq!(
                status,
                ffi::ggml_status_GGML_STATUS_SUCCESS,
                "over_rows: ggml_backend_tensor_alloc failed"
            );
        }
        t
    }
}
