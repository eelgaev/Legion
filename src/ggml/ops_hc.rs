//! Ops for qwen4exp: the fused DeepSeek-V4 hyper-connection ops, the element-wise helpers of its
//! per-layer embedding gate, and the indexer ops of its sparse attention. Thin wrappers, as in
//! `ops.rs`.

use super::context::{Context, Tensor};
use super::ffi;

impl Context {
    pub fn abs<'a>(&'a self, a: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_abs(self.raw(), a.raw()) })
    }

    pub fn sgn<'a>(&'a self, a: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_sgn(self.raw(), a.raw()) })
    }

    pub fn sqrt<'a>(&'a self, a: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_sqrt(self.raw(), a.raw()) })
    }

    /// `a` repeated (broadcast) to shape `ne`; every dim of `ne` must be a multiple of `a`'s.
    pub fn repeat_4d<'a>(&'a self, a: Tensor<'a>, ne: [i64; 4]) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_repeat_4d(self.raw(), a.raw(), ne[0], ne[1], ne[2], ne[3]) })
    }

    /// `scale * sum_h x[i, h, t] * sigmoid(gate[i, h, t])`: x and gate `[n_embd, hc, n]`,
    /// result `[n_embd, n]`.
    pub fn hc_pre_gated<'a>(&'a self, x: Tensor<'a>, gate: Tensor<'a>, scale: f32) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_dsv4_hc_pre_gated(self.raw(), x.raw(), gate.raw(), scale) })
    }

    /// `residual[i, h, t] + x[i, t] * post[h, t]` (identity stream mixing): x `[n_embd, n]`,
    /// residual `[n_embd, hc, n]`, post `[hc, n]`.
    pub fn hc_post<'a>(
        &'a self,
        x: Tensor<'a>,
        residual: Tensor<'a>,
        post: Tensor<'a>,
    ) -> Tensor<'a> {
        self.wrap(unsafe {
            ffi::ggml_dsv4_hc_post(
                self.raw(),
                x.raw(),
                residual.raw(),
                post.raw(),
                std::ptr::null_mut(),
            )
        })
    }
}

impl Context {
    pub fn sub<'a>(&'a self, a: Tensor<'a>, b: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_sub(self.raw(), a.raw(), b.raw()) })
    }

    pub fn reshape_1d<'a>(&'a self, a: Tensor<'a>, ne0: i64) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_reshape_1d(self.raw(), a.raw(), ne0) })
    }

    /// A tensor shaped like `a` with every element `c` (`a`'s data is not read).
    pub fn fill<'a>(&'a self, a: Tensor<'a>, c: f32) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_fill(self.raw(), a.raw(), c) })
    }

    /// Running sum along dim 0.
    pub fn cumsum<'a>(&'a self, a: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_cumsum(self.raw(), a.raw()) })
    }

    /// `a * s + b`.
    pub fn scale_bias<'a>(&'a self, a: Tensor<'a>, s: f32, b: f32) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_scale_bias(self.raw(), a.raw(), s, b) })
    }

    /// Indices (I32) of the `k` largest values of every row, in no particular order.
    pub fn top_k<'a>(&'a self, a: Tensor<'a>, k: i32) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_top_k(self.raw(), a.raw(), k) })
    }

    /// DeepSeek's lightning indexer: `score[j, t] = sum_h w[h, t] relu(q[:, h, t] · k[:, j])`
    /// plus `mask[j, t]` (F16). q `[d, heads, n]`, k `[d, 1, n_kv]`, w `[heads, n]`.
    pub fn lightning_indexer<'a>(
        &'a self,
        q: Tensor<'a>,
        k: Tensor<'a>,
        w: Tensor<'a>,
        mask: Tensor<'a>,
    ) -> Tensor<'a> {
        self.wrap(unsafe {
            ffi::ggml_lightning_indexer(self.raw(), q.raw(), k.raw(), w.raw(), mask.raw())
        })
    }
}

impl<'a> Tensor<'a> {
    /// The bytes of an allocated tensor in host memory (a mapped weight, say), borrowed in
    /// place. None for device memory.
    pub fn host_bytes(self) -> Option<&'a [u8]> {
        if !self.is_host() {
            return None;
        }
        // SAFETY: host memory of `nbytes()` bytes owned by the tensor's buffer, which lives as
        // long as the context ('a). Readers only: weights are never written after loading.
        unsafe {
            let data = (*self.raw()).data as *const u8;
            Some(std::slice::from_raw_parts(data, self.nbytes()))
        }
    }
}
