//! Tensor operations beyond the basics in `context.rs`. Thin wrappers: shapes are in ggml order
//! (`ne[0]` contiguous) and strides in bytes, exactly as the ggml C API documents them.

use super::context::{Context, Tensor};
use super::ffi;

/// F32s per 4 KiB, the smallest page size: [`Tensor::f32_range`] writes one per page.
const PAGE_F32: usize = 1024;

impl Context {
    pub fn reshape_2d<'a>(&'a self, a: Tensor<'a>, ne0: i64, ne1: i64) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_reshape_2d(self.raw(), a.raw(), ne0, ne1) })
    }

    pub fn reshape_3d<'a>(&'a self, a: Tensor<'a>, ne0: i64, ne1: i64, ne2: i64) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_reshape_3d(self.raw(), a.raw(), ne0, ne1, ne2) })
    }

    pub fn reshape_4d<'a>(&'a self, a: Tensor<'a>, ne: [i64; 4]) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_reshape_4d(self.raw(), a.raw(), ne[0], ne[1], ne[2], ne[3]) })
    }

    /// A strided view. `nb` are the strides of dims 1.. in bytes; `offset` in bytes.
    pub fn view_2d<'a>(
        &'a self,
        a: Tensor<'a>,
        ne: [i64; 2],
        nb1: usize,
        offset: usize,
    ) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_view_2d(self.raw(), a.raw(), ne[0], ne[1], nb1, offset) })
    }

    pub fn view_3d<'a>(
        &'a self,
        a: Tensor<'a>,
        ne: [i64; 3],
        nb: [usize; 2],
        offset: usize,
    ) -> Tensor<'a> {
        self.wrap(unsafe {
            ffi::ggml_view_3d(
                self.raw(),
                a.raw(),
                ne[0],
                ne[1],
                ne[2],
                nb[0],
                nb[1],
                offset,
            )
        })
    }

    pub fn view_4d<'a>(
        &'a self,
        a: Tensor<'a>,
        ne: [i64; 4],
        nb: [usize; 3],
        offset: usize,
    ) -> Tensor<'a> {
        self.wrap(unsafe {
            ffi::ggml_view_4d(
                self.raw(),
                a.raw(),
                ne[0],
                ne[1],
                ne[2],
                ne[3],
                nb[0],
                nb[1],
                nb[2],
                offset,
            )
        })
    }

    pub fn cont<'a>(&'a self, a: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_cont(self.raw(), a.raw()) })
    }

    pub fn cont_2d<'a>(&'a self, a: Tensor<'a>, ne0: i64, ne1: i64) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_cont_2d(self.raw(), a.raw(), ne0, ne1) })
    }

    /// Axis `i` of `a` becomes axis `axes[i]` of the result.
    pub fn permute<'a>(&'a self, a: Tensor<'a>, axes: [i32; 4]) -> Tensor<'a> {
        self.wrap(unsafe {
            ffi::ggml_permute(self.raw(), a.raw(), axes[0], axes[1], axes[2], axes[3])
        })
    }

    pub fn transpose<'a>(&'a self, a: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_transpose(self.raw(), a.raw()) })
    }

    pub fn concat<'a>(&'a self, a: Tensor<'a>, b: Tensor<'a>, dim: i32) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_concat(self.raw(), a.raw(), b.raw(), dim) })
    }

    /// Converts to another type (e.g. F32 -> F16, as a KV cache stores it).
    pub fn cast<'a>(&'a self, a: Tensor<'a>, ty: ffi::ggml_type) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_cast(self.raw(), a.raw(), ty) })
    }

    /// [`Self::cast`] for matmul inputs: every call for the same `a` and `ty` in this context
    /// returns the same node, so the matmuls that read one activation share one conversion.
    /// Views get a cast of their own each time (their data may be another node's, written in
    /// place), unless [`Self::set_shared_cast`] gave them one, or they are a reshape of a node
    /// with a shared cast: that cast reshaped (a reshape is its source's values in its order).
    pub fn cast_shared<'a>(&'a self, a: Tensor<'a>, ty: ffi::ggml_type) -> Tensor<'a> {
        if let Some(c) = self.shared_cast(a, ty) {
            return c;
        }
        // SAFETY: a live node of this context.
        let t = unsafe { &*a.raw() };
        if !t.view_src.is_null() {
            if t.op == ffi::ggml_op_GGML_OP_RESHAPE
                && !t.src[0].is_null()
                && let Some(c) = self.shared_cast(self.wrap(t.src[0]), ty)
            {
                return self.reshape_4d(c, a.ne());
            }
            return self.cast(a, ty);
        }
        let c = self.cast(a, ty);
        self.shared_casts.borrow_mut().push((
            std::ptr::NonNull::new(a.raw()).expect("a tensor"),
            ty,
            std::ptr::NonNull::new(c.raw()).expect("a tensor"),
        ));
        c
    }

    /// Makes `c` what [`Self::cast_shared`] returns for `a` and `ty` from now on: an op that
    /// writes `ty` directly instead of `a` and a cast. `c` must hold exactly `cast(a, ty)`'s
    /// values (the caller's promise); `a` is still computed if anything else reads it.
    pub fn set_shared_cast<'a>(&'a self, a: Tensor<'a>, ty: ffi::ggml_type, c: Tensor<'a>) {
        assert!(
            c.ty() == ty && c.ne() == a.ne() && c.is_contiguous(),
            "set_shared_cast: {:?} {:?} is not a contiguous cast of {:?} to {ty}",
            c.ty(),
            c.ne(),
            a.ne()
        );
        let src = std::ptr::NonNull::new(a.raw()).expect("a tensor");
        let mut casts = self.shared_casts.borrow_mut();
        casts.retain(|&(s, t, _)| !(s == src && t == ty));
        casts.push((src, ty, std::ptr::NonNull::new(c.raw()).expect("a tensor")));
    }

    /// The node [`Self::cast_shared`] made for `a` and `ty` in this context, if any.
    pub fn shared_cast<'a>(&'a self, a: Tensor<'a>, ty: ffi::ggml_type) -> Option<Tensor<'a>> {
        let src = a.raw();
        let casts = self.shared_casts.borrow();
        let &(_, _, c) = casts
            .iter()
            .find(|&&(s, t, _)| s.as_ptr() == src && t == ty)?;
        Some(self.wrap(c.as_ptr()))
    }

    /// [`Self::cast_shared`] for other conversions of a matmul input that depend on `a` alone
    /// (e.g. [`crate::model::fp8`]'s per-token FP8 quantization): the node `make` builds,
    /// made once per `a` and `key` in this context and returned again for later calls. Views
    /// get a node of their own each time, as in `cast_shared`.
    pub fn shared_node<'a>(
        &'a self,
        a: Tensor<'a>,
        key: &'static str,
        make: impl FnOnce() -> Tensor<'a>,
    ) -> Tensor<'a> {
        let src = a.raw();
        if let Some(&(_, _, n)) = self
            .shared_nodes
            .borrow()
            .iter()
            .find(|&&(s, k, _)| s.as_ptr() == src && k == key)
        {
            return self.wrap(n.as_ptr());
        }
        let n = make();
        // SAFETY: a live node of this context.
        if unsafe { (*src).view_src }.is_null() {
            self.shared_nodes.borrow_mut().push((
                std::ptr::NonNull::new(src).expect("a tensor"),
                key,
                std::ptr::NonNull::new(n.raw()).expect("a tensor"),
            ));
        }
        n
    }

    /// Whether [`Self::shared_node`] has a node for `a` and `key` already (made or set).
    pub fn has_shared_node(&self, a: Tensor<'_>, key: &'static str) -> bool {
        let src = a.raw();
        self.shared_nodes
            .borrow()
            .iter()
            .any(|&(s, k, _)| s.as_ptr() == src && k == key)
    }

    /// Makes `n` what [`Self::shared_node`] returns for `a` and `key` from now on, views
    /// included (as [`Self::set_shared_cast`]): an op that computes the conversion from `a`'s
    /// own inputs, so `a` itself is computed only if anything else reads it. `n` must hold
    /// exactly what the `make` of every later call would build (the caller's promise).
    pub fn set_shared_node<'a>(&'a self, a: Tensor<'a>, key: &'static str, n: Tensor<'a>) {
        let src = std::ptr::NonNull::new(a.raw()).expect("a tensor");
        let mut nodes = self.shared_nodes.borrow_mut();
        nodes.retain(|&(s, k, _)| !(s == src && k == key));
        nodes.push((src, key, std::ptr::NonNull::new(n.raw()).expect("a tensor")));
    }

    /// Copies `a` into `b` (converting the type, e.g. F32 -> F16) and returns `b`. `b` is
    /// usually a view into a persistent buffer (a KV cache row, a recurrent state); the copy
    /// only happens if the result is expanded into the graph. Elements are copied in logical
    /// order, so `a` and `b` may have different strides.
    pub fn cpy<'a>(&'a self, a: Tensor<'a>, b: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_cpy(self.raw(), a.raw(), b.raw()) })
    }

    /// Appends `p[i]` zeros at the end of dim `i` (F32).
    pub fn pad<'a>(&'a self, a: Tensor<'a>, p: [i32; 4]) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_pad(self.raw(), a.raw(), p[0], p[1], p[2], p[3]) })
    }

    pub fn sigmoid<'a>(&'a self, a: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_sigmoid(self.raw(), a.raw()) })
    }

    pub fn softplus<'a>(&'a self, a: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_softplus(self.raw(), a.raw()) })
    }

    pub fn div<'a>(&'a self, a: Tensor<'a>, b: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_div(self.raw(), a.raw(), b.raw()) })
    }

    pub fn sum_rows<'a>(&'a self, a: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_sum_rows(self.raw(), a.raw()) })
    }

    pub fn clamp<'a>(&'a self, a: Tensor<'a>, min: f32, max: f32) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_clamp(self.raw(), a.raw(), min, max) })
    }

    pub fn soft_max<'a>(&'a self, a: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_soft_max(self.raw(), a.raw()) })
    }

    /// `softmax(a * scale + mask)`, mask broadcast over dims 2..; no ALiBi when `max_bias` is 0.
    pub fn soft_max_ext<'a>(
        &'a self,
        a: Tensor<'a>,
        mask: Tensor<'a>,
        scale: f32,
        max_bias: f32,
    ) -> Tensor<'a> {
        self.wrap(unsafe {
            ffi::ggml_soft_max_ext(self.raw(), a.raw(), mask.raw(), scale, max_bias)
        })
    }

    /// Fused attention `softmax(q·kᵀ · scale + mask) · v`, accumulated in F32: `q` `[d, n_tok,
    /// n_head, n_seqs]`, `k` and `v` `[d, n_kv, n_head_kv, n_seqs]` (F16), `mask` F16
    /// `[n_kv, n_tok, 1, n_seqs]`. Returns `[d_v, n_head, n_tok, n_seqs]`.
    pub fn flash_attn_ext<'a>(
        &'a self,
        q: Tensor<'a>,
        k: Tensor<'a>,
        v: Tensor<'a>,
        mask: Tensor<'a>,
        scale: f32,
    ) -> Tensor<'a> {
        // SAFETY: tensors of this context (the lifetime ties them to it); ggml checks shapes.
        let t = self.wrap(unsafe {
            ffi::ggml_flash_attn_ext(
                self.raw(),
                q.raw(),
                k.raw(),
                v.raw(),
                mask.raw(),
                scale,
                0.0,
                0.0,
            )
        });
        // SAFETY: `t` is the flash_attn_ext op just made; this sets one of its op params.
        unsafe { ffi::ggml_flash_attn_ext_set_prec(t.raw(), ffi::ggml_prec_GGML_PREC_F32) };
        t
    }

    /// Multi-section RoPE (`GGML_ROPE_TYPE_MROPE`/`IMROPE`). `pos` is I32 with
    /// `4 × n_tokens` entries, one block of positions per section.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_multi<'a>(
        &'a self,
        a: Tensor<'a>,
        pos: Tensor<'a>,
        n_dims: i32,
        mut sections: [i32; 4],
        mode: i32,
        n_ctx_orig: i32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> Tensor<'a> {
        self.wrap(unsafe {
            ffi::ggml_rope_multi(
                self.raw(),
                a.raw(),
                pos.raw(),
                std::ptr::null_mut(),
                n_dims,
                sections.as_mut_ptr(),
                mode,
                n_ctx_orig,
                freq_base,
                freq_scale,
                ext_factor,
                attn_factor,
                beta_fast,
                beta_slow,
            )
        })
    }

    /// Depthwise causal conv: `sx` is `[d_conv - 1 + n_tokens, channels, n_seqs]`, `c` is
    /// `[d_conv, channels]`; result `[channels, n_tokens, n_seqs]`.
    pub fn ssm_conv<'a>(&'a self, sx: Tensor<'a>, c: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_ssm_conv(self.raw(), sx.raw(), c.raw()) })
    }

    /// Fused gated delta rule over a whole chunk of tokens. Output holds the per-token outputs
    /// followed by `k` state snapshots (the last one is the final state).
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_net<'a>(
        &'a self,
        q: Tensor<'a>,
        k: Tensor<'a>,
        v: Tensor<'a>,
        g: Tensor<'a>,
        beta: Tensor<'a>,
        state: Tensor<'a>,
        snapshots: i64,
    ) -> Tensor<'a> {
        self.wrap(unsafe {
            ffi::ggml_gated_delta_net(
                self.raw(),
                q.raw(),
                k.raw(),
                v.raw(),
                g.raw(),
                beta.raw(),
                state.raw(),
                snapshots,
            )
        })
    }

    /// The least precise accumulator the matmul `a` may use (`ggml_prec_set_acc`), e.g.
    /// [`ffi::ggml_prec_GGML_PREC_BF16`]. Only backends with such kernels (CUDA) look at it.
    pub fn prec_set_acc(&self, a: Tensor<'_>, prec: ffi::ggml_prec) {
        unsafe { ffi::ggml_prec_set_acc(a.raw(), prec) };
    }

    /// The lowest precision the matmul `a` may quantize its source `idx` to
    /// (`ggml_prec_set_src`): [`ffi::ggml_prec_GGML_PREC_Q8`] keeps CUDA's FP4 matmuls from
    /// quantizing activations to 4 bits. The CPU backend always uses Q8_0 activations.
    pub fn prec_set_src(&self, a: Tensor<'_>, prec: ffi::ggml_prec, idx: i32) {
        unsafe { ffi::ggml_prec_set_src(a.raw(), prec, idx) };
    }

    /// Per-row expert matmul: `as` is `[k, m, n_experts]`, `b` is `[k, n_used, n_tokens]`,
    /// `ids` is I32 `[n_used, n_tokens]`; result `[m, n_used, n_tokens]`.
    pub fn mul_mat_id<'a>(
        &'a self,
        experts: Tensor<'a>,
        b: Tensor<'a>,
        ids: Tensor<'a>,
    ) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_mul_mat_id(self.raw(), experts.raw(), b.raw(), ids.raw()) })
    }

    /// Writes the rows of `src` (F32 or F16 `[n, m]`) into rows `ids[i]` of `dst` (`[n, rows]`,
    /// converting to its type) and returns `dst`. Like [`Self::cpy`], it only happens if the
    /// result is expanded into the graph; the destinations are data (`ids`, I32), not part of
    /// the graph's shape, so the same graph can write different rows every run.
    pub fn set_rows<'a>(&'a self, dst: Tensor<'a>, src: Tensor<'a>, ids: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_set_rows(self.raw(), dst.raw(), src.raw(), ids.raw()) })
    }

    /// The bytes of the contiguous tensor `a` seen as another plain (non-block) type and shape,
    /// e.g. an F16 KV cache as I32 rows of the same byte length, so that [`Self::get_rows`]
    /// (which converts every float type to F32) copies them bit for bit. A view: no data moves.
    pub fn reinterpret<'a>(
        &'a self,
        a: Tensor<'a>,
        ty: ffi::ggml_type,
        ne: [i64; 4],
    ) -> Tensor<'a> {
        assert!(a.is_contiguous(), "reinterpret needs a contiguous tensor");
        let es = unsafe { ffi::ggml_type_size(ty) };
        assert_eq!(
            unsafe { ffi::ggml_blck_size(ty) },
            1,
            "reinterpret to a block type"
        );
        let bytes = ne.iter().product::<i64>() as usize * es;
        assert_eq!(bytes, a.nbytes(), "reinterpret changes the size");
        let v = unsafe { ffi::ggml_view_1d(self.raw(), a.raw(), a.elements() as i64, 0) };
        // SAFETY: `v` is a fresh view node of this context that nothing else references yet;
        // the new type and shape cover exactly the same bytes, contiguously.
        unsafe {
            (*v).type_ = ty;
            (*v).ne = ne;
            (*v).nb[0] = es;
            for i in 1..4 {
                (*v).nb[i] = (*v).nb[i - 1] * ne[i - 1] as usize;
            }
        }
        self.wrap(v)
    }

    /// F32 `[start, start + step, ...)` up to `stop` (exclusive).
    pub fn arange(&self, start: f32, stop: f32, step: f32) -> Tensor<'_> {
        self.wrap(unsafe { ffi::ggml_arange(self.raw(), start, stop, step) })
    }

    /// Index (I32) of the largest value of each row: `[n, rows]` -> `[rows]`.
    pub fn argmax<'a>(&'a self, a: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_argmax(self.raw(), a.raw()) })
    }

    /// Indices (I32) of the `k` largest values of each row.
    pub fn argsort_top_k<'a>(&'a self, a: Tensor<'a>, k: i32) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_argsort_top_k(self.raw(), a.raw(), k) })
    }
}

impl<'a> Tensor<'a> {
    /// All four dims (`ne`), untrimmed.
    pub fn ne(self) -> [i64; 4] {
        unsafe { (*self.raw()).ne }
    }

    /// All four strides in bytes (`nb`).
    pub fn nb(self) -> [usize; 4] {
        unsafe { (*self.raw()).nb }
    }

    pub fn element_size(self) -> usize {
        unsafe { ffi::ggml_element_size(self.raw()) }
    }

    pub fn is_contiguous(self) -> bool {
        unsafe { ffi::ggml_is_contiguous(self.raw()) }
    }

    /// For I32 tensors.
    pub fn to_i32(self) -> Vec<i32> {
        assert_eq!(
            self.ty(),
            ffi::ggml_type_GGML_TYPE_I32,
            "to_i32 on a non-I32 tensor"
        );
        self.bytes()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_ne_bytes(*c))
            .collect()
    }

    /// Gives this not-yet-allocated tensor the memory of `base` starting `offset` bytes in, so
    /// a graph op writes its result there directly. The compute allocator then leaves it alone.
    ///
    /// # Safety
    /// The bytes are shared with `base` (and whatever else lives there): the caller must make
    /// sure the op may write them, including when they are also one of its inputs.
    pub unsafe fn place_in(self, base: Tensor<'_>, offset: usize) {
        assert!(
            offset
                .checked_add(self.nbytes())
                .is_some_and(|end| end <= base.nbytes()),
            "place_in: {} bytes at {offset} do not fit in `{}`",
            self.nbytes(),
            base.name()
        );
        unsafe {
            assert!(
                (*self.raw()).buffer.is_null() && (*self.raw()).data.is_null(),
                "place_in: tensor already has memory"
            );
            let buffer = (*base.raw()).buffer;
            assert!(!buffer.is_null(), "place_in: base has no memory");
            let addr = (*base.raw()).data.cast::<u8>().add(offset);
            let status = ffi::ggml_backend_tensor_alloc(buffer, self.raw(), addr.cast());
            assert_eq!(
                status,
                ffi::ggml_status_GGML_STATUS_SUCCESS,
                "place_in failed"
            );
        }
    }

    /// Gives this not-yet-allocated tensor the memory of `parts`, tensors that lie back to back
    /// in one buffer (each starts where the one before ends) and together are exactly this
    /// tensor's size: one tensor reads them all (#256: two weight matrices as one matmul).
    /// Returns false, and leaves the tensor alone, where they do not.
    ///
    /// # Safety
    /// The bytes stay `parts`' too: the caller must not let an op write them through this
    /// tensor unless it may write them through `parts`. `parts` must be live tensors.
    pub unsafe fn place_over(self, parts: &[Tensor<'_>]) -> bool {
        let Some(first) = parts.first() else {
            return false;
        };
        unsafe {
            assert!(
                (*self.raw()).buffer.is_null() && (*self.raw()).data.is_null(),
                "place_over: tensor already has memory"
            );
            let buffer = (*first.raw()).buffer;
            // A meta buffer's tensors are slices on several devices, not bytes at `data`.
            if buffer.is_null() || ffi::ggml_backend_buffer_is_meta(buffer) {
                return false;
            }
            let start = (*first.raw()).data as usize;
            let mut end = start;
            for p in parts {
                if (*p.raw()).buffer != buffer || (*p.raw()).data as usize != end {
                    return false;
                }
                end += p.nbytes();
            }
            if end - start != self.nbytes() {
                return false;
            }
            let status =
                ffi::ggml_backend_tensor_alloc(buffer, self.raw(), (*first.raw()).data.cast());
            status == ffi::ggml_status_GGML_STATUS_SUCCESS
        }
    }

    /// `n` F32 values starting at element `start`, copied out (host or device).
    pub fn f32_range(self, start: usize, n: usize) -> Vec<f32> {
        assert_eq!(
            self.ty(),
            ffi::ggml_type_GGML_TYPE_F32,
            "f32_range on a non-F32 tensor"
        );
        assert!((start + n) * 4 <= self.nbytes(), "f32_range out of bounds");
        let mut out = Vec::<f32>::with_capacity(n);
        // The pages are mapped by the CPU before the device writes them (#1176). A new vector's
        // memory is often fresh (malloc grows a thread's arena without touching it, and calloc
        // skips zeroing such memory), and on a GB10 (ATS) a device copy into unmapped pages
        // faults page by page: a 1 MB logits row then took 3-20 ms instead of ~20 µs.
        // SAFETY: the writes stay within the capacity; the read fills all `n` values (any bytes
        // are valid f32s) before the length is set.
        unsafe {
            let p = out.as_mut_ptr();
            for i in (0..n).step_by(PAGE_F32).chain(n.checked_sub(1)) {
                p.add(i).write_volatile(0.0);
            }
            ffi::ggml_backend_tensor_get(self.raw(), p.cast(), start * 4, n * 4);
            out.set_len(n);
        }
        out
    }

    /// Copies `src`'s contents (same shape and type) into this tensor, across backends and
    /// devices (a peer copy between GPUs).
    pub fn copy_from(self, src: Tensor<'_>) {
        assert_eq!(self.nbytes(), src.nbytes(), "copy_from: sizes differ");
        unsafe { ffi::ggml_backend_tensor_copy(src.raw(), self.raw()) }
    }

    /// `n` I32 values starting at element `start`, copied out (host or device).
    pub fn i32_range(self, start: usize, n: usize) -> Vec<i32> {
        assert_eq!(
            self.ty(),
            ffi::ggml_type_GGML_TYPE_I32,
            "i32_range on a non-I32 tensor"
        );
        assert!((start + n) * 4 <= self.nbytes(), "i32_range out of bounds");
        let mut out = vec![0i32; n];
        unsafe {
            ffi::ggml_backend_tensor_get(self.raw(), out.as_mut_ptr().cast(), start * 4, n * 4)
        };
        out
    }

    /// Sets `len` bytes starting `offset` bytes in to `value` (host or device memory).
    pub fn memset(self, offset: usize, len: usize, value: u8) {
        assert!(
            offset
                .checked_add(len)
                .is_some_and(|end| end <= self.nbytes()),
            "tensor `{}`: memset range out of bounds",
            self.name()
        );
        unsafe { ffi::ggml_backend_tensor_memset(self.raw(), value, offset, len) }
    }

    /// Mark as a graph input: the compute allocator gives it memory before any intermediate, so
    /// it can be filled after [`Context::alloc_graph`] and before compute. That memory is only
    /// its own until its last consumer ran: ggml frees an input then like any other tensor (only
    /// outputs are never freed), and a later node may be placed over it. Fill it again before
    /// every run of the graph, or mark it [`Self::as_output`] too to keep a value set once
    /// (#836: a bench that filled its input once read a stale token 0 on every later run).
    pub fn as_input(self) -> Self {
        unsafe { ffi::ggml_set_input(self.raw()) };
        self
    }

    /// Keep this tensor's memory intact until the graph finishes (graph outputs).
    pub fn as_output(self) -> Self {
        unsafe { ffi::ggml_set_output(self.raw()) };
        self
    }
}
