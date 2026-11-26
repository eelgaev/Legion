//! GLM-5.3-Flash's vision tower on a CUDA GPU (#1219): [`crate::vision::Encoder`]'s `glm5v`
//! graph on the kernels of `src/cuda/vision.cu`, as [`super::QwenGpu`] runs Qwen3-VL's.
//!
//! The weights are uploaded once from the CPU tower's, matrices F16: the two temporal patch
//! kernels added into one (its rows padded with zeros to a multiple of 8 values, as the patches'
//! rows are), each block's gate and up matrices (and biases) stacked into one, so each is one
//! GEMM, as are the projector's; the 2×2 convolution as a matrix over a block's four patches
//! side by side. An image then runs on the encoder's stream: patches (merge order) to rows with
//! the bias; per block an RMS norm, the `qkv` GEMM with its bias, the per-head q / k norms and
//! Qwen3-VL's rotation, full attention at `1 / sqrt(d)` in chunks of queries, the output GEMM
//! with its bias into the residual, an RMS norm, the gate-and-up GEMM, the clamped SwiGLU, the
//! down GEMM with its bias into the residual; then the final RMS norm, the convolution GEMM,
//! the projector's linear, LayerNorm and exact GELU, and its clamped SwiGLU MLP. Every kernel is
//! sm_70+ (WMMA 16×16×16 and plain CUDA), on capped grids like Qwen3-VL's
//! ([`crate::vision::share`]).
//!
//! The CPU tower is the oracle: the rows are close to its rows, not the same bits.

use std::os::raw::{c_int, c_void};

use super::{
    ATTN_BUDGET, CC_MAJOR, D2H, DevBuf, Gemm, H2D, Kernel, NON_BLOCKING, OUT_F16, Op, Pinned, TILE,
    cudaDeviceGetAttribute, cudaFreeAsync, cudaMallocAsync, cudaMemcpyAsync, cudaMemsetAsync,
    cudaSetDevice, cudaStreamCreateWithFlags, cudaStreamDestroy, cudaStreamSynchronize, gemm_on,
    launch, ok, sm_count, values, vector,
};
use crate::cpu::gpu_exact::f16_to_f32;
use crate::vision::glm5v::{MERGER_EPS, ROPE_THETA};
use crate::vision::qwen3vl::{Encoder, EncoderConfig, ImageEmbedding};
use crate::vision::share::{Caps, Turns};
use crate::vision::{Preprocessed, VisionError, VisionHparams};

/// A matrix as F16 rows on the device (`n` rows of `k`) and its F32 bias, if it has one.
struct Lin {
    w: DevBuf,
    b: Option<DevBuf>,
    n: usize,
    k: usize,
}

impl Lin {
    /// The linears `names` (`name.weight`, each `[k, n_i]`, and `name.bias` when there; all or
    /// none have one) stacked row-wise into one.
    fn stacked(enc: &Encoder, names: &[&str]) -> Result<Self, VisionError> {
        let (mut rows, mut bias) = (Vec::new(), Vec::new());
        let (mut n, mut k, mut biased) = (0, 0, None);
        for name in names {
            let t = enc.weight(&format!("{name}.weight"))?;
            let shape = t.shape();
            if k != 0 && shape[0] as usize != k {
                return Err(VisionError::Unsupported(format!("`{name}`: width")));
            }
            k = shape[0] as usize;
            n += shape[1] as usize;
            rows.extend(values(t)?);
            let b = enc.weight(&format!("{name}.bias")).ok();
            if biased.is_some_and(|had| had != b.is_some()) {
                return Err(VisionError::Unsupported(format!("`{name}`: a bias")));
            }
            biased = Some(b.is_some());
            if let Some(b) = b {
                bias.extend(values(b)?);
            }
        }
        Ok(Self {
            w: DevBuf::f16s(&rows)?,
            b: if bias.is_empty() {
                None
            } else {
                Some(DevBuf::f32s(&bias)?)
            },
            n,
            k,
        })
    }

    fn one(enc: &Encoder, name: &str) -> Result<Self, VisionError> {
        Self::stacked(enc, &[name])
    }

    fn bias(&self) -> *const f32 {
        self.b
            .as_ref()
            .map_or(std::ptr::null(), |b| b.0.cast_const().cast())
    }
}

struct Block {
    ln1: DevBuf,
    ln2: DevBuf,
    q_norm: DevBuf,
    k_norm: DevBuf,
    qkv: Lin,
    out: Lin,
    gate_up: Lin,
    down: Lin,
}

/// The tower's widths the scratch layout depends on.
#[derive(Clone, Copy, Debug)]
struct Dims {
    e: usize,
    heads: usize,
    /// The patch rows' padded width.
    patch_k: usize,
    n_ff: usize,
    /// The 2×2 convolution's output width.
    merger: usize,
    /// The projector's linear width.
    mw: usize,
    mff: usize,
    n_out: usize,
}

impl Dims {
    /// The queries per attention chunk of `n` patches (`kpad` keys): its scores (F32) and
    /// probabilities (F16) for every head within [`ATTN_BUDGET`], a multiple of [`TILE`] when
    /// that fits.
    fn chunk(&self, n: usize, kpad: usize) -> usize {
        let per = ATTN_BUDGET / (self.heads * kpad * 6).max(1);
        if per >= TILE {
            per / TILE * TILE
        } else {
            per.max(1)
        }
        .min(n.next_multiple_of(TILE))
    }

    /// The scratch of `n` patches, `n_pix` pixel values and a rotation table of `n_cs` values.
    fn scratch(&self, n: usize, n_pix: usize, n_cs: usize) -> Scratch {
        let (e, heads) = (self.e, self.heads);
        let d = e / heads;
        let kpad = n.next_multiple_of(8);
        let chunk = self.chunk(n, kpad);
        let n_tok = n / 4;
        let (mw, mff) = (self.mw, self.mff);
        let mut at = 0usize;
        let mut take = |bytes: usize| {
            let o = at;
            at += bytes.next_multiple_of(256);
            o
        };
        let pix = take(n_pix * 4);
        let cs = take(n_cs * 4);
        let x = take(n * e * 4);
        let h = take((n * e * 2).max(n_tok * mw * 2));
        let big = take(
            (n * self.patch_k * 2)
                .max(n * 3 * e * 4)
                .max(n * 2 * self.n_ff * 2)
                .max(n_tok * mw * 4)
                .max(n_tok * 2 * mff * 2),
        );
        let ff = take(
            (n * self.n_ff * 2)
                .max(n_tok * self.merger * 2)
                .max(n_tok * mff * 2),
        );
        let q = take(heads * n * d * 2);
        let k = take(heads * n * d * 2);
        let vt = take(heads * d * kpad * 2);
        let s = take(heads * chunk * kpad * 4);
        let p = take(heads * chunk * kpad * 2);
        let out = take(n_tok * self.n_out * 4);
        Scratch {
            pix,
            cs,
            x,
            h,
            big,
            ff,
            q,
            k,
            vt,
            s,
            p,
            out,
            total: at,
            chunk,
        }
    }

    /// The most scratch an image of at most `max_px` pixels (patches of `patch` pixels, heads
    /// of `d_head`) can need: the largest over every grid up to that many patches, not just
    /// the largest grid's. Every region but the attention's grows with the patch count, but
    /// the chunk does not: a smaller grid's chunk can round up to more queries, so its
    /// `chunk × kpad` can pass the largest grid's. A grid's patches are a multiple of 4 (an
    /// even width and height); the largest count is taken too, with all `max_px` pixels, and
    /// never fewer than one merged patch's 4 (the hparams refuse a budget under that).
    fn scratch_bound(&self, max_px: usize, patch: usize, d_head: usize) -> usize {
        let n_max = (max_px / (patch * patch).max(1)).max(4);
        // A rotation table of at most one row per patch.
        let at = |n: usize, n_pix: usize| self.scratch(n, n_pix, n * d_head / 2).total;
        (4..=n_max)
            .step_by(4)
            .map(|n| at(n, 3 * n * patch * patch))
            .fold(at(n_max, 3 * max_px), usize::max)
    }
}

/// Byte offsets into an image's scratch, its size and the queries per attention chunk.
struct Scratch {
    pix: usize,
    cs: usize,
    x: usize,
    h: usize,
    big: usize,
    ff: usize,
    q: usize,
    k: usize,
    vt: usize,
    s: usize,
    p: usize,
    out: usize,
    total: usize,
    chunk: usize,
}

/// GLM-5.3-Flash's tower on one GPU.
pub struct GlmGpu {
    pub(super) cfg: EncoderConfig,
    pub(super) hp: VisionHparams,
    pub(super) device: i32,
    stream: *mut c_void,
    /// The summed patch kernels, `e` rows of `ld` (`3 p p` padded to a multiple of 8).
    patch: Lin,
    blocks: Vec<Block>,
    post_ln: DevBuf,
    /// The 2×2 convolution: rows of `4 e`, a block's four patches (its merge order) side by side.
    merger: Lin,
    fc: Lin,
    post_norm_w: DevBuf,
    post_norm_b: DevBuf,
    mm_gate_up: Lin,
    mm_down: Lin,
    n_ff: usize,
    pin_in: Pinned,
    pin_out: Pinned,
    /// The shares of the GPU to encode with, in turn ([`crate::vision::share`]).
    pub(super) turns: Turns,
    /// The share of the encode under way.
    caps: Caps,
}

// SAFETY: as `QwenGpu`: device / pinned allocations and a stream owned by the encoder, used by
// one thread at a time.
unsafe impl Send for GlmGpu {}

impl GlmGpu {
    pub(super) fn new(enc: &Encoder, device: usize) -> Result<Option<Self>, VisionError> {
        let c = enc.config().clone();
        let (e, d) = (c.n_embd, c.d_head());
        let pp3 = 3 * c.patch_size * c.patch_size;
        let ld = pp3.next_multiple_of(8);
        let mut major = 0;
        // SAFETY: plain device queries and the calling thread's current device.
        unsafe {
            ok(cudaSetDevice(device as c_int), "cudaSetDevice")?;
            ok(
                cudaDeviceGetAttribute(&mut major, CC_MAJOR, device as c_int),
                "cudaDeviceGetAttribute",
            )?;
        }
        let sms = sm_count(device)?;
        let n_ff = enc.weight("v.blk.0.ffn_gate.weight")?.shape()[1] as usize;
        let width = |name: &str| -> Result<usize, VisionError> {
            Ok(enc.weight(&format!("{name}.weight"))?.shape()[0] as usize)
        };
        let (mw, mlp, mff) = (width("mm.model.fc")?, width("mm.gate")?, width("mm.down")?);
        // Tensor cores (sm_70+), rows of a multiple of 8 halves (every GEMM's input), heads of
        // at most 128 (the q / k kernel's shared rows) and of whole quarters.
        if major < 7
            || !e.is_multiple_of(8)
            || !d.is_multiple_of(8)
            || d > 128
            || !n_ff.is_multiple_of(8)
            || !mw.is_multiple_of(8)
            || !mlp.is_multiple_of(8)
            || !mff.is_multiple_of(8)
            || c.merge != 2
        {
            return Ok(None);
        }
        let mut blocks = Vec::with_capacity(c.n_layer);
        for il in 0..c.n_layer {
            let t = |s: &str| format!("v.blk.{il}.{s}");
            let v = |s: &str| -> Result<DevBuf, VisionError> {
                vector(enc.weight(&format!("{}.weight", t(s)))?)
            };
            blocks.push(Block {
                ln1: v("ln1")?,
                ln2: v("ln2")?,
                q_norm: v("attn_q_norm")?,
                k_norm: v("attn_k_norm")?,
                qkv: Lin::one(enc, &t("attn_qkv"))?,
                out: Lin::one(enc, &t("attn_out"))?,
                gate_up: Lin::stacked(enc, &[&t("ffn_gate"), &t("ffn_up")])?,
                down: Lin::one(enc, &t("ffn_down"))?,
            });
        }
        // The patch kernels [p, p, 3, e]: e rows of 3 p p, summed, padded to ld.
        let w0 = values(enc.weight("v.patch_embd.weight")?)?;
        let w1 = values(enc.weight("v.patch_embd.weight.1")?)?;
        if w0.len() != e * pp3 || w1.len() != e * pp3 {
            return Err(VisionError::Unsupported(format!(
                "`v.patch_embd.weight`: {} and {} values, not {e} x {pp3}",
                w0.len(),
                w1.len()
            )));
        }
        let mut patch = vec![0f32; e * ld];
        for o in 0..e {
            for k in 0..pp3 {
                patch[o * ld + k] = w0[o * pp3 + k] + w1[o * pp3 + k];
            }
        }
        // The convolution [kw, kh, e, o] as rows [o][(kh * 2 + kw) * e + c]: a block's patch
        // (dy, dx) is its (kh, kw), its rows side by side in merge order.
        let conv = values(enc.weight("mm.patch_merger.weight")?)?;
        if conv.is_empty() || !conv.len().is_multiple_of(4 * e) {
            return Err(VisionError::Unsupported(format!(
                "`mm.patch_merger.weight`: {} values, not rows of 4 x {e}",
                conv.len()
            )));
        }
        let n_mw = conv.len() / (4 * e);
        let mut merger = vec![0f32; conv.len()];
        for o in 0..n_mw {
            for ch in 0..e {
                for q in 0..4 {
                    merger[o * 4 * e + q * e + ch] = conv[(o * e + ch) * 4 + q];
                }
            }
        }
        drop((w0, w1, conv));
        let hp = enc.hparams().clone();
        let max_px = hp.image_max_pixels as usize;
        let max_patches = max_px / (c.patch_size * c.patch_size);
        let mut pin_in = Pinned(std::ptr::null_mut(), 0);
        let mut pin_out = Pinned(std::ptr::null_mut(), 0);
        pin_in.reserve((3 * max_px + max_patches * d / 2) * 4)?;
        pin_out.reserve(max_patches / 4 * c.n_out * 4)?;
        let mut stream = std::ptr::null_mut();
        // SAFETY: a stream of the current device, owned by the encoder.
        unsafe {
            ok(
                cudaStreamCreateWithFlags(&mut stream, NON_BLOCKING),
                "cudaStreamCreateWithFlags",
            )?
        };
        let g = Self {
            patch: Lin {
                w: DevBuf::f16s(&patch)?,
                b: Some(vector(enc.weight("v.patch_embd.bias")?)?),
                n: e,
                k: ld,
            },
            blocks,
            post_ln: vector(enc.weight("v.post_ln.weight")?)?,
            merger: Lin {
                w: DevBuf::f16s(&merger)?,
                b: Some(vector(enc.weight("mm.patch_merger.bias")?)?),
                n: n_mw,
                k: 4 * e,
            },
            fc: Lin::one(enc, "mm.model.fc")?,
            post_norm_w: vector(enc.weight("mm.post_norm.weight")?)?,
            post_norm_b: vector(enc.weight("mm.post_norm.bias")?)?,
            mm_gate_up: Lin::stacked(enc, &["mm.gate", "mm.up"])?,
            mm_down: Lin::one(enc, "mm.down")?,
            n_ff,
            cfg: c,
            hp,
            device: device as i32,
            stream,
            pin_in,
            pin_out,
            turns: Turns::new(sms),
            caps: Caps::default(),
        };
        g.check_shapes()?;
        Ok(Some(g))
    }

    /// Every matrix's width is the width of what feeds it (an mmproj's shapes are not checked
    /// on load), before any encode indexes by them.
    fn check_shapes(&self) -> Result<(), VisionError> {
        let (e, ff) = (self.cfg.n_embd, self.n_ff);
        let mut want: Vec<(String, &Lin, usize, usize)> = Vec::new();
        for (il, b) in self.blocks.iter().enumerate() {
            let t = |s: &str| format!("v.blk.{il}.{s}");
            want.push((t("attn_qkv"), &b.qkv, 3 * e, e));
            want.push((t("attn_out"), &b.out, e, e));
            want.push((t("ffn_gate / ffn_up"), &b.gate_up, 2 * ff, e));
            want.push((t("ffn_down"), &b.down, e, ff));
        }
        let (mw, mff) = (self.merger.n, self.mm_down.k);
        want.push(("mm.model.fc".into(), &self.fc, self.fc.n, mw));
        want.push((
            "mm.gate / mm.up".into(),
            &self.mm_gate_up,
            2 * mff,
            self.fc.n,
        ));
        want.push(("mm.down".into(), &self.mm_down, self.cfg.n_out, mff));
        for (name, lin, n, k) in want {
            if lin.n != n || lin.k != k {
                return Err(VisionError::Unsupported(format!(
                    "`{name}`: {} x {}, not {n} x {k}",
                    lin.n, lin.k
                )));
            }
        }
        Ok(())
    }

    /// Kernel `k` over `items` work items, on the encoder's stream under the encode's caps.
    fn launch<T>(
        &self,
        k: Kernel,
        items: usize,
        block: u32,
        args: &mut T,
    ) -> Result<(), VisionError> {
        launch(self.stream, self.caps, k, items, block, args)
    }

    /// The GEMM kernel over `batch` products, as [`Self::launch`].
    fn gemm(&self, g: Gemm, batch: usize) -> Result<(), VisionError> {
        gemm_on(self.stream, self.caps, g, batch)
    }

    /// `y = x W^T + b (+ res)` of `lin` over `m` rows of F16 `x` (row stride `lin.k`), F32 or
    /// F16 (`flags`).
    fn linear(
        &self,
        lin: &Lin,
        x: *const c_void,
        m: usize,
        y: *mut c_void,
        res: *const f32,
        flags: i32,
    ) -> Result<(), VisionError> {
        self.gemm(
            Gemm {
                a: x,
                b: lin.w.0,
                c: y,
                bias: lin.bias(),
                res,
                lda: lin.k as i64,
                ldb: lin.k as i64,
                ldc: lin.n as i64,
                ldr: lin.n as i64,
                sa: 0,
                sb: 0,
                sc: 0,
                m: m as i32,
                n: lin.n as i32,
                k: lin.k as i32,
                alpha: 1.0,
                flags,
                batch: 1,
            },
            1,
        )
    }

    /// A warp-per-row kernel over `rows` rows: a block per 8 rows.
    fn rows(&self, k: Kernel, rows: usize, o: &mut Op) -> Result<(), VisionError> {
        self.launch(k, rows.div_ceil(8), 256, o)
    }

    /// What the scratch layout depends on.
    fn dims(&self) -> Dims {
        Dims {
            e: self.cfg.n_embd,
            heads: self.cfg.n_head,
            patch_k: self.patch.k,
            n_ff: self.n_ff,
            merger: self.merger.n,
            mw: self.fc.n,
            mff: self.mm_down.k,
            n_out: self.cfg.n_out,
        }
    }

    /// The scratch of an image of `n` patches, `n_pix` pixel values and a rotation table of
    /// `n_cs` values.
    fn scratch(&self, n: usize, n_pix: usize, n_cs: usize) -> Scratch {
        self.dims().scratch(n, n_pix, n_cs)
    }

    /// The scratch any image the preprocessing gives can need (up to `image_max_pixels`).
    pub(super) fn scratch_bytes(&self) -> usize {
        let c = &self.cfg;
        self.dims()
            .scratch_bound(self.hp.image_max_pixels as usize, c.patch_size, c.d_head())
    }

    /// The tower's output for `p`; `alone`: nothing else runs on the GPU meanwhile.
    pub(super) fn encode(
        &mut self,
        p: &Preprocessed,
        alone: bool,
    ) -> Result<ImageEmbedding, VisionError> {
        self.encode_tapped(p, alone, None)
    }

    /// `count` values (F16 or F32) at `src` on the device, once the stream gets there.
    fn read_back(
        &self,
        src: *const c_void,
        count: usize,
        half: bool,
    ) -> Result<Vec<f32>, VisionError> {
        let size = if half { 2 } else { 4 };
        let mut bytes = vec![0u8; count * size];
        // SAFETY: `count` values at `src` in the scratch, into a host buffer of that size; the
        // stream is drained before the buffer is read.
        unsafe {
            ok(
                cudaMemcpyAsync(
                    bytes.as_mut_ptr().cast(),
                    src,
                    bytes.len(),
                    D2H,
                    self.stream,
                ),
                "cudaMemcpyAsync",
            )?;
            ok(cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")?;
        }
        Ok(if half {
            bytes
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        } else {
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        })
    }

    /// [`Self::encode`], and with `taps` each stage's rows as the CPU tower names them
    /// ([`crate::vision::glm5v`]'s `encode_tapped`), for tests: the stream is drained after
    /// each one.
    pub(super) fn encode_tapped(
        &mut self,
        p: &Preprocessed,
        alone: bool,
        mut taps: Option<&mut Vec<(String, Vec<f32>)>>,
    ) -> Result<ImageEmbedding, VisionError> {
        let c = self.cfg.clone();
        let (gx, gy) = (p.grid_w as usize, p.grid_h as usize);
        if gx % 2 != 0 || gy % 2 != 0 || gx == 0 || gy == 0 {
            return Err(VisionError::Dimensions);
        }
        let (w_px, h_px) = (p.width as usize, p.height as usize);
        // Within the pixel budget too, which the scratch is reserved for (`scratch_bytes`).
        if w_px != gx * c.patch_size
            || h_px != gy * c.patch_size
            || p.pixels.len() != 3 * w_px * h_px
            || w_px * h_px > self.hp.image_max_pixels as usize
        {
            return Err(VisionError::Dimensions);
        }
        let n = gx * gy;
        let e = c.n_embd;
        let heads = c.n_head;
        let d = e / heads;
        let quarter = d / 4;
        let kpad = n.next_multiple_of(8);
        let n_tok = n / 4;
        let n_out = c.n_out;

        // Rotation table: ggml's `ggml_mrope_cache_init` for GGML_ROPE_TYPE_VISION, each
        // section's theta restarting at the patch's row (column) and multiplied down per pair.
        let side = gx.max(gy);
        let theta_scale = ROPE_THETA.powf(-2.0 / (d / 2) as f32);
        let mut cs = Vec::with_capacity(side * quarter * 2);
        for pos in 0..side {
            let mut theta = pos as f32;
            for _ in 0..quarter {
                cs.push(theta.cos());
                cs.push(theta.sin());
                theta *= theta_scale;
            }
        }

        let Scratch {
            pix: o_pix,
            cs: o_cs,
            x: o_x,
            h: o_h,
            big: o_big,
            ff: o_ff,
            q: o_q,
            k: o_k,
            vt: o_vt,
            s: o_s,
            p: o_p,
            out: o_out,
            total,
            chunk,
        } = self.scratch(n, p.pixels.len(), cs.len());

        let in_bytes = (p.pixels.len() + cs.len()) * 4;
        let out_bytes = n_tok * n_out * 4;
        // SAFETY: the calling thread's current device.
        unsafe { ok(cudaSetDevice(self.device), "cudaSetDevice")? };
        self.caps = self.turns.next_caps(alone);
        let pin_in = self.pin_in.reserve(in_bytes)?;
        let pin_out = self.pin_out.reserve(out_bytes)?;
        // SAFETY: the pinned buffers hold at least these many bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(p.pixels.as_ptr(), pin_in.cast::<f32>(), p.pixels.len());
            std::ptr::copy_nonoverlapping(
                cs.as_ptr(),
                pin_in.cast::<f32>().add(p.pixels.len()),
                cs.len(),
            );
        }

        let st = self.stream;
        let mut base = std::ptr::null_mut::<c_void>();
        // SAFETY: the encoder's device and stream; `base` is freed on the stream below.
        unsafe { ok(cudaMallocAsync(&mut base, total, st), "cudaMallocAsync")? };
        let ptr = |o: usize| -> *mut c_void {
            // SAFETY: offsets within the allocation.
            unsafe { base.cast::<u8>().add(o).cast() }
        };
        let result = (|| -> Result<(), VisionError> {
            // SAFETY: copies into the scratch from the pinned buffers, on the stream.
            unsafe {
                let px_bytes = p.pixels.len() * 4;
                ok(
                    cudaMemcpyAsync(ptr(o_pix), pin_in, px_bytes, H2D, st),
                    "cudaMemcpyAsync",
                )?;
                ok(
                    cudaMemcpyAsync(
                        ptr(o_cs),
                        pin_in.cast::<u8>().add(px_bytes).cast(),
                        in_bytes - px_bytes,
                        H2D,
                        st,
                    ),
                    "cudaMemcpyAsync",
                )?;
                ok(
                    cudaMemsetAsync(ptr(o_vt), 0, heads * d * kpad * 2, st),
                    "cudaMemsetAsync",
                )?;
            }
            let x = ptr(o_x).cast::<f32>();
            let h = ptr(o_h).cast::<u16>();
            let bigp = ptr(o_big);
            let ffp = ptr(o_ff);
            let (q, k, vt) = (
                ptr(o_q).cast::<u16>(),
                ptr(o_k).cast::<u16>(),
                ptr(o_vt).cast::<u16>(),
            );
            let geo = Op {
                n: n as i32,
                e: e as i32,
                heads: heads as i32,
                d: d as i32,
                gx: gx as i32,
                gy: gy as i32,
                p: c.patch_size as i32,
                w_px: w_px as i32,
                h_px: h_px as i32,
                kpad: kpad as i32,
                chunk: chunk as i32,
                eps: c.eps,
                lim: c.swiglu_limit,
                ..Op::default()
            };
            let null = std::ptr::null::<f32>();
            let mut tap = |name: String, src: *mut c_void, count: usize, half: bool| match taps
                .as_deref_mut()
            {
                Some(t) => self.read_back(src, count, half).map(|v| t.push((name, v))),
                None => Ok(()),
            };

            // Patches (merge order, rows of ld), then their products with the summed kernels
            // and the bias.
            let mut o = Op {
                x: ptr(o_pix).cast(),
                h0: bigp.cast(),
                ld: self.patch.k as i32,
                ..geo
            };
            self.launch(Kernel::Im2col, n, 256, &mut o)?;
            self.linear(&self.patch, bigp, n, x.cast(), null, 0)?;
            tap("patches".into(), x.cast::<c_void>(), n * e, false)?;

            let scale = 1.0 / (d as f32).sqrt();
            for (il, blk) in self.blocks.iter().enumerate() {
                let mut o = Op {
                    x,
                    w: blk.ln1.0.cast(),
                    h0: h,
                    ..geo
                };
                self.rows(Kernel::Rms, n, &mut o)?;
                self.linear(&blk.qkv, h.cast(), n, bigp, null, 0)?;
                let mut o = Op {
                    x: bigp.cast(),
                    w: blk.q_norm.0.cast(),
                    b: blk.k_norm.0.cast(),
                    cs: ptr(o_cs).cast(),
                    h0: q,
                    h1: k,
                    h2: vt,
                    ..geo
                };
                self.launch(Kernel::GlmQkv, n, 256, &mut o)?;
                let mut q0 = 0;
                while q0 < n {
                    let cm = chunk.min(n - q0);
                    // Scores of the chunk's queries, every head (batch), scaled.
                    self.gemm(
                        Gemm {
                            // SAFETY: within q.
                            a: unsafe { q.add(q0 * d) }.cast(),
                            b: k.cast(),
                            c: ptr(o_s),
                            bias: std::ptr::null(),
                            res: std::ptr::null(),
                            lda: d as i64,
                            ldb: d as i64,
                            ldc: kpad as i64,
                            ldr: 0,
                            sa: (n * d) as i64,
                            sb: (n * d) as i64,
                            sc: (chunk * kpad) as i64,
                            m: cm as i32,
                            n: n as i32,
                            k: d as i32,
                            alpha: scale,
                            flags: 0,
                            batch: 1,
                        },
                        heads,
                    )?;
                    let mut o = Op {
                        x: ptr(o_s).cast(),
                        h0: ptr(o_p).cast(),
                        cm: cm as i32,
                        ..geo
                    };
                    self.launch(Kernel::Softmax, heads * chunk, 256, &mut o)?;
                    // Their values, into the attention output [patch][head * d].
                    self.gemm(
                        Gemm {
                            a: ptr(o_p),
                            b: vt.cast(),
                            // SAFETY: within h.
                            c: unsafe { h.add(q0 * e) }.cast(),
                            bias: std::ptr::null(),
                            res: std::ptr::null(),
                            lda: kpad as i64,
                            ldb: kpad as i64,
                            ldc: e as i64,
                            ldr: 0,
                            sa: (chunk * kpad) as i64,
                            sb: (d * kpad) as i64,
                            sc: d as i64,
                            m: cm as i32,
                            n: d as i32,
                            k: kpad as i32,
                            alpha: 1.0,
                            flags: OUT_F16,
                            batch: 1,
                        },
                        heads,
                    )?;
                    q0 += cm;
                }
                self.linear(&blk.out, h.cast(), n, x.cast(), x, 0)?;
                tap(format!("blk.{il}.attn"), x.cast::<c_void>(), n * e, false)?;
                let mut o = Op {
                    x,
                    w: blk.ln2.0.cast(),
                    h0: h,
                    ..geo
                };
                self.rows(Kernel::Rms, n, &mut o)?;
                self.linear(&blk.gate_up, h.cast(), n, bigp, null, OUT_F16)?;
                let mut o = Op {
                    h1: bigp.cast(),
                    h0: ffp.cast(),
                    e: self.n_ff as i32,
                    ..geo
                };
                self.rows(Kernel::SwigluClamp, n, &mut o)?;
                self.linear(&blk.down, ffp, n, x.cast(), x, 0)?;
                tap(format!("blk.{il}"), x.cast::<c_void>(), n * e, false)?;
            }
            // The merger: the final norm; each block's four normed rows (merge order) are one
            // row of 4 e for the convolution.
            let mut o = Op {
                x,
                w: self.post_ln.0.cast(),
                h0: h,
                ..geo
            };
            self.rows(Kernel::Rms, n, &mut o)?;
            tap("post_ln".into(), h.cast::<c_void>(), n * e, true)?;
            self.linear(&self.merger, h.cast(), n_tok, ffp, null, OUT_F16)?;
            tap("downsample".into(), ffp, n_tok * self.merger.n, true)?;
            self.linear(&self.fc, ffp, n_tok, bigp, null, 0)?;
            tap("fc".into(), bigp, n_tok * self.fc.n, false)?;
            let mut o = Op {
                x: bigp.cast(),
                w: self.post_norm_w.0.cast(),
                b: self.post_norm_b.0.cast(),
                h0: h,
                n: n_tok as i32,
                e: self.fc.n as i32,
                eps: MERGER_EPS,
                ..geo
            };
            self.rows(Kernel::LnGelu, n_tok, &mut o)?;
            tap(
                "ln_gelu".into(),
                h.cast::<c_void>(),
                n_tok * self.fc.n,
                true,
            )?;
            self.linear(&self.mm_gate_up, h.cast(), n_tok, bigp, null, OUT_F16)?;
            let mut o = Op {
                h1: bigp.cast(),
                h0: ffp.cast(),
                n: n_tok as i32,
                e: self.mm_down.k as i32,
                ..geo
            };
            self.rows(Kernel::SwigluClamp, n_tok, &mut o)?;
            self.linear(&self.mm_down, ffp, n_tok, ptr(o_out), null, 0)?;
            // SAFETY: into the pinned output buffer, on the stream.
            unsafe {
                ok(
                    cudaMemcpyAsync(pin_out, ptr(o_out), out_bytes, D2H, st),
                    "cudaMemcpyAsync",
                )
            }
        })();
        // SAFETY: the scratch is freed in stream order; the stream is then drained so the
        // pinned buffers are free to reuse.
        let done = unsafe {
            cudaFreeAsync(base, st);
            cudaStreamSynchronize(st)
        };
        result?;
        ok(done, "the vision tower's stream")?;
        let mut data = vec![0f32; n_tok * n_out];
        // SAFETY: the stream finished writing `out_bytes` into the pinned buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(pin_out.cast::<f32>(), data.as_mut_ptr(), data.len())
        };
        if let Some(t) = taps {
            t.push(("out".into(), data.clone()));
        }
        Ok(ImageEmbedding {
            nx: gx / 2,
            ny: gy / 2,
            n_embd: n_out,
            data,
        })
    }
}

#[cfg(test)]
mod tests;

impl Drop for GlmGpu {
    fn drop(&mut self) {
        // SAFETY: our own stream, idle after every encode.
        unsafe {
            cudaSetDevice(self.device);
            cudaStreamDestroy(self.stream);
        }
    }
}
