//! Gemma 4's vision tower on a CUDA GPU (#1217): [`crate::vision::Encoder`]'s `gemma4v` graph
//! on the kernels of `src/cuda/vision.cu`, as [`super::QwenGpu`] runs Qwen3-VL's.
//!
//! The weights are uploaded once from the CPU tower's, matrices F16: `q`, `k` and `v` stacked
//! into one matrix, the gate and up matrices into another, so each is one GEMM. An image then
//! runs on the encoder's stream: patches (scaled to [-1, 1]) to rows, the column and row tables
//! added; per block an RMS norm, the `qkv` GEMM, the per-head norms and the half-and-half
//! rotation, full attention at scale 1 in chunks of queries, the output GEMM and its post-norm
//! into the residual, an RMS norm, the gate-and-up GEMM, the gated GELU, the down GEMM and its
//! post-norm into the residual; then the 3×3 pool, its standardization and norm, and the
//! projection. Every kernel is sm_70+ (WMMA 16×16×16 and plain CUDA), on capped grids like
//! Qwen3-VL's, so the language model's kernels find room beside them ([`crate::vision::share`]).
//!
//! The CPU tower is the oracle: the rows are close to its rows, not the same bits.

use std::os::raw::{c_int, c_void};

use super::{
    ATTN_BUDGET, CC_MAJOR, D2H, DevBuf, Gemm, H2D, Kernel, NON_BLOCKING, OUT_F16, Op, Pinned, TILE,
    cudaDeviceGetAttribute, cudaFreeAsync, cudaMallocAsync, cudaMemcpyAsync, cudaMemsetAsync,
    cudaSetDevice, cudaStreamCreateWithFlags, cudaStreamDestroy, cudaStreamSynchronize, gemm_on,
    launch, ok, sm_count, values, vector,
};
use crate::vision::gemma4v::ROPE_THETA;
use crate::vision::qwen3vl::{Encoder, EncoderConfig, ImageEmbedding};
use crate::vision::share::{Caps, Turns};
use crate::vision::{Preprocessed, VisionError, VisionHparams};

/// A bias-free matrix as F16 rows on the device: `n` rows of `k`.
struct Mat {
    w: DevBuf,
    n: usize,
    k: usize,
}

impl Mat {
    /// The tensors `names` (each `[k, n_i]`) stacked row-wise into one matrix.
    fn stacked(enc: &Encoder, names: &[String]) -> Result<Self, VisionError> {
        let mut rows = Vec::new();
        let (mut n, mut k) = (0, 0);
        for name in names {
            let t = enc.weight(name)?;
            let shape = t.shape();
            if k != 0 && shape[0] as usize != k {
                return Err(VisionError::Unsupported(format!("`{name}`: width")));
            }
            k = shape[0] as usize;
            n += shape[1] as usize;
            rows.extend(values(t)?);
        }
        Ok(Self {
            w: DevBuf::f16s(&rows)?,
            n,
            k,
        })
    }

    fn one(enc: &Encoder, name: &str) -> Result<Self, VisionError> {
        Self::stacked(enc, &[name.to_string()])
    }
}

struct Block {
    ln1: DevBuf,
    ln2: DevBuf,
    attn_post: DevBuf,
    ffn_post: DevBuf,
    q_norm: DevBuf,
    k_norm: DevBuf,
    qkv: Mat,
    out: Mat,
    gate_up: Mat,
    down: Mat,
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

/// Gemma 4's tower on one GPU.
pub struct GemmaGpu {
    pub(super) cfg: EncoderConfig,
    pub(super) hp: VisionHparams,
    pub(super) device: i32,
    stream: *mut c_void,
    patch: Mat,
    /// The column table's first `pos_rows` rows, then the row table's.
    pos: DevBuf,
    pos_rows: usize,
    blocks: Vec<Block>,
    std_bias: DevBuf,
    std_scale: DevBuf,
    proj: Mat,
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
unsafe impl Send for GemmaGpu {}

impl GemmaGpu {
    pub(super) fn new(enc: &Encoder, device: usize) -> Result<Option<Self>, VisionError> {
        let c = enc.config().clone();
        let d = c.d_head();
        let pp3 = 3 * c.patch_size * c.patch_size;
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
        // Tensor cores (sm_70+), rows of a multiple of 8 halves, heads of at most 128 (the
        // q / k / v kernel's shared rows), halves of whole pairs.
        if major < 7
            || !c.n_embd.is_multiple_of(8)
            || !d.is_multiple_of(8)
            || d > 128
            || !pp3.is_multiple_of(8)
            || !n_ff.is_multiple_of(8)
        {
            return Ok(None);
        }
        let mut blocks = Vec::with_capacity(c.n_layer);
        for il in 0..c.n_layer {
            let t = |s: &str| format!("v.blk.{il}.{s}.weight");
            let v = |s: &str| -> Result<DevBuf, VisionError> { vector(enc.weight(&t(s))?) };
            blocks.push(Block {
                ln1: v("ln1")?,
                ln2: v("ln2")?,
                attn_post: v("attn_post_norm")?,
                ffn_post: v("ffn_post_norm")?,
                q_norm: v("attn_q_norm")?,
                k_norm: v("attn_k_norm")?,
                qkv: Mat::stacked(enc, &[t("attn_q"), t("attn_k"), t("attn_v")])?,
                out: Mat::one(enc, &t("attn_out"))?,
                gate_up: Mat::stacked(enc, &[t("ffn_gate"), t("ffn_up")])?,
                down: Mat::one(enc, &t("ffn_down"))?,
            });
        }
        // The tables' rows a grid within the pixel budget can reach (a 48-pixel-wide image
        // that tall), not all of them.
        let hp = enc.hparams().clone();
        let align = (hp.patch_size * hp.spatial_merge_size) as usize;
        let max_side = hp.image_max_pixels as usize / align / c.patch_size;
        let pos_rows = c.pos_side.min(max_side.max(1));
        let all = values(enc.weight("v.position_embd.weight")?)?;
        let e = c.n_embd;
        let mut pos = Vec::with_capacity(2 * pos_rows * e);
        for table in 0..2 {
            pos.extend_from_slice(&all[table * c.pos_side * e..][..pos_rows * e]);
        }
        drop(all);
        let max_px = hp.image_max_pixels as usize;
        let max_patches = max_px / (c.patch_size * c.patch_size);
        let mut pin_in = Pinned(std::ptr::null_mut(), 0);
        let mut pin_out = Pinned(std::ptr::null_mut(), 0);
        pin_in.reserve((3 * max_px + max_side * d / 2) * 4)?;
        pin_out.reserve(max_patches / (c.merge * c.merge) * c.n_out * 4)?;
        let mut stream = std::ptr::null_mut();
        // SAFETY: a stream of the current device, owned by the encoder.
        unsafe {
            ok(
                cudaStreamCreateWithFlags(&mut stream, NON_BLOCKING),
                "cudaStreamCreateWithFlags",
            )?
        };
        Ok(Some(Self {
            patch: Mat::one(enc, "v.patch_embd.weight").map(|m| Mat {
                // The kernel [p, p, 3, e]: e rows of 3 p p.
                n: e,
                k: pp3,
                w: m.w,
            })?,
            pos: DevBuf::f32s(&pos)?,
            pos_rows,
            blocks,
            std_bias: vector(enc.weight("v.std_bias")?)?,
            std_scale: vector(enc.weight("v.std_scale")?)?,
            proj: Mat::one(enc, "mm.input_projection.weight")?,
            n_ff,
            cfg: c,
            hp,
            device: device as i32,
            stream,
            pin_in,
            pin_out,
            turns: Turns::new(sms),
            caps: Caps::default(),
        }))
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

    /// `y = x W^T` of `w` over `m` rows of F16 `x` (row stride `w.k`), F32 or F16 (`flags`).
    fn linear(
        &self,
        w: &Mat,
        x: *const c_void,
        m: usize,
        y: *mut c_void,
        flags: i32,
    ) -> Result<(), VisionError> {
        self.gemm(
            Gemm {
                a: x,
                b: w.w.0,
                c: y,
                bias: std::ptr::null(),
                res: std::ptr::null(),
                lda: w.k as i64,
                ldb: w.k as i64,
                ldc: w.n as i64,
                ldr: 0,
                sa: 0,
                sb: 0,
                sc: 0,
                m: m as i32,
                n: w.n as i32,
                k: w.k as i32,
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

    /// The scratch of an image of `n` patches, `n_pix` pixel values and a rotation table of
    /// `n_cs` values.
    fn scratch(&self, n: usize, n_pix: usize, n_cs: usize) -> Scratch {
        let c = &self.cfg;
        let (e, heads) = (c.n_embd, c.n_head);
        let d = e / heads;
        let pp3 = 3 * c.patch_size * c.patch_size;
        let kpad = n.next_multiple_of(8);
        let n_tok = n / (c.merge * c.merge);
        let per = ATTN_BUDGET / (heads * kpad * 6);
        let chunk = if per >= TILE {
            per / TILE * TILE
        } else {
            per.max(1)
        }
        .min(n.next_multiple_of(TILE));
        let mut at = 0usize;
        let mut take = |bytes: usize| {
            let o = at;
            at += bytes.next_multiple_of(256);
            o
        };
        let pix = take(n_pix * 4);
        let cs = take(n_cs * 4);
        let x = take(n * e * 4);
        let h = take(n * e * 2);
        let big = take(
            (n * pp3 * 2)
                .max(n * 3 * e * 4)
                .max(n * 2 * self.n_ff * 2)
                .max(n_tok * e * 4),
        );
        let ff = take(n * self.n_ff * 2);
        let q = take(heads * n * d * 2);
        let k = take(heads * n * d * 2);
        let vt = take(heads * d * kpad * 2);
        let s = take(heads * chunk * kpad * 4);
        let p = take(heads * chunk * kpad * 2);
        let out = take(n_tok * c.n_out * 4);
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

    pub(super) fn scratch_bytes(&self) -> usize {
        let c = &self.cfg;
        let px = self.hp.image_max_pixels as usize;
        let n = px / (c.patch_size * c.patch_size);
        self.scratch(n, 3 * px, self.pos_rows * c.d_head() / 2)
            .total
    }

    pub(super) fn encode(&mut self, p: &Preprocessed) -> Result<ImageEmbedding, VisionError> {
        let c = self.cfg.clone();
        let (gx, gy) = (p.grid_w as usize, p.grid_h as usize);
        let kp = c.merge;
        if gx % kp != 0 || gy % kp != 0 || gx == 0 || gy == 0 || gx.max(gy) > self.pos_rows {
            return Err(VisionError::Dimensions);
        }
        let (w_px, h_px) = (p.width as usize, p.height as usize);
        if w_px != gx * c.patch_size
            || h_px != gy * c.patch_size
            || p.pixels.len() != 3 * w_px * h_px
        {
            return Err(VisionError::Dimensions);
        }
        let n = gx * gy;
        let e = c.n_embd;
        let heads = c.n_head;
        let d = e / heads;
        let q4 = d / 4;
        let kpad = n.next_multiple_of(8);
        let n_tok = n / (kp * kp);
        let n_out = c.n_out;

        // Rotation table: each half of a head turns pair j by pos * base^(-2 j / (d / 2)), the
        // angle multiplied up pair by pair as ggml's CPU rope cache does.
        let side = gx.max(gy);
        let theta_scale = ROPE_THETA.powf(-2.0 / (d / 2) as f32);
        let mut cs = Vec::with_capacity(side * q4 * 2);
        for pos in 0..side {
            let mut theta = pos as f32;
            for _ in 0..q4 {
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
        self.caps = self.turns.next_caps();
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
                side: self.pos_rows as i32,
                p: c.patch_size as i32,
                w_px: w_px as i32,
                h_px: h_px as i32,
                kpad: kpad as i32,
                chunk: chunk as i32,
                pool: kp as i32,
                eps: c.eps,
                ..Op::default()
            };

            // Patches, then their products with the kernel; the tables.
            let mut o = Op {
                x: ptr(o_pix).cast(),
                h0: bigp.cast(),
                ..geo
            };
            self.launch(Kernel::G4Im2col, n, 256, &mut o)?;
            self.linear(&self.patch, bigp, n, x.cast(), 0)?;
            let mut o = Op {
                y: x,
                w: self.pos.0.cast(),
                ..geo
            };
            self.launch(Kernel::G4Pos, n, 256, &mut o)?;

            for blk in &self.blocks {
                let mut o = Op {
                    x,
                    w: blk.ln1.0.cast(),
                    h0: h,
                    ..geo
                };
                self.rows(Kernel::Rms, n, &mut o)?;
                self.linear(&blk.qkv, h.cast(), n, bigp, 0)?;
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
                self.launch(Kernel::G4Qkv, n, 256, &mut o)?;
                let mut q0 = 0;
                while q0 < n {
                    let cm = chunk.min(n - q0);
                    // Scores of the chunk's queries, every head (batch), at scale 1.
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
                            alpha: 1.0,
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
                self.linear(&blk.out, h.cast(), n, bigp, 0)?;
                let mut o = Op {
                    x: bigp.cast(),
                    y: x,
                    w: blk.attn_post.0.cast(),
                    ..geo
                };
                self.rows(Kernel::RmsAdd, n, &mut o)?;
                let mut o = Op {
                    x,
                    w: blk.ln2.0.cast(),
                    h0: h,
                    ..geo
                };
                self.rows(Kernel::Rms, n, &mut o)?;
                self.linear(&blk.gate_up, h.cast(), n, bigp, OUT_F16)?;
                let mut o = Op {
                    h1: bigp.cast(),
                    h0: ffp.cast(),
                    e: self.n_ff as i32,
                    ..geo
                };
                self.rows(Kernel::Geglu, n, &mut o)?;
                self.linear(&blk.down, ffp, n, bigp, 0)?;
                let mut o = Op {
                    x: bigp.cast(),
                    y: x,
                    w: blk.ffn_post.0.cast(),
                    ..geo
                };
                self.rows(Kernel::RmsAdd, n, &mut o)?;
            }
            // The pooler, then the projection.
            let mut o = Op {
                x,
                y: bigp.cast(),
                w: self.std_scale.0.cast(),
                b: self.std_bias.0.cast(),
                h0: h,
                ..geo
            };
            self.launch(Kernel::G4Pool, n_tok, 256, &mut o)?;
            self.linear(&self.proj, h.cast(), n_tok, ptr(o_out), 0)?;
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
        Ok(ImageEmbedding {
            nx: gx / kp,
            ny: gy / kp,
            n_embd: n_out,
            data,
        })
    }
}

impl Drop for GemmaGpu {
    fn drop(&mut self) {
        // SAFETY: our own stream, idle after every encode.
        unsafe {
            cudaSetDevice(self.device);
            cudaStreamDestroy(self.stream);
        }
    }
}
