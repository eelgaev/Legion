//! The vision tower on a CUDA GPU (#1195): [`super::Encoder`]'s graph on Legion's own kernels
//! (`src/cuda/vision.cu`, launched from here with `cudaLaunchKernel`), no ggml-cuda compute.
//!
//! The weights are uploaded once from the CPU tower's: matrices as F16 (as ggml's CPU rounds an
//! F16 matrix's activations, every product here is F16 x F16 into F32), the two temporal patch
//! kernels added into one, vectors and the position table F32. An image then runs on a stream
//! of its own, so the language model's steps on other streams go on meanwhile: patches to rows
//! (one GEMM), positions added, each block's LayerNorm, `qkv` GEMM, the 2-D rotation, full
//! attention per head in chunks of queries (scores GEMM, softmax, values GEMM), output GEMM
//! with the residual, LayerNorm and the GELU MLP; then the merger. Its scratch memory is
//! stream-ordered (`cudaMallocAsync`), freed when the image is done.
//!
//! The CPU tower is the oracle: the rows are close to its rows, not the same bits
//! (`LEGION_DEBUG_VISION_CPU=1` runs the CPU tower instead, for A/B runs).

use std::os::raw::{c_int, c_void};

use crate::cpu::gpu_exact::{f16_to_f32, f32_to_f16};
use crate::ggml::{Tensor, types};

use super::qwen3vl::{Encoder, EncoderConfig, ImageEmbedding};
use super::{Preprocessed, VisionError, VisionHparams};

/// Scores and probabilities of one chunk of queries (all heads) take at most this much.
const ATTN_BUDGET: usize = 128 << 20;
/// The GEMM kernel's tile (`BM`, `BN`).
const TILE: usize = 128;

/// `Gemm` of `src/cuda/vision.cu`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Gemm {
    a: *const c_void,
    b: *const c_void,
    c: *mut c_void,
    bias: *const f32,
    res: *const f32,
    lda: i64,
    ldb: i64,
    ldc: i64,
    ldr: i64,
    sa: i64,
    sb: i64,
    sc: i64,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    flags: i32,
}

const OUT_F16: i32 = 1;
const GELU: i32 = 2;

/// `Op` of `src/cuda/vision.cu`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Op {
    x: *const f32,
    w: *const f32,
    b: *const f32,
    cs: *const f32,
    y: *mut f32,
    h0: *mut u16,
    h1: *mut u16,
    h2: *mut u16,
    n: i32,
    e: i32,
    heads: i32,
    d: i32,
    gx: i32,
    gy: i32,
    side: i32,
    p: i32,
    w_px: i32,
    h_px: i32,
    kpad: i32,
    chunk: i32,
    cm: i32,
    eps: f32,
    sfx: f32,
    sfy: f32,
}

impl Default for Op {
    fn default() -> Self {
        use std::ptr::{null, null_mut};
        Self {
            x: null(),
            w: null(),
            b: null(),
            cs: null(),
            y: null_mut(),
            h0: null_mut(),
            h1: null_mut(),
            h2: null_mut(),
            n: 0,
            e: 0,
            heads: 0,
            d: 0,
            gx: 0,
            gy: 0,
            side: 0,
            p: 0,
            w_px: 0,
            h_px: 0,
            kpad: 0,
            chunk: 0,
            cm: 0,
            eps: 0.0,
            sfx: 1.0,
            sfy: 1.0,
        }
    }
}

/// `dim3`.
#[repr(C)]
struct Dim3 {
    x: u32,
    y: u32,
    z: u32,
}

/// The kernels' order in `legion_vision_kernels`.
#[derive(Clone, Copy)]
enum Kernel {
    Gemm,
    Im2col,
    PosAdd,
    Norm,
    Rope,
    Softmax,
}

unsafe extern "C" {
    /// The kernels as the CUDA runtime knows them (their host stubs), in [`Kernel`] order.
    static legion_vision_kernels: [*const c_void; 6];
    fn cudaLaunchKernel(
        func: *const c_void,
        grid: Dim3,
        block: Dim3,
        args: *mut *mut c_void,
        shared: usize,
        stream: *mut c_void,
    ) -> i32;
    fn cudaSetDevice(device: c_int) -> c_int;
    fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
    fn cudaStreamCreateWithFlags(s: *mut *mut c_void, flags: u32) -> c_int;
    fn cudaStreamDestroy(s: *mut c_void) -> c_int;
    fn cudaStreamSynchronize(s: *mut c_void) -> c_int;
    fn cudaMalloc(p: *mut *mut c_void, size: usize) -> c_int;
    fn cudaFree(p: *mut c_void) -> c_int;
    fn cudaMallocAsync(p: *mut *mut c_void, size: usize, s: *mut c_void) -> c_int;
    fn cudaFreeAsync(p: *mut c_void, s: *mut c_void) -> c_int;
    fn cudaMallocHost(p: *mut *mut c_void, size: usize) -> c_int;
    fn cudaFreeHost(p: *mut c_void) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, n: usize, kind: c_int) -> c_int;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        n: usize,
        kind: c_int,
        s: *mut c_void,
    ) -> c_int;
    fn cudaMemsetAsync(p: *mut c_void, v: c_int, n: usize, s: *mut c_void) -> c_int;
}

const H2D: c_int = 1;
const D2H: c_int = 2;
const NON_BLOCKING: u32 = 1;
const CC_MAJOR: c_int = 75;

fn ok(e: c_int, what: &str) -> Result<(), VisionError> {
    if e == 0 {
        Ok(())
    } else {
        Err(VisionError::Compute(format!("{what}: CUDA error {e}")))
    }
}

/// Device memory, freed on drop.
struct DevBuf(*mut c_void);

impl DevBuf {
    fn upload(bytes: &[u8]) -> Result<Self, VisionError> {
        let mut p = std::ptr::null_mut();
        // SAFETY: a fresh allocation of `bytes.len()` bytes, filled from `bytes`.
        unsafe {
            ok(cudaMalloc(&mut p, bytes.len().max(1)), "cudaMalloc")?;
            let buf = Self(p);
            ok(
                cudaMemcpy(p, bytes.as_ptr().cast(), bytes.len(), H2D),
                "cudaMemcpy",
            )?;
            Ok(buf)
        }
    }

    fn f32s(values: &[f32]) -> Result<Self, VisionError> {
        Self::upload(
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    }

    fn f16s(values: &[f32]) -> Result<Self, VisionError> {
        Self::upload(
            &values
                .iter()
                .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                .collect::<Vec<_>>(),
        )
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        // SAFETY: our own allocation; the encoder's stream is synchronized after every image.
        unsafe { cudaFree(self.0) };
    }
}

/// Pinned host memory that grows to the largest request.
struct Pinned(*mut c_void, usize);

impl Pinned {
    fn reserve(&mut self, bytes: usize) -> Result<*mut c_void, VisionError> {
        if bytes > self.1 {
            // SAFETY: our own allocation; nothing in flight reads it (the stream was synchronized).
            unsafe {
                if !self.0.is_null() {
                    cudaFreeHost(self.0);
                }
                self.0 = std::ptr::null_mut();
                self.1 = 0;
                ok(cudaMallocHost(&mut self.0, bytes), "cudaMallocHost")?;
            }
            self.1 = bytes;
        }
        Ok(self.0)
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: our own allocation.
            unsafe { cudaFreeHost(self.0) };
        }
    }
}

/// A tensor's values as F32 (F32, F16 or BF16).
fn values(t: Tensor<'_>) -> Result<Vec<f32>, VisionError> {
    let b = t.bytes();
    match t.ty() {
        types::F32 => Ok(b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        types::F16 => Ok(b
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        types::BF16 => Ok(b
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()),
        _ => Err(VisionError::Unsupported(format!(
            "`{}`: a matrix type the GPU tower does not take",
            t.name()
        ))),
    }
}

/// A matrix as F16 on the device: an F16 tensor's bytes as they are.
fn matrix(t: Tensor<'_>) -> Result<DevBuf, VisionError> {
    if t.ty() == types::F16 {
        DevBuf::upload(&t.bytes())
    } else {
        DevBuf::f16s(&values(t)?)
    }
}

fn vector(t: Tensor<'_>) -> Result<DevBuf, VisionError> {
    DevBuf::f32s(&values(t)?)
}

/// Byte offsets into an image's scratch, its size and the queries per attention chunk.
struct Scratch {
    pix: usize,
    cs: usize,
    x: usize,
    h: usize,
    big: usize,
    q: usize,
    k: usize,
    vt: usize,
    s: usize,
    p: usize,
    out: usize,
    total: usize,
    chunk: usize,
}

struct Linear {
    w: DevBuf,
    b: DevBuf,
    n: usize,
    k: usize,
}

impl Linear {
    fn load(enc: &Encoder, name: &str) -> Result<Self, VisionError> {
        let w = enc.weight(&format!("{name}.weight"))?;
        let shape = w.shape();
        Ok(Self {
            w: matrix(w)?,
            b: vector(enc.weight(&format!("{name}.bias"))?)?,
            n: shape[1] as usize,
            k: shape[0] as usize,
        })
    }
}

struct Norm {
    w: DevBuf,
    b: DevBuf,
}

impl Norm {
    fn load(enc: &Encoder, name: &str) -> Result<Self, VisionError> {
        Ok(Self {
            w: vector(enc.weight(&format!("{name}.weight"))?)?,
            b: vector(enc.weight(&format!("{name}.bias"))?)?,
        })
    }
}

struct Block {
    ln1: Norm,
    qkv: Linear,
    out: Linear,
    ln2: Norm,
    up: Linear,
    down: Linear,
}

/// The tower on one GPU.
pub struct GpuEncoder {
    cfg: EncoderConfig,
    hp: VisionHparams,
    device: i32,
    stream: *mut c_void,
    patch: DevBuf,
    patch_bias: DevBuf,
    pos: DevBuf,
    blocks: Vec<Block>,
    post_ln: Norm,
    mm0: Linear,
    mm2: Linear,
    n_ff: usize,
    pin_in: Pinned,
    pin_out: Pinned,
}

// SAFETY: the raw pointers are device / pinned allocations and a stream owned by the encoder,
// used by one thread at a time (callers hold it behind a Mutex).
unsafe impl Send for GpuEncoder {}

impl GpuEncoder {
    /// The CPU tower `enc`'s weights on CUDA device `device`. `Ok(None)`: the device or the
    /// tower's shapes do not fit the kernels (the CPU tower runs then).
    pub fn new(enc: &Encoder, device: usize) -> Result<Option<Self>, VisionError> {
        let c = enc.config().clone();
        let d_head = c.n_embd / c.n_head;
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
        let n_ff = enc.weight("v.blk.0.ffn_up.weight")?.shape()[1] as usize;
        // The kernels: tensor cores (sm_70+), rows of a multiple of 8 halves.
        if major < 7
            || !c.n_embd.is_multiple_of(8)
            || !d_head.is_multiple_of(8)
            || !pp3.is_multiple_of(8)
            || !n_ff.is_multiple_of(8)
        {
            return Ok(None);
        }
        // Every linear's input width too (the merger's are its own), before any upload.
        let mut linears = vec!["mm.0".to_string(), "mm.2".to_string()];
        for il in 0..c.n_layer {
            for l in ["attn_qkv", "attn_out", "ffn_up", "ffn_down"] {
                linears.push(format!("v.blk.{il}.{l}"));
            }
        }
        for name in &linears {
            if !(enc.weight(&format!("{name}.weight"))?.shape()[0] as usize).is_multiple_of(8) {
                return Ok(None);
            }
        }
        let w0 = values(enc.weight("v.patch_embd.weight")?)?;
        let w1 = values(enc.weight("v.patch_embd.weight.1")?)?;
        let patch: Vec<f32> = w0.iter().zip(&w1).map(|(a, b)| a + b).collect();
        let mut blocks = Vec::with_capacity(c.n_layer);
        for il in 0..c.n_layer {
            let blk = format!("v.blk.{il}");
            blocks.push(Block {
                ln1: Norm::load(enc, &format!("{blk}.ln1"))?,
                qkv: Linear::load(enc, &format!("{blk}.attn_qkv"))?,
                out: Linear::load(enc, &format!("{blk}.attn_out"))?,
                ln2: Norm::load(enc, &format!("{blk}.ln2"))?,
                up: Linear::load(enc, &format!("{blk}.ffn_up"))?,
                down: Linear::load(enc, &format!("{blk}.ffn_down"))?,
            });
        }
        let patch = DevBuf::f16s(&patch)?;
        let patch_bias = vector(enc.weight("v.patch_embd.bias")?)?;
        let pos = vector(enc.weight("v.position_embd.weight")?)?;
        let post_ln = Norm::load(enc, "v.post_ln")?;
        let mm0 = Linear::load(enc, "mm.0")?;
        let mm2 = Linear::load(enc, "mm.2")?;
        // Pinned copies of the largest image's pixels (and rotation table) and rows, now:
        // growing them later (cudaFreeHost) would wait for the whole device, the language
        // model's work too.
        let max_px = enc.hparams().image_max_pixels as usize;
        let max_patches = max_px / (c.patch_size * c.patch_size);
        let mut pin_in = Pinned(std::ptr::null_mut(), 0);
        let mut pin_out = Pinned(std::ptr::null_mut(), 0);
        pin_in.reserve((3 * max_px + max_patches * (c.n_embd / c.n_head) / 2) * 4)?;
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
            hp: enc.hparams().clone(),
            device: device as i32,
            stream,
            patch,
            patch_bias,
            pos,
            blocks,
            post_ln,
            mm0,
            mm2,
            n_ff,
            cfg: c,
            pin_in,
            pin_out,
        }))
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.cfg
    }

    pub fn hparams(&self) -> &VisionHparams {
        &self.hp
    }

    pub fn device(&self) -> usize {
        self.device as usize
    }

    fn launch<T>(
        &self,
        k: Kernel,
        grid: (usize, usize, usize),
        block: u32,
        args: &mut T,
    ) -> Result<(), VisionError> {
        if grid.0 == 0 || grid.1 == 0 || grid.2 == 0 {
            return Ok(());
        }
        let mut argv = [(args as *mut T).cast::<c_void>()];
        // SAFETY: a kernel of the table taking `T` by value (the launch copies it); its
        // pointers are device memory of the encoder's device.
        let e = unsafe {
            cudaLaunchKernel(
                legion_vision_kernels[k as usize],
                Dim3 {
                    x: grid.0 as u32,
                    y: grid.1 as u32,
                    z: grid.2 as u32,
                },
                Dim3 {
                    x: block,
                    y: 1,
                    z: 1,
                },
                argv.as_mut_ptr(),
                0,
                self.stream,
            )
        };
        ok(e, "vision kernel launch")
    }

    fn gemm(&self, mut g: Gemm, batch: usize) -> Result<(), VisionError> {
        debug_assert!(g.k % 8 == 0 && g.lda % 8 == 0 && g.ldb % 8 == 0);
        let grid = (
            (g.n as usize).div_ceil(TILE),
            (g.m as usize).div_ceil(TILE),
            batch,
        );
        self.launch(Kernel::Gemm, grid, 256, &mut g)
    }

    /// `y = x W^T + b` of `lin` over `m` rows of F16 `x`, then GELU (`flags`) or `+ res`.
    fn linear(
        &self,
        lin: &Linear,
        x: *const u16,
        m: usize,
        y: *mut c_void,
        res: *const f32,
        flags: i32,
    ) -> Result<(), VisionError> {
        self.gemm(
            Gemm {
                a: x.cast(),
                b: lin.w.0,
                c: y,
                bias: lin.b.0.cast(),
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
            },
            1,
        )
    }

    fn norm(&self, n: &Norm, x: *const f32, rows: usize, out: *mut u16) -> Result<(), VisionError> {
        let mut o = Op {
            x,
            w: n.w.0.cast(),
            b: n.b.0.cast(),
            h0: out,
            n: rows as i32,
            e: self.cfg.n_embd as i32,
            eps: self.cfg.eps,
            ..Op::default()
        };
        self.launch(Kernel::Norm, (rows.div_ceil(8), 1, 1), 256, &mut o)
    }

    /// The scratch of an image of `n` patches, `n_pix` pixel values and a rotation table of
    /// `n_cs` values.
    fn scratch(&self, n: usize, n_pix: usize, n_cs: usize) -> Scratch {
        let c = &self.cfg;
        let (e, heads) = (c.n_embd, c.n_head);
        let d = e / heads;
        let pp3 = 3 * c.patch_size * c.patch_size;
        let kpad = n.next_multiple_of(8);
        let n_tok = n / 4;
        // Whole tiles of queries where the budget takes one, fewer queries where it does not
        // (the GEMM takes partial tiles).
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
                .max(n * self.n_ff * 2)
                .max(n_tok * self.mm0.n * 2),
        );
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

    /// Device memory the largest image's scratch takes (`image_max_pixels`), beside the
    /// weights: for the language model's budgets.
    pub fn scratch_bytes(&self) -> usize {
        let c = &self.cfg;
        let px = self.hp.image_max_pixels as usize;
        let n = px / (c.patch_size * c.patch_size);
        // A rotation table of at most one row per patch.
        self.scratch(n, 3 * px, n * (c.n_embd / c.n_head) / 2).total
    }

    /// The tower's output for `p`, as [`Encoder::encode`] gives it.
    pub fn encode(&mut self, p: &Preprocessed) -> Result<ImageEmbedding, VisionError> {
        let c = self.cfg.clone();
        let (gx, gy) = (p.grid_w as usize, p.grid_h as usize);
        if gx % c.merge != 0 || gy % c.merge != 0 || gx == 0 || gy == 0 || c.merge != 2 {
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
        let quarter = d / 4;
        let pp3 = 3 * c.patch_size * c.patch_size;
        let kpad = n.next_multiple_of(8);
        let n_tok = n / 4;
        let n_out = c.n_out;

        // Rotation table: ggml's `ggml_mrope_cache_init` for GGML_ROPE_TYPE_VISION, each
        // section's theta restarting at the patch's row (column) and multiplied down per pair.
        let side = gx.max(gy);
        let theta_scale = 10000f32.powf(-2.0 / (d / 2) as f32);
        let mut cs = Vec::with_capacity(side * quarter * 2);
        for pos in 0..side {
            let mut theta = pos as f32;
            for _ in 0..quarter {
                cs.push(theta.cos());
                cs.push(theta.sin());
                theta *= theta_scale;
            }
        }

        // Scratch, one stream-ordered allocation.
        let Scratch {
            pix: o_pix,
            cs: o_cs,
            x: o_x,
            h: o_h,
            big: o_big,
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
            // SAFETY: copies into the scratch from the pinned buffers, on the stream; the
            // pixels and the rotation table each to their own (aligned) offset.
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
                side: c.pos_side as i32,
                p: c.patch_size as i32,
                w_px: w_px as i32,
                h_px: h_px as i32,
                kpad: kpad as i32,
                chunk: chunk as i32,
                ..Op::default()
            };

            // Patches, then their products with the (summed) temporal kernels.
            let mut o = Op {
                x: ptr(o_pix).cast(),
                h0: bigp.cast(),
                ..geo
            };
            self.launch(Kernel::Im2col, (n, 1, 1), 256, &mut o)?;
            self.gemm(
                Gemm {
                    a: bigp,
                    b: self.patch.0,
                    c: x.cast(),
                    bias: std::ptr::null(),
                    res: std::ptr::null(),
                    lda: pp3 as i64,
                    ldb: pp3 as i64,
                    ldc: e as i64,
                    ldr: 0,
                    sa: 0,
                    sb: 0,
                    sc: 0,
                    m: n as i32,
                    n: e as i32,
                    k: pp3 as i32,
                    alpha: 1.0,
                    flags: 0,
                },
                1,
            )?;
            // ggml's align-corners scale factors, as its CPU computes them.
            let sf = |out: usize, from: usize| {
                if out > 1 && from > 1 {
                    (out - 1) as f32 / (from - 1) as f32
                } else {
                    out as f32 / from as f32
                }
            };
            let mut o = Op {
                y: x,
                w: self.pos.0.cast(),
                b: self.patch_bias.0.cast(),
                sfx: sf(gx, c.pos_side),
                sfy: sf(gy, c.pos_side),
                ..geo
            };
            self.launch(Kernel::PosAdd, (n, 1, 1), 256, &mut o)?;

            let scale = 1.0 / (d as f32).sqrt();
            for blk in &self.blocks {
                self.norm(&blk.ln1, x, n, h)?;
                self.linear(&blk.qkv, h, n, bigp, std::ptr::null(), 0)?;
                let mut o = Op {
                    x: bigp.cast(),
                    cs: ptr(o_cs).cast(),
                    h0: q,
                    h1: k,
                    h2: vt,
                    ..geo
                };
                self.launch(Kernel::Rope, (n, 1, 1), 256, &mut o)?;
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
                        },
                        heads,
                    )?;
                    let mut o = Op {
                        x: ptr(o_s).cast(),
                        h0: ptr(o_p).cast(),
                        cm: cm as i32,
                        ..geo
                    };
                    self.launch(Kernel::Softmax, (heads * chunk, 1, 1), 256, &mut o)?;
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
                        },
                        heads,
                    )?;
                    q0 += cm;
                }
                self.linear(&blk.out, h, n, x.cast(), x, 0)?;
                self.norm(&blk.ln2, x, n, h)?;
                self.linear(&blk.up, h, n, bigp, std::ptr::null(), GELU | OUT_F16)?;
                self.linear(&blk.down, bigp.cast(), n, x.cast(), x, 0)?;
            }
            // The merger: each 2x2 block's four normed rows are one row of 4 e.
            self.norm(&self.post_ln, x, n, h)?;
            self.linear(&self.mm0, h, n_tok, bigp, std::ptr::null(), GELU | OUT_F16)?;
            self.linear(
                &self.mm2,
                bigp.cast(),
                n_tok,
                ptr(o_out),
                std::ptr::null(),
                0,
            )?;
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
            nx: gx / c.merge,
            ny: gy / c.merge,
            n_embd: n_out,
            data,
        })
    }
}

impl Drop for GpuEncoder {
    fn drop(&mut self) {
        // SAFETY: our own stream, idle after every encode.
        unsafe {
            cudaSetDevice(self.device);
            cudaStreamDestroy(self.stream);
        }
    }
}

#[cfg(test)]
mod tests;
