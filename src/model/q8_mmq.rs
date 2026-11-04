//! Q8_0 matmuls of MTP verify steps on a GB10 without ggml-cuda's stream-k fixup (#836):
//! `w @ x` for one Q8_0 weight and [`MIN_TOKENS`] to [`MAX_TOKENS`] F32 columns, as an external
//! op ([`crate::ggml::ExternalOp`]) with ggml's MMQ bits.
//!
//! ggml-cuda runs each of these matmuls as MMQ: a quantization of the input, the MMQ kernel
//! and, whenever its tiles do not fill the SMs' waves (most verify shapes: a 2560-row weight at
//! 48 columns is 20 tiles on 48 SMs), a stream-k fixup kernel that adds up the tiles split over
//! several CTAs. Serving 16 sequences with MTP drafts on Qwen3.8-Flash-Next, a verify step runs
//! ~500 of them (`attn_qkv`, `attn_gate`, `ssm_out`, the attention projections, the shared
//! experts, the hyper-connection mixers' down and up), the fixups ~2.3 ms a round (#836 part 16).
//!
//! The op (`src/cuda/q8_mmq.cu`) runs ggml's quantization and ggml's own MMQ tile code
//! (`mul_mat_q_process_tile`) on ggml's grid ([`plan`]: its tile width, its CTA count from the
//! GPU's SMs, its split of each tile's K blocks over the CTAs), so every CTA computes what
//! ggml's CTA of the same index does. The pieces of a split tile are added up by the CTA that
//! finishes last (a counter per tile), in ggml's fixup order: two launches and ggml's bits.
//! Where that CTA would add many or wide pieces, a fixup launch of its own (ggml's) adds them
//! instead, in the same order ([`Fixup`], `legion serve --q8-mmq-fixup`).
//!
//! The node is ggml's `mul_mat`'s shape and size (`[rows, n]` F32, the input and the weight its
//! sources), so ggml-alloc places the graph's tensors as before (#819: ggml-cuda's fusions
//! depend on that placement). Its scratch (the quantized input, the tile counters, the partial
//! sums) is a buffer per device and stream outside the graphs ([`cuda`]), as ggml's pool is.
//!
//! Taken on compute capability 12.x ([`set_gpu`]: the GB10, whose ggml MMQ config for Q8_0 the
//! kernels copy) inside stage graphs (between ggml's CUDA graphs each op would split one), for
//! the plans it runs faster than ggml's three launches on a DGX Spark ([`beats_ggml`]; on by
//! default since #858, `legion serve --q8-mmq-ggml` keeps ggml's MMQ for all). Consecutive op
//! nodes on the same input quantize it once ([`quant_run`]). The CPU implementation is a
//! reference for the tests (the pieces summed in the same order, not the GPU's bits).

use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

pub mod check;

#[cfg(test)]
mod tests;

static Q8_MMQ: ExternalOp = ExternalOp {
    name: "q8_mmq",
    cpu: cpu_q8_mmq,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::q8_mmq),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Fewest columns the op takes: ggml-cuda runs up to 8 on MMVQ.
pub const MIN_TOKENS: i64 = 9;

/// Most columns the op takes: from 256 the prompt paths (shared quantization, #363) take over.
pub const MAX_TOKENS: i64 = 255;

/// Rows per MMQ tile (`I` of ggml's Q8_0 config).
pub const TILE_ROWS: i64 = 128;

/// Values per Q8_0 block.
const QK: i64 = 32;

/// Bytes per Q8_0 block: an F16 scale, 32 int8.
const BLOCK_BYTES: usize = 34;

/// Q8_0 blocks per MMQ K iteration (`MMQ_ITER_K / QK8_0`): stream-k shares start on them.
const BLOCKS_PER_ITER: i64 = 8;

/// Values the quantized input's rows are padded to (`MATRIX_ROW_PADDING`).
const ROW_PADDING: i64 = 512;

/// Bytes of a `block_q8_1_mmq` (128 values: 4 F32 scales, 128 int8).
const Q8_1_MMQ_BYTES: usize = 144;

/// Values per `block_q8_1_mmq`.
const Q8_1_MMQ_VALUES: i64 = 128;

/// Threads of an MMQ CTA for Q8_0 (ggml's config: 256 at every width).
const MMQ_THREADS: u32 = 256;

/// Most output tiles of one matmul (tile counters in the scratch).
pub const MAX_TILES: i64 = 4096;

/// Most SMs (CTAs of a split matmul) the kernels list a tile's pieces for (`MAX_CTAS`).
pub const MAX_CTAS: i64 = 256;

/// Bytes of the quantized input the scratch holds (with ggml's read-ahead padding).
pub const MAX_QUANT_BYTES: usize = 8 << 20;

/// ggml-cuda's MMQ tile widths for a Q8_0 weight (`mmq-config-ampere.cuh`, which Blackwell
/// takes for Q8_0): rows a multiple of 128, and the `fallback` set for other row counts.
const WIDTHS: [i64; 11] = [8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128];
const WIDTHS_FALLBACK: [i64; 5] = [8, 16, 32, 64, 128];

/// Whether graphs built from now on may take the op ([`set_q8_mmq`]; on by default since #858,
/// for the plans it runs faster, [`beats_ggml`]).
static ON: AtomicBool = AtomicBool::new(true);

/// The engine GPU's SMs where it is one the op runs on (compute capability 12.x), else 0.
static GPU_SMS: AtomicI32 = AtomicI32::new(0);

/// The op's matmul kernel for graphs built from now on ([`set_q8_mmq_fast`]): 0 ggml's tile
/// code, 8 or 4 the streaming kernel with that many warps per CTA.
static FAST: AtomicI32 = AtomicI32::new(0);

/// Values per K stage of the streaming kernel's split (`k` a multiple of it).
const FAST_K: i64 = 256;

/// Sets the op's matmul kernel for graphs built from now on: `warps` 8 or 4 the streaming
/// kernel (`src/cuda/sm80/q8_mmq_fast.cuh`, ggml's bits from a pipelined main loop, #858) with
/// that many warps of 16 rows per CTA, 0 ggml's MMQ tile code. Shapes the streaming kernel does
/// not take ([`fast_fits`]) run ggml's tile code (the same bits).
pub fn set_q8_mmq_fast(warps: i32) {
    FAST.store(
        if matches!(warps, 4 | 8) { warps } else { 0 },
        Ordering::Relaxed,
    );
}

/// The kernel [`set_q8_mmq_fast`] stored.
pub fn q8_mmq_fast() -> i32 {
    FAST.load(Ordering::Relaxed)
}

/// Sets whether graphs built from now on run Q8_0 matmuls of verify-sized steps on the op where
/// it applies ([`takes`]; `true`, the default); `false` (`legion serve --q8-mmq-ggml`) keeps
/// ggml's MMQ for all.
pub fn set_q8_mmq(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// The engine's GPU: the op runs on compute capability 12.x (`cc_major`: GB10) with `sms`
/// streaming multiprocessors (ggml's stream-k split depends on them).
pub fn set_gpu(cc_major: Option<i32>, sms: i32) {
    GPU_SMS.store(
        if cc_major == Some(12) { sms.max(0) } else { 0 },
        Ordering::Relaxed,
    );
}

/// The SMs [`set_gpu`] stored (0: the op is off on this GPU).
pub fn gpu_sms() -> i32 {
    GPU_SMS.load(Ordering::Relaxed)
}

/// ggml-cuda's MMQ tile width for a Q8_0 weight of `rows` rows times `n` columns
/// (`ggml_cuda_mul_mat_q`): the first width, from 8 up, with the fewest column tiles.
pub fn tile_width(rows: i64, n: i64) -> i64 {
    let widths: &[i64] = if rows % TILE_ROWS != 0 {
        &WIDTHS_FALLBACK
    } else {
        &WIDTHS
    };
    let (mut best_j, mut best) = (0, i64::MAX);
    let mut j = 8;
    while j <= 128 && best > 1 {
        if widths.contains(&j) {
            let ntx = (n + j - 1) / j;
            if ntx < best {
                (best_j, best) = (j, ntx);
            }
        }
        j += 8;
    }
    best_j
}

/// How ggml-cuda's MMQ runs a Q8_0 weight's matmul ([`plan`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Tile width (columns).
    pub j: i64,
    /// Rows not a multiple of [`TILE_ROWS`]: ggml's `fallback` kernels.
    pub fallback: bool,
    /// Column tiles.
    pub ntx: i64,
    /// Row tiles.
    pub nty: i64,
    /// K blocks per tile.
    pub bpt: i64,
    /// CTAs.
    pub ctas: i64,
}

impl Plan {
    pub fn tiles(&self) -> i64 {
        self.ntx * self.nty
    }

    /// Whether some tile is split over several CTAs (ggml's fixup runs).
    pub fn split(&self) -> bool {
        self.tiles() % self.ctas != 0
    }

    /// CTA `b`'s first K block (`mul_mat_q`'s `kbc`, moved back to its K iteration's start).
    pub fn start(&self, b: i64) -> i64 {
        let total = self.tiles() * self.bpt;
        let kbc = b * total / self.ctas;
        kbc - (kbc % self.bpt) % BLOCKS_PER_ITER
    }

    /// The pieces of tile `t` in ggml's CTA order: `(kb0, kb1)` K block ranges.
    pub fn pieces(&self, t: i64) -> Vec<(i64, i64)> {
        let (t0, t1) = (t * self.bpt, (t + 1) * self.bpt);
        (0..self.ctas)
            .filter_map(|b| {
                let (s, e) = (self.start(b).max(t0), self.start(b + 1).min(t1));
                (s < e).then_some((s - t0, e - t0))
            })
            .collect()
    }

    /// The matmul kernel's dynamic shared memory (`mmq_get_nbytes_shared`: the column ids, the
    /// weight tile in the MMA layout, the input tile padded to the CTA's 4-byte loads).
    pub fn smem(&self) -> usize {
        const TILE_X: usize = (TILE_ROWS as usize) * 76 * 4;
        let j = self.j as usize;
        let chunk = MMQ_THREADS as usize * 4;
        j * 4 + TILE_X + (j * Q8_1_MMQ_BYTES).div_ceil(chunk) * chunk
    }

    /// The table index of the matmul kernel (`legion_q8_mmq_kernels`): `J / 8`, or 17 to 21
    /// for the fallback widths 8 to 128; 21 more for grids of a CTA per tile (no split tiles).
    pub fn kernel(&self) -> usize {
        let k = if self.fallback {
            17 + (self.j / 8).trailing_zeros() as usize
        } else {
            (self.j / 8) as usize
        };
        if self.split() { k } else { k + 21 }
    }

    /// As [`Plan::kernel`] for a split plan whose pieces are added up by the fixup launch
    /// ([`Plan::fixup_kernel`]) instead of the CTA that counts last: 42 more (ggml's
    /// `mul_mat_q`).
    pub fn kernel_no_count(&self) -> usize {
        self.kernel() + 42
    }

    /// The table index of the fixup launch for tiles `j` wide (`split_fixup<J>`).
    pub fn fixup_kernel(&self) -> usize {
        64 + (self.j / 8) as usize - 1
    }
}

/// How a split plan's pieces are added up ([`set_q8_mmq_fixup`], `legion serve
/// --q8-mmq-fixup`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum Fixup {
    /// Per plan, the faster of the two as measured on the GB10.
    #[default]
    Auto,
    /// By the CTA that counts a tile's last piece, inside the matmul kernel (two launches).
    Last,
    /// By a fixup launch after the matmul, ggml's stream-k fixup (three launches).
    Launch,
}

impl Fixup {
    fn code(self) -> i32 {
        match self {
            Fixup::Auto => 0,
            Fixup::Last => 1,
            Fixup::Launch => 2,
        }
    }

    fn from_code(c: i32) -> Self {
        match c {
            1 => Fixup::Last,
            2 => Fixup::Launch,
            _ => Fixup::Auto,
        }
    }

    /// Whether plan `p` adds its pieces up in a launch of its own (on either kernel).
    pub fn launch(self, p: &Plan) -> bool {
        p.split()
            && match self {
                Fixup::Last => false,
                Fixup::Launch => true,
                Fixup::Auto => auto_launch(p),
            }
    }
}

/// [`Fixup::Auto`]'s rule for a split plan, from `legion q8-mmq-check --fixup last,launch` on a
/// DGX Spark (GB10, 48 SMs; Qwen3.8-Flash-Next's 10 Q8_0 shapes at 16, 36 and 144 columns,
/// #858). The fixup launch costs about 1.5 us more than the in-kernel add-up, and saves the
/// chain of latencies on the CTA that counts last (fence, count, reads, dst update) where that
/// CTA adds many pieces or wide ones:
/// - tiles split over 4+ CTAs each (2560x512, 2560x640, 10240x320 weights, `k x rows`): the
///   launch for tiles 32+ columns wide (36 and 144 columns: 1.01-1.13 of ggml's time against
///   1.09-1.22), the in-kernel add-up for narrower ones (16 columns: 0.98-1.05 against
///   1.09-1.12);
/// - tiles split over 1-4 CTAs each (640x2560, 6144x2560, 10240x2560): the launch (0.99-1.07
///   against 1.01-1.37);
/// - fewer CTAs than tiles (at most 2 pieces a tile): the launch for K of 2048+ (2560x10240:
///   1.00-1.02 against 1.01-1.04), else the in-kernel add-up (320x10240: 0.99-1.04 against
///   1.01-1.07).
///
/// On ggml's tile code the sum of the 30 cases' best times goes from +3.8% (all in-kernel) and
/// +0.9% (all launches) of ggml's to +0.5%; on the streaming kernel from +6.2% / +4.9% to +4.6%.
fn auto_launch(p: &Plan) -> bool {
    let (tiles, ctas) = (p.tiles(), p.ctas);
    if ctas >= 4 * tiles {
        p.j >= 32
    } else if ctas > tiles {
        true
    } else {
        p.bpt >= 64
    }
}

/// The split-tile add-up for graphs built from now on ([`set_q8_mmq_fixup`]).
static FIXUP: AtomicI32 = AtomicI32::new(0);

/// Sets how graphs built from now on add up the pieces of split tiles (see [`Fixup`]). Every
/// mode adds them in ggml's order: the same bits.
pub fn set_q8_mmq_fixup(f: Fixup) {
    FIXUP.store(f.code(), Ordering::Relaxed);
}

/// The mode [`set_q8_mmq_fixup`] stored.
pub fn q8_mmq_fixup() -> Fixup {
    Fixup::from_code(FIXUP.load(Ordering::Relaxed))
}

/// ggml-cuda's MMQ of a Q8_0 weight of `rows` rows of `k` values times `n` columns on a GPU
/// with `sms` SMs (`launch_mul_mat_q`): one CTA per tile when the tiles fill their waves to 90%,
/// else `sms` CTAs sharing the tiles' K blocks.
pub fn plan(rows: i64, k: i64, n: i64, sms: i64) -> Plan {
    let j = tile_width(rows, n);
    let (ntx, nty, bpt) = ((n + j - 1) / j, (rows + TILE_ROWS - 1) / TILE_ROWS, k / QK);
    let ntiles = ntx * nty;
    let waves = (ntiles + sms - 1) / sms;
    let ctas = if 100 * ntiles / (sms * waves) >= 90 {
        ntiles
    } else {
        sms
    };
    Plan {
        j,
        fallback: rows % TILE_ROWS != 0,
        ntx,
        nty,
        bpt,
        ctas,
    }
}

/// `k` padded as ggml pads the quantized input's rows.
fn k_padded(k: i64) -> i64 {
    (k + ROW_PADDING - 1) / ROW_PADDING * ROW_PADDING
}

/// Bytes of the quantized input of `n` columns of `k` values for tiles `j` wide, with ggml's
/// padding for a tile's reads past the last column.
pub fn quant_bytes(k: i64, n: i64, j: i64) -> usize {
    let chunk = MMQ_THREADS as usize * 4;
    (k_padded(k) / Q8_1_MMQ_VALUES * n) as usize * Q8_1_MMQ_BYTES
        + (j as usize * Q8_1_MMQ_BYTES).div_ceil(chunk) * chunk
}

/// Bytes of the partial sums for `ctas` CTAs of tiles `j` wide (ggml's `tmp_fixup`).
pub fn part_bytes(ctas: i64, j: i64) -> usize {
    (ctas * j * TILE_ROWS) as usize * 4
}

/// ggml's `init_fastdiv_values` for a divisor `d` (1 to 2^31): `(mp, l)` with
/// `n / d == (umulhi(n, mp) + n) >> l` for every `n` below 2^31, so the matmul kernel divides by
/// the K blocks per tile and the column tiles as ggml's `mul_mat_q` does (#858).
pub fn fastdiv(d: u32) -> (u32, u32) {
    assert!(d > 0, "fastdiv: divisor 0");
    let mut l = 0;
    while l < 32 && (1u64 << l) < d as u64 {
        l += 1;
    }
    let mp = ((1u64 << 32) * ((1u64 << l) - d as u64) / d as u64 + 1) as u32;
    (mp, l)
}

/// Row slices (CTAs per ggml CTA) of the streaming kernel with `warps` warps of 16 rows.
pub fn fast_slices(warps: i64) -> i64 {
    TILE_ROWS / (16 * warps)
}

/// Bytes of the streaming kernel's quantized input (`[k / 32][ntx * j]` 8 words of quants,
/// then the scales at a 256-byte boundary).
pub fn fast_quant_bytes(k: i64, p: &Plan) -> usize {
    let blocks = (k / QK) as usize * (p.ntx * p.j) as usize;
    (blocks * 32).next_multiple_of(256) + blocks * 4
}

/// Whether the streaming kernel with `warps` warps takes plan `p` of a `k`-value weight with
/// `w_ld` blocks between rows: `k` a multiple of 256 (its stages of 8 blocks, ggml's split
/// points), `w_ld` a multiple of 8 (a row's stage is 272 bytes, 16-byte aligned), its counters
/// and quantized input within the scratch. Others run ggml's MMQ.
pub fn fast_fits(p: &Plan, k: i64, w_ld: i64, warps: i64) -> bool {
    matches!(warps, 4 | 8)
        && k % FAST_K == 0
        && w_ld % 8 == 0
        && p.tiles() * fast_slices(warps) <= 2 * MAX_TILES
        && p.ctas <= MAX_CTAS
        && fast_quant_bytes(k, p) <= MAX_QUANT_BYTES
}

/// Whether `w @ x` takes the op: switched on, a 12.x GPU ([`set_gpu`]), inside stage graphs, a
/// weight in one GPU's memory, shapes the op takes ([`fits`]) and a plan it runs faster than
/// ggml's MMQ ([`beats_ggml`]; the others stay on ggml's, shared quantization included).
pub fn takes(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    let sms = gpu_sms() as i64;
    ON.load(Ordering::Relaxed)
        && cfg!(feature = "cuda")
        && sms > 0
        && crate::ggml::stage_graph::enabled()
        && !w.is_host()
        && !w.is_meta()
        && fits((w.ty(), w.ne(), w.nb()), (x.ty(), x.ne(), x.nb()), sms)
        && beats_ggml(&plan(w.ne()[1], w.ne()[0], x.ne()[1], sms))
}

/// The plans the op (ggml's tile code, [`Fixup::Auto`]) runs faster than ggml's three launches,
/// from `legion q8-mmq-check` on a DGX Spark (GB10, 48 SMs; Qwen3.8-Flash-Next's 10 Q8_0 shapes
/// at 16, 36 and 144 columns, three sessions, #858): tiles at most 16 columns wide, each split
/// over 4+ CTAs (added up in the kernel) with 128+ K blocks, i.e. the 10240x320 weight
/// (`k x rows`) at 9-16 columns: 0.97-0.99 of ggml's time. Every other plan measured
/// 1.00-1.06 of ggml's time (noise to slower), so it stays on ggml's MMQ.
pub fn beats_ggml(p: &Plan) -> bool {
    p.split() && p.ctas >= 4 * p.tiles() && p.j <= 16 && p.bpt >= 128
}

/// [`takes`] on shapes: a Q8_0 weight `[k, rows]` with contiguous rows (`k` a multiple of 32);
/// an F32 `x` `[k, n]`, [`MIN_TOKENS`] to [`MAX_TOKENS`] columns, contiguous values; at most
/// [`MAX_TILES`] tiles and [`MAX_QUANT_BYTES`] of quantized input.
pub fn fits(
    w: (ffi::ggml_type, [i64; 4], [usize; 4]),
    x: (ffi::ggml_type, [i64; 4], [usize; 4]),
    sms: i64,
) -> bool {
    let (wty, [k, rows, w2, w3], wnb) = w;
    let (xty, [xk, n, x2, x3], xnb) = x;
    let shapes = wty == types::Q8_0
        && xty == types::F32
        && k > 0
        && k % QK == 0
        && xk == k
        && rows > 0
        && rows * MAX_TOKENS < 1 << 31
        && w2 == 1
        && w3 == 1
        && x2 == 1
        && x3 == 1
        && wnb[0] == BLOCK_BYTES
        && wnb[1] % BLOCK_BYTES == 0
        && wnb[1] >= (k / QK) as usize * BLOCK_BYTES
        && xnb[0] == 4
        && xnb[1] % 4 == 0
        && (MIN_TOKENS..=MAX_TOKENS).contains(&n)
        && sms > 0
        && sms <= MAX_CTAS;
    if !shapes {
        return false;
    }
    let p = plan(rows, k, n, sms);
    p.tiles() <= MAX_TILES
        && p.tiles() * p.bpt < 1 << 30
        && quant_bytes(k, n, p.j) <= MAX_QUANT_BYTES
}

/// `w @ x` (see [`takes`]) on the op: F32 `[rows, n]`, ggml's `mul_mat`'s shape, with its bits.
pub fn mul_mat<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>) -> Tensor<'a> {
    let sms = gpu_sms() as i64;
    mul_mat_fast(ctx, w, x, sms, q8_mmq_fast(), q8_mmq_fixup())
}

/// [`mul_mat`] planned for `sms` SMs on ggml's tile code (tests and the check).
pub fn mul_mat_sms<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>, sms: i64) -> Tensor<'a> {
    mul_mat_fast(ctx, w, x, sms, 0, Fixup::Auto)
}

/// [`mul_mat`] planned for `sms` SMs on kernel `fast` (see [`set_q8_mmq_fast`]) with the
/// split-tile add-up `fix` (see [`set_q8_mmq_fixup`]).
pub fn mul_mat_fast<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    sms: i64,
    fast: i32,
    fix: Fixup,
) -> Tensor<'a> {
    assert!(
        fits((w.ty(), w.ne(), w.nb()), (x.ty(), x.ne(), x.nb()), sms),
        "q8_mmq: shapes"
    );
    let rows = w.ne()[1];
    let n = x.ne()[1];
    ctx.external_with(
        &Q8_MMQ,
        types::F32,
        [rows, n, 1, 1],
        &[x, w],
        &[sms as i32, fast, fix.code()],
    )
}

/// The node's [`Plan`].
///
/// # Safety
/// `dst` is a node made by [`mul_mat`].
unsafe fn node_plan(dst: &ffi::ggml_tensor) -> Plan {
    // SAFETY: a node made by `mul_mat`: src[0] the input, src[1] the weight.
    unsafe {
        let (x, w) = (&*dst.src[0], &*dst.src[1]);
        let sms = crate::ggml::external_params(dst)[0] as i64;
        plan(w.ne[1], x.ne[0], x.ne[1], sms)
    }
}

/// Whether the node runs ggml's MMQ tile code (else the streaming kernel, [`fast_fits`]): the
/// tile code's quantized input is ggml's `block_q8_1_mmq` layout, the same for every plan.
///
/// # Safety
/// `dst` is a node made by [`mul_mat`].
unsafe fn runs_tile(dst: &ffi::ggml_tensor) -> bool {
    // SAFETY: the caller's: src[0] the input, src[1] the weight.
    unsafe {
        let (x, w) = (&*dst.src[0], &*dst.src[1]);
        let fast = crate::ggml::external_params(dst)[1] as i64;
        let w_ld = (w.nb[1] / BLOCK_BYTES) as i64;
        !(fast_fits(&node_plan(dst), x.ne[0], w_ld, fast) && (w.data as usize).is_multiple_of(16))
    }
}

/// Whether `node` is one of the op's.
///
/// # Safety
/// `node` is a valid tensor.
unsafe fn is_ours(node: *const ffi::ggml_tensor) -> bool {
    // SAFETY: the caller's.
    unsafe { crate::ggml::op_of(node).is_some_and(|o| std::ptr::eq(o, &Q8_MMQ)) }
}

/// How the op's node `nodes[at]` gets its quantized input (#858). `None`: the last node of the
/// op before it in `nodes` (the graph's order, in which the nodes run on one stream, directly or
/// captured in a stage graph) read the same input and both run ggml's tile code
/// ([`runs_tile`]), so the input's blocks are in the stream's scratch already, as ggml's own
/// quantization would write them: no `quantize` launch. `Some(tiles)`: it quantizes, and zeroes
/// the tile counters of the most tiles among it and the nodes after it that reuse its blocks
/// (each split tile's counter goes back to 0 once its last piece is counted, so the counters
/// stay zero along such a run). The same graph gives the same answer on every run, so a capture
/// records the launches a direct run makes.
///
/// # Safety
/// Every pointer of `nodes` is a valid tensor; `nodes[at]` is a node made by [`mul_mat`].
pub unsafe fn quant_run(nodes: &[*mut ffi::ggml_tensor], at: usize) -> Option<i64> {
    // SAFETY: the caller's; the op's nodes are nodes made by `mul_mat`.
    unsafe {
        let me = &*nodes[at];
        let x = me.src[0];
        let tile = runs_tile(me);
        if tile
            && let Some(&prev) = nodes[..at].iter().rev().find(|&&n| is_ours(n))
            && (*prev).src[0] == x
            && runs_tile(&*prev)
        {
            return None;
        }
        let mut tiles = node_plan(me).tiles();
        if tile {
            for &n in nodes[at + 1..].iter().filter(|&&n| is_ours(n)) {
                if (*n).src[0] != x || !runs_tile(&*n) {
                    break;
                }
                tiles = tiles.max(node_plan(&*n).tiles());
            }
        }
        Some(tiles)
    }
}

/// A column of `x` quantized as `quantize_mmq_q8_1` does for a Q8_0 weight: per block of 32,
/// the quants and the scale.
pub fn quantize_col(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let mut q = Vec::with_capacity(x.len());
    let mut d = Vec::with_capacity(x.len() / QK as usize);
    for v in x.chunks(QK as usize) {
        let amax = v.iter().fold(0f32, |m, x| m.max(x.abs()));
        let d_inv = 127.0 / amax;
        // NaN (an all-zero block) converts to 0, as on the GPU.
        q.extend(v.iter().map(|x| (x * d_inv).round() as i8));
        d.push(1.0 / d_inv);
    }
    (q, d)
}

unsafe fn cpu_q8_mmq(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat`, host tensors.
    unsafe {
        let (x, w) = (&*dst.src[0], &*dst.src[1]);
        let (k, n, rows) = (x.ne[0] as usize, x.ne[1] as usize, w.ne[1] as usize);
        let p = node_plan(dst);
        let cols: Vec<(Vec<i8>, Vec<f32>)> = (0..n)
            .map(|c| {
                let col = (x.data as *const u8).add(c * x.nb[1]) as *const f32;
                quantize_col(std::slice::from_raw_parts(col, k))
            })
            .collect();
        let out = dst.data as *mut f32;
        let piece_sum = |r: usize, c: usize, (kb0, kb1): (i64, i64)| {
            let row = (w.data as *const u8).add(r * w.nb[1]);
            let (q, d) = &cols[c];
            let mut s = 0f32;
            for b in kb0 as usize..kb1 as usize {
                let wb = row.add(b * BLOCK_BYTES);
                let dw = ffi::ggml_fp16_to_fp32(u16::from_le_bytes([*wb, *wb.add(1)]));
                let mut c32 = 0i32;
                for v in 0..QK as usize {
                    c32 += *wb.add(2 + v) as i8 as i32 * q[b * QK as usize + v] as i32;
                }
                s += c32 as f32 * dw * d[b];
            }
            s
        };
        for r in (ith..rows).step_by(nth) {
            let it = r as i64 / TILE_ROWS;
            for c in 0..n {
                let pieces = p.pieces(it * p.ntx + c as i64 / p.j);
                let (last, rest) = pieces.split_last().expect("a tile has pieces");
                let mut v = piece_sum(r, c, *last);
                if !rest.is_empty() {
                    // ggml's fixup: the pieces before the last, walking back, added to it.
                    let mut s = 0f32;
                    for &pc in rest.iter().rev() {
                        s += piece_sum(r, c, pc);
                    }
                    v += s;
                }
                *out.add(c * rows + r) = v;
            }
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::cell::Cell;
    use std::collections::HashMap;
    use std::os::raw::{c_int, c_void};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicI8, Ordering};

    use super::*;

    /// `legion_q8_mmq` of `src/cuda/q8_mmq.cu`.
    #[repr(C)]
    struct Params {
        w: *const c_void,
        x: *const f32,
        yq: *mut c_void,
        cnt: *mut i32,
        part: *mut f32,
        dst: *mut f32,
        x_ld: i64,
        w_ld: i32,
        k: i32,
        k_pad: i32,
        n: i32,
        rows: i32,
        ntiles: i32,
        nzero: i32,
        /// [`fastdiv`] of the K blocks per tile and of the column tiles: `mp`, `l` each.
        div: [u32; 4],
    }

    /// `legion_q8_fast` of `src/cuda/q8_mmq.cu`.
    #[repr(C)]
    struct FastParams {
        w: *const c_void,
        x: *const f32,
        yq: *mut i32,
        yd: *mut f32,
        cnt: *mut i32,
        part: *mut f32,
        dst: *mut f32,
        x_ld: i64,
        w_ld: i32,
        k: i32,
        n: i32,
        n_pad: i32,
        rows: i32,
        ntiles: i32,
        ntx: i32,
        ctas: i32,
        ncnt: i32,
        sep: i32,
    }

    /// Kernels in `legion_q8_fast_kernels`: the quantization, then the matmul by tile width
    /// (`J / 8`) with 8 warps, then with 4.
    const FAST_KERNELS: usize = 33;

    /// `dim3`.
    #[repr(C)]
    struct Dim3 {
        x: u32,
        y: u32,
        z: u32,
    }

    /// Kernels in `legion_q8_mmq_kernels`: the quantization, the matmuls by tile width (split
    /// with the count, unsplit, split without the count), the fixup launches by tile width.
    const KERNELS: usize = 80;

    unsafe extern "C" {
        static legion_q8_mmq_kernels: [*const c_void; KERNELS];
        static legion_q8_fast_kernels: [*const c_void; FAST_KERNELS];
        fn cudaLaunchKernel(
            func: *const c_void,
            grid: Dim3,
            block: Dim3,
            args: *mut *mut c_void,
            shared: usize,
            stream: *mut c_void,
        ) -> c_int;
        fn cudaGetDevice(device: *mut c_int) -> c_int;
        fn cudaFuncSetAttribute(func: *const c_void, attr: c_int, value: c_int) -> c_int;
        fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> c_int;
        fn cudaGetLastError() -> c_int;
    }

    const FUNC_ATTR_MAX_DYNAMIC_SHARED: c_int = 8;

    /// Devices the shared-memory cache covers.
    const DEVICES: usize = 64;

    /// Each device's shared-memory opt-in of each kernel: 0 not yet, 1 done, -1 refused.
    static SMEM: [[AtomicI8; KERNELS]; DEVICES] =
        [const { [const { AtomicI8::new(0) }; KERNELS] }; DEVICES];

    /// One stream's scratch: the quantized input, the tile counters, the partial sums.
    #[derive(Clone, Copy)]
    struct Scratch {
        yq: usize,
        cnt: usize,
        part: usize,
        part_bytes: usize,
    }

    /// The scratch of each (device, stream), allocated on first use at the most any node takes
    /// ([`MAX_QUANT_BYTES`], [`MAX_TILES`], partial sums for the SMs' CTAs of the widest tile)
    /// and kept: graphs captured with these pointers replay them. Ops on one stream run one
    /// after the other, so they share it as ggml's MMQ shares its pool.
    static SCRATCH: Mutex<Option<HashMap<(c_int, usize), Scratch>>> = Mutex::new(None);

    fn scratch(dev: c_int, stream: *mut c_void, sms: i64) -> Result<Scratch, String> {
        let mut map = SCRATCH.lock().unwrap();
        let map = map.get_or_insert_with(HashMap::new);
        if let Some(s) = map.get(&(dev, stream as usize)) {
            return Ok(*s);
        }
        let part_bytes = part_bytes(sms, 128);
        // Counters for the streaming kernel's row slices too ([`fast_fits`]).
        let sizes = [MAX_QUANT_BYTES, 2 * MAX_TILES as usize * 4, part_bytes];
        let mut ptrs = [0usize; 3];
        for (p, &bytes) in ptrs.iter_mut().zip(&sizes) {
            let mut q = std::ptr::null_mut();
            // SAFETY: a plain allocation on the current device (a relaxed capture allows it).
            if unsafe { cudaMalloc(&mut q, bytes) } != 0 {
                // SAFETY: clears the error the failed call left.
                unsafe { cudaGetLastError() };
                return Err(format!("q8_mmq: cudaMalloc of {bytes} bytes"));
            }
            *p = q as usize;
        }
        let s = Scratch {
            yq: ptrs[0],
            cnt: ptrs[1],
            part: ptrs[2],
            part_bytes,
        };
        map.insert((dev, stream as usize), s);
        Ok(s)
    }

    fn opt_in(dev: c_int, kernel: usize, bytes: usize) -> Result<(), String> {
        let slot = &SMEM[dev as usize][kernel];
        let mut state = slot.load(Ordering::Relaxed);
        if state == 0 {
            // SAFETY: a kernel of the table and its shared memory size.
            state = unsafe {
                if cudaFuncSetAttribute(
                    legion_q8_mmq_kernels[kernel],
                    FUNC_ATTR_MAX_DYNAMIC_SHARED,
                    bytes as c_int,
                ) == 0
                {
                    1
                } else {
                    cudaGetLastError();
                    -1
                }
            };
            slot.store(state, Ordering::Relaxed);
        }
        if state < 0 {
            return Err(format!("q8_mmq: {bytes} bytes of shared memory"));
        }
        Ok(())
    }

    fn launch(
        kernel: usize,
        grid: Dim3,
        block: Dim3,
        shared: usize,
        p: &mut Params,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let mut argv = [(p as *mut Params).cast::<c_void>()];
        // SAFETY: a kernel of the table taking `legion_q8_mmq` by value, from the launch's copy
        // of `p`; its pointers are device memory of the stream's device.
        match unsafe {
            cudaLaunchKernel(
                legion_q8_mmq_kernels[kernel],
                grid,
                block,
                argv.as_mut_ptr(),
                shared,
                stream,
            )
        } {
            0 => Ok(()),
            e => Err(format!("q8_mmq: CUDA error {e}")),
        }
    }

    /// The streaming kernel's shared memory with `warps` warps (bytes): two stages of 16 rows a
    /// warp, 68 words (8 blocks) a row.
    fn fast_smem(warps: i64) -> usize {
        2 * 16 * warps as usize * 68 * 4
    }

    /// Each device's shared-memory opt-in of each streaming kernel: 0 not yet, 1 done, -1
    /// refused.
    static FAST_SMEM: [[AtomicI8; FAST_KERNELS]; DEVICES] =
        [const { [const { AtomicI8::new(0) }; FAST_KERNELS] }; DEVICES];

    fn fast_opt_in(dev: c_int, kernel: usize, bytes: usize) -> Result<(), String> {
        let slot = &FAST_SMEM[dev as usize][kernel];
        let mut state = slot.load(Ordering::Relaxed);
        if state == 0 {
            // SAFETY: a kernel of the table and its shared memory size.
            state = unsafe {
                if cudaFuncSetAttribute(
                    legion_q8_fast_kernels[kernel],
                    FUNC_ATTR_MAX_DYNAMIC_SHARED,
                    bytes as c_int,
                ) == 0
                {
                    1
                } else {
                    cudaGetLastError();
                    -1
                }
            };
            slot.store(state, Ordering::Relaxed);
        }
        if state < 0 {
            return Err(format!("q8_mmq: {bytes} bytes of shared memory"));
        }
        Ok(())
    }

    fn launch_fast(
        kernel: usize,
        grid: Dim3,
        block: Dim3,
        shared: usize,
        p: &mut FastParams,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let mut argv = [(p as *mut FastParams).cast::<c_void>()];
        // SAFETY: a kernel of the table taking `legion_q8_fast` by value, from the launch's
        // copy of `p`; its pointers are device memory of the stream's device.
        match unsafe {
            cudaLaunchKernel(
                legion_q8_fast_kernels[kernel],
                grid,
                block,
                argv.as_mut_ptr(),
                shared,
                stream,
            )
        } {
            0 => Ok(()),
            e => Err(format!("q8_mmq: CUDA error {e}")),
        }
    }

    thread_local! {
        /// The graph and index of the last node of the op this thread launched: the next one
        /// is usually a little after it, so [`quant_in`] looks there first.
        static LAST: Cell<(usize, usize)> = const { Cell::new((0, usize::MAX)) };
    }

    /// [`quant_run`] for `dst` in graph `g`; a node outside a graph (`g` null or not holding
    /// it) quantizes and zeroes its own counters.
    ///
    /// # Safety
    /// `g` is null or a valid graph; `dst` a node made by [`mul_mat`].
    unsafe fn quant_in(g: *mut ffi::ggml_cgraph, dst: &ffi::ggml_tensor) -> Option<i64> {
        // SAFETY: the caller's: the graph's node array holds `n` valid tensor pointers.
        unsafe {
            let own = Some(node_plan(dst).tiles());
            if g.is_null() {
                return own;
            }
            let n = ffi::ggml_graph_n_nodes(g) as usize;
            let nodes = std::slice::from_raw_parts(ffi::ggml_graph_nodes(g), n);
            let node = dst as *const ffi::ggml_tensor;
            let (lg, li) = LAST.with(Cell::get);
            // Past the last launched node first (it is one of the op's while the graph is the
            // same), else from the start.
            let from = if lg == g as usize && li < n && is_ours(nodes[li]) {
                li
            } else {
                0
            };
            let at = (from..n)
                .find(|&i| std::ptr::eq(nodes[i], node))
                .or_else(|| (0..from).find(|&i| std::ptr::eq(nodes[i], node)));
            let Some(at) = at else { return own };
            LAST.with(|c| c.set((g as usize, at)));
            quant_run(nodes, at)
        }
    }

    pub(super) unsafe fn q8_mmq(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        // SAFETY: a node made by `mul_mat` (its shapes passed `fits`), device tensors of the
        // stream's device, the current one.
        unsafe {
            let (x, w) = (&*dst.src[0], &*dst.src[1]);
            let p = node_plan(dst);
            let sms = crate::ggml::external_params(dst)[0] as i64;
            let mut dev: c_int = 0;
            if cudaGetDevice(&mut dev) != 0 || !(0..DEVICES as c_int).contains(&dev) {
                return Err("q8_mmq: no current device".into());
            }
            let s = scratch(dev, stream, sms)?;
            if part_bytes(p.ctas, p.j) > s.part_bytes && p.split() {
                return Err("q8_mmq: more CTAs than the scratch holds".into());
            }
            let (k, n) = (x.ne[0], x.ne[1]);
            let fix = Fixup::from_code(crate::ggml::external_params(dst)[2]);
            let sep = fix.launch(&p);
            if !runs_tile(dst) {
                let fast = crate::ggml::external_params(dst)[1] as i64;
                return q8_fast(dst, &p, &s, dev, fast, sep, stream);
            }
            let quant = quant_in(crate::ggml::current_cuda_graph(), dst);
            let ((bpt_mp, bpt_l), (ntx_mp, ntx_l)) = (fastdiv(p.bpt as u32), fastdiv(p.ntx as u32));
            let mut a = Params {
                w: w.data as *const c_void,
                x: x.data as *const f32,
                yq: s.yq as *mut c_void,
                cnt: s.cnt as *mut i32,
                part: s.part as *mut f32,
                dst: dst.data as *mut f32,
                x_ld: (x.nb[1] / 4) as i64,
                w_ld: (w.nb[1] / BLOCK_BYTES) as i32,
                k: k as i32,
                k_pad: k_padded(k) as i32,
                n: n as i32,
                rows: w.ne[1] as i32,
                ntiles: p.tiles() as i32,
                nzero: quant.unwrap_or(0) as i32,
                div: [bpt_mp, bpt_l, ntx_mp, ntx_l],
            };
            if quant.is_some() {
                let quant_grid = Dim3 {
                    x: n as u32,
                    y: (k_padded(k) / ROW_PADDING) as u32,
                    z: 1,
                };
                let quant_block = Dim3 { x: 128, y: 1, z: 1 };
                launch(0, quant_grid, quant_block, 0, &mut a, stream)?;
            }
            let kernel = if sep { p.kernel_no_count() } else { p.kernel() };
            let smem = p.smem();
            opt_in(dev, kernel, smem)?;
            let grid = Dim3 {
                x: p.ctas as u32,
                y: 1,
                z: 1,
            };
            let block = Dim3 {
                x: 32,
                y: MMQ_THREADS / 32,
                z: 1,
            };
            launch(kernel, grid, block, smem, &mut a, stream)?;
            if sep {
                launch_fixup(p, &mut a, stream)?;
            }
            Ok(())
        }
    }

    /// The fixup launch of split plan `p` (`split_fixup<J>`): ggml's grid of the matmul's
    /// CTAs times 4 slices of 32 rows, 32 x 4 threads. `a`: the partial sums, dst and the
    /// shape (`k`, `n`, `rows`, `ntiles`).
    fn launch_fixup(p: Plan, a: &mut Params, stream: *mut c_void) -> Result<(), String> {
        let grid = Dim3 {
            x: p.ctas as u32,
            y: (TILE_ROWS / 32) as u32,
            z: 1,
        };
        let block = Dim3 { x: 32, y: 4, z: 1 };
        launch(p.fixup_kernel(), grid, block, 0, a, stream)
    }

    /// The node on the streaming kernel (`fast` warps per CTA): its quantization, then ggml's
    /// grid times the row slices.
    ///
    /// # Safety
    /// As [`q8_mmq`], and the plan passed [`fast_fits`].
    unsafe fn q8_fast(
        dst: &ffi::ggml_tensor,
        p: &Plan,
        s: &Scratch,
        dev: c_int,
        fast: i64,
        sep: bool,
        stream: *mut c_void,
    ) -> Result<(), String> {
        // SAFETY: as the caller's.
        unsafe {
            let (x, w) = (&*dst.src[0], &*dst.src[1]);
            let (k, n) = (x.ne[0], x.ne[1]);
            let slices = fast_slices(fast);
            let n_pad = p.ntx * p.j;
            let yd_off = ((k / QK) as usize * n_pad as usize * 32).next_multiple_of(256);
            let mut a = FastParams {
                w: w.data as *const c_void,
                x: x.data as *const f32,
                yq: s.yq as *mut i32,
                yd: (s.yq + yd_off) as *mut f32,
                cnt: s.cnt as *mut i32,
                part: s.part as *mut f32,
                dst: dst.data as *mut f32,
                x_ld: (x.nb[1] / 4) as i64,
                w_ld: (w.nb[1] / BLOCK_BYTES) as i32,
                k: k as i32,
                n: n as i32,
                n_pad: n_pad as i32,
                rows: w.ne[1] as i32,
                ntiles: p.tiles() as i32,
                ntx: p.ntx as i32,
                ctas: p.ctas as i32,
                ncnt: (p.tiles() * slices) as i32,
                sep: sep as i32,
            };
            let quant_grid = Dim3 {
                x: n as u32,
                y: (k / FAST_K) as u32,
                z: 1,
            };
            let quant_block = Dim3 { x: 64, y: 1, z: 1 };
            launch_fast(0, quant_grid, quant_block, 0, &mut a, stream)?;
            let kernel = (p.j / 8) as usize + if fast == 8 { 0 } else { 16 };
            let grid = Dim3 {
                x: p.ctas as u32,
                y: slices as u32,
                z: 1,
            };
            let block = Dim3 {
                x: 32 * fast as u32,
                y: 1,
                z: 1,
            };
            let smem = fast_smem(fast);
            fast_opt_in(dev, kernel, smem)?;
            launch_fast(kernel, grid, block, smem, &mut a, stream)?;
            if sep {
                let mut f = Params {
                    w: std::ptr::null(),
                    x: std::ptr::null(),
                    yq: std::ptr::null_mut(),
                    cnt: std::ptr::null_mut(),
                    part: a.part,
                    dst: a.dst,
                    x_ld: 0,
                    w_ld: 0,
                    k: a.k,
                    k_pad: 0,
                    n: a.n,
                    rows: a.rows,
                    ntiles: a.ntiles,
                    nzero: 0,
                    div: [0; 4],
                };
                launch_fixup(*p, &mut f, stream)?;
            }
            Ok(())
        }
    }
}
