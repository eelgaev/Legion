//! Paged attention over TurboQuant caches (#552, [`crate::model::core::turboquant`]): a layer whose
//! K or V cache (or both) is `tq3` / `tq4`. Its own op and kernels (`src/cuda/turboquant.cu`), so
//! the paged kernels of other cache types stay as they are.
//!
//! The math runs in the rotated basis of the layer's transform `R`: with a TurboQuant K the query
//! is rotated once (`R q`) and scored against the stored centroids (`n · g[idx] / sqrt(d)`, no
//! inverse transform per key); with a TurboQuant V the weighted sum of the stored centroids is
//! rotated back once (`R^T`) per output row. A K or V of another type (F16, q8_0, q4_0) is read
//! as it is.
//!
//! On CUDA:
//! - groups of up to [`DIRECT_TOK`] tokens per sequence (decode, speculative verify steps):
//!   flash-decoding over the blocks in place, each (sequence, token) one query, reading the
//!   indices straight from the pool (`tq_decode`), then a combine of the splits;
//! - longer groups (prefill chunks): pieces of at most `cap` blocks, dequantized (in the rotated
//!   basis) into F16 staging slots, which the existing F16 prefill kernels of
//!   `src/cuda/paged_attn/` read through their tiered-pool pieces (#352), unchanged: on every
//!   GPU from sm_70 (the WMMA, m8n8k4 and mma.sync kernels all read F16).
//!
//! On the CPU, a reference implementation in F32 (tests).

#[cfg(feature = "cuda")]
use std::os::raw::c_void;

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};
use crate::model::core::KV_BLOCK;
use crate::model::core::turboquant::{self, Cache};

/// Most tokens per sequence a group may have to take the direct (decode) kernel; longer groups
/// go through the staged prefill.
pub const DIRECT_TOK: usize = 8;

/// Bytes of F16 staging (K and V) a prefill group's scratch holds at most: pieces of as many
/// blocks.
const STAGE_BUDGET: usize = 64 << 20;

/// A layer's K and V caches: TurboQuant ones (`Some`), or of a ggml type (`None`, read as the
/// paged kernels read them). At least one is TurboQuant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TqCaches {
    pub k: Option<Cache>,
    pub v: Option<Cache>,
}

impl TqCaches {
    fn any(&self) -> Cache {
        self.k.or(self.v).expect("a TurboQuant K or V")
    }
}

/// Whether this op reads caches of `types` on CUDA (any GPU from sm_70): a TurboQuant K or V,
/// the other TurboQuant, F16, q8_0 or q4_0.
pub fn reads(types: crate::model::core::KvTypes) -> bool {
    use crate::model::core::KvType::*;
    let ok = |t| matches!(t, F16 | Q8_0 | Q4_0 | Tq3 | Tq4);
    cfg!(feature = "cuda") && types.has_tq() && ok(types.k) && ok(types.v)
}

static TQ_ATTN: ExternalOp = ExternalOp {
    name: "tq_paged_attn",
    cpu: cpu_tq_attn,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_tq_attn),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// A cache's code for the kernels: 0 F16, 1 q8_0, 2 q4_0, 3 tq3, 4 tq4; `None` if neither reads
/// it.
fn code(c: Option<Cache>, ty: ffi::ggml_type) -> Option<i32> {
    match c {
        Some(c) => Some(c.layout.bits as i32),
        None if ty == types::F16 => Some(0),
        None if ty == types::Q8_0 => Some(1),
        None if ty == ffi::ggml_type_GGML_TYPE_Q4_0 => Some(2),
        None => None,
    }
}

/// The scratch's parts, in F32 elements, after the output (`o - q` of them).
#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct Plan {
    /// Splits per query of the direct kernel.
    splits: usize,
    /// Blocks per piece of the staged prefill.
    cap: usize,
    direct: bool,
    /// Offsets: the contiguous (rotated) query, the rotated output, the splits' partials or
    /// the prefill's (m, l) rows, its I32 piece arrays, its F16 staging.
    q: usize,
    o: usize,
    part: usize,
    ints: usize,
    stage: usize,
    len: usize,
}

impl Plan {
    fn new(d: usize, n_head: usize, n_kv: usize, n_tok: usize, n_seqs: usize, tw: usize) -> Self {
        let qn = d * n_head * n_tok * n_seqs;
        let direct = n_tok <= DIRECT_TOK;
        let splits = super::decode_splits(n_seqs * n_tok, n_kv);
        let kv_row = d * n_kv;
        let cap = (STAGE_BUDGET / (KV_BLOCK * kv_row * 2 * 2)).clamp(1, (tw * n_seqs).max(1));
        let a4 = |x: usize| x.next_multiple_of(4);
        let q = a4(qn);
        let o = a4(q + qn);
        let part = a4(o + qn);
        let (ints, stage, len) = if direct {
            let end = part + (d + 2) * splits * n_head * n_tok * n_seqs;
            (end, end, end)
        } else {
            let ints = a4(part + 2 * n_head * n_tok * n_seqs);
            let stage = a4(ints + 2 * tw * n_seqs + n_seqs + 1 + cap);
            (ints, stage, stage + cap * KV_BLOCK * kv_row)
        };
        Self {
            splits,
            cap,
            direct,
            q,
            o,
            part,
            ints,
            stage,
            len,
        }
    }
}

/// Attention of `q` (F32 `[d, n_head, n_tok, n_seqs]`, may be a strided view) over the pools `k`,
/// `v` of a layer with TurboQuant caches `caches`, through the block tables `table` from the
/// positions `n_past`, with softmax scale `scale`, sliding window `window` and ring `ring` as in
/// [`super::attention_tiered`]. Returns F32 `[d, n_head, n_tok, n_seqs]`, contiguous (a view of
/// the op's node, whose tail is the kernels' scratch).
#[allow(clippy::too_many_arguments)]
pub fn attention<'a>(
    ctx: &'a Context,
    q: Tensor<'a>,
    k: Tensor<'a>,
    v: Tensor<'a>,
    table: Tensor<'a>,
    n_past: Tensor<'a>,
    caches: TqCaches,
    scale: f32,
    window: usize,
    ring: usize,
) -> Tensor<'a> {
    let [d, n_head, n_tok, n_seqs] = q.ne();
    let c = caches.any();
    assert_eq!(d as usize, c.layout.d, "tq attention: head size");
    let k_code = code(caches.k, k.ty()).expect("tq attention: K type");
    let v_code = code(caches.v, v.ty()).expect("tq attention: V type");
    let plan = Plan::new(
        d as usize,
        n_head as usize,
        c.layout.n_kv,
        n_tok as usize,
        n_seqs as usize,
        table.ne()[0] as usize,
    );
    let params = [
        scale.to_bits() as i32,
        window as i32,
        ring as i32,
        k_code,
        v_code,
        c.seed as i32,
        c.layout.n_kv as i32,
        // The query's shape, which the node's own (1-D) shape does not keep.
        n_head as i32,
        n_tok as i32,
        n_seqs as i32,
    ];
    let node = ctx.external_with(
        &TQ_ATTN,
        types::F32,
        [plan.len as i64, 1, 1, 1],
        &[q, k, v, table, n_past],
        &params,
    );
    let f = 4usize;
    ctx.view_4d(
        node,
        [d, n_head, n_tok, n_seqs],
        [
            d as usize * f,
            (d * n_head) as usize * f,
            (d * n_head * n_tok) as usize * f,
        ],
        0,
    )
}

/// The op's operands, read from the node.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct Operands {
    d: usize,
    n_head: usize,
    n_kv: usize,
    n_tok: usize,
    n_seqs: usize,
    q: *const u8,
    q_nb: [usize; 4],
    k: *const u8,
    v: *const u8,
    k_ty: ffi::ggml_type,
    v_ty: ffi::ggml_type,
    k_cell: usize,
    v_cell: usize,
    k_code: i32,
    v_code: i32,
    table: *const i32,
    table_width: usize,
    n_past: *const i32,
    dst: *mut f32,
    scale: f32,
    window: usize,
    ring: usize,
    seed: u32,
    plan: Plan,
}

impl Operands {
    /// # Safety
    /// `dst` must be a node made by [`attention`].
    unsafe fn of(dst: &ffi::ggml_tensor) -> Self {
        unsafe {
            let src = |i: usize| &*dst.src[i];
            let (q, k, v, table, n_past) = (src(0), src(1), src(2), src(3), src(4));
            let p = crate::ggml::external_params(dst);
            let d = q.ne[0] as usize;
            let (n_head, n_kv, n_tok, n_seqs) =
                (p[7] as usize, p[6] as usize, p[8] as usize, p[9] as usize);
            Self {
                d,
                n_head,
                n_kv,
                n_tok,
                n_seqs,
                q: q.data as *const u8,
                q_nb: q.nb,
                k: k.data as *const u8,
                v: v.data as *const u8,
                k_ty: k.type_,
                v_ty: v.type_,
                k_cell: k.nb[1],
                v_cell: v.nb[1],
                k_code: p[3],
                v_code: p[4],
                table: table.data as *const i32,
                table_width: table.ne[0] as usize,
                n_past: n_past.data as *const i32,
                dst: dst.data as *mut f32,
                scale: f32::from_bits(p[0] as u32),
                window: p[1] as usize,
                ring: p[2] as usize,
                seed: p[5] as u32,
                plan: Plan::new(d, n_head, n_kv, n_tok, n_seqs, table.ne[0] as usize),
            }
        }
    }

    fn layout(&self, code: i32) -> turboquant::Layout {
        turboquant::Layout {
            d: self.d,
            n_kv: self.n_kv,
            bits: code as u32,
        }
    }

    /// Head `kvh` of cell `c` of the pool at `base` (type `ty`, code `code`, `bytes` per cell),
    /// in the attention's basis: rotated for a TurboQuant cache.
    ///
    /// # Safety
    /// `base` must hold cell `c`, in host memory.
    #[allow(clippy::too_many_arguments)]
    unsafe fn head(
        &self,
        base: *const u8,
        ty: ffi::ggml_type,
        code: i32,
        bytes: usize,
        c: usize,
        kvh: usize,
        out: &mut [f32],
    ) {
        unsafe {
            let raw = std::slice::from_raw_parts(base.add(c * bytes), bytes);
            if code >= 3 {
                let l = self.layout(code);
                let (w, n) = l.head(raw, kvh);
                out.copy_from_slice(&turboquant::decode_head_rotated(l.bits, &w, n, self.d));
            } else if ty == types::F16 {
                ffi::ggml_fp16_to_fp32_row(
                    (raw.as_ptr() as *const u16).add(kvh * self.d),
                    out.as_mut_ptr(),
                    self.d as i64,
                );
            } else {
                let all = crate::ggml::dequantize(ty, raw, self.d * self.n_kv);
                out.copy_from_slice(&all[kvh * self.d..(kvh + 1) * self.d]);
            }
        }
    }
}

/// The CPU implementation: per (sequence, token, head) row, in F32, the rotated-basis math the
/// kernels do (the query rotated for a TurboQuant K, the output rotated back for a TurboQuant
/// V). Only the output part of the node is written.
unsafe fn cpu_tq_attn(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `attention`, host tensors.
    unsafe {
        let o = Operands::of(dst);
        let g = o.n_head / o.n_kv;
        let rows = o.n_seqs * o.n_tok * o.n_head;
        let mut kf = vec![0f32; o.d];
        let mut scores = Vec::new();
        let mut out = vec![0f32; o.d];
        for row in (ith..rows).step_by(nth) {
            let h = row % o.n_head;
            let t = row / o.n_head % o.n_tok;
            let s = row / (o.n_head * o.n_tok);
            let kvh = h / g;
            let mut q = std::slice::from_raw_parts(
                o.q.add(s * o.q_nb[3] + t * o.q_nb[2] + h * o.q_nb[1]) as *const f32,
                o.d,
            )
            .to_vec();
            if o.k_code >= 3 {
                q = turboquant::rotate(o.seed, &q);
            }
            let visible = *o.n_past.add(s) as usize + t + 1;
            let first = if o.window > 0 {
                visible.saturating_sub(o.window)
            } else {
                0
            };
            let cell = |p: usize| {
                let entry = if o.ring > 0 {
                    p / KV_BLOCK % o.ring
                } else {
                    p / KV_BLOCK
                };
                let b = *o.table.add(s * o.table_width + entry) as usize;
                b * KV_BLOCK + p % KV_BLOCK
            };
            scores.clear();
            for p in first..visible {
                o.head(o.k, o.k_ty, o.k_code, o.k_cell, cell(p), kvh, &mut kf);
                let dot: f32 = q.iter().zip(&kf).map(|(a, b)| a * b).sum();
                scores.push(dot * o.scale);
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for x in &mut scores {
                *x = (*x - max).exp();
                sum += *x;
            }
            out.fill(0.0);
            for (p, w) in (first..visible).zip(&scores) {
                o.head(o.v, o.v_ty, o.v_code, o.v_cell, cell(p), kvh, &mut kf);
                for (acc, x) in out.iter_mut().zip(&kf) {
                    *acc += w / sum * x;
                }
            }
            if o.v_code >= 3 {
                out = turboquant::unrotate(o.seed, &out);
            }
            std::slice::from_raw_parts_mut(o.dst.add(row * o.d), o.d).copy_from_slice(&out);
        }
    }
}

/// `TqAttn` in `src/cuda/turboquant.cu`.
#[cfg(feature = "cuda")]
#[repr(C)]
struct TqAttn {
    q: *const f32,
    k: *const u8,
    v: *const u8,
    k_cell: i64,
    v_cell: i64,
    k_code: i32,
    v_code: i32,
    table: *const i32,
    table_width: i32,
    n_past: *const i32,
    d: i32,
    n_head: i32,
    n_kv: i32,
    n_tok: i32,
    n_seqs: i32,
    scale: f32,
    rsd: f32,
    window: i32,
    ring: i32,
    seed: u32,
    out: *mut f32,
    part: *mut f32,
    max_split: i32,
    piece_blk: *mut i32,
    piece_ent: *mut i32,
    piece_n: *mut i32,
    piece_cnt: *mut i32,
    src: *mut i32,
    cap: i32,
    piece: i32,
    k_stage: *mut u16,
    v_stage: *mut u16,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_tq_transform(
        x: *const f32,
        nb1: i64,
        nb2: i64,
        nb3: i64,
        d: i32,
        n_head: i32,
        n_tok: i32,
        n_seqs: i32,
        mode: i32,
        seed: u32,
        rsd: f32,
        out: *mut f32,
        stream: *mut c_void,
    ) -> i32;
    fn legion_tq_decode(a: *const TqAttn, t: *const turboquant::Tables, stream: *mut c_void)
    -> i32;
    fn legion_tq_stage(a: *const TqAttn, t: *const turboquant::Tables, stream: *mut c_void) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_tq_attn(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `attention`, all tensors in device memory of the stream's device.
    let o = unsafe { Operands::of(dst) };
    let pl = o.plan;
    let base = o.dst;
    let at = |off: usize| unsafe { base.add(off) };
    let check = |rc: i32, what: &str| match rc {
        0 => Ok(()),
        1 => Err(format!(
            "{what}: unsupported shape: head size {}, {} heads over {} kv heads",
            o.d, o.n_head, o.n_kv
        )),
        e => Err(format!("{what}: CUDA launch failed ({e})")),
    };
    let rsd = turboquant::inv_sqrt(o.d);
    let (d, nh, nt, ns) = (o.d as i32, o.n_head as i32, o.n_tok as i32, o.n_seqs as i32);
    // The query, contiguous: rotated for a TurboQuant K (mode 1), else copied (0).
    let qr = at(pl.q);
    check(
        unsafe {
            legion_tq_transform(
                o.q as *const f32,
                o.q_nb[1] as i64,
                o.q_nb[2] as i64,
                o.q_nb[3] as i64,
                d,
                nh,
                nt,
                ns,
                i32::from(o.k_code >= 3),
                o.seed,
                rsd,
                qr,
                stream,
            )
        },
        "tq query",
    )?;
    let v_tq = o.v_code >= 3;
    // Where the attention writes: the output, or the rotated output for a TurboQuant V.
    let out = if v_tq { at(pl.o) } else { o.dst };
    let ints = at(pl.ints) as *mut i32;
    let tw = o.table_width * o.n_seqs;
    let mut a = TqAttn {
        q: qr,
        k: o.k,
        v: o.v,
        k_cell: o.k_cell as i64,
        v_cell: o.v_cell as i64,
        k_code: o.k_code,
        v_code: o.v_code,
        table: o.table,
        table_width: o.table_width as i32,
        n_past: o.n_past,
        d,
        n_head: nh,
        n_kv: o.n_kv as i32,
        n_tok: nt,
        n_seqs: ns,
        scale: o.scale,
        rsd,
        window: o.window as i32,
        ring: o.ring as i32,
        seed: o.seed,
        out,
        part: at(pl.part),
        max_split: pl.splits as i32,
        piece_blk: ints,
        piece_ent: unsafe { ints.add(tw) },
        piece_n: unsafe { ints.add(2 * tw) },
        piece_cnt: unsafe { ints.add(2 * tw + o.n_seqs) },
        src: unsafe { ints.add(2 * tw + o.n_seqs + 1) },
        cap: pl.cap as i32,
        piece: 0,
        k_stage: at(pl.stage) as *mut u16,
        v_stage: unsafe { (at(pl.stage) as *mut u16).add(pl.cap * KV_BLOCK * o.d * o.n_kv) },
    };
    let tables = turboquant::Tables::get();
    if pl.direct {
        check(unsafe { legion_tq_decode(&a, tables, stream) }, "tq decode")?;
    } else {
        // The staged pieces through the F16 prefill kernels (paged_attn.cu, unchanged).
        let kv_row = o.d * o.n_kv;
        let mut p = super::LaunchParams {
            q: qr,
            q_nb1: (o.d * 4) as i64,
            q_nb2: (o.d * o.n_head * 4) as i64,
            q_nb3: (o.d * o.n_head * o.n_tok * 4) as i64,
            k: a.k_stage,
            v: a.v_stage,
            kv_row: kv_row as i64,
            table: o.table,
            table_width: o.table_width as i32,
            n_past: o.n_past,
            dst: out,
            scratch: at(pl.part),
            max_split: 1,
            d,
            n_head: nh,
            n_head_kv: o.n_kv as i32,
            n_tok: nt,
            n_seqs: ns,
            scale: o.scale,
            window: o.window as i32,
            ring: 0,
            k_type: 0,
            v_type: 0,
            k_cell: (kv_row * 2) as i64,
            v_cell: (kv_row * 2) as i64,
            piece_blk: a.piece_blk,
            piece_ent: a.piece_ent,
            piece_n: a.piece_n,
            piece_cnt: a.piece_cnt,
            k_stage: a.k_stage as *const u8,
            v_stage: a.v_stage as *const u8,
            ml: at(pl.part),
            piece: 0,
            split: std::ptr::null_mut(),
            split_seqs: 0,
            compact: 0,
            head_stride: o.d as i64,
        };
        // An upper bound on the pieces: every table entry walked (idle pieces return early).
        let pieces = tw.div_ceil(pl.cap).max(1);
        for j in 0..pieces {
            a.piece = j as i32;
            p.piece = j as i32;
            check(unsafe { legion_tq_stage(&a, tables, stream) }, "tq stage")?;
            check(
                unsafe { super::legion_paged_attn_launch(&p, stream) },
                "tq prefill",
            )?;
        }
    }
    if v_tq {
        let f = 4i64;
        check(
            unsafe {
                legion_tq_transform(
                    out,
                    d as i64 * f,
                    (d * nh) as i64 * f,
                    (d * nh * nt) as i64 * f,
                    d,
                    nh,
                    nt,
                    ns,
                    2,
                    o.seed,
                    rsd,
                    o.dst,
                    stream,
                )
            },
            "tq output",
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
