//! The CUDA side of the fused expert op: the kernels' arguments, the order of their launches,
//! and the staging of host-resident experts into VRAM. The kernels themselves are in
//! `src/cuda/moe_gemm.cu`.
//!
//! Steps, all on the op's stream, with no host sync:
//! 1. `prep`: x (F32) to FP16 rows; each expert's routed (token, slot) pairs counted.
//! 2. `plan`: one block scans the counts into each expert's first pair and first tile.
//! 3. `scatter`: the pairs listed expert by expert.
//! 4. `gemm` gate/up: per (tile of up to [`TILE_PAIRS`] pairs of one expert, [`TILE_ROWS`]
//!    interleaved gate and up rows), SwiGLU in the epilogue, FP16 rows in list order.
//! 5. `gemm` down: those rows times down, each scaled by its router weight, stored at its pair.
//! 6. `reduce`: per token, the used experts' rows summed in slot order. On the GB10 the down
//!    GEMM can make these sums itself where its fragment kernel runs ([`super::set_down_sum`],
//!    opt-in).
//!
//! Experts in host memory are staged into VRAM before the op (see the parent module).

use std::os::raw::{c_int, c_uint, c_void};

use super::*;

type Stream = *mut c_void;

/// `legion_moe_args` of `src/cuda/moe_gemm.cu`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Args {
    up: *const u8,
    gate: *const u8,
    down: *const u8,
    gu_nb1: i64,
    gu_nb2: i64,
    down_nb1: i64,
    down_nb2: i64,
    table: *const i32,
    n_expert: c_int,
    n_used: c_int,
    n_embd: i64,
    n_ff: i64,
    n_tok: i64,
    x: *const f32,
    x_ld: i64,
    ids: *const i32,
    ids_ld: i64,
    w: *const f32,
    w_ld: i64,
    out: *mut f32,
    xh: *mut c_void,
    count: *mut i32,
    first: *mut i32,
    cursor: *mut i32,
    tile0: *mut i32,
    list: *mut i32,
    h: *mut c_void,
    d: *mut f32,
    tm: c_int,
    n_slots: c_int,
    d_fmt: c_int,
    sh: *const f32,
    sh_ld: i64,
    sg: *const f32,
    sg_ld: i64,
    e_lo: c_int,
    e_hi: c_int,
    limit: f32,
    q8: c_int,
    raw_off: c_int,
    sum_cnt: *mut i32,
    frag_ycap: c_int,
}

/// The sm_80+ fragment GEMMs' block-row cap (`frag_ycap`; 0: one block row per plan tile) for a
/// step of `pairs` (token, expert) pairs: a decode or verify step's grid sizes one partial tile per
/// expert it could touch (up to `n_expert`), but it touches far fewer, and the blocks past the
/// plan's tiles only return. Capped, each block row walks the tiles `y, y + cap, ...`. Prefill
/// steps (pairs at least four per expert) keep the full grid.
pub(super) fn frag_ycap(pairs: i64, n_expert: i64, max_tiles: i64) -> c_int {
    if pairs >= 4 * n_expert {
        return 0;
    }
    let cap = n_expert / 2 + pairs / TILE_PAIRS;
    if cap < max_tiles { cap as c_int } else { 0 }
}

unsafe extern "C" {
    fn legion_moe_prep(a: *const Args, stream: Stream) -> c_int;
    fn legion_moe_plan(a: *const Args, stream: Stream) -> c_int;
    fn legion_moe_scatter(a: *const Args, stream: Stream) -> c_int;
    fn legion_moe_gemm(
        a: *const Args,
        kind: c_int,
        wtype: c_int,
        bn: c_int,
        grid_x: c_uint,
        grid_y: c_uint,
        stream: Stream,
    ) -> c_int;
    fn legion_moe_gemm_glm(
        a: *const Args,
        kind: c_int,
        wtype: c_int,
        bn: c_int,
        grid_x: c_uint,
        grid_y: c_uint,
        stream: Stream,
    ) -> c_int;
    fn legion_moe_reduce(a: *const Args, stream: Stream) -> c_int;
    fn legion_moe_down_sum(a: *const Args, down_type: c_int) -> c_int;
    fn legion_moe_down_sum_init(a: *const Args, stream: Stream) -> c_int;
    fn legion_moe_mma16_supported() -> c_int;
    fn legion_moe_glm_q8(a: *const Args, gu_type: c_int, down_type: c_int) -> c_int;
}

/// Whether the current CUDA device runs the op's GEMMs on `mma.m16n8k16` (see
/// [`super::mma16`]).
pub(super) fn mma16_supported() -> bool {
    // SAFETY: a query of the current device, no arguments.
    unsafe { legion_moe_mma16_supported() == 1 }
}

/// A launcher's status: CUDA error codes, or -1 for a shape the kernels lack.
fn launched(what: &str, code: c_int) -> Result<(), String> {
    match code {
        0 => Ok(()),
        -1 => Err(format!("{what}: no kernel for this shape or type")),
        c => Err(format!("{what}: CUDA error {c}")),
    }
}

/// The kernels' assumptions about the operands, checked before launching: a violation is an
/// error here rather than a misaligned load or an out-of-bounds index on the GPU.
/// - `plan_kernel` scans at most 4096 experts; `prep_kernel` reads ids with 256 threads.
/// - `prep_kernel` reads X as `float2`: 8-aligned columns of an even length.
/// - Weight blocks are read in 16-byte pieces for K-quants (144/176-byte blocks), 8-byte ones
///   for Q5_1 (24 bytes) and IQ4_XS (136), 4-byte ones for Q2_K (84) and 2-byte ones for Q8_0 (34),
///   Q6_K (210) and the other IQ formats: the base, the row and the expert strides must keep that
///   alignment.
fn check(a: &Args, x: &ffi::ggml_tensor, weights: [&ffi::ggml_tensor; 3]) -> Result<(), String> {
    let align = |t: &ffi::ggml_tensor| match t.type_ {
        types::Q4_K | types::Q5_K => 16,
        types::Q5_1 | types::IQ4_XS => 8,
        types::Q2_K => 4,
        _ => 2,
    };
    let aligned = |t: &ffi::ggml_tensor| {
        let al = align(t);
        (t.data as usize).is_multiple_of(al)
            && t.nb[1].is_multiple_of(al)
            && t.nb[2].is_multiple_of(al)
    };
    let err = |what: &str| Err(format!("moe_ffn: {what}"));
    if !(1..=4096).contains(&a.n_expert) {
        return err(&format!(
            "{} experts (the plan kernel takes 1..=4096)",
            a.n_expert
        ));
    }
    if !(1..=256).contains(&a.n_used) {
        return err(&format!("{} experts per token (at most 256)", a.n_used));
    }
    if !(x.data as usize).is_multiple_of(8) || a.x_ld % 2 != 0 || a.n_embd % 2 != 0 {
        return err("activations not 8-byte aligned for float2 loads");
    }
    if !weights.iter().all(|w| aligned(w)) {
        return err("weights not aligned for the kernels' vector loads");
    }
    if a.n_slots < 1 {
        return err("no expert slots");
    }
    Ok(())
}

/// The op's CUDA implementation (see the module docs).
pub(super) unsafe fn launch(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `moe_ffn`; activations, routing and the node in device memory,
    // weights device-visible or staged.
    unsafe {
        let src = |i: usize| &*dst.src[i];
        let (x, ids, w, up, gate, down) = (src(0), src(1), src(2), src(3), src(4), src(5));
        let table = (dst.op_params[P_TABLE] != 0).then(|| src(6));
        // The shared expert and its gate logits after the table, if the op adds them.
        let shexp = shexp_srcs(dst);
        let n_expert = table.map_or(up.ne[2], |t| t.ne[0]);
        let (n_embd, n_ff, n, n_used) = (up.ne[0], up.ne[1], x.ne[1], ids.ne[0]);
        let sc = Scratch::new(n_expert, n_used, n_embd, n_ff, n);
        let out_floats = (n_embd * n) as usize;
        let base = (dst.data as *mut u8).add(out_floats.div_ceil(64) * 256);
        let mut a = Args {
            up: up.data as *const u8,
            gate: gate.data as *const u8,
            down: down.data as *const u8,
            gu_nb1: up.nb[1] as i64,
            gu_nb2: up.nb[2] as i64,
            down_nb1: down.nb[1] as i64,
            down_nb2: down.nb[2] as i64,
            table: table.map_or(std::ptr::null(), |t| t.data as *const i32),
            n_expert: n_expert as c_int,
            n_used: n_used as c_int,
            n_embd,
            n_ff,
            n_tok: n,
            x: x.data as *const f32,
            x_ld: (x.nb[1] / 4) as i64,
            ids: ids.data as *const i32,
            ids_ld: (ids.nb[1] / 4) as i64,
            w: w.data as *const f32,
            w_ld: (w.nb[1] / 4) as i64,
            out: dst.data as *mut f32,
            xh: base.add(sc.xh).cast(),
            count: base.add(sc.count).cast(),
            first: base.add(sc.first).cast(),
            cursor: base.add(sc.cursor).cast(),
            tile0: base.add(sc.tile0).cast(),
            list: base.add(sc.list).cast(),
            h: base.add(sc.h).cast(),
            d: base.add(sc.d).cast(),
            tm: TILE_PAIRS as c_int,
            n_slots: up.ne[2] as c_int,
            d_fmt: dst.op_params[P_PAIR] as c_int,
            sh: shexp.map_or(std::ptr::null(), |(sh, _)| sh.data as *const f32),
            sh_ld: shexp.map_or(0, |(sh, _)| (sh.nb[1] / 4) as i64),
            sg: shexp.map_or(std::ptr::null(), |(_, g)| g.data as *const f32),
            sg_ld: shexp.map_or(0, |(_, g)| (g.nb[1] / 4) as i64),
            e_lo: 0,
            e_hi: n_expert as c_int,
            limit: f32::from_bits(dst.op_params[P_CLAMP] as u32),
            q8: 0,
            raw_off: !raw_staging() as c_int,
            sum_cnt: std::ptr::null_mut(),
            frag_ycap: 0,
        };
        check(&a, x, [up, gate, down])?;
        // GLM-5.3-Flash's int8 GEMMs (opt-in, #603) where this step and the device take them;
        // prep then writes x as Q8 rows for them
        if dst.op_params[P_GLM] != 0 && dst.op_params[P_GLM_Q8] != 0 {
            a.q8 = legion_moe_glm_q8(&a, up.type_ as c_int, down.type_ as c_int);
        }
        let deferred = dst.op_params[P_SHEXP] & SUM_DEFERRED != 0;
        // The down GEMM's own per-token sums (#826) where its kernel takes them: their counters
        // in x's FP16 rows, which only the gate/up GEMM reads (they are set after it), so the op
        // needs no more scratch. GLM-5.3-Flash's nodes keep the reduce kernel.
        if down_sum() && !deferred && dst.op_params[P_GLM] == 0 {
            a.sum_cnt = a.xh.cast();
            if legion_moe_down_sum(&a, down.type_ as c_int) != 1 {
                a.sum_cnt = std::ptr::null_mut();
            }
        }
        crate::residency::cuda::zero_async(a.count.cast(), 4 * n_expert as usize, stream)?;
        launched("moe prep", legion_moe_prep(&a, stream))?;
        launched("moe plan", legion_moe_plan(&a, stream))?;
        launched("moe scatter", legion_moe_scatter(&a, stream))?;
        // Tiles: the full ones, plus at most one partial tile per expert.
        let pairs = n * n_used;
        let max_tiles = (pairs / TILE_PAIRS + pairs.min(n_expert)) as c_uint;
        a.frag_ycap = frag_ycap(pairs, n_expert, max_tiles as i64);
        let rows = |m: i64| (m / TILE_ROWS) as c_uint;
        // GLM-5.3-Flash's IQ2 kernels where the node allows them (they fall back to the shared
        // ones for other types and GPUs)
        let gemm = if dst.op_params[P_GLM] != 0 {
            legion_moe_gemm_glm
        } else {
            legion_moe_gemm
        };
        // The GEMMs over the experts [lo, hi) (their tiles only; the blocks past them return).
        let gemms = |lo: usize, hi: usize| {
            if lo >= hi {
                return Ok(());
            }
            let a = Args {
                e_lo: lo as c_int,
                e_hi: hi as c_int,
                ..a
            };
            launched(
                "moe gate/up",
                gemm(
                    &a,
                    0,
                    up.type_ as c_int,
                    TILE_ROWS as c_int,
                    rows(2 * n_ff),
                    max_tiles,
                    stream,
                ),
            )?;
            if !a.sum_cnt.is_null() {
                // legion_moe_down_sum took one range of all experts (the fragment kernels run
                // after the whole staging copy)
                launched("moe down-sum init", legion_moe_down_sum_init(&a, stream))?;
            }
            launched(
                "moe down",
                gemm(
                    &a,
                    1,
                    down.type_ as c_int,
                    TILE_ROWS as c_int,
                    rows(n_embd),
                    max_tiles,
                    stream,
                ),
            )
        };
        match dst.op_params[P_STAGE] {
            0 => gemms(0, n_expert as usize)?,
            // Staged weights still being copied: each range after its part of the copy. The
            // sm_80 fragment kernels walk experts their own way: one range, after all of it.
            id => crate::residency::staging::split_wait(
                id as usize,
                dst.op_params[P_STAGE_ROW] as usize,
                n_expert as usize,
                mma16_supported(),
                stream,
                gemms,
            )?,
        }
        if deferred || !a.sum_cnt.is_null() {
            // The reader sums the pairs itself (`defer_sum`), or the down GEMM did.
            return Ok(());
        }
        launched("moe reduce", legion_moe_reduce(&a, stream))
    }
}

/// The shared expert and its gate logits of a `moe_ffn` node, after the table in the sources,
/// if the op adds them.
///
/// # Safety
/// `dst` is a node made by `moe_ffn_with`.
unsafe fn shexp_srcs(dst: &ffi::ggml_tensor) -> Option<(&ffi::ggml_tensor, &ffi::ggml_tensor)> {
    // SAFETY: the node's sources as `moe_ffn_with` placed them.
    unsafe {
        (dst.op_params[P_SHEXP] & SHEXP_ADDED != 0).then(|| {
            let at = 6 + (dst.op_params[P_TABLE] != 0) as usize;
            (&*dst.src[at], &*dst.src[at + 1])
        })
    }
}

/// `legion_moe_pair_sum` of `src/cuda/fused.cu`: what a reader of a deferred sum
/// ([`super::defer_sum`]) needs to make it from the op's pairs as `reduce_kernel` does.
#[repr(C)]
pub(crate) struct PairSum {
    d: *const f32,
    ids: *const i32,
    ids_ld: i64,
    table: *const i32,
    n_expert: c_int,
    n_used: c_int,
    n_slots: c_int,
    sh: *const f32,
    sh_ld: i64,
    sg: *const f32,
    sg_ld: i64,
}

/// The [`PairSum`] of `node`, a `moe_ffn` node with F32 pair rows (device pointers).
///
/// # Safety
/// `node` is a node made by `moe_ffn_with`, its sources and data in device memory.
pub(crate) unsafe fn pair_sum(node: &ffi::ggml_tensor) -> PairSum {
    // SAFETY: the node's sources and scratch as `moe_ffn_with` and `launch` lay them out.
    unsafe {
        let src = |i: usize| &*node.src[i];
        let (x, ids, up) = (src(0), src(1), src(3));
        let table = (node.op_params[P_TABLE] != 0).then(|| src(6));
        let shexp = shexp_srcs(node);
        let n_expert = table.map_or(up.ne[2], |t| t.ne[0]);
        let (n_embd, n_ff, n, n_used) = (up.ne[0], up.ne[1], x.ne[1], ids.ne[0]);
        let sc = Scratch::new(n_expert, n_used, n_embd, n_ff, n);
        let base = (node.data as *const u8).add(((n_embd * n) as usize).div_ceil(64) * 256);
        PairSum {
            d: base.add(sc.d).cast(),
            ids: ids.data as *const i32,
            ids_ld: (ids.nb[1] / 4) as i64,
            table: table.map_or(std::ptr::null(), |t| t.data as *const i32),
            n_expert: n_expert as c_int,
            n_used: n_used as c_int,
            n_slots: up.ne[2] as c_int,
            sh: shexp.map_or(std::ptr::null(), |(sh, _)| sh.data as *const f32),
            sh_ld: shexp.map_or(0, |(sh, _)| (sh.nb[1] / 4) as i64),
            sg: shexp.map_or(std::ptr::null(), |(_, g)| g.data as *const f32),
            sg_ld: shexp.map_or(0, |(_, g)| (g.nb[1] / 4) as i64),
        }
    }
}
