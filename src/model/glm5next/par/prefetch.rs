//! `--glm-prefetch` (#747): each GPU copies the next layer's likely uncached experts into VRAM
//! while a layer computes.
//!
//! Under GLM's parallel decode a GPU reads the chosen experts its cache misses in place from
//! host memory, about 12 times the time of a cached one, and those reads bunch up inside each
//! layer. At MoE layer `L - 1` (any layer before a MoE layer `L`), [`Model::glm_prefetch`]
//! adds a node that, on a side stream of its own:
//! - applies layer `L`'s router (its weights and bias, F32) to layer `L - 1`'s FFN input, the
//!   input `L`'s router will see minus one sublayer's change (#745 measured 60-80% of the
//!   misses named), and takes the top `n_used` per token (`route_kernel` in
//!   src/cuda/glm_prefetch.cu);
//! - copies up to `cap` of the guessed experts this GPU owns and does not cache into its half
//!   of the device's staging ring ([`crate::residency::ep_prefetch::RingLayout`]), with
//!   `stage_copy`, the #253 staging copy.
//!
//! At layer `L`, [`match_staged`] waits for those copies and gives each of the step's real
//! pairs the slot its expert was staged in (`-1`: read in place as before); the expert matvec
//! reads staged experts from VRAM ([`crate::model::core::expert_mv::ahead`]): the same bytes,
//! the same code, the same bits. The guess only decides which experts are copied, never what
//! the step computes. With `--glm-pair-balance`, [`cached_with_staged`] counts each GPU's
//! staged experts as cached, so the balance leaves them with their owner; both GPUs of a pair
//! compute it from the same guess.
//!
//! The ring holds two halves (layer `L` in half `L % 2`): a layer's copies fill one while the
//! layer before it reads the other. Steps of more than
//! [`crate::residency::ep_prefetch::MAX_TOKENS`] tokens are not predicted.

// The parameters' indices and the layout's rebuild serve the CUDA ops only.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

#[cfg(test)]
mod tests;

#[cfg(feature = "cuda")]
use std::os::raw::{c_int, c_uint, c_void};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};
use crate::model::core::expert_mv::ahead::Ahead;
use crate::model::core::{Model, ModelError, Pass};
use crate::residency::ep_prefetch::{self, RingLayout};

/// What layer `L - 1`'s prefetch leaves for layer `L`.
#[derive(Clone, Copy)]
pub(crate) struct Staged<'a> {
    /// The device's staging ring and its layout.
    pub ring: Tensor<'a>,
    pub layout: RingLayout,
    /// The prefetch node (for the order of the graph).
    pub node: Tensor<'a>,
    /// The half the guess is in (0 or 1).
    pub half: usize,
    /// The step's tokens and experts per token.
    pub n: i64,
    pub n_used: i64,
    /// The FFN input's width.
    pub k: usize,
    /// A slot's offsets of the gate, up and down rows.
    pub off: [usize; 3],
}

// The prefetch node's parameters: the half, experts per token, this GPU's experts
// (`lo..hi`), the next layer's VRAM slots, the experts copied at most, a slot's bytes.
const P_HALF: usize = 0;
const P_USED: usize = 1;
const P_LO: usize = 2;
const P_HI: usize = 3;
const P_VRAM: usize = 4;
const P_CAP: usize = 5;
const P_SLOT: usize = 6;

/// [`match_staged`]'s and [`cached_with_staged`]'s parameters after [`P_HALF`]: the ring's
/// shape ([`RingLayout::new`]'s `cap`, slot bytes, `k`, `n_expert`), then theirs.
const Q_CAP: usize = 1;
const Q_SLOT: usize = 2;
const Q_K: usize = 3;
const Q_EXPERTS: usize = 4;
/// [`match_staged`]: the layer's VRAM slots. [`cached_with_staged`]: the pair's experts.
const Q_VRAM: usize = 5;
const Q_LO: usize = 5;
const Q_MID: usize = 6;
const Q_HI: usize = 7;

static PREDICT: ExternalOp = ExternalOp {
    name: "glm_prefetch",
    cpu: cpu_nothing,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_predict),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static MATCH: ExternalOp = ExternalOp {
    name: "glm_prefetch_match",
    cpu: cpu_nothing,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_match),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static CACHED: ExternalOp = ExternalOp {
    name: "glm_prefetch_cached",
    cpu: cpu_cached,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_cached),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

impl Model {
    /// At layer `il` of GPU `me`'s step over the FFN input `x` (`[n_embd, n]`): the prefetch
    /// of layer `il + 1`'s guessed uncached experts (see the module docs), expanded into the
    /// graph now, so that it runs beside the layer's own work. `None` (and nothing added) when
    /// `--glm-prefetch` is off, the next layer is not a tiered MoE layer of this GPU, the step
    /// is too long, or the router is not one F32 matrix.
    pub(super) fn glm_prefetch<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        me: usize,
        x: Tensor<'a>,
    ) -> Result<Option<Staged<'a>>, ModelError> {
        let plan = &self.glm_par.as_ref().expect("a parallel step").plan;
        let next = il + 1;
        let (n, n_used) = (x.ne()[1], self.hp.n_expert_used);
        if plan.prefetch == 0
            || next >= self.hp.n_layer
            || next < self.glm().n_dense
            || !ep_prefetch::predicts(n as usize, n_used as usize)
        {
            return Ok(None);
        }
        let Some(t) = self.expert_tier_on(next, me) else {
            return Ok(None);
        };
        let Some((ring, layout)) = t.prefetch_ring() else {
            return Ok(None);
        };
        if t.gathers(n) {
            return Ok(None);
        }
        let ctx = pass.ctx;
        let mirror = plan.mirror_sfx(next, me);
        let w = self.lw(next, &format!("ffn_gate_inp.weight{mirror}"))?;
        let bias = self.lw(next, &format!("exp_probs_b.bias{mirror}"))?;
        let n_expert = w.ne()[1];
        let [up, gate, down] = [t.weight(0), t.weight(1), t.weight(2)];
        let fits = w.ty() == types::F32
            && w.is_contiguous()
            && bias.ty() == types::F32
            && w.ne()[0] == x.ne()[0]
            && x.ne()[0] % 4 == 0
            && n_expert as usize <= 512
            && n_expert == bias.ne()[0]
            && (gate.nb()[2] + up.nb()[2] + down.nb()[2]) <= layout.slot;
        if !fits {
            return Ok(None);
        }
        let x = if x.is_contiguous() { x } else { ctx.cont(x) };
        let owned = plan.owned(me);
        let half = next % 2;
        let params = [
            half as i32,
            n_used as i32,
            owned.start as i32,
            owned.end as i32,
            t.n_slots() as i32,
            layout.cap as i32,
            layout.slot as i32,
        ];
        let node = ctx.external_with(
            &PREDICT,
            types::I32,
            [1, 1, 1, 1],
            &[x, w, bias, t.table_view(ctx), gate, up, down, ring],
            &params,
        );
        pass.graph.borrow_mut().expand(node);
        let (g, u) = (gate.nb()[2], up.nb()[2]);
        Ok(Some(Staged {
            ring,
            layout,
            node,
            half,
            n,
            n_used,
            k: x.ne()[0] as usize,
            off: [0, g, g + u],
        }))
    }
}

impl<'a> Staged<'a> {
    /// The ring's shape as the nodes after the prefetch pass it.
    fn shape(&self, n_expert: i64) -> [i32; 5] {
        [
            self.half as i32,
            self.layout.cap as i32,
            self.layout.slot as i32,
            self.k as i32,
            n_expert as i32,
        ]
    }

    /// Whether the step's choices `ids` are the ones this prefetch guessed for.
    pub fn takes(&self, ids: Tensor<'a>) -> bool {
        ids.ne()[0] == self.n_used && ids.ne()[1] == self.n
    }
}

/// The staged slots of the step's choices `ids` (this GPU's tiered ids, `[n_used, n]` I32, -1
/// for those it does not compute) on a layer of `vram_slots` cache slots and `n_expert`
/// experts, after waiting for `s`'s copies: where the expert matvec reads them.
pub(crate) fn match_staged<'a>(
    ctx: &'a Context,
    s: &Staged<'a>,
    ids: Tensor<'a>,
    vram_slots: usize,
    n_expert: i64,
) -> Ahead<'a> {
    let mut params = s.shape(n_expert).to_vec();
    params.push(vram_slots as i32);
    let ids = if ids.is_contiguous() {
        ids
    } else {
        ctx.cont(ids)
    };
    let after = ctx.external_with(
        &MATCH,
        types::I32,
        [1, 1, 1, 1],
        &[ids, s.ring, s.node],
        &params,
    );
    let half = s.half * s.layout.half;
    Ahead {
        ring: s.ring,
        after,
        table: half + s.layout.real,
        slots: half + s.layout.slots,
        slot: s.layout.slot,
        off: s.off,
    }
}

/// `--glm-pair-balance`'s cached experts (`cached`, I32 `[n_expert]`) plus those `s` stages on
/// their owner, of the GPU pair `lo..hi` whose second GPU starts at `mid`.
pub(crate) fn cached_with_staged<'a>(
    ctx: &'a Context,
    s: &Staged<'a>,
    cached: Tensor<'a>,
    (lo, mid, hi): (u32, u32, u32),
) -> Tensor<'a> {
    let n_expert = cached.ne()[0];
    let mut params = s.shape(n_expert).to_vec();
    params.extend([lo as i32, mid as i32, hi as i32]);
    let cached = if cached.is_contiguous() {
        cached
    } else {
        ctx.cont(cached)
    };
    ctx.external_with(
        &CACHED,
        types::I32,
        [n_expert, 1, 1, 1],
        &[cached, s.ring, s.node],
        &params,
    )
}

/// The ring's layout as a node after the prefetch gives it.
fn layout_of(p: &[i32]) -> RingLayout {
    RingLayout::new(
        p[Q_CAP] as usize,
        p[Q_SLOT] as usize,
        p[Q_K] as usize,
        p[Q_EXPERTS] as usize,
    )
}

/// On the CPU nothing is staged (no ring): the step reads every expert as before.
unsafe fn cpu_nothing(_dst: &ffi::ggml_tensor, _ith: usize, _nth: usize) {}

/// On the CPU the cached experts as they are (nothing is staged).
unsafe fn cpu_cached(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    // SAFETY: a node of `cached_with_staged`: contiguous I32 host source and dst of one value
    // per expert.
    unsafe {
        let n = dst.ne[0] as usize;
        let src = std::slice::from_raw_parts((*dst.src[0]).data as *const i32, n);
        std::slice::from_raw_parts_mut(dst.data as *mut i32, n).copy_from_slice(src);
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use crate::ggml::external_params;
    use crate::model::core::expert_mv::overlap::{self, StageParams};

    /// `dim3`.
    #[repr(C)]
    struct Dim3 {
        x: u32,
        y: u32,
        z: u32,
    }

    unsafe extern "C" {
        static legion_glm_prefetch_route_kernel: *const c_void;
        static legion_glm_prefetch_match_kernel: *const c_void;
        static legion_glm_prefetch_cached_kernel: *const c_void;
        fn cudaLaunchKernel(
            func: *const c_void,
            grid: Dim3,
            block: Dim3,
            args: *mut *mut c_void,
            shared: usize,
            stream: *mut c_void,
        ) -> i32;
        fn cudaGetDevice(d: *mut c_int) -> c_int;
        fn cudaStreamCreateWithFlags(s: *mut *mut c_void, flags: c_uint) -> c_int;
        fn cudaEventCreateWithFlags(e: *mut *mut c_void, flags: c_uint) -> c_int;
        fn cudaEventRecord(e: *mut c_void, s: *mut c_void) -> c_int;
        fn cudaStreamWaitEvent(s: *mut c_void, e: *mut c_void, flags: c_uint) -> c_int;
        fn cudaMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            n: usize,
            kind: c_int,
            s: *mut c_void,
        ) -> c_int;
    }

    /// `route_kernel`'s parameters (`Route` in src/cuda/glm_prefetch.cu).
    #[repr(C)]
    struct Route {
        x: *mut f32,
        w: *const f32,
        bias: *const f32,
        table: *const i32,
        logits: *mut f32,
        raw: *mut i32,
        tiered: *mut i32,
        done: *mut u32,
        k: i32,
        n_expert: i32,
        n_used: i32,
        n: i32,
        lo: i32,
        hi: i32,
    }

    /// `match_kernel`'s (`Match`).
    #[repr(C)]
    struct Match {
        ids: *const i32,
        ids_nb1: i64,
        n_used: i32,
        n_tok: i32,
        guess: *mut i32,
        slot: *mut i32,
        n_guess: i32,
        vram_below: i32,
        out: *mut i32,
    }

    /// `cached_kernel`'s (`Cached`).
    #[repr(C)]
    struct Cached {
        cached: *const i32,
        raw: *mut i32,
        n_guess: i32,
        n_expert: i32,
        lo: i32,
        mid: i32,
        hi: i32,
        cap: i32,
        out: *mut i32,
    }

    /// A device's side stream and its events: the fork, and per half the guess and the copies
    /// done.
    #[derive(Clone, Copy)]
    struct Side {
        dev: c_int,
        stream: *mut c_void,
        fork: *mut c_void,
        routed: [*mut c_void; 2],
        copied: [*mut c_void; 2],
    }

    // SAFETY: CUDA streams and events may be used from any host thread.
    unsafe impl Send for Side {}

    static SIDES: std::sync::Mutex<Vec<Side>> = std::sync::Mutex::new(Vec::new());

    fn rt(e: c_int, what: &str) -> Result<(), String> {
        match e {
            0 => Ok(()),
            e => Err(format!("glm prefetch: {what}: CUDA error {e}")),
        }
    }

    /// The current device's side stream, made on first use.
    fn side() -> Result<Side, String> {
        let mut dev = 0;
        // SAFETY: an out pointer.
        rt(unsafe { cudaGetDevice(&mut dev) }, "device")?;
        let mut sides = SIDES.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = sides.iter().find(|s| s.dev == dev) {
            return Ok(*s);
        }
        let event = || {
            let mut e = std::ptr::null_mut();
            // SAFETY: an out pointer; 2 = cudaEventDisableTiming.
            rt(unsafe { cudaEventCreateWithFlags(&mut e, 2) }, "event")?;
            Ok::<_, String>(e)
        };
        let mut stream = std::ptr::null_mut();
        // SAFETY: an out pointer; 1 = cudaStreamNonBlocking.
        rt(
            unsafe { cudaStreamCreateWithFlags(&mut stream, 1) },
            "stream",
        )?;
        let s = Side {
            dev,
            stream,
            fork: event()?,
            routed: [event()?, event()?],
            copied: [event()?, event()?],
        };
        sides.push(s);
        Ok(s)
    }

    /// Launches `kernel` with its one by-value parameter struct `arg`.
    ///
    /// # Safety
    /// `kernel` takes one parameter of type `T`, whose pointers are live device memory.
    unsafe fn launch<T>(
        kernel: *const c_void,
        (grid, block): (u32, u32),
        arg: &mut T,
        stream: *mut c_void,
        what: &str,
    ) -> Result<(), String> {
        let mut args = [(arg as *mut T).cast::<c_void>()];
        // SAFETY: as the caller promises; the launch copies the argument.
        rt(
            unsafe {
                cudaLaunchKernel(
                    kernel,
                    Dim3 {
                        x: grid,
                        y: 1,
                        z: 1,
                    },
                    Dim3 {
                        x: block,
                        y: 1,
                        z: 1,
                    },
                    args.as_mut_ptr(),
                    0,
                    stream,
                )
            },
            what,
        )
    }

    pub(super) unsafe fn cuda_predict(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let p = external_params(dst);
        // SAFETY: a node of `Model::glm_prefetch` on the stream's device: its sources are
        // allocated, the ring of the layout its parameters give.
        unsafe {
            let src = |i: usize| &*dst.src[i];
            let (x, w, bias, table) = (src(0), src(1), src(2), src(3));
            let mats = [src(4), src(5), src(6)];
            let ring = src(7);
            let (k, n, n_expert) = (x.ne[0], x.ne[1], bias.ne[0]);
            let l = RingLayout::new(
                p[P_CAP] as usize,
                p[P_SLOT] as usize,
                k as usize,
                n_expert as usize,
            );
            let h = p[P_HALF] as usize;
            let base = (ring.data as *mut u8).add(h * l.half);
            let s = side()?;
            // The last reads of this half (two layers back) are done; x's copy, then the side
            // stream goes on from here.
            rt(cudaStreamWaitEvent(stream, s.copied[h], 0), "wait")?;
            rt(
                cudaMemcpyAsync(
                    base.add(l.x).cast(),
                    x.data,
                    (k * n * 4) as usize,
                    3,
                    stream,
                ),
                "input copy",
            )?;
            rt(cudaEventRecord(s.fork, stream), "fork")?;
            rt(cudaStreamWaitEvent(s.stream, s.fork, 0), "fork")?;
            let n_used = p[P_USED];
            let mut r = Route {
                x: base.add(l.x).cast(),
                w: w.data as *const f32,
                bias: bias.data as *const f32,
                table: table.data as *const i32,
                logits: base.add(l.logits).cast(),
                raw: base.add(l.raw).cast(),
                tiered: base.add(l.guess).cast(),
                done: base.add(l.done).cast(),
                k: k as i32,
                n_expert: n_expert as i32,
                n_used,
                n: n as i32,
                lo: p[P_LO],
                hi: p[P_HI],
            };
            launch(
                legion_glm_prefetch_route_kernel,
                ((n_expert as u32).div_ceil(8), 256),
                &mut r,
                s.stream,
                "route",
            )?;
            rt(cudaEventRecord(s.routed[h], s.stream), "routed")?;
            // The guess's uncached experts of this GPU, in rank order, into the half's slots.
            let st = StageParams {
                ids: base.add(l.guess).cast::<i32>(),
                ids_nb1: n * 4,
                n_used: n as i32,
                n_tok: n_used,
                vram_below: p[P_VRAM],
                n_w: mats[0].ne[2] as i32,
                stage: base.add(l.copy_table),
                hdr: (l.slots - l.copy_table) as i64,
                slot: l.slot as i64,
                cap: l.cap as i32,
                table: 1,
                n_seg: 3,
                src: mats.map(|m| m.data as *const u8),
                nb2: mats.map(|m| m.nb[2] as i64),
                off: [
                    0,
                    mats[0].nb[2] as i64,
                    (mats[0].nb[2] + mats[1].nb[2]) as i64,
                ],
                ctas: overlap::CTAS as i32,
                t_every: 1,
                t_take: 1,
                c_every: 1,
                c_take: 1,
                c_flip: 0,
            };
            overlap::stage(&st, s.stream)?;
            rt(cudaEventRecord(s.copied[h], s.stream), "copied")
        }
    }

    pub(super) unsafe fn cuda_match(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let p = external_params(dst);
        // SAFETY: a node of `match_staged` after its prefetch node on this device.
        unsafe {
            let (ids, ring) = (&*dst.src[0], &*dst.src[1]);
            let l = layout_of(&p);
            let h = p[P_HALF] as usize;
            let base = (ring.data as *mut u8).add(h * l.half);
            let s = side()?;
            rt(cudaStreamWaitEvent(stream, s.copied[h], 0), "wait")?;
            let n_guess = (ids.ne[0] * ids.ne[1]) as i32;
            let mut m = Match {
                ids: ids.data as *const i32,
                ids_nb1: ids.nb[1] as i64,
                n_used: ids.ne[0] as i32,
                n_tok: ids.ne[1] as i32,
                guess: base.add(l.guess).cast(),
                slot: base.add(l.copy_table).cast(),
                n_guess,
                vram_below: p[Q_VRAM],
                out: base.add(l.real).cast(),
            };
            launch(
                legion_glm_prefetch_match_kernel,
                (1, 256),
                &mut m,
                stream,
                "match",
            )
        }
    }

    pub(super) unsafe fn cuda_cached(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let p = external_params(dst);
        // SAFETY: a node of `cached_with_staged` after its prefetch node on this device.
        unsafe {
            let (cached, ring, node) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
            let l = layout_of(&p);
            let h = p[P_HALF] as usize;
            let base = (ring.data as *mut u8).add(h * l.half);
            let s = side()?;
            rt(cudaStreamWaitEvent(stream, s.routed[h], 0), "wait")?;
            // The guess's pairs: the prefetch node's step (its input's tokens x n_used).
            let n_guess = ((*node.src[0]).ne[1] as i32) * external_params(node)[P_USED];
            let mut c = Cached {
                cached: cached.data as *const i32,
                raw: base.add(l.raw).cast(),
                n_guess,
                n_expert: cached.ne[0] as i32,
                lo: p[Q_LO],
                mid: p[Q_MID],
                hi: p[Q_HI],
                cap: l.cap as i32,
                out: dst.data as *mut i32,
            };
            launch(
                legion_glm_prefetch_cached_kernel,
                (1, 256),
                &mut c,
                stream,
                "cached",
            )
        }
    }
}

#[cfg(feature = "cuda")]
use cuda::{cuda_cached, cuda_match, cuda_predict};
