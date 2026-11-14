//! `LEGION_DEBUG_EXPERT_DUMP` (docs/debug-env.md, #37): records of what the expert matvec
//! kernels read and wrote in real steps, for `legion expert-cpu-proof check` to recompute on the
//! host ([`crate::cpu::gpu_exact`]) and compare bit for bit.
//!
//! A record is one op of up to [`MAX_TOKENS`] tokens: its ids, the first [`PRINT`] bytes of each
//! pair's expert as the kernel addressed it (which names the layer and the expert offline,
//! whatever tier slot the id is), the Q8_1 activation blocks the kernels multiplied, and a
//! digest of each pair's output rows (the first records carry the rows themselves).
//!
//! A SwiGLU op's output is `silu(gate) * up`, which the host does not reproduce (the GPU's
//! exponential), so its record holds the raw gate and up sums instead: two more launches of the
//! same kernels, one matrix each, into scratch memory, and a third kernel that applies the same
//! inline SwiGLU to them; the record says per pair whether that equals the op's real output, so
//! the raw sums are the ones the op activated.
//!
//! While the variable is set, stage graphs do not capture expert ops (the hook synchronizes the
//! stream after each recorded op), and recorded steps are slower. Output does not change.

use std::io::{self, Read};
use std::sync::OnceLock;

/// Bytes of an expert kept per pair.
pub const PRINT: usize = 64;
/// Most tokens of a recorded op.
pub const MAX_TOKENS: usize = 8;

const MAGIC: u32 = u32::from_le_bytes(*b"EMVD");
const HEADER_WORDS: usize = 16;

/// The record's flags: the activations came quantized (`glu_tiered_preq`) ...
pub const F_PREQ: u32 = 1;
/// ... a verify step's row map follows the ids ...
pub const F_PAD: u32 = 2;
/// ... the output rows follow the digests ...
pub const F_FULL: u32 = 4;
/// ... a SwiGLU op (two matrices).
pub const F_GLU: u32 = 8;

/// What the variable asks for: `<file>[,every=<n>][,full=<n>][,max=<n>]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub path: String,
    /// Every `every`-th eligible op is recorded (1: all).
    pub every: u64,
    /// The first `full` records carry their output rows.
    pub full: u64,
    /// Records at most.
    pub max: u64,
}

impl Config {
    pub fn parse(v: &str) -> Option<Self> {
        let mut parts = v.split(',');
        let path = parts.next().filter(|p| !p.is_empty())?.to_string();
        let mut c = Self {
            path,
            every: 1,
            full: 0,
            max: u64::MAX,
        };
        for part in parts {
            let (key, n) = part.split_once('=')?;
            let n: u64 = n.parse().ok()?;
            match key {
                "every" if n > 0 => c.every = n,
                "full" => c.full = n,
                "max" => c.max = n,
                _ => return None,
            }
        }
        Some(c)
    }
}

fn config() -> Option<&'static Config> {
    static CONFIG: OnceLock<Option<Config>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let v = std::env::var("LEGION_DEBUG_EXPERT_DUMP").ok()?;
            let c = Config::parse(&v);
            if c.is_none() {
                eprintln!(
                    "LEGION_DEBUG_EXPERT_DUMP: expected <file>[,every=<n>][,full=<n>][,max=<n>]"
                );
            }
            c
        })
        .as_ref()
}

/// Whether expert ops are being recorded.
pub fn enabled() -> bool {
    config().is_some()
}

/// Two 64-bit digests of `bytes` (independent multiply-xorshift folds): equal digests stand for
/// equal bytes in the offline check.
pub fn digest(bytes: &[u8]) -> [u64; 2] {
    let fold = |seed: u64, mul: u64| {
        let mut h = seed ^ (bytes.len() as u64).wrapping_mul(mul);
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            let v = u64::from_le_bytes(c.try_into().expect("8 bytes"));
            h = (h ^ v).wrapping_mul(mul);
            h ^= h >> 29;
        }
        for &b in chunks.remainder() {
            h = (h ^ u64::from(b)).wrapping_mul(mul);
            h ^= h >> 31;
        }
        h = (h ^ (h >> 32)).wrapping_mul(mul);
        h ^ (h >> 29)
    };
    [
        fold(0x9e37_79b9_7f4a_7c15, 0xff51_afd7_ed55_8ccd),
        fold(0xc2b2_ae3d_27d4_eb4f, 0xc4ce_b9fe_1a85_ec53),
    ]
}

/// One recorded op.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Record {
    /// The op's number among the eligible ones.
    pub seq: u64,
    pub flags: u32,
    /// The weights' ggml type.
    pub ty: u32,
    pub k: usize,
    pub rows: usize,
    pub n_used: usize,
    pub n_tok: usize,
    /// Activation columns per token (1, or `n_used`: a column per pair).
    pub n_x: usize,
    /// The SwiGLU clamp's bits (0: none).
    pub limit: u32,
    /// Experts the stack holds, and how many of its first are VRAM slots.
    pub n_w: u32,
    pub vram_below: i32,
    /// `[n_tok][n_used]`, as the kernel read them (tier slots).
    pub ids: Vec<i32>,
    /// [`F_PAD`]: the row map (`[n_tok]`).
    pub pad: Vec<i32>,
    /// Per matrix (gate and up, or the one): `[pairs][PRINT]`.
    pub prints: Vec<Vec<u8>>,
    /// `[n_tok * n_x]` columns of `k / 32` `block_q8_1`.
    pub xq: Vec<u8>,
    /// Per matrix: each pair's digest of its `rows` f32 sums (a SwiGLU op: the raw gate and up
    /// sums; else the op's output).
    pub digests: Vec<Vec<[u64; 2]>>,
    /// A SwiGLU op, per pair: 1 if the SwiGLU of the raw sums is the op's output, bit for bit.
    pub act_equal: Vec<u8>,
    /// [`F_FULL`], per matrix: `[pairs][rows]` sums.
    pub full: Vec<Vec<f32>>,
}

impl Record {
    pub fn glu(&self) -> bool {
        self.flags & F_GLU != 0
    }

    pub fn pairs(&self) -> usize {
        self.n_used * self.n_tok
    }

    fn mats(&self) -> usize {
        if self.glu() { 2 } else { 1 }
    }

    /// Whether pair `(t, j)` is a dead padding pair of a verify step (its rows 0, not computed):
    /// `t` a padding row and no real row chose its expert.
    pub fn dead(&self, t: usize, j: usize) -> bool {
        if self.pad.is_empty() || self.pad[t] == t as i32 {
            return false;
        }
        let id = self.ids[t * self.n_used + j];
        !(0..self.n_tok)
            .filter(|&r| self.pad[r] == r as i32)
            .any(|r| self.ids[r * self.n_used..][..self.n_used].contains(&id))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut body = Vec::new();
        let i32s = |v: &[i32], out: &mut Vec<u8>| {
            v.iter()
                .for_each(|x| out.extend_from_slice(&x.to_le_bytes()));
        };
        i32s(&self.ids, &mut body);
        i32s(&self.pad, &mut body);
        self.prints.iter().for_each(|p| body.extend_from_slice(p));
        body.extend_from_slice(&self.xq);
        for d in self.digests.iter().flatten() {
            body.extend_from_slice(&d[0].to_le_bytes());
            body.extend_from_slice(&d[1].to_le_bytes());
        }
        body.extend_from_slice(&self.act_equal);
        for v in self.full.iter().flatten() {
            body.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        let words: [u32; HEADER_WORDS] = [
            MAGIC,
            self.flags,
            self.seq as u32,
            (self.seq >> 32) as u32,
            self.ty,
            self.k as u32,
            self.rows as u32,
            self.n_used as u32,
            self.n_tok as u32,
            self.n_x as u32,
            self.limit,
            self.n_w,
            self.vram_below as u32,
            0,
            0,
            body.len() as u32,
        ];
        let mut out = Vec::with_capacity(HEADER_WORDS * 4 + body.len());
        words
            .iter()
            .for_each(|w| out.extend_from_slice(&w.to_le_bytes()));
        out.extend_from_slice(&body);
        out
    }

    /// The next record of `r`; `None` at its end.
    pub fn read(r: &mut impl Read) -> io::Result<Option<Self>> {
        let bad = |what: &str| io::Error::new(io::ErrorKind::InvalidData, what.to_string());
        let mut head = [0u8; HEADER_WORDS * 4];
        match r.read(&mut head[..1])? {
            0 => return Ok(None),
            _ => r.read_exact(&mut head[1..])?,
        }
        let w: Vec<u32> = head
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect();
        if w[0] != MAGIC {
            return Err(bad("not an expert dump record"));
        }
        let mut rec = Self {
            flags: w[1],
            seq: u64::from(w[2]) | u64::from(w[3]) << 32,
            ty: w[4],
            k: w[5] as usize,
            rows: w[6] as usize,
            n_used: w[7] as usize,
            n_tok: w[8] as usize,
            n_x: w[9] as usize,
            limit: w[10],
            n_w: w[11],
            vram_below: w[12] as i32,
            ..Self::default()
        };
        let mut body = vec![0u8; w[15] as usize];
        r.read_exact(&mut body)?;
        let (pairs, mats) = (rec.pairs(), rec.mats());
        let n_pad = if rec.flags & F_PAD != 0 { rec.n_tok } else { 0 };
        let xq = rec.n_tok * rec.n_x * (rec.k / 32) * 36;
        let n_full = if rec.flags & F_FULL != 0 { mats } else { 0 };
        let want = pairs * 4
            + n_pad * 4
            + mats * pairs * PRINT
            + xq
            + mats * pairs * 16
            + if rec.glu() { pairs } else { 0 }
            + n_full * pairs * rec.rows * 4;
        if pairs == 0 || rec.n_tok > MAX_TOKENS || body.len() != want {
            return Err(bad("an expert dump record of the wrong size"));
        }
        let mut at = 0;
        let mut take = |n: usize| {
            at += n;
            &body[at - n..at]
        };
        let i32s = |b: &[u8]| -> Vec<i32> {
            b.chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().expect("4 bytes")))
                .collect()
        };
        rec.ids = i32s(take(pairs * 4));
        rec.pad = i32s(take(n_pad * 4));
        rec.prints = (0..mats).map(|_| take(pairs * PRINT).to_vec()).collect();
        rec.xq = take(xq).to_vec();
        rec.digests = (0..mats)
            .map(|_| {
                take(pairs * 16)
                    .chunks_exact(16)
                    .map(|c| {
                        [
                            u64::from_le_bytes(c[..8].try_into().expect("8 bytes")),
                            u64::from_le_bytes(c[8..].try_into().expect("8 bytes")),
                        ]
                    })
                    .collect()
            })
            .collect();
        rec.act_equal = take(if rec.glu() { pairs } else { 0 }).to_vec();
        rec.full = (0..n_full)
            .map(|_| {
                take(pairs * rec.rows * 4)
                    .chunks_exact(4)
                    .map(|c| f32::from_bits(u32::from_le_bytes(c.try_into().expect("4 bytes"))))
                    .collect()
            })
            .collect();
        Ok(Some(rec))
    }
}

#[cfg(test)]
mod tests;

#[cfg(feature = "cuda")]
pub(super) use hook::after;

#[cfg(feature = "cuda")]
mod hook {
    use std::fs::File;
    use std::io::{BufWriter, Write};
    use std::os::raw::{c_int, c_void};
    use std::sync::Mutex;
    use std::sync::atomic::Ordering;

    use super::super::{LaunchParams, Operands, legion_expert_mv_launch};
    use super::{Config, F_FULL, F_GLU, F_PAD, F_PREQ, MAX_TOKENS, PRINT, Record, digest};
    use crate::ggml::{ffi, types};

    unsafe extern "C" {
        fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> c_int;
        fn cudaFree(p: *mut c_void) -> c_int;
        fn cudaMemcpy(dst: *mut c_void, src: *const c_void, n: usize, kind: c_int) -> c_int;
        fn cudaStreamSynchronize(s: *mut c_void) -> c_int;
        fn legion_expert_mv_debug_peek(
            w: *const c_void,
            nb2: i64,
            ids: *const c_void,
            ids_nb1: i64,
            n_used: c_int,
            n_tok: c_int,
            n_w: c_int,
            out: *mut c_void,
            stream: *mut c_void,
        ) -> c_int;
        fn legion_expert_mv_debug_swiglu(
            g: *const f32,
            u: *const f32,
            out: *mut f32,
            n: c_int,
            limit: f32,
            stream: *mut c_void,
        ) -> c_int;
    }

    const DEVICE_TO_HOST: c_int = 2;

    struct State {
        cfg: &'static Config,
        file: BufWriter<File>,
        /// Eligible ops seen, records written, ops skipped (not eligible).
        seen: u64,
        written: u64,
        skipped: u64,
        /// Device scratch (an address) and its bytes.
        dev: usize,
        dev_bytes: usize,
        failed: bool,
    }

    fn state() -> Option<&'static Mutex<State>> {
        static STATE: std::sync::OnceLock<Option<Mutex<State>>> = std::sync::OnceLock::new();
        STATE
            .get_or_init(|| {
                let cfg = super::config()?;
                match File::create(&cfg.path) {
                    Ok(f) => Some(Mutex::new(State {
                        cfg,
                        file: BufWriter::new(f),
                        seen: 0,
                        written: 0,
                        skipped: 0,
                        dev: 0,
                        dev_bytes: 0,
                        failed: false,
                    })),
                    Err(e) => {
                        eprintln!("LEGION_DEBUG_EXPERT_DUMP: {}: {e}", cfg.path);
                        None
                    }
                }
            })
            .as_ref()
    }

    fn cuda(what: &str, e: c_int) -> Result<(), String> {
        match e {
            0 => Ok(()),
            e => Err(format!("{what}: CUDA error {e}")),
        }
    }

    /// `n` bytes of device memory at `src`.
    unsafe fn fetch(src: *const u8, n: usize) -> Result<Vec<u8>, String> {
        let mut v = vec![0u8; n];
        // SAFETY: the caller's `src` holds `n` device bytes; the stream was synchronized.
        let e = unsafe { cudaMemcpy(v.as_mut_ptr().cast(), src.cast(), n, DEVICE_TO_HOST) };
        cuda("copy to host", e).map(|()| v)
    }

    fn floats(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_bits(u32::from_le_bytes(c.try_into().expect("4 bytes"))))
            .collect()
    }

    /// Records the op `dst` just launched on `stream`, if it is one to record. Never fails the
    /// op: a problem is printed once and ends the dump.
    ///
    /// # Safety
    /// As `cuda_expert_mv`, right after its launches.
    pub(in super::super) unsafe fn after(dst: &ffi::ggml_tensor, stream: *mut c_void, preq: bool) {
        let Some(state) = state() else { return };
        let mut st = state.lock().unwrap_or_else(|e| e.into_inner());
        if st.failed || st.written >= st.cfg.max {
            return;
        }
        // SAFETY: the caller's node.
        let o = unsafe { Operands::of(dst) };
        let typed = [types::Q4_K, types::Q5_K, types::Q5_1, types::Q8_0].contains(&o.ty);
        let plain = o.halves == 0
            && o.map.is_null()
            && o.id_base == 0
            && o.ahead.is_none()
            && o.n_tok <= MAX_TOKENS
            && o.n_used > 0
            && o.ids_nb1 == o.n_used * 4
            && o.dst_nb[1] == o.rows * 4
            && o.dst_nb[2] == o.n_used * o.rows * 4;
        if !(typed && plain) {
            if st.skipped == 0 {
                eprintln!(
                    "expert dump: not recording ops like this one: {} k {} rows {}, {} x {} \
                     pairs, half {}, id base {}, map {}, staged ahead {}",
                    crate::ggml::type_name(o.ty),
                    o.k,
                    o.rows,
                    o.n_used,
                    o.n_tok,
                    o.halves,
                    o.id_base,
                    !o.map.is_null(),
                    o.ahead.is_some()
                );
            }
            st.skipped += 1;
            return;
        }
        st.seen += 1;
        if !(st.seen - 1).is_multiple_of(st.cfg.every) {
            return;
        }
        // SAFETY: as the caller promises.
        if let Err(e) = unsafe { record(&mut st, &o, stream, preq) } {
            eprintln!("LEGION_DEBUG_EXPERT_DUMP: stopped: {e}");
            st.failed = true;
        }
        if st.written.is_multiple_of(4096) || st.written >= st.cfg.max {
            let _ = st.file.flush();
            eprintln!(
                "expert dump: {} records of {} eligible ops, {} ops not eligible",
                st.written, st.seen, st.skipped
            );
        }
    }

    unsafe fn record(
        st: &mut State,
        o: &Operands,
        stream: *mut c_void,
        preq: bool,
    ) -> Result<(), String> {
        let glu = !o.w2.is_null();
        let pairs = o.n_used * o.n_tok;
        let out_bytes = pairs * o.rows * 4;
        // Device scratch: the experts' first bytes per matrix, then (SwiGLU) the raw gate and
        // up sums and their SwiGLU.
        let prints = (2 * pairs * PRINT).next_multiple_of(256);
        let need = prints + 3 * out_bytes;
        if st.dev_bytes < need {
            let mut p = std::ptr::null_mut();
            // SAFETY: plain allocations on the current device (the op's), kept for the process.
            unsafe {
                if st.dev != 0 {
                    cudaFree(st.dev as *mut c_void);
                }
                st.dev = 0;
                st.dev_bytes = 0;
                cuda("scratch", cudaMalloc(&mut p, need))?;
            }
            st.dev = p as usize;
            st.dev_bytes = need;
        }
        let dev = st.dev as *mut u8;
        // SAFETY: offsets inside the scratch.
        let at = |i: usize| unsafe { dev.add(prints + i * out_bytes) };
        let peek = |w: *const u8, m: usize| {
            // SAFETY: the op's ids and stack; `pairs * PRINT` bytes of scratch per matrix.
            let e = unsafe {
                legion_expert_mv_debug_peek(
                    w.cast(),
                    o.w_nb[2] as i64,
                    o.ids.cast(),
                    o.ids_nb1 as i64,
                    o.n_used as c_int,
                    o.n_tok as c_int,
                    o.n_w as c_int,
                    dev.add(m * pairs * PRINT).cast(),
                    stream,
                )
            };
            cuda("peek", e)
        };
        peek(o.w, 0)?;
        if glu {
            peek(o.w2, 1)?;
            // One matrix a launch, from the activations the op quantized, every pair in place.
            let rows_per_cta = super::super::rows_per_cta(o.w_nb[1], 1);
            let smem = (rows_per_cta * o.w_nb[1]).next_multiple_of(16);
            for (m, w) in [o.w, o.w2].into_iter().enumerate() {
                let p = LaunchParams {
                    w,
                    w2: std::ptr::null(),
                    w_nb1: o.w_nb[1] as i64,
                    w_nb2: o.w_nb[2] as i64,
                    x: o.x as *const f32,
                    x_nb1: o.x_nb[1] as i64,
                    x_nb2: o.x_nb[2] as i64,
                    n_x: o.n_x as i32,
                    ids: o.ids as *const i32,
                    ids_nb1: o.ids_nb1 as i64,
                    dst: at(m) as *mut f32,
                    dst_nb1: o.dst_nb[1] as i64,
                    dst_nb2: o.dst_nb[2] as i64,
                    k: o.k as i32,
                    rows: o.rows as i32,
                    n_used: o.n_used as i32,
                    n_tok: o.n_tok as i32,
                    rows_per_cta: rows_per_cta as i32,
                    xq: o.xq,
                    vram_below: o.vram_below,
                    vram_kernel: 0,
                    n_w: o.n_w as i32,
                    part: 0,
                    steps: 2,
                    stage: std::ptr::null(),
                    stage_hdr: 0,
                    stage_slot: 0,
                    stage_w: 0,
                    stage_w2: 0,
                    dot: super::super::DOT.load(Ordering::Relaxed) as i32,
                    vram_pf: 0,
                    limit: 0.0,
                    id_base: 0,
                    zero_skipped: 1,
                    pipe_tiles: super::super::pipe_tiles(o.ty, false, o.n_tok) as i32,
                    q4k_units: 0,
                    map: std::ptr::null(),
                    n_map: 0,
                    link_ctas: 0,
                    halves: 0,
                    q4k_hold: 0,
                    balance: i32::from(super::super::balance()),
                    half_tiles: i32::from(super::super::half_tiles_for(o.ty, false, o.n_tok)),
                    pad_rows: std::ptr::null(),
                };
                // SAFETY: the op's own operands (checked by its launch), one matrix, the scratch
                // as output.
                let e = unsafe { legion_expert_mv_launch(o.ty as i32, &p, smem as i32, stream) };
                cuda("raw launch", e)?;
            }
            // SAFETY: three arrays of `out_bytes` in the scratch.
            let e = unsafe {
                legion_expert_mv_debug_swiglu(
                    at(0) as *const f32,
                    at(1) as *const f32,
                    at(2) as *mut f32,
                    (out_bytes / 4) as c_int,
                    o.limit,
                    stream,
                )
            };
            cuda("swiglu", e)?;
        }
        // SAFETY: the op's stream; then synchronous copies of device memory the op's node and
        // the scratch hold.
        let (ids, pad, xq, print, outs) = unsafe {
            cuda("synchronize", cudaStreamSynchronize(stream))?;
            let ids = fetch(o.ids, pairs * 4)?;
            let pad = if o.pad.is_null() {
                Vec::new()
            } else {
                fetch(o.pad.cast(), o.n_tok * 4)?
            };
            let xq = fetch(o.xq, o.n_tok * o.n_x * (o.k / 32) * 36)?;
            let print = fetch(dev, 2 * pairs * PRINT)?;
            let mut outs = vec![fetch(o.dst, out_bytes)?];
            if glu {
                for m in 0..3 {
                    outs.push(fetch(at(m), out_bytes)?);
                }
            }
            (ids, pad, xq, print, outs)
        };
        let i32s = |b: &[u8]| -> Vec<i32> {
            b.chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().expect("4 bytes")))
                .collect()
        };
        let row = o.rows * 4;
        // The sums the record holds: the raw gate and up ones, or the op's output.
        let sums: &[Vec<u8>] = if glu { &outs[1..3] } else { &outs[..1] };
        let full = st.written < st.cfg.full;
        let rec = Record {
            seq: st.seen - 1,
            flags: if glu { F_GLU } else { 0 }
                | if preq { F_PREQ } else { 0 }
                | if pad.is_empty() { 0 } else { F_PAD }
                | if full { F_FULL } else { 0 },
            ty: o.ty as u32,
            k: o.k,
            rows: o.rows,
            n_used: o.n_used,
            n_tok: o.n_tok,
            n_x: o.n_x,
            limit: o.limit.to_bits(),
            n_w: o.n_w as u32,
            vram_below: o.vram_below,
            ids: i32s(&ids),
            pad: i32s(&pad),
            prints: (0..sums.len())
                .map(|m| print[m * pairs * PRINT..][..pairs * PRINT].to_vec())
                .collect(),
            xq,
            digests: sums
                .iter()
                .map(|s| s.chunks_exact(row).map(digest).collect())
                .collect(),
            act_equal: if glu {
                outs[0]
                    .chunks_exact(row)
                    .zip(outs[3].chunks_exact(row))
                    .map(|(a, b)| u8::from(a == b))
                    .collect()
            } else {
                Vec::new()
            },
            full: if full {
                sums.iter().map(|s| floats(s)).collect()
            } else {
                Vec::new()
            },
        };
        // Flushed per record: the process may be killed.
        st.file
            .write_all(&rec.to_bytes())
            .and_then(|()| st.file.flush())
            .map_err(|e| e.to_string())?;
        st.written += 1;
        Ok(())
    }
}
