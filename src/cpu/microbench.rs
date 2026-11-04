//! Exactness checks and micro-benchmarks of the host kernels, behind `legion cpu-bench`
//! (a development command, #637) and the ignored `cpu::bench` test.
//!
//! - [`check`]: every kernel this CPU supports against the scalar reference, bit for bit, on
//!   random and extreme block bytes, several shapes and token counts.
//! - [`kernels`]: one `rows x columns` matmul per format on one thread, each ISA against the
//!   scalar reference and ggml's CPU `vec_dot`.
//! - [`experts`]: one MoE expert (gate, up, down) at a time on one thread, cycling through more
//!   experts than the caches hold, each ISA against ggml.
//!
//! GB/s is packed weight bytes per second of one whole matmul (all tokens), so multi-token
//! rows show the reuse of each decoded weight block. ggml runs its own `vec_dot` per (row,
//! token) with activations in its `vec_dot_type` (Q8_K for the K-quants and IQ4_XS, Q8_1 for
//! Q5_1, Q8_0 for Q8_0).

use std::time::{Duration, Instant};

use super::*;
use crate::ggml::{self, ffi};

/// xorshift64*: deterministic data without a dependency.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next() >> 56) as u8).collect()
    }

    /// A random finite f16 (normal or subnormal, either sign).
    fn finite_f16(&mut self) -> u16 {
        let h = (self.next() >> 48) as u16;
        if h & 0x7c00 == 0x7c00 { h & !0x4000 } else { h }
    }

    /// Uniform in [-1, 1).
    pub fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

impl QuantFormat {
    /// The ggml type of this format's blocks.
    pub fn ggml_type(self) -> ffi::ggml_type {
        match self {
            Self::Q4K => ggml::types::Q4_K,
            Self::Q5K => ggml::types::Q5_K,
            Self::Q6K => ggml::types::Q6_K,
            Self::Q8_0 => ggml::types::Q8_0,
            Self::Q5_1 => ggml::types::Q5_1,
            Self::Iq4Xs => ffi::ggml_type_GGML_TYPE_IQ4_XS,
        }
    }

    /// Byte offsets of the f16 scale fields inside one block.
    fn f16_fields(self) -> &'static [usize] {
        match self {
            Self::Q8_0 | Self::Iq4Xs => &[0],
            Self::Q5_1 | Self::Q4K | Self::Q5K => &[0, 2],
            Self::Q6K => &[208],
        }
    }
}

/// Best-of time of `f` over at least `min_runs` runs and roughly `budget` wall time.
pub fn time(budget: Duration, min_runs: usize, mut f: impl FnMut()) -> Duration {
    let start = Instant::now();
    let mut best = Duration::MAX;
    let mut runs = 0;
    while runs < min_runs || start.elapsed() < budget {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed());
        runs += 1;
    }
    best
}

/// Packed weights or Q8_0 activations filled with `byte`, or random bytes when `None`; the f16
/// scale fields always finite (`one_scale`: 1.0, else random).
fn blocks(
    rng: &mut Rng,
    n: usize,
    block_bytes: usize,
    f16_fields: &[usize],
    byte: Option<u8>,
    one_scale: bool,
) -> Vec<u8> {
    let mut b = match byte {
        Some(v) => vec![v; n],
        None => rng.bytes(n),
    };
    for blk in b.chunks_exact_mut(block_bytes) {
        for &at in f16_fields {
            let h = if one_scale { 0x3c00 } else { rng.finite_f16() };
            blk[at..at + 2].copy_from_slice(&h.to_le_bytes());
        }
    }
    b
}

/// Checks every quantized kernel of every ISA this CPU supports against the scalar reference,
/// bit for bit: random block bytes and scales, all-equal extreme bytes (0x00, 0xff, 0x80,
/// 0x7f weights against -128, 127, -127, 0 activations), whole multiples of a kernel's block
/// step and remainders, 1 to 9 tokens. Returns the cases run, or the first mismatch.
pub fn check() -> Result<usize, String> {
    let mut rng = Rng::new(637);
    let mut cases = 0;
    let isas: Vec<Isa> = supported_isas()
        .into_iter()
        .filter(|&i| i != Isa::Scalar)
        .collect();
    for format in QuantFormat::ALL {
        let blocks_per_row: &[usize] = if format.block_len() == 32 {
            &[1, 3, 4, 5, 7, 8, 12, 13, 20, 24, 80]
        } else {
            &[1, 2, 3, 10]
        };
        for &nb in blocks_per_row {
            let columns = nb * format.block_len();
            let rb = format.row_bytes(columns).expect("whole blocks");
            for tokens in [1, 2, 3, 4, 5, 9] {
                let mut data: Vec<(Option<u8>, Option<u8>)> = vec![(None, None); 3];
                if nb <= 3 && tokens <= 3 {
                    for w in [0x00, 0xff, 0x80, 0x7f] {
                        for x in [0x80, 0x7f, 0x81, 0x00] {
                            data.push((Some(w), Some(x)));
                        }
                    }
                }
                for (wbyte, xbyte) in data {
                    let rows = 3;
                    let fields = format.f16_fields();
                    let w = blocks(
                        &mut rng,
                        rows * rb,
                        format.block_bytes(),
                        fields,
                        wbyte,
                        false,
                    );
                    let n = columns / GROUP * tokens * Q8_0_BLOCK_BYTES;
                    let packed =
                        blocks(&mut rng, n, Q8_0_BLOCK_BYTES, &[0], xbyte, xbyte.is_some());
                    let x = Q8Activations::from_q8_0(&packed, columns, tokens).expect("shape");
                    let mut want = vec![f32::NAN; rows * tokens];
                    matmul_q8_with_isa(Isa::Scalar, format, &w, &x, rows, &mut want)
                        .map_err(|e| e.to_string())?;
                    for &isa in &isas {
                        let mut got = vec![f32::NAN; rows * tokens];
                        matmul_q8_with_isa(isa, format, &w, &x, rows, &mut got)
                            .map_err(|e| e.to_string())?;
                        if got
                            .iter()
                            .zip(&want)
                            .any(|(a, b)| a.to_bits() != b.to_bits())
                        {
                            return Err(format!(
                                "{} {format:?}, {columns} columns, {tokens} tokens, weights \
                                 {wbyte:?}, activations {xbyte:?}: {got:?} != scalar {want:?}",
                                isa.name()
                            ));
                        }
                        cases += 1;
                    }
                }
            }
        }
    }
    Ok(cases)
}

/// A ggml `vec_dot` matmul of one format: activations converted to its `vec_dot_type` once.
struct GgmlMatmul {
    vec_dot: ffi::ggml_vec_dot_t,
    from_float: ffi::ggml_from_float_t,
    y_row: usize,
    w_row: usize,
}

impl GgmlMatmul {
    fn new(format: QuantFormat, columns: usize) -> Self {
        // SAFETY: static traits tables.
        let traits = unsafe { &*ffi::ggml_get_type_traits_cpu(format.ggml_type()) };
        let dot_traits = unsafe { &*ffi::ggml_get_type_traits_cpu(traits.vec_dot_type) };
        Self {
            vec_dot: traits.vec_dot,
            from_float: dot_traits.from_float,
            y_row: ggml::row_size(traits.vec_dot_type, columns),
            w_row: format.row_bytes(columns).expect("whole blocks"),
        }
    }

    /// `xf` (`tokens x columns`) in the `vec_dot_type`.
    fn convert(&self, xf: &[f32], columns: usize) -> Vec<u8> {
        let from_float = self.from_float.expect("from_float");
        let tokens = xf.len() / columns;
        let mut y = vec![0u8; self.y_row * tokens];
        for (src, dst) in xf.chunks_exact(columns).zip(y.chunks_exact_mut(self.y_row)) {
            // SAFETY: `dst` holds one row of `columns` values in the vec_dot type.
            unsafe { from_float(src.as_ptr(), dst.as_mut_ptr().cast(), columns as i64) };
        }
        y
    }

    /// `out[t * rows + r]` for the `rows` rows of `w` and the converted tokens of `y`.
    fn run(&self, w: &[u8], y: &[u8], columns: usize, out: &mut [f32]) {
        let vec_dot = self.vec_dot.expect("vec_dot");
        let rows = w.len() / self.w_row;
        for (t, yt) in y.chunks_exact(self.y_row).enumerate() {
            for r in 0..rows {
                // SAFETY: one weight row and one activation row of `columns` values each.
                unsafe {
                    vec_dot(
                        columns as i32,
                        &mut out[t * rows + r],
                        0,
                        w[r * self.w_row..].as_ptr().cast(),
                        0,
                        yt.as_ptr().cast(),
                        0,
                        1,
                    )
                };
            }
        }
    }
}

/// One `rows x columns` matmul per format and token count on this thread: every supported ISA
/// against the scalar reference and ggml. Calls `line` with each table line.
pub fn kernels(
    rows: usize,
    columns: usize,
    formats: &[QuantFormat],
    tokens: &[usize],
    budget: Duration,
    line: &mut dyn FnMut(String),
) {
    // SAFETY: initializes ggml's CPU tables; idempotent.
    unsafe { ffi::ggml_cpu_init() };
    let mut rng = Rng::new(42);
    line(format!(
        "host kernels, {rows}x{columns}, single thread, detected {:?}, supported {:?}",
        detected_isa(),
        supported_isas()
    ));
    line("format  tokens  path         ms        GB/s   vs scalar  vs ggml".into());
    for &format in formats {
        let wf: Vec<f32> = (0..rows * columns).map(|_| rng.unit()).collect();
        let w = ggml::quantize(format.ggml_type(), &wf, columns);
        let gb = w.len() as f64 / 1e9;
        let g = GgmlMatmul::new(format, columns);
        for &n in tokens {
            let xf: Vec<f32> = (0..n * columns).map(|_| rng.unit()).collect();
            let x = Q8Activations::quantize(&xf, columns, n).expect("shape");
            let y = g.convert(&xf, columns);
            let mut out = vec![0.0f32; rows * n];
            let ggml_t = time(budget, 3, || {
                g.run(&w, &y, columns, &mut out);
                std::hint::black_box(&out);
            });
            let mut scalar_t = Duration::ZERO;
            for isa in supported_isas() {
                let runs = if isa == Isa::Scalar { 1 } else { 3 };
                let t = time(budget, runs, || {
                    matmul_q8_with_isa(isa, format, &w, &x, rows, &mut out).expect("shape");
                    std::hint::black_box(&out);
                });
                if isa == Isa::Scalar {
                    scalar_t = t;
                }
                line(format!(
                    "{:<7} {n:>6}  {:<11} {:>8.3} {:>9.2} {:>9.2}x {:>7.2}x",
                    format!("{format:?}"),
                    isa.name(),
                    t.as_secs_f64() * 1e3,
                    gb / t.as_secs_f64(),
                    scalar_t.as_secs_f64() / t.as_secs_f64(),
                    ggml_t.as_secs_f64() / t.as_secs_f64(),
                ));
            }
            line(format!(
                "{:<7} {n:>6}  {:<11} {:>8.3} {:>9.2}",
                format!("{format:?}"),
                "ggml",
                ggml_t.as_secs_f64() * 1e3,
                gb / ggml_t.as_secs_f64(),
            ));
        }
    }
}

/// The shape and formats of one MoE expert: gate and up `n_ff x n_embd`, down `n_embd x n_ff`.
#[derive(Debug, Clone, Copy)]
pub struct ExpertShape {
    pub n_embd: usize,
    pub n_ff: usize,
    pub up_gate: QuantFormat,
    pub down: QuantFormat,
}

impl ExpertShape {
    /// Qwen3.8-Flash-Next's experts in the UD-Q4_K_XL GGUF (most layers).
    pub const QWEN_FLASH_NEXT: Self = Self {
        n_embd: 2560,
        n_ff: 640,
        up_gate: QuantFormat::Q4K,
        down: QuantFormat::Q5_1,
    };

    pub fn gate_bytes(&self) -> usize {
        self.n_ff * self.up_gate.row_bytes(self.n_embd).expect("whole blocks")
    }

    pub fn down_bytes(&self) -> usize {
        self.n_embd * self.down.row_bytes(self.n_ff).expect("whole blocks")
    }

    /// Bytes of one expert's gate, up and down weights.
    pub fn bytes(&self) -> usize {
        2 * self.gate_bytes() + self.down_bytes()
    }
}

/// `n_expert` experts of `shape` in three buffers (up, gate, down), quantized by ggml from
/// random values: four distinct experts repeated, so filling many is quick and every expert
/// still has its own addresses.
pub fn random_experts(shape: ExpertShape, n_expert: usize) -> [Vec<u8>; 3] {
    let mut rng = Rng::new(637);
    let distinct = n_expert.min(4);
    let mut kind = |format: QuantFormat, rows: usize, columns: usize| {
        let wf: Vec<f32> = (0..distinct * rows * columns)
            .map(|_| rng.unit() * 0.1)
            .collect();
        let q = ggml::quantize(format.ggml_type(), &wf, columns);
        let one = q.len() / distinct;
        let mut all = Vec::with_capacity(one * n_expert);
        for e in 0..n_expert {
            all.extend_from_slice(&q[(e % distinct) * one..][..one]);
        }
        all
    };
    let up = kind(shape.up_gate, shape.n_ff, shape.n_embd);
    let gate = kind(shape.up_gate, shape.n_ff, shape.n_embd);
    let down = kind(shape.down, shape.n_embd, shape.n_ff);
    [up, gate, down]
}

/// One expert at a time on this thread (gate and up, SwiGLU, the activations re-quantized,
/// down), cycling through `n_expert` experts so their weights come from memory: microseconds
/// per expert and GB/s for every supported ISA and ggml. Calls `line` with each table line.
pub fn experts(
    shape: ExpertShape,
    n_expert: usize,
    tokens: &[usize],
    budget: Duration,
    line: &mut dyn FnMut(String),
) {
    // SAFETY: initializes ggml's CPU tables; idempotent.
    unsafe { ffi::ggml_cpu_init() };
    let ExpertShape { n_embd, n_ff, .. } = shape;
    let [up, gate, down] = random_experts(shape, n_expert);
    let (gb, db) = (shape.gate_bytes(), shape.down_bytes());
    let mb = shape.bytes() as f64 / 1e6;
    line(format!(
        "one expert on one thread: {:?}/{:?}, {n_embd} x {n_ff}, {mb:.2} MB, {n_expert} experts cycled",
        shape.up_gate, shape.down
    ));
    line("tokens  path          us/expert    GB/s   vs ggml".into());
    let mut rng = Rng::new(7);
    let swiglu = |g: f32, u: f32| g / (1.0 + (-g).exp()) * u;
    for &n in tokens {
        let xf: Vec<f32> = (0..n * n_embd).map(|_| rng.unit()).collect();
        let (mut g, mut u, mut o) = (
            vec![0.0; n * n_ff],
            vec![0.0; n * n_ff],
            vec![0.0; n * n_embd],
        );
        let mut act = vec![0.0f32; n * n_ff];
        let per = |t: Duration| t.as_secs_f64() / n_expert as f64;
        let (gg, gd) = (
            GgmlMatmul::new(shape.up_gate, n_embd),
            GgmlMatmul::new(shape.down, n_ff),
        );
        let ggml_t = per(time(budget, 2, || {
            for e in 0..n_expert {
                let y = gg.convert(&xf, n_embd);
                gg.run(&gate[e * gb..][..gb], &y, n_embd, &mut g);
                gg.run(&up[e * gb..][..gb], &y, n_embd, &mut u);
                act.iter_mut()
                    .zip(g.iter().zip(&u))
                    .for_each(|(a, (&g, &u))| *a = swiglu(g, u));
                let ya = gd.convert(&act, n_ff);
                gd.run(&down[e * db..][..db], &ya, n_ff, &mut o);
                std::hint::black_box(&o);
            }
        }));
        for isa in supported_isas() {
            let t = per(time(budget, 2, || {
                for e in 0..n_expert {
                    let x = Q8Activations::quantize(&xf, n_embd, n).expect("shape");
                    let gw = &gate[e * gb..][..gb];
                    let uw = &up[e * gb..][..gb];
                    matmul_q8_with_isa(isa, shape.up_gate, gw, &x, n_ff, &mut g).expect("gate");
                    matmul_q8_with_isa(isa, shape.up_gate, uw, &x, n_ff, &mut u).expect("up");
                    act.iter_mut()
                        .zip(g.iter().zip(&u))
                        .for_each(|(a, (&g, &u))| *a = swiglu(g, u));
                    let a = Q8Activations::quantize(&act, n_ff, n).expect("shape");
                    let dw = &down[e * db..][..db];
                    matmul_q8_with_isa(isa, shape.down, dw, &a, n_embd, &mut o).expect("down");
                    std::hint::black_box(&o);
                }
            }));
            line(format!(
                "{n:>6}  {:<11} {:>11.1} {:>7.2} {:>8.2}x",
                isa.name(),
                t * 1e6,
                mb / 1e3 / t,
                ggml_t / t
            ));
        }
        line(format!(
            "{n:>6}  {:<11} {:>11.1} {:>7.2}",
            "ggml",
            ggml_t * 1e6,
            mb / 1e3 / ggml_t
        ));
    }
}

#[cfg(test)]
mod tests;
