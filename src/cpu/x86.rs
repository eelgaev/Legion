//! x86-64 kernels: AVX2/FMA/F16C, AVX-VNNI and AVX-512 VNNI (plus BF16, and a 512-bit form),
//! chosen by `dispatch`.
//!
//! The quantized kernels are written once (`quant_kernels!`) and compiled per tier with that
//! tier's `#[target_feature]` set, so the per-tier integer-dot primitives inline. The kernels
//! unpack a superblock's eight 32-value groups into 256-bit vectors and `pack` them into the
//! tier's `Wq`; only the primitives that take a `Wq` differ between tiers:
//!
//! - `dots_us(w, x)` / `dots_signed(w, x)`: lane `g` = the exact dot of group `g` of unsigned
//!   (at most 128) or signed (never -128) weights with signed activations.
//! - `q6_dots(w, x, sc)`: the same for Q6_K, each 16-value half times its sub-scale.
//! - `q8_prep` / `q8_dots`: exact dots of eight Q8_0 blocks with arbitrary signed bytes (weights
//!   may hold -128, which the usual sign trick gets wrong).
//!
//! The 256-bit tiers (`ymm_dots!`) build those on `dot_us(u, s)`: eight i32 lanes, lane `k` = the
//! dot of bytes `4k..4k + 4` of unsigned `u` and signed `s`. Exact when every `u` is at most 128
//! (AVX2's `vpmaddubsw` sums byte pairs in saturating i16; `128 * 127 * 2` and `128 * -128 * 2`
//! still fit) and always exact with `vpdpbusd`. The 512-bit tier holds two groups per vector and
//! reduces four `vpdpbusd` results at once. The integer group dots are exact in any summation
//! order, so the tiers differ only in how they reach them.
//!
//! Signed weights (Q6_K, IQ4_XS) use the sign trick `dot_us(|x|, sign(w, x))`; it is exact
//! because those weights are never -128. Unsigned weights (Q4_K, Q5_K, Q5_1) are at most 31.
//! Every group's float term is computed lane-wise exactly as in `scalar`, so results match the
//! scalar reference bit for bit.
//!
//! The packing tricks (nibble unpacking, `bytes_from_bits_32`, the sign trick) are adapted from
//! ggml's MIT-licensed `ggml-cpu/arch/x86/quants.c`, pinned under `third_party/ggml`.

use std::arch::x86_64::*;

use super::dispatch::{Bf16Dot, Isa, Kernels};

fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
        && is_x86_feature_detected!("f16c")
}

fn has_avx_vnni() -> bool {
    has_avx2() && is_x86_feature_detected!("avxvnni")
}

fn has_avx512_vnni() -> bool {
    has_avx2()
        && is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vl")
        && is_x86_feature_detected!("avx512vnni")
}

fn has_avx512_bf16() -> bool {
    has_avx512_vnni() && is_x86_feature_detected!("avx512bf16")
}

/// The kernels of `isa`, after checking at run time that this CPU has every feature they use.
pub(super) fn kernels_for(isa: Isa) -> Option<Kernels> {
    // SAFETY (every arm): the `has_*` checks above detect exactly the features each set of
    // kernels is compiled for.
    match isa {
        Isa::Avx2 => has_avx2().then(|| unsafe { avx2::kernels(isa, avx2_bf16_dot) }),
        Isa::AvxVnni => has_avx_vnni().then(|| unsafe { avxvnni::kernels(isa, avx2_bf16_dot) }),
        Isa::Avx512Vnni => has_avx512_vnni().then(|| unsafe { vnni::kernels(isa, avx2_bf16_dot) }),
        Isa::Avx512Bf16 => {
            has_avx512_bf16().then(|| unsafe { vnni::kernels(isa, avx512_bf16_dot) })
        }
        Isa::Avx512Zmm => has_avx512_vnni().then(|| {
            let bf16_dot: Bf16Dot = if has_avx512_bf16() {
                avx512_bf16_dot
            } else {
                avx2_bf16_dot
            };
            unsafe { zmm::kernels(isa, bf16_dot) }
        }),
        _ => None,
    }
}

/// Safe entry points for the `#[target_feature]` kernels of one tier.
macro_rules! entry {
    ($($name:ident => $imp:ident),*) => {$(
        fn $name(w: &[u8], x: &Q8Activations, rows: usize, out: &mut [f32]) {
            // SAFETY: these functions are reachable only through the tier's `kernels`, whose
            // caller detected every feature the kernel is compiled for.
            unsafe { $imp(w, x, rows, out) }
        }
    )*};
}

macro_rules! quant_kernels {
    ($feat:literal) => {
        use super::super::dispatch::{Bf16Dot, Isa, Kernels};
        use super::super::{GROUP, Q8Activations, scalar};

        /// Tokens that share one decoded weight block (a verify step's k+1 tokens fit in two).
        const TB: usize = 4;
        const Q8_0: usize = 34;
        const Q5_1: usize = 24;

        /// The kernels of this tier, with `bf16_dot` for the BF16 path.
        ///
        /// # Safety
        /// The CPU must support every feature in this module's `target_feature` set.
        pub(super) unsafe fn kernels(isa: Isa, bf16_dot: Bf16Dot) -> Kernels {
            Kernels {
                isa,
                q8_0,
                q5_1,
                q4_k,
                q5_k,
                q6_k,
                iq4_xs,
                bf16_dot,
            }
        }

        entry!(q8_0 => q8_0_imp, q5_1 => q5_1_imp, q4_k => q4_k_imp, q5_k => q5_k_imp,
               q6_k => q6_k_imp, iq4_xs => iq4_xs_imp);

        #[target_feature(enable = $feat)]
        #[inline]
        fn load32(b: &[u8]) -> __m256i {
            let b = &b[..32];
            // SAFETY: `b` holds 32 readable bytes and `loadu` has no alignment requirement.
            unsafe { _mm256_loadu_si256(b.as_ptr().cast()) }
        }

        #[target_feature(enable = $feat)]
        #[inline]
        fn load16(b: &[u8]) -> __m128i {
            let b = &b[..16];
            // SAFETY: `b` holds 16 readable bytes; unaligned load.
            unsafe { _mm_loadu_si128(b.as_ptr().cast()) }
        }

        #[target_feature(enable = $feat)]
        #[inline]
        fn load_ps(v: &[f32]) -> __m256 {
            let v = &v[..8];
            // SAFETY: `v` holds 8 readable f32; unaligned load.
            unsafe { _mm256_loadu_ps(v.as_ptr()) }
        }

        #[target_feature(enable = $feat)]
        #[inline]
        fn load_epi32(v: &[i32]) -> __m256i {
            let v = &v[..8];
            // SAFETY: `v` holds 8 readable i32; unaligned load.
            unsafe { _mm256_loadu_si256(v.as_ptr().cast()) }
        }

        #[target_feature(enable = $feat)]
        #[inline]
        fn lanes(v: __m256) -> [f32; 8] {
            let mut a = [0.0f32; 8];
            // SAFETY: `a` has room for 8 f32; unaligned store.
            unsafe { _mm256_storeu_ps(a.as_mut_ptr(), v) };
            a
        }

        /// Eight f16 bit patterns to f32 (F16C converts exactly, as `scalar::half_to_f32`).
        #[target_feature(enable = $feat)]
        #[inline]
        fn f16x8(h: &[u16; 8]) -> __m256 {
            // SAFETY: `h` is 16 readable bytes; unaligned load.
            _mm256_cvtph_ps(unsafe { _mm_loadu_si128(h.as_ptr().cast()) })
        }

        /// The f16 at `b[0..2]`, converted exactly and broadcast.
        #[target_feature(enable = $feat)]
        #[inline]
        fn f16_splat(b: &[u8]) -> __m256 {
            let h = i32::from(u16::from_le_bytes([b[0], b[1]]));
            _mm256_broadcastss_ps(_mm_cvtph_ps(_mm_cvtsi32_si128(h)))
        }

        /// Byte `i` is 0xff where bit `i` of `bits` is set (ggml's `bytes_from_bits_32`).
        #[target_feature(enable = $feat)]
        #[inline]
        fn bytes_from_bits(bits: u32) -> __m256i {
            let t = _mm256_set1_epi32(bits as i32);
            let shuf = _mm256_setr_epi64x(
                0,
                0x0101_0101_0101_0101,
                0x0202_0202_0202_0202,
                0x0303_0303_0303_0303,
            );
            let b = _mm256_shuffle_epi8(t, shuf);
            let b = _mm256_or_si256(b, _mm256_set1_epi64x(0x7fbf_dfef_f7fb_fdfe));
            _mm256_cmpeq_epi8(b, _mm256_set1_epi64x(-1))
        }

        /// Writes the outputs of tokens `t0..t0 + acc.len()` for row `r`.
        #[target_feature(enable = $feat)]
        #[inline]
        fn store_out(acc: &[__m256], t0: usize, r: usize, rows: usize, out: &mut [f32]) {
            for (j, &a) in acc.iter().enumerate() {
                out[(t0 + j) * rows + r] = scalar::reduce8(lanes(a));
            }
        }

        #[target_feature(enable = $feat)]
        fn q8_0_imp(w: &[u8], x: &Q8Activations, rows: usize, out: &mut [f32]) {
            let (nb, tokens) = (x.columns() / GROUP, x.tokens());
            let full = nb / 8;
            for (r, row) in w.chunks_exact(nb * Q8_0).take(rows).enumerate() {
                for t0 in (0..tokens).step_by(TB) {
                    let tn = TB.min(tokens - t0);
                    let mut acc = [_mm256_setzero_ps(); TB];
                    for c in 0..full {
                        let mut dh = [0u16; 8];
                        let mut wq = [_mm256_setzero_si256(); 8];
                        for (b, blk) in row[c * 8 * Q8_0..][..8 * Q8_0].chunks_exact(Q8_0).enumerate()
                        {
                            dh[b] = u16::from_le_bytes([blk[0], blk[1]]);
                            wq[b] = q8_prep(load32(&blk[2..]));
                        }
                        let wq = pack(&wq);
                        let dw = f16x8(&dh);
                        for (j, a) in acc[..tn].iter_mut().enumerate() {
                            let t = t0 + j;
                            let xq = &x.qs(t)[c * 256..][..256];
                            let isum = q8_dots(&wq, xq, &x.sums(t)[c * 8..]);
                            let xd = load_ps(&x.scales(t)[c * 8..]);
                            let term = _mm256_mul_ps(_mm256_mul_ps(dw, xd), _mm256_cvtepi32_ps(isum));
                            *a = _mm256_add_ps(*a, term);
                        }
                    }
                    for (j, a) in acc[..tn].iter_mut().enumerate() {
                        let t = t0 + j;
                        let mut l = lanes(*a);
                        for b in full * 8..nb {
                            let blk = &row[b * Q8_0..][..Q8_0];
                            let xq = &x.qs(t)[GROUP * b..][..GROUP];
                            l[b % 8] += scalar::q8_0_block(blk, xq, x.scales(t)[b]);
                        }
                        *a = load_ps(&l);
                    }
                    store_out(&acc[..tn], t0, r, rows, out);
                }
            }
        }

        #[target_feature(enable = $feat)]
        fn q5_1_imp(w: &[u8], x: &Q8Activations, rows: usize, out: &mut [f32]) {
            let (nb, tokens) = (x.columns() / GROUP, x.tokens());
            let full = nb / 8;
            let m4 = _mm_set1_epi8(0x0f);
            let b4 = _mm256_set1_epi8(0x10);
            for (r, row) in w.chunks_exact(nb * Q5_1).take(rows).enumerate() {
                for t0 in (0..tokens).step_by(TB) {
                    let tn = TB.min(tokens - t0);
                    let mut acc = [_mm256_setzero_ps(); TB];
                    for c in 0..full {
                        let mut dh = [0u16; 8];
                        let mut mh = [0u16; 8];
                        let mut wq = [_mm256_setzero_si256(); 8];
                        for (b, blk) in row[c * 8 * Q5_1..][..8 * Q5_1].chunks_exact(Q5_1).enumerate()
                        {
                            dh[b] = u16::from_le_bytes([blk[0], blk[1]]);
                            mh[b] = u16::from_le_bytes([blk[2], blk[3]]);
                            let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
                            let v = load16(&blk[8..]);
                            let lo = _mm_and_si128(v, m4);
                            let hi = _mm_and_si128(_mm_srli_epi16::<4>(v), m4);
                            let high = _mm256_and_si256(bytes_from_bits(qh), b4);
                            wq[b] = _mm256_or_si256(_mm256_set_m128i(hi, lo), high);
                        }
                        let wq = pack(&wq);
                        let (d, m) = (f16x8(&dh), f16x8(&mh));
                        for (j, a) in acc[..tn].iter_mut().enumerate() {
                            let t = t0 + j;
                            let xq = &x.qs(t)[c * 256..][..256];
                            let dots = dots_us(&wq, xq);
                            let xd = load_ps(&x.scales(t)[c * 8..]);
                            let sums = _mm256_cvtepi32_ps(load_epi32(&x.sums(t)[c * 8..]));
                            let inner = _mm256_add_ps(
                                _mm256_mul_ps(d, _mm256_cvtepi32_ps(dots)),
                                _mm256_mul_ps(m, sums),
                            );
                            *a = _mm256_add_ps(*a, _mm256_mul_ps(xd, inner));
                        }
                    }
                    for (j, a) in acc[..tn].iter_mut().enumerate() {
                        let t = t0 + j;
                        let mut l = lanes(*a);
                        for b in full * 8..nb {
                            let blk = &row[b * Q5_1..][..Q5_1];
                            let xq = &x.qs(t)[GROUP * b..][..GROUP];
                            l[b % 8] += scalar::q5_1_block(blk, xq, x.scales(t)[b], x.sums(t)[b]);
                        }
                        *a = load_ps(&l);
                    }
                    store_out(&acc[..tn], t0, r, rows, out);
                }
            }
        }

        /// Q4_K and Q5_K: `HIGH` adds Q5_K's fifth bit from `qh`.
        #[target_feature(enable = $feat)]
        fn kmin_imp<const HIGH: bool>(w: &[u8], x: &Q8Activations, rows: usize, out: &mut [f32]) {
            let bb = if HIGH { 176 } else { 144 };
            let qs_at = if HIGH { 48 } else { 16 };
            let (nsb, tokens) = (x.columns() / 256, x.tokens());
            let m4 = _mm256_set1_epi8(0x0f);
            let b4 = _mm256_set1_epi8(0x10);
            for (r, row) in w.chunks_exact(nsb * bb).take(rows).enumerate() {
                for t0 in (0..tokens).step_by(TB) {
                    let tn = TB.min(tokens - t0);
                    let mut acc = [_mm256_setzero_ps(); TB];
                    for (sb, blk) in row.chunks_exact(bb).enumerate() {
                        let (d, dmin) = (f16_splat(blk), f16_splat(&blk[2..]));
                        let sm = load16(&scalar::k_scales_packed(&blk[4..16]));
                        // Scales and mins are 6-bit, so their f32 products with the group dots
                        // and sums (below 2^24) are exact: equal to the reference's i32 products.
                        let sc = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(sm));
                        let mn = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(sm)));
                        let qs = &blk[qs_at..qs_at + 128];
                        let mut wq = [_mm256_setzero_si256(); 8];
                        for p in 0..4 {
                            let v = load32(&qs[32 * p..]);
                            wq[2 * p] = _mm256_and_si256(v, m4);
                            wq[2 * p + 1] = _mm256_and_si256(_mm256_srli_epi16::<4>(v), m4);
                        }
                        if HIGH {
                            let qh = load32(&blk[16..48]);
                            for (g, q) in wq.iter_mut().enumerate() {
                                let bit = _mm256_set1_epi8((1u8 << g) as i8);
                                let set = _mm256_cmpeq_epi8(_mm256_and_si256(qh, bit), bit);
                                *q = _mm256_or_si256(*q, _mm256_and_si256(set, b4));
                            }
                        }
                        let wq = pack(&wq);
                        for (j, a) in acc[..tn].iter_mut().enumerate() {
                            let t = t0 + j;
                            let xq = &x.qs(t)[sb * 256..][..256];
                            let dots = dots_us(&wq, xq);
                            let xd = load_ps(&x.scales(t)[sb * 8..]);
                            let sums = _mm256_cvtepi32_ps(load_epi32(&x.sums(t)[sb * 8..]));
                            let pos = _mm256_mul_ps(d, _mm256_mul_ps(sc, _mm256_cvtepi32_ps(dots)));
                            let neg = _mm256_mul_ps(dmin, _mm256_mul_ps(mn, sums));
                            *a = _mm256_add_ps(*a, _mm256_mul_ps(xd, _mm256_sub_ps(pos, neg)));
                        }
                    }
                    store_out(&acc[..tn], t0, r, rows, out);
                }
            }
        }

        #[target_feature(enable = $feat)]
        fn q4_k_imp(w: &[u8], x: &Q8Activations, rows: usize, out: &mut [f32]) {
            kmin_imp::<false>(w, x, rows, out)
        }

        #[target_feature(enable = $feat)]
        fn q5_k_imp(w: &[u8], x: &Q8Activations, rows: usize, out: &mut [f32]) {
            kmin_imp::<true>(w, x, rows, out)
        }

        #[target_feature(enable = $feat)]
        fn q6_k_imp(w: &[u8], x: &Q8Activations, rows: usize, out: &mut [f32]) {
            let (nsb, tokens) = (x.columns() / 256, x.tokens());
            let m4 = _mm256_set1_epi8(0x0f);
            let m30 = _mm256_set1_epi8(0x30);
            let o32 = _mm256_set1_epi8(32);
            for (r, row) in w.chunks_exact(nsb * 210).take(rows).enumerate() {
                for t0 in (0..tokens).step_by(TB) {
                    let tn = TB.min(tokens - t0);
                    let mut acc = [_mm256_setzero_ps(); TB];
                    for (sb, blk) in row.chunks_exact(210).enumerate() {
                        let d = f16_splat(&blk[208..]);
                        let sc = q6_scales(load16(&blk[192..208]));
                        let mut wq = [_mm256_setzero_si256(); 8];
                        for n in 0..2 {
                            let a = load32(&blk[64 * n..]);
                            let b = load32(&blk[64 * n + 32..]);
                            let h = load32(&blk[128 + 32 * n..]);
                            let q = [
                                _mm256_or_si256(
                                    _mm256_and_si256(a, m4),
                                    _mm256_and_si256(_mm256_slli_epi16::<4>(h), m30),
                                ),
                                _mm256_or_si256(
                                    _mm256_and_si256(b, m4),
                                    _mm256_and_si256(_mm256_slli_epi16::<2>(h), m30),
                                ),
                                _mm256_or_si256(
                                    _mm256_and_si256(_mm256_srli_epi16::<4>(a), m4),
                                    _mm256_and_si256(h, m30),
                                ),
                                _mm256_or_si256(
                                    _mm256_and_si256(_mm256_srli_epi16::<4>(b), m4),
                                    _mm256_and_si256(_mm256_srli_epi16::<2>(h), m30),
                                ),
                            ];
                            for (k, v) in q.into_iter().enumerate() {
                                wq[4 * n + k] = _mm256_sub_epi8(v, o32);
                            }
                        }
                        let wq = pack(&wq);
                        for (j, a) in acc[..tn].iter_mut().enumerate() {
                            let t = t0 + j;
                            let xq = &x.qs(t)[sb * 256..][..256];
                            let isum = q6_dots(&wq, xq, &sc);
                            let xd = load_ps(&x.scales(t)[sb * 8..]);
                            let term = _mm256_mul_ps(_mm256_mul_ps(d, xd), _mm256_cvtepi32_ps(isum));
                            *a = _mm256_add_ps(*a, term);
                        }
                    }
                    store_out(&acc[..tn], t0, r, rows, out);
                }
            }
        }

        #[target_feature(enable = $feat)]
        fn iq4_xs_imp(w: &[u8], x: &Q8Activations, rows: usize, out: &mut [f32]) {
            let (nsb, tokens) = (x.columns() / 256, x.tokens());
            let m4 = _mm256_set1_epi8(0x0f);
            let table = scalar::IQ4_VALUES.map(|v| v as u8);
            let table = _mm256_broadcastsi128_si256(load16(&table));
            for (r, row) in w.chunks_exact(nsb * 136).take(rows).enumerate() {
                for t0 in (0..tokens).step_by(TB) {
                    let tn = TB.min(tokens - t0);
                    let mut acc = [_mm256_setzero_ps(); TB];
                    for (sb, blk) in row.chunks_exact(136).enumerate() {
                        let d = f16_splat(blk);
                        // `ls` is in -32..=31, so its f32 product with a group dot (below 2^24) is
                        // exact: equal to the reference's i32 product.
                        let ls = _mm256_cvtepi32_ps(load_epi32(&scalar::iq4_xs_scales(blk)));
                        let mut wq = [_mm256_setzero_si256(); 8];
                        for p in 0..4 {
                            let v = load32(&blk[8 + 32 * p..]);
                            let lo = _mm256_shuffle_epi8(table, _mm256_and_si256(v, m4));
                            let hi = _mm256_shuffle_epi8(
                                table,
                                _mm256_and_si256(_mm256_srli_epi16::<4>(v), m4),
                            );
                            wq[2 * p] = _mm256_permute2x128_si256::<0x20>(lo, hi);
                            wq[2 * p + 1] = _mm256_permute2x128_si256::<0x31>(lo, hi);
                        }
                        let wq = pack(&wq);
                        for (j, a) in acc[..tn].iter_mut().enumerate() {
                            let t = t0 + j;
                            let xq = &x.qs(t)[sb * 256..][..256];
                            let isum = _mm256_mul_ps(ls, _mm256_cvtepi32_ps(dots_signed(&wq, xq)));
                            let xd = load_ps(&x.scales(t)[sb * 8..]);
                            let term = _mm256_mul_ps(_mm256_mul_ps(d, xd), isum);
                            *a = _mm256_add_ps(*a, term);
                        }
                    }
                    store_out(&acc[..tn], t0, r, rows, out);
                }
            }
        }
    };
}

/// The `Wq` primitives of a 256-bit tier, built on its `dot_us`: one vector per group.
macro_rules! ymm_dots {
    ($feat:literal) => {
        /// A superblock's eight groups, one vector each.
        type Wq = [__m256i; 8];
        /// Q6_K sub-scales in the lane order of `hsum_halves`.
        type Q6Scales = (__m256i, __m256i);

        #[target_feature(enable = $feat)]
        #[inline]
        fn pack(wq: &[__m256i; 8]) -> Wq {
            *wq
        }

        #[target_feature(enable = $feat)]
        #[inline]
        fn load32i(b: &[i8]) -> __m256i {
            let b = &b[..32];
            // SAFETY: `b` holds 32 readable bytes and `loadu` has no alignment requirement.
            unsafe { _mm256_loadu_si256(b.as_ptr().cast()) }
        }

        /// Half-group sums: lanes `0..4` of the first result are the sums of lanes `0..4` (bytes
        /// `0..16`) of `v[0..4]`, lanes `4..8` the sums of their lanes `4..8`; the second result
        /// does the same for `v[4..8]`.
        #[target_feature(enable = $feat)]
        #[inline]
        fn hsum_halves(v: &[__m256i; 8]) -> (__m256i, __m256i) {
            let s01 = _mm256_hadd_epi32(v[0], v[1]);
            let s23 = _mm256_hadd_epi32(v[2], v[3]);
            let s45 = _mm256_hadd_epi32(v[4], v[5]);
            let s67 = _mm256_hadd_epi32(v[6], v[7]);
            (_mm256_hadd_epi32(s01, s23), _mm256_hadd_epi32(s45, s67))
        }

        /// Adds the two halves of every group from [`hsum_halves`]: lane `g` = group `g`.
        #[target_feature(enable = $feat)]
        #[inline]
        fn fold_halves(lo: __m256i, hi: __m256i) -> __m256i {
            _mm256_add_epi32(
                _mm256_permute2x128_si256::<0x20>(lo, hi),
                _mm256_permute2x128_si256::<0x31>(lo, hi),
            )
        }

        /// Lane `g` of the result is the sum of the eight lanes of `v[g]`.
        #[target_feature(enable = $feat)]
        #[inline]
        fn hsum8(v: &[__m256i; 8]) -> __m256i {
            let (lo, hi) = hsum_halves(v);
            fold_halves(lo, hi)
        }

        /// Sign-trick dot of signed weights `w` (never -128) and activations `x`.
        #[target_feature(enable = $feat)]
        #[inline]
        fn dot_signed(w: __m256i, x: __m256i) -> __m256i {
            dot_us(_mm256_abs_epi8(x), _mm256_sign_epi8(w, x))
        }

        #[target_feature(enable = $feat)]
        #[inline]
        fn dots_us(w: &Wq, x: &[i8]) -> __m256i {
            let mut d = [_mm256_setzero_si256(); 8];
            for (g, d) in d.iter_mut().enumerate() {
                *d = dot_us(w[g], load32i(&x[32 * g..]));
            }
            hsum8(&d)
        }

        #[target_feature(enable = $feat)]
        #[inline]
        fn dots_signed(w: &Wq, x: &[i8]) -> __m256i {
            let mut d = [_mm256_setzero_si256(); 8];
            for (g, d) in d.iter_mut().enumerate() {
                *d = dot_signed(w[g], load32i(&x[32 * g..]));
            }
            hsum8(&d)
        }

        /// Sub-scale `2g + h` scales half `h` of group `g`; reorder them to the lane order of
        /// `hsum_halves`: [0 2 4 6 1 3 5 7] and [8 10 .. 13 15].
        #[target_feature(enable = $feat)]
        #[inline]
        fn q6_scales(sc: __m128i) -> Q6Scales {
            let order = _mm_setr_epi8(0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15);
            let sc = _mm_shuffle_epi8(sc, order);
            (
                _mm256_cvtepi8_epi32(sc),
                _mm256_cvtepi8_epi32(_mm_srli_si128::<8>(sc)),
            )
        }

        #[target_feature(enable = $feat)]
        #[inline]
        fn q6_dots(w: &Wq, x: &[i8], sc: &Q6Scales) -> __m256i {
            let mut d = [_mm256_setzero_si256(); 8];
            for (g, d) in d.iter_mut().enumerate() {
                *d = dot_signed(w[g], load32i(&x[32 * g..]));
            }
            let (lo, hi) = hsum_halves(&d);
            fold_halves(_mm256_mullo_epi32(lo, sc.0), _mm256_mullo_epi32(hi, sc.1))
        }
    };
}

/// AVX2 + FMA + F16C: `vpmaddubsw` + `vpmaddwd` integer dots.
mod avx2 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    fn dot_us(u: __m256i, s: __m256i) -> __m256i {
        _mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1))
    }

    /// Q8_0 weights are used as loaded.
    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    fn q8_prep(w: __m256i) -> __m256i {
        w
    }

    /// The exact dots of eight Q8_0 weight blocks with `x` (lane `b` = block `b`): bytes widened
    /// to i16 and `vpmaddwd`, since `vpmaddubsw` would saturate or mishandle -128.
    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    fn q8_dots(w: &Wq, x: &[i8], _sums: &[i32]) -> __m256i {
        let mut d = [_mm256_setzero_si256(); 8];
        for (b, d) in d.iter_mut().enumerate() {
            let xv = load32i(&x[32 * b..]);
            let a0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w[b]));
            let a1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(w[b]));
            let b0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(xv));
            let b1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(xv));
            *d = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), _mm256_madd_epi16(a1, b1));
        }
        hsum8(&d)
    }

    ymm_dots!("avx2,fma,f16c");
    quant_kernels!("avx2,fma,f16c");
}

/// AVX2 + AVX-VNNI: VEX `vpdpbusd` on 256-bit vectors, Q8_0 as on the AVX-512 VNNI tier.
mod avxvnni {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma,f16c,avxvnni")]
    #[inline]
    fn dot_us(u: __m256i, s: __m256i) -> __m256i {
        _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), u, s)
    }

    /// Q8_0 weights are biased to `w + 128` (unsigned) once per block.
    #[target_feature(enable = "avx2,fma,f16c,avxvnni")]
    #[inline]
    fn q8_prep(w: __m256i) -> __m256i {
        _mm256_xor_si256(w, _mm256_set1_epi8(i8::MIN))
    }

    /// As `vnni::q8_dots`: `(w + 128) . x - 128 * sum(x)`.
    #[target_feature(enable = "avx2,fma,f16c,avxvnni")]
    #[inline]
    fn q8_dots(w: &Wq, x: &[i8], sums: &[i32]) -> __m256i {
        _mm256_sub_epi32(dots_us(w, x), _mm256_slli_epi32::<7>(load_epi32(sums)))
    }

    ymm_dots!("avx2,fma,f16c,avxvnni");
    quant_kernels!("avx2,fma,f16c,avxvnni");
}

/// AVX-512 F/BW/VL + VNNI: `vpdpbusd` on 256-bit vectors (Zen 4 runs 512-bit ops on 256-bit
/// datapaths, so wider vectors are not assumed to pay off).
mod vnni {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")]
    #[inline]
    fn dot_us(u: __m256i, s: __m256i) -> __m256i {
        _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s)
    }

    /// Q8_0 weights are biased to `w + 128` (unsigned) once per block.
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")]
    #[inline]
    fn q8_prep(w: __m256i) -> __m256i {
        _mm256_xor_si256(w, _mm256_set1_epi8(i8::MIN))
    }

    /// The exact dots of eight biased Q8_0 weight blocks with `x`: `(w + 128) . x - 128 * sum(x)`,
    /// using the precomputed group sums. `vpdpbusd` never saturates, so any bytes are exact.
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")]
    #[inline]
    fn q8_dots(w: &Wq, x: &[i8], sums: &[i32]) -> __m256i {
        _mm256_sub_epi32(dots_us(w, x), _mm256_slli_epi32::<7>(load_epi32(sums)))
    }

    ymm_dots!("avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni");
    quant_kernels!("avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni");
}

/// AVX-512 VNNI on 512-bit vectors: vector `k` of a `Wq` holds groups `2k` (lanes `0..8`) and
/// `2k + 1` (lanes `8..16`), so a superblock takes four `vpdpbusd` instead of eight.
mod zmm {
    use std::arch::x86_64::*;

    /// A superblock's eight groups, two per vector.
    type Wq = [__m512i; 4];
    /// Q6_K sub-scales, each repeated over the four i32 lanes of its half group.
    type Q6Scales = [__m512i; 4];

    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")]
    #[inline]
    fn pack(wq: &[__m256i; 8]) -> Wq {
        std::array::from_fn(|k| {
            _mm512_inserti64x4::<1>(_mm512_castsi256_si512(wq[2 * k]), wq[2 * k + 1])
        })
    }

    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")]
    #[inline]
    fn load64i(b: &[i8]) -> __m512i {
        let b = &b[..64];
        // SAFETY: `b` holds 64 readable bytes and `loadu` has no alignment requirement.
        unsafe { _mm512_loadu_si512(b.as_ptr().cast()) }
    }

    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")]
    #[inline]
    fn dot_us(u: __m512i, s: __m512i) -> __m512i {
        _mm512_dpbusd_epi32(_mm512_setzero_si512(), u, s)
    }

    /// Sign-trick dot of signed weights `w` (never -128) and activations `x`. AVX-512 has no
    /// `vpsignb`: negate `w` where `x` is negative and zero it where `x` is zero.
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")]
    #[inline]
    fn dot_signed(w: __m512i, x: __m512i) -> __m512i {
        let zero = _mm512_setzero_si512();
        let neg = _mm512_mask_sub_epi8(w, _mm512_movepi8_mask(x), zero, w);
        let sw = _mm512_maskz_mov_epi8(_mm512_test_epi8_mask(x, x), neg);
        dot_us(_mm512_abs_epi8(x), sw)
    }

    /// Lane `g` of the result is the sum of the eight i32 lanes of group `g` in `v` (laid out
    /// as in `Wq`).
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")]
    #[inline]
    fn hsum_pairs(v: &[__m512i; 4]) -> __m256i {
        // 128-bit lane `j` of `v[k]` is the lo (j even) or hi (j odd) half of group `2k + j / 2`;
        // adding the halves leaves lane `j` of `s[i]` = the four partial sums of group `4i + j`.
        let fold = |a: __m512i, b: __m512i| {
            _mm512_add_epi32(
                _mm512_shuffle_i32x4::<0b10_00_10_00>(a, b),
                _mm512_shuffle_i32x4::<0b11_01_11_01>(a, b),
            )
        };
        let (s0, s1) = (fold(v[0], v[1]), fold(v[2], v[3]));
        // Per 128-bit lane `j`: [a0 + a2, b0 + b2, a1 + a3, b1 + b3] with `a` = group `j` and
        // `b` = group `j + 4`, then both 64-bit halves hold [group j, group j + 4].
        let w = _mm512_add_epi32(_mm512_unpacklo_epi32(s0, s1), _mm512_unpackhi_epi32(s0, s1));
        let w = _mm512_add_epi32(w, _mm512_shuffle_epi32::<0b01_00_11_10>(w));
        let order = _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 0, 0, 0, 0, 0, 0, 0, 0);
        _mm512_castsi512_si256(_mm512_permutexvar_epi32(order, w))
    }

    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")]
    #[inline]
    fn dots_us(w: &Wq, x: &[i8]) -> __m256i {
        hsum_pairs(&std::array::from_fn(|k| {
            dot_us(w[k], load64i(&x[64 * k..]))
        }))
    }

    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")]
    #[inline]
    fn dots_signed(w: &Wq, x: &[i8]) -> __m256i {
        hsum_pairs(&std::array::from_fn(|k| {
            dot_signed(w[k], load64i(&x[64 * k..]))
        }))
    }

    /// Lane `l` of vector `k` covers half `(l % 8) / 4` of group `2k + l / 8`: sub-scale
    /// `4k + l / 4`.
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")]
    #[inline]
    fn q6_scales(sc: __m128i) -> Q6Scales {
        let sc = _mm512_cvtepi8_epi32(sc);
        std::array::from_fn(|k| {
            let i = 4 * k as i32;
            let (a, b, c, d) = (i, i + 1, i + 2, i + 3);
            let order = _mm512_setr_epi32(a, a, a, a, b, b, b, b, c, c, c, c, d, d, d, d);
            _mm512_permutexvar_epi32(order, sc)
        })
    }

    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")]
    #[inline]
    fn q6_dots(w: &Wq, x: &[i8], sc: &Q6Scales) -> __m256i {
        hsum_pairs(&std::array::from_fn(|k| {
            _mm512_mullo_epi32(dot_signed(w[k], load64i(&x[64 * k..])), sc[k])
        }))
    }

    /// Q8_0 weights are biased to `w + 128` (unsigned) once per block.
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")]
    #[inline]
    fn q8_prep(w: __m256i) -> __m256i {
        _mm256_xor_si256(w, _mm256_set1_epi8(i8::MIN))
    }

    /// As `vnni::q8_dots`: `(w + 128) . x - 128 * sum(x)`.
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")]
    #[inline]
    fn q8_dots(w: &Wq, x: &[i8], sums: &[i32]) -> __m256i {
        _mm256_sub_epi32(dots_us(w, x), _mm256_slli_epi32::<7>(load_epi32(sums)))
    }

    quant_kernels!("avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni");
}

fn avx2_bf16_dot(a: &[u16], b: &[u16]) -> f32 {
    assert_eq!(a.len(), b.len());
    // SAFETY: this function is only placed in `Kernels` by `kernels_for` after `has_avx2`.
    unsafe { avx2_bf16_dot_imp(a, b) }
}

/// Eight BF16 values widened to f32 (exact: BF16 is the top half of an f32).
#[target_feature(enable = "avx2,fma")]
#[inline]
fn bf16x8(v: &[u16; 8]) -> __m256 {
    // SAFETY: `v` is 16 readable bytes; unaligned load.
    let h = unsafe { _mm_loadu_si128(v.as_ptr().cast()) };
    _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(h)))
}

/// Four independent accumulators hide the FMA latency.
#[target_feature(enable = "avx2,fma")]
fn avx2_bf16_dot_imp(a: &[u16], b: &[u16]) -> f32 {
    let mut acc = [_mm256_setzero_ps(); 4];
    let ((a8, a_tail), (b8, b_tail)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    let ((a4, a_rest), (b4, b_rest)) = (a8.as_chunks::<4>(), b8.as_chunks::<4>());
    for (x, y) in a4.iter().zip(b4) {
        for (k, acc) in acc.iter_mut().enumerate() {
            *acc = _mm256_fmadd_ps(bf16x8(&x[k]), bf16x8(&y[k]), *acc);
        }
    }
    for (x, y) in a_rest.iter().zip(b_rest) {
        acc[0] = _mm256_fmadd_ps(bf16x8(x), bf16x8(y), acc[0]);
    }
    let sum = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
    let mut l = [0.0f32; 8];
    // SAFETY: `l` has room for 8 f32; unaligned store.
    unsafe { _mm256_storeu_ps(l.as_mut_ptr(), sum) };
    l.into_iter().sum::<f32>() + super::scalar::bf16_dot(a_tail, b_tail)
}

fn avx512_bf16_dot(a: &[u16], b: &[u16]) -> f32 {
    assert_eq!(a.len(), b.len());
    // SAFETY: this function is only placed in `Kernels` by `kernels_for` after `has_avx512_bf16`.
    unsafe { avx512_bf16_dot_imp(a, b) }
}

/// 32 BF16 values as a `vdpbf16ps` operand.
#[target_feature(enable = "avx512f,avx512bf16")]
#[inline]
fn bf16x32(v: &[u16; 32]) -> __m512bh {
    // SAFETY: `v` is 64 readable bytes; unaligned load. `__m512i` and `__m512bh` are both 512
    // plain bits, so the transmute only relabels the vector.
    unsafe { std::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512(v.as_ptr().cast())) }
}

/// Four independent accumulators hide the `vdpbf16ps` latency.
#[target_feature(enable = "avx512f,avx512bf16")]
fn avx512_bf16_dot_imp(a: &[u16], b: &[u16]) -> f32 {
    let mut acc = [_mm512_setzero_ps(); 4];
    let ((a32, a_tail), (b32, b_tail)) = (a.as_chunks::<32>(), b.as_chunks::<32>());
    let ((a4, a_rest), (b4, b_rest)) = (a32.as_chunks::<4>(), b32.as_chunks::<4>());
    for (x, y) in a4.iter().zip(b4) {
        for (k, acc) in acc.iter_mut().enumerate() {
            *acc = _mm512_dpbf16_ps(*acc, bf16x32(&x[k]), bf16x32(&y[k]));
        }
    }
    for (x, y) in a_rest.iter().zip(b_rest) {
        acc[0] = _mm512_dpbf16_ps(acc[0], bf16x32(x), bf16x32(y));
    }
    let sum = _mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3]));
    _mm512_reduce_add_ps(sum) + super::scalar::bf16_dot(a_tail, b_tail)
}

#[cfg(test)]
mod tests;
