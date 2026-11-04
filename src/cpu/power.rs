//! POWER9 kernels (ppc64le, VMX/VSX), chosen by `dispatch`: register-resident, multi-token rows
//! that return the scalar reference's results bit for bit.
//!
//! Rust's POWER vector intrinsics are not stable, so each kernel is one `asm!` block per weight
//! row that loops over the row's blocks and over the tokens:
//!
//! - The quants are loaded once per block, unpacked in vector registers (`vand` / `vsrb` for
//!   the nibbles, `vperm` + `vrlb` for Q5_1's fifth bits, `vrlb` for Q5_K's), and stay there
//!   while every token's activations meet them (a verify step's k+1 tokens share each decoded
//!   block, and the row is read from memory once).
//! - The integer dots are `vmsummbm` (signed activation bytes times unsigned weight bytes into
//!   words). Q8_0's signed weights become unsigned with `w ^ 0x80` and the dot is corrected by
//!   `128 * sum(x)`, exact for any byte.
//! - A 4x4 word transpose (`vmrghw` / `vmrglw` / `vadduwm`) turns four groups' word partial
//!   sums into one vector of four group dots; the group terms are then computed four groups at
//!   a time in VSX single precision (`xvcvsxwsp`, `xvmulsp`, `xvaddsp` / `xvsubsp`), scales
//!   converted from f16 by `xvcvhpsp` (POWER9). The operations and their order are the scalar
//!   reference's (separate multiplies and adds, no fused multiply-add; the integer products
//!   that the reference converts are exact in f32, so multiplying their converted factors is
//!   the same), so every result is bit-identical to `scalar.rs`.
//! - Each group's term is added to accumulator `g % 8` of its token, which lives in a small
//!   array the kernel loads and stores per block; the caller sums the eight with
//!   [`reduce8`]. Q5_1 / Q8_0 rows run four 32-value blocks per step; a row's last
//!   `blocks % 4` blocks are added by the scalar block functions after the kernel, in order.
//!
//! Byte order: `lxv` (DQ form) reverses a vector's bytes on little-endian, `lxvb16x` does not.
//! A dot product only needs its weights and activations loaded the same way, so the Q4_K /
//! Q5_K kernels use `lxv` for both; anything that places bytes or words by position (scales,
//! the Q5_1 bit expansion, the f16 gather) uses `lxvb16x` / `lxvw4x`, which keep memory order
//! in the ISA's (big-endian) element numbering the instructions are specified in.
//!
//! Q6_K and IQ4_XS run one `asm!` block per superblock (their sub-scales are unpacked in Rust
//! first); Q6_K's signed quants are dotted as unsigned `q` less `32 * sum(x)` per half group
//! (`vsum4sbs`), IQ4_XS's codebook values biased by 128 like Q8_0's. The BF16 dot widens with
//! `vmrghh` / `vmrglh` against zero (exact) and multiply-adds in VSX, within the bound stated on
//! `matmul_bf16`. Everything here is POWER9 (ISA 3.0) or older: no POWER10 instructions.
//!
//! The decoding tricks follow ggml's MIT-licensed `ggml-cpu/arch/powerpc/quants.c` (pinned
//! under `third_party/ggml`); the CPU expert share that needs them follows Strata-AC922's
//! POWER9 VSX expert kernels (eelgaev/Strata-AC922 bd1a96d, MIT).

use std::arch::asm;

use super::dispatch::{Isa, Kernels};
use super::scalar::{
    LANES, bf16_dot as bf16_dot_scalar, f16_at, iq4_xs_scales, q5_1_block, q8_0_block, reduce8,
};
use super::{GROUP, Q8Activations, QuantFormat};

/// The kernels of `isa`: VMX and VSX are in every ppc64le CPU's baseline; the kernels also use
/// POWER9 (ISA 3.0) instructions (`lxv`, `lxvb16x`, `xvcvhpsp`), and POWER9 is the only POWER
/// host Legion targets.
pub(super) fn kernels_for(isa: Isa) -> Option<Kernels> {
    (isa == Isa::Vsx).then_some(Kernels {
        isa,
        q8_0,
        q5_1,
        q4_k,
        q5_k,
        q6_k,
        iq4_xs,
        bf16_dot,
    })
}

/// Constant byte vectors the Q5_1 / Q8_0 kernels load (in memory order, `lxvb16x`).
#[repr(C, align(16))]
struct Consts([u8; 80]);

static CONSTS: Consts = Consts([
    // 0: replicate the `qh` byte 0 (block byte 4) into bytes 0..8 and byte 1 into 8..16.
    4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, //
    // 16: the same for `qh` bytes 2 and 3 (block bytes 6, 7).
    6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, //
    // 32: per byte, the left rotation that brings bit `k % 8` to bit 4.
    4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, //
    // 48: two blocks' f16 at bytes 0..2 and 2..4 of each into the low halves of four words:
    // [first's 0..2, second's 0..2, first's 2..4, second's 2..4] (little-endian halves).
    0, 0, 1, 0, 0, 0, 17, 16, 0, 0, 3, 2, 0, 0, 19, 18, //
    // 64: the IQ4_NL / IQ4_XS codebook (`scalar::IQ4_VALUES`) plus 128, as unsigned bytes.
    1, 24, 45, 63, 79, 93, 106, 118, 129, 141, 153, 166, 181, 197, 217, 241,
]);

/// The per-token pointers and strides every row kernel takes: token `t`'s activation quants
/// are at `x + t * x_stride` bytes, its group scales and sums at `t * g_stride` bytes from
/// `xd` / `xs`, its eight accumulators at `acc + 8 * t`.
#[derive(Clone, Copy)]
struct Tokens {
    x: *const i8,
    xd: *const f32,
    xs: *const i32,
    acc: *mut f32,
    n: usize,
    x_stride: usize,
    g_stride: usize,
}

/// Runs `row` on every weight row of `w` (`row_bytes` each) for all of `acts`' tokens, then
/// `tail` (the blocks the kernel leaves to the scalar code), and writes the reduced sums.
fn matmul(
    w: &[u8],
    acts: &Q8Activations,
    rows: usize,
    out: &mut [f32],
    row_bytes: usize,
    row: unsafe fn(&[u8], Tokens),
    tail: impl Fn(&[u8], &mut [f32; LANES], usize),
) {
    let (n, columns) = (acts.tokens(), acts.columns());
    if n == 0 {
        return;
    }
    let mut acc = vec![[0.0f32; LANES]; n];
    for (r, wrow) in w.chunks_exact(row_bytes).take(rows).enumerate() {
        acc.iter_mut().for_each(|a| *a = [0.0; LANES]);
        let tokens = Tokens {
            x: acts.qs.as_ptr(),
            xd: acts.scales.as_ptr(),
            xs: acts.sums.as_ptr(),
            acc: acc.as_mut_ptr().cast(),
            n,
            x_stride: columns,
            g_stride: columns / GROUP * 4,
        };
        // SAFETY: `wrow` is one whole row of the format's blocks, the activations hold `n`
        // tokens of `columns` values (strides as above), `acc` holds `n` x 8 floats.
        unsafe { row(wrow, tokens) };
        for (t, a) in acc.iter_mut().enumerate() {
            tail(wrow, a, t);
            out[t * rows + r] = reduce8(*a);
        }
    }
}

fn q4_k(w: &[u8], acts: &Q8Activations, rows: usize, out: &mut [f32]) {
    let rb = QuantFormat::Q4K
        .row_bytes(acts.columns())
        .expect("whole blocks");
    matmul(w, acts, rows, out, rb, q4_k_row, |_, _, _| {});
}

fn q5_k(w: &[u8], acts: &Q8Activations, rows: usize, out: &mut [f32]) {
    let rb = QuantFormat::Q5K
        .row_bytes(acts.columns())
        .expect("whole blocks");
    matmul(w, acts, rows, out, rb, q5_k_row, |_, _, _| {});
}

fn q5_1(w: &[u8], acts: &Q8Activations, rows: usize, out: &mut [f32]) {
    let bb = QuantFormat::Q5_1.block_bytes();
    let rb = QuantFormat::Q5_1
        .row_bytes(acts.columns())
        .expect("whole blocks");
    matmul(w, acts, rows, out, rb, q5_1_row, |row, acc, t| {
        let (xq, xd, xs) = (acts.qs(t), acts.scales(t), acts.sums(t));
        let nb = row.len() / bb;
        for b in nb / 4 * 4..nb {
            let blk = &row[b * bb..][..bb];
            acc[b % LANES] += q5_1_block(blk, &xq[GROUP * b..], xd[b], xs[b]);
        }
    });
}

fn q6_k(w: &[u8], acts: &Q8Activations, rows: usize, out: &mut [f32]) {
    let rb = QuantFormat::Q6K
        .row_bytes(acts.columns())
        .expect("whole blocks");
    matmul(w, acts, rows, out, rb, q6_k_row, |_, _, _| {});
}

fn iq4_xs(w: &[u8], acts: &Q8Activations, rows: usize, out: &mut [f32]) {
    let rb = QuantFormat::Iq4Xs
        .row_bytes(acts.columns())
        .expect("whole blocks");
    matmul(w, acts, rows, out, rb, iq4_xs_row, |_, _, _| {});
}

fn q8_0(w: &[u8], acts: &Q8Activations, rows: usize, out: &mut [f32]) {
    let bb = QuantFormat::Q8_0.block_bytes();
    let rb = QuantFormat::Q8_0
        .row_bytes(acts.columns())
        .expect("whole blocks");
    matmul(w, acts, rows, out, rb, q8_0_row, |row, acc, t| {
        let (xq, xd) = (acts.qs(t), acts.scales(t));
        let nb = row.len() / bb;
        for b in nb / 4 * 4..nb {
            let blk = &row[b * bb..][..bb];
            acc[b % LANES] += q8_0_block(blk, &xq[GROUP * b..], xd[b]);
        }
    });
}

/// One Q4_K / Q5_K row: per 256-value superblock, the quants are unpacked into v3..v18 (group
/// `j`'s two 16-byte halves in v(3 + 2j), v(4 + 2j)) by `$unpack` (which may use v19..v31),
/// the eight 6-bit scales and mins into vs0..vs3 (f32), `d` / `dmin` splat into vs4 / vs5;
/// then per token: 16 `vmsummbm` into v19..v26, two transposes into the group dots, the eight
/// terms `xd * (d * (sc * dot) - dmin * (m * sum))` added to the token's accumulators.
macro_rules! k_quant_row {
    ($name:ident, $block:expr, $($unpack:literal),* $(,)?) => {
        /// # Safety
        /// `w` holds whole superblocks; `tk` describes live activations and accumulators.
        unsafe fn $name(w: &[u8], tk: Tokens) {
            let nsb = w.len() / $block;
            if nsb == 0 {
                return;
            }
            // SAFETY: per superblock the loads read its `$block` bytes, 256 activation bytes,
            // 8 scales and 8 sums of each token, and the token's 8 accumulators, all inside
            // the caller's buffers; every register written is declared clobbered.
            unsafe {
                asm!(
                    "vspltisb 0, 15",
                    "vspltisb 1, 4",
                    "vspltisb 2, 0",
                    "2:",
                    $($unpack,)*
                    // Scales and mins (`k_scales`): Q = bytes 4..16 of the block.
                    "lxvb16x 51, {i4}, {w}",
                    "vspltisb 20, 6",
                    "vsrb 21, 19, 20",
                    "vslb 22, 21, 20",
                    "vsububm 22, 19, 22",
                    "vslb 21, 21, 1",
                    "vand 23, 19, 0",
                    "vsrb 24, 19, 1",
                    "vsldoi 23, 23, 23, 8",
                    "vor 23, 23, 21",
                    "vsldoi 24, 24, 24, 4",
                    "vor 24, 24, 21",
                    "vupkhsb 25, 22",
                    "vupkhsb 23, 23",
                    "vupkhsb 24, 24",
                    "vupkhsh 26, 25",
                    "vupklsh 25, 25",
                    "vupkhsh 23, 23",
                    "vupklsh 24, 24",
                    "xvcvsxwsp 0, 58",
                    "xvcvsxwsp 1, 55",
                    "xvcvsxwsp 2, 57",
                    "xvcvsxwsp 3, 56",
                    // d (low half of word 0) and dmin (high half).
                    "lxvw4x 52, 0, {w}",
                    "vspltisw 21, -16",
                    "vsrw 21, 20, 21",
                    "xvcvhpsp 52, 52",
                    "xvcvhpsp 53, 53",
                    "xxspltw 4, 52, 0",
                    "xxspltw 5, 53, 0",
                    // Tokens.
                    "mr {xp}, {x}",
                    "mr {dp}, {xd}",
                    "mr {sp}, {xs}",
                    "mr {ap}, {acc}",
                    "mr {t}, {nt}",
                    "3:",
                    "lxv 59, 0({xp})",
                    "lxv 60, 16({xp})",
                    "lxv 61, 32({xp})",
                    "lxv 62, 48({xp})",
                    "vmsummbm 19, 27, 3, 2",
                    "vmsummbm 20, 29, 5, 2",
                    "vmsummbm 19, 28, 4, 19",
                    "vmsummbm 20, 30, 6, 20",
                    "lxv 59, 64({xp})",
                    "lxv 60, 80({xp})",
                    "lxv 61, 96({xp})",
                    "lxv 62, 112({xp})",
                    "vmsummbm 21, 27, 7, 2",
                    "vmsummbm 22, 29, 9, 2",
                    "vmsummbm 21, 28, 8, 21",
                    "vmsummbm 22, 30, 10, 22",
                    "lxv 59, 128({xp})",
                    "lxv 60, 144({xp})",
                    "lxv 61, 160({xp})",
                    "lxv 62, 176({xp})",
                    "vmsummbm 23, 27, 11, 2",
                    "vmsummbm 24, 29, 13, 2",
                    "vmsummbm 23, 28, 12, 23",
                    "vmsummbm 24, 30, 14, 24",
                    "lxv 59, 192({xp})",
                    "lxv 60, 208({xp})",
                    "lxv 61, 224({xp})",
                    "lxv 62, 240({xp})",
                    "vmsummbm 25, 27, 15, 2",
                    "vmsummbm 26, 29, 17, 2",
                    "vmsummbm 25, 28, 16, 25",
                    "vmsummbm 26, 30, 18, 26",
                    // Groups 0..4 -> v29, 4..8 -> v30.
                    "vmrghw 27, 19, 21",
                    "vmrglw 28, 19, 21",
                    "vmrghw 29, 20, 22",
                    "vmrglw 30, 20, 22",
                    "vadduwm 27, 27, 28",
                    "vadduwm 28, 29, 30",
                    "vmrghw 29, 27, 28",
                    "vmrglw 31, 27, 28",
                    "vadduwm 29, 29, 31",
                    "vmrghw 27, 23, 25",
                    "vmrglw 28, 23, 25",
                    "vmrghw 30, 24, 26",
                    "vmrglw 31, 24, 26",
                    "vadduwm 27, 27, 28",
                    "vadduwm 28, 30, 31",
                    "vmrghw 30, 27, 28",
                    "vmrglw 31, 27, 28",
                    "vadduwm 30, 30, 31",
                    // Terms.
                    "xvcvsxwsp 6, 61",
                    "xvcvsxwsp 7, 62",
                    "xvmulsp 6, 6, 0",
                    "xvmulsp 7, 7, 1",
                    "xvmulsp 6, 6, 4",
                    "xvmulsp 7, 7, 4",
                    "lxvw4x 8, 0, {sp}",
                    "lxvw4x 9, {i16}, {sp}",
                    "xvcvsxwsp 8, 8",
                    "xvcvsxwsp 9, 9",
                    "xvmulsp 8, 8, 2",
                    "xvmulsp 9, 9, 3",
                    "xvmulsp 8, 8, 5",
                    "xvmulsp 9, 9, 5",
                    "xvsubsp 6, 6, 8",
                    "xvsubsp 7, 7, 9",
                    "lxvw4x 10, 0, {dp}",
                    "lxvw4x 11, {i16}, {dp}",
                    "xvmulsp 6, 6, 10",
                    "xvmulsp 7, 7, 11",
                    "lxvw4x 12, 0, {ap}",
                    "lxvw4x 13, {i16}, {ap}",
                    "xvaddsp 12, 12, 6",
                    "xvaddsp 13, 13, 7",
                    "stxvw4x 12, 0, {ap}",
                    "stxvw4x 13, {i16}, {ap}",
                    "add {xp}, {xp}, {xstr}",
                    "add {dp}, {dp}, {gstr}",
                    "add {sp}, {sp}, {gstr}",
                    "addi {ap}, {ap}, 32",
                    "addi {t}, {t}, -1",
                    "cmpdi 0, {t}, 0",
                    "bne 0, 3b",
                    // Next superblock.
                    "addi {w}, {w}, {block}",
                    "addi {x}, {x}, 256",
                    "addi {xd}, {xd}, 32",
                    "addi {xs}, {xs}, 32",
                    "addi {n}, {n}, -1",
                    "cmpdi 0, {n}, 0",
                    "bne 0, 2b",
                    block = const $block,
                    w = inout(reg_nonzero) w.as_ptr() => _,
                    x = inout(reg_nonzero) tk.x => _,
                    xd = inout(reg_nonzero) tk.xd => _,
                    xs = inout(reg_nonzero) tk.xs => _,
                    n = inout(reg_nonzero) nsb => _,
                    acc = in(reg_nonzero) tk.acc,
                    nt = in(reg_nonzero) tk.n,
                    xstr = in(reg_nonzero) tk.x_stride,
                    gstr = in(reg_nonzero) tk.g_stride,
                    i4 = in(reg_nonzero) 4usize,
                    i16 = in(reg_nonzero) 16usize,
                    xp = out(reg_nonzero) _,
                    dp = out(reg_nonzero) _,
                    sp = out(reg_nonzero) _,
                    ap = out(reg_nonzero) _,
                    t = out(reg_nonzero) _,
                    out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
                    out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _,
                    out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _,
                    out("v15") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                    out("v20") _, out("v21") _, out("v22") _, out("v23") _, out("v24") _,
                    out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _,
                    out("v30") _, out("v31") _,
                    out("f0") _, out("f1") _, out("f2") _, out("f3") _, out("f4") _,
                    out("f5") _, out("f6") _, out("f7") _, out("f8") _, out("f9") _,
                    out("f10") _, out("f11") _, out("f12") _, out("f13") _,
                    options(nostack),
                );
            }
        }
    };
}

// Q4_K: `qs` (128 bytes at 16): chunk `c`'s 32 bytes hold group 2c in their low nibbles and
// group 2c + 1 in their high ones.
k_quant_row!(
    q4_k_row,
    144,
    "lxv 37, 16({w})",
    "lxv 38, 32({w})",
    "vand 3, 5, 0",
    "vand 4, 6, 0",
    "vsrb 5, 5, 1",
    "vsrb 6, 6, 1",
    "lxv 41, 48({w})",
    "lxv 42, 64({w})",
    "vand 7, 9, 0",
    "vand 8, 10, 0",
    "vsrb 9, 9, 1",
    "vsrb 10, 10, 1",
    "lxv 45, 80({w})",
    "lxv 46, 96({w})",
    "vand 11, 13, 0",
    "vand 12, 14, 0",
    "vsrb 13, 13, 1",
    "vsrb 14, 14, 1",
    "lxv 49, 112({w})",
    "lxv 50, 128({w})",
    "vand 15, 17, 0",
    "vand 16, 18, 0",
    "vsrb 17, 17, 1",
    "vsrb 18, 18, 1",
);

// Q5_K: Q4_K's nibbles from `qs` (128 bytes at 48), plus each value's fifth bit: bit `j` (its
// group) of the `qh` byte (32 bytes at 16) at its position in the group, as 0x10. `qh` is
// loaded like `qs` and the activations (`lxv`), so its bytes line up with theirs; rotating
// each byte left by `4 - j` (mod 8) brings bit `j` to bit 4 (v21 = 0x10, v22 = 7: one more
// left rotation by 7 moves on to the next group's bit).
k_quant_row!(
    q5_k_row,
    176,
    "lxv 37, 48({w})",
    "lxv 38, 64({w})",
    "vand 3, 5, 0",
    "vand 4, 6, 0",
    "vsrb 5, 5, 1",
    "vsrb 6, 6, 1",
    "lxv 41, 80({w})",
    "lxv 42, 96({w})",
    "vand 7, 9, 0",
    "vand 8, 10, 0",
    "vsrb 9, 9, 1",
    "vsrb 10, 10, 1",
    "lxv 45, 112({w})",
    "lxv 46, 128({w})",
    "vand 11, 13, 0",
    "vand 12, 14, 0",
    "vsrb 13, 13, 1",
    "vsrb 14, 14, 1",
    "lxv 49, 144({w})",
    "lxv 50, 160({w})",
    "vand 15, 17, 0",
    "vand 16, 18, 0",
    "vsrb 17, 17, 1",
    "vsrb 18, 18, 1",
    "lxv 51, 16({w})",
    "lxv 52, 32({w})",
    "vspltisb 21, 1",
    "vslb 21, 21, 1",
    "vspltisb 22, 7",
    "vrlb 19, 19, 1",
    "vrlb 20, 20, 1",
    "vand 23, 19, 21",
    "vand 24, 20, 21",
    "vor 3, 3, 23",
    "vor 4, 4, 24",
    "vrlb 19, 19, 22",
    "vrlb 20, 20, 22",
    "vand 23, 19, 21",
    "vand 24, 20, 21",
    "vor 5, 5, 23",
    "vor 6, 6, 24",
    "vrlb 19, 19, 22",
    "vrlb 20, 20, 22",
    "vand 23, 19, 21",
    "vand 24, 20, 21",
    "vor 7, 7, 23",
    "vor 8, 8, 24",
    "vrlb 19, 19, 22",
    "vrlb 20, 20, 22",
    "vand 23, 19, 21",
    "vand 24, 20, 21",
    "vor 9, 9, 23",
    "vor 10, 10, 24",
    "vrlb 19, 19, 22",
    "vrlb 20, 20, 22",
    "vand 23, 19, 21",
    "vand 24, 20, 21",
    "vor 11, 11, 23",
    "vor 12, 12, 24",
    "vrlb 19, 19, 22",
    "vrlb 20, 20, 22",
    "vand 23, 19, 21",
    "vand 24, 20, 21",
    "vor 13, 13, 23",
    "vor 14, 14, 24",
    "vrlb 19, 19, 22",
    "vrlb 20, 20, 22",
    "vand 23, 19, 21",
    "vand 24, 20, 21",
    "vor 15, 15, 23",
    "vor 16, 16, 24",
    "vrlb 19, 19, 22",
    "vrlb 20, 20, 22",
    "vand 23, 19, 21",
    "vand 24, 20, 21",
    "vor 17, 17, 23",
    "vor 18, 18, 24",
);

/// One Q5_1 row, four 24-byte blocks (`d`, `m`, `qh[4]`, `qs[16]`) per step: block `j`'s
/// values 0..16 (`qs` low nibbles | fifth bits 0..16) in v(8 + 2j), 16..32 in v(9 + 2j); `d`
/// and `m` of the four blocks in vs0 / vs1; then per token four dots (v20..v23), one
/// transpose, the terms `xd * (d * dot + m * sum)` added to accumulators `b % 8`.
///
/// # Safety
/// `w` holds whole blocks; `tk` describes live activations and accumulators.
unsafe fn q5_1_row(w: &[u8], tk: Tokens) {
    let nq = w.len() / QuantFormat::Q5_1.block_bytes() / 4;
    if nq == 0 {
        return;
    }
    // SAFETY: per step the loads read the four blocks' 96 bytes, 128 activation bytes, 4
    // scales and 4 sums of each token, and 4 of the token's accumulators, all inside the
    // caller's buffers, and the 64 bytes of `CONSTS`; every register written is declared
    // clobbered.
    unsafe {
        asm!(
            "vspltisb 0, 15",
            "vspltisb 1, 4",
            "vspltisb 2, 0",
            "lxvb16x 35, 0, {c}",
            "lxvb16x 36, {i16}, {c}",
            "lxvb16x 37, {i32}, {c}",
            "lxvb16x 39, {i48}, {c}",
            "vspltisb 6, 1",
            "vslb 6, 6, 1",
            "2:",
            "lxvb16x 48, 0, {w}",
            "lxvb16x 52, {i8}, {w}",
            "addi {w}, {w}, 24",
            "lxvb16x 49, 0, {w}",
            "lxvb16x 53, {i8}, {w}",
            "addi {w}, {w}, 24",
            "lxvb16x 50, 0, {w}",
            "lxvb16x 54, {i8}, {w}",
            "addi {w}, {w}, 24",
            "lxvb16x 51, 0, {w}",
            "lxvb16x 55, {i8}, {w}",
            "addi {w}, {w}, 24",
            // Block 0 -> v8, v9 (v20: qs, v24..v27 temporaries).
            "vperm 24, 16, 16, 3",
            "vperm 25, 16, 16, 4",
            "vrlb 24, 24, 5",
            "vrlb 25, 25, 5",
            "vand 24, 24, 6",
            "vand 25, 25, 6",
            "vand 8, 20, 0",
            "vsrb 9, 20, 1",
            "vor 8, 8, 24",
            "vor 9, 9, 25",
            "vperm 26, 17, 17, 3",
            "vperm 27, 17, 17, 4",
            "vrlb 26, 26, 5",
            "vrlb 27, 27, 5",
            "vand 26, 26, 6",
            "vand 27, 27, 6",
            "vand 10, 21, 0",
            "vsrb 11, 21, 1",
            "vor 10, 10, 26",
            "vor 11, 11, 27",
            "vperm 24, 18, 18, 3",
            "vperm 25, 18, 18, 4",
            "vrlb 24, 24, 5",
            "vrlb 25, 25, 5",
            "vand 24, 24, 6",
            "vand 25, 25, 6",
            "vand 12, 22, 0",
            "vsrb 13, 22, 1",
            "vor 12, 12, 24",
            "vor 13, 13, 25",
            "vperm 26, 19, 19, 3",
            "vperm 27, 19, 19, 4",
            "vrlb 26, 26, 5",
            "vrlb 27, 27, 5",
            "vand 26, 26, 6",
            "vand 27, 27, 6",
            "vand 14, 23, 0",
            "vsrb 15, 23, 1",
            "vor 14, 14, 26",
            "vor 15, 15, 27",
            // d and m of the four blocks.
            "vperm 24, 16, 17, 7",
            "vperm 25, 18, 19, 7",
            "xxpermdi 0, 56, 57, 0",
            "xxpermdi 1, 56, 57, 3",
            "xvcvhpsp 0, 0",
            "xvcvhpsp 1, 1",
            // Tokens.
            "mr {xp}, {x}",
            "mr {dp}, {xd}",
            "mr {sp}, {xs}",
            "add {ap}, {acc}, {ao}",
            "mr {t}, {nt}",
            "3:",
            "lxvb16x 56, 0, {xp}",
            "lxvb16x 57, {i16}, {xp}",
            "lxvb16x 58, {i32}, {xp}",
            "lxvb16x 59, {i48}, {xp}",
            "vmsummbm 20, 24, 8, 2",
            "vmsummbm 21, 26, 10, 2",
            "vmsummbm 20, 25, 9, 20",
            "vmsummbm 21, 27, 11, 21",
            "addi {xp}, {xp}, 64",
            "lxvb16x 56, 0, {xp}",
            "lxvb16x 57, {i16}, {xp}",
            "lxvb16x 58, {i32}, {xp}",
            "lxvb16x 59, {i48}, {xp}",
            "vmsummbm 22, 24, 12, 2",
            "vmsummbm 23, 26, 14, 2",
            "vmsummbm 22, 25, 13, 22",
            "vmsummbm 23, 27, 15, 23",
            "add {xp}, {xp}, {xstep}",
            "vmrghw 28, 20, 22",
            "vmrglw 29, 20, 22",
            "vmrghw 30, 21, 23",
            "vmrglw 31, 21, 23",
            "vadduwm 28, 28, 29",
            "vadduwm 29, 30, 31",
            "vmrghw 30, 28, 29",
            "vmrglw 31, 28, 29",
            "vadduwm 30, 30, 31",
            "xvcvsxwsp 2, 62",
            "xvmulsp 2, 2, 0",
            "lxvw4x 3, 0, {sp}",
            "xvcvsxwsp 3, 3",
            "xvmulsp 3, 3, 1",
            "xvaddsp 2, 2, 3",
            "lxvw4x 4, 0, {dp}",
            "xvmulsp 2, 2, 4",
            "lxvw4x 5, 0, {ap}",
            "xvaddsp 5, 5, 2",
            "stxvw4x 5, 0, {ap}",
            "add {dp}, {dp}, {gstr}",
            "add {sp}, {sp}, {gstr}",
            "addi {ap}, {ap}, 32",
            "addi {t}, {t}, -1",
            "cmpdi 0, {t}, 0",
            "bne 0, 3b",
            // Next four blocks: accumulators 0..4 and 4..8 alternate.
            "addi {x}, {x}, 128",
            "addi {xd}, {xd}, 16",
            "addi {xs}, {xs}, 16",
            "xori {ao}, {ao}, 16",
            "addi {n}, {n}, -1",
            "cmpdi 0, {n}, 0",
            "bne 0, 2b",
            c = in(reg_nonzero) CONSTS.0.as_ptr(),
            w = inout(reg_nonzero) w.as_ptr() => _,
            x = inout(reg_nonzero) tk.x => _,
            xd = inout(reg_nonzero) tk.xd => _,
            xs = inout(reg_nonzero) tk.xs => _,
            n = inout(reg_nonzero) nq => _,
            ao = inout(reg) 0usize => _,
            acc = in(reg_nonzero) tk.acc,
            nt = in(reg_nonzero) tk.n,
            xstep = in(reg) tk.x_stride.wrapping_sub(64),
            gstr = in(reg_nonzero) tk.g_stride,
            i8 = in(reg_nonzero) 8usize,
            i16 = in(reg_nonzero) 16usize,
            i32 = in(reg_nonzero) 32usize,
            i48 = in(reg_nonzero) 48usize,
            xp = out(reg_nonzero) _,
            dp = out(reg_nonzero) _,
            sp = out(reg_nonzero) _,
            ap = out(reg_nonzero) _,
            t = out(reg_nonzero) _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
            out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _,
            out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _,
            out("v15") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _, out("v24") _,
            out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _,
            out("v30") _, out("v31") _,
            out("f0") _, out("f1") _, out("f2") _, out("f3") _, out("f4") _, out("f5") _,
            options(nostack),
        );
    }
}

/// One Q8_0 row, four 34-byte blocks (`d`, `qs[32]`) per step: block `j`'s quants plus 128
/// (`^ 0x80`, unsigned) in v(4 + 2j), v(5 + 2j), the four `d` in vs0; per token four dots
/// (v16..v19) less `128 * sum(x)`, the terms `(d * xd) * dot` added to accumulators `b % 8`.
///
/// # Safety
/// `w` holds whole blocks; `tk` describes live activations and accumulators.
unsafe fn q8_0_row(w: &[u8], tk: Tokens) {
    let nq = w.len() / QuantFormat::Q8_0.block_bytes() / 4;
    if nq == 0 {
        return;
    }
    // SAFETY: as `q5_1_row`, with 136 weight bytes per step.
    unsafe {
        asm!(
            "vspltisb 0, -1",
            "vspltisb 3, 7",
            "vslb 0, 0, 3",
            "vspltisw 3, 7",
            "vspltisb 1, 0",
            "lxvb16x 34, {i48}, {c}",
            "2:",
            "lxvb16x 44, 0, {w}",
            "lxvb16x 36, {i2}, {w}",
            "lxvb16x 37, {i18}, {w}",
            "addi {w}, {w}, 34",
            "lxvb16x 45, 0, {w}",
            "lxvb16x 38, {i2}, {w}",
            "lxvb16x 39, {i18}, {w}",
            "addi {w}, {w}, 34",
            "lxvb16x 46, 0, {w}",
            "lxvb16x 40, {i2}, {w}",
            "lxvb16x 41, {i18}, {w}",
            "addi {w}, {w}, 34",
            "lxvb16x 47, 0, {w}",
            "lxvb16x 42, {i2}, {w}",
            "lxvb16x 43, {i18}, {w}",
            "addi {w}, {w}, 34",
            "vxor 4, 4, 0",
            "vxor 5, 5, 0",
            "vxor 6, 6, 0",
            "vxor 7, 7, 0",
            "vxor 8, 8, 0",
            "vxor 9, 9, 0",
            "vxor 10, 10, 0",
            "vxor 11, 11, 0",
            "vperm 16, 12, 13, 2",
            "vperm 17, 14, 15, 2",
            "xxpermdi 0, 48, 49, 0",
            "xvcvhpsp 0, 0",
            // Tokens.
            "mr {xp}, {x}",
            "mr {dp}, {xd}",
            "mr {sp}, {xs}",
            "add {ap}, {acc}, {ao}",
            "mr {t}, {nt}",
            "3:",
            "lxvb16x 52, 0, {xp}",
            "lxvb16x 53, {i16}, {xp}",
            "lxvb16x 54, {i32}, {xp}",
            "lxvb16x 55, {i48}, {xp}",
            "vmsummbm 16, 20, 4, 1",
            "vmsummbm 17, 22, 6, 1",
            "vmsummbm 16, 21, 5, 16",
            "vmsummbm 17, 23, 7, 17",
            "addi {xp}, {xp}, 64",
            "lxvb16x 52, 0, {xp}",
            "lxvb16x 53, {i16}, {xp}",
            "lxvb16x 54, {i32}, {xp}",
            "lxvb16x 55, {i48}, {xp}",
            "vmsummbm 18, 20, 8, 1",
            "vmsummbm 19, 22, 10, 1",
            "vmsummbm 18, 21, 9, 18",
            "vmsummbm 19, 23, 11, 19",
            "add {xp}, {xp}, {xstep}",
            "vmrghw 24, 16, 18",
            "vmrglw 25, 16, 18",
            "vmrghw 26, 17, 19",
            "vmrglw 27, 17, 19",
            "vadduwm 24, 24, 25",
            "vadduwm 25, 26, 27",
            "vmrghw 26, 24, 25",
            "vmrglw 27, 24, 25",
            "vadduwm 26, 26, 27",
            "lxvw4x 62, 0, {sp}",
            "vslw 30, 30, 3",
            "vsubuwm 26, 26, 30",
            "lxvw4x 2, 0, {dp}",
            "xvmulsp 2, 0, 2",
            "xvcvsxwsp 3, 58",
            "xvmulsp 2, 2, 3",
            "lxvw4x 4, 0, {ap}",
            "xvaddsp 4, 4, 2",
            "stxvw4x 4, 0, {ap}",
            "add {dp}, {dp}, {gstr}",
            "add {sp}, {sp}, {gstr}",
            "addi {ap}, {ap}, 32",
            "addi {t}, {t}, -1",
            "cmpdi 0, {t}, 0",
            "bne 0, 3b",
            "addi {x}, {x}, 128",
            "addi {xd}, {xd}, 16",
            "addi {xs}, {xs}, 16",
            "xori {ao}, {ao}, 16",
            "addi {n}, {n}, -1",
            "cmpdi 0, {n}, 0",
            "bne 0, 2b",
            c = in(reg_nonzero) CONSTS.0.as_ptr(),
            w = inout(reg_nonzero) w.as_ptr() => _,
            x = inout(reg_nonzero) tk.x => _,
            xd = inout(reg_nonzero) tk.xd => _,
            xs = inout(reg_nonzero) tk.xs => _,
            n = inout(reg_nonzero) nq => _,
            ao = inout(reg) 0usize => _,
            acc = in(reg_nonzero) tk.acc,
            nt = in(reg_nonzero) tk.n,
            xstep = in(reg) tk.x_stride.wrapping_sub(64),
            gstr = in(reg_nonzero) tk.g_stride,
            i2 = in(reg_nonzero) 2usize,
            i18 = in(reg_nonzero) 18usize,
            i16 = in(reg_nonzero) 16usize,
            i32 = in(reg_nonzero) 32usize,
            i48 = in(reg_nonzero) 48usize,
            xp = out(reg_nonzero) _,
            dp = out(reg_nonzero) _,
            sp = out(reg_nonzero) _,
            ap = out(reg_nonzero) _,
            t = out(reg_nonzero) _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
            out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _,
            out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _,
            out("v15") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _, out("v24") _,
            out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _,
            out("v30") _, out("v31") _,
            out("f0") _, out("f1") _, out("f2") _, out("f3") _, out("f4") _,
            options(nostack),
        );
    }
}

impl Tokens {
    /// The pointers advanced to superblock `sb` (256 values, 8 groups) of every token.
    fn superblock(self, sb: usize) -> Self {
        // `wrapping_add` only computes addresses; the kernels load inside the buffers.
        Self {
            x: self.x.wrapping_add(256 * sb),
            xd: self.xd.wrapping_add(8 * sb),
            xs: self.xs.wrapping_add(8 * sb),
            ..self
        }
    }
}

/// One Q6_K row, one `asm!` block per superblock (`ql[128]`, `qh[64]`, `scales[16]`, `d`): the
/// quants 0..=63 unpacked into v3..v18 (half `h` of group `g` in v(3 + 2g + h)); per token each
/// half's dot is `vmsummbm` minus `32 * sum(x)` (`vsum4sbs`), which is the dot with `q - 32`
/// without a signed weight byte. Four half dots are transposed into one vector, multiplied by
/// their sub-scales (`vmuluwm`), the two halves added: the reference's `isum`; then
/// `(d * xd) * isum` is added to accumulators `g % 8`. The sub-scales go in as words in that
/// order, `[0 2 4 6] [1 3 5 7] [8 10 12 14] [9 11 13 15]`, and `d` splat.
///
/// # Safety
/// `w` holds whole superblocks; `tk` describes live activations and accumulators.
unsafe fn q6_k_row(w: &[u8], tk: Tokens) {
    for (sb, blk) in w
        .as_chunks::<{ QuantFormat::Q6K.block_bytes() }>()
        .0
        .iter()
        .enumerate()
    {
        let sc: [i32; 16] = std::array::from_fn(|i| {
            i32::from(blk[192 + 8 * (i / 8) + 2 * (i % 4) + (i / 4) % 2] as i8)
        });
        let d = [f16_at(blk, 208); 4];
        let tk = tk.superblock(sb);
        // SAFETY: the loads read the superblock's 210 bytes, 256 activation bytes and 8 scales
        // of each token, the token's 8 accumulators, `sc` and `d`, all inside live buffers;
        // every register written is declared clobbered.
        unsafe {
            asm!(
                // Unpack: v23 = 15, v24 = 4, v25 = 2, v26 = 0x30; v27/v28 = `ql` at 64n + 16h and
                // 64n + 32 + 16h, v29 = `qh` at 128 + 32n + 16h; groups 4n + k, half h.
                "vspltisb 0, 0",
                "vspltisw 1, 5",
                "vspltisb 23, 15",
                "vspltisb 24, 4",
                "vspltisb 25, 2",
                "vspltisb 26, 3",
                "vslb 26, 26, 24",
                "lxv 59, 0({w})",
                "lxv 60, 32({w})",
                "lxv 61, 128({w})",
                "vand 30, 27, 23",
                "vslb 31, 29, 24",
                "vand 31, 31, 26",
                "vor 3, 30, 31",
                "vand 30, 28, 23",
                "vslb 31, 29, 25",
                "vand 31, 31, 26",
                "vor 5, 30, 31",
                "vsrb 30, 27, 24",
                "vand 31, 29, 26",
                "vor 7, 30, 31",
                "vsrb 30, 28, 24",
                "vsrb 31, 29, 25",
                "vand 31, 31, 26",
                "vor 9, 30, 31",
                "lxv 59, 16({w})",
                "lxv 60, 48({w})",
                "lxv 61, 144({w})",
                "vand 30, 27, 23",
                "vslb 31, 29, 24",
                "vand 31, 31, 26",
                "vor 4, 30, 31",
                "vand 30, 28, 23",
                "vslb 31, 29, 25",
                "vand 31, 31, 26",
                "vor 6, 30, 31",
                "vsrb 30, 27, 24",
                "vand 31, 29, 26",
                "vor 8, 30, 31",
                "vsrb 30, 28, 24",
                "vsrb 31, 29, 25",
                "vand 31, 31, 26",
                "vor 10, 30, 31",
                "lxv 59, 64({w})",
                "lxv 60, 96({w})",
                "lxv 61, 160({w})",
                "vand 30, 27, 23",
                "vslb 31, 29, 24",
                "vand 31, 31, 26",
                "vor 11, 30, 31",
                "vand 30, 28, 23",
                "vslb 31, 29, 25",
                "vand 31, 31, 26",
                "vor 13, 30, 31",
                "vsrb 30, 27, 24",
                "vand 31, 29, 26",
                "vor 15, 30, 31",
                "vsrb 30, 28, 24",
                "vsrb 31, 29, 25",
                "vand 31, 31, 26",
                "vor 17, 30, 31",
                "lxv 59, 80({w})",
                "lxv 60, 112({w})",
                "lxv 61, 176({w})",
                "vand 30, 27, 23",
                "vslb 31, 29, 24",
                "vand 31, 31, 26",
                "vor 12, 30, 31",
                "vand 30, 28, 23",
                "vslb 31, 29, 25",
                "vand 31, 31, 26",
                "vor 14, 30, 31",
                "vsrb 30, 27, 24",
                "vand 31, 29, 26",
                "vor 16, 30, 31",
                "vsrb 30, 28, 24",
                "vsrb 31, 29, 25",
                "vand 31, 31, 26",
                "vor 18, 30, 31",
                "lxvw4x 4, 0, {d}",
                "mr {t}, {nt}",
                "3:",
                // Groups 0..4: low halves -> v28, high halves -> v29.
                "lxv 55, 0({x})",
                "vmsummbm 19, 23, 3, 0",
                "vsum4sbs 24, 23, 0",
                "vslw 24, 24, 1",
                "vsubuwm 19, 19, 24",
                "lxv 55, 32({x})",
                "vmsummbm 20, 23, 5, 0",
                "vsum4sbs 24, 23, 0",
                "vslw 24, 24, 1",
                "vsubuwm 20, 20, 24",
                "lxv 55, 64({x})",
                "vmsummbm 21, 23, 7, 0",
                "vsum4sbs 24, 23, 0",
                "vslw 24, 24, 1",
                "vsubuwm 21, 21, 24",
                "lxv 55, 96({x})",
                "vmsummbm 22, 23, 9, 0",
                "vsum4sbs 24, 23, 0",
                "vslw 24, 24, 1",
                "vsubuwm 22, 22, 24",
                "vmrghw 25, 19, 21",
                "vmrglw 26, 19, 21",
                "vadduwm 25, 25, 26",
                "vmrghw 26, 20, 22",
                "vmrglw 27, 20, 22",
                "vadduwm 26, 26, 27",
                "vmrghw 27, 25, 26",
                "vmrglw 28, 25, 26",
                "vadduwm 28, 27, 28",
                "lxv 55, 16({x})",
                "vmsummbm 19, 23, 4, 0",
                "vsum4sbs 24, 23, 0",
                "vslw 24, 24, 1",
                "vsubuwm 19, 19, 24",
                "lxv 55, 48({x})",
                "vmsummbm 20, 23, 6, 0",
                "vsum4sbs 24, 23, 0",
                "vslw 24, 24, 1",
                "vsubuwm 20, 20, 24",
                "lxv 55, 80({x})",
                "vmsummbm 21, 23, 8, 0",
                "vsum4sbs 24, 23, 0",
                "vslw 24, 24, 1",
                "vsubuwm 21, 21, 24",
                "lxv 55, 112({x})",
                "vmsummbm 22, 23, 10, 0",
                "vsum4sbs 24, 23, 0",
                "vslw 24, 24, 1",
                "vsubuwm 22, 22, 24",
                "vmrghw 25, 19, 21",
                "vmrglw 26, 19, 21",
                "vadduwm 25, 25, 26",
                "vmrghw 26, 20, 22",
                "vmrglw 27, 20, 22",
                "vadduwm 26, 26, 27",
                "vmrghw 27, 25, 26",
                "vmrglw 29, 25, 26",
                "vadduwm 29, 27, 29",
                "lxvw4x 62, {s0}, {sc}",
                "vmuluwm 28, 28, 30",
                "lxvw4x 62, {s1}, {sc}",
                "vmuluwm 29, 29, 30",
                "vadduwm 28, 28, 29",
                "xvcvsxwsp 6, 60",
                "lxvw4x 7, {o0}, {xd}",
                "xvmulsp 7, 4, 7",
                "xvmulsp 6, 7, 6",
                "lxvw4x 8, {o0}, {acc}",
                "xvaddsp 8, 8, 6",
                "stxvw4x 8, {o0}, {acc}",
                // Groups 4..8: low halves -> v28, high halves -> v29.
                "lxv 55, 128({x})",
                "vmsummbm 19, 23, 11, 0",
                "vsum4sbs 24, 23, 0",
                "vslw 24, 24, 1",
                "vsubuwm 19, 19, 24",
                "lxv 55, 160({x})",
                "vmsummbm 20, 23, 13, 0",
                "vsum4sbs 24, 23, 0",
                "vslw 24, 24, 1",
                "vsubuwm 20, 20, 24",
                "lxv 55, 192({x})",
                "vmsummbm 21, 23, 15, 0",
                "vsum4sbs 24, 23, 0",
                "vslw 24, 24, 1",
                "vsubuwm 21, 21, 24",
                "lxv 55, 224({x})",
                "vmsummbm 22, 23, 17, 0",
                "vsum4sbs 24, 23, 0",
                "vslw 24, 24, 1",
                "vsubuwm 22, 22, 24",
                "vmrghw 25, 19, 21",
                "vmrglw 26, 19, 21",
                "vadduwm 25, 25, 26",
                "vmrghw 26, 20, 22",
                "vmrglw 27, 20, 22",
                "vadduwm 26, 26, 27",
                "vmrghw 27, 25, 26",
                "vmrglw 28, 25, 26",
                "vadduwm 28, 27, 28",
                "lxv 55, 144({x})",
                "vmsummbm 19, 23, 12, 0",
                "vsum4sbs 24, 23, 0",
                "vslw 24, 24, 1",
                "vsubuwm 19, 19, 24",
                "lxv 55, 176({x})",
                "vmsummbm 20, 23, 14, 0",
                "vsum4sbs 24, 23, 0",
                "vslw 24, 24, 1",
                "vsubuwm 20, 20, 24",
                "lxv 55, 208({x})",
                "vmsummbm 21, 23, 16, 0",
                "vsum4sbs 24, 23, 0",
                "vslw 24, 24, 1",
                "vsubuwm 21, 21, 24",
                "lxv 55, 240({x})",
                "vmsummbm 22, 23, 18, 0",
                "vsum4sbs 24, 23, 0",
                "vslw 24, 24, 1",
                "vsubuwm 22, 22, 24",
                "vmrghw 25, 19, 21",
                "vmrglw 26, 19, 21",
                "vadduwm 25, 25, 26",
                "vmrghw 26, 20, 22",
                "vmrglw 27, 20, 22",
                "vadduwm 26, 26, 27",
                "vmrghw 27, 25, 26",
                "vmrglw 29, 25, 26",
                "vadduwm 29, 27, 29",
                "lxvw4x 62, {s2}, {sc}",
                "vmuluwm 28, 28, 30",
                "lxvw4x 62, {s3}, {sc}",
                "vmuluwm 29, 29, 30",
                "vadduwm 28, 28, 29",
                "xvcvsxwsp 6, 60",
                "lxvw4x 7, {o1}, {xd}",
                "xvmulsp 7, 4, 7",
                "xvmulsp 6, 7, 6",
                "lxvw4x 8, {o1}, {acc}",
                "xvaddsp 8, 8, 6",
                "stxvw4x 8, {o1}, {acc}",
                "add {x}, {x}, {xstr}",
                "add {xd}, {xd}, {gstr}",
                "addi {acc}, {acc}, 32",
                "addi {t}, {t}, -1",
                "cmpdi 0, {t}, 0",
                "bne 0, 3b",
                w = in(reg_nonzero) blk.as_ptr(),
                sc = in(reg_nonzero) sc.as_ptr(),
                d = in(reg_nonzero) d.as_ptr(),
                x = inout(reg_nonzero) tk.x => _,
                xd = inout(reg_nonzero) tk.xd => _,
                acc = inout(reg_nonzero) tk.acc => _,
                nt = in(reg_nonzero) tk.n,
                xstr = in(reg_nonzero) tk.x_stride,
                gstr = in(reg_nonzero) tk.g_stride,
                s0 = in(reg) 0usize,
                s1 = in(reg_nonzero) 16usize,
                s2 = in(reg_nonzero) 32usize,
                s3 = in(reg_nonzero) 48usize,
                o0 = in(reg) 0usize,
                o1 = in(reg_nonzero) 16usize,
                t = out(reg_nonzero) _,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
                out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _,
                out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _,
                out("v15") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                out("v20") _, out("v21") _, out("v22") _, out("v23") _, out("v24") _,
                out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _,
                out("v30") _, out("v31") _,
                out("f4") _, out("f6") _, out("f7") _, out("f8") _,
                options(nostack),
            );
        }
    }
}

/// One IQ4_XS row, one `asm!` block per superblock (`d`, `scales_h`, `scales_l[4]`, `qs[128]`):
/// the codebook values plus 128 (unsigned, `vperm` through the table at `CONSTS[64..80]`) in
/// v3..v18 (group `j`'s two halves in v(3 + 2j), v(4 + 2j)); per token the group dots are
/// `vmsummbm` less `128 * sum(x)` (as Q8_0's), times the sub-scales (`vmuluwm`): the reference's
/// `ls * dot`, then `(d * xd) * isum` added to accumulators `g % 8`.
///
/// # Safety
/// `w` holds whole superblocks; `tk` describes live activations and accumulators.
unsafe fn iq4_xs_row(w: &[u8], tk: Tokens) {
    for (sb, blk) in w
        .as_chunks::<{ QuantFormat::Iq4Xs.block_bytes() }>()
        .0
        .iter()
        .enumerate()
    {
        let ls = iq4_xs_scales(blk);
        let d = [f16_at(blk, 0); 4];
        let tk = tk.superblock(sb);
        // SAFETY: the loads read the superblock's 136 bytes, 256 activation bytes, 8 scales and
        // 8 sums of each token, the token's 8 accumulators, `ls`, `d` and 16 bytes of `CONSTS`,
        // all inside live buffers; every register written is declared clobbered.
        unsafe {
            asm!(
                // Unpack: v2 = the biased codebook, v25 = 15, v26 = 4; group j's low nibbles (values
                // 0..16) through the codebook into v(3 + 2j), its high nibbles into v(4 + 2j).
                "vspltisb 0, 0",
                "vspltisw 1, 7",
                "vspltisb 25, 15",
                "vspltisb 26, 4",
                "lxvb16x 34, {i64}, {c}",
                "addi {q}, {w}, 8",
                "lxvb16x 59, 0, {q}",
                "addi {q}, {q}, 16",
                "vand 28, 27, 25",
                "vperm 3, 2, 2, 28",
                "vsrb 28, 27, 26",
                "vperm 4, 2, 2, 28",
                "lxvb16x 59, 0, {q}",
                "addi {q}, {q}, 16",
                "vand 28, 27, 25",
                "vperm 5, 2, 2, 28",
                "vsrb 28, 27, 26",
                "vperm 6, 2, 2, 28",
                "lxvb16x 59, 0, {q}",
                "addi {q}, {q}, 16",
                "vand 28, 27, 25",
                "vperm 7, 2, 2, 28",
                "vsrb 28, 27, 26",
                "vperm 8, 2, 2, 28",
                "lxvb16x 59, 0, {q}",
                "addi {q}, {q}, 16",
                "vand 28, 27, 25",
                "vperm 9, 2, 2, 28",
                "vsrb 28, 27, 26",
                "vperm 10, 2, 2, 28",
                "lxvb16x 59, 0, {q}",
                "addi {q}, {q}, 16",
                "vand 28, 27, 25",
                "vperm 11, 2, 2, 28",
                "vsrb 28, 27, 26",
                "vperm 12, 2, 2, 28",
                "lxvb16x 59, 0, {q}",
                "addi {q}, {q}, 16",
                "vand 28, 27, 25",
                "vperm 13, 2, 2, 28",
                "vsrb 28, 27, 26",
                "vperm 14, 2, 2, 28",
                "lxvb16x 59, 0, {q}",
                "addi {q}, {q}, 16",
                "vand 28, 27, 25",
                "vperm 15, 2, 2, 28",
                "vsrb 28, 27, 26",
                "vperm 16, 2, 2, 28",
                "lxvb16x 59, 0, {q}",
                "addi {q}, {q}, 16",
                "vand 28, 27, 25",
                "vperm 17, 2, 2, 28",
                "vsrb 28, 27, 26",
                "vperm 18, 2, 2, 28",
                "lxvw4x 4, 0, {d}",
                "mr {t}, {nt}",
                "3:",
                "mr {xp}, {x}",
                // Groups 0..4 -> v29.
                "lxvb16x 55, 0, {xp}",
                "lxvb16x 56, {i16}, {xp}",
                "addi {xp}, {xp}, 32",
                "vmsummbm 19, 23, 3, 0",
                "vmsummbm 19, 24, 4, 19",
                "lxvb16x 55, 0, {xp}",
                "lxvb16x 56, {i16}, {xp}",
                "addi {xp}, {xp}, 32",
                "vmsummbm 20, 23, 5, 0",
                "vmsummbm 20, 24, 6, 20",
                "lxvb16x 55, 0, {xp}",
                "lxvb16x 56, {i16}, {xp}",
                "addi {xp}, {xp}, 32",
                "vmsummbm 21, 23, 7, 0",
                "vmsummbm 21, 24, 8, 21",
                "lxvb16x 55, 0, {xp}",
                "lxvb16x 56, {i16}, {xp}",
                "addi {xp}, {xp}, 32",
                "vmsummbm 22, 23, 9, 0",
                "vmsummbm 22, 24, 10, 22",
                "vmrghw 27, 19, 21",
                "vmrglw 28, 19, 21",
                "vadduwm 27, 27, 28",
                "vmrghw 28, 20, 22",
                "vmrglw 30, 20, 22",
                "vadduwm 28, 28, 30",
                "vmrghw 30, 27, 28",
                "vmrglw 29, 27, 28",
                "vadduwm 29, 30, 29",
                "lxvw4x 62, {o0}, {xs}",
                "vslw 30, 30, 1",
                "vsubuwm 29, 29, 30",
                "lxvw4x 63, {o0}, {ls}",
                "vmuluwm 29, 29, 31",
                "xvcvsxwsp 6, 61",
                "lxvw4x 7, {o0}, {xd}",
                "xvmulsp 7, 4, 7",
                "xvmulsp 6, 7, 6",
                "lxvw4x 8, {o0}, {acc}",
                "xvaddsp 8, 8, 6",
                "stxvw4x 8, {o0}, {acc}",
                // Groups 4..8 -> v29.
                "lxvb16x 55, 0, {xp}",
                "lxvb16x 56, {i16}, {xp}",
                "addi {xp}, {xp}, 32",
                "vmsummbm 19, 23, 11, 0",
                "vmsummbm 19, 24, 12, 19",
                "lxvb16x 55, 0, {xp}",
                "lxvb16x 56, {i16}, {xp}",
                "addi {xp}, {xp}, 32",
                "vmsummbm 20, 23, 13, 0",
                "vmsummbm 20, 24, 14, 20",
                "lxvb16x 55, 0, {xp}",
                "lxvb16x 56, {i16}, {xp}",
                "addi {xp}, {xp}, 32",
                "vmsummbm 21, 23, 15, 0",
                "vmsummbm 21, 24, 16, 21",
                "lxvb16x 55, 0, {xp}",
                "lxvb16x 56, {i16}, {xp}",
                "addi {xp}, {xp}, 32",
                "vmsummbm 22, 23, 17, 0",
                "vmsummbm 22, 24, 18, 22",
                "vmrghw 27, 19, 21",
                "vmrglw 28, 19, 21",
                "vadduwm 27, 27, 28",
                "vmrghw 28, 20, 22",
                "vmrglw 30, 20, 22",
                "vadduwm 28, 28, 30",
                "vmrghw 30, 27, 28",
                "vmrglw 29, 27, 28",
                "vadduwm 29, 30, 29",
                "lxvw4x 62, {o1}, {xs}",
                "vslw 30, 30, 1",
                "vsubuwm 29, 29, 30",
                "lxvw4x 63, {o1}, {ls}",
                "vmuluwm 29, 29, 31",
                "xvcvsxwsp 6, 61",
                "lxvw4x 7, {o1}, {xd}",
                "xvmulsp 7, 4, 7",
                "xvmulsp 6, 7, 6",
                "lxvw4x 8, {o1}, {acc}",
                "xvaddsp 8, 8, 6",
                "stxvw4x 8, {o1}, {acc}",
                "add {x}, {x}, {xstr}",
                "add {xd}, {xd}, {gstr}",
                "add {xs}, {xs}, {gstr}",
                "addi {acc}, {acc}, 32",
                "addi {t}, {t}, -1",
                "cmpdi 0, {t}, 0",
                "bne 0, 3b",
                c = in(reg_nonzero) CONSTS.0.as_ptr(),
                w = in(reg_nonzero) blk.as_ptr(),
                ls = in(reg_nonzero) ls.as_ptr(),
                d = in(reg_nonzero) d.as_ptr(),
                x = inout(reg_nonzero) tk.x => _,
                xd = inout(reg_nonzero) tk.xd => _,
                xs = inout(reg_nonzero) tk.xs => _,
                acc = inout(reg_nonzero) tk.acc => _,
                nt = in(reg_nonzero) tk.n,
                xstr = in(reg_nonzero) tk.x_stride,
                gstr = in(reg_nonzero) tk.g_stride,
                i16 = in(reg_nonzero) 16usize,
                i64 = in(reg_nonzero) 64usize,
                o0 = in(reg) 0usize,
                o1 = in(reg_nonzero) 16usize,
                q = out(reg_nonzero) _,
                xp = out(reg_nonzero) _,
                t = out(reg_nonzero) _,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
                out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _,
                out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _,
                out("v15") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                out("v20") _, out("v21") _, out("v22") _, out("v23") _, out("v24") _,
                out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _,
                out("v30") _, out("v31") _,
                out("f4") _, out("f6") _, out("f7") _, out("f8") _,
                options(nostack),
            );
        }
    }
}

fn bf16_dot(a: &[u16], b: &[u16]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n8 = a.len() / 8;
    let mut acc = [0.0f32; 8];
    if n8 > 0 {
        // SAFETY: the loop reads `8 * n8` halfwords of `a` and of `b` and stores 32 bytes into
        // `acc`; every register written is declared clobbered.
        unsafe {
            asm!(
                // vs0 / vs1: four f32 lanes each. `lxvh8x` keeps the halfwords in memory order;
                // merging them with zero halfwords puts each BF16 in the top half of a word,
                // which is its exact f32. The products are exact, so `xvmaddasp` rounds only
                // the sum.
                "xxlxor 0, 0, 0",
                "xxlxor 1, 1, 1",
                "vspltisb 0, 0",
                "2:",
                "lxvh8x 33, 0, {a}",
                "lxvh8x 34, 0, {b}",
                "vmrghh 3, 1, 0",
                "vmrglh 4, 1, 0",
                "vmrghh 5, 2, 0",
                "vmrglh 6, 2, 0",
                "xvmaddasp 0, 35, 37",
                "xvmaddasp 1, 36, 38",
                "addi {a}, {a}, 16",
                "addi {b}, {b}, 16",
                "addi {n}, {n}, -1",
                "cmpdi 0, {n}, 0",
                "bne 0, 2b",
                "stxvw4x 0, 0, {o}",
                "stxvw4x 1, {i16}, {o}",
                a = inout(reg_nonzero) a.as_ptr() => _,
                b = inout(reg_nonzero) b.as_ptr() => _,
                n = inout(reg_nonzero) n8 => _,
                o = in(reg_nonzero) acc.as_mut_ptr(),
                i16 = in(reg_nonzero) 16usize,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
                out("v5") _, out("v6") _, out("f0") _, out("f1") _,
                options(nostack),
            );
        }
    }
    acc.into_iter().sum::<f32>() + bf16_dot_scalar(&a[8 * n8..], &b[8 * n8..])
}

#[cfg(test)]
mod tests;
