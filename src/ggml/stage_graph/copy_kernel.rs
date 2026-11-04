//! Copy kernels in place of a big capture's device-to-device memcpy nodes (#836): the kernels
//! and how one is launched for a copy.
//!
//! `cudaGraphExecUpdate` refuses to move a memcpy node's pointers unless the copy is 1-D (and
//! its memory comes from the same context as before), and ggml-cuda captures a same-type copy of
//! strided rows as a 2-D `cudaMemcpy2DAsync`. A verify graph has such a copy into per-sequence
//! state, so a capture of another pool size could never update the executable graph of an
//! earlier one ([`super::spare`]) and was instantiated instead. While spares or steals are on,
//! each device-to-device memcpy node of a big capture is made a kernel node of the PTX below
//! (`src/cuda/ptx/stage_copy.ptx`) with the copy's operands as its parameters, which an update
//! may change. The kernels move the same bytes, so the output bits do not change.

#[cfg(test)]
mod tests;

/// The kernels' PTX text.
pub(crate) static MODULE: crate::ptx::Module =
    crate::ptx::Module::new("stage_copy", include_str!("../../cuda/ptx/stage_copy.ptx"));

/// The kernels by element size: 16, 4 and 1 bytes.
pub(crate) const B16: &str = "legion_stage_copy_b16";
pub(crate) const B4: &str = "legion_stage_copy_b4";
pub(crate) const B1: &str = "legion_stage_copy_b1";

/// Threads per CTA.
pub(crate) const THREADS: u32 = 256;
/// At most this many CTAs along a row, and in all (the rest is strided by the grid).
const MAX_X: u64 = 1024;
const MAX_CTAS: u64 = 4096;
/// The grid's y limit.
const MAX_Y: u64 = 65535;

/// A copy of `rows` rows of `width` bytes, row `r` from `src + r * spitch` to
/// `dst + r * dpitch` (device addresses).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Copy2d {
    pub dst: u64,
    pub dpitch: u64,
    pub src: u64,
    pub spitch: u64,
    pub width: u64,
    pub rows: u64,
}

/// A kernel launch for a [`Copy2d`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Launch {
    /// The kernel: the widest element every address and pitch is aligned to.
    pub entry: &'static str,
    /// Elements per row.
    pub n: u64,
    pub grid: [u32; 3],
    pub block: [u32; 3],
}

/// The launch for copy `c`; `None` for an empty copy, or rows wider than their pitch.
pub(crate) fn plan(c: Copy2d) -> Option<Launch> {
    if c.width == 0 || c.rows == 0 {
        return None;
    }
    // One row: the pitches are never used.
    let pitches = if c.rows > 1 {
        if c.dpitch < c.width || c.spitch < c.width {
            return None;
        }
        c.dpitch | c.spitch
    } else {
        0
    };
    let align = c.dst | c.src | c.width | pitches;
    let (entry, size) = if align.is_multiple_of(16) {
        (B16, 16)
    } else if align.is_multiple_of(4) {
        (B4, 4)
    } else {
        (B1, 1)
    };
    let n = c.width / size;
    let gx = n.div_ceil(THREADS as u64).min(MAX_X);
    let gy = c.rows.min((MAX_CTAS / gx).max(1)).min(MAX_Y);
    Some(Launch {
        entry,
        n,
        grid: [gx as u32, gy as u32, 1],
        block: [THREADS, 1, 1],
    })
}
