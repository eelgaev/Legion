//! A pass's small input copies as one kernel launch (#814).
//!
//! [`super::PinnedIo`] queues each input as its own `cudaMemcpyAsync` from pinned memory. On
//! Qwen3.8-Flash-Next over four V100s with MTP (C=1), the first stage's ~15 copies per verify
//! step and the last GPU's ~25 (its stage's and the two draft passes') are queued one at a time
//! while the GPUs wait for them: the profile on #814 counted 143 µs of all-GPU idle per step in
//! front of GPU0's small copies and 75 µs in front of GPU3's, most of it the host between calls.
//!
//! Between [`super::PinnedIo::begin_packed`] and [`super::PinnedIo::flush`] the writes only land
//! in the pinned buffer and are noted as [`Piece`]s; the flush launches `input_pack_kernel`
//! (`src/cuda/input_pack.cu`), which reads the pinned memory over the host link and writes every
//! piece into its tensor, on the same stream and ahead of the graph, as the copies were. The bytes
//! are the same, and the graphs are not touched (no tensor added, no layout moved), so outputs are
//! byte-identical. A write that overlaps one still held first launches what is held.
//!
//! On by default; `--no-input-pack` keeps one copy per input. Only where the pinned buffer has a
//! device address (`cudaHostGetDevicePointer`) and the backend is a plain CUDA one.

use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(test)]
mod tests;

static ON: AtomicBool = AtomicBool::new(true);

/// Turns the packed input copies on or off (`--no-input-pack`; on by default).
pub fn set_input_pack(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// Whether [`super::PinnedIo::begin_packed`] packs.
pub(super) fn enabled() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Bytes one thread block of the kernel copies at most (`PACK_CHUNK`).
pub(super) const CHUNK: usize = 4096;
/// Pieces one launch takes (`PACK_MAX`).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(super) const MAX_PIECES: usize = 200;

/// `len` bytes from offset `src` of the pinned buffer into device address `dst` (`Piece` of
/// `src/cuda/input_pack.cu`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(super) struct Piece {
    pub dst: u64,
    pub src: u32,
    pub len: u32,
}

/// The writes held since [`super::PinnedIo::begin_packed`], not launched yet.
#[derive(Default, Debug)]
pub(super) struct Pending {
    pub pieces: Vec<Piece>,
    /// Each write's device range `[start, end)`.
    ranges: Vec<(u64, u64)>,
}

impl Pending {
    /// Holds a write of `len` bytes at pinned offset `src` into device address `dst`, split into
    /// pieces of at most [`CHUNK`] bytes.
    pub fn push(&mut self, dst: u64, src: usize, len: usize) {
        if len == 0 {
            return;
        }
        self.ranges.push((dst, dst + len as u64));
        let mut at = 0;
        while at < len {
            let n = CHUNK.min(len - at);
            self.pieces.push(Piece {
                dst: dst + at as u64,
                src: u32::try_from(src + at).expect("pinned offset fits in u32"),
                len: n as u32,
            });
            at += n;
        }
    }

    /// Whether a write of `len` bytes into `dst` overlaps one held.
    pub fn overlaps(&self, dst: u64, len: usize) -> bool {
        let end = dst + len as u64;
        len > 0 && self.ranges.iter().any(|&(a, b)| dst < b && a < end)
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    pub fn clear(&mut self) {
        self.pieces.clear();
        self.ranges.clear();
    }
}

#[cfg(feature = "cuda")]
pub(super) use cuda::{device_pointer, launch};

#[cfg(feature = "cuda")]
mod cuda {
    use std::ffi::c_void;
    use std::sync::Once;

    use super::{MAX_PIECES, Piece};

    /// `Args` of `src/cuda/input_pack.cu`.
    #[repr(C)]
    struct Args {
        base: *const u8,
        pieces: [Piece; MAX_PIECES],
    }

    const _: () = assert!(std::mem::size_of::<Args>() == 8 + 16 * MAX_PIECES);

    /// `dim3`.
    #[repr(C)]
    struct Dim3 {
        x: u32,
        y: u32,
        z: u32,
    }

    /// The kernel's threads per block (`PACK_THREADS`).
    const THREADS: u32 = 256;
    const _: () = assert!(THREADS as usize * 16 == super::CHUNK);

    /// `cudaMemcpyHostToDevice`.
    const H2D: i32 = 1;

    unsafe extern "C" {
        /// `input_pack_kernel`'s host stub.
        static legion_input_pack_kernel: *const c_void;
        fn cudaLaunchKernel(
            func: *const c_void,
            grid: Dim3,
            block: Dim3,
            args: *mut *mut c_void,
            shared: usize,
            stream: *mut c_void,
        ) -> i32;
        fn cudaHostGetDevicePointer(dev: *mut *mut c_void, host: *mut c_void, flags: u32) -> i32;
        fn cudaMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            n: usize,
            kind: i32,
            stream: *mut c_void,
        ) -> i32;
        fn cudaGetLastError() -> i32;
    }

    /// The device address of pinned host memory at `host` (`cudaMallocHost`), if it has one.
    pub fn device_pointer(host: *const u8) -> Option<usize> {
        let mut dev = std::ptr::null_mut();
        // SAFETY: an out pointer; the call only looks `host` up.
        let e = unsafe { cudaHostGetDevicePointer(&mut dev, host.cast_mut().cast(), 0) };
        if e != 0 {
            // Clear the error: memory that is not pinned is not an error here.
            // SAFETY: no arguments.
            unsafe { cudaGetLastError() };
            return None;
        }
        Some(dev as usize)
    }

    /// Queues `pieces` on `stream` (of the current device): the kernel, [`MAX_PIECES`] pieces a
    /// launch; if a launch fails, its pieces as one `cudaMemcpyAsync` each from `host`.
    ///
    /// # Safety
    /// `base` is the device address of the pinned buffer at `host`; every piece's source range
    /// lies in it and nothing writes it until the stream is synchronized; every piece's
    /// destination is device memory of the stream's device that nothing else writes meanwhile.
    pub unsafe fn launch(stream: *mut c_void, base: usize, host: *const u8, pieces: &[Piece]) {
        static WARN: Once = Once::new();
        for part in pieces.chunks(MAX_PIECES) {
            let mut a = Args {
                base: base as *const u8,
                pieces: [Piece {
                    dst: 0,
                    src: 0,
                    len: 0,
                }; MAX_PIECES],
            };
            a.pieces[..part.len()].copy_from_slice(part);
            let mut argv = [(&raw mut a).cast::<c_void>()];
            let grid = Dim3 {
                x: part.len() as u32,
                y: 1,
                z: 1,
            };
            let block = Dim3 {
                x: THREADS,
                y: 1,
                z: 1,
            };
            // SAFETY: the caller's contract; the launch copies `a`.
            let e = unsafe {
                cudaLaunchKernel(
                    legion_input_pack_kernel,
                    grid,
                    block,
                    argv.as_mut_ptr(),
                    0,
                    stream,
                )
            };
            if e == 0 {
                continue;
            }
            // SAFETY: no arguments.
            unsafe { cudaGetLastError() };
            WARN.call_once(|| {
                tracing::warn!(
                    error = e,
                    "input pack kernel failed to launch; plain copies"
                );
            });
            for p in part {
                // SAFETY: the caller's contract, the same ranges as the kernel's.
                let e = unsafe {
                    cudaMemcpyAsync(
                        p.dst as *mut c_void,
                        host.add(p.src as usize).cast(),
                        p.len as usize,
                        H2D,
                        stream,
                    )
                };
                assert_eq!(e, 0, "input copy failed: CUDA error {e}");
            }
        }
    }
}
