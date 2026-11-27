//! The lent expert slots' VRAM handed to whoever maps next (#37): a long prompt's lend unmaps
//! the slots and, instead of releasing their allocations, keeps them here; the prompt's step
//! compute buffers map them ([`lent_type`]), and the restore after it maps them back into the
//! slots ([`take_exact`]). Neither pays the driver clearing freed VRAM before it can be had
//! again (a release followed by a `cuMemCreate` of the same bytes: ~0.9 ms per 48 MiB on an
//! RTX 4090, the bulk of a lend's unmapping and of the restore's mapping).
//!
//! What the pool holds counts as free VRAM ([`super::vram`], [`super::free_vram`]): whatever
//! plans a lend's steps sees the same free memory as when the lend released it. Everything
//! that needs plain free VRAM while the pool holds some (the KV tail's pieces, the spill
//! buffers, a step buffer's rest) releases pooled allocations first ([`make_raw`], and one
//! more after each out of memory in [`create`]); a lent
//! step's allocation ([`flush`]) and the end of the restore ([`close_all`]) release what is
//! left, so nothing but the pool's own takers ever waits on it.
//!
//! `LEGION_DEBUG_LEND_HANDOFF=0` releases the slots' allocations at once, as before.

use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// Pooled allocations: `(device, handle, bytes)`, unmapped.
static POOL: Mutex<Vec<(usize, CUmemGenericAllocationHandle, usize)>> = Mutex::new(Vec::new());

/// The devices (bit `device`) whose pool takes allocations: from a lend's first one until the
/// restore after it is done ([`close_all`]).
static OPEN: AtomicU64 = AtomicU64::new(0);

/// Plain free VRAM a pool's taker leaves on top of what it asked for: not all of what the
/// driver reports free is to be had ([`shed`]).
const RAW_SLACK: usize = 256 << 20;

fn pool() -> std::sync::MutexGuard<'static, Vec<(usize, CUmemGenericAllocationHandle, usize)>> {
    POOL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether the hand-over is on (`LEGION_DEBUG_LEND_HANDOFF=0`: off).
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LEGION_DEBUG_LEND_HANDOFF").map_or(true, |v| v != "0"))
}

fn bit(device: usize) -> u64 {
    1u64.checked_shl(device as u32).unwrap_or(0)
}

/// Whether `device`'s pool takes allocations (a lend gave it some, the restore is not done).
pub(crate) fn is_open(device: usize) -> bool {
    OPEN.load(Ordering::Acquire) & bit(device) != 0
}

/// Bytes `device`'s pool holds.
pub(crate) fn bytes(device: usize) -> usize {
    if OPEN.load(Ordering::Acquire) == 0 {
        return 0;
    }
    pool().iter().filter(|p| p.0 == device).map(|p| p.2).sum()
}

fn release(device: usize, parts: impl IntoIterator<Item = (CUmemGenericAllocationHandle, usize)>) {
    for (h, n) in parts {
        // SAFETY: an unmapped allocation the pool owned.
        timed("cuMemRelease", device, n, || unsafe { cuMemRelease(h) });
    }
}

/// Takes the lent slots' allocations `mem` (unmapped, of one device) into the pool, which
/// opens for the device; released at once with the hand-over off.
pub(crate) fn put(mem: DeviceMem) {
    let device = mem.device;
    let parts = mem.into_parts();
    if !enabled() || bit(device) == 0 {
        release(device, parts);
        return;
    }
    OPEN.fetch_or(bit(device), Ordering::AcqRel);
    pool().extend(parts.into_iter().map(|(h, n)| (device, h, n)));
}

/// Gives a step buffer's unmapped allocations back: into the pool while it is open, else
/// released.
pub(crate) fn give_back(device: usize, parts: Vec<(CUmemGenericAllocationHandle, usize)>) {
    if is_open(device) {
        pool().extend(parts.into_iter().map(|(h, n)| (device, h, n)));
    } else {
        release(device, parts);
    }
}

/// A pooled allocation of `device` of exactly `n` bytes, if there is one.
pub(crate) fn take_exact(device: usize, n: usize) -> Option<CUmemGenericAllocationHandle> {
    if !is_open(device) {
        return None;
    }
    let mut p = pool();
    let i = p.iter().position(|x| x.0 == device && x.2 == n)?;
    Some(p.swap_remove(i).1)
}

/// Pooled allocations of `device` of at most `n` bytes in all, the largest first.
pub(crate) fn take_upto(device: usize, n: usize) -> Vec<(CUmemGenericAllocationHandle, usize)> {
    let mut out = Vec::new();
    if !is_open(device) {
        return out;
    }
    let mut p = pool();
    let mut mine: Vec<usize> = (0..p.len()).filter(|&i| p[i].0 == device).collect();
    mine.sort_by_key(|&i| std::cmp::Reverse(p[i].2));
    let mut left = n;
    let mut taken = Vec::new();
    for i in mine {
        if p[i].2 <= left {
            left -= p[i].2;
            taken.push(i);
        }
    }
    taken.sort_unstable_by(|a, b| b.cmp(a));
    for i in taken {
        let (_, h, n) = p.swap_remove(i);
        out.push((h, n));
    }
    out
}

/// Releases pooled allocations of `device` until at least `n` bytes (and [`RAW_SLACK`]) of
/// it are free outside the pool, for a `cuMemCreate` or `cudaMalloc` of them.
pub(crate) fn make_raw(device: usize, n: usize) {
    if !is_open(device) {
        return;
    }
    let raw = raw_free_vram(device).unwrap_or(0);
    let mut short = n.saturating_add(RAW_SLACK).saturating_sub(raw);
    if short == 0 {
        return;
    }
    let mut gone = Vec::new();
    {
        let mut p = pool();
        // The largest first: a step buffer's rest (no slot's size) before the slots' own,
        // which the restore maps again.
        while short > 0 {
            let pick = p
                .iter()
                .enumerate()
                .filter(|(_, x)| x.0 == device)
                .max_by_key(|(_, x)| x.2)
                .map(|(i, _)| i);
            let Some(i) = pick else { break };
            let (_, h, b) = p.swap_remove(i);
            short = short.saturating_sub(b);
            gone.push((h, b));
        }
    }
    if gone.is_empty() {
        return;
    }
    tracing::debug!(
        device,
        n_mib = n >> 20,
        raw_mib = raw >> 20,
        released_mib = gone.iter().map(|g| g.1).sum::<usize>() >> 20,
        "lent VRAM released for an allocation"
    );
    release(device, gone);
}

/// How long [`shed`] waits for released VRAM to show as free at most.
const RAW_WAIT: std::time::Duration = std::time::Duration::from_millis(100);

/// `cuMemCreate` of `n` bytes of `device` for `h` after [`make_raw`]; out of memory, tried
/// again after each pooled allocation released ([`shed`]) until the pool is empty.
///
/// # Safety
/// As `cuMemCreate`.
pub(crate) unsafe fn create(
    device: usize,
    h: &mut CUmemGenericAllocationHandle,
    n: usize,
    prop: &CUmemAllocationProp,
) -> CUresult {
    make_raw(device, n);
    loop {
        // SAFETY: the caller's.
        let r = timed("cuMemCreate", device, n, || unsafe {
            cuMemCreate(h, n, prop, 0)
        });
        if r != CUDA_ERROR_OUT_OF_MEMORY || bytes(device) == 0 {
            if r == CUDA_ERROR_OUT_OF_MEMORY {
                tracing::debug!(
                    device,
                    mib = n >> 20,
                    raw_mib = raw_free_vram(device).unwrap_or(0) >> 20,
                    pooled_mib = bytes(device) >> 20,
                    "lent VRAM: out of memory"
                );
            }
            return r;
        }
        // Free VRAM the driver reports is not all to be had (out of memory with ~200 MiB
        // reported free for a 66 MiB allocation, the restore after a 32K prompt on an RTX
        // 4090): the largest pooled allocation goes whatever it reports.
        shed(device);
    }
}

/// Releases the largest pooled allocation of `device` and waits until the driver reports its
/// bytes free (at most [`RAW_WAIT`]).
fn shed(device: usize) {
    let raw = raw_free_vram(device).unwrap_or(0);
    let gone = {
        let mut p = pool();
        let pick = p
            .iter()
            .enumerate()
            .filter(|(_, x)| x.0 == device)
            .max_by_key(|(_, x)| x.2)
            .map(|(i, _)| i);
        pick.map(|i| p.swap_remove(i))
    };
    let Some((_, h, n)) = gone else { return };
    release(device, [(h, n)]);
    let t = std::time::Instant::now();
    while raw_free_vram(device).unwrap_or(usize::MAX) < raw + n && t.elapsed() < RAW_WAIT {
        std::thread::sleep(std::time::Duration::from_micros(200));
    }
}

/// Releases what `device`'s pool holds (the pool stays open for the restore); the bytes.
pub(crate) fn flush(device: usize) -> usize {
    if !is_open(device) {
        return 0;
    }
    let gone: Vec<_> = {
        let mut p = pool();
        let (gone, keep) = std::mem::take(&mut *p)
            .into_iter()
            .partition(|x| x.0 == device);
        *p = keep;
        gone
    };
    let n: usize = gone.iter().map(|x| x.2).sum();
    if n > 0 {
        tracing::debug!(device, mib = n >> 20, "lent VRAM nobody mapped released");
    }
    release(device, gone.into_iter().map(|(_, h, n)| (h, n)));
    n
}

/// Closes every device's pool (the restore is done) and releases what is left in it.
pub fn close_all() {
    let open = OPEN.swap(0, Ordering::AcqRel);
    if open == 0 {
        return;
    }
    let gone = std::mem::take(&mut *pool());
    for (device, h, n) in gone {
        release(device, [(h, n)]);
    }
}
