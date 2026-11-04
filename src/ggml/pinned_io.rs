//! Pinned host memory that a small graph's inputs and outputs go through, as copies queued on
//! the backend's stream (#362).
//!
//! [`super::Tensor::set_bytes`] and [`super::Tensor::bytes`] each wait for their copy. On CUDA
//! that means a copy on the per-thread stream plus a `cudaStreamSynchronize`, about 15-25 µs
//! even for a few bytes. An MTP draft pass sets about seven inputs and reads three outputs, so
//! the waits added up to about 0.3 ms of a 4.8 ms two-pass draft window on a V100, with the GPU
//! idle between passes. Through [`PinnedIo`] the copies are queued in order with the graph:
//! inputs before it, outputs after it. One [`PinnedIo::finish`], the synchronize the pass did
//! anyway, waits for all of them. The bytes are the same either way.
//!
//! Elsewhere (meta backends, CPU, tensors outside the backend's own memory, or no pinned
//! memory) every transfer falls back to the plain synchronous call. A fallback read runs at
//! [`PinnedIo::take`], after the graph.
//!
//! [`PinnedIo::begin_packed`] holds a pass's queueable input writes and [`PinnedIo::flush`]
//! copies them all with one kernel launch ([`pack`], #814).

use super::backend::{Backend, Event};
use super::context::{Context, Tensor};
use super::{ffi, types};

pub mod pack;

/// Transfers start at multiples of this in the pinned buffer (CUDA's copies like aligned
/// host addresses; any 4-byte value is aligned).
const ALIGN: usize = 256;

/// Pinned host memory for one backend's small transfers (see the module docs). The caller
/// keeps one per backend.
#[derive(Default)]
pub struct PinnedIo {
    /// One host tensor `pinned_io` in the device's pinned host buffer type, if it has one.
    host: Option<Context>,
    /// Bytes handed out since [`Self::begin`].
    used: usize,
    /// Bytes the queueable transfers since [`Self::begin`] asked for, handed out or not: what
    /// the next [`Self::begin`] grows the buffer to.
    wanted: usize,
    /// Copies may still be queued on the backend.
    pending: bool,
    /// A write that cannot be queued waits for the backend first ([`Self::set_ordered`]).
    ordered: bool,
    /// Between [`Self::begin_packed`] and [`Self::flush`]: the pinned buffer's device address
    /// and the writes held for one kernel launch ([`pack`]).
    pack: Option<(usize, pack::Pending)>,
    /// The pinned buffer's host address and its device address, if it has one (looked up once
    /// per buffer).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    dev_base: Option<(usize, Option<usize>)>,
    /// The queued reads that end at or before this offset have landed ([`Self::wait`]).
    landed: usize,
    /// Events for [`Self::fence`], free for reuse.
    events: Vec<Event>,
    /// [`Self::begin`] calls so far.
    begun: u64,
}

/// The copies queued up to some point ([`PinnedIo::fence`]): [`PinnedIo::wait`] waits for them
/// and not for the work queued after.
#[must_use]
pub struct Fence {
    event: Event,
    /// [`PinnedIo::used`] when it was recorded.
    upto: usize,
    /// [`PinnedIo::begun`] when it was recorded: the offsets are that pass's.
    begun: u64,
}

/// An output read queued by [`PinnedIo::read`]; [`PinnedIo::take`] returns its bytes.
#[must_use]
pub enum Read<'t> {
    /// Queued into the pinned buffer at this offset, this many bytes.
    Queued { at: usize, len: usize },
    /// Read synchronously at [`PinnedIo::take`].
    Later(Tensor<'t>),
}

impl Read<'_> {
    /// Whether the read is queued on the stream (not one made at [`PinnedIo::take`]).
    pub fn is_queued(&self) -> bool {
        matches!(self, Read::Queued { .. })
    }

    /// A queued read, apart from its tensor's lifetime; `None` for one made at
    /// [`PinnedIo::take`].
    pub fn detach(self) -> Option<Read<'static>> {
        match self {
            Read::Queued { at, len } => Some(Read::Queued { at, len }),
            Read::Later(_) => None,
        }
    }
}

impl PinnedIo {
    /// Starts a pass's transfers on `backend`. It first waits for any copies still queued
    /// from the last pass, then grows the pinned buffer to what the last pass's transfers
    /// wanted, up to `max` bytes. The first pass of a bigger shape copies synchronously
    /// whatever does not fit; repeated shapes (decode) then fit.
    pub fn begin(&mut self, backend: &mut Backend, max: usize) {
        self.assert_flushed();
        self.pack = None;
        if self.pending {
            backend.synchronize();
            self.pending = false;
        }
        let want = std::mem::take(&mut self.wanted).min(max);
        self.used = 0;
        self.landed = 0;
        self.begun += 1;
        if self.capacity() < want {
            // Room for a quarter more: bigger batches come.
            self.grow(backend, want + want / 4);
        }
    }

    /// [`Self::begin`], then holds the queueable writes until [`Self::flush`], which copies them
    /// all with one kernel launch ([`pack`], #814; plain [`Self::begin`] where that is off or
    /// cannot run). The caller must [`Self::flush`] before queuing the graph that reads them.
    pub fn begin_packed(&mut self, backend: &mut Backend, max: usize) {
        self.begin(backend, max);
        if pack::enabled()
            && let Some(base) = self.device_base(backend)
        {
            self.pack = Some((base, pack::Pending::default()));
        }
    }

    /// Queues the writes held since [`Self::begin_packed`] (one kernel launch) and ends the
    /// holding: later writes are queued one copy each again. Nothing without
    /// [`Self::begin_packed`].
    pub fn flush(&mut self, backend: &mut Backend) {
        self.launch_held(backend);
        self.pack = None;
    }

    /// Whether writes are held for [`Self::flush`] now.
    pub fn is_packing(&self) -> bool {
        self.pack.is_some()
    }

    /// Launches the held writes; the holding goes on.
    fn launch_held(&mut self, backend: &mut Backend) {
        // SAFETY: reading the data pointer of the live pinned tensor.
        let host = self
            .buffer()
            .map(|t| unsafe { (*t.raw()).data.cast::<u8>().cast_const() });
        let Some((base, held)) = self.pack.as_mut() else {
            return;
        };
        if held.is_empty() {
            return;
        }
        #[cfg(feature = "cuda")]
        {
            let stream = backend
                .cuda_launch_stream()
                .expect("packed writes on a CUDA backend");
            let host = host.expect("packed writes have a pinned buffer");
            // SAFETY: `base` is the buffer's device address (`device_base`); the pieces are
            // claimed ranges of it that stay untouched until the next `begin` or `finish`
            // synchronizes the backend (`pending`), written into tensors in the backend's own
            // memory (`claim`).
            unsafe { pack::launch(stream, *base, host, &held.pieces) };
        }
        // Nothing is held without CUDA (`device_base`).
        #[cfg(not(feature = "cuda"))]
        let _ = (backend, base, host);
        held.clear();
        self.pending = true;
    }

    /// The pinned buffer's device address, where the held writes can be launched as a kernel:
    /// a CUDA backend (not a meta one) and pinned memory the device reads.
    #[cfg(feature = "cuda")]
    fn device_base(&mut self, backend: &Backend) -> Option<usize> {
        backend.cuda_stream()?;
        // SAFETY: reading the data pointer of the live pinned tensor.
        let host = unsafe { (*self.buffer()?.raw()).data } as usize;
        match self.dev_base {
            Some((h, d)) if h == host => d,
            _ => {
                let d = pack::device_pointer(host as *const u8);
                self.dev_base = Some((host, d));
                d
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn device_base(&mut self, _backend: &Backend) -> Option<usize> {
        None
    }

    /// Held writes never launched would leave their tensors stale: a caller bug.
    fn assert_flushed(&self) {
        assert!(
            self.pack.as_ref().is_none_or(|(_, p)| p.is_empty()),
            "PinnedIo: packed input writes were never flushed"
        );
    }

    /// The caller knows the copies queued through this buffer are done (the graph queued after
    /// them has finished): the next [`Self::begin`] does not wait for the backend, which may
    /// run other work by then (`--long-dataflow-async`, two buffers taking turns).
    pub fn settle(&mut self) {
        self.pending = false;
    }

    /// While on, a write that cannot be queued (no pinned room, or memory the backend does
    /// not queue copies of) first waits for all the work queued on the backend: inputs set
    /// while the backend still runs a graph that reads them then stay behind it, as the
    /// queued ones do.
    pub fn set_ordered(&mut self, on: bool) {
        self.ordered = on;
    }

    /// Grows the pinned buffer to at least `bytes` now, unless it has them: a caller whose
    /// passes run beside other GPUs' work reserves what its passes will want before they
    /// start. Freeing and allocating pinned memory waits for the work running on every GPU
    /// (#794: a long prompt's last stage, whose inputs grow with the context, waited up to
    /// 330 ms for the first stage's graph when its buffer grew mid-step). Call it between
    /// passes (after [`Self::finish`] or before the first [`Self::begin`]).
    pub fn reserve(&mut self, backend: &mut Backend, bytes: usize) {
        if self.capacity() < bytes {
            if self.pending {
                backend.synchronize();
                self.pending = false;
            }
            self.grow(backend, bytes);
        }
    }

    /// Replaces the pinned buffer with one of `bytes` (none if the device has no pinned
    /// memory or it is out of it).
    fn grow(&mut self, backend: &Backend, bytes: usize) {
        let t = std::time::Instant::now();
        self.host = None;
        self.host = Self::alloc(backend, bytes);
        tracing::debug!(
            device = %backend.name(),
            mib = self.capacity() as f64 / (1u64 << 20) as f64,
            ms = t.elapsed().as_secs_f64() * 1e3,
            "pinned io buffer grew"
        );
    }

    fn alloc(backend: &Backend, bytes: usize) -> Option<Context> {
        let buft = backend.device().host_buffer_type()?;
        let ctx = Context::new(1, 0);
        let n = bytes.div_ceil(4);
        ctx.new_tensor(types::I32, &[n as i64]).named("pinned_io");
        ctx.alloc_tensors(buft).ok()?;
        Some(ctx)
    }

    /// The pinned buffer's size in bytes, 0 without one.
    pub fn capacity(&self) -> usize {
        self.buffer().map_or(0, |t| t.nbytes())
    }

    fn buffer(&self) -> Option<Tensor<'_>> {
        self.host.as_ref().and_then(|c| c.tensor("pinned_io"))
    }

    /// Whether queued transfers of `lens` bytes each, made now, all fit in what is left of the
    /// buffer since [`Self::begin`] (none falls back to a synchronous copy for want of room).
    /// If they do not, the next [`Self::begin`] grows the buffer for them too.
    pub fn has_room(&mut self, lens: &[usize]) -> bool {
        let need: usize = lens.iter().map(|l| l.next_multiple_of(ALIGN)).sum();
        if need > self.capacity().saturating_sub(self.used) {
            self.wanted += need;
            return false;
        }
        true
    }

    /// `len` bytes of pinned memory for a queued copy of `t`, if `backend` queues copies of
    /// it and they fit; the offset into the buffer.
    fn claim(&mut self, backend: &Backend, t: Tensor<'_>, len: usize) -> Option<usize> {
        if !backend.queues_copies_of(t) {
            return None;
        }
        self.wanted += len.next_multiple_of(ALIGN);
        let room = self.capacity().saturating_sub(self.used);
        if len > room {
            return None;
        }
        let at = self.used;
        self.used += len.next_multiple_of(ALIGN).min(room);
        Some(at)
    }

    /// The pinned bytes `at..at + len`.
    ///
    /// # Safety
    /// No queued copy reads or writes them while the slice lives.
    unsafe fn bytes_mut(&mut self, at: usize, len: usize) -> &mut [u8] {
        let t = self.buffer().expect("pinned buffer");
        assert!(at + len <= t.nbytes(), "pinned_io range out of bounds");
        // SAFETY: an allocated host tensor of at least `at + len` bytes, owned by `self.host`,
        // which only this struct hands out; the caller keeps queued copies off the range.
        unsafe {
            let data = (*t.raw()).data.cast::<u8>();
            std::slice::from_raw_parts_mut(data.add(at), len)
        }
    }

    /// Sets all of input `t` to the `len` bytes that `fill` writes, through pinned memory as a
    /// queued copy when possible.
    fn write_with(
        &mut self,
        backend: &mut Backend,
        t: Tensor<'_>,
        len: usize,
        fill: impl FnOnce(&mut [u8]),
    ) {
        assert_eq!(len, t.nbytes(), "tensor `{}` size", t.name());
        // SAFETY: reading the data pointer of a live tensor (a view's includes its offset).
        let dev = unsafe { (*t.raw()).data } as u64;
        // A write over one still held goes after it (`pack`).
        let over = self
            .pack
            .as_ref()
            .is_some_and(|(_, p)| p.overlaps(dev, len));
        if over {
            self.launch_held(backend);
        }
        match self.claim(backend, t, len) {
            Some(at) => {
                // SAFETY: a range claimed for this copy only, nothing queued on it yet.
                let dst = unsafe { self.bytes_mut(at, len) };
                fill(dst);
                let src = dst.as_ptr();
                if let Some((_, held)) = self.pack.as_mut() {
                    // Copied at `flush`, with the other held writes.
                    held.push(dev, at, len);
                } else {
                    // SAFETY: `queues_copies_of(t)` held (claim); the range stays untouched
                    // until `finish` or the next `begin` synchronizes the backend (`pending`).
                    unsafe { backend.set_async(t, src, len) };
                }
                self.pending = true;
            }
            None => {
                let mut v = vec![0u8; len];
                fill(&mut v);
                if over {
                    // The held write launched above must land first.
                    backend.synchronize();
                    self.pending = false;
                }
                if self.ordered {
                    backend.synchronize();
                    self.pending = false;
                }
                t.set_bytes(&v);
            }
        }
    }

    /// Sets I32 input `t` to `values` (see [`Self::write_with`]).
    pub fn write_i32(&mut self, backend: &mut Backend, t: Tensor<'_>, values: &[i32]) {
        assert_eq!(
            t.ty(),
            ffi::ggml_type_GGML_TYPE_I32,
            "write_i32 on a non-I32 tensor"
        );
        let bytes = as_bytes(values);
        self.write_with(backend, t, bytes.len(), |dst| dst.copy_from_slice(bytes));
    }

    /// Sets F32 input `t` to `values` (see [`Self::write_with`]).
    pub fn write_f32(&mut self, backend: &mut Backend, t: Tensor<'_>, values: &[f32]) {
        self.write_f32_parts(backend, t, &[values]);
    }

    /// Sets input `t` to `bytes` (see [`Self::write_with`]).
    pub fn write_bytes(&mut self, backend: &mut Backend, t: Tensor<'_>, bytes: &[u8]) {
        self.write_with(backend, t, bytes.len(), |dst| dst.copy_from_slice(bytes));
    }

    /// Sets F32 input `t` to `parts` one after another (see [`Self::write_with`]), gathered
    /// straight into the pinned memory.
    pub fn write_f32_parts(&mut self, backend: &mut Backend, t: Tensor<'_>, parts: &[&[f32]]) {
        assert_eq!(
            t.ty(),
            ffi::ggml_type_GGML_TYPE_F32,
            "write_f32 on a non-F32 tensor"
        );
        let len = parts.iter().map(|p| std::mem::size_of_val(*p)).sum();
        self.write_with(backend, t, len, |dst| {
            let mut at = 0;
            for p in parts {
                let b = as_bytes(p);
                dst[at..at + b.len()].copy_from_slice(b);
                at += b.len();
            }
        });
    }

    /// Queues a read of all of output `t` after the work queued on `backend` so far (call it
    /// after starting the graph); [`Self::take`] returns the bytes after [`Self::finish`].
    pub fn read<'t>(&mut self, backend: &mut Backend, t: Tensor<'t>) -> Read<'t> {
        self.assert_flushed();
        let len = t.nbytes();
        match self.claim(backend, t, len) {
            Some(at) => {
                // SAFETY: a range claimed for this copy only; nothing reads it before `finish`
                // (`take` asserts that).
                let dst = unsafe { self.bytes_mut(at, len) }.as_mut_ptr();
                // SAFETY: `queues_copies_of(t)` held (claim); see above for `dst`.
                unsafe { backend.get_async(t, dst, len) };
                self.pending = true;
                Read::Queued { at, len }
            }
            None => Read::Later(t),
        }
    }

    /// Waits for every queued copy and for the graph: the backend's synchronize.
    pub fn finish(&mut self, backend: &mut Backend) {
        self.assert_flushed();
        self.pack = None;
        backend.synchronize();
        self.pending = false;
    }

    /// Marks the copies queued so far on `backend` (since [`Self::begin`]): [`Self::wait`] on
    /// it waits for them, not for the work queued after. `None` where the device has no events
    /// (then [`Self::finish`] is the way to wait).
    pub fn fence(&mut self, backend: &mut Backend) -> Option<Fence> {
        let event = self.events.pop().or_else(|| backend.new_event())?;
        backend.record(&event);
        Some(Fence {
            event,
            upto: self.used,
            begun: self.begun,
        })
    }

    /// Waits for the copies queued before `fence` ([`Self::fence`], since the last
    /// [`Self::begin`]): their reads can be taken, while later work may still run.
    ///
    /// When the fence covers every copy queued through the buffer since that begin (nothing
    /// was claimed or held after it), none is left on the backend: the next [`Self::begin`]
    /// reuses the buffer without waiting for the work queued after the fence. Between two MTP
    /// draft passes that is the last stage's verify commit (`commit_hold`), which the next
    /// pass's input copies then queue behind instead of the host waiting for it (#814).
    pub fn wait(&mut self, fence: Fence) {
        fence.event.synchronize();
        if fence.begun == self.begun {
            self.landed = self.landed.max(fence.upto);
            let held = self.pack.as_ref().is_some_and(|(_, p)| !p.is_empty());
            if covers_all(fence.upto, self.used, held) {
                self.pending = false;
            }
        }
        self.events.push(fence.event);
    }

    /// The bytes of a read (after [`Self::finish`], or a [`Self::wait`] on a fence after it).
    pub fn take(&mut self, r: Read<'_>) -> Vec<u8> {
        assert!(
            !self.pending || matches!(r, Read::Queued { at, len } if at + len <= self.landed),
            "PinnedIo::take before finish"
        );
        match r {
            // SAFETY: the backend is synchronized, or the copies up to `landed` are done: no
            // copy touches the range.
            Read::Queued { at, len } => unsafe { self.bytes_mut(at, len) }.to_vec(),
            Read::Later(t) => t.bytes(),
        }
    }

    /// [`Self::take`] as F32 values.
    pub fn take_f32(&mut self, r: Read<'_>) -> Vec<f32> {
        from_bytes(&self.take(r), f32::from_ne_bytes)
    }

    /// [`Self::take`] as I32 values.
    pub fn take_i32(&mut self, r: Read<'_>) -> Vec<i32> {
        from_bytes(&self.take(r), i32::from_ne_bytes)
    }
}

/// The bytes of 4-byte plain values (`f32`, `i32`) in memory order.
fn as_bytes<T: Copy>(values: &[T]) -> &[u8] {
    const { assert!(std::mem::size_of::<T>() == 4) };
    // SAFETY: the slice's memory, read as bytes; any byte is a valid u8.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn from_bytes<T>(bytes: &[u8], f: fn([u8; 4]) -> T) -> Vec<T> {
    bytes.as_chunks::<4>().0.iter().map(|&c| f(c)).collect()
}

/// Whether a fence recorded with `upto` bytes of the buffer handed out covers all `used`
/// now, with no write `held` for a packed launch: then no copy through the buffer is queued
/// after it ([`PinnedIo::wait`]).
fn covers_all(upto: usize, used: usize, held: bool) -> bool {
    upto >= used && !held
}

#[cfg(test)]
mod tests;
