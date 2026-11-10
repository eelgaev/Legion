//! Devices and backends: what ggml can compute on, and handles to compute with.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::ptr::NonNull;
use std::sync::Mutex;

use super::context::{Graph, Tensor};
use super::{GgmlError, ffi};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Gpu,
    /// A GPU sharing memory with the host (e.g. GB10 on DGX Spark).
    IntegratedGpu,
    Accelerator,
    Other,
}

/// A compute device from ggml's backend registry. Devices live for the whole process.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Device(NonNull<ffi::ggml_backend_device>);

// SAFETY: registry devices are created once, never freed, and their getters are thread-safe.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("name", &self.name())
            .field("kind", &self.kind())
            .finish()
    }
}

/// Every device ggml knows about, CPU included, in registry order.
pub fn devices() -> Vec<Device> {
    // SAFETY: plain registry queries; indices are in range.
    unsafe {
        (0..ffi::ggml_backend_dev_count())
            .filter_map(|i| NonNull::new(ffi::ggml_backend_dev_get(i)).map(Device))
            .collect()
    }
}

/// The first device of a kind, if any.
pub fn device_by_kind(kind: DeviceKind) -> Option<Device> {
    devices().into_iter().find(|d| d.kind() == kind)
}

/// The first discrete GPU, else the first GPU sharing memory with the host (the GB10 reports
/// itself as integrated), as the engine's layout picks them.
pub fn gpu_device() -> Option<Device> {
    device_by_kind(DeviceKind::Gpu).or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
}

fn str_of(p: *const std::os::raw::c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: ggml returns NUL-terminated strings that outlive the call.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

impl Device {
    pub(crate) fn raw(self) -> ffi::ggml_backend_dev_t {
        self.0.as_ptr()
    }

    /// A device ggml made outside the registry (a meta device, [`super::meta`]).
    ///
    /// # Safety
    /// `raw` must be a ggml backend device that stays valid for the rest of the process
    /// (`Device` is `Copy` and never frees it), as registry devices do.
    pub(crate) unsafe fn from_raw(raw: NonNull<ffi::ggml_backend_device>) -> Self {
        Self(raw)
    }

    pub fn name(self) -> String {
        str_of(unsafe { ffi::ggml_backend_dev_name(self.raw()) })
    }

    /// The device's description (a GPU's name), asked once per device: ggml-cuda reads it
    /// from `cudaGetDeviceProperties` (milliseconds on a GB10, #827) on every call.
    pub fn description(self) -> String {
        static KNOWN: Mutex<Option<HashMap<usize, String>>> = Mutex::new(None);
        let key = self.raw() as usize;
        if let Some(d) = KNOWN.lock().unwrap().get_or_insert_default().get(&key) {
            return d.clone();
        }
        let d = str_of(unsafe { ffi::ggml_backend_dev_description(self.raw()) });
        KNOWN
            .lock()
            .unwrap()
            .get_or_insert_default()
            .insert(key, d.clone());
        d
    }

    /// The kind of device, asked once per device: ggml-cuda tells a GPU from an integrated
    /// one by `cudaGetDeviceProperties` (12.4 ms on a GB10, #827) on every call, and a step
    /// graph's build asks it.
    pub fn kind(self) -> DeviceKind {
        static KNOWN: Mutex<Option<HashMap<usize, DeviceKind>>> = Mutex::new(None);
        let key = self.raw() as usize;
        if let Some(&k) = KNOWN.lock().unwrap().get_or_insert_default().get(&key) {
            return k;
        }
        let k = match unsafe { ffi::ggml_backend_dev_type(self.raw()) } {
            ffi::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU => DeviceKind::Cpu,
            ffi::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU => DeviceKind::Gpu,
            ffi::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU => DeviceKind::IntegratedGpu,
            ffi::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_ACCEL => DeviceKind::Accelerator,
            _ => DeviceKind::Other,
        };
        KNOWN.lock().unwrap().get_or_insert_default().insert(key, k);
        k
    }

    /// `(free, total)` bytes of device memory as ggml reports it.
    pub fn memory(self) -> (usize, usize) {
        #[cfg(feature = "cuda")]
        if let Some(m) = self
            .discrete_cuda()
            .and_then(|c| crate::residency::cuda::vram(c).ok())
        {
            return m;
        }
        #[cfg(feature = "cuda")]
        if let Some((free, total)) = self
            .uma_cuda()
            .and_then(|c| crate::residency::cuda::vram(c).ok())
        {
            // As ggml-cuda reports a UMA device's memory, without its device properties query.
            let free = std::fs::read("/proc/meminfo")
                .ok()
                .and_then(|m| super::uma_memory::free_bytes(&m))
                .unwrap_or(free);
            return (free, total);
        }
        let (mut free, mut total) = (0usize, 0usize);
        unsafe { ffi::ggml_backend_dev_memory(self.raw(), &mut free, &mut total) };
        (free, total)
    }

    /// The CUDA index of an integrated GPU (the GB10) that is not split into virtual devices
    /// (`GGML_CUDA_DEVICES`), decided once per device: [`Self::memory`] reads its memory as
    /// ggml-cuda does (`cudaMemGetInfo`, then `/proc/meminfo`, see `super::uma_memory`), without
    /// the `cudaGetDeviceProperties` ggml-cuda asks on every memory query (12.4 ms on a GB10,
    /// #827).
    #[cfg(feature = "cuda")]
    fn uma_cuda(self) -> Option<usize> {
        static KNOWN: Mutex<Option<HashMap<usize, Option<usize>>>> = Mutex::new(None);
        let key = self.raw() as usize;
        if let Some(&c) = KNOWN.lock().unwrap().get_or_insert_default().get(&key) {
            return c;
        }
        let plain = std::env::var_os("GGML_CUDA_DEVICES").is_none();
        let c = self
            .name()
            .strip_prefix("CUDA")
            .and_then(|n| n.parse().ok())
            .filter(|_| plain && self.kind() == DeviceKind::IntegratedGpu);
        KNOWN.lock().unwrap().get_or_insert_default().insert(key, c);
        c
    }

    /// The CUDA index of a discrete GPU whose memory ggml reports as the driver does, decided
    /// once per device. ggml-cuda asks `cudaGetDeviceProperties` (1-3 ms on a V100) on every
    /// memory query, which every long-prompt step graph's allocation makes (twice per pipeline
    /// round on the bottleneck stage); [`Self::memory`] asks the driver directly for these.
    /// Not for integrated GPUs or forced unified memory (ggml reports host memory), nor for
    /// virtual devices (`GGML_CUDA_DEVICES`: ggml splits a GPU's memory between them).
    #[cfg(feature = "cuda")]
    pub(crate) fn discrete_cuda(self) -> Option<usize> {
        static KNOWN: Mutex<Option<HashMap<usize, Option<usize>>>> = Mutex::new(None);
        let key = self.raw() as usize;
        if let Some(&c) = KNOWN.lock().unwrap().get_or_insert_default().get(&key) {
            return c;
        }
        let plain = std::env::var_os("GGML_CUDA_DEVICES").is_none()
            && std::env::var_os("GGML_CUDA_ENABLE_UNIFIED_MEMORY").is_none();
        let c = self
            .name()
            .strip_prefix("CUDA")
            .and_then(|n| n.parse().ok())
            .filter(|_| plain && self.kind() == DeviceKind::Gpu);
        KNOWN.lock().unwrap().get_or_insert_default().insert(key, c);
        c
    }

    /// Without CUDA there is no discrete CUDA GPU.
    #[cfg(not(feature = "cuda"))]
    pub(crate) fn discrete_cuda(self) -> Option<usize> {
        None
    }

    /// The device's own memory.
    pub fn buffer_type(self) -> BufferType {
        BufferType(
            NonNull::new(unsafe { ffi::ggml_backend_dev_buffer_type(self.raw()) })
                .expect("device without a buffer type"),
        )
    }

    /// Pinned host memory the device reads fast, if the device has such a type.
    pub fn host_buffer_type(self) -> Option<BufferType> {
        NonNull::new(unsafe { ffi::ggml_backend_dev_host_buffer_type(self.raw()) }).map(BufferType)
    }

    /// Starts a backend (a stream / thread pool) on this device.
    pub fn init(self) -> Result<Backend, GgmlError> {
        let raw = unsafe { ffi::ggml_backend_dev_init(self.raw(), std::ptr::null()) };
        let raw = NonNull::new(raw).ok_or_else(|| GgmlError::Init(self.name()))?;
        // Asked once here: CUDA's device type query reads the device properties, slow enough
        // (0.2-8 ms seen) to dominate a draft pass if asked per transfer.
        // SAFETY: a live backend; the calls read its interface and default buffer type.
        let queued_buft = unsafe {
            let gpu = matches!(self.kind(), DeviceKind::Gpu | DeviceKind::IntegratedGpu);
            if gpu && !ffi::ggml_backend_is_meta(raw.as_ptr()) {
                NonNull::new(ffi::ggml_backend_get_default_buffer_type(raw.as_ptr()))
            } else {
                None
            }
        };
        Ok(Backend {
            raw,
            device: self,
            kept_state: KeptState::default(),
            queued_buft,
        })
    }
}

/// A kind of memory tensors can live in (device VRAM, pinned host, plain host).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BufferType(pub(crate) NonNull<ffi::ggml_backend_buffer_type>);

// SAFETY: buffer types are static descriptors owned by their backend registry.
unsafe impl Send for BufferType {}
unsafe impl Sync for BufferType {}

impl std::fmt::Debug for BufferType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BufferType({})", self.name())
    }
}

impl BufferType {
    pub fn name(self) -> String {
        str_of(unsafe { ffi::ggml_backend_buft_name(self.0.as_ptr()) })
    }

    pub fn is_host(self) -> bool {
        unsafe { ffi::ggml_backend_buft_is_host(self.0.as_ptr()) }
    }

    /// True for ggml's plain CPU buffer type (not pinned host memory of a device).
    pub fn is_plain_cpu(self) -> bool {
        self.0.as_ptr() == unsafe { ffi::ggml_backend_cpu_buffer_type() }
    }

    pub(crate) fn raw(self) -> ffi::ggml_backend_buffer_type_t {
        self.0.as_ptr()
    }
}

/// A point in the work queued on a backend's stream ([`Backend::record`]); the host can wait
/// for the work before it, and not the work queued after it ([`Event::synchronize`]).
pub struct Event(NonNull<ffi::ggml_backend_event>);

// SAFETY: as a backend: used from one thread at a time.
unsafe impl Send for Event {}

impl Event {
    /// Waits for the work queued before the last [`Backend::record`] of this event.
    pub fn synchronize(&self) {
        // SAFETY: a live event.
        unsafe { ffi::ggml_backend_event_synchronize(self.0.as_ptr()) }
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: a live event, freed once.
        unsafe { ffi::ggml_backend_event_free(self.0.as_ptr()) }
    }
}

/// A started backend. Graphs whose tensors live in memory this backend can use run on it.
pub struct Backend {
    raw: NonNull<ffi::ggml_backend>,
    device: Device,
    /// On a meta backend, which kept graphs' per-device tensors may have been dropped (see
    /// [`Self::ready_kept`]).
    kept_state: KeptState,
    /// On a GPU backend that is not a meta one, its default buffer type: the memory whose
    /// tensors [`Self::set_async`] / [`Self::get_async`] queue copies of. Fixed at init.
    queued_buft: Option<NonNull<ffi::ggml_backend_buffer_type>>,
}

/// What a meta backend's computes did to its buffers' per-device tensors. A meta buffer keeps
/// the per-device tensors that graphs make in it (intermediates, inputs, views of its static
/// tensors) only until graphs using it have been rebuilt: so a kept graph needs initializing
/// again once another graph ran over a buffer it keeps tensors in, and only then.
#[derive(Default)]
struct KeptState {
    /// Bumped by every compute.
    epoch: u64,
    /// The last compute of a graph that is not kept (which may use any buffer).
    plain: u64,
    /// When each kept graph ([`super::KeptGraph::serial`]) was last made ready.
    ready: std::collections::HashMap<u64, u64>,
    /// Per buffer: the last two computes over it by different kept graphs, `(epoch, serial)`,
    /// latest first.
    used: std::collections::HashMap<usize, [(u64, u64); 2]>,
    /// Re-initializations so far, and the tensors they initialized again.
    reinits: (u64, u64),
}

impl KeptState {
    /// The buffers in which `kept`'s per-device tensors may have been dropped since it was last
    /// made ready (all of its buffers if it never was, or a plain graph ran since).
    fn stale(&self, kept: &super::KeptGraph) -> Vec<usize> {
        let resident = kept.resident_buffers();
        let Some(&ready) = self.ready.get(&kept.serial()) else {
            return resident.to_vec();
        };
        if self.plain > ready {
            return resident.to_vec();
        }
        resident
            .iter()
            .copied()
            .filter(|b| {
                self.used.get(b).is_some_and(|last| {
                    last.iter()
                        .any(|&(epoch, serial)| serial != kept.serial() && epoch > ready)
                })
            })
            .collect()
    }

    fn computed(&mut self, kept: Option<&super::KeptGraph>) {
        self.epoch += 1;
        let Some(kept) = kept else {
            self.plain = self.epoch;
            return;
        };
        let (epoch, serial) = (self.epoch, kept.serial());
        for &b in kept.used_buffers() {
            let last = self.used.entry(b).or_insert([(0, u64::MAX); 2]);
            if last[0].1 == serial {
                last[0].0 = epoch;
            } else {
                last[1] = last[0];
                last[0] = (epoch, serial);
            }
        }
    }
}

// SAFETY: a backend may move between threads; it is used from one thread at a time (&mut).
unsafe impl Send for Backend {}

impl Drop for Backend {
    fn drop(&mut self) {
        unsafe { ffi::ggml_backend_free(self.raw.as_ptr()) }
    }
}

impl Backend {
    /// The CPU backend.
    pub fn cpu() -> Result<Self, GgmlError> {
        device_by_kind(DeviceKind::Cpu)
            .ok_or_else(|| GgmlError::Init("cpu".into()))?
            .init()
    }

    pub fn name(&self) -> String {
        str_of(unsafe { ffi::ggml_backend_name(self.raw.as_ptr()) })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// The memory this backend computes in by default.
    pub fn buffer_type(&self) -> BufferType {
        BufferType(
            NonNull::new(unsafe { ffi::ggml_backend_get_default_buffer_type(self.raw.as_ptr()) })
                .expect("backend without a buffer type"),
        )
    }

    /// Worker threads for the CPU backend. No effect on other backends.
    pub fn set_threads(&mut self, n: usize) {
        if self.device.kind() == DeviceKind::Cpu {
            unsafe { ffi::ggml_backend_cpu_set_n_threads(self.raw.as_ptr(), n.max(1) as i32) }
        }
    }

    /// CPU backend only: run reference kernels (no llamafile GEMM, no op fusion). Their results
    /// do not depend on how many tokens a batch has, so a one-token decode step reproduces the
    /// rows of a batched pass bit for bit (the default kernels switch from GEMV to tiled GEMM
    /// at two tokens, which sums in another order). Slower prefill; for exactness tests.
    pub fn set_use_ref(&mut self, on: bool) {
        if self.device.kind() == DeviceKind::Cpu {
            unsafe { ffi::ggml_backend_cpu_set_use_ref(self.raw.as_ptr(), on) }
        }
    }

    /// Runs a kept graph to completion ([`Self::compute`]). On a meta backend (tensor
    /// parallelism) a kept graph other than the last one computed is initialized again first
    /// (see [`super::KeptGraph::reinit`]).
    pub fn compute_kept(&mut self, kept: &super::KeptGraph) -> Result<(), GgmlError> {
        self.compute_kept_with(kept, true, None)
    }

    /// [`Self::compute_kept`] without waiting ([`Self::compute_async`]).
    pub fn compute_kept_async(&mut self, kept: &super::KeptGraph) -> Result<(), GgmlError> {
        self.compute_kept_with(kept, false, None)
    }

    /// [`Self::compute_kept`] (to completion when `sync`, else [`Self::compute_kept_async`])
    /// with an input filled late: on CUDA, `late`'s fill runs once the nodes before the input's
    /// first reader are launched, so it overlaps them (the per-layer n-gram rows of qwen4exp,
    /// read from the SSD, overlap its first layer). Elsewhere it runs before the graph.
    pub fn compute_kept_with(
        &mut self,
        kept: &super::KeptGraph,
        sync: bool,
        late: Option<LateInput<'_>>,
    ) -> Result<(), GgmlError> {
        self.prepare_kept(kept)?;
        let plain = self.kept_state.plain;
        self.kept_state.computed(None);
        // SAFETY: a live backend and an allocated kept graph, used from this thread only (&mut).
        let r = unsafe {
            run_graph(
                self.raw.as_ptr(),
                kept.graph().raw(),
                sync,
                late,
                Some(kept.kept_run()),
            )
        };
        self.kept_state.plain = plain;
        self.kept_state.computed(Some(kept));
        r
    }

    /// Records `kept`'s stage-graph captures now, without running it, for its first
    /// [`Self::compute_kept_with`] (given `late` as its late input's tensor, if any) to launch
    /// rather than run its kernels one by one (#836: a verify graph built ahead). Work queued
    /// before keeps running meanwhile. Whether captures were made: only on a CUDA backend (not
    /// a meta one) with stage graphs on and early captures on; elsewhere nothing happens.
    pub fn precapture_kept(&mut self, kept: &super::KeptGraph, late: Option<Tensor<'_>>) -> bool {
        #[cfg(feature = "cuda")]
        {
            let raw = self.raw.as_ptr();
            // SAFETY: a live backend; the calls read its interface.
            let cuda = unsafe { ffi::ggml_backend_is_cuda(raw) } && !self.is_meta();
            if cuda && super::stage_graph::enabled() && !super::op_profile::enabled() {
                // SAFETY: a live CUDA backend and an allocated kept graph (`late` one of its
                // tensors), used from this thread only (&mut); nothing runs it meanwhile.
                return unsafe {
                    super::stage_graph::precapture(raw, kept.graph().raw(), late.map(|t| t.raw()))
                };
            }
        }
        let _ = (kept, late);
        false
    }

    /// On a meta backend: kept-graph re-initializations so far ([`Self::ready_kept`]), and
    /// the tensors they initialized again.
    pub fn reinit_counts(&self) -> (u64, u64) {
        self.kept_state.reinits
    }

    /// Whether this is a meta backend (tensor parallelism, [`super::meta`]).
    pub fn is_meta(&self) -> bool {
        // SAFETY: a live backend; the call only reads its interface pointer.
        unsafe { ffi::ggml_backend_is_meta(self.raw.as_ptr()) }
    }

    fn prepare_kept(&mut self, kept: &super::KeptGraph) -> Result<(), GgmlError> {
        self.ready_kept(kept)
    }

    /// Makes `kept` ready for its inputs to be set and to run: on a meta backend, a kept graph
    /// is initialized again ([`super::KeptGraph::reinit`]) when another graph ran over a
    /// buffer it keeps per-device tensors in since it was last made ready, since the meta
    /// buffer drops those tensors (inputs included) then. Graphs with separate compute buffers
    /// that only read each other's static tensors (the target's steps and the MTP head's)
    /// alternate without it. Call it before setting the inputs of a kept graph that may not be
    /// the last one run; [`Self::compute_kept`] calls it too.
    pub fn ready_kept(&mut self, kept: &super::KeptGraph) -> Result<(), GgmlError> {
        if !self.is_meta() {
            return Ok(());
        }
        let stale = self.kept_state.stale(kept);
        if !stale.is_empty() || !self.kept_state.ready.contains_key(&kept.serial()) {
            let t = std::time::Instant::now();
            let n = kept.reinit_where(|b| stale.binary_search(&b).is_ok())?;
            super::meta::count_reinit(t.elapsed());
            self.kept_state.reinits.0 += 1;
            self.kept_state.reinits.1 += n as u64;
            let epoch = self.kept_state.epoch;
            self.kept_state.ready.insert(kept.serial(), epoch);
            // A new id: the meta backend rebuilds its per-device subgraphs over the tensors
            // initialized now, and reuses them for the runs of this graph that follow with
            // nothing else run in between (see `super::graph_uid`).
            // SAFETY: a live kept graph, not running (this backend is idle for it: &mut).
            unsafe { super::graph_uid::set_uid(kept.graph().raw(), super::graph_uid::next_uid()) };
        }
        Ok(())
    }

    /// Runs `graph` to completion. Every tensor in it must be allocated (see
    /// [`super::Context::alloc_tensors`] and [`super::Context::alloc_graph`]).
    ///
    /// External ops ([`super::ExternalOp`]) run as ggml custom ops on the CPU; on CUDA the graph
    /// runs in segments around them (see `super::external`).
    pub fn compute(&mut self, graph: &Graph<'_>) -> Result<(), GgmlError> {
        self.kept_state.computed(None);
        // SAFETY: a live backend and an allocated graph, used from this thread only (&mut).
        unsafe { run_graph(self.raw.as_ptr(), graph.raw(), true, None, None) }
    }

    /// Starts `graph` and returns before it finishes (call [`Self::synchronize`] before
    /// reading results or touching its inputs). On CUDA, external ops are launched in order on
    /// the backend's stream like the graph's segments.
    pub fn compute_async(&mut self, graph: &Graph<'_>) -> Result<(), GgmlError> {
        self.kept_state.computed(None);
        // SAFETY: as in `compute`.
        unsafe { run_graph(self.raw.as_ptr(), graph.raw(), false, None, None) }
    }

    /// [`Self::compute_kept`] run by another thread: the bookkeeping now, the compute when the
    /// returned launcher runs. The launcher holds the backend's and the graph's raw handles
    /// only, so the thread that runs it borrows nothing of their owner, which may go on with
    /// its other stages meanwhile (see `Executor::step`).
    ///
    /// # Safety
    /// Until the launcher's [`Launcher::run`] has returned (or the launcher is dropped unrun),
    /// the caller keeps this backend and `kept` alive and uses neither: no compute or
    /// synchronize on the backend, no reading or writing `kept`'s tensors, no dropping or
    /// rebuilding `kept`.
    pub unsafe fn launcher(&mut self, kept: &super::KeptGraph) -> Result<Launcher, GgmlError> {
        self.prepare_kept(kept)?;
        // What `compute_kept` records, in its order.
        let plain = self.kept_state.plain;
        self.kept_state.computed(None);
        self.kept_state.plain = plain;
        self.kept_state.computed(Some(kept));
        Ok(Launcher {
            raw: self.raw,
            graph: NonNull::new(kept.graph().raw()).expect("a kept graph"),
            serial: kept.serial(),
            slot: NonNull::from(kept.kept_run().slot),
        })
    }

    pub fn synchronize(&mut self) {
        unsafe { ffi::ggml_backend_synchronize(self.raw.as_ptr()) }
    }

    /// The `cudaStream_t` this backend computes on, if it is a CUDA backend (not a meta one).
    #[cfg(feature = "cuda")]
    pub fn cuda_stream(&self) -> Option<*mut std::ffi::c_void> {
        // SAFETY: a live backend.
        unsafe { cuda_stream_of(self.raw.as_ptr()) }
    }

    /// [`Self::cuda_stream`], with the backend's device made current on this thread: for a
    /// kernel Legion launches there outside a graph (`pinned_io::pack`).
    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_launch_stream(&self) -> Option<*mut std::ffi::c_void> {
        let raw = self.raw.as_ptr();
        // SAFETY: a live backend; the shim calls read its CUDA context.
        unsafe {
            ffi::ggml_backend_is_cuda(raw).then(|| {
                super::external::cuda::legion_cuda_set_device(raw);
                super::external::cuda::legion_cuda_stream(raw)
            })
        }
    }

    /// A new event on this backend's device, `None` where the device has none (the CPU).
    pub fn new_event(&self) -> Option<Event> {
        // SAFETY: a live device; null if it has no events.
        NonNull::new(unsafe { ffi::ggml_backend_event_new(self.device.raw()) }).map(Event)
    }

    /// Records `event` after the work queued on this backend so far.
    pub fn record(&mut self, event: &Event) {
        // SAFETY: a live event of this backend's device and a live backend.
        unsafe { ffi::ggml_backend_event_record(event.0.as_ptr(), self.raw.as_ptr()) }
    }

    /// Queues a copy of `src` (in this backend's memory) into `dst` (in `to`'s) after the work
    /// queued on this backend; the work queued on `to` from now on waits for it, the host does
    /// not (CUDA: a peer copy on this backend's stream and an event `to`'s stream waits on).
    /// A backend that cannot queue it waits for both backends and copies.
    ///
    /// Until this backend is synchronized, the host must not write `src`'s memory.
    pub fn copy_to(&self, to: &Backend, src: Tensor<'_>, dst: Tensor<'_>) {
        assert_eq!(dst.nbytes(), src.nbytes(), "copy_to: sizes differ");
        unsafe {
            ffi::ggml_backend_tensor_copy_async(
                self.raw.as_ptr(),
                to.raw.as_ptr(),
                src.raw(),
                dst.raw(),
            )
        }
    }

    /// Whether [`Self::set_async`] / [`Self::get_async`] queue copies of `t` on this backend's
    /// stream: a GPU backend (not a meta one) and `t` allocated in its own memory (CUDA's
    /// queued copies take nothing else). Elsewhere copy synchronously.
    pub(crate) fn queues_copies_of(&self, t: Tensor<'_>) -> bool {
        let Some(buft) = self.queued_buft else {
            return false;
        };
        // SAFETY: a live tensor; its buffer (or its view source's) is a live buffer if set.
        let buf = unsafe {
            let raw = t.raw();
            let src = (*raw).view_src;
            if src.is_null() {
                (*raw).buffer
            } else {
                (*src).buffer
            }
        };
        // SAFETY: a live buffer; the call reads its type field.
        !buf.is_null() && unsafe { ffi::ggml_backend_buffer_get_type(buf) } == buft.as_ptr()
    }

    /// Queues a copy of `len` bytes from host memory at `src` into the start of `t` on this
    /// backend's stream: ordered with its computes, done by its next [`Self::synchronize`].
    ///
    /// # Safety
    /// [`Self::queues_copies_of`] holds for `t`; `src` points to `len` readable bytes (pinned
    /// host memory, for the copy to be asynchronous at all) that nothing writes until the
    /// backend is synchronized; `len` fits in `t`.
    pub(crate) unsafe fn set_async(&mut self, t: Tensor<'_>, src: *const u8, len: usize) {
        // SAFETY: the caller's contract; ggml asserts the range.
        unsafe {
            ffi::ggml_backend_tensor_set_async(self.raw.as_ptr(), t.raw(), src.cast(), 0, len)
        }
    }

    /// Queues a copy of the first `len` bytes of `t` into host memory at `dst` on this
    /// backend's stream (after the computes queued before it), done by its next
    /// [`Self::synchronize`].
    ///
    /// # Safety
    /// [`Self::queues_copies_of`] holds for `t`; `dst` points to `len` writable bytes that
    /// nothing reads or writes until the backend is synchronized; `len` fits in `t`.
    pub(crate) unsafe fn get_async(&mut self, t: Tensor<'_>, dst: *mut u8, len: usize) {
        // SAFETY: the caller's contract; ggml asserts the range.
        unsafe {
            ffi::ggml_backend_tensor_get_async(self.raw.as_ptr(), t.raw(), dst.cast(), 0, len)
        }
    }
}

/// A kept graph's compute for another thread ([`Backend::launcher`]).
pub struct Launcher {
    raw: NonNull<ffi::ggml_backend>,
    graph: NonNull<ffi::ggml_cgraph>,
    /// The kept graph's serial and stage-graph slot ([`super::KeptGraph::kept_run`]).
    serial: u64,
    slot: NonNull<super::stage_graph::kept_words::KeptSlot>,
}

// SAFETY: raw handles to a backend and a graph that nothing else uses until `run` returns
// (`Backend::launcher`'s contract).
unsafe impl Send for Launcher {}

impl Launcher {
    /// Runs the graph to completion.
    pub fn run(self) -> Result<(), GgmlError> {
        // SAFETY: a live backend and its allocated kept graph, used by this thread only until
        // this returns (`Backend::launcher`'s contract).
        unsafe {
            run_graph(
                self.raw.as_ptr(),
                self.graph.as_ptr(),
                true,
                None,
                self.kept(),
            )
        }
    }

    /// The kept graph, as its run sees it.
    fn kept(&self) -> Option<super::stage_graph::kept_words::KeptRun<'_>> {
        Some(super::stage_graph::kept_words::KeptRun {
            serial: self.serial,
            // SAFETY: the kept graph (and its slot) lives until the run returns
            // (`Backend::launcher`'s contract).
            slot: unsafe { self.slot.as_ref() },
        })
    }

    /// Starts the graph and returns once it is queued ([`Backend::compute_kept_async`]). The
    /// owner synchronizes the backend before it reads the graph's results or touches its
    /// inputs, and keeps the backend and the graph alive until then.
    pub fn start(self) -> Result<(), GgmlError> {
        // SAFETY: as in `run`; the owner waits for the queued work (above).
        unsafe {
            run_graph(
                self.raw.as_ptr(),
                self.graph.as_ptr(),
                false,
                None,
                self.kept(),
            )
        }
    }

    /// The stream [`Self::start`] queues the graph on ([`Backend::cuda_stream`]).
    #[cfg(feature = "cuda")]
    pub fn cuda_stream(&self) -> Option<*mut std::ffi::c_void> {
        // SAFETY: a live backend (`Backend::launcher`'s contract).
        unsafe { cuda_stream_of(self.raw.as_ptr()) }
    }
}

/// The compute stream of `raw`, if it is a CUDA backend.
///
/// # Safety
/// `raw` is a live backend.
#[cfg(feature = "cuda")]
unsafe fn cuda_stream_of(raw: *mut ffi::ggml_backend) -> Option<*mut std::ffi::c_void> {
    // SAFETY: the caller's.
    unsafe {
        ffi::ggml_backend_is_cuda(raw).then(|| super::external::cuda::legion_cuda_stream(raw))
    }
}

/// A graph input written while the graph already runs ([`Backend::compute_kept_with`]).
pub struct LateInput<'f> {
    /// The input tensor: none of the graph's nodes before its first reader may write it.
    pub tensor: Tensor<'f>,
    /// Writes the tensor's data (with [`Tensor::set_f32`] or the like, which returns once the
    /// bytes are on the device).
    pub fill: &'f mut dyn FnMut(),
}

/// Runs graph `g` on backend `raw`, to completion when `sync` (see [`Backend::compute`] and
/// [`Backend::compute_async`]), filling `late` before its first reader runs. `kept`: the kept
/// graph `g` is, if it is one (its serial and slot, for the stage graphs' words check and
/// cache entry).
///
/// # Safety
/// `raw` is a live backend, `g` a graph allocated in memory it computes in, and nothing else
/// uses either until this returns. With `kept`, `g` is that kept graph's.
unsafe fn run_graph(
    raw: *mut ffi::ggml_backend,
    g: *mut ffi::ggml_cgraph,
    sync: bool,
    late: Option<LateInput<'_>>,
    kept: Option<super::stage_graph::kept_words::KeptRun<'_>>,
) -> Result<(), GgmlError> {
    let mut late = late;
    if let Some(k) = &kept {
        super::input_audit::on_run(k.serial);
    }
    #[cfg(not(feature = "cuda"))]
    let _ = kept;
    #[cfg(feature = "cuda")]
    if super::op_profile::enabled() && unsafe { ffi::ggml_backend_is_cuda(raw) } {
        if let Some(l) = late.take() {
            (l.fill)();
        }
        // SAFETY: a CUDA backend and an allocated graph, as for any compute.
        return unsafe { super::op_profile::compute(raw, g) };
    }
    #[cfg(feature = "cuda")]
    if super::stage_graph::enabled() && unsafe { ffi::ggml_backend_is_cuda(raw) } {
        // SAFETY: the caller's.
        return unsafe { super::stage_graph::compute(raw, g, sync, &mut late, kept) };
    }
    #[cfg(feature = "cuda")]
    // SAFETY: the caller's.
    if let Some(done) = unsafe { compute_with_external(raw, g, sync, &mut late) } {
        return done;
    }
    // No segments to put it between: before the graph.
    if let Some(l) = late.take() {
        (l.fill)();
    }
    // SAFETY: the caller's.
    let status = unsafe {
        if sync {
            ffi::ggml_backend_graph_compute(raw, g)
        } else {
            ffi::ggml_backend_graph_compute_async(raw, g)
        }
    };
    if status == ffi::ggml_status_GGML_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(GgmlError::Compute(status))
    }
}

/// On CUDA, for a graph with external ops: its segments and the ops' kernels, in order on the
/// backend's stream, with `late` (taken) filled just before the segment of its first reader.
/// `None` when there is nothing to split (or not on CUDA); `late` is left then.
///
/// # Safety
/// As for [`run_graph`].
#[cfg(feature = "cuda")]
unsafe fn compute_with_external(
    raw: *mut ffi::ggml_backend,
    g: *mut ffi::ggml_cgraph,
    sync: bool,
    late: &mut Option<LateInput<'_>>,
) -> Option<Result<(), GgmlError>> {
    use super::external::{cuda, op_of};
    // SAFETY: valid graph; node indices in range.
    let n = unsafe { ffi::ggml_graph_n_nodes(g) };
    let ext: Vec<_> = (0..n)
        .filter_map(|i| unsafe {
            let node = ffi::ggml_graph_node(g, i);
            op_of(node).map(|op| (i, node, op))
        })
        .collect();
    if ext.is_empty() || !unsafe { ffi::ggml_backend_is_cuda(raw) } {
        return None;
    }
    // The late input's first reader (the graph's start if none reads it).
    let mut late = late.take().map(|l| {
        let t = l.tensor.raw();
        // SAFETY: valid graph; node indices in range; `src` entries are null or tensors.
        let at = (0..n)
            .find(|&i| unsafe {
                let node = ffi::ggml_graph_node(g, i);
                (*node).src.contains(&t)
            })
            .unwrap_or(0);
        (at, l.fill)
    });
    let mut run = || -> Result<(), GgmlError> {
        // SAFETY: the graph is allocated (as for any compute); segments are views of it, and
        // every kernel goes on the stream the segments run on, so all of it runs in order.
        unsafe {
            // The external ops launch on the current device, which a graph starting with
            // one has not set yet on this thread.
            cuda::legion_cuda_set_device(raw);
            let stream = cuda::legion_cuda_stream(raw);
            let _backend = cuda::BackendScope::enter(raw, g);
            let segment = |i0: i32, i1: i32| {
                if i0 >= i1 {
                    return Ok(());
                }
                let status = cuda::legion_cuda_compute_range(raw, g, i0, i1);
                if status == ffi::ggml_status_GGML_STATUS_SUCCESS {
                    Ok(())
                } else {
                    Err(GgmlError::Compute(status))
                }
            };
            let mut start = 0;
            // The nodes before the late input's reader are queued: fill it while they run. Its
            // write returns once the bytes are on the device, before the reader is queued.
            let mut fill_before = |i: i32, start: &mut i32| -> Result<(), GgmlError> {
                if let Some((at, fill)) = late.take_if(|(at, _)| *at <= i) {
                    segment(*start, at)?;
                    fill();
                    *start = at;
                }
                Ok(())
            };
            for &(i, node, op) in &ext {
                fill_before(i, &mut start)?;
                segment(start, i)?;
                let launch = op
                    .cuda
                    .ok_or_else(|| GgmlError::External(format!("{}: no CUDA kernel", op.name)))?;
                launch(&*node, stream).map_err(|e| {
                    let name = str_of((*node).name.as_ptr());
                    GgmlError::External(format!("{} (node {i} `{name}`): {e}", op.name))
                })?;
                start = i + 1;
            }
            fill_before(n, &mut start)?;
            segment(start, n)
        }
    };
    let done = run();
    if let Err(e) = &done {
        // The sync below aborts on a sticky CUDA error (an illegal address): name the op first.
        tracing::error!("graph compute failed: {e}");
    }
    if sync || done.is_err() {
        // SAFETY: the caller's.
        unsafe { ffi::ggml_backend_synchronize(raw) };
    }
    Some(done)
}

/// Initializes the backend `name` (e.g. `"CUDA0"`).
pub fn backend_by_name(name: &str) -> Result<Backend, GgmlError> {
    let c = CString::new(name).map_err(|_| GgmlError::Init(name.into()))?;
    let dev = unsafe { ffi::ggml_backend_dev_by_name(c.as_ptr()) };
    NonNull::new(dev)
        .map(Device)
        .ok_or_else(|| GgmlError::Init(name.into()))?
        .init()
}
