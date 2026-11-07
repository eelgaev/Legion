//! Contexts, tensors and graphs.
//!
//! Soundness rule: every byte of memory backing a tensor is owned by the [`Context`] the tensor
//! was created in, and tensors and graphs borrow their context. So no tensor can outlive its data.
//! - Weights: [`Context::alloc_tensors`] allocates all not-yet-allocated tensors in a backend buffer
//!   that the context keeps until it is dropped.
//! - Mapped weights: [`Context::map_tensors_from_file`] maps a file and points tensors into the
//!   mapping through a non-owning buffer. The context owns the mapping too, and frees the buffer
//!   before the mapping (`Drop` body first, then fields), so the data outlives every tensor.
//! - Intermediates: [`Context::alloc_graph`] moves a [`ComputeAllocator`] into the context, once.
//!   The allocator can grow its buffer, so a context never runs it twice; get it back for the next
//!   step with [`Context::into_allocator`], which consumes the context (and so ends all borrows).

use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::ptr::NonNull;

use super::backend::BufferType;
use super::{GgmlError, ffi};

/// A node of [`Context::cast_shared`]: its source, its type and the cast.
pub(super) type SharedCast = (
    NonNull<ffi::ggml_tensor>,
    ffi::ggml_type,
    NonNull<ffi::ggml_tensor>,
);

/// A node of [`Context::shared_node`]: its source, its key and the node.
pub(super) type SharedNode = (
    NonNull<ffi::ggml_tensor>,
    &'static str,
    NonNull<ffi::ggml_tensor>,
);

/// A ggml context holding tensor and graph metadata (never tensor data itself).
pub struct Context {
    raw: NonNull<ffi::ggml_context>,
    buffers: RefCell<Vec<NonNull<ffi::ggml_backend_buffer>>>,
    allocator: RefCell<Option<Alloc>>,
    /// File mappings that mapped-weight buffers point into. `Drop::drop` frees the buffers
    /// first, then this field unmaps.
    mappings: RefCell<Vec<memmap2::MmapMut>>,
    /// [`Context::cast_shared`]'s nodes: (source, type, cast).
    pub(super) shared_casts: RefCell<Vec<SharedCast>>,
    /// [`Context::shared_node`]'s nodes: (source, key, node).
    pub(super) shared_nodes: RefCell<Vec<SharedNode>>,
    /// The metadata memory, when it came from [`META_POOL`] (ggml's own otherwise); it goes back
    /// there after `ggml_free`.
    meta: Option<MetaBuffer>,
    /// [`Context::index_names`]' index.
    names: std::sync::OnceLock<NameIndex>,
    /// ggml contexts are not thread-safe: Send, but not Sync.
    _not_sync: PhantomData<Cell<()>>,
}

// SAFETY: the context and its buffers can move to another thread as a whole.
unsafe impl Send for Context {}

impl Drop for Context {
    fn drop(&mut self) {
        // Intermediates first, then weight buffers, then the metadata they describe.
        self.allocator.get_mut().take();
        unsafe {
            for b in self.buffers.get_mut().drain(..) {
                ffi::ggml_backend_buffer_free(b.as_ptr());
            }
            ffi::ggml_free(self.raw.as_ptr());
        }
        // ggml no longer points into it.
        if let Some(m) = self.meta.take() {
            m.give_back();
        }
    }
}

/// A context's tensors by name ([`Context::index_names`]), valid while ggml's used metadata
/// stays `used` bytes (no object made since).
struct NameIndex {
    used: usize,
    slots: std::collections::HashMap<String, TensorSlot>,
}

/// Contexts with at least this many bytes of metadata (graph contexts: a step graph's is about
/// 7 MB) take their metadata memory from [`META_POOL`].
pub(crate) const META_POOL_MIN: usize = 1 << 20;

/// Freed metadata buffers kept for the next context of the same size.
pub(crate) const META_POOL_KEEP: usize = 4;

/// Metadata memory of dropped big contexts ([`META_POOL_MIN`]), for the next ones: every step
/// graph build makes a context of [`crate::model`]'s 16384-node size (~7 MB), and a buffer
/// from here is already paged in, where a new allocation is page-faulted again as the graph
/// fills it and unmapped when it is freed. ggml initializes every object it places in the
/// buffer, so what a buffer held before never shows.
static META_POOL: std::sync::Mutex<Vec<MetaBuffer>> = std::sync::Mutex::new(Vec::new());

/// Buffers in [`META_POOL`] now (for tests).
#[cfg(test)]
pub(crate) fn meta_pool_len() -> usize {
    META_POOL.lock().unwrap_or_else(|e| e.into_inner()).len()
}

/// Memory for a ggml context's metadata (ggml's `mem_buffer`), aligned as ggml's own.
struct MetaBuffer {
    ptr: NonNull<u8>,
    layout: std::alloc::Layout,
}

// SAFETY: plain memory, owned by one context (or the pool) at a time.
unsafe impl Send for MetaBuffer {}

impl MetaBuffer {
    /// ggml_aligned_malloc's alignment.
    const ALIGN: usize = 64;

    /// A buffer of `size` bytes: one from the pool if it holds one of that size.
    fn take(size: usize) -> Self {
        let mut pool = META_POOL.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = pool.iter().rposition(|m| m.layout.size() == size) {
            return pool.remove(i);
        }
        drop(pool);
        let layout =
            std::alloc::Layout::from_size_align(size, Self::ALIGN).expect("a metadata size");
        // SAFETY: a non-zero size (callers ask for at least META_POOL_MIN).
        let ptr = unsafe { std::alloc::alloc(layout) };
        Self {
            ptr: NonNull::new(ptr).unwrap_or_else(|| std::alloc::handle_alloc_error(layout)),
            layout,
        }
    }

    /// Back to the pool; when it is full, the oldest buffer there is freed.
    fn give_back(self) {
        let mut pool = META_POOL.lock().unwrap_or_else(|e| e.into_inner());
        pool.push(self);
        let old = (pool.len() > META_POOL_KEEP).then(|| pool.remove(0));
        drop(pool);
        drop(old);
    }
}

impl Drop for MetaBuffer {
    fn drop(&mut self) {
        // SAFETY: allocated in `take` with this layout.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) }
    }
}

/// ggml-cuda's `MATRIX_ROW_PADDING` (ggml-cuda/common.cuh): its quantized matmuls (MMQ) read
/// whole K tiles, so the last row of a tensor is read up to the next multiple of this many
/// elements.
const CUDA_MATRIX_ROW_PADDING: i64 = 512;

/// Bytes past the end of a mapped tensor of type `ty` with rows of `ne0` elements that must be
/// zero for a GPU to read it in place (`--host-experts`): ggml's CUDA buffers allocate and zero
/// this padding (`ggml_backend_cuda_buffer_type_get_alloc_size`), while in a file the next
/// tensor's bytes are there, and read as quantization scales they can be Inf or NaN, which
/// times the zero-padded activations poisons the result. 0 for unquantized types.
pub fn mapped_tail_padding(ty: ffi::ggml_type, ne0: i64) -> usize {
    // SAFETY: plain type-traits lookup.
    if !unsafe { ffi::ggml_is_quantized(ty) } || ne0 % CUDA_MATRIX_ROW_PADDING == 0 {
        return 0;
    }
    super::row_size(
        ty,
        (CUDA_MATRIX_ROW_PADDING - ne0 % CUDA_MATRIX_ROW_PADDING) as usize,
    )
}

impl Context {
    /// Room for `max_tensors` tensors and, if `max_graph_nodes > 0`, one graph of that size.
    pub fn new(max_tensors: usize, max_graph_nodes: usize) -> Self {
        // SAFETY: size queries.
        let mem = unsafe {
            let mut mem = ffi::ggml_tensor_overhead() * max_tensors.max(1);
            if max_graph_nodes > 0 {
                mem += ffi::ggml_graph_overhead_custom(max_graph_nodes, false);
            }
            // As ggml pads the size of memory it allocates itself (GGML_MEM_ALIGN, at most 16).
            mem.next_multiple_of(16)
        };
        // Big ones in pooled memory (see META_POOL), ggml allocating the rest itself.
        let meta = (mem >= META_POOL_MIN).then(|| MetaBuffer::take(mem));
        // SAFETY: ggml_init with no_alloc only stores metadata in mem_size bytes, in `meta`'s
        // buffer (mem bytes, aligned) when given: the context holds it until after ggml_free.
        let raw = unsafe {
            ffi::ggml_init(ffi::ggml_init_params {
                mem_size: mem,
                mem_buffer: meta
                    .as_ref()
                    .map_or(std::ptr::null_mut(), |m| m.ptr.as_ptr().cast()),
                no_alloc: true,
            })
        };
        Self {
            meta,
            names: std::sync::OnceLock::new(),
            raw: NonNull::new(raw).expect("ggml_init failed"),
            buffers: RefCell::new(Vec::new()),
            allocator: RefCell::new(None),
            mappings: RefCell::new(Vec::new()),
            shared_casts: RefCell::new(Vec::new()),
            shared_nodes: RefCell::new(Vec::new()),
            _not_sync: PhantomData,
        }
    }

    pub(crate) fn raw(&self) -> *mut ffi::ggml_context {
        self.raw.as_ptr()
    }

    pub(crate) fn wrap(&self, t: *mut ffi::ggml_tensor) -> Tensor<'_> {
        Tensor {
            raw: NonNull::new(t).expect("ggml returned a null tensor (context full?)"),
            _ctx: PhantomData,
        }
    }

    /// A new tensor. `shape` is in ggml order: `shape[0]` is the contiguous dimension.
    pub fn new_tensor(&self, ty: ffi::ggml_type, shape: &[i64]) -> Tensor<'_> {
        assert!((1..=ffi::GGML_MAX_DIMS as usize).contains(&shape.len()));
        self.wrap(unsafe {
            ffi::ggml_new_tensor(self.raw(), ty, shape.len() as i32, shape.as_ptr())
        })
    }

    /// The tensor named `name` in this context. ggml compares the name with every tensor's in
    /// turn: for many lookups in a big context, keep [`Tensor::slot`]s and use
    /// [`Self::tensor_at`], or [`Self::index_names`] once its tensors are made.
    pub fn tensor(&self, name: &str) -> Option<Tensor<'_>> {
        if let Some(found) = self.indexed(name) {
            return found;
        }
        let c = std::ffi::CString::new(name).ok()?;
        let t = unsafe { ffi::ggml_get_tensor(self.raw(), c.as_ptr()) };
        NonNull::new(t).map(|raw| Tensor {
            raw,
            _ctx: PhantomData,
        })
    }

    /// Makes [`Self::tensor`] a hash lookup from now on, for a context whose tensors are all
    /// made and named (weight copies, cache and state pools, which every graph build looks up
    /// by name). It finds what ggml's search finds (the first tensor of a name); once a tensor
    /// is made later, ggml searches again, but a tensor renamed later is not seen: call it
    /// after the last [`Tensor::named`], once (later calls do nothing). Nothing is indexed when
    /// a name is not UTF-8.
    pub fn index_names(&self) {
        if self.names.get().is_some() {
            return;
        }
        let mut slots = std::collections::HashMap::new();
        // SAFETY: walks a live context's tensors; their names are NUL-terminated.
        unsafe {
            let mut t = ffi::ggml_get_first_tensor(self.raw());
            while let Some(raw) = NonNull::new(t) {
                let Ok(name) = std::ffi::CStr::from_ptr((*t).name.as_ptr()).to_str() else {
                    return;
                };
                slots.entry(name.to_owned()).or_insert(TensorSlot(raw));
                t = ffi::ggml_get_next_tensor(self.raw(), t);
            }
        }
        // SAFETY: a live context.
        let used = unsafe { ffi::ggml_used_mem(self.raw()) };
        let _ = self.names.set(NameIndex { used, slots });
    }

    /// [`Self::tensor`] through the name index, if there is one still valid.
    fn indexed(&self, name: &str) -> Option<Option<Tensor<'_>>> {
        let index = self.names.get()?;
        // SAFETY: a live context.
        if unsafe { ffi::ggml_used_mem(self.raw()) } != index.used {
            // Objects made since: ggml's search sees them.
            return None;
        }
        Some(index.slots.get(name).map(|s| Tensor {
            raw: s.0,
            _ctx: PhantomData,
        }))
    }

    /// The tensor of this context that `slot` was taken from ([`Tensor::slot`]), without a
    /// search.
    ///
    /// # Safety
    /// `slot` is a tensor of this context: ggml frees a tensor's metadata only with its
    /// context, so it stays valid for as long as the context lives.
    pub unsafe fn tensor_at(&self, slot: TensorSlot) -> Tensor<'_> {
        debug_assert!({
            // SAFETY: plain reads of a live context's metadata region.
            let (base, size) = unsafe {
                (
                    ffi::ggml_get_mem_buffer(self.raw()) as usize,
                    ffi::ggml_get_mem_size(self.raw()),
                )
            };
            (base..base + size).contains(&(slot.0.as_ptr() as usize))
        });
        Tensor {
            raw: slot.0,
            _ctx: PhantomData,
        }
    }

    /// Allocates every tensor of this context that has no memory yet in a new buffer of `buft`.
    /// Returns the buffer size in bytes (0 if there was nothing to allocate).
    pub fn alloc_tensors(&self, buft: BufferType) -> Result<usize, GgmlError> {
        // SAFETY: valid context and buffer type; the buffer is owned by self from here on.
        unsafe {
            let size = ffi::ggml_backend_alloc_ctx_tensors_from_buft_size(self.raw(), buft.raw());
            if size == 0 {
                return Ok(0);
            }
            let buf = ffi::ggml_backend_alloc_ctx_tensors_from_buft(self.raw(), buft.raw());
            let buf = NonNull::new(buf).ok_or(GgmlError::Alloc {
                bytes: size,
                buffer: buft.name(),
            })?;
            ffi::ggml_backend_buffer_set_usage(
                buf.as_ptr(),
                ffi::ggml_backend_buffer_usage_GGML_BACKEND_BUFFER_USAGE_WEIGHTS,
            );
            self.buffers.borrow_mut().push(buf);
            Ok(size)
        }
    }

    /// Places `t` (a tensor of this context with no memory yet) `offset` bytes into `buffer`,
    /// which this context owns from now on and frees when dropped, like the buffers of
    /// [`Self::alloc_tensors`]. For memory made outside ggml's allocators (see
    /// `crate::residency`).
    ///
    /// # Safety
    /// `buffer` must be a live buffer that nothing else frees, and `t` must fit in it from
    /// `offset`.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) unsafe fn adopt_buffer(
        &self,
        t: Tensor<'_>,
        buffer: NonNull<ffi::ggml_backend_buffer>,
        offset: usize,
    ) -> Result<(), GgmlError> {
        self.buffers.borrow_mut().push(buffer);
        // SAFETY: per the contract; ggml checks that the tensor fits the buffer.
        let status = unsafe {
            let base = ffi::ggml_backend_buffer_get_base(buffer.as_ptr()).cast::<u8>();
            ffi::ggml_backend_tensor_alloc(buffer.as_ptr(), t.raw(), base.add(offset).cast())
        };
        if status == ffi::ggml_status_GGML_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(GgmlError::Alloc {
                bytes: t.nbytes(),
                buffer: "adopted buffer".into(),
            })
        }
    }

    /// Maps `file` copy-on-write (private; writes never reach the file) and places the named
    /// tensors of this context at the given byte offsets inside it, with no read: the data is
    /// whatever the page cache holds. `placements` is `(tensor name, file offset)`.
    ///
    /// Only for tensors that have no memory yet and are meant for the plain CPU backend. A
    /// tensor is mapped only if its offset is a multiple of ggml's tensor alignment (any offset
    /// for an `I8` tensor, whose elements are bytes: a safetensors checkpoint's FP8 n-gram table,
    /// which only the host reads) and its bytes lie inside the file; the others are left untouched (and unallocated) for the caller to
    /// load another way. Returns the indices into `placements` of the tensors that were mapped.
    ///
    /// A quantized tensor also gets its [`mapped_tail_padding`] zeroed in the (private) mapping,
    /// as a GPU reading it in place needs; one whose padding would run past the mapping's last
    /// page is not mapped. The padding is left alone where it overlaps another tensor mapped
    /// here.
    ///
    /// Soundness: the context keeps the mapping and the buffer; the buffer (which does not own
    /// the memory) is freed first on drop, then the mapping is unmapped, and tensors borrow the
    /// context, so no tensor outlives its bytes. The mapping is `PROT_READ|PROT_WRITE` private,
    /// so a stray write through a tensor faults nowhere and never alters the file. The usual
    /// mmap caveat remains: if another process truncates the file while it is mapped, touching
    /// the vanished pages raises SIGBUS.
    pub fn map_tensors_from_file(
        &self,
        file: &std::fs::File,
        placements: &[(&str, u64)],
    ) -> Result<Vec<usize>, GgmlError> {
        // SAFETY: see above; the file is only read through the mapping.
        let map = unsafe { memmap2::MmapOptions::new().map_copy(file) }
            .map_err(|e| GgmlError::Map(e.to_string()))?;
        let mut map = map;
        // Ask for readahead of the whole mapping; best effort.
        let _ = map.advise(memmap2::Advice::WillNeed);
        let len = map.len();
        let base = map.as_mut_ptr();
        // SAFETY: the pointer is page aligned (so aligned for ggml) and valid for `len` bytes
        // until `map` is dropped, which this context delays until after the buffer is freed.
        let buf = unsafe { ffi::ggml_backend_cpu_buffer_from_ptr(base.cast(), len) };
        let buf = NonNull::new(buf).ok_or(GgmlError::Alloc {
            bytes: len,
            buffer: "CPU_Mapped".into(),
        })?;
        // SAFETY: valid buffer.
        unsafe {
            ffi::ggml_backend_buffer_set_usage(
                buf.as_ptr(),
                ffi::ggml_backend_buffer_usage_GGML_BACKEND_BUFFER_USAGE_WEIGHTS,
            );
        }
        self.buffers.borrow_mut().push(buf);
        self.mappings.borrow_mut().push(map);
        // The mapping's last page reads as zeros past the end of the file (4 KiB, the smallest
        // page size, is a lower bound for where that page ends).
        let mapped_len = (len as u64).next_multiple_of(4096);
        let mut mapped = Vec::new();
        // Byte ranges of the mapped tensors, and the tail paddings to zero once all are known.
        let mut ranges: Vec<(u64, u64)> = Vec::new();
        let mut tails: Vec<(u64, u64)> = Vec::new();
        // SAFETY: static buffer type.
        let align =
            unsafe { ffi::ggml_backend_buft_get_alignment(ffi::ggml_backend_cpu_buffer_type()) }
                .max(1) as u64;
        for (i, &(name, offset)) in placements.iter().enumerate() {
            let Some(t) = self.tensor(name) else { continue };
            let nbytes = t.nbytes() as u64;
            let pad = mapped_tail_padding(t.ty(), t.shape()[0]) as u64;
            let aligned = offset.is_multiple_of(align) || t.ty() == super::types::I8;
            let end = offset.saturating_add(nbytes);
            if !aligned
                || t.is_allocated()
                || end > len as u64
                || end.saturating_add(pad) > mapped_len
            {
                continue;
            }
            // SAFETY: the tensor is in this context, unallocated, and addr..addr+size is inside
            // the buffer (checked above).
            let status = unsafe {
                ffi::ggml_backend_tensor_alloc(
                    buf.as_ptr(),
                    t.raw(),
                    base.add(offset as usize).cast(),
                )
            };
            if status == ffi::ggml_status_GGML_STATUS_SUCCESS {
                mapped.push(i);
                ranges.push((offset, end));
                if pad > 0 {
                    tails.push((end, end + pad));
                }
            }
        }
        for (lo, hi) in tails {
            if ranges.iter().any(|&(a, b)| a < hi && lo < b) {
                continue;
            }
            // Past the end of the file the page is zero already.
            let hi = hi.min(len as u64);
            if lo < hi {
                // SAFETY: lo..hi is inside the mapping and no tensor's bytes; the mapping is
                // private, so the zeros never reach the file.
                unsafe { std::ptr::write_bytes(base.add(lo as usize), 0, (hi - lo) as usize) };
            }
        }
        Ok(mapped)
    }

    /// Sets every byte of the buffers made by [`Self::alloc_tensors`] to `value`.
    pub fn clear_buffers(&self, value: u8) {
        for b in self.buffers.borrow().iter() {
            unsafe { ffi::ggml_backend_buffer_clear(b.as_ptr(), value) }
        }
    }

    /// A graph with room for `max_nodes` nodes, in this context's metadata memory.
    pub fn new_graph(&self, max_nodes: usize) -> Graph<'_> {
        let raw = unsafe { ffi::ggml_new_graph_custom(self.raw(), max_nodes, false) };
        Graph {
            raw: NonNull::new(raw).expect("no room for the graph in this context"),
            ctx: self.raw,
            _ctx: PhantomData,
        }
    }

    /// Gives memory to every intermediate tensor of `graph`, using `allocator`. Tensors that
    /// already have memory (weights, inputs allocated with `alloc_tensors`) are left alone.
    /// A context holds one allocator: calling this twice is an error.
    pub fn alloc_graph(
        &self,
        graph: &Graph<'_>,
        allocator: ComputeAllocator,
    ) -> Result<(), GgmlError> {
        // The intermediates' memory must belong to the context that owns them.
        if graph.ctx != self.raw {
            return Err(GgmlError::ForeignGraph);
        }
        let mut slot = self.allocator.borrow_mut();
        if slot.is_some() {
            return Err(GgmlError::GraphAlreadyAllocated);
        }
        let ok = allocator.alloc(graph);
        let buffer = allocator.buft.name();
        let buft = allocator.buft;
        *slot = Some(Alloc::Own(allocator));
        if ok {
            Ok(())
        } else {
            Err(GgmlError::Alloc {
                bytes: graph_buffer_size(buft, graph),
                buffer,
            })
        }
    }

    /// [`Self::alloc_graph`] from an allocator other graphs share: their intermediates
    /// overlap in its one buffer, so only one of them may run at a time, each with its inputs
    /// set again first. Returns whether the buffer grew: it then moved, and the graphs
    /// allocated from it before must not run again.
    pub fn alloc_graph_shared(
        &self,
        graph: &Graph<'_>,
        allocator: &SharedAllocator,
    ) -> Result<bool, GgmlError> {
        if graph.ctx != self.raw {
            return Err(GgmlError::ForeignGraph);
        }
        let mut slot = self.allocator.borrow_mut();
        if slot.is_some() {
            return Err(GgmlError::GraphAlreadyAllocated);
        }
        let a = allocator.lock().unwrap_or_else(|e| e.into_inner());
        let before = a.buffer_size();
        let ok = a.alloc(graph);
        // The allocator plans one buffer (id 0) and only reallocates it to grow: a buffer
        // the same size did not move. (Were ggml to grow it by adding a chunk instead, the
        // size would grow too, and the earlier graphs would be dropped needlessly, not kept
        // pointing at freed memory.)
        let grew = a.buffer_size() > before;
        let buffer = a.buft.name();
        drop(a);
        *slot = Some(Alloc::Shared(allocator.clone()));
        if ok {
            Ok(grew)
        } else {
            Err(GgmlError::Alloc { bytes: 0, buffer })
        }
    }

    /// Bytes the compute allocator of an allocated graph reserves (0 before one).
    pub fn allocator_size(&self) -> usize {
        match &*self.allocator.borrow() {
            None => 0,
            Some(Alloc::Own(a)) => a.buffer_size(),
            Some(Alloc::Shared(a)) => a.lock().unwrap_or_else(|e| e.into_inner()).buffer_size(),
        }
    }

    /// Ends this context and returns its compute allocator (with its grown buffer) for reuse;
    /// `None` for a shared one ([`Self::alloc_graph_shared`]).
    pub fn into_allocator(self) -> Option<ComputeAllocator> {
        match self.allocator.borrow_mut().take() {
            Some(Alloc::Own(a)) => Some(a),
            _ => None,
        }
    }

    // ---- operations (add more as model code needs them) ----

    /// `a @ b^T` in ggml terms: `a` is `[k, m]` (weights), `b` is `[k, n]`; result `[m, n]`.
    pub fn mul_mat<'a>(&'a self, a: Tensor<'a>, b: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_mul_mat(self.raw(), a.raw(), b.raw()) })
    }

    pub fn add<'a>(&'a self, a: Tensor<'a>, b: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_add(self.raw(), a.raw(), b.raw()) })
    }

    /// Element-wise product (with broadcasting of `b`).
    pub fn mul<'a>(&'a self, a: Tensor<'a>, b: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_mul(self.raw(), a.raw(), b.raw()) })
    }

    pub fn scale<'a>(&'a self, a: Tensor<'a>, s: f32) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_scale(self.raw(), a.raw(), s) })
    }

    pub fn rms_norm<'a>(&'a self, a: Tensor<'a>, eps: f32) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_rms_norm(self.raw(), a.raw(), eps) })
    }

    pub fn silu<'a>(&'a self, a: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_silu(self.raw(), a.raw()) })
    }

    /// Rows of `a` selected by the I32 tensor `ids` (embedding lookup).
    pub fn get_rows<'a>(&'a self, a: Tensor<'a>, ids: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_get_rows(self.raw(), a.raw(), ids.raw()) })
    }

    /// [`Self::get_rows`] into F16 rows: a float or quantized `a` converted (dequantized) to
    /// F16 by the kernel itself, without the F32 rows `get_rows` makes and a cast after. Only
    /// for backends whose get_rows writes F16 (CUDA's does; the CPU's writes F32 only).
    pub fn get_rows_f16<'a>(&'a self, a: Tensor<'a>, ids: Tensor<'a>) -> Tensor<'a> {
        let t = self.get_rows(a, ids);
        // SAFETY: a fresh, unallocated get_rows node of this context that nothing reads yet:
        // its type and contiguous strides change before any allocation or use.
        unsafe {
            let r = t.raw();
            (*r).type_ = ffi::ggml_type_GGML_TYPE_F16;
            (*r).nb[0] = ffi::ggml_type_size((*r).type_);
            for i in 1..4 {
                (*r).nb[i] = (*r).nb[i - 1] * (*r).ne[i - 1] as usize;
            }
        }
        t
    }
}

/// A tensor (metadata in a [`Context`]; data in a buffer that context owns).
#[derive(Clone, Copy)]
pub struct Tensor<'a> {
    raw: NonNull<ffi::ggml_tensor>,
    _ctx: PhantomData<&'a Context>,
}

impl std::fmt::Debug for Tensor<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("name", &self.name())
            .field("type", &super::type_name(self.ty()))
            .field("shape", &self.shape())
            .finish()
    }
}

/// Where a tensor's metadata sits in its context: [`Context::tensor_at`] gets it back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorSlot(NonNull<ffi::ggml_tensor>);

// SAFETY: an address only; reading through it takes `Context::tensor_at` on the owning context.
unsafe impl Send for TensorSlot {}
unsafe impl Sync for TensorSlot {}

impl<'a> Tensor<'a> {
    pub(crate) fn raw(self) -> *mut ffi::ggml_tensor {
        self.raw.as_ptr()
    }

    /// This tensor's place in its context ([`Context::tensor_at`]).
    pub fn slot(self) -> TensorSlot {
        TensorSlot(self.raw)
    }

    /// Another tensor of the same context (a source of this one, say).
    pub(super) fn with_raw(self, raw: NonNull<ffi::ggml_tensor>) -> Tensor<'a> {
        Tensor {
            raw,
            _ctx: PhantomData,
        }
    }

    pub fn ty(self) -> ffi::ggml_type {
        unsafe { (*self.raw()).type_ }
    }

    /// ggml order, trailing 1s trimmed.
    pub fn shape(self) -> Vec<i64> {
        let ne = unsafe { (*self.raw()).ne };
        let n = ne.iter().rposition(|&d| d != 1).map_or(1, |i| i + 1);
        ne[..n].to_vec()
    }

    pub fn elements(self) -> usize {
        unsafe { ffi::ggml_nelements(self.raw()) as usize }
    }

    pub fn nbytes(self) -> usize {
        unsafe { ffi::ggml_nbytes(self.raw()) }
    }

    pub fn name(self) -> String {
        unsafe { std::ffi::CStr::from_ptr((*self.raw()).name.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    /// Sets the name (truncated by ggml to 63 bytes). Returns the tensor for chaining.
    pub fn named(self, name: &str) -> Self {
        let c = std::ffi::CString::new(name).expect("tensor name with NUL");
        unsafe { ffi::ggml_set_name(self.raw(), c.as_ptr()) };
        self
    }

    fn is_allocated(self) -> bool {
        unsafe { !(*self.raw()).buffer.is_null() }
    }

    /// Copies `bytes` into the tensor's memory (host or device).
    ///
    /// Panics if the tensor has no memory yet or the length isn't exactly `nbytes()`.
    pub fn set_bytes(self, bytes: &[u8]) {
        assert!(
            self.is_allocated(),
            "tensor `{}` has no memory yet",
            self.name()
        );
        assert_eq!(bytes.len(), self.nbytes(), "tensor `{}` size", self.name());
        unsafe { ffi::ggml_backend_tensor_set(self.raw(), bytes.as_ptr().cast(), 0, bytes.len()) }
    }

    /// Copies `bytes` into the tensor's memory starting `offset` bytes in (host or device).
    ///
    /// Panics if the tensor has no memory yet or the range is not inside `nbytes()`.
    pub fn set_bytes_at(self, offset: usize, bytes: &[u8]) {
        assert!(
            self.is_allocated(),
            "tensor `{}` has no memory yet",
            self.name()
        );
        assert!(
            offset
                .checked_add(bytes.len())
                .is_some_and(|end| end <= self.nbytes()),
            "tensor `{}`: range {offset}+{} out of bounds",
            self.name(),
            bytes.len()
        );
        unsafe {
            ffi::ggml_backend_tensor_set(self.raw(), bytes.as_ptr().cast(), offset, bytes.len())
        }
    }

    /// Copies `len` bytes of the tensor starting `offset` bytes in to `dst` starting
    /// `dst_offset` bytes in, where `dst` or the tensor lives in host memory (CPU or pinned): the
    /// other side is read or written through its buffer (a DMA for pinned memory), synchronously.
    /// Copies between two devices' memory go through a graph instead.
    ///
    /// Panics if either tensor has no memory, neither is host memory, or a range is out of
    /// bounds.
    pub fn copy_bytes_to(self, offset: usize, dst: Tensor<'_>, dst_offset: usize, len: usize) {
        assert!(
            self.is_allocated() && dst.is_allocated(),
            "tensors `{}` / `{}` have no memory yet",
            self.name(),
            dst.name()
        );
        let fits = |t: Tensor<'_>, at: usize| at.checked_add(len).is_some_and(|e| e <= t.nbytes());
        assert!(
            fits(self, offset) && fits(dst, dst_offset),
            "copy of {len} bytes from `{}`+{offset} to `{}`+{dst_offset} out of bounds",
            self.name(),
            dst.name()
        );
        // SAFETY: both tensors are allocated and the ranges are inside them (asserted); a host
        // tensor's bytes are addressable at `data`, and the caller orders this copy against any
        // graph using either tensor (as for `set_bytes` / `bytes`).
        unsafe {
            if dst.is_host() {
                let to = (*dst.raw()).data.cast::<u8>().add(dst_offset);
                ffi::ggml_backend_tensor_get(self.raw(), to.cast(), offset, len);
            } else {
                assert!(
                    self.is_host(),
                    "copy_bytes_to: neither tensor is in host memory"
                );
                let from = (*self.raw()).data.cast::<u8>().add(offset);
                ffi::ggml_backend_tensor_set(dst.raw(), from.cast(), dst_offset, len);
            }
        }
    }

    /// True if the tensor lives in a meta buffer (tensor parallelism, `super::meta`): split or
    /// mirrored over several devices, written through ggml only, in whole slices.
    pub fn is_meta(self) -> bool {
        // SAFETY: an allocated tensor's `buffer` is a live buffer; the call reads its type.
        self.is_allocated() && unsafe { ffi::ggml_backend_buffer_is_meta((*self.raw()).buffer) }
    }

    /// True if the tensor has memory in a host-visible buffer (CPU or pinned host memory).
    pub fn is_host(self) -> bool {
        // A meta buffer (tensor parallelism, `super::meta`) over host devices says it is host
        // memory, but its tensors' data are the devices' slices, not at `data`.
        // SAFETY: an allocated tensor's `buffer` is a live buffer; the calls read its type.
        self.is_allocated()
            && unsafe {
                let buf = (*self.raw()).buffer;
                ffi::ggml_backend_buffer_is_host(buf) && !ffi::ggml_backend_buffer_is_meta(buf)
            }
    }

    /// The CUDA device ordinal of the device whose buffer holds the tensor (ggml's `CUDA<n>`;
    /// spill buffers in host memory name the GPU they serve); `None` when it has no memory yet
    /// or its buffer's device is not a CUDA device (CPU, meta). No driver call: three ggml
    /// queries and the name.
    pub fn cuda_device(self) -> Option<usize> {
        if !self.is_allocated() {
            return None;
        }
        // SAFETY: an allocated tensor's `buffer` is a live buffer; its type and device are
        // static descriptors and the device name a NUL-terminated string they own.
        let name = unsafe {
            let buft = ffi::ggml_backend_buffer_get_type((*self.raw()).buffer);
            if buft.is_null() {
                return None;
            }
            let dev = ffi::ggml_backend_buft_get_device(buft);
            if dev.is_null() {
                return None;
            }
            let name = ffi::ggml_backend_dev_name(dev);
            if name.is_null() {
                return None;
            }
            std::ffi::CStr::from_ptr(name)
        };
        let digits = name.to_bytes().strip_prefix(b"CUDA")?;
        if digits.is_empty() || !digits.iter().all(u8::is_ascii_digit) {
            return None;
        }
        std::str::from_utf8(digits).ok()?.parse().ok()
    }

    /// The tensor's memory as a mutable byte slice, or `None` if it has no memory yet or lives
    /// in a buffer the host cannot address (device VRAM).
    ///
    /// Why this is sound: the memory belongs to the buffer the owning [`Context`] keeps until it
    /// is dropped, and the returned slice carries the context lifetime `'a` of the tensor, so
    /// the slice cannot outlive the memory. Host buffers are plain addressable bytes of exactly
    /// `nbytes()` length, starting at `tensor->data`.
    ///
    /// It is `unsafe` because `Tensor` is `Copy`, so Rust cannot see other handles to the same
    /// bytes. The caller must guarantee that, while the slice lives, nothing else reads or
    /// writes this tensor's data: no other slice from this method, no `bytes()`/`set_bytes()`,
    /// and no graph computing with it. The weight loader satisfies this because it fills
    /// tensors before handing them out, each tensor having exactly one writer.
    ///
    /// # Safety
    /// The caller upholds the exclusivity rule above.
    pub unsafe fn host_bytes_mut(&mut self) -> Option<&'a mut [u8]> {
        if !self.is_host() {
            return None;
        }
        let data = unsafe { (*self.raw()).data };
        if data.is_null() {
            return None;
        }
        // SAFETY: see above; the buffer is host memory of at least nbytes() at `data`.
        Some(unsafe { std::slice::from_raw_parts_mut(data.cast::<u8>(), self.nbytes()) })
    }

    /// Copies the tensor's contents out (host or device).
    pub fn bytes(self) -> Vec<u8> {
        assert!(
            self.is_allocated(),
            "tensor `{}` has no memory yet",
            self.name()
        );
        let mut out = vec![0u8; self.nbytes()];
        unsafe { ffi::ggml_backend_tensor_get(self.raw(), out.as_mut_ptr().cast(), 0, out.len()) };
        out
    }

    /// Copies the tensor's first `n` bytes out (host or device): the first rows of a matrix.
    pub fn bytes_prefix(self, n: usize) -> Vec<u8> {
        assert!(
            self.is_allocated(),
            "tensor `{}` has no memory yet",
            self.name()
        );
        assert!(n <= self.nbytes(), "{n} bytes of `{}`", self.name());
        let mut out = vec![0u8; n];
        unsafe { ffi::ggml_backend_tensor_get(self.raw(), out.as_mut_ptr().cast(), 0, n) };
        out
    }

    /// Copies `len` bytes of the tensor starting `offset` bytes in out (host or device).
    pub fn bytes_at(self, offset: usize, len: usize) -> Vec<u8> {
        assert!(
            self.is_allocated(),
            "tensor `{}` has no memory yet",
            self.name()
        );
        assert!(
            offset
                .checked_add(len)
                .is_some_and(|end| end <= self.nbytes()),
            "tensor `{}`: range {offset}+{len} out of bounds",
            self.name()
        );
        let mut out = vec![0u8; len];
        unsafe { ffi::ggml_backend_tensor_get(self.raw(), out.as_mut_ptr().cast(), offset, len) };
        out
    }

    /// For F32 tensors.
    pub fn set_f32(self, values: &[f32]) {
        assert_eq!(
            self.ty(),
            ffi::ggml_type_GGML_TYPE_F32,
            "set_f32 on a non-F32 tensor"
        );
        self.set_bytes(as_bytes(values));
    }

    /// For F32 tensors.
    pub fn to_f32(self) -> Vec<f32> {
        assert_eq!(
            self.ty(),
            ffi::ggml_type_GGML_TYPE_F32,
            "to_f32 on a non-F32 tensor"
        );
        assert!(
            self.is_allocated(),
            "tensor `{}` has no memory yet",
            self.name()
        );
        // Straight into the result: a step's captured rows run to hundreds of MB.
        let mut out = vec![0f32; self.nbytes() / 4];
        unsafe {
            ffi::ggml_backend_tensor_get(self.raw(), out.as_mut_ptr().cast(), 0, out.len() * 4)
        };
        out
    }

    /// For F32 tensors: like [`Self::to_f32`], but the bytes land in host tensor `via` first
    /// (pinned memory: one direct DMA, where reading into pageable memory crawls through the
    /// driver's bounce buffers).
    pub fn to_f32_via(self, via: Tensor<'_>) -> Vec<f32> {
        assert_eq!(
            self.ty(),
            ffi::ggml_type_GGML_TYPE_F32,
            "to_f32_via on a non-F32 tensor"
        );
        assert!(
            self.is_allocated() && via.is_allocated() && via.is_host(),
            "to_f32_via needs allocated tensors and a host one to read through"
        );
        let n = self.nbytes();
        assert!(
            via.nbytes() >= n,
            "to_f32_via: `{}` is too small",
            via.name()
        );
        // SAFETY: `via` is allocated host memory of at least `n` bytes, which the read fills
        // before the slice is taken; any bytes are valid f32s.
        unsafe {
            let data = (*via.raw()).data;
            ffi::ggml_backend_tensor_get(self.raw(), data, 0, n);
            std::slice::from_raw_parts(data.cast::<f32>(), n / 4).to_vec()
        }
    }

    /// For F32 tensors: the DMA half of [`Self::to_f32_via`] only: the bytes land in host tensor
    /// `via` (pinned memory), which the caller then copies out (returns `via`'s address and the
    /// byte count).
    pub fn read_via(self, via: Tensor<'_>) -> (usize, usize) {
        assert_eq!(
            self.ty(),
            ffi::ggml_type_GGML_TYPE_F32,
            "read_via on a non-F32 tensor"
        );
        assert!(
            self.is_allocated() && via.is_allocated() && via.is_host(),
            "read_via needs allocated tensors and a host one to read through"
        );
        let n = self.nbytes();
        assert!(via.nbytes() >= n, "read_via: `{}` is too small", via.name());
        // SAFETY: `via` is allocated host memory of at least `n` bytes.
        unsafe {
            let data = (*via.raw()).data;
            ffi::ggml_backend_tensor_get(self.raw(), data, 0, n);
            (data as usize, n)
        }
    }

    /// [`Self::read_via`] of the tensor's bytes from `offset` on only: they land at the start
    /// of `via` (returns `via`'s address and the byte count read).
    pub fn read_tail_via(self, via: Tensor<'_>, offset: usize) -> (usize, usize) {
        assert_eq!(
            self.ty(),
            ffi::ggml_type_GGML_TYPE_F32,
            "read_tail_via on a non-F32 tensor"
        );
        assert!(
            self.is_allocated() && via.is_allocated() && via.is_host(),
            "read_tail_via needs allocated tensors and a host one to read through"
        );
        let n = self.nbytes();
        assert!(offset <= n, "read_tail_via: offset past the tensor");
        let n = n - offset;
        assert!(
            via.nbytes() >= n,
            "read_tail_via: `{}` is too small",
            via.name()
        );
        // SAFETY: `via` is allocated host memory of at least `n` bytes; the read stays within
        // the tensor (`offset + n` is its size).
        unsafe {
            let data = (*via.raw()).data;
            if n > 0 {
                ffi::ggml_backend_tensor_get(self.raw(), data, offset, n);
            }
            (data as usize, n)
        }
    }

    /// For F32 tensors: like [`Self::set_f32`], through host tensor `via` (see
    /// [`Self::to_f32_via`]).
    pub fn set_f32_via(self, values: &[f32], via: Tensor<'_>) {
        assert_eq!(
            std::mem::size_of_val(values),
            self.nbytes(),
            "tensor `{}` size",
            self.name()
        );
        self.fill_f32_via(via, |dst| dst.copy_from_slice(values));
    }

    /// For F32 tensors: sets all of the tensor through host tensor `via` (see
    /// [`Self::to_f32_via`]), whose first floats (as many as the tensor's) `fill` writes first.
    pub fn fill_f32_via(self, via: Tensor<'_>, fill: impl FnOnce(&mut [f32])) {
        assert_eq!(
            self.ty(),
            ffi::ggml_type_GGML_TYPE_F32,
            "set_f32_via on a non-F32 tensor"
        );
        assert!(
            self.is_allocated() && via.is_allocated() && via.is_host(),
            "set_f32_via needs allocated tensors and a host one to write through"
        );
        let n = self.nbytes();
        assert!(
            via.nbytes() >= n,
            "set_f32_via: `{}` is too small",
            via.name()
        );
        // SAFETY: `via` is allocated host memory of at least `n` bytes, which nothing else
        // references while the slice lives; any bytes are valid f32s.
        unsafe {
            let data = (*via.raw()).data;
            fill(std::slice::from_raw_parts_mut(data.cast::<f32>(), n / 4));
            ffi::ggml_backend_tensor_set(self.raw(), data, 0, n);
        }
    }

    /// For I32 tensors.
    pub fn set_i32(self, values: &[i32]) {
        assert_eq!(
            self.ty(),
            ffi::ggml_type_GGML_TYPE_I32,
            "set_i32 on a non-I32 tensor"
        );
        self.set_bytes(as_bytes(values));
    }

    /// For I32 tensors: writes `values` from element `start` on, leaving the rest alone.
    pub fn set_i32_at(self, start: usize, values: &[i32]) {
        assert_eq!(
            self.ty(),
            ffi::ggml_type_GGML_TYPE_I32,
            "set_i32_at on a non-I32 tensor"
        );
        self.set_bytes_at(start * 4, as_bytes(values));
    }
}

/// The bytes of 4-byte plain values (`f32`, `i32`) in memory order.
fn as_bytes<T: Copy>(values: &[T]) -> &[u8] {
    const { assert!(std::mem::size_of::<T>() == 4) };
    // SAFETY: the slice's memory, read as bytes; any byte is a valid u8.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

/// A computation graph.
pub struct Graph<'a> {
    raw: NonNull<ffi::ggml_cgraph>,
    /// The context the graph (and its intermediates) were created in.
    ctx: NonNull<ffi::ggml_context>,
    _ctx: PhantomData<&'a Context>,
}

impl<'a> Graph<'a> {
    pub(crate) fn raw(&self) -> *mut ffi::ggml_cgraph {
        self.raw.as_ptr()
    }

    /// The context the graph's nodes are made in.
    pub(super) fn ctx_raw(&self) -> *mut ffi::ggml_context {
        self.ctx.as_ptr()
    }

    /// Adds `t` and everything it depends on.
    pub fn expand(&mut self, t: Tensor<'a>) {
        unsafe { ffi::ggml_build_forward_expand(self.raw(), t.raw()) }
    }

    pub fn nodes(&self) -> usize {
        unsafe { ffi::ggml_graph_n_nodes(self.raw()) as usize }
    }

    /// Moves the nodes from index `from` on (the last ones added) to just after `anchor`, a
    /// node ahead of them, the others keeping their order. Every source of a moved node that is
    /// a node must be `anchor` or ahead of it (checked).
    pub fn move_after(&mut self, anchor: Tensor<'a>, from: usize) {
        let n = self.nodes();
        assert!(from <= n, "move_after: from {from} past {n} nodes");
        // SAFETY: the graph's node array holds `n` valid tensor pointers.
        let nodes = unsafe { std::slice::from_raw_parts_mut(ffi::ggml_graph_nodes(self.raw()), n) };
        let at = nodes[..from]
            .iter()
            .position(|&t| t == anchor.raw())
            .expect("move_after: the anchor is not a node ahead of the moved ones");
        let between: std::collections::HashSet<_> = nodes[at + 1..from].iter().copied().collect();
        for &t in &nodes[from..] {
            // SAFETY: a node of the graph; `src` entries are null or tensors.
            for s in unsafe { (*t).src } {
                assert!(
                    !between.contains(&s),
                    "move_after: a moved node reads a node between the anchor and it"
                );
            }
        }
        nodes[at + 1..].rotate_right(n - from);
    }
}

/// A graph kept together with the context it was built in (and that context's compute
/// allocator), to run it again with new input values instead of rebuilding and re-planning it.
/// Every run then sees the same tensors at the same addresses, which is what lets the CUDA
/// backend replay the graph as a captured CUDA graph.
pub struct KeptGraph {
    graph: NonNull<ffi::ggml_cgraph>,
    tensors: Vec<NonNull<ffi::ggml_tensor>>,
    /// Unique among kept graphs (a dropped graph's address may be reused, its serial not).
    serial: u64,
    /// The buffers its nodes and their sources live in (what a compute of it rebuilds over),
    /// and those holding its own per-device tensors: intermediates, inputs and views.
    used: Vec<usize>,
    resident: Vec<usize>,
    /// Owns the graph and tensor metadata above and the intermediates' memory.
    ctx: Context,
    /// What a run of the graph must read of the weights ([`Self::set_model_bytes`]).
    model_bytes: super::model_bytes::GraphBytes,
    /// Its stage-graph cache entry between unchanged runs ([`Self::kept_run`]).
    stage: super::stage_graph::kept_words::KeptSlot,
    /// Its entry in the input check across graphs, while it is kept (`--audit-graph-inputs`).
    _audit: super::input_audit::Registration,
}

// SAFETY: like Context, the whole bundle may move to another thread.
unsafe impl Send for KeptGraph {}

/// The buffers of `graph`'s nodes and sources (`used`) and of the tensors whose per-device
/// copies a meta buffer makes per graph (`resident`): every node (an intermediate or a view),
/// and every source that is an input (`kept`, or flagged) or a view. Static leaves (weights,
/// caches) are used but not resident.
fn graph_buffers(
    graph: *mut ffi::ggml_cgraph,
    kept: &[NonNull<ffi::ggml_tensor>],
) -> (Vec<usize>, Vec<usize>) {
    let (mut used, mut resident) = (Vec::new(), Vec::new());
    let buf = |t: *mut ffi::ggml_tensor| -> usize {
        // SAFETY: tensors of a live graph; a view's memory is its source's buffer.
        unsafe {
            let b = (*t).buffer;
            if b.is_null() && !(*t).view_src.is_null() {
                (*(*t).view_src).buffer as usize
            } else {
                b as usize
            }
        }
    };
    let input = ffi::ggml_tensor_flag_GGML_TENSOR_FLAG_INPUT as i32;
    // SAFETY: the graph and its tensors are live (the caller keeps them); srcs may be null.
    unsafe {
        for i in 0..ffi::ggml_graph_n_nodes(graph) {
            let node = ffi::ggml_graph_node(graph, i);
            used.push(buf(node));
            resident.push(buf(node));
            for &src in &(*node).src {
                if src.is_null() {
                    continue;
                }
                used.push(buf(src));
                if (*src).flags & input != 0 || !(*src).view_src.is_null() || (*src).op != 0 {
                    resident.push(buf(src));
                }
            }
        }
    }
    resident.extend(kept.iter().map(|t| buf(t.as_ptr())));
    for v in [&mut used, &mut resident] {
        v.retain(|&b| b != 0);
        v.sort_unstable();
        v.dedup();
    }
    (used, resident)
}

/// What [`KeptGraph::new`] keeps of a graph, without borrowing its context (so the context
/// can then move into the [`KeptGraph`]).
pub struct KeptParts {
    graph: NonNull<ffi::ggml_cgraph>,
    ctx: NonNull<ffi::ggml_context>,
    tensors: Vec<NonNull<ffi::ggml_tensor>>,
}

impl KeptParts {
    /// `graph` and `tensors` (the inputs to refill and outputs to read, addressed by index
    /// later), all of one context.
    pub fn new(graph: &Graph<'_>, tensors: &[Tensor<'_>]) -> Self {
        Self {
            graph: graph.raw,
            ctx: graph.ctx,
            tensors: tensors.iter().map(|t| t.raw).collect(),
        }
    }
}

impl KeptGraph {
    /// Keeps the graph of `parts`, built and allocated ([`Context::alloc_graph`]) in `ctx`.
    ///
    /// # Safety
    /// The tensors of `parts` must belong to `ctx`. The graph also references tensors of other
    /// contexts (weights, caches) that the borrow checker no longer tracks once `ctx` moves in
    /// here: the caller must keep them alive, at the same memory, for as long as it runs this
    /// graph.
    pub unsafe fn new(ctx: Context, parts: KeptParts) -> Self {
        assert_eq!(parts.ctx, ctx.raw, "the graph belongs to another context");
        assert!(
            ctx.allocator.borrow().is_some(),
            "keep a graph only after allocating it"
        );
        static SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let (used, resident) = graph_buffers(parts.graph.as_ptr(), &parts.tensors);
        let serial = SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut audit = super::input_audit::Registration::default();
        if super::input_audit::enabled() {
            let listed: Vec<_> = parts.tensors.iter().map(|t| t.as_ptr()).collect();
            // SAFETY: an allocated graph (asserted above) of live tensors, the caller's
            // contract; the listed tensors are what its runs fill.
            unsafe {
                super::input_audit::check(parts.graph.as_ptr(), &listed);
                audit = super::input_audit::register(serial, parts.graph.as_ptr(), &listed);
            }
        }
        super::stage_graph::stats::count_build();
        Self {
            graph: parts.graph,
            tensors: parts.tensors,
            serial,
            used,
            resident,
            ctx,
            model_bytes: Default::default(),
            stage: Default::default(),
            _audit: audit,
        }
    }

    /// Counts the weight bytes a run of the graph reads ([`super::model_bytes::count`]), with
    /// `is_weight` saying which leaf tensors are weights.
    pub fn set_model_bytes(&mut self, is_weight: &dyn Fn(*const ffi::ggml_tensor) -> bool) {
        // SAFETY: the kept graph and its tensors live as long as self.
        self.model_bytes = unsafe { super::model_bytes::count(self.graph.as_ptr(), is_weight) };
    }

    /// See [`Self::set_model_bytes`] (zero until it is called).
    pub fn model_bytes(&self) -> &super::model_bytes::GraphBytes {
        &self.model_bytes
    }

    /// Marks the graph as the MTP head's, for `--stage-graph-stats`' split of the counts.
    pub fn mark_mtp(&self) {
        self.stage
            .mtp
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// See the field.
    pub fn serial(&self) -> u64 {
        self.serial
    }

    /// The graph as a run of it sees it: its serial and its stage-graph slot
    /// ([`super::stage_graph::kept_words::KeptSlot`]).
    pub(crate) fn kept_run(&self) -> super::stage_graph::kept_words::KeptRun<'_> {
        super::stage_graph::kept_words::KeptRun {
            serial: self.serial,
            slot: &self.stage,
        }
    }

    /// The buffers a compute of this graph uses (sorted, as addresses).
    pub fn used_buffers(&self) -> &[usize] {
        &self.used
    }

    /// The buffers holding this graph's own per-device tensors on a meta backend:
    /// intermediates, inputs and views (sorted, as addresses).
    pub fn resident_buffers(&self) -> &[usize] {
        &self.resident
    }

    pub fn graph(&self) -> Graph<'_> {
        Graph {
            raw: self.graph,
            ctx: self.ctx.raw,
            _ctx: PhantomData,
        }
    }

    /// The `i`th tensor given to [`Self::new`].
    pub fn tensor(&self, i: usize) -> Tensor<'_> {
        Tensor {
            raw: self.tensors[i],
            _ctx: PhantomData,
        }
    }

    /// Ends the graph and returns its compute allocator for reuse.
    pub fn into_allocator(self) -> Option<ComputeAllocator> {
        self.ctx.into_allocator()
    }

    /// Initializes the graph's tensors in their buffers once more (same memory): its inputs, its
    /// ops' results and every view, sources first. A meta backend (tensor parallelism) keeps
    /// the per-device tensors of a graph's intermediates and views only until it computes
    /// another graph, so a kept graph needs this before it runs again after another one.
    /// Static tensors (weights, caches: leaves that are not inputs) keep theirs.
    pub fn reinit(&self) -> Result<(), GgmlError> {
        self.reinit_where(|_| true).map(|_| ())
    }

    /// [`Self::reinit`] of the tensors whose per-device copies a meta buffer `stale` (by
    /// address) may have dropped, and of every node that reads one of them (its per-device
    /// copy points at its sources' copies). A buffer where anything is initialized again is
    /// initialized again whole: a meta buffer files new per-device copies in its current
    /// container and drops the other one at the next rebuild, so the graph's copies in one
    /// buffer must all be new together. A graph whose own compute buffer is intact
    /// re-initializes only its tensors in the others (its cache writes, say).
    /// Returns how many tensors it initialized again.
    pub fn reinit_where(&self, stale: impl Fn(usize) -> bool) -> Result<usize, GgmlError> {
        let g = self.graph.as_ptr();
        // SAFETY: the graph and its tensors live as long as `self` (its context); every node
        // and src pointer ggml returns is a tensor of that graph, null-checked where it may be
        // null (a missing src, an unallocated tensor's buffer). Re-initializing a tensor in its
        // own meta buffer only rebuilds the per-device views of the same memory.
        unsafe {
            let count = std::cell::Cell::new(0usize);
            let init = |t: *mut ffi::ggml_tensor| -> Result<(), GgmlError> {
                let buf = (*t).buffer;
                if buf.is_null() || !ffi::ggml_backend_buffer_is_meta(buf) {
                    return Ok(());
                }
                count.set(count.get() + 1);
                // Views too: the meta buffer derives a view's per-device tensors from its source's.
                let status = ffi::ggml_backend_buffer_init_tensor(buf, t);
                if status == ffi::ggml_status_GGML_STATUS_SUCCESS {
                    Ok(())
                } else {
                    Err(GgmlError::Compute(status))
                }
            };
            // The buffers to redo whole: the stale ones, and those holding a node that reads a
            // tensor redone, until nothing changes.
            let mut redo: std::collections::HashSet<usize> = self
                .resident
                .iter()
                .copied()
                .filter(|&b| stale(b))
                .collect();
            loop {
                let before = redo.len();
                let mut marked = std::collections::HashSet::new();
                for i in 0..ffi::ggml_graph_n_nodes(g) {
                    let node = ffi::ggml_graph_node(g, i);
                    // A static leaf (weights, a cache) keeps its per-device copies for good.
                    let per_graph = |t: *mut ffi::ggml_tensor| {
                        (*t).op != 0
                            || !(*t).view_src.is_null()
                            || (*t).flags & ffi::ggml_tensor_flag_GGML_TENSOR_FLAG_INPUT as i32 != 0
                    };
                    let reads = (*node).src.iter().any(|&src| {
                        !src.is_null()
                            && (marked.contains(&(src as usize))
                                || (per_graph(src) && redo.contains(&((*src).buffer as usize))))
                    });
                    if reads || redo.contains(&((*node).buffer as usize)) {
                        marked.insert(node as usize);
                        redo.insert((*node).buffer as usize);
                    }
                }
                if redo.len() == before {
                    break;
                }
            }
            let in_stale = |t: *mut ffi::ggml_tensor| redo.contains(&((*t).buffer as usize));
            let input = |t: *mut ffi::ggml_tensor| {
                (*t).flags & ffi::ggml_tensor_flag_GGML_TENSOR_FLAG_INPUT as i32 != 0
            };
            // Initialized again: read by a node, it is initialized again too.
            let mut redone = std::collections::HashSet::new();
            // The kept tensors first: inputs no node reads still get filled.
            for t in &self.tensors {
                let t = t.as_ptr();
                if input(t) && in_stale(t) && redone.insert(t as usize) {
                    init(t)?;
                }
            }
            for i in 0..ffi::ggml_graph_n_nodes(g) {
                let node = ffi::ggml_graph_node(g, i);
                let mut reads_redone = false;
                // The inputs (leaves) a node reads, before it.
                for &src in &(*node).src {
                    if src.is_null() {
                        continue;
                    }
                    if input(src) && in_stale(src) && redone.insert(src as usize) {
                        init(src)?;
                    }
                    reads_redone |= redone.contains(&(src as usize));
                }
                if (reads_redone || in_stale(node)) && redone.insert(node as usize) {
                    init(node)?;
                }
            }
            Ok(count.get())
        }
    }

    /// The graph's identity, for telling kept graphs apart.
    pub fn id(&self) -> usize {
        self.graph.as_ptr() as usize
    }

    /// Bytes its compute allocator reserves.
    pub fn allocator_size(&self) -> usize {
        self.ctx.allocator_size()
    }
}

/// Plans and holds the memory for a graph's intermediate tensors. Its buffer grows to the
/// largest graph it has seen and is reused, so keep one per backend across steps.
/// A compute allocator several kept graphs allocate from ([`Context::alloc_graph_shared`]).
pub type SharedAllocator = std::sync::Arc<std::sync::Mutex<ComputeAllocator>>;

/// A context's compute allocator: its own, or one it shares.
enum Alloc {
    Own(ComputeAllocator),
    // Held to keep the shared buffer alive while the graph is.
    Shared(#[allow(dead_code)] SharedAllocator),
}

pub struct ComputeAllocator {
    raw: NonNull<ffi::ggml_gallocr>,
    buft: BufferType,
    /// Plan every graph afresh ([`set_fresh_plans`]).
    fresh: bool,
    /// A relayout taken on every graph (tests); `None`: `post_alloc`'s, on CUDA graphs.
    relayout: Option<super::post_alloc::Relayout>,
}

/// The tensors of `graph` an allocation gives memory: its nodes, their sources and its leafs
/// without memory yet (views too: they get their source's).
///
/// # Safety
/// A live graph.
unsafe fn unplanned(graph: *mut ffi::ggml_cgraph) -> Vec<*mut ffi::ggml_tensor> {
    // SAFETY: the caller's; nodes, their sources and leafs are valid tensors.
    unsafe {
        let n = ffi::ggml_graph_n_nodes(graph) as usize;
        let nodes = std::slice::from_raw_parts(ffi::ggml_graph_nodes(graph), n);
        let mut out = Vec::with_capacity(4 * n);
        let bare = |t: *mut ffi::ggml_tensor| (*t).data.is_null() && (*t).buffer.is_null();
        for &t in nodes.iter().chain(super::graph_uid::leafs(graph)) {
            if bare(t) {
                out.push(t);
            }
            out.extend(
                (*t).src
                    .iter()
                    .copied()
                    .filter(|&s| !s.is_null() && bare(s)),
            );
        }
        out
    }
}

/// Takes the memory an allocation gave `tensors` ([`unplanned`]) back off them, so the next
/// allocation plans them again.
///
/// # Safety
/// Valid tensors no node computed into yet.
unsafe fn forget_plan(tensors: &[*mut ffi::ggml_tensor]) {
    for &t in tensors {
        // SAFETY: the caller's.
        unsafe {
            (*t).data = std::ptr::null_mut();
            (*t).buffer = std::ptr::null_mut();
        }
    }
}

/// Whether compute allocators made from now on plan every graph afresh ([`set_fresh_plans`]).
static FRESH_PLANS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Makes the compute allocators created from now on plan each graph they allocate afresh.
///
/// ggml's graph allocator keeps the plan (every intermediate's offset in its buffer) it made for
/// its previous graph and reuses it for the next one whenever that has as many nodes and leaves
/// and each fits its predecessor's slot: a smaller step of the same shape gets the larger one's
/// offsets. Which tensors overlap then follows the graphs the allocator served before, and
/// ggml-cuda fuses some ops (the MoE weighted sum, #410) only when their memory does not
/// overlap, rounding differently when it does. A planned-afresh graph's layout depends on the
/// graph alone, so a step's sums do not depend on what the server ran before (#459). Costs one
/// planning pass per graph build; kept graphs are built once per shape. `--deterministic`
/// turns it on; off by default, where the plans (and so the bits) stay as they were.
pub fn set_fresh_plans(on: bool) {
    FRESH_PLANS.store(on, std::sync::atomic::Ordering::Relaxed);
}

// SAFETY: owned exclusively; moved between contexts, never shared.
unsafe impl Send for ComputeAllocator {}

impl Drop for ComputeAllocator {
    fn drop(&mut self) {
        unsafe { ffi::ggml_gallocr_free(self.raw.as_ptr()) }
    }
}

impl ComputeAllocator {
    pub fn new(buft: BufferType) -> Self {
        let raw = unsafe { ffi::ggml_gallocr_new(buft.raw()) };
        Self {
            raw: NonNull::new(raw).expect("ggml_gallocr_new failed"),
            buft,
            fresh: FRESH_PLANS.load(std::sync::atomic::Ordering::Relaxed),
            relayout: None,
        }
    }

    /// This allocator with fresh plans on or off ([`set_fresh_plans`] sets the default).
    pub fn with_fresh_plans(mut self, on: bool) -> Self {
        self.fresh = on;
        self
    }

    /// Allocates `graph`'s intermediates: from the previous graph's plan when it fits (ggml's
    /// rule), or, with fresh plans, from a plan of `graph` alone (the buffer is kept unless it
    /// must grow).
    fn alloc(&self, graph: &Graph<'_>) -> bool {
        let raw = graph.raw();
        // A rewrite needing a plan of its own (`post_alloc::set_relayout`): planned as built
        // first, rewritten from that layout, then planned again.
        // SAFETY: a live graph, not allocated yet.
        let relayout = match self.relayout {
            Some(r) if unsafe { (r.wants)(raw, false) } => Some((r, false)),
            Some(_) => None,
            None => unsafe { super::post_alloc::relayout(raw, self.buft) }.map(|r| (r, true)),
        };
        // SAFETY: as above.
        let unplanned = relayout.map(|_| unsafe { unplanned(raw) });
        let mut ok = self.plan(raw, self.fresh);
        if ok
            && let Some((r, cuda)) = relayout
            && let Some(relaid) = unsafe { (r.rewrite)(raw, cuda) }
        {
            let unplanned = unplanned.as_deref().unwrap_or_default();
            // SAFETY: allocated just now, nothing computed; the tensors the plan gave memory
            // are planned afresh, as rewritten, and if the rewrite no longer holds then, as
            // built again.
            unsafe {
                forget_plan(unplanned);
                ok = self.plan(raw, true);
                if ok && !relaid.holds(raw) {
                    relaid.undo(raw);
                    forget_plan(unplanned);
                    ok = self.plan(raw, true);
                }
            }
        }
        if ok {
            // SAFETY: allocated just now, nothing computed yet; a graph is allocated once
            // (`Context::alloc_graph`), so the rewritten nodes are never planned again.
            unsafe { super::post_alloc::run(graph.raw(), self.buft) };
        }
        ok
    }

    /// Plans `graph` (`fresh`: from a plan of its own, else from the previous one when it
    /// fits) and gives its intermediates memory.
    fn plan(&self, raw: *mut ffi::ggml_cgraph, fresh: bool) -> bool {
        // SAFETY: a live allocator and graph; reserving plans the graph and grows the buffer
        // if needed, and the allocation then finds the plan valid for this graph.
        unsafe {
            (!fresh || ffi::ggml_gallocr_reserve(self.raw.as_ptr(), raw))
                && ffi::ggml_gallocr_alloc_graph(self.raw.as_ptr(), raw)
        }
    }

    /// This allocator taking `r` on every graph it allocates (whatever its buffer type), as
    /// `cuda: false`: for tests of a [`super::post_alloc::Relayout`] on the CPU.
    #[cfg(test)]
    pub(crate) fn with_relayout(mut self, r: super::post_alloc::Relayout) -> Self {
        self.relayout = Some(r);
        self
    }

    /// [`Self::new`] over a buffer type that allocates more than the graph needs (a quarter
    /// and 64 MiB more), so graphs that each need a little more than the one before fit the
    /// buffer for a while instead of reallocating it every time (see `padded`). Costs that
    /// memory per allocator.
    pub fn padded(buft: BufferType) -> Self {
        Self::over(buft, super::padded::Pad::Quarter)
    }

    /// [`Self::padded`] with a little headroom only (a sixty-fourth and 16 MiB): for graphs that
    /// each need a few KiB more than the one before (#363: an integrated GPU's prompt chunks,
    /// whose 1.5 GiB buffer was freed and allocated again for every second chunk).
    pub fn headroom(buft: BufferType) -> Self {
        Self::over(buft, super::padded::Pad::Light)
    }

    fn over(buft: BufferType, pad: super::padded::Pad) -> Self {
        let over = super::padded::over(buft.raw(), pad);
        let raw = unsafe { ffi::ggml_gallocr_new(over) };
        Self {
            raw: NonNull::new(raw).expect("ggml_gallocr_new failed"),
            buft: BufferType(NonNull::new(over).expect("a padded buffer type")),
            fresh: FRESH_PLANS.load(std::sync::atomic::Ordering::Relaxed),
            relayout: None,
        }
    }

    /// Bytes currently reserved for intermediates.
    pub fn buffer_size(&self) -> usize {
        unsafe { ffi::ggml_gallocr_get_buffer_size(self.raw.as_ptr(), 0) }
    }

    /// [`Self::new`] with its buffer allocated now, `bytes` big: a graph that needs no more
    /// later allocates from it without a device allocation of its own. One that needs more
    /// grows it as usual.
    pub fn reserved(buft: BufferType, bytes: usize) -> Result<Self, GgmlError> {
        let a = Self::new(buft);
        // A graph of one leaf of `bytes` bytes: the allocator plans it and allocates its buffer.
        let ctx = Context::new(1, 1);
        let t = ctx.new_tensor(super::types::I8, &[bytes.max(1) as i64]);
        let mut graph = ctx.new_graph(1);
        graph.expand(t);
        // SAFETY: a live allocator and graph; reserving plans the graph and allocates the
        // allocator's buffer without touching the graph's tensors (it remembers no pointer to
        // them past the next plan).
        let ok = unsafe { ffi::ggml_gallocr_reserve(a.raw.as_ptr(), graph.raw()) };
        if ok {
            Ok(a)
        } else {
            Err(GgmlError::Alloc {
                bytes,
                buffer: buft.name(),
            })
        }
    }

    /// This allocator (new: it has allocated nothing yet) with its buffer allocated now, at
    /// least `bytes` big, and written once (zeros), so the buffer's memory is in place before a
    /// graph runs in it. Its plan is then one of a single leaf, as [`Self::reserved`] leaves
    /// it: the next graph is planned afresh, as in a new allocator, and only the buffer's size
    /// differs.
    pub fn touched(self, bytes: usize) -> Result<Self, GgmlError> {
        let ctx = Context::new(1, 1);
        let t = ctx.new_tensor(super::types::I8, &[bytes.max(1) as i64]);
        let mut graph = ctx.new_graph(1);
        graph.expand(t);
        // SAFETY: a live allocator and graph; allocating plans the graph's one leaf, allocates
        // the buffer and points the leaf into it (the leaf is dropped with `ctx`, and the
        // allocator keeps no pointer to it past its next plan).
        let ok = unsafe { ffi::ggml_gallocr_alloc_graph(self.raw.as_ptr(), graph.raw()) };
        if !ok {
            return Err(GgmlError::Alloc {
                bytes,
                buffer: self.buft.name(),
            });
        }
        t.memset(0, t.nbytes(), 0);
        Ok(self)
    }
}

/// The compute buffer `graph`'s intermediates need in `buft`, planned by a fresh allocator
/// without allocating (what an allocation that failed asked for).
fn graph_buffer_size(buft: BufferType, graph: &Graph<'_>) -> usize {
    let a = ComputeAllocator::new(buft);
    let mut size = 0usize;
    // SAFETY: a fresh allocator of one buffer (null buffer ids: all in buffer 0) plans the
    // live graph without allocating or touching its tensors; `size` has room for one buffer.
    unsafe {
        ffi::ggml_gallocr_reserve_n_size(
            a.raw.as_ptr(),
            graph.raw(),
            std::ptr::null(),
            std::ptr::null(),
            &mut size,
        )
    };
    size
}

#[cfg(test)]
mod tests;
