//! Host tiers of tensor-parallel KV caches (#317): each GPU's slice of a cache tensor is one
//! virtual range whose first rows are VRAM and the rest pinned memory on the GPU's host NUMA
//! node, so ggml's ops read a row through the same pointer wherever it lives (as
//! [`super::tiered_rows`] does for a one-GPU cache).
//!
//! ggml's meta buffer type allocates each GPU's slices with that GPU's own buffer type
//! (`ggml_backend_dev_buffer_type`), so the meta device for such a cache is made over wrappers
//! of the CUDA devices that differ from them only there: the wrapper's buffer type below.
//!
//! An alias type's buffers are the range of the live buffer of the same GPU and size, with no
//! memory of their own: a second meta buffer over the same cache, whose tensors another kind
//! of graph views. A meta buffer drops the per-device copies of the views one graph made when
//! another graph that views it runs, and Legion then re-initializes the whole graph (see
//! `Context::reinit_where`); the MTP head's graphs view the cache through the alias, so the
//! target's and the head's graphs, which alternate every step, keep theirs.

use super::*;
use crate::ggml::Device;

/// One GPU's tiered buffer type: tensors get `hbm_rows` rows in VRAM (rounded up to the
/// mapping granule), the rest on host node `node`; with `slot_rows`, each slot of that many
/// rows does.
struct TierType {
    device: usize,
    node: u32,
    hbm_rows: usize,
    slot_rows: usize,
    /// An alias: buffers over the live buffer of the same GPU and size.
    alias: bool,
    /// Slots whose host rows are mapped as their sequences grow ([`tp_tier_back`]), not all
    /// at once: slots whose whole context the host could not hold (#1131).
    lazy: bool,
    gran: usize,
    name: CString,
    buft: ffi::ggml_backend_buffer_type,
    /// The CUDA device with this buffer type in its place.
    dev: ffi::ggml_backend_device,
}

static TIER_TYPES: Registry<TierType> = Registry(Mutex::new(Vec::new()));

/// The live (not alias) buffers: GPU, range, size; most recent last.
static LIVE: Mutex<Vec<(usize, CUdeviceptr, usize)>> = Mutex::new(Vec::new());

/// A buffer: a reserved range, mapped tensor by tensor as ggml places them.
struct TierBuffer {
    device: usize,
    va: CUdeviceptr,
    size: usize,
    /// Mapped pieces: offset, bytes, physical handle.
    maps: Vec<(usize, usize, CUmemGenericAllocationHandle)>,
    /// Another buffer's range (an alias type's): nothing to map or free.
    alias: bool,
    /// A lazy type's slots (see [`TierType::lazy`]).
    lazy: Option<LazySlots>,
}

/// Rows of a lazy slot backed at once: [`tp_tier_back`] rounds up to them.
pub const LAZY_CHUNK_ROWS: usize = 4096;

/// A lazy tier buffer's slots: their VRAM rows are mapped with the tensors, their host rows as
/// [`tp_tier_back`] asks. Until then a host chunk ([`LAZY_CHUNK_ROWS`] rows) maps one shared
/// zeroed piece of host memory: attention reads a batch's longest span in every sequence's slot
/// (masked, by an exact 0 weight), so each slot's whole range must hold finite numbers, and
/// nothing writes to rows past a sequence's own.
struct LazySlots {
    slot_rows: usize,
    node: u32,
    /// Per tensor: its offset in the range, its row bytes and each slot's VRAM bytes.
    tensors: Vec<(usize, usize, usize)>,
    /// Per tensor and slot: the bytes from the slot's start backed so far (by VRAM or its own
    /// host memory; the shared zeros past them).
    backed: Vec<Vec<usize>>,
    /// Per slot: the host pieces mapped for it (offset, bytes, handle).
    pieces: Vec<Vec<(usize, usize, CUmemGenericAllocationHandle)>>,
    /// The shared zeroed pieces: bytes, handle.
    zeros: Vec<(usize, CUmemGenericAllocationHandle)>,
}

/// The live lazy tier buffers (their [`TierBuffer`] boxes).
static LAZY: Mutex<Vec<usize>> = Mutex::new(Vec::new());

impl TierBuffer {
    fn release(&self) {
        // SAFETY: only what this buffer mapped and reserved.
        unsafe {
            for &(off, n, h) in &self.maps {
                cuMemUnmap(self.va + off as u64, n);
                cuMemRelease(h);
            }
            if let Some(l) = &self.lazy {
                for (t, &(off, row, _)) in l.tensors.iter().enumerate() {
                    let (slot, chunk) = (l.slot_rows * row, LAZY_CHUNK_ROWS * row);
                    for (s, &have) in l.backed[t].iter().enumerate() {
                        for at in (have..slot).step_by(chunk) {
                            cuMemUnmap(self.va + (off + s * slot + at) as u64, chunk);
                        }
                    }
                }
                for &(off, n, h) in l.pieces.iter().flatten() {
                    cuMemUnmap(self.va + off as u64, n);
                    cuMemRelease(h);
                }
                for &(_, h) in &l.zeros {
                    cuMemRelease(h);
                }
            }
            cuMemAddressFree(self.va, self.size);
        }
    }

    /// Backs `n` bytes at `off` with new memory at `location` and lets the device use them.
    fn map(&mut self, off: usize, n: usize, location: (u32, i32)) -> Result<(), String> {
        if n == 0 {
            return Ok(());
        }
        let h = self.map_piece(off, n, location)?;
        self.maps.push((off, n, h));
        Ok(())
    }

    /// [`Self::map`] without keeping the piece in `maps`: the caller keeps its handle.
    fn map_piece(
        &self,
        off: usize,
        n: usize,
        location: (u32, i32),
    ) -> Result<CUmemGenericAllocationHandle, String> {
        let prop = alloc_prop(location.0, location.1);
        let mut h: CUmemGenericAllocationHandle = 0;
        // SAFETY: a new allocation mapped into this buffer's reserved range.
        unsafe {
            cu(cuMemCreate(&mut h, n, &prop, 0), "cuMemCreate(KV tier)")?;
            if let Err(e) = cu(
                cuMemMap(self.va + off as u64, n, 0, h, 0),
                "cuMemMap(KV tier)",
            ) {
                cuMemRelease(h);
                return Err(e);
            }
            let a = access(CU_MEM_LOCATION_TYPE_DEVICE, self.device as i32);
            if let Err(e) = cu(
                cuMemSetAccess(self.va + off as u64, n, &a, 1),
                "cuMemSetAccess(KV tier)",
            ) {
                cuMemUnmap(self.va + off as u64, n);
                cuMemRelease(h);
                return Err(e);
            }
        }
        Ok(h)
    }
}

fn tier_type<'a>(buft: ffi::ggml_backend_buffer_type_t) -> &'a TierType {
    // SAFETY: the context is the registered TierType.
    unsafe { &*((*buft).context as *const TierType) }
}

fn tier_buffer<'a>(buffer: ffi::ggml_backend_buffer_t) -> &'a mut TierBuffer {
    // SAFETY: the context is the boxed TierBuffer of this buffer.
    unsafe { &mut *((*buffer).context as *mut TierBuffer) }
}

unsafe extern "C" fn tier_name(buft: ffi::ggml_backend_buffer_type_t) -> *const c_char {
    tier_type(buft).name.as_ptr()
}

unsafe extern "C" fn tier_alignment(buft: ffi::ggml_backend_buffer_type_t) -> usize {
    tier_type(buft).gran
}

/// Whole granules per tensor: no two tensors share one, so each maps its own.
unsafe extern "C" fn tier_alloc_size(
    buft: ffi::ggml_backend_buffer_type_t,
    t: *const ffi::ggml_tensor,
) -> usize {
    // SAFETY: a live tensor.
    round_up(unsafe { ffi::ggml_nbytes(t) }, tier_type(buft).gran)
}

unsafe extern "C" fn tier_alloc(
    buft: ffi::ggml_backend_buffer_type_t,
    size: usize,
) -> ffi::ggml_backend_buffer_t {
    let ty = tier_type(buft);
    let size = round_up(size.max(1), ty.gran);
    if let Err(e) = use_device(ty.device) {
        set_error(e);
        return null_mut();
    }
    let mut va: CUdeviceptr = 0;
    if ty.alias {
        let live = LIVE.lock().unwrap();
        match live.iter().rev().find(|l| l.0 == ty.device && l.2 == size) {
            Some(l) => va = l.1,
            None => {
                set_error(format!(
                    "KV tier alias: no live {size}-byte buffer on CUDA{}",
                    ty.device
                ));
                return null_mut();
            }
        }
    } else {
        // SAFETY: reserves a range; nothing is mapped until the tensors are placed.
        if let Err(e) = cu(
            unsafe { cuMemAddressReserve(&mut va, size, ty.gran, 0, 0) },
            "cuMemAddressReserve(KV tier)",
        ) {
            set_error(e);
            return null_mut();
        }
        LIVE.lock().unwrap().push((ty.device, va, size));
    }
    let b = Box::new(TierBuffer {
        device: ty.device,
        va,
        size,
        maps: Vec::new(),
        alias: ty.alias,
        lazy: (ty.lazy && !ty.alias).then(|| LazySlots {
            slot_rows: ty.slot_rows,
            node: ty.node,
            tensors: Vec::new(),
            backed: Vec::new(),
            pieces: Vec::new(),
            zeros: Vec::new(),
        }),
    });
    if b.lazy.is_some() {
        LAZY.lock().unwrap().push(&*b as *const TierBuffer as usize);
    }
    let iface = ffi::ggml_backend_buffer_i {
        free_buffer: Some(tier_free),
        get_base: Some(tier_base),
        init_tensor: Some(tier_init_tensor),
        memset_tensor: Some(tier_memset),
        set_tensor: Some(tier_set),
        get_tensor: Some(tier_get),
        set_tensor_2d: None,
        get_tensor_2d: None,
        cpy_tensor: None,
        clear: Some(tier_clear),
        reset: None,
    };
    // SAFETY: ggml takes the interface and the boxed context (freed by tier_free).
    unsafe { ffi::ggml_backend_buffer_init(buft, iface, Box::into_raw(b) as *mut c_void, size) }
}

unsafe extern "C" fn tier_free(buffer: ffi::ggml_backend_buffer_t) {
    let b = tier_buffer(buffer);
    if !b.alias {
        LIVE.lock().unwrap().retain(|l| l.1 != b.va);
    }
    if b.lazy.is_some() {
        let me = &*b as *const TierBuffer as usize;
        LAZY.lock().unwrap().retain(|&p| p != me);
    }
    // SAFETY: no kernel may still read the range; then frees the box handed to ggml.
    unsafe {
        if !b.alias {
            cudaSetDevice(b.device as c_int);
            cudaDeviceSynchronize();
            b.release();
        }
        drop(Box::from_raw((*buffer).context as *mut TierBuffer));
    }
}

unsafe extern "C" fn tier_base(buffer: ffi::ggml_backend_buffer_t) -> *mut c_void {
    tier_buffer(buffer).va as *mut c_void
}

/// Maps a tensor's granules as ggml places it: its first `hbm_rows` rows in VRAM, the rest on
/// the host node; or each of its slots of `slot_rows` rows so. A slot need not be whole
/// granules (a q8_0 row is 17/16 of a power of two): a granule is in VRAM when it lies within
/// a slot's first `hbm_rows` rows, so a slot may keep a granule's rows less there. Views share
/// their source's memory.
unsafe extern "C" fn tier_init_tensor(
    buffer: ffi::ggml_backend_buffer_t,
    t: *mut ffi::ggml_tensor,
) -> ffi::ggml_status {
    // SAFETY: a live tensor of this buffer.
    let (view, data, nbytes, row) = unsafe {
        (
            !(*t).view_src.is_null(),
            (*t).data as u64,
            ffi::ggml_nbytes(t),
            (*t).nb[1],
        )
    };
    if view || nbytes == 0 || tier_buffer(buffer).alias {
        return ffi::ggml_status_GGML_STATUS_SUCCESS;
    }
    // SAFETY: the buffer's type is a TierType.
    let ty = tier_type(unsafe { (*buffer).buft });
    let b = tier_buffer(buffer);
    let off = (data - b.va) as usize;
    let bytes = round_up(nbytes, ty.gran);
    let slot = match ty.slot_rows * row {
        0 => bytes,
        n => n.min(bytes),
    };
    let host = (CU_MEM_LOCATION_TYPE_HOST_NUMA, ty.node as i32);
    let device = (CU_MEM_LOCATION_TYPE_DEVICE, ty.device as i32);
    let done = if b.lazy.is_some() && ty.slot_rows > 0 {
        lazy_init(b, ty, off, bytes, row, slot)
    } else if ty.slot_rows == 0 {
        let vram = round_up(ty.hbm_rows * row, ty.gran).min(bytes);
        b.map(off, vram, device)
            .and_then(|_| b.map(off + vram, bytes - vram, host))
    } else {
        let vram = (ty.hbm_rows * row).min(slot);
        let in_vram = |g: usize| {
            let (start, end) = (g * ty.gran, (g + 1) * ty.gran);
            end <= start / slot * slot + vram
        };
        // Runs of granules in one place, each mapped at once.
        let n = bytes / ty.gran;
        let mut first = 0;
        (1..=n).try_for_each(|g| {
            if g < n && in_vram(g) == in_vram(first) {
                return Ok(());
            }
            let at = (first * ty.gran, (g - first) * ty.gran);
            let r = b.map(off + at.0, at.1, if in_vram(first) { device } else { host });
            first = g;
            r
        })
    };
    match done {
        Ok(()) => ffi::ggml_status_GGML_STATUS_SUCCESS,
        Err(e) => {
            tracing::error!(device = ty.device, node = ty.node, "KV tier: {e}");
            set_error(e);
            ffi::ggml_status_GGML_STATUS_ALLOC_FAILED
        }
    }
}

/// A lazy type's tensor at `off` (`bytes`, whole granules; rows of `row` bytes; slots of
/// `slot` bytes): each slot's VRAM rows mapped now, its host chunks to the shared zeros; the
/// rows past the last whole slot (pad cells) on the host now. Chunks must be whole granules, so
/// that each slot's memory is its own.
fn lazy_init(
    b: &mut TierBuffer,
    ty: &TierType,
    off: usize,
    bytes: usize,
    row: usize,
    slot: usize,
) -> Result<(), String> {
    let vram = (ty.hbm_rows * row).min(slot);
    let chunk = LAZY_CHUNK_ROWS * row;
    if chunk % ty.gran != 0 || slot % chunk != 0 || vram % chunk != 0 {
        return Err(format!(
            "lazy KV slots of {slot} bytes ({vram} in VRAM) are not whole {chunk}-byte chunks \
             of {}-byte granules",
            ty.gran
        ));
    }
    let n_slots = bytes / slot;
    let device = (CU_MEM_LOCATION_TYPE_DEVICE, ty.device as i32);
    let zero = b.zero_piece(chunk)?;
    for s in 0..n_slots {
        b.map(off + s * slot, vram, device)?;
        for at in (vram..slot).step_by(chunk) {
            b.map_zero(off + s * slot + at, chunk, zero)?;
        }
    }
    let tail = n_slots * slot;
    b.map(
        off + tail,
        bytes - tail,
        (CU_MEM_LOCATION_TYPE_HOST_NUMA, ty.node as i32),
    )?;
    let l = b.lazy.as_mut().expect("a lazy buffer");
    l.tensors.push((off, row, vram));
    l.backed.push(vec![vram; n_slots]);
    if l.pieces.len() < n_slots {
        l.pieces.resize_with(n_slots, Vec::new);
    }
    Ok(())
}

impl TierBuffer {
    /// The shared zeroed host piece of `bytes` of a lazy buffer, made (and zeroed) on first use.
    fn zero_piece(&mut self, bytes: usize) -> Result<CUmemGenericAllocationHandle, String> {
        let l = self.lazy.as_ref().expect("a lazy buffer");
        if let Some(&(_, h)) = l.zeros.iter().find(|z| z.0 == bytes) {
            return Ok(h);
        }
        let node = l.node;
        // Zeroed through a short-lived range of its own.
        let prop = alloc_prop(CU_MEM_LOCATION_TYPE_HOST_NUMA, node as i32);
        let mut h: CUmemGenericAllocationHandle = 0;
        let mut va: CUdeviceptr = 0;
        // SAFETY: a new allocation, mapped into a new range to be zeroed, then unmapped.
        unsafe {
            cu(
                cuMemCreate(&mut h, bytes, &prop, 0),
                "cuMemCreate(KV tier zeros)",
            )?;
            let zeroed = (|| {
                cu(
                    cuMemAddressReserve(&mut va, bytes, 0, 0, 0),
                    "cuMemAddressReserve(KV tier zeros)",
                )?;
                let r = (|| {
                    cu(cuMemMap(va, bytes, 0, h, 0), "cuMemMap(KV tier zeros)")?;
                    let a = access(CU_MEM_LOCATION_TYPE_DEVICE, self.device as i32);
                    let r = cu(
                        cuMemSetAccess(va, bytes, &a, 1),
                        "cuMemSetAccess(KV tier zeros)",
                    )
                    .and_then(|_| {
                        rt(
                            cudaMemset(va as *mut c_void, 0, bytes),
                            "cudaMemset(KV tier zeros)",
                        )
                    })
                    .and_then(|_| rt(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));
                    cuMemUnmap(va, bytes);
                    r
                })();
                cuMemAddressFree(va, bytes);
                r
            })();
            if let Err(e) = zeroed {
                cuMemRelease(h);
                return Err(e);
            }
        }
        self.lazy
            .as_mut()
            .expect("a lazy buffer")
            .zeros
            .push((bytes, h));
        Ok(h)
    }

    /// Maps shared piece `h` (`n` bytes) at `off`, which the device may then read and write.
    fn map_zero(
        &self,
        off: usize,
        n: usize,
        h: CUmemGenericAllocationHandle,
    ) -> Result<(), String> {
        let at = self.va + off as u64;
        // SAFETY: an unmapped stretch of this buffer's reserved range.
        unsafe {
            cu(cuMemMap(at, n, 0, h, 0), "cuMemMap(KV tier zeros)")?;
            let a = access(CU_MEM_LOCATION_TYPE_DEVICE, self.device as i32);
            if let Err(e) = cu(
                cuMemSetAccess(at, n, &a, 1),
                "cuMemSetAccess(KV tier zeros)",
            ) {
                cuMemUnmap(at, n);
                return Err(e);
            }
        }
        Ok(())
    }
}

/// Backs sequence slot `slot` of every lazy tier buffer up to `rows` rows (#1131): each
/// tensor's slot rows past those it has, up to `rows` rounded up to whole chunks
/// ([`LAZY_CHUNK_ROWS`], at most the slot), in new pinned memory on the GPU's host node, zeroed
/// (as the shared zeros it replaces). Rows backed before stay.
pub fn tp_tier_back(slot: usize, rows: usize) -> Result<(), String> {
    let list = LAZY.lock().unwrap();
    for &p in list.iter() {
        // SAFETY: a live lazy buffer: `tier_free` takes it off the list before freeing it, under
        // the same lock.
        let b = unsafe { &mut *(p as *mut TierBuffer) };
        let Some(l) = b.lazy.as_ref() else {
            continue;
        };
        let rows = rows.next_multiple_of(LAZY_CHUNK_ROWS).min(l.slot_rows);
        let mut new = Vec::new();
        for (t, &(off, row, _)) in l.tensors.iter().enumerate() {
            let Some(&have) = l.backed[t].get(slot) else {
                continue;
            };
            if rows * row > have {
                new.push((t, off + slot * l.slot_rows * row, have, rows * row, row));
            }
        }
        if new.is_empty() {
            continue;
        }
        use_device(b.device)?;
        let node = l.node;
        // SAFETY: waits for whatever may still read the shared zeros there.
        rt(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize")?;
        for (t, base, have, want, row) in new {
            let chunk = LAZY_CHUNK_ROWS * row;
            // SAFETY: the shared zeros' mappings in the stretch, each mapped on its own.
            unsafe {
                for at in (have..want).step_by(chunk) {
                    cuMemUnmap(b.va + (base + at) as u64, chunk);
                }
            }
            let host = (CU_MEM_LOCATION_TYPE_HOST_NUMA, node as i32);
            let h = match b.map_piece(base + have, want - have, host) {
                Ok(h) => h,
                Err(e) => {
                    // The zeros back in place: the range stays readable.
                    let zero = b.zero_piece(chunk)?;
                    for at in (have..want).step_by(chunk) {
                        b.map_zero(base + at, chunk, zero)?;
                    }
                    return Err(e);
                }
            };
            let l = b.lazy.as_mut().expect("a lazy buffer");
            l.pieces[slot].push((base + have, want - have, h));
            l.backed[t][slot] = want;
            // SAFETY: the piece just mapped, which the device may access.
            rt(
                unsafe { cudaMemset((b.va + (base + have) as u64) as *mut c_void, 0, want - have) },
                "cudaMemset(KV tier)",
            )?;
        }
        // The zeros land before any stream reads them (ggml's streams do not wait for the
        // legacy stream's memset).
        // SAFETY: a plain synchronization of the current device.
        rt(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize")?;
    }
    Ok(())
}

/// Gives back the host memory [`tp_tier_back`] mapped for slot `slot` of every lazy tier
/// buffer (no sequence is in the slot), once the GPUs finished what may still read it: the
/// shared zeros take its place. Its VRAM rows stay.
pub fn tp_tier_trim(slot: usize) -> Result<(), String> {
    let list = LAZY.lock().unwrap();
    for &p in list.iter() {
        // SAFETY: as in `tp_tier_back`.
        let b = unsafe { &mut *(p as *mut TierBuffer) };
        let va = b.va;
        if b.lazy
            .as_ref()
            .is_none_or(|l| l.pieces.get(slot).is_none_or(|p| p.is_empty()))
        {
            continue;
        }
        use_device(b.device)?;
        let l = b.lazy.as_mut().expect("a lazy buffer");
        let pieces = std::mem::take(&mut l.pieces[slot]);
        // SAFETY: waits for the device, then unmaps and releases this slot's own pieces.
        unsafe {
            rt(cudaDeviceSynchronize(), "cudaDeviceSynchronize")?;
            for (off, n, h) in pieces {
                cuMemUnmap(va + off as u64, n);
                cuMemRelease(h);
            }
        }
        let l = b.lazy.as_ref().expect("a lazy buffer");
        let stretches: Vec<(usize, usize, usize, usize)> = l
            .tensors
            .iter()
            .enumerate()
            .map(|(t, &(off, row, vram))| {
                (t, off + slot * l.slot_rows * row, vram, l.backed[t][slot])
            })
            .collect();
        let rows: Vec<usize> = l.tensors.iter().map(|t| t.1).collect();
        for (t, base, vram, have) in stretches {
            let chunk = LAZY_CHUNK_ROWS * rows[t];
            let zero = b.zero_piece(chunk)?;
            for at in (vram..have).step_by(chunk) {
                b.map_zero(base + at, chunk, zero)?;
            }
            b.lazy.as_mut().expect("a lazy buffer").backed[t][slot] = vram;
        }
    }
    Ok(())
}

unsafe extern "C" fn tier_memset(
    buffer: ffi::ggml_backend_buffer_t,
    t: *mut ffi::ggml_tensor,
    v: u8,
    off: usize,
    n: usize,
) {
    // SAFETY: a range of the tensor, addressable by the device.
    unsafe {
        cudaSetDevice(tier_buffer(buffer).device as c_int);
        cudaMemset(
            ((*t).data as *mut u8).add(off) as *mut c_void,
            v as c_int,
            n,
        );
    }
}

unsafe extern "C" fn tier_set(
    buffer: ffi::ggml_backend_buffer_t,
    t: *mut ffi::ggml_tensor,
    data: *const c_void,
    off: usize,
    n: usize,
) {
    // SAFETY: as above.
    unsafe {
        cudaSetDevice(tier_buffer(buffer).device as c_int);
        cudaMemcpy(
            ((*t).data as *mut u8).add(off) as *mut c_void,
            data,
            n,
            CUDA_MEMCPY_DEFAULT,
        );
    }
}

unsafe extern "C" fn tier_get(
    buffer: ffi::ggml_backend_buffer_t,
    t: *const ffi::ggml_tensor,
    data: *mut c_void,
    off: usize,
    n: usize,
) {
    // SAFETY: as above.
    unsafe {
        cudaSetDevice(tier_buffer(buffer).device as c_int);
        cudaMemcpy(
            data,
            ((*t).data as *const u8).add(off) as *const c_void,
            n,
            CUDA_MEMCPY_DEFAULT,
        );
    }
}

unsafe extern "C" fn tier_clear(buffer: ffi::ggml_backend_buffer_t, v: u8) {
    let b = tier_buffer(buffer);
    // SAFETY: the mapped pieces only (the rest of the range has no memory behind it).
    unsafe {
        cudaSetDevice(b.device as c_int);
        for &(off, n, _) in &b.maps {
            cudaMemset((b.va + off as u64) as *mut c_void, v as c_int, n);
        }
        cudaDeviceSynchronize();
    }
}

unsafe extern "C" fn tier_dev_buffer_type(
    dev: ffi::ggml_backend_dev_t,
) -> ffi::ggml_backend_buffer_type_t {
    let mut list = TIER_TYPES.0.lock().unwrap();
    match list.iter_mut().find(|t| std::ptr::eq(&t.dev, dev)) {
        Some(t) => &mut t.buft,
        None => null_mut(),
    }
}

/// The CUDA devices `devices` (ggml's indices) as devices whose buffer type keeps a tensor's
/// first `hbm_rows` rows in the device's memory and the rest in pinned memory on its host NUMA
/// node (with `slot_rows` > 0: each slot's of that many rows), for a meta device
/// ([`crate::ggml::meta::meta_device`]) over them: its buffer type allocates a tensor-parallel
/// KV cache with a host tier. Fails where the driver cannot map host NUMA memory (VMM).
/// `alias`: devices of the alias type (see the module), for a second meta buffer over the
/// cache allocated just before with the others. `lazy` (with slots): each slot's host rows are
/// mapped only as [`tp_tier_back`] asks, for slots the host could not hold all at once.
pub fn tp_tier_devices(
    devices: &[usize],
    hbm_rows: usize,
    slot_rows: usize,
    alias: bool,
    lazy: bool,
) -> Result<Vec<Device>, String> {
    devices
        .iter()
        .map(|&device| {
            let caps = caps(device).ok_or("cannot query the GPU")?;
            if !caps.host_numa_vmm || caps.host_numa_id < 0 {
                return Err(format!(
                    "CUDA{device}'s driver cannot map host NUMA memory (cuMemCreate)"
                ));
            }
            let node = caps.host_numa_id as u32;
            let mut list = TIER_TYPES.0.lock().unwrap();
            if let Some(t) = list.iter_mut().find(|t| {
                t.device == device
                    && t.hbm_rows == hbm_rows
                    && t.slot_rows == slot_rows
                    && t.alias == alias
                    && t.lazy == lazy
            }) {
                // SAFETY: a registered device, never freed.
                return Ok(unsafe { Device::from_raw(NonNull::from(&mut t.dev)) });
            }
            // SAFETY: ggml's CUDA device for this index; its interface, registry and context
            // are static.
            let cuda = unsafe {
                *ffi::ggml_backend_buft_get_device(ffi::ggml_backend_cuda_buffer_type(
                    device as c_int,
                ))
            };
            let gran = caps.device_granularity.max(caps.host_granularity).max(256);
            let t: &'static mut TierType = Box::leak(Box::new(TierType {
                device,
                node,
                hbm_rows,
                slot_rows,
                alias,
                lazy,
                gran,
                name: CString::new(format!(
                    "CUDA{device}_KvTier{node}{}{}",
                    if lazy { "Lazy" } else { "" },
                    if alias { "Alias" } else { "" }
                ))
                .unwrap(),
                buft: ffi::ggml_backend_buffer_type {
                    iface: ffi::ggml_backend_buffer_type_i {
                        get_name: Some(tier_name),
                        alloc_buffer: Some(tier_alloc),
                        alloc_buffer_n: None,
                        get_alignment: Some(tier_alignment),
                        get_max_size: None,
                        get_alloc_size: Some(tier_alloc_size),
                        get_alloc_size_n: None,
                        is_host: None,
                    },
                    // SAFETY: as above.
                    device: unsafe {
                        ffi::ggml_backend_buft_get_device(ffi::ggml_backend_cuda_buffer_type(
                            device as c_int,
                        ))
                    },
                    context: null_mut(),
                },
                dev: ffi::ggml_backend_device {
                    iface: ffi::ggml_backend_device_i {
                        get_buffer_type: Some(tier_dev_buffer_type),
                        ..cuda.iface
                    },
                    ..cuda
                },
            }));
            t.buft.context = t as *mut TierType as *mut c_void;
            let dev = NonNull::from(&mut t.dev);
            list.push(t);
            // SAFETY: registered for the process.
            Ok(unsafe { Device::from_raw(dev) })
        })
        .collect()
}
