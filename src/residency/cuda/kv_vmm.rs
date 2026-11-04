//! A GPU's KV pools in a reserved virtual range (`--kv-vmm`, #814): the buffer type
//! reserves the range, and each cache tensor, placed on a mapping granule, gets VRAM
//! allocations of its own of at most [`KV_VMM_CHUNK`] bytes ([`crate::residency::kv_vmm`]),
//! all mapped when ggml places it. Kernels and ggml's ops read and write it as device memory.
//! What a pool in it adds is that the memory behind a tensor's last blocks is a set of
//! allocations of its own, which can be unmapped without touching the rest.

use super::*;
use crate::residency::kv_vmm::{KV_VMM_CHUNK, pieces};

struct VmmType {
    device: usize,
    gran: usize,
    name: CString,
    buft: ffi::ggml_backend_buffer_type,
}

static VMM_TYPES: Registry<VmmType> = Registry(Mutex::new(Vec::new()));

/// A buffer: a reserved range, mapped tensor by tensor as ggml places them.
struct VmmBuffer {
    device: usize,
    va: CUdeviceptr,
    size: usize,
    /// The tensors' pieces: offset, bytes, physical handle (0: unmapped, lent by
    /// [`kv_vmm_unmap`]).
    maps: Vec<(usize, usize, CUmemGenericAllocationHandle)>,
}

impl VmmBuffer {
    fn release(&self) {
        // SAFETY: only what this buffer mapped and reserved.
        unsafe {
            for &(off, n, h) in self.maps.iter().filter(|m| m.2 != 0) {
                cuMemUnmap(self.va + off as u64, n);
                cuMemRelease(h);
            }
            cuMemAddressFree(self.va, self.size);
        }
    }

    /// Backs `n` bytes at `off` with new VRAM and lets the device use them.
    fn map(&mut self, off: usize, n: usize) -> Result<(), String> {
        let prop = alloc_prop(CU_MEM_LOCATION_TYPE_DEVICE, self.device as i32);
        let mut h: CUmemGenericAllocationHandle = 0;
        let va = self.va + off as u64;
        // SAFETY: a new allocation mapped into this buffer's reserved, unmapped range.
        unsafe {
            cu(cuMemCreate(&mut h, n, &prop, 0), "cuMemCreate(KV pool)")?;
            if let Err(e) = cu(cuMemMap(va, n, 0, h, 0), "cuMemMap(KV pool)") {
                cuMemRelease(h);
                return Err(e);
            }
            self.maps.push((off, n, h));
            let a = access(CU_MEM_LOCATION_TYPE_DEVICE, self.device as i32);
            cu(cuMemSetAccess(va, n, &a, 1), "cuMemSetAccess(KV pool)")
        }
    }

    /// Backs piece `i`, unmapped by [`kv_vmm_unmap`], with new VRAM again and zeroes it.
    fn back(&mut self, i: usize) -> Result<(), String> {
        let (off, n, _) = self.maps[i];
        let prop = alloc_prop(CU_MEM_LOCATION_TYPE_DEVICE, self.device as i32);
        let mut h: CUmemGenericAllocationHandle = 0;
        let va = self.va + off as u64;
        // SAFETY: a new allocation mapped into this buffer's piece, which has none now.
        unsafe {
            cu(cuMemCreate(&mut h, n, &prop, 0), "cuMemCreate(KV pool)")?;
            if let Err(e) = cu(cuMemMap(va, n, 0, h, 0), "cuMemMap(KV pool)") {
                cuMemRelease(h);
                return Err(e);
            }
            let a = access(CU_MEM_LOCATION_TYPE_DEVICE, self.device as i32);
            if let Err(e) = cu(cuMemSetAccess(va, n, &a, 1), "cuMemSetAccess(KV pool)") {
                cuMemUnmap(va, n);
                cuMemRelease(h);
                return Err(e);
            }
            self.maps[i].2 = h;
            // Masked cells are still multiplied (by an exact 0): finite, as a new pool's.
            rt(cudaMemset(va as *mut c_void, 0, n), "cudaMemset(KV pool)")
        }
    }
}

/// The `--kv-vmm` buffer of tensor `t` and its offset in it, if `t` lives in one.
fn vmm_of(t: Tensor<'_>) -> Option<(&'static mut VmmBuffer, usize)> {
    // SAFETY: reading a live tensor's buffer; a buffer whose type is a registered VmmType has
    // a VmmBuffer as its context.
    unsafe {
        let raw = t.raw();
        let buffer = (*raw).buffer;
        if buffer.is_null() || !(*raw).view_src.is_null() {
            return None;
        }
        let buft = (*buffer).buft;
        let ours = VMM_TYPES
            .0
            .lock()
            .unwrap()
            .iter()
            .any(|ty| std::ptr::eq(&ty.buft, buft));
        if !ours {
            return None;
        }
        let b = vmm_buffer(buffer);
        let off = ((*raw).data as u64 - b.va) as usize;
        Some((b, off))
    }
}

/// The KV lend (#814): unmaps and frees the pieces of `--kv-vmm` tensor `t` that lie wholly
/// within bytes `from..to` of it, after the device finished everything queued. Returns the
/// bytes given back. Nothing may read those bytes until [`kv_vmm_remap`] backs them again.
pub fn kv_vmm_unmap(t: Tensor<'_>, from: usize, to: usize) -> Result<usize, String> {
    let (b, at) = vmm_of(t).ok_or("not a --kv-vmm tensor")?;
    let (from, to) = (at + from, at + to);
    use_device(b.device)?;
    // SAFETY: waits for every queued kernel (one of an earlier step may still read a piece),
    // then unmaps and frees pieces of this buffer only.
    unsafe {
        rt(cudaDeviceSynchronize(), "cudaDeviceSynchronize")?;
        let mut freed = 0;
        for m in b.maps.iter_mut() {
            let (off, n, h) = *m;
            if h == 0 || off < from || off + n > to {
                continue;
            }
            cu(cuMemUnmap(b.va + off as u64, n), "cuMemUnmap(KV pool)")?;
            cu(cuMemRelease(h), "cuMemRelease(KV pool)")?;
            m.2 = 0;
            freed += n;
        }
        Ok(freed)
    }
}

/// Backs every piece of `--kv-vmm` tensor `t` that [`kv_vmm_unmap`] freed with new, zeroed
/// VRAM, done before this returns. Returns the bytes mapped.
pub fn kv_vmm_remap(t: Tensor<'_>) -> Result<usize, String> {
    let (b, at) = vmm_of(t).ok_or("not a --kv-vmm tensor")?;
    let end = at + t.nbytes();
    use_device(b.device)?;
    let mut mapped = 0;
    for i in 0..b.maps.len() {
        let (off, n, h) = b.maps[i];
        if h != 0 || off < at || off >= end {
            continue;
        }
        b.back(i)?;
        mapped += n;
    }
    // SAFETY: the zeroing above is done before any step can use the blocks.
    rt(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize")?;
    Ok(mapped)
}

fn vmm_type<'a>(buft: ffi::ggml_backend_buffer_type_t) -> &'a VmmType {
    // SAFETY: the context is the registered VmmType.
    unsafe { &*((*buft).context as *const VmmType) }
}

fn vmm_buffer<'a>(buffer: ffi::ggml_backend_buffer_t) -> &'a mut VmmBuffer {
    // SAFETY: the context is the boxed VmmBuffer of this buffer.
    unsafe { &mut *((*buffer).context as *mut VmmBuffer) }
}

unsafe extern "C" fn vmm_name(buft: ffi::ggml_backend_buffer_type_t) -> *const c_char {
    vmm_type(buft).name.as_ptr()
}

unsafe extern "C" fn vmm_alignment(buft: ffi::ggml_backend_buffer_type_t) -> usize {
    vmm_type(buft).gran
}

/// Whole granules per tensor: no two tensors share one, so each maps its own.
unsafe extern "C" fn vmm_alloc_size(
    buft: ffi::ggml_backend_buffer_type_t,
    t: *const ffi::ggml_tensor,
) -> usize {
    // SAFETY: a live tensor.
    round_up(unsafe { ffi::ggml_nbytes(t) }, vmm_type(buft).gran)
}

unsafe extern "C" fn vmm_alloc(
    buft: ffi::ggml_backend_buffer_type_t,
    size: usize,
) -> ffi::ggml_backend_buffer_t {
    let ty = vmm_type(buft);
    let size = round_up(size.max(1), ty.gran);
    if let Err(e) = use_device(ty.device) {
        set_error(e);
        return null_mut();
    }
    let mut va: CUdeviceptr = 0;
    // SAFETY: reserves a range; nothing is mapped until the tensors are placed.
    if let Err(e) = cu(
        unsafe { cuMemAddressReserve(&mut va, size, ty.gran, 0, 0) },
        "cuMemAddressReserve(KV pool)",
    ) {
        set_error(e);
        return null_mut();
    }
    let b = Box::new(VmmBuffer {
        device: ty.device,
        va,
        size,
        maps: Vec::new(),
    });
    let iface = ffi::ggml_backend_buffer_i {
        free_buffer: Some(vmm_free),
        get_base: Some(vmm_base),
        init_tensor: Some(vmm_init_tensor),
        memset_tensor: Some(vmm_memset),
        set_tensor: Some(vmm_set),
        get_tensor: Some(vmm_get),
        set_tensor_2d: None,
        get_tensor_2d: None,
        cpy_tensor: None,
        clear: Some(vmm_clear),
        reset: None,
    };
    // SAFETY: ggml takes the interface and the boxed context (freed by vmm_free).
    unsafe { ffi::ggml_backend_buffer_init(buft, iface, Box::into_raw(b) as *mut c_void, size) }
}

unsafe extern "C" fn vmm_free(buffer: ffi::ggml_backend_buffer_t) {
    let b = vmm_buffer(buffer);
    // SAFETY: no kernel may still read the range; then frees the box handed to ggml.
    unsafe {
        cudaSetDevice(b.device as c_int);
        cudaDeviceSynchronize();
        b.release();
        drop(Box::from_raw((*buffer).context as *mut VmmBuffer));
    }
}

unsafe extern "C" fn vmm_base(buffer: ffi::ggml_backend_buffer_t) -> *mut c_void {
    vmm_buffer(buffer).va as *mut c_void
}

/// Maps a tensor's granules as ggml places it, in pieces of at most [`KV_VMM_CHUNK`] bytes.
/// Views share their source's memory.
unsafe extern "C" fn vmm_init_tensor(
    buffer: ffi::ggml_backend_buffer_t,
    t: *mut ffi::ggml_tensor,
) -> ffi::ggml_status {
    // SAFETY: a live tensor of this buffer.
    let (view, data, nbytes) = unsafe {
        (
            !(*t).view_src.is_null(),
            (*t).data as u64,
            ffi::ggml_nbytes(t),
        )
    };
    if view || nbytes == 0 {
        return ffi::ggml_status_GGML_STATUS_SUCCESS;
    }
    // SAFETY: the buffer's type is a VmmType.
    let ty = vmm_type(unsafe { (*buffer).buft });
    let b = vmm_buffer(buffer);
    let off = (data - b.va) as usize;
    let bytes = round_up(nbytes, ty.gran);
    let done = pieces(off, bytes, ty.gran, KV_VMM_CHUNK)
        .into_iter()
        .try_for_each(|(at, n)| b.map(at, n));
    match done {
        Ok(()) => ffi::ggml_status_GGML_STATUS_SUCCESS,
        Err(e) => {
            tracing::error!(device = ty.device, "KV pool (--kv-vmm): {e}");
            set_error(e);
            ffi::ggml_status_GGML_STATUS_ALLOC_FAILED
        }
    }
}

unsafe extern "C" fn vmm_memset(
    buffer: ffi::ggml_backend_buffer_t,
    t: *mut ffi::ggml_tensor,
    v: u8,
    off: usize,
    n: usize,
) {
    // SAFETY: a range of the tensor, mapped for the device.
    unsafe {
        cudaSetDevice(vmm_buffer(buffer).device as c_int);
        cudaMemset(
            ((*t).data as *mut u8).add(off) as *mut c_void,
            v as c_int,
            n,
        );
    }
}

unsafe extern "C" fn vmm_set(
    buffer: ffi::ggml_backend_buffer_t,
    t: *mut ffi::ggml_tensor,
    data: *const c_void,
    off: usize,
    n: usize,
) {
    // SAFETY: as above.
    unsafe {
        cudaSetDevice(vmm_buffer(buffer).device as c_int);
        cudaMemcpy(
            ((*t).data as *mut u8).add(off) as *mut c_void,
            data,
            n,
            CUDA_MEMCPY_DEFAULT,
        );
    }
}

unsafe extern "C" fn vmm_get(
    buffer: ffi::ggml_backend_buffer_t,
    t: *const ffi::ggml_tensor,
    data: *mut c_void,
    off: usize,
    n: usize,
) {
    // SAFETY: as above.
    unsafe {
        cudaSetDevice(vmm_buffer(buffer).device as c_int);
        cudaMemcpy(
            data,
            ((*t).data as *const u8).add(off) as *const c_void,
            n,
            CUDA_MEMCPY_DEFAULT,
        );
    }
}

unsafe extern "C" fn vmm_clear(buffer: ffi::ggml_backend_buffer_t, v: u8) {
    let b = vmm_buffer(buffer);
    // SAFETY: the mapped pieces only (the rest of the range has no memory behind it).
    unsafe {
        cudaSetDevice(b.device as c_int);
        for &(off, n, _) in b.maps.iter().filter(|m| m.2 != 0) {
            cudaMemset((b.va + off as u64) as *mut c_void, v as c_int, n);
        }
        cudaDeviceSynchronize();
    }
}

/// The buffer type of CUDA device `device`'s KV pools under `--kv-vmm`: a reserved virtual
/// range per buffer, each tensor on a granule of its own and backed by VRAM allocations of at
/// most [`KV_VMM_CHUNK`] bytes (see the module). Fails where the device has no virtual memory
/// management.
pub fn kv_vmm_type(device: usize) -> Result<BufferType, String> {
    let caps = caps(device).ok_or("cannot query the GPU")?;
    if !caps.vmm || caps.device_granularity == 0 {
        return Err(format!(
            "CUDA{device} has no virtual memory management (cuMemCreate)"
        ));
    }
    let mut list = VMM_TYPES.0.lock().unwrap();
    if let Some(t) = list.iter_mut().find(|t| t.device == device) {
        return Ok(BufferType(NonNull::from(&mut t.buft)));
    }
    let t: &'static mut VmmType = Box::leak(Box::new(VmmType {
        device,
        gran: caps.device_granularity.max(256),
        name: CString::new(format!("CUDA{device}_KvVmm")).unwrap(),
        buft: ffi::ggml_backend_buffer_type {
            iface: ffi::ggml_backend_buffer_type_i {
                get_name: Some(vmm_name),
                alloc_buffer: Some(vmm_alloc),
                alloc_buffer_n: None,
                get_alignment: Some(vmm_alignment),
                get_max_size: None,
                get_alloc_size: Some(vmm_alloc_size),
                get_alloc_size_n: None,
                is_host: None,
            },
            // SAFETY: the CUDA device of ggml's own buffer type for it.
            device: unsafe {
                ffi::ggml_backend_buft_get_device(ffi::ggml_backend_cuda_buffer_type(
                    device as c_int,
                ))
            },
            context: null_mut(),
        },
    }));
    t.buft.context = t as *mut VmmType as *mut c_void;
    let buft = BufferType(NonNull::from(&mut t.buft));
    list.push(t);
    Ok(buft)
}
