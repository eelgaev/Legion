//! Step compute buffers in unified memory: VRAM as far as it is free, the rest pinned host
//! memory on the GPU's NUMA node, mapped back to back in one virtual range the device reads
//! through the same pointer (the host part over the link). A step whose compute buffer no
//! longer fits beside the expert cache then runs, a little slower, instead of failing.
//!
//! The same memory holds a decode state that does not fit beside the weights (#1131): the
//! sliding-window rings of many sequences, with more VRAM kept free for the steps and the KV
//! cache ([`spill_type_keeping`]); under tensor parallelism through wrappers of the CUDA
//! devices whose buffer type is this one ([`spill_devices`]), for a meta device over them.

use super::*;
use crate::ggml::Device;

/// VRAM a spill buffer leaves free: what the backend allocates while graphs compute (cuBLAS
/// workspaces, ggml-cuda's pool), as the lent expert cache's steps keep free (`LENT_SCRATCH`).
const SPILL_KEEPS_FREE: usize = 512 << 20;

struct SpillType {
    device: usize,
    node: u32,
    gran: usize,
    /// VRAM its buffers leave free.
    keep: usize,
    name: CString,
    buft: ffi::ggml_backend_buffer_type,
    /// The CUDA device with this buffer type in its place ([`spill_devices`]).
    dev: ffi::ggml_backend_device,
}

static SPILLS: Registry<SpillType> = Registry(Mutex::new(Vec::new()));

struct SpillBuffer {
    device: usize,
    va: CUdeviceptr,
    size: usize,
    vram: CUmemGenericAllocationHandle,
    vram_bytes: usize,
    host: CUmemGenericAllocationHandle,
    host_bytes: usize,
    mapped: (bool, bool),
}

impl SpillBuffer {
    /// Unmaps and releases what was mapped and created, then the range.
    fn release(&self) {
        // SAFETY: only what this buffer created, mapped and reserved.
        unsafe {
            if self.mapped.0 {
                cuMemUnmap(self.va, self.vram_bytes);
            }
            if self.mapped.1 {
                cuMemUnmap(self.va + self.vram_bytes as u64, self.host_bytes);
            }
            if self.vram != 0 {
                cuMemRelease(self.vram);
            }
            if self.host != 0 {
                cuMemRelease(self.host);
            }
            if self.va != 0 {
                cuMemAddressFree(self.va, self.size);
            }
        }
    }
}

/// VRAM bytes of a `total`-byte spill buffer with `free` VRAM free: as much as leaves `keep`
/// free, in whole granules.
fn vram_part(total: usize, free: usize, keep: usize, gran: usize) -> usize {
    total.min(free.saturating_sub(keep) / gran * gran)
}

fn spill_ctx<'a>(buffer: ffi::ggml_backend_buffer_t) -> &'a SpillBuffer {
    // SAFETY: the context is the boxed SpillBuffer of this buffer.
    unsafe { &*((*buffer).context as *const SpillBuffer) }
}

unsafe extern "C" fn spill_name(buft: ffi::ggml_backend_buffer_type_t) -> *const c_char {
    // SAFETY: the context is the registered SpillType.
    unsafe { (*((*buft).context as *const SpillType)).name.as_ptr() }
}

/// The CUDA device's own buffer type, for the alignment and the allocation sizes.
fn cuda_buft(device: usize) -> ffi::ggml_backend_buffer_type_t {
    // SAFETY: ggml's static buffer type of a device index.
    unsafe { ffi::ggml_backend_cuda_buffer_type(device as c_int) }
}

unsafe extern "C" fn spill_alignment(buft: ffi::ggml_backend_buffer_type_t) -> usize {
    // SAFETY: the context is the registered SpillType.
    unsafe {
        let t = &*((*buft).context as *const SpillType);
        ffi::ggml_backend_buft_get_alignment(cuda_buft(t.device))
    }
}

unsafe extern "C" fn spill_alloc_size(
    buft: ffi::ggml_backend_buffer_type_t,
    tensor: *const ffi::ggml_tensor,
) -> usize {
    // SAFETY: as above; a live tensor.
    unsafe {
        let t = &*((*buft).context as *const SpillType);
        ffi::ggml_backend_buft_get_alloc_size(cuda_buft(t.device), tensor)
    }
}

unsafe extern "C" fn spill_alloc(
    buft: ffi::ggml_backend_buffer_type_t,
    size: usize,
) -> ffi::ggml_backend_buffer_t {
    // SAFETY: the context is the registered SpillType.
    let t = unsafe { &*((*buft).context as *const SpillType) };
    match spill_map(t, size) {
        Ok(b) => spill_wrap(buft, b),
        Err(e) => {
            set_error(e);
            null_mut()
        }
    }
}

fn spill_map(t: &SpillType, size: usize) -> Result<Box<SpillBuffer>, String> {
    let free = free_vram(t.device)?;
    let total = round_up(size.max(1), t.gran);
    let vram_bytes = vram_part(total, free, t.keep, t.gran);
    let mut b = Box::new(SpillBuffer {
        device: t.device,
        va: 0,
        size: total,
        vram: 0,
        vram_bytes,
        host: 0,
        host_bytes: total - vram_bytes,
        mapped: (false, false),
    });
    // SAFETY: reserves, creates and maps within the new range; on failure releases it.
    let r = unsafe {
        (|| -> Result<(), String> {
            cu(
                cuMemAddressReserve(&mut b.va, total, t.gran, 0, 0),
                "cuMemAddressReserve",
            )?;
            if b.vram_bytes > 0 {
                let prop = alloc_prop(CU_MEM_LOCATION_TYPE_DEVICE, t.device as i32);
                cu(
                    cuMemCreate(&mut b.vram, b.vram_bytes, &prop, 0),
                    "cuMemCreate(spill VRAM)",
                )?;
                cu(
                    cuMemMap(b.va, b.vram_bytes, 0, b.vram, 0),
                    "cuMemMap(spill VRAM)",
                )?;
                b.mapped.0 = true;
            }
            if b.host_bytes > 0 {
                let prop = host_prop(t.node);
                cu(
                    cuMemCreate(&mut b.host, b.host_bytes, &prop, 0),
                    "cuMemCreate(spill host NUMA)",
                )?;
                cu(
                    cuMemMap(b.va + b.vram_bytes as u64, b.host_bytes, 0, b.host, 0),
                    "cuMemMap(spill host)",
                )?;
                b.mapped.1 = true;
            }
            let a = access(CU_MEM_LOCATION_TYPE_DEVICE, t.device as i32);
            cu(cuMemSetAccess(b.va, total, &a, 1), "cuMemSetAccess(spill)")
        })()
    };
    if let Err(e) = r {
        b.release();
        return Err(e);
    }
    if b.host_bytes > 0 && t.keep == SPILL_KEEPS_FREE {
        tracing::info!(
            device = t.device,
            node = t.node,
            vram_mib = b.vram_bytes >> 20,
            host_mib = b.host_bytes >> 20,
            "step compute buffer spills into host memory"
        );
    } else if b.host_bytes > 0 {
        // Reallocated with every cache size the load tries: the load's warning says it once.
        tracing::debug!(
            device = t.device,
            node = t.node,
            vram_mib = b.vram_bytes >> 20,
            host_mib = b.host_bytes >> 20,
            "decode state spills into host memory"
        );
    }
    Ok(b)
}

fn spill_wrap(
    buft: ffi::ggml_backend_buffer_type_t,
    b: Box<SpillBuffer>,
) -> ffi::ggml_backend_buffer_t {
    let iface = ffi::ggml_backend_buffer_i {
        free_buffer: Some(spill_free),
        get_base: Some(spill_base),
        init_tensor: None,
        memset_tensor: Some(spill_memset),
        set_tensor: Some(spill_set),
        get_tensor: Some(spill_get),
        set_tensor_2d: Some(spill_set_2d),
        get_tensor_2d: Some(spill_get_2d),
        cpy_tensor: None,
        clear: Some(spill_clear),
        reset: None,
    };
    let size = b.size;
    // SAFETY: ggml takes the interface and the boxed context (freed by spill_free).
    unsafe { ffi::ggml_backend_buffer_init(buft, iface, Box::into_raw(b) as *mut c_void, size) }
}

unsafe extern "C" fn spill_free(buffer: ffi::ggml_backend_buffer_t) {
    let b = spill_ctx(buffer);
    // SAFETY: no kernel may still use the range.
    timed("cudaDeviceSynchronize", b.device, b.size, || unsafe {
        cudaSetDevice(b.device as c_int);
        cudaDeviceSynchronize();
    });
    timed("release range", b.device, b.size, || b.release());
    // SAFETY: frees the box handed to ggml in `spill_wrap`.
    drop(unsafe { Box::from_raw((*buffer).context as *mut SpillBuffer) });
}

unsafe extern "C" fn spill_base(buffer: ffi::ggml_backend_buffer_t) -> *mut c_void {
    spill_ctx(buffer).va as *mut c_void
}

unsafe extern "C" fn spill_memset(
    buffer: ffi::ggml_backend_buffer_t,
    t: *mut ffi::ggml_tensor,
    v: u8,
    off: usize,
    n: usize,
) {
    // SAFETY: a range of the tensor, addressable by the device.
    unsafe {
        cudaSetDevice(spill_ctx(buffer).device as c_int);
        cudaMemset(
            ((*t).data as *mut u8).add(off) as *mut c_void,
            v as c_int,
            n,
        );
    }
}

unsafe extern "C" fn spill_set(
    buffer: ffi::ggml_backend_buffer_t,
    t: *mut ffi::ggml_tensor,
    data: *const c_void,
    off: usize,
    n: usize,
) {
    // SAFETY: as above.
    unsafe {
        cudaSetDevice(spill_ctx(buffer).device as c_int);
        cudaMemcpy(
            ((*t).data as *mut u8).add(off) as *mut c_void,
            data,
            n,
            CUDA_MEMCPY_DEFAULT,
        );
    }
}

unsafe extern "C" fn spill_get(
    buffer: ffi::ggml_backend_buffer_t,
    t: *const ffi::ggml_tensor,
    data: *mut c_void,
    off: usize,
    n: usize,
) {
    // SAFETY: as above.
    unsafe {
        cudaSetDevice(spill_ctx(buffer).device as c_int);
        cudaMemcpy(
            data,
            ((*t).data as *const u8).add(off) as *const c_void,
            n,
            CUDA_MEMCPY_DEFAULT,
        );
    }
}

/// `n_copies` rows of `size` bytes, `stride_tensor` apart in the tensor and `stride_data` in
/// `data`, in one 2D copy: ggml's fallback copies each row on its own, and a tensor-parallel
/// ring's save copies each GPU's share of every cell so.
unsafe extern "C" fn spill_set_2d(
    buffer: ffi::ggml_backend_buffer_t,
    t: *mut ffi::ggml_tensor,
    data: *const c_void,
    off: usize,
    size: usize,
    n_copies: usize,
    stride_tensor: usize,
    stride_data: usize,
) {
    // SAFETY: rows of the tensor, addressable by the device (ggml checked the range).
    unsafe {
        cudaSetDevice(spill_ctx(buffer).device as c_int);
        cudaMemcpy2D(
            ((*t).data as *mut u8).add(off) as *mut c_void,
            stride_tensor,
            data,
            stride_data,
            size,
            n_copies,
            CUDA_MEMCPY_DEFAULT,
        );
    }
}

/// The reverse of [`spill_set_2d`].
unsafe extern "C" fn spill_get_2d(
    buffer: ffi::ggml_backend_buffer_t,
    t: *const ffi::ggml_tensor,
    data: *mut c_void,
    off: usize,
    size: usize,
    n_copies: usize,
    stride_tensor: usize,
    stride_data: usize,
) {
    // SAFETY: as above.
    unsafe {
        cudaSetDevice(spill_ctx(buffer).device as c_int);
        cudaMemcpy2D(
            data,
            stride_data,
            ((*t).data as *const u8).add(off) as *const c_void,
            stride_tensor,
            size,
            n_copies,
            CUDA_MEMCPY_DEFAULT,
        );
    }
}

unsafe extern "C" fn spill_clear(buffer: ffi::ggml_backend_buffer_t, v: u8) {
    let b = spill_ctx(buffer);
    // SAFETY: the whole mapped range.
    unsafe {
        cudaSetDevice(b.device as c_int);
        cudaMemset(b.va as *mut c_void, v as c_int, b.size);
        cudaDeviceSynchronize();
    }
}

/// A buffer type for `device`'s step compute buffers that spills into pinned host memory on
/// the device's NUMA node (host memory of no node where it has none: [`alloc_node`]) when VRAM
/// runs short. `None` when this driver cannot map pinned host memory through VMM for it.
pub fn spill_type(device: usize) -> Option<BufferType> {
    spill_type_keeping(device, SPILL_KEEPS_FREE)
}

/// [`spill_type`] whose buffers leave `keep` bytes of VRAM free (at least the steps'
/// [`SPILL_KEEPS_FREE`]): for a decode state that does not fit beside the weights, with the
/// room the steps and the KV cache need kept for them.
pub fn spill_type_keeping(device: usize, keep: usize) -> Option<BufferType> {
    registered(device, keep.max(SPILL_KEEPS_FREE)).map(|t| BufferType(NonNull::from(&mut t.buft)))
}

/// The CUDA devices `devices` (ggml's indices) as devices whose buffer type is
/// [`spill_type_keeping`]`(device, keep)`, for a meta device
/// ([`crate::ggml::meta::meta_device`]) over them: its buffer type allocates each GPU's slice
/// of a tensor-parallel decode state in VRAM as far as `keep` stays free, the rest in pinned
/// memory on the GPU's host node. Fails where the driver cannot map host NUMA memory (VMM).
pub fn spill_devices(devices: &[usize], keep: usize) -> Result<Vec<Device>, String> {
    devices
        .iter()
        .map(|&device| {
            let t = registered(device, keep.max(SPILL_KEEPS_FREE)).ok_or_else(|| {
                format!("CUDA{device}'s driver cannot map host NUMA memory (cuMemCreate)")
            })?;
            // SAFETY: a registered device, never freed.
            Ok(unsafe { Device::from_raw(NonNull::from(&mut t.dev)) })
        })
        .collect()
}

unsafe extern "C" fn spill_dev_buffer_type(
    dev: ffi::ggml_backend_dev_t,
) -> ffi::ggml_backend_buffer_type_t {
    let mut list = SPILLS.0.lock().unwrap();
    match list.iter_mut().find(|t| std::ptr::eq(&t.dev, dev)) {
        Some(t) => &mut t.buft,
        None => null_mut(),
    }
}

/// The registered spill type of `device` keeping `keep` free, made on first use.
fn registered(device: usize, keep: usize) -> Option<&'static mut SpillType> {
    let mut list = SPILLS.0.lock().unwrap();
    if let Some(t) = list
        .iter_mut()
        .find(|t| t.device == device && t.keep == keep)
    {
        // SAFETY: registered for the process, never freed; ggml reaches it through raw
        // pointers as before.
        return Some(unsafe { &mut *(&mut **t as *mut SpillType) });
    }
    // A GPU with no host NUMA node spills to host memory of no node (`alloc_node`).
    let node = alloc_node(device)?;
    use_device(device).ok()?;
    let (dg, hg) = (
        granularity(CU_MEM_LOCATION_TYPE_DEVICE, device as i32),
        host_granularity(node),
    );
    if dg == 0 || hg == 0 || (dg.max(hg) % dg.min(hg)) != 0 {
        return None;
    }
    // SAFETY: ggml's CUDA device for this index; its interface, registry and context are
    // static.
    let cuda = unsafe { *ffi::ggml_backend_buft_get_device(cuda_buft(device)) };
    let t: &'static mut SpillType = Box::leak(Box::new(SpillType {
        device,
        node,
        gran: dg.max(hg),
        keep,
        name: CString::new(if keep == SPILL_KEEPS_FREE {
            format!("CUDA{device}_Spill")
        } else {
            format!("CUDA{device}_SpillKeep{}M", keep >> 20)
        })
        .unwrap(),
        buft: ffi::ggml_backend_buffer_type {
            iface: ffi::ggml_backend_buffer_type_i {
                get_name: Some(spill_name),
                alloc_buffer: Some(spill_alloc),
                alloc_buffer_n: None,
                get_alignment: Some(spill_alignment),
                get_max_size: None,
                get_alloc_size: Some(spill_alloc_size),
                get_alloc_size_n: None,
                is_host: None,
            },
            // SAFETY: the CUDA device of ggml's own buffer type for it.
            device: unsafe { ffi::ggml_backend_buft_get_device(cuda_buft(device)) },
            context: null_mut(),
        },
        dev: ffi::ggml_backend_device {
            iface: ffi::ggml_backend_device_i {
                get_buffer_type: Some(spill_dev_buffer_type),
                ..cuda.iface
            },
            ..cuda
        },
    }));
    t.buft.context = t as *mut SpillType as *mut c_void;
    let p: *mut SpillType = &mut *t;
    list.push(t);
    // SAFETY: as above.
    Some(unsafe { &mut *p })
}

#[cfg(test)]
mod tests;
