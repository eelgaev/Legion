//! Tensor parallelism through ggml's meta backend (`ggml-backend-meta.cpp`): one device that
//! stands for several, whose buffers hold every tensor split along an axis (or mirrored) across
//! them, and whose backend runs each op on every device's slice, inserting the all-reduces where
//! a partial sum meets an op that needs the whole (on CUDA through `ggml-cuda/allreduce.cu`).
//! llama.cpp's `-sm tensor` is built on it.
//!
//! Which axis a weight or cache tensor is split along is the caller's choice ([`SplitRule`],
//! by tensor name and shape); tensors made by ops inherit a compatible split from their
//! sources. Every device gets an equal share of the axis, in whole multiples of the rule's
//! granularity (heads, GQA groups, quantization blocks).

use std::os::raw::c_void;
use std::ptr::NonNull;

use super::backend::Device;
use super::ffi;

/// How one tensor is laid out over the devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    /// A full copy on every device.
    Mirrored,
    /// Each device holds a contiguous slice of dimension `axis`, a multiple of `granularity`
    /// elements long (the last device takes the rest).
    Axis { axis: usize, granularity: i64 },
}

/// Chooses the [`Split`] of a statically allocated tensor from its name, its shape and type.
pub type SplitRule = dyn Fn(&str, [i64; 4], ffi::ggml_type) -> Split + Send + Sync;

struct Rule {
    rule: Box<SplitRule>,
    n_devices: usize,
}

/// The per-device lengths of a dimension of `ne` elements split `n` ways in multiples of `g`.
pub fn shares(ne: i64, n: usize, g: i64) -> Vec<i64> {
    let g = g.max(1);
    let mut out = Vec::with_capacity(n);
    let mut low = 0;
    for j in 0..n {
        let high = if j + 1 == n {
            ne
        } else {
            let h = ne * (j as i64 + 1) / n as i64;
            h - h % g
        };
        out.push(high - low);
        low = high;
    }
    out
}

unsafe extern "C" fn split_state(
    tensor: *const ffi::ggml_tensor,
    ud: *mut c_void,
) -> ffi::ggml_backend_meta_split_state {
    // SAFETY: `ud` is the leaked `Rule` of `meta_device`; ggml passes a valid tensor.
    let (rule, t) = unsafe { (&*(ud as *const Rule), &*tensor) };
    // SAFETY: ggml keeps tensor names NUL-terminated within their fixed array.
    let name = unsafe { std::ffi::CStr::from_ptr(t.name.as_ptr()) }.to_string_lossy();
    // SAFETY: plain old data; every field is set below.
    let mut s: ffi::ggml_backend_meta_split_state = unsafe { std::mem::zeroed() };
    s.n_segments = 1;
    s.nr[0] = 1;
    match (rule.rule)(&name, t.ne, t.type_) {
        Split::Mirrored => {
            s.axis = ffi::ggml_backend_meta_split_axis_GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        }
        Split::Axis { axis, granularity } => {
            s.axis = axis as ffi::ggml_backend_meta_split_axis;
            // A quantized row splits only between whole blocks.
            let g = if axis == 0 {
                // SAFETY: a pure lookup in ggml's type table; `t.type_` is a valid type.
                let blck = unsafe { ffi::ggml_blck_size(t.type_) };
                lcm(granularity.max(1), blck)
            } else {
                granularity
            };
            for (j, ne) in shares(t.ne[axis], rule.n_devices, g)
                .into_iter()
                .enumerate()
            {
                s.ne[j] = ne;
            }
        }
    }
    if std::env::var_os("LEGION_META_DEBUG").is_some() {
        eprintln!(
            "meta split {name} {:?} -> axis {} {:?}",
            t.ne,
            s.axis,
            &s.ne[..rule.n_devices]
        );
    }
    s
}

static REINITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REINIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Counts one re-initialization of a kept graph ([`crate::ggml::KeptGraph::reinit`]).
pub(crate) fn count_reinit(took: std::time::Duration) {
    use std::sync::atomic::Ordering::Relaxed;
    REINITS.fetch_add(1, Relaxed);
    REINIT_NS.fetch_add(took.as_nanos() as u64, Relaxed);
}

/// Kept-graph re-initializations on meta backends so far, and the host time they took (for
/// profiles: each is a host-side cost before the graph launches).
pub fn reinit_stats() -> (u64, std::time::Duration) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        REINITS.load(Relaxed),
        std::time::Duration::from_nanos(REINIT_NS.load(Relaxed)),
    )
}

fn lcm(a: i64, b: i64) -> i64 {
    fn gcd(a: i64, b: i64) -> i64 {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    a / gcd(a, b) * b
}

/// A meta device over `devices` (in order), splitting statically allocated tensors by `rule`.
/// Its buffer type allocates split tensors, its backend computes over all of them. The rule
/// lives as long as the process (meta devices are never freed).
pub fn meta_device(devices: &[Device], rule: Box<SplitRule>) -> Option<Device> {
    let ud = Box::into_raw(Box::new(Rule {
        rule,
        n_devices: devices.len(),
    }));
    let mut raws: Vec<ffi::ggml_backend_dev_t> = devices.iter().map(|d| d.raw()).collect();
    // ggml makes meta devices without locking (a TODO in ggml-backend-meta.cpp): one at a
    // time.
    static MAKING: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _one = MAKING.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: valid devices; the callback and its userdata live for the process; calls are
    // serialized by `MAKING`.
    let dev = unsafe {
        ffi::ggml_backend_meta_device(
            raws.as_mut_ptr(),
            raws.len(),
            Some(split_state),
            ud as *mut c_void,
        )
    };
    let dev = NonNull::new(dev)?;
    DEVICES
        .lock()
        .unwrap()
        .push((dev.as_ptr() as usize, ud as usize));
    // SAFETY: a device ggml made and never frees.
    Some(unsafe { Device::from_raw(dev) })
}

/// The meta devices made so far and their rules (leaked, never freed): device, `Rule`.
static DEVICES: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());

/// How an allocated tensor of a meta buffer is laid out: the devices it is over and its
/// [`Split`] by the meta device's rule (`None` outside a meta buffer). For code that has to
/// know each device's slice of a statically allocated tensor (offsets of views within it).
pub(crate) fn tensor_split(t: super::Tensor<'_>) -> Option<(usize, Split)> {
    if !t.is_meta() {
        return None;
    }
    // SAFETY: an allocated tensor's buffer and its type are live; reads only.
    let dev = unsafe {
        let buft = ffi::ggml_backend_buffer_get_type((*t.raw()).buffer);
        ffi::ggml_backend_buft_get_device(buft) as usize
    };
    let ud = DEVICES.lock().unwrap().iter().find(|d| d.0 == dev)?.1;
    // SAFETY: the leaked `Rule` of `meta_device`, alive for the process.
    let rule = unsafe { &*(ud as *const Rule) };
    Some((rule.n_devices, (rule.rule)(&t.name(), t.ne(), t.ty())))
}

#[cfg(test)]
mod tests;
