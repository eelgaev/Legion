//! Answers kept per CUDA device (#1201). A question about the GPU (its compute capability, its
//! SM count, whether an op's kernels run on it) asked once per process gives every device the
//! first one's answer, and one box can hold GPUs of different architectures. These slots keep
//! one answer per device ordinal, worked out the first time it is asked: a relaxed load on the
//! hot path, no driver call after the first.

use std::sync::atomic::{AtomicI32, Ordering};

/// Device ordinals with a slot; a higher one is asked every time (no box has that many).
pub const DEVICES: usize = 64;

/// A slot's value before its first answer (never an answer itself: see [`PerDevice::get`]).
const UNSET: i32 = i32::MIN;

/// One `i32` answer per device ordinal.
pub struct PerDevice([AtomicI32; DEVICES]);

impl PerDevice {
    pub const fn new() -> Self {
        Self([const { AtomicI32::new(UNSET) }; DEVICES])
    }

    /// Device `dev`'s answer: kept from an earlier call, else `probe(dev)` (kept from now on).
    /// A probe answering `i32::MIN` is asked again next time. Two threads asking a device for
    /// the first time may both probe it; a probe gives the same answer each time.
    pub fn get(&self, dev: usize, probe: impl FnOnce(usize) -> i32) -> i32 {
        let Some(slot) = self.0.get(dev) else {
            return probe(dev);
        };
        let v = slot.load(Ordering::Relaxed);
        if v != UNSET {
            return v;
        }
        let v = probe(dev);
        slot.store(v, Ordering::Relaxed);
        v
    }
}

impl Default for PerDevice {
    fn default() -> Self {
        Self::new()
    }
}

/// CUDA device `dev`'s compute capability as `10 * major + minor` (121 on a GB10), asked once;
/// `None` when it cannot be asked (no such device, or a build without CUDA).
pub fn compute_capability(dev: usize) -> Option<i32> {
    static CC: PerDevice = PerDevice::new();
    let cc = CC.get(dev, cuda::compute_capability);
    (cc > 0).then_some(cc)
}

/// CUDA device `dev`'s compute capability as `(major, minor)` ((12, 1) on a GB10), asked once;
/// `None` when it cannot be asked.
pub fn cc(dev: usize) -> Option<(i32, i32)> {
    compute_capability(dev).map(|c| (c / 10, c % 10))
}

/// What a switch chosen by compute capability answers for an op on CUDA device `dev` (the
/// device of the weight it multiplies, or of the state it updates): `rule` of that device's
/// compute capability (asked once per device). Where the op's tensors name no CUDA device
/// (`None`: not placed yet, or host memory), `engine`: the answer for the GPU the engine loaded
/// on. On a box of one kind of GPU the two are the same answer.
pub fn pick<T>(dev: Option<usize>, engine: T, rule: impl FnOnce(Option<(i32, i32)>) -> T) -> T {
    pick_by(dev, engine, rule, cc)
}

/// [`pick`] with the compute capabilities `cc_of` gives (the tests' fake boxes).
pub fn pick_by<T>(
    dev: Option<usize>,
    engine: T,
    rule: impl FnOnce(Option<(i32, i32)>) -> T,
    cc_of: impl FnOnce(usize) -> Option<(i32, i32)>,
) -> T {
    match dev {
        Some(dev) => rule(cc_of(dev)),
        None => engine,
    }
}

/// The tests' fake mixed box for [`pick_by`]: a GB10 (device 0), an RTX 4090 (1), a V100 (2),
/// an A100 (3), an RTX 5090 (4); none from device 5 on.
#[cfg(test)]
pub fn test_cc(dev: usize) -> Option<(i32, i32)> {
    [(12, 1), (8, 9), (7, 0), (8, 0), (12, 0)].get(dev).copied()
}

/// The CUDA device current on the calling thread (a launcher's: the stream's device), `None`
/// when it cannot be asked.
pub fn current() -> Option<usize> {
    cuda::current()
}

/// CUDA device `dev`'s streaming multiprocessors, asked once; 0 when unknown.
pub fn sm_count(dev: usize) -> i32 {
    static SMS: PerDevice = PerDevice::new();
    SMS.get(dev, cuda::sm_count)
}

/// Device `dev`'s SM count when its compute capability is 12.x (GB10, RTX 50), else 0: the GPUs
/// the ops tuned on a GB10 are chosen for.
pub fn sms_if_cc12(dev: usize) -> i32 {
    sms_if_cc12_on(Some(dev), 0)
}

/// [`sms_if_cc12`] for an op on device `dev` (the weight's); `engine` (the engine's GPU's
/// answer) where the op's tensors name no CUDA device, as [`pick`].
pub fn sms_if_cc12_on(dev: Option<usize>, engine: i32) -> i32 {
    sms_if_cc12_by(dev, engine, cc, sm_count)
}

/// [`sms_if_cc12_on`] with the compute capabilities `cc_of` and SM counts `sms_of` give (the
/// tests' fake boxes).
pub fn sms_if_cc12_by(
    dev: Option<usize>,
    engine: i32,
    cc_of: impl FnOnce(usize) -> Option<(i32, i32)>,
    sms_of: impl FnOnce(usize) -> i32,
) -> i32 {
    match dev {
        Some(dev) => match cc_of(dev) {
            Some((12, _)) => sms_of(dev).max(0),
            _ => 0,
        },
        None => engine,
    }
}

/// Whether compute capability `(major, minor)` is Ampere: 8.0 to 8.7 (A100 and A30, RTX 30, the
/// A-series cards, Orin), not 8.9 (Ada). The CUDA launchers ask the same of the current device
/// (`legion_device::ampere_cc`, #1212).
pub fn ampere((major, minor): (i32, i32)) -> bool {
    major == 8 && (0..=7).contains(&minor)
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::ffi::c_int;

    unsafe extern "C" {
        fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
        fn cudaGetDevice(device: *mut c_int) -> c_int;
    }

    pub(super) fn current() -> Option<usize> {
        let mut dev = 0;
        // SAFETY: a plain query of the calling thread's device.
        if unsafe { cudaGetDevice(&mut dev) } != 0 {
            return None;
        }
        usize::try_from(dev).ok()
    }

    const DEV_ATTR_SM_COUNT: c_int = 16;
    const DEV_ATTR_CC_MAJOR: c_int = 75;
    const DEV_ATTR_CC_MINOR: c_int = 76;

    fn attribute(dev: usize, attr: c_int) -> Option<i32> {
        let dev = c_int::try_from(dev).ok()?;
        let mut v = 0;
        // SAFETY: a plain query of device `dev` (an error for a device that does not exist).
        (unsafe { cudaDeviceGetAttribute(&mut v, attr, dev) } == 0).then_some(v)
    }

    pub(super) fn compute_capability(dev: usize) -> i32 {
        match (
            attribute(dev, DEV_ATTR_CC_MAJOR),
            attribute(dev, DEV_ATTR_CC_MINOR),
        ) {
            (Some(major), Some(minor)) => 10 * major + minor,
            _ => 0,
        }
    }

    pub(super) fn sm_count(dev: usize) -> i32 {
        attribute(dev, DEV_ATTR_SM_COUNT).unwrap_or(0).max(0)
    }
}

#[cfg(not(feature = "cuda"))]
mod cuda {
    pub(super) fn current() -> Option<usize> {
        None
    }

    pub(super) fn compute_capability(_: usize) -> i32 {
        0
    }

    pub(super) fn sm_count(_: usize) -> i32 {
        0
    }
}

#[cfg(test)]
mod tests;
