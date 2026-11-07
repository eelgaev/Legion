//! Whether this binary has GPU code of a device's architecture (#1201).
//!
//! Legion's kernels for newer GPUs are built only where `__CUDA_ARCH__` is high enough and trap
//! elsewhere (src/cuda/README.md), and the host picks them by the device's compute capability.
//! A binary built for older architectures only still runs on a newer GPU (the driver compiles
//! the newest PTX it holds), but there every kernel is the older architecture's code: the paths
//! picked for the GPU's compute capability would trap, or run their older fallbacks where they
//! must not (programmatic dependent launch without its wait). So the engine asks once per
//! device, at load, which architecture the code it runs was compiled for, and refuses a GPU it
//! has no code of its architecture's level for, with how to build for it.

/// The architecture levels the kernels are built at (`__CUDA_ARCH__ >= 800`, `>= 890`,
/// `>= 900`, as `10 * major + minor`), 70 the oldest Legion runs on.
const LEVELS: [i32; 4] = [70, 80, 89, 90];

/// The newest architecture level at or below `cc` (`10 * major + minor`); `None` below 7.0.
pub fn level(cc: i32) -> Option<i32> {
    LEVELS.iter().rev().copied().find(|&l| l <= cc)
}

/// Why device `dev` (`name`, compute capability `cc`) cannot run on this binary, whose code on
/// it was compiled for architecture `image` (`10 * major + minor`; `None` when it has no code
/// the device runs); `None` when it can.
pub fn missing_code(dev: usize, name: &str, cc: i32, image: Option<i32>) -> Option<String> {
    let need = level(cc)?;
    let has = image.and_then(level);
    if has.is_some_and(|h| h >= need) {
        return None;
    }
    let (major, minor) = (cc / 10, cc % 10);
    let built = match image {
        Some(i) => format!("for compute capability {}.{} and older", i / 10, i % 10),
        None => "without code it runs".to_string(),
    };
    // Compute capability 9.0+ parts take their architecture-specific target (90a, 120a, 121a).
    let target = if major >= 9 {
        format!("{major}{minor}a")
    } else {
        format!("{major}{minor}")
    };
    Some(format!(
        "GPU {dev} ({name}, compute capability {major}.{minor}): this binary was built \
         {built}; rebuild with CMAKE_CUDA_ARCHITECTURES including {target} (or native on this \
         machine)"
    ))
}

/// What [`image`] found on a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Image {
    /// The architecture (`10 * major + minor`) the code the device runs was compiled for.
    Arch(i32),
    /// The binary has no code the device runs.
    None,
    /// The question failed (or a build without CUDA): nothing to refuse on.
    Unknown,
}

/// The GPU code device `dev` runs, asked once per device: the PTX version of one of Legion's
/// kernels there (every file of src/cuda is built for the same architectures).
pub fn image(dev: usize) -> Image {
    static IMAGE: super::per_device::PerDevice = super::per_device::PerDevice::new();
    match IMAGE.get(dev, cuda::image) {
        v if v > 0 => Image::Arch(v),
        0 => Image::None,
        _ => Image::Unknown,
    }
}

/// The error [`missing_code`] gives for CUDA device `dev` (its name from `name`, asked only
/// then), from its compute capability and [`image`]; `None` when it runs, or when either
/// question fails.
pub fn check(dev: usize, name: impl FnOnce() -> String) -> Option<String> {
    let cc = super::per_device::compute_capability(dev)?;
    let image = match image(dev) {
        Image::Arch(a) => Some(a),
        Image::None => None,
        Image::Unknown => return None,
    };
    let need = level(cc)?;
    if image.and_then(level).is_some_and(|h| h >= need) {
        return None;
    }
    missing_code(dev, &name(), cc, image)
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::ffi::{c_int, c_void};

    /// `cudaFuncAttributes`, its fields up to `binaryVersion`, padded past the struct's size.
    #[repr(C)]
    struct FuncAttributes {
        shared: usize,
        constant: usize,
        local: usize,
        max_threads: c_int,
        regs: c_int,
        ptx_version: c_int,
        binary_version: c_int,
        rest: [u8; 256],
    }

    unsafe extern "C" {
        /// q8_group.cu's kernels (their host stubs), built for every architecture.
        static legion_q8_group_kernels: [*const c_void; 33];
        fn cudaGetDevice(device: *mut c_int) -> c_int;
        fn cudaSetDevice(device: c_int) -> c_int;
        fn cudaGetLastError() -> c_int;
        fn cudaFuncGetAttributes(attr: *mut FuncAttributes, func: *const c_void) -> c_int;
    }

    /// `cudaErrorInvalidDeviceFunction`, `cudaErrorNoKernelImageForDevice`: no code for the device.
    const NO_CODE: [c_int; 2] = [98, 209];

    /// The PTX version (the `__CUDA_ARCH__` its code was compiled with, divided by 10) of the
    /// quantization kernel on device `dev`; 0 when the device has no code for it, -1 when the
    /// question fails otherwise.
    pub(super) fn image(dev: usize) -> i32 {
        let Ok(dev) = c_int::try_from(dev) else {
            return -1;
        };
        // SAFETY: plain queries; the current device is put back before returning.
        unsafe {
            let mut was = 0;
            if cudaGetDevice(&mut was) != 0 {
                cudaGetLastError();
                return -1;
            }
            if was != dev && cudaSetDevice(dev) != 0 {
                cudaGetLastError();
                return -1;
            }
            let mut a: FuncAttributes = std::mem::zeroed();
            let err = cudaFuncGetAttributes(&mut a, legion_q8_group_kernels[0]);
            if err != 0 {
                cudaGetLastError();
            }
            if was != dev {
                cudaSetDevice(was);
            }
            if err != 0 {
                return if NO_CODE.contains(&err) { 0 } else { -1 };
            }
            // ptxVersion is the virtual architecture the code was compiled for (on a GPU newer
            // than every image, the driver's JIT output reports the GPU's own binaryVersion).
            if a.ptx_version > 0 {
                a.ptx_version
            } else if a.binary_version > 0 {
                a.binary_version
            } else {
                -1
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
mod cuda {
    pub(super) fn image(_: usize) -> i32 {
        -1
    }
}

#[cfg(test)]
mod tests;
