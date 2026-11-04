//! Hand-written PTX kernels, loaded at run time through the CUDA driver API (#703).
//!
//! The `.cu` kernels are built by nvcc into `liblegion_cuda.a` (build.rs). A [`Module`] is
//! instead PTX text embedded in the binary (`include_str!` of a file under `src/cuda/ptx/`),
//! JIT-compiled by the driver for the GPU it runs on: `cuModuleLoadDataEx` once per CUDA context
//! (the runtime's primary context of each device, so once per device), with the JIT's error log
//! in the error when the text does not assemble. [`function`] resolves an entry point with
//! `cuModuleGetFunction` and caches it; [`launch`] runs it with `cuLaunchKernel` on a stream the
//! caller already has (a ggml backend's stream: a runtime `cudaStream_t` is a driver `CUstream`).
//!
//! Nothing here knows about models: an op that wants a PTX kernel declares its `static`
//! [`Module`] and launches from its CUDA callback. Loading happens on the first launch on a
//! device, outside any graph capture (external ops run between ggml's captured segments).

#[cfg(feature = "cuda")]
use std::ffi::c_void;

/// One `.ptx` file: a name for messages and its text.
pub struct Module {
    pub name: &'static str,
    pub text: &'static str,
}

impl Module {
    pub const fn new(name: &'static str, text: &'static str) -> Self {
        Self { name, text }
    }

    /// The `.entry` points the text declares, in order.
    pub fn entries(&self) -> Vec<&'static str> {
        entries(self.text)
    }

    /// The compute capability the text targets (`.target sm_XY` as `XY`), if it says.
    pub fn target(&self) -> Option<u32> {
        self.text.lines().find_map(|l| {
            l.trim()
                .strip_prefix(".target")?
                .trim()
                .split([',', ' '])
                .next()?
                .strip_prefix("sm_")?
                .trim_end_matches(|c: char| c.is_ascii_alphabetic())
                .parse()
                .ok()
        })
    }
}

/// The names after `.entry` in PTX text (comments skipped).
pub fn entries(text: &'static str) -> Vec<&'static str> {
    text.lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .filter_map(|l| {
            let rest = l.split_once(".entry")?.1.trim_start();
            let end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '$'))?;
            (end > 0).then(|| &rest[..end])
        })
        .collect()
}

/// A kernel resolved in the current context (a driver `CUfunction`).
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
pub struct Function(*mut c_void);

// SAFETY: a CUfunction is a handle the driver lets any thread use while its module is loaded;
// modules stay loaded for the life of the process.
#[cfg(feature = "cuda")]
unsafe impl Send for Function {}

#[cfg(feature = "cuda")]
impl Function {
    /// The driver's `CUfunction` (for a graph's kernel node).
    pub fn raw(self) -> *mut c_void {
        self.0
    }
}

#[cfg(feature = "cuda")]
#[allow(non_snake_case)]
mod sys {
    use std::ffi::{c_char, c_int, c_uint, c_void};

    pub type CUresult = c_int;

    /// cuda.h's `CUjit_option` values used here.
    pub const CU_JIT_INFO_LOG_BUFFER: c_uint = 3;
    pub const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: c_uint = 4;
    pub const CU_JIT_ERROR_LOG_BUFFER: c_uint = 5;
    pub const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_uint = 6;

    unsafe extern "C" {
        pub fn cuGetErrorString(e: CUresult, s: *mut *const c_char) -> CUresult;
        pub fn cuCtxGetCurrent(ctx: *mut *mut c_void) -> CUresult;
        pub fn cuModuleLoadDataEx(
            module: *mut *mut c_void,
            image: *const c_void,
            n_options: c_uint,
            options: *mut c_uint,
            values: *mut *mut c_void,
        ) -> CUresult;
        pub fn cuModuleGetFunction(
            f: *mut *mut c_void,
            module: *mut c_void,
            name: *const c_char,
        ) -> CUresult;
        #[allow(clippy::too_many_arguments)]
        pub fn cuLaunchKernel(
            f: *mut c_void,
            grid_x: c_uint,
            grid_y: c_uint,
            grid_z: c_uint,
            block_x: c_uint,
            block_y: c_uint,
            block_z: c_uint,
            smem: c_uint,
            stream: *mut c_void,
            params: *mut *mut c_void,
            extra: *mut *mut c_void,
        ) -> CUresult;
    }
}

#[cfg(feature = "cuda")]
fn cu_error(r: sys::CUresult) -> String {
    let mut s = std::ptr::null();
    // SAFETY: the driver returns a static string or leaves `s` null.
    unsafe { sys::cuGetErrorString(r, &mut s) };
    if s.is_null() {
        format!("CUDA driver error {r}")
    } else {
        // SAFETY: a NUL-terminated static string from the driver.
        let msg = unsafe { std::ffi::CStr::from_ptr(s) };
        format!("{} ({r})", msg.to_string_lossy())
    }
}

/// A module loaded in one context, with the functions resolved so far.
#[cfg(feature = "cuda")]
struct Loaded {
    ctx: usize,
    module: &'static str,
    handle: usize,
    functions: Vec<(&'static str, Function)>,
}

#[cfg(feature = "cuda")]
static LOADED: std::sync::Mutex<Vec<Loaded>> = std::sync::Mutex::new(Vec::new());

/// `entry` of `m` in the current CUDA context, loading the module there first if needed.
#[cfg(feature = "cuda")]
pub fn function(m: &'static Module, entry: &'static str) -> Result<Function, String> {
    let mut ctx = std::ptr::null_mut();
    // SAFETY: writes the current context (or null) to `ctx`.
    let r = unsafe { sys::cuCtxGetCurrent(&mut ctx) };
    if r != 0 || ctx.is_null() {
        return Err(format!(
            "ptx {}: no current CUDA context ({})",
            m.name,
            cu_error(r)
        ));
    }
    let mut loaded = LOADED.lock().unwrap_or_else(|e| e.into_inner());
    let i = match loaded
        .iter()
        .position(|l| l.ctx == ctx as usize && std::ptr::eq(l.module, m.name))
    {
        Some(i) => i,
        None => {
            let handle = load(m)?;
            loaded.push(Loaded {
                ctx: ctx as usize,
                module: m.name,
                handle,
                functions: Vec::new(),
            });
            loaded.len() - 1
        }
    };
    let l = &mut loaded[i];
    if let Some(&(_, f)) = l.functions.iter().find(|(n, _)| *n == entry) {
        return Ok(f);
    }
    let name = std::ffi::CString::new(entry).map_err(|e| e.to_string())?;
    let mut f = std::ptr::null_mut();
    // SAFETY: a module loaded in this context; a NUL-terminated name.
    let r = unsafe { sys::cuModuleGetFunction(&mut f, l.handle as *mut c_void, name.as_ptr()) };
    if r != 0 {
        return Err(format!(
            "ptx {}: no entry {entry} ({})",
            m.name,
            cu_error(r)
        ));
    }
    l.functions.push((entry, Function(f)));
    Ok(Function(f))
}

/// JIT-compiles `m` in the current context; the JIT's error log in the error.
#[cfg(feature = "cuda")]
fn load(m: &Module) -> Result<usize, String> {
    const LOG: usize = 16 * 1024;
    let text = std::ffi::CString::new(m.text).map_err(|e| format!("ptx {}: {e}", m.name))?;
    let mut info = vec![0u8; LOG];
    let mut error = vec![0u8; LOG];
    let mut options = [
        sys::CU_JIT_INFO_LOG_BUFFER,
        sys::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        sys::CU_JIT_ERROR_LOG_BUFFER,
        sys::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    ];
    let mut values = [
        info.as_mut_ptr() as *mut c_void,
        LOG as *mut c_void,
        error.as_mut_ptr() as *mut c_void,
        LOG as *mut c_void,
    ];
    let mut module = std::ptr::null_mut();
    // SAFETY: NUL-terminated PTX; option values as cuda.h documents them (buffers and their
    // sizes, a size passed as the pointer value).
    let r = unsafe {
        sys::cuModuleLoadDataEx(
            &mut module,
            text.as_ptr() as *const c_void,
            options.len() as u32,
            options.as_mut_ptr(),
            values.as_mut_ptr(),
        )
    };
    let log = |b: &[u8]| {
        let n = b.iter().position(|&c| c == 0).unwrap_or(b.len());
        String::from_utf8_lossy(&b[..n]).trim().to_string()
    };
    if r != 0 {
        return Err(format!(
            "ptx {}: cuModuleLoadDataEx failed: {}; JIT log: {}",
            m.name,
            cu_error(r),
            log(&error)
        ));
    }
    let info = log(&info);
    tracing::debug!(
        module = m.name,
        "ptx module loaded{}{info}",
        if info.is_empty() { "" } else { ": " }
    );
    Ok(module as usize)
}

/// Launches `f` on `stream` (a runtime `cudaStream_t` of the current context). `params` holds
/// one pointer per kernel parameter, to its value.
///
/// # Safety
/// `params` must match the kernel's parameter list, and every address the kernel reads or
/// writes through them must be valid for the launch.
#[cfg(feature = "cuda")]
pub unsafe fn launch(
    f: Function,
    grid: [u32; 3],
    block: [u32; 3],
    smem: u32,
    stream: *mut c_void,
    params: &mut [*mut c_void],
) -> Result<(), String> {
    // SAFETY: as the caller promises.
    let r = unsafe {
        sys::cuLaunchKernel(
            f.0,
            grid[0],
            grid[1],
            grid[2],
            block[0],
            block[1],
            block[2],
            smem,
            stream,
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    if r != 0 {
        return Err(format!("ptx launch failed: {}", cu_error(r)));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
