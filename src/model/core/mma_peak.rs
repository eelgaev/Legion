//! The tensor cores' real `mma.sync` peak on the GPU at hand (#1183): register-only MMA loops
//! (kernels in `src/cuda/mma_peak.cu`), no memory traffic, for each form a kernel of Legion's
//! uses or could use: m16n8k16 BF16 and F16 with F32 sums, F16 with F16 sums, m16n8k32 E4M3
//! with F32 sums (sm_89+), and on sm_120a / sm_121a (a GB10, RTX 50) `kind::f8f6f4` E4M3 and the
//! block-scaled `kind::mxf8f6f4` (E4M3, UE8M0 scales) and `kind::mxf4nvf4` (E2M1, UE4M3 scales)
//! forms. Each form runs over [`WARPS`] warps an SM (one CTA an SM) times [`CHAINS`] independent
//! accumulators a warp; the best of the sweep is the form's peak, the GEMMs' ceiling to measure
//! a kernel's TFLOPS against.
//!
//! The SM clock comes from the kernels themselves (`%clock64` cycles of a CTA over the CUDA
//! events' time), so no `nvidia-smi` is needed; each run also counts the SMs its CTAs ran on
//! (`%smid`). A form the device's code lacks (an older target, or a 12.x build without the `a`
//! suffix) is reported as such, never launched (a probe kernel says which forms the code has).
//!
//! Run it on a GPU: `cargo test --release --features cuda --lib mma_peak -- --ignored
//! --nocapture`.

#[cfg(test)]
mod tests;

/// One MMA form of the kernels (`src/cuda/mma_peak.cu`'s FORM, the index in [`FORMS`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Form {
    /// The form as the table prints it.
    pub name: &'static str,
    /// The oldest code it needs.
    pub needs: &'static str,
    /// Its `m · n · k`: an MMA is twice as many floating-point operations, a warp's.
    pub mnk: u64,
}

/// The forms, in the kernels' order.
pub const FORMS: [Form; 7] = [
    Form {
        name: "m16n8k16 BF16 -> F32",
        needs: "sm_80",
        mnk: 16 * 8 * 16,
    },
    Form {
        name: "m16n8k16 F16 -> F32",
        needs: "sm_80",
        mnk: 16 * 8 * 16,
    },
    Form {
        name: "m16n8k16 F16 -> F16",
        needs: "sm_80",
        mnk: 16 * 8 * 16,
    },
    Form {
        name: "m16n8k32 E4M3 -> F32",
        needs: "sm_89",
        mnk: 16 * 8 * 32,
    },
    Form {
        name: "f8f6f4 m16n8k32 E4M3 -> F32",
        needs: "sm_120a",
        mnk: 16 * 8 * 32,
    },
    Form {
        name: "mxf8f6f4 1X m16n8k32 E4M3 -> F32",
        needs: "sm_120a",
        mnk: 16 * 8 * 32,
    },
    Form {
        name: "mxf4nvf4 4X m16n8k64 E2M1 -> F32",
        needs: "sm_120a",
        mnk: 16 * 8 * 64,
    },
];

/// Warps an SM the sweep runs (one CTA an SM of this many warps).
pub const WARPS: [u32; 4] = [4, 8, 16, 32];

/// Independent accumulators a warp the sweep runs (the kernels' CHAINS, in their order).
pub const CHAINS: [u32; 4] = [1, 2, 4, 8];

/// One run of a form.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Run {
    /// The form's index in [`FORMS`].
    pub form: usize,
    pub warps: u32,
    pub chains: u32,
    /// The launch's time by CUDA events.
    pub ms: f64,
    pub tflops: f64,
    /// The SM clock over the run: a CTA's `%clock64` cycles over `ms`.
    pub mhz: f64,
    /// Distinct SMs the CTAs ran on.
    pub sms_used: usize,
}

/// TFLOPS of `ctas` CTAs of `warps` warps, each issuing `chains` MMAs of `mnk` a loop trip for
/// `iters` trips, in `ms` milliseconds.
pub fn tflops(ctas: u64, warps: u64, chains: u64, iters: u64, mnk: u64, ms: f64) -> f64 {
    let flops = (ctas * warps * chains * iters) as f64 * (2 * mnk) as f64;
    if ms > 0.0 {
        flops / (ms * 1e-3) / 1e12
    } else {
        0.0
    }
}

/// Loop trips for a run of about `target_ms`, from a calibration run of `iters` trips that
/// took `ms`: at least `iters`, at most 2^30.
pub fn scaled_iters(iters: u64, ms: f64, target_ms: f64) -> u64 {
    if ms <= 0.0 {
        return iters;
    }
    ((iters as f64 * target_ms / ms) as u64).clamp(iters, 1 << 30)
}

/// Each measured form's fastest run, in [`FORMS`]' order.
pub fn best(runs: &[Run]) -> Vec<Run> {
    let mut out: Vec<Run> = Vec::new();
    for r in runs {
        match out.iter_mut().find(|b| b.form == r.form) {
            Some(b) if r.tflops > b.tflops => *b = *r,
            Some(_) => {}
            None => out.push(*r),
        }
    }
    out.sort_by_key(|r| r.form);
    out
}

/// The report: every run, then each form's peak with its warps, chains, clock and FLOPs a clock
/// an SM; `missing` the forms the device's code lacks (their indices).
pub fn report(sms: usize, runs: &[Run], missing: &[usize]) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(s, "mma.sync peak sweep ({sms} SMs, one CTA an SM)");
    let _ = writeln!(
        s,
        "{:<34} {:>5} {:>6} {:>9} {:>9} {:>7} {:>4}",
        "form", "warps", "chains", "ms", "TFLOPS", "MHz", "SMs"
    );
    for r in runs {
        let _ = writeln!(
            s,
            "{:<34} {:>5} {:>6} {:>9.3} {:>9.1} {:>7.0} {:>4}",
            FORMS[r.form].name, r.warps, r.chains, r.ms, r.tflops, r.mhz, r.sms_used
        );
    }
    let _ = writeln!(s, "\npeaks:");
    for r in best(runs) {
        let per_clk = if r.mhz > 0.0 && sms > 0 {
            r.tflops * 1e12 / (r.mhz * 1e6) / sms as f64
        } else {
            0.0
        };
        let _ = writeln!(
            s,
            "{:<34} {:>7.1} TFLOPS  ({} warps x {} chains, {:.0} MHz, {:.0} FLOP/clk/SM)",
            FORMS[r.form].name, r.tflops, r.warps, r.chains, r.mhz, per_clk
        );
    }
    for &f in missing {
        let _ = writeln!(
            s,
            "{:<34} not in this build's code for the device (needs {})",
            FORMS[f].name, FORMS[f].needs
        );
    }
    s
}

/// The sweep on CUDA device `dev`: every form its code has, [`WARPS`] x [`CHAINS`], each run
/// about `target_ms` long after a calibration run. Returns the SM count, the runs and the
/// forms the device's code lacks.
#[cfg(feature = "cuda")]
pub fn measure(dev: i32, target_ms: f64) -> Result<(usize, Vec<Run>, Vec<usize>), String> {
    cuda::measure(dev, target_ms)
}

#[cfg(feature = "cuda")]
mod cuda {
    //! The launches: `cudaLaunchKernel` from Rust (no host code in the `.cu`).

    use std::os::raw::{c_int, c_void};

    use super::{CHAINS, FORMS, Run, WARPS};

    /// `legion_mma_peak` (src/cuda/mma_peak.cu).
    #[repr(C)]
    struct Params {
        out: *mut f32,
        clk: *mut i64,
        iters: i32,
        seed: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Dim3 {
        x: u32,
        y: u32,
        z: u32,
    }

    /// `cudaDevAttrMultiProcessorCount`.
    const DEV_ATTR_SMS: c_int = 16;
    /// `cudaMemcpyDeviceToHost`.
    const D2H: c_int = 2;
    /// The largest CTA of the sweep (32 warps).
    const MAX_THREADS: usize = 1024;
    /// Loop trips of a calibration run.
    const CALIBRATE_ITERS: u64 = 1024;

    unsafe extern "C" {
        static legion_mma_peak_kernels: [[*const c_void; 4]; 7];
        static legion_mma_peak_probe: *const c_void;
        fn cudaSetDevice(device: c_int) -> c_int;
        fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
        fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> c_int;
        fn cudaFree(p: *mut c_void) -> c_int;
        fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: c_int) -> c_int;
        fn cudaDeviceSynchronize() -> c_int;
        fn cudaGetLastError() -> c_int;
        fn cudaLaunchKernel(
            func: *const c_void,
            grid: Dim3,
            block: Dim3,
            args: *mut *mut c_void,
            shared: usize,
            stream: *mut c_void,
        ) -> c_int;
        fn cudaEventCreate(e: *mut *mut c_void) -> c_int;
        fn cudaEventRecord(e: *mut c_void, s: *mut c_void) -> c_int;
        fn cudaEventSynchronize(e: *mut c_void) -> c_int;
        fn cudaEventElapsedTime(ms: *mut f32, a: *mut c_void, b: *mut c_void) -> c_int;
        fn cudaEventDestroy(e: *mut c_void) -> c_int;
    }

    fn rt(e: c_int, what: &str) -> Result<(), String> {
        match e {
            0 => Ok(()),
            e => Err(format!("mma_peak: {what}: CUDA error {e}")),
        }
    }

    /// Device memory freed on drop.
    struct Buf(*mut c_void);

    impl Buf {
        fn new(bytes: usize) -> Result<Self, String> {
            let mut p = std::ptr::null_mut();
            // SAFETY: an out pointer.
            rt(unsafe { cudaMalloc(&mut p, bytes) }, "alloc")?;
            Ok(Buf(p))
        }
    }

    impl Drop for Buf {
        fn drop(&mut self) {
            // SAFETY: memory from cudaMalloc, freed once.
            unsafe { cudaFree(self.0) };
        }
    }

    /// Two events freed on drop.
    struct Events([*mut c_void; 2]);

    impl Drop for Events {
        fn drop(&mut self) {
            for e in self.0 {
                // SAFETY: events from cudaEventCreate, destroyed once.
                unsafe { cudaEventDestroy(e) };
            }
        }
    }

    /// The forms the current device's code has (a bit each), by the probe kernel.
    fn probe() -> Result<u32, String> {
        let buf = Buf::new(4)?;
        let mut p = buf.0 as *mut c_int;
        let mut args = [(&mut p as *mut *mut c_int).cast::<c_void>()];
        let one = Dim3 { x: 1, y: 1, z: 1 };
        let mut forms: c_int = 0;
        // SAFETY: the probe takes one `int *`; then a copy of the int it wrote.
        unsafe {
            rt(
                cudaLaunchKernel(
                    legion_mma_peak_probe,
                    one,
                    one,
                    args.as_mut_ptr(),
                    0,
                    std::ptr::null_mut(),
                ),
                "probe",
            )?;
            rt(
                cudaMemcpy((&mut forms as *mut c_int).cast(), buf.0, 4, D2H),
                "probe copy",
            )?;
        }
        Ok(forms as u32)
    }

    /// A sweep's buffers and events: `sms` CTAs a launch.
    struct Bench {
        sms: usize,
        out: Buf,
        clk: Buf,
        ev: Events,
    }

    /// One launch of form `f` with chains index `c`: `b.sms` CTAs of `warps` warps, `iters`
    /// trips. Returns (ms, cycles of CTA 0, distinct SMs).
    fn launch(
        b: &Bench,
        f: usize,
        c: usize,
        warps: u32,
        iters: u64,
    ) -> Result<(f64, i64, usize), String> {
        let (sms, out, clk, ev) = (b.sms, &b.out, &b.clk, &b.ev);
        let mut p = Params {
            out: out.0 as *mut f32,
            clk: clk.0 as *mut i64,
            iters: i32::try_from(iters).map_err(|_| "mma_peak: too many trips".to_string())?,
            seed: 0x1234_5678 ^ ((f as u32) << 8) ^ warps,
        };
        let mut args = [(&mut p as *mut Params).cast::<c_void>()];
        let grid = Dim3 {
            x: sms as u32,
            y: 1,
            z: 1,
        };
        let block = Dim3 {
            x: warps * 32,
            y: 1,
            z: 1,
        };
        let mut ms = 0f32;
        let mut clocks = vec![0i64; 3 * sms];
        // SAFETY: the kernel takes one `Params`, writes `sms · warps · 32` floats and `3 · sms`
        // i64s, within the buffers; events and the default stream of the current device.
        unsafe {
            rt(cudaEventRecord(ev.0[0], std::ptr::null_mut()), "event")?;
            rt(
                cudaLaunchKernel(
                    legion_mma_peak_kernels[f][c],
                    grid,
                    block,
                    args.as_mut_ptr(),
                    0,
                    std::ptr::null_mut(),
                ),
                "launch",
            )?;
            rt(cudaEventRecord(ev.0[1], std::ptr::null_mut()), "event")?;
            rt(cudaEventSynchronize(ev.0[1]), "sync")?;
            rt(cudaGetLastError(), "kernel")?;
            rt(cudaEventElapsedTime(&mut ms, ev.0[0], ev.0[1]), "time")?;
            rt(
                cudaMemcpy(
                    clocks.as_mut_ptr().cast(),
                    clk.0,
                    clocks.len() * std::mem::size_of::<i64>(),
                    D2H,
                ),
                "clock copy",
            )?;
        }
        let mut seen: Vec<i64> = clocks.chunks(3).map(|c| c[2]).collect();
        seen.sort_unstable();
        seen.dedup();
        Ok((f64::from(ms), clocks[1] - clocks[0], seen.len()))
    }

    pub(super) fn measure(
        dev: i32,
        target_ms: f64,
    ) -> Result<(usize, Vec<Run>, Vec<usize>), String> {
        let mut sms: c_int = 0;
        // SAFETY: plain device calls; an out pointer.
        unsafe {
            rt(cudaSetDevice(dev), "device")?;
            rt(
                cudaDeviceGetAttribute(&mut sms, DEV_ATTR_SMS, dev),
                "SM count",
            )?;
        }
        let sms = sms.max(1) as usize;
        let forms = probe()?;
        let mut ev = Events([std::ptr::null_mut(); 2]);
        for e in &mut ev.0 {
            // SAFETY: an out pointer.
            rt(unsafe { cudaEventCreate(e) }, "event")?;
        }
        let b = Bench {
            sms,
            out: Buf::new(sms * MAX_THREADS * 4)?,
            clk: Buf::new(sms * 3 * 8)?,
            ev,
        };
        let mut runs = Vec::new();
        let mut missing = Vec::new();
        for (f, form) in FORMS.iter().enumerate() {
            if forms & (1 << f) == 0 {
                missing.push(f);
                continue;
            }
            for &warps in &WARPS {
                for (c, &chains) in CHAINS.iter().enumerate() {
                    // Warm-up and calibration, then the timed run.
                    launch(&b, f, c, warps, CALIBRATE_ITERS)?;
                    let (ms, ..) = launch(&b, f, c, warps, CALIBRATE_ITERS)?;
                    let iters = super::scaled_iters(CALIBRATE_ITERS, ms, target_ms);
                    let (ms, cycles, used) = launch(&b, f, c, warps, iters)?;
                    runs.push(Run {
                        form: f,
                        warps,
                        chains,
                        ms,
                        tflops: super::tflops(
                            sms as u64,
                            u64::from(warps),
                            u64::from(chains),
                            iters,
                            form.mnk,
                            ms,
                        ),
                        mhz: if ms > 0.0 {
                            cycles as f64 / (ms * 1e3)
                        } else {
                            0.0
                        },
                        sms_used: used,
                    });
                }
            }
        }
        // SAFETY: a plain sync.
        rt(unsafe { cudaDeviceSynchronize() }, "sync")?;
        Ok((sms, runs, missing))
    }
}
