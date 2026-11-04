//! Parsing of build-environment switches forwarded to cmake. Shared with
//! tests/cmake_env.rs so the tests run the build script's own code.

/// Normalises a boolean env value (`1/0/on/off/true/false/yes/no`, any case) to
/// cmake's `ON`/`OFF`. Unknown values are an error rather than silently OFF.
pub fn cmake_bool(v: &str) -> Result<&'static str, String> {
    match v.trim().to_ascii_lowercase().as_str() {
        "1" | "on" | "true" | "yes" => Ok("ON"),
        "0" | "off" | "false" | "no" => Ok("OFF"),
        other => Err(format!("expected ON or OFF, got `{other}`")),
    }
}

/// A CUDA architecture as Legion's kernels are built for it: plain `12X` (with or without
/// `-real` / `-virtual`) becomes the architecture-specific `12Xa`, as ggml's CMakeLists does,
/// since the block-scaled FP4 tensor-core instructions (src/cuda/sm120/nvfp4_a4.cu) exist only
/// there. Everything else is kept.
pub fn arch_specific(arch: &str) -> String {
    let (num, kind) = arch
        .split_once('-')
        .map_or((arch, None), |(n, k)| (n, Some(k)));
    let blackwell =
        num.len() == 3 && num.starts_with("12") && num.bytes().all(|b| b.is_ascii_digit());
    let num = if blackwell {
        format!("{num}a")
    } else {
        num.to_string()
    };
    match kind {
        Some(k) => format!("{num}-{k}"),
        None => num,
    }
}

/// The architectures of `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` output
/// ("12.1" per GPU), deduplicated in order; empty when there is none.
pub fn archs_from_compute_caps(out: &str) -> Vec<String> {
    let mut archs: Vec<String> = Vec::new();
    for line in out.lines() {
        let Some((major, minor)) = line.trim().split_once('.') else {
            continue;
        };
        if major.parse::<u32>().is_err() || minor.parse::<u32>().is_err() {
            continue;
        }
        let a = format!("{major}{minor}");
        if !archs.contains(&a) {
            archs.push(a);
        }
    }
    archs
}
