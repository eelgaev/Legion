//! Builds the pinned ggml in `third_party/ggml` as static libraries and generates
//! bindings for ggml, gguf and ggml-backend (plus ggml-cuda with the `cuda` feature).

use std::env;
use std::path::{Path, PathBuf};

// Shared with tests/git_watch.rs so the tests run the build script's own code.
#[path = "build/git_watch.rs"]
mod git_watch;
// Shared with tests/cmake_env.rs.
#[path = "build/cmake_env.rs"]
mod cmake_env;
#[path = "build/cuda_wrap.rs"]
mod cuda_wrap;
#[path = "build/stb.rs"]
mod stb;

/// ggml's x86 instruction-set switches, passed to its CMake build when set in the environment.
const X86_ISA_OPTIONS: [&str; 8] = [
    "GGML_SSE42",
    "GGML_AVX",
    "GGML_AVX_VNNI",
    "GGML_AVX2",
    "GGML_BMI2",
    "GGML_AVX512",
    "GGML_FMA",
    "GGML_F16C",
];

/// Short git commit of the checkout at `dir`, or `unknown` when there is no git
/// metadata (tarball builds, no git on PATH, empty repo), so `--version` always
/// prints an honest build identity.
fn git_hash(dir: &str) -> String {
    // Without this check `git` would walk up to whatever repo contains the tarball.
    if !Path::new(dir).join(".git").exists() {
        return "unknown".into();
    }
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(dir)
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

fn main() {
    stb::build();
    let ggml = PathBuf::from("third_party/ggml");
    if !ggml.join("CMakeLists.txt").exists() {
        panic!("third_party/ggml is empty: run `git submodule update --init third_party/ggml`");
    }
    let cuda = env::var_os("CARGO_FEATURE_CUDA").is_some();

    // Build identity for `legion --version` and the `serve` startup log (src/cli/version.rs).
    println!("cargo:rustc-env=LEGION_GIT_HASH={}", git_hash("."));
    println!(
        "cargo:rustc-env=LEGION_GGML_HASH={}",
        git_hash("third_party/ggml")
    );
    println!(
        "cargo:rustc-env=LEGION_CUDA={}",
        if cuda { "on" } else { "off" }
    );
    // Re-run when a commit moves HEAD or its ref file, so `--version` never
    // keeps printing a stale hash until something else triggers a rebuild.
    for dir in [".", "third_party/ggml"] {
        for line in git_watch::rerun_lines(Path::new(dir)) {
            println!("{line}");
        }
    }

    let mut cfg = cmake::Config::new(&ggml);
    cfg.define("BUILD_SHARED_LIBS", "OFF")
        .define("GGML_BUILD_TESTS", "OFF")
        .define("GGML_BUILD_EXAMPLES", "OFF")
        // The CPU backend is only a fallback here; skipping OpenMP keeps linking simple.
        .define("GGML_OPENMP", "OFF")
        // llamafile's sgemm for F32/F16/Q8_0 CPU matmuls: faster, and what llama.cpp builds
        // with by default, so CPU results match it bit for bit (ggml alone defaults to OFF).
        .define("GGML_LLAMAFILE", "ON")
        .define("GGML_CUDA", if cuda { "ON" } else { "OFF" })
        // Replays a decode step's kernels as one captured CUDA graph when the step's ggml graph
        // is unchanged (the engine reuses it); llama.cpp builds with this on, ggml alone off.
        .define("GGML_CUDA_GRAPHS", if cuda { "ON" } else { "OFF" })
        .profile("Release");
    if let Ok(archs) = env::var("CMAKE_CUDA_ARCHITECTURES") {
        cfg.define("CMAKE_CUDA_ARCHITECTURES", archs);
    }
    // GGML_NATIVE tunes the CPU backend for the build machine (-march=native or the
    // POWER/ARM equivalent). Default ON for local builds; set it to OFF for binaries
    // that run on other CPUs.
    let native = env::var("GGML_NATIVE").unwrap_or_else(|_| "ON".into());
    cfg.define(
        "GGML_NATIVE",
        cmake_env::cmake_bool(&native).unwrap_or_else(|e| panic!("GGML_NATIVE: {e}")),
    );
    // With GGML_NATIVE=OFF, ggml's x86 build turns AVX2, FMA and BMI2 on by default, which an
    // AVX-only CPU (Sandy/Ivy Bridge) cannot run. Its instruction-set switches pass through when
    // set (e.g. GGML_AVX2=OFF GGML_FMA=OFF GGML_BMI2=OFF for such a CPU, plus GGML_F16C=OFF on
    // Sandy Bridge, which has no F16C).
    for var in X86_ISA_OPTIONS {
        if let Some(v) = env::var_os(var).filter(|v| !v.is_empty()) {
            let v = v.to_string_lossy();
            cfg.define(
                var,
                cmake_env::cmake_bool(&v).unwrap_or_else(|e| panic!("{var}: {e}")),
            );
        }
    }
    // A newer host toolchain (e.g. a gcc-toolset on RHEL) is selected with CC/CXX/CUDAHOSTCXX.
    for (var, define) in [
        ("CC", "CMAKE_C_COMPILER"),
        ("CXX", "CMAKE_CXX_COMPILER"),
        ("CUDAHOSTCXX", "CMAKE_CUDA_HOST_COMPILER"),
    ] {
        if let Some(v) = env::var_os(var).filter(|v| !v.is_empty()) {
            cfg.define(define, v);
        }
    }
    let dst = cfg.build();

    for dir in ["lib", "lib64"] {
        println!("cargo:rustc-link-search=native={}", dst.join(dir).display());
    }
    if cuda {
        build_cuda_kernels(&ggml);
        for arg in cuda_wrap::link_args() {
            println!("cargo:rustc-link-arg={arg}");
        }
    }
    let mut libs = vec!["ggml", "ggml-base", "ggml-cpu"];
    if cuda {
        libs.push("ggml-cuda");
    }
    for lib in libs {
        println!("cargo:rustc-link-lib=static={lib}");
    }
    if cuda {
        // CUDA_PATH / CUDA_HOME, else the usual install locations: /usr/local/cuda (NVIDIA's
        // installers, most distros), then /opt/cuda (Arch).
        let cuda_root = env::var("CUDA_PATH")
            .or_else(|_| env::var("CUDA_HOME"))
            .unwrap_or_else(|_| {
                ["/usr/local/cuda", "/opt/cuda"]
                    .into_iter()
                    .find(|d| Path::new(d).exists())
                    .unwrap_or("/usr/local/cuda")
                    .into()
            });
        for dir in [
            "lib64",
            "lib",
            "targets/x86_64-linux/lib",
            "targets/sbsa-linux/lib",
            "targets/ppc64le-linux/lib",
        ] {
            println!("cargo:rustc-link-search=native={cuda_root}/{dir}");
        }
        for lib in ["cudart", "cublas", "cublasLt", "cuda"] {
            println!("cargo:rustc-link-lib={lib}");
        }
        // ggml-cuda uses NCCL for multi-GPU all-reduce when cmake finds it.
        let cache = std::fs::read_to_string(dst.join("build/CMakeCache.txt")).unwrap_or_default();
        if let Some(lib) = cache
            .lines()
            .find_map(|l| l.strip_prefix("NCCL_LIBRARY:FILEPATH="))
            .filter(|p| !p.ends_with("-NOTFOUND"))
        {
            let lib = PathBuf::from(lib);
            if let Some(dir) = lib.parent() {
                println!("cargo:rustc-link-search=native={}", dir.display());
            }
            println!("cargo:rustc-link-lib=nccl");
        }
    }
    println!("cargo:rustc-link-lib=stdc++");

    let mut builder = bindgen::Builder::default()
        .header("src/ggml/wrapper.h")
        .clang_arg(format!("-I{}", ggml.join("include").display()))
        // ggml-backend-impl.h: the buffer interfaces Rust implements (src/residency).
        .clang_arg(format!("-I{}", ggml.join("src").display()))
        .allowlist_function("(ggml|gguf)_.*")
        .allowlist_type("(ggml|gguf)_.*")
        .allowlist_var("(GGML|GGUF)_.*")
        .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));
    if cuda {
        builder = builder.clang_arg("-DLEGION_CUDA");
    }
    let bindings = builder
        .generate()
        .expect("bindgen failed on src/ggml/wrapper.h");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("ggml_bindings.rs");
    bindings.write_to_file(out).expect("writing ggml bindings");

    println!("cargo:rerun-if-changed=src/ggml/wrapper.h");
    println!("cargo:rerun-if-changed=src/cuda");
    println!("cargo:rerun-if-changed=third_party/ggml/include");
    for var in [
        "CMAKE_CUDA_ARCHITECTURES",
        "GGML_NATIVE",
        "CC",
        "CXX",
        "CUDAHOSTCXX",
        "CUDA_PATH",
        "CUDA_HOME",
    ]
    .into_iter()
    .chain(X86_ISA_OPTIONS)
    {
        println!("cargo:rerun-if-env-changed={var}");
    }
}

/// nvcc flags for a CMake-style architecture list ("89", "86-real;90-virtual", "native").
fn cuda_arch_flags(archs: &str) -> Vec<String> {
    let mut flags = Vec::new();
    for a in archs
        .split([';', ','])
        .map(str::trim)
        .filter(|a| !a.is_empty())
        .map(cmake_env::arch_specific)
    {
        let a = a.as_str();
        if a == "native" || a == "all" || a == "all-major" {
            flags.push(format!("-arch={a}"));
            continue;
        }
        let (num, kind) = a.split_once('-').unwrap_or((a, ""));
        match kind {
            "real" => flags.push(format!("-gencode=arch=compute_{num},code=sm_{num}")),
            "virtual" => flags.push(format!("-gencode=arch=compute_{num},code=compute_{num}")),
            _ => flags.push(format!(
                "-gencode=arch=compute_{num},code=[sm_{num},compute_{num}]"
            )),
        }
    }
    flags
}

/// Legion's own CUDA code (src/cuda): kernels that run between ggml graph segments, and the
/// shim into ggml's CUDA backend they need. Built against the pinned ggml's headers, for the
/// architectures ggml is built for (`CMAKE_CUDA_ARCHITECTURES`, default `native`).
fn build_cuda_kernels(ggml: &Path) {
    let mut archs = env::var("CMAKE_CUDA_ARCHITECTURES").unwrap_or_else(|_| "native".into());
    // nvcc's -arch=native picks 12X, not 12Xa, for consumer Blackwell (GB10, RTX 50): name the
    // build machine's GPUs instead when nvidia-smi can.
    if archs == "native" {
        let caps = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| cmake_env::archs_from_compute_caps(&s))
            .unwrap_or_default();
        if !caps.is_empty() {
            archs = caps.join(";");
        }
    }
    let mut b = cc::Build::new();
    b.cuda(true)
        .cudart("shared")
        .include(ggml.join("include"))
        .include(ggml.join("src"))
        // The slowest translation units first, so the parallel build starts them first (#955:
        // expert_mv and paged_attn are split by weight type and head size, nvfp4_mm by kernel
        // and tile width).
        .file("src/cuda/paged_attn/d512.cu")
        .file("src/cuda/expert_mv/iq2.cu")
        .file("src/cuda/expert_mv/iq3.cu")
        .file("src/cuda/expert_mv/q4_k.cu")
        .file("src/cuda/expert_mv/q5_k.cu")
        .file("src/cuda/expert_mv/q5_1.cu")
        .file("src/cuda/expert_mv/q6_k.cu")
        .file("src/cuda/expert_mv/q8_0.cu")
        .file("src/cuda/paged_attn/d256.cu")
        .file("src/cuda/paged_attn/d512_compact.cu")
        .file("src/cuda/paged_attn/d128.cu")
        .file("src/cuda/expert_mv/k_low.cu")
        .file("src/cuda/expert_mv/dispatch.cu")
        .file("src/cuda/paged_attn/dispatch.cu")
        .file("src/cuda/shim.cu")
        .file("src/cuda/qsa_attn.cu")
        .file("src/cuda/moe_gemm.cu")
        .file("src/cuda/qwen4_hc.cu")
        .file("src/cuda/fp8_mm.cu")
        .file("src/cuda/fp8_act.cu")
        .file("src/cuda/fp8_block.cu")
        .file("src/cuda/bf16_experts.cu")
        .file("src/cuda/nvfp4_mm/mvg_nt1.cu")
        .file("src/cuda/nvfp4_mm/mv_nt4.cu")
        .file("src/cuda/nvfp4_mm/gemm_mi2.cu")
        .file("src/cuda/nvfp4_mm/mv_nt2.cu")
        .file("src/cuda/nvfp4_mm/gemm_mi1.cu")
        .file("src/cuda/nvfp4_mm/gemm_glu.cu")
        .file("src/cuda/nvfp4_mm/mv_nt1.cu")
        .file("src/cuda/nvfp4_mm/mv_nt8.cu")
        .file("src/cuda/nvfp4_mm/swiglu_mv.cu")
        .file("src/cuda/nvfp4_mm/mvg_nt4.cu")
        .file("src/cuda/nvfp4_mm/mvg_nt2.cu")
        .file("src/cuda/nvfp4_mm/dispatch.cu")
        .file("src/cuda/fused.cu")
        .file("src/cuda/router_mv.cu")
        .file("src/cuda/mmq_preq.cu")
        .file("src/cuda/mmq_ids.cu")
        .file("src/cuda/hc_up_mix.cu")
        .file("src/cuda/hc_mix16.cu")
        .file("src/cuda/hc_chain.cu")
        .file("src/cuda/gate_q8.cu")
        .file("src/cuda/decode_fuse.cu")
        .file("src/cuda/node_fuse.cu")
        .file("src/cuda/ple.cu")
        .file("src/cuda/gdn_scan.cu")
        .file("src/cuda/gdn_mma.cu")
        .file("src/cuda/gdn.cu")
        .file("src/cuda/norm_rope.cu")
        .file("src/cuda/sm120/nvfp4_a4.cu")
        .file("src/cuda/sm120/nvfp4_mvq.cu")
        .file("src/cuda/dequant.cu")
        .file("src/cuda/cpu_share.cu")
        .file("src/cuda/dense_mv.cu")
        .file("src/cuda/q8_mv.cu")
        .file("src/cuda/q8_group.cu")
        .file("src/cuda/nvfp4_once.cu")
        .file("src/cuda/q8_head.cu")
        .file("src/cuda/q4k_head.cu")
        .file("src/cuda/head_mm.cu")
        .file("src/cuda/q8_mmq.cu")
        .file("src/cuda/dflash_draft.cu")
        .file("src/cuda/tu_mmq.cu")
        .file("src/cuda/sm90/cluster_select.cu")
        .file("src/cuda/glm5next.cu")
        .file("src/cuda/deepseek41.cu")
        .file("src/cuda/glm_exchange.cu")
        .file("src/cuda/glm_prefetch.cu")
        .file("src/cuda/pair_link.cu")
        .file("src/cuda/mtp_cond.cu")
        .file("src/cuda/stage_pull.cu")
        .file("src/cuda/input_pack.cu")
        .file("src/cuda/turboquant.cu")
        .file("src/cuda/tp_net.cu")
        // As ggml-cuda is built: CUDA graphs on (it changes ggml_backend_cuda_context).
        .define("GGML_CUDA_USE_GRAPHS", None)
        .flag("-std=c++17")
        .flag("-O3")
        .flag("--use_fast_math")
        .warnings(false);
    for f in cuda_arch_flags(&archs) {
        b.flag(&f);
    }
    if let Some(host) = env::var_os("CUDAHOSTCXX").filter(|v| !v.is_empty()) {
        // Instead of cc's own -ccbin (the C++ compiler), which nvcc would warn about twice.
        b.ccbin(false);
        b.flag(format!("-ccbin={}", host.to_string_lossy()));
    }
    b.compile("legion_cuda");
}
