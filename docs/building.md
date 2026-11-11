# Building Legion

Legion builds the pinned ggml in `third_party/ggml` through `build.rs`. A CPU-only build needs
a C/C++ toolchain, cmake and git:

```sh
git submodule update --init third_party/ggml
cargo build --release
```

`rust-toolchain.toml` pins the Rust version, so `cargo clippy -- -D warnings` gives the same
result on every machine; rustup fetches that version on first use (or run
`rustup toolchain install` in the checkout). A version bump is its own PR: change the pin,
fix the new lints.

The CUDA build adds `--features cuda` and needs a CUDA toolkit. `scripts/build.sh <target>`
wraps the per-target settings below:

```sh
scripts/build.sh spark    # or: ac922
```

Extra arguments go to `cargo build`. Variables you export beforehand override the defaults.

## Environment variables read by `build.rs`

| Variable | Effect |
|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | GPU architectures to compile for (e.g. `70`, `120` or `121`; a `12x` gets its `a` suffix automatically). Always set it explicitly: without it ggml uses `native`, which needs a visible GPU at build time, or falls back to a long list of architectures. |
| `GGML_NATIVE` | `ON` (default) tunes the CPU backend for the build machine. Set `OFF` for binaries that must run on other CPUs. |
| `GGML_SSE42`, `GGML_AVX`, `GGML_AVX2`, `GGML_FMA`, `GGML_F16C`, `GGML_BMI2`, `GGML_AVX_VNNI`, `GGML_AVX512` | ggml's x86 instruction sets, passed through when set (`ON`/`OFF`). With `GGML_NATIVE=OFF` ggml turns AVX2, FMA and BMI2 on by default; an AVX-only CPU (Sandy/Ivy Bridge) needs `GGML_AVX2=OFF GGML_FMA=OFF GGML_BMI2=OFF`. |
| `CC`, `CXX` | C and C++ compiler for ggml. |
| `CUDAHOSTCXX` | Host compiler nvcc uses (cmake `CMAKE_CUDA_HOST_COMPILER`). Must be compatible with the CUDA version. |
| `CUDA_PATH` / `CUDA_HOME` | CUDA toolkit root used for linking (default `/opt/cuda`). The `lib64`, `lib` and `targets/{x86_64,sbsa,ppc64le}-linux/lib` directories are searched. |

`build.rs` re-runs when any of these change.

## IBM AC922 (POWER9 ppc64le, V100)

The full machine setup (RHEL 8, kernel, driver, NVLink) is in [ac922.md](ac922.md).

- RHEL 8, CUDA 12.x, V100 = `sm_70`, so `CMAKE_CUDA_ARCHITECTURES=70`.
- Recommended kernel: a current mainline kernel with the [power9-nvidia-nvlink2](https://github.com/eelgaev/power9-nvidia-nvlink2) patches. Linux dropped the
  POWER9 NVLink2 / NPU support in 5.3 and 5.13; these patches bring it back (CPU-GPU NVLink, ATS,
  GPU memory as coherent NUMA nodes, unified memory) with NVIDIA's last ppc64le driver,
  550.54.15. Legion's host-resident experts read host memory over this link. The AC922 numbers
  in the README were measured on a 6.12 kernel with these patches. Smaller memory blocks there
  also leave each V100 about 0.75 GiB more usable VRAM than RHEL 8's 4.18 kernel.
- The system gcc is too old. Install `gcc-toolset-12` and enable it in every shell that builds:
  `source /opt/rh/gcc-toolset-12/enable`.
- ggml's CPU backend on POWER uses the VSX `vector` keyword, which needs GNU C++ (`gnu++17`).
  ggml sets `CMAKE_CXX_STANDARD 17` with cmake's default GNU extensions, so nothing extra is
  needed, but do not force `CMAKE_CXX_EXTENSIONS=OFF`.
- `GGML_NATIVE=ON` (default) enables the POWER9 flags for the build machine.
- The machine has two NUMA nodes; GPUs 0-1 are attached to node 0 and GPUs 2-3 to node 8.
  This matters at run time (pin threads and memory next to the GPUs in use), not at build time.

```sh
scripts/build.sh ac922
# equivalent:
source /opt/rh/gcc-toolset-12/enable
CC=gcc CXX=g++ CUDAHOSTCXX=g++ CMAKE_CUDA_ARCHITECTURES=70 cargo build --release --features cuda
```

## DGX Spark (GB10, aarch64)

- Ubuntu on aarch64 (sbsa), CUDA 13, GB10 = `sm_121`, so `CMAKE_CUDA_ARCHITECTURES=121`.
- Packages: `build-essential cmake git` and the CUDA 13 toolkit (`/usr/local/cuda`).
- CPU and GPU share unified memory.
- CUDA libraries live under `targets/sbsa-linux/lib`, which `build.rs` searches.

```sh
export CUDA_PATH=/usr/local/cuda
scripts/build.sh spark
# equivalent:
CMAKE_CUDA_ARCHITECTURES=121 cargo build --release --features cuda
```

## Desktop Blackwell (RTX 50, sm_120)

- RTX 5090 / 5080 / 5070 = `sm_120`, so `CMAKE_CUDA_ARCHITECTURES=120`. CUDA 12.8 or newer
  (12.9 or newer when the list also has `121`).
- `build.rs` turns any `12x` into the architecture-specific `12xa` (`120` builds `sm_120a`), as
  ggml does: the FP4 tensor-core kernels (`--nvfp4-a4`) exist only there. Do not add the `a`
  yourself.
- An `sm_121a` (DGX Spark) binary does not run on an RTX 50, and an `sm_120a` one does not run
  on a GB10. `CMAKE_CUDA_ARCHITECTURES="120;121"` puts both GPUs' code in one binary. The list
  names GPU targets only: the host side is still built for the build machine's CPU, so an x86
  RTX 50 PC and an aarch64 Spark each need their own build (with `GGML_NATIVE=OFF` if the
  binary must run on other CPUs of the same architecture).
- The two share the SM, so an RTX 50 takes the same kernels as the GB10 where the choice is
  about the instructions (the hyper-connection mixer, GLM's MLA prompt kernel, programmatic
  dependent launch). Choices tuned to the GB10's memory (unified LPDDR5X) stay GB10-only.
- `--q8-group` is opt-in on an RTX 50 for now (unmeasured there).
- If `--nvfp4-a4` is set and the binary has no FP4 code for the GPU, the load log says so and the
  NVFP4 matmuls stay W4A16.

```sh
CMAKE_CUDA_ARCHITECTURES=120 cargo build --release --features cuda
# RTX 50 and GB10 code in one binary (one host architecture):
CMAKE_CUDA_ARCHITECTURES="120;121" cargo build --release --features cuda
```

## Turing (sm_75: RTX 20, GTX 16, T4)

- `CMAKE_CUDA_ARCHITECTURES=75`. Turing runs the V100's kernels (an `sm_70` binary runs too),
  with 64 KB of shared memory a block instead of 96: the WMMA prompt attention takes 32 query
  rows a CTA where 64 do not fit (head size 512, Gemma 4's global layers), with the same bits.
- GTX 16 cards (TU11x) have no tensor cores, but the WMMA and ggml's MMA kernels still run on them
  (checked on a GTX 1660 Ti).
- Quantized KV caches need compute capability 8.0+, as on the V100.

```sh
CMAKE_CUDA_ARCHITECTURES=75 cargo build --release --features cuda
# a binary for an AVX-only host CPU (Ivy Bridge; on Sandy Bridge, which has no F16C, add GGML_F16C=OFF):
GGML_NATIVE=OFF GGML_AVX2=OFF GGML_FMA=OFF GGML_BMI2=OFF CMAKE_CUDA_ARCHITECTURES=75 \
  cargo build --release --features cuda
```

## Status

The AC922 and DGX Spark profiles are the ones the project builds and benchmarks on; fix this page
if a step differs on your machine.
