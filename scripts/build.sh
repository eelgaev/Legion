#!/usr/bin/env bash
# Build Legion with CUDA for a supported target. See docs/building.md.
#   scripts/build.sh <gaming-pc|spark|ac922> [extra cargo args...]
# Any variable you export beforehand (CMAKE_CUDA_ARCHITECTURES, GGML_NATIVE, CC, CXX,
# CUDAHOSTCXX, CUDA_PATH) wins over the target's default.
set -euo pipefail

usage() {
    echo "usage: $0 <gaming-pc|spark|ac922> [cargo build args...]" >&2
    exit 2
}

[ $# -ge 1 ] || usage
target=$1
shift

cd "$(dirname "$0")/.."

case "$target" in
gaming-pc)
    export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-86;89;120}"
    ;;
spark)
    export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-121}"
    ;;
ac922)
    export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-70}"
    # The system gcc is too old; use gcc-toolset-12 unless a compiler is already chosen.
    if [ -z "${CXX:-}" ]; then
        toolset=/opt/rh/gcc-toolset-12/enable
        if [ ! -r "$toolset" ]; then
            echo "error: $toolset not found; install gcc-toolset-12 (see docs/building.md)" >&2
            exit 1
        fi
        # shellcheck disable=SC1090
        . "$toolset"
        export CC="${CC:-gcc}" CXX=g++ CUDAHOSTCXX="${CUDAHOSTCXX:-g++}"
    fi
    ;;
*)
    usage
    ;;
esac

git submodule update --init third_party/ggml
echo "target=$target CMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES" >&2
exec cargo build --release --features cuda "$@"
