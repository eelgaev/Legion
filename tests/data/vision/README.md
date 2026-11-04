# Qwen3-VL preprocessing reference

`test-1.jpeg` is copied unchanged from `tools/mtmd/test-1.jpeg` in llama.cpp
commit `f3363593f457169cdfbdfe22ea399b1cc139ba18`. The image and the preprocessing
code adapted in `src/vision` are covered by the accompanying `llama.cpp.LICENSE`
(MIT, copyright the ggml authors).

`test-1.json` was produced by `tools/llama-ref/vision_ref.cpp` using the
Qwen3.6-35B-A3B mmproj-BF16 GGUF. It contains dimensions, the patch grid before
spatial merging, loaded preprocessing metadata, every 97th planar RGB float,
and a double-precision sum of all floats. No model weights are needed by the
Rust tests. The dumper uses the actual clip metadata loader with `no_alloc`,
GPU disabled and warmup disabled; it does not evaluate the encoder.

To reproduce, set `LLAMA` to the pinned llama.cpp checkout, `BUILD` to a build
directory, and `MMPROJ` to that mmproj file:

```sh
cmake -S "$LLAMA" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_MTMD=ON
cmake --build "$BUILD" --target mtmd -j 6
g++ -O2 -std=c++17 tools/llama-ref/vision_ref.cpp \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" -I"$LLAMA/tools/mtmd" \
  -L"$BUILD/bin" -lmtmd -lllama -lggml -lggml-base \
  -Wl,-rpath,"$BUILD/bin" -o "$BUILD/vision_ref"
"$BUILD/vision_ref" "$MMPROJ" tests/data/vision/test-1.jpeg \
  tests/data/vision/test-1.json
```

PNG, JPEG, GIF and BMP decoding uses the same stb_image v2.30 as mtmd (vendored
in `third_party/stb`). The 640x480 reference image is already aligned and within
the pixel budget, so no resampling occurs. The strict fixture and tolerance
are unchanged. WebP uses the `image` crate and has no llama.cpp parity requirement.

The tiny synthetic format fixtures were generated with Pillow: `transparent.png`
is a 2x1 RGBA image filled with `(12, 34, 56, 0)`; `color.bmp` is a 2x1 RGB image
filled with `(12, 34, 56)`; `animated.gif` has that same first frame followed by
a red `(255, 0, 0)` frame, each 100 ms, looping forever. They test alpha discard,
first-frame selection, resizing and centered padding. These generated fixtures
are covered by Legion's Apache-2.0 license.
