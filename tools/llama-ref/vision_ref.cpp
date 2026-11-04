// CPU-only preprocessing reference, using llama.cpp's actual metadata loader and
// mtmd dynamic-size preprocessor. No encoder evaluation or GPU initialization.
//
// Build mtmd with LLAMA_BUILD_MTMD=ON, GGML_CUDA=OFF, GGML_METAL=OFF, then:
//   g++ -O2 -std=c++17 vision_ref.cpp -I$LLAMA/include -I$LLAMA/ggml/include \
//       -I$LLAMA/tools/mtmd -L$BUILD/bin -lmtmd -lllama -lggml -lggml-base \
//       -Wl,-rpath,$BUILD/bin -o vision_ref
//   vision_ref mmproj.gguf image.jpeg reference.json [pixels.f32]
//
// The compact JSON samples every 97th value of the graph's planar RGB input,
// and sums ALL values in double precision. Optional pixels.f32 is the full input.
#include "mtmd-image.h"
#include "mtmd-helper.h"

#include <cstdio>
#include <exception>
#include <fstream>
#include <iomanip>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "usage: vision_ref mmproj.gguf image reference.json [pixels.f32]\n");
        return 2;
    }
    try {
        clip_context_params params{};
        params.use_gpu = false;
        params.warmup = false;
        params.no_alloc = true; // metadata and tensor shapes suffice for preprocessing
        params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
        auto contexts = clip_init(argv[1], params);
        if (!contexts.ctx_v) return 1;
        auto * ctx = contexts.ctx_v;
        auto bitmap = mtmd_helper_bitmap_init_from_file(nullptr, argv[2], false, {});
        if (!bitmap.bitmap) return 1;
        const int w = mtmd_bitmap_get_nx(bitmap.bitmap);
        const int h = mtmd_bitmap_get_ny(bitmap.bitmap);
        const auto * bytes = mtmd_bitmap_get_data(bitmap.bitmap);
        clip_image_u8 image;
        image.set_size({w, h}, false);
        image.cpy_buf(std::vector<uint8_t>(bytes, bytes + w * h * 3));
        const auto hp = *clip_get_hparams(ctx);
        const auto result = mtmd_image_preprocessor_dyn_size(ctx).preprocess(image);
        const auto & resized = result.entries.at(0);
        const auto & rgb = resized.get_ro_buf();
        const size_t n = resized.nx() * resized.ny();
        std::vector<float> planar(3 * n);
        // Same interleaved -> planar conversion as clip.cpp's inp_raw upload.
        double sum = 0;
        for (size_t c = 0; c < 3; ++c) {
            for (size_t i = 0; i < n; ++i) {
                planar[c * n + i] = rgb[3 * i + c];
                sum += planar[c * n + i];
            }
        }
        std::ofstream out(argv[3]);
        out << std::setprecision(17)
            << "{\n  \"width\": " << resized.nx() << ", \"height\": " << resized.ny()
            << ",\n  \"grid_w\": " << resized.nx() / hp.patch_size
            << ", \"grid_h\": " << resized.ny() / hp.patch_size
            << ",\n  \"patch_size\": " << hp.patch_size
            << ", \"spatial_merge_size\": " << hp.n_merge
            << ",\n  \"image_min_pixels\": " << hp.image_min_pixels
            << ", \"image_max_pixels\": " << hp.image_max_pixels
            << ",\n  \"image_mean\": [" << hp.image_mean[0] << ", " << hp.image_mean[1] << ", " << hp.image_mean[2]
            << "],\n  \"image_std\": [" << hp.image_std[0] << ", " << hp.image_std[1] << ", " << hp.image_std[2]
            << "],\n  \"stride\": 97, \"sum\": " << sum << ",\n  \"samples\": [\n";
        for (size_t i = 0; i < planar.size(); i += 97) {
            if (i) out << ",\n";
            out << "    " << planar[i];
        }
        out << "\n  ]\n}\n";
        if (!out) return 1;
        if (argc == 5) {
            std::ofstream raw(argv[4], std::ios::binary);
            raw.write(reinterpret_cast<const char *>(planar.data()), planar.size() * sizeof(float));
            if (!raw) return 1;
        }
        mtmd_bitmap_free(bitmap.bitmap);
        clip_free(contexts.ctx_v);
        if (contexts.ctx_a) clip_free(contexts.ctx_a);
        if (contexts.ctx_gen_a) clip_free(contexts.ctx_gen_a);
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
