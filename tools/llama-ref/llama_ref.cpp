// llama_ref: reference outputs from llama.cpp (CPU) for Legion's forward-pass tests.
//
//   llama_ref <model.gguf> <ids> <out.f32> [dump_dir [name-regex]]
//
// Runs the comma-separated token ids (or a file holding them) through llama.cpp in one batch and
// writes the logits of every position as raw float32, [n_tokens][n_vocab]. With dump_dir, every
// named F32/I32 intermediate whose name matches the regex (default: all) is written in full to
// dump_dir/<name>.<k>.bin (k counts repeated names) and listed in dump_dir/manifest.txt as
// `file op type ne0 ne1 ne2 ne3`.
//
// REF_LAST=n writes only the last n positions' logits. REF_GEN=n then decodes n greedy tokens one
// at a time and writes their ids to <out.f32>.gen.
//
// REF_MODE=exact (default) turns off flash attention and weight repacking, so llama.cpp runs the
// same ggml kernels as Legion's graph and the outputs should match bit for bit. REF_MODE=default
// keeps llama.cpp's own CPU defaults (flash attention, CPU_REPACK weights): what a user of
// llama.cpp actually gets. REF_THREADS sets the thread count (default 8). The KV cache is F16.
// REF_GPU=1 offloads every layer (a llama.cpp build with a GPU backend), to measure how far
// llama.cpp's own GPU run is from its CPU run.
//
// REF_ASSISTANT=<assistant.gguf> (a Gemma 4 `gemma4-assistant`): after the prompt, runs
// REF_DRAFT_STEPS (default 3) draft steps the way llama.cpp's MTP driver runs a head that shares
// the target's KV cache: every step at the position after the prompt, the first fed the
// target's greedy next token and its post-norm hidden state at the last prompt position, each
// next one the step's greedy draft and the assistant's own output residual. Writes the steps'
// logits to <out.f32>.draft.f32 ([steps][n_vocab]), their residuals to <out.f32>.draft_h.f32
// ([steps][n_embd of the target]) and the tokens fed to <out.f32>.draft.ids.
//
// Build against a CPU-only llama.cpp build (see tests/qwen35.rs):
//   g++ -O2 -std=c++17 llama_ref.cpp -I$LLAMA/include -I$LLAMA/ggml/include -I$LLAMA/src -L$BUILD/bin \
//       -lllama -lggml -lggml-base -Wl,-rpath,$BUILD/bin -o llama_ref
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "llama-ext.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

struct Dump {
    std::string dir;
    std::regex re;
    std::map<std::string, int> seen;
    FILE * manifest = nullptr;
};

static bool on_tensor(ggml_tensor * t, bool ask, void * ud) {
    auto * d = (Dump *) ud;
    const std::string name = ggml_get_name(t);
    if (name.empty() || name.find(' ') != std::string::npos) return false;
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32) return false;
    if (!std::regex_search(name, d->re)) return false;
    if (ask) return true;

    std::vector<char> raw;
    const char * base = (const char *) t->data;
    if (!ggml_backend_buffer_is_host(t->buffer)) {
        raw.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        base = raw.data();
    }
    // Views are read through their strides, so every file is dense in ggml order.
    std::vector<char> out;
    out.reserve(ggml_nelements(t) * 4);
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++)
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++)
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++)
                for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                    const char * p = base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                    out.insert(out.end(), p, p + 4);
                }
    const std::string file = name + "." + std::to_string(d->seen[name]++) + ".bin";
    FILE * f = fopen((d->dir + "/" + file).c_str(), "wb");
    fwrite(out.data(), 1, out.size(), f);
    fclose(f);
    fprintf(d->manifest, "%s %s %s %lld %lld %lld %lld\n", file.c_str(), ggml_op_desc(t), ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.gguf ids|ids-file out.f32 [dump_dir [regex]]\n", argv[0]);
        return 2;
    }
    std::string ids_text = argv[2];
    if (std::ifstream in{ids_text}) {
        std::stringstream ss;
        ss << in.rdbuf();
        ids_text = ss.str();
    }
    std::vector<llama_token> ids;
    for (char * p = strtok(ids_text.data(), ", \n"); p; p = strtok(nullptr, ", \n")) ids.push_back(atoi(p));

    const char * mode = getenv("REF_MODE");
    const bool exact = !mode || strcmp(mode, "default") != 0;
    const char * th = getenv("REF_THREADS");

    Dump dump;
    if (argc > 4) {
        dump.dir = argv[4];
        dump.re = std::regex(argc > 5 ? argv[5] : ".");
        dump.manifest = fopen((dump.dir + "/manifest.txt").c_str(), "w");
        if (!dump.manifest) { fprintf(stderr, "cannot write to %s\n", argv[4]); return 1; }
    }

    llama_backend_init();
    auto mp = llama_model_default_params();
    const char * gpu = getenv("REF_GPU");
    mp.n_gpu_layers = gpu && atoi(gpu) ? 999 : 0;
    if (exact) mp.use_extra_bufts = false;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    auto cp = llama_context_default_params();
    const char * gen_env = getenv("REF_GEN");
    const int n_gen = gen_env ? atoi(gen_env) : 0;
    cp.n_ctx = 512;
    while (cp.n_ctx < ids.size() + n_gen) cp.n_ctx *= 2;
    cp.n_batch = cp.n_ubatch = cp.n_ctx;
    cp.n_threads = cp.n_threads_batch = th ? atoi(th) : 8;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    if (exact) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (dump.manifest) {
        cp.cb_eval = on_tensor;
        cp.cb_eval_user_data = &dump;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }

    llama_batch batch = llama_batch_init((int32_t) ids.size(), 0, 1);
    for (size_t i = 0; i < ids.size(); i++) {
        batch.token[i] = ids[i];
        batch.pos[i] = (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }
    batch.n_tokens = (int32_t) ids.size();
    const char * assistant = getenv("REF_ASSISTANT");
    // The assistant's context shares the target's memory: made before the target runs, as
    // llama.cpp's server makes it (made after, its attention sees none of the target's cells).
    llama_model * md = nullptr;
    llama_context * cd = nullptr;
    if (assistant) {
        llama_set_embeddings_nextn(ctx, true, /*masked*/ false);
        md = llama_model_load_from_file(assistant, mp);
        if (!md) { fprintf(stderr, "assistant load failed\n"); return 1; }
        auto dp = cp;
        dp.cb_eval = nullptr;
        dp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        dp.ctx_other = ctx;
        cd = llama_init_from_model(md, dp);
        if (!cd) { fprintf(stderr, "assistant context failed\n"); return 1; }
        llama_set_embeddings_nextn(cd, true, /*masked*/ true);
    }
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    if (dump.manifest) fclose(dump.manifest);

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    // REF_LAST=n keeps only the last n positions' logits (long prompts make huge files).
    const char * last = getenv("REF_LAST");
    const size_t skip = last && (size_t) atoi(last) < ids.size() ? ids.size() - atoi(last) : 0;
    FILE * f = fopen(argv[3], "wb");
    for (size_t i = skip; i < ids.size(); i++) fwrite(llama_get_logits_ith(ctx, (int32_t) i), sizeof(float), n_vocab, f);
    fclose(f);
    printf("%s: %zu tokens x %d logits (%s mode)\n", argv[3], ids.size(), n_vocab, exact ? "exact" : "default");

    // REF_GEN=n: then n greedy tokens, one decode call each, written to <out>.gen as ids.
    if (n_gen > 0) {
        std::string gen;
        const float * row = llama_get_logits_ith(ctx, (int32_t) ids.size() - 1);
        for (int k = 0; k < n_gen; k++) {
            int best = 0;
            for (int v = 1; v < n_vocab; v++) if (row[v] > row[best]) best = v;
            gen += (k ? "," : "") + std::to_string(best);
            llama_batch one = llama_batch_init(1, 0, 1);
            one.token[0] = best;
            one.pos[0] = (llama_pos) (ids.size() + k);
            one.n_seq_id[0] = 1;
            one.seq_id[0][0] = 0;
            one.logits[0] = true;
            one.n_tokens = 1;
            if (llama_decode(ctx, one) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
            llama_batch_free(one);
            row = llama_get_logits_ith(ctx, 0);
        }
        FILE * g = fopen((std::string(argv[3]) + ".gen").c_str(), "w");
        fprintf(g, "%s\n", gen.c_str());
        fclose(g);
        printf("%s.gen: %d greedy tokens\n", argv[3], n_gen);
    }

    if (assistant) {
        const char * st = getenv("REF_DRAFT_STEPS");
        const int steps = st ? atoi(st) : 3;
        const int n_embd = llama_model_n_embd(model);
        const float * h0 = llama_get_embeddings_nextn_ith(ctx, (int32_t) ids.size() - 1);
        std::vector<float> h(h0, h0 + n_embd);
        const float * last_row = llama_get_logits_ith(ctx, (int32_t) ids.size() - 1);
        llama_token tok = 0;
        for (int v = 1; v < n_vocab; v++) if (last_row[v] > last_row[tok]) tok = v;
        const llama_pos pos = (llama_pos) ids.size();
        FILE * fl = fopen((std::string(argv[3]) + ".draft.f32").c_str(), "wb");
        FILE * fh = fopen((std::string(argv[3]) + ".draft_h.f32").c_str(), "wb");
        std::string fed;
        for (int k = 0; k < steps; k++) {
            fed += (k ? "," : "") + std::to_string(tok);
            llama_batch_ext * b = llama_batch_ext_init(cd);
            const int32_t idx = llama_batch_ext_add_token(b, 0, tok);
            llama_batch_ext_set_pos(b, idx, &pos);
            llama_batch_ext_set_embd_state(b, idx, { h.data(), 1, (size_t) n_embd });
            llama_batch_ext_set_output_logits(b, idx, true);
            if (llama_process(cd, LLAMA_PROCESS_TYPE_DECODE, b) != 0) { fprintf(stderr, "draft failed\n"); return 1; }
            llama_batch_ext_free(b);
            const float * lg = llama_get_logits_ith(cd, 0);
            const float * hn = llama_get_embeddings_nextn_ith(cd, 0);
            fwrite(lg, sizeof(float), n_vocab, fl);
            fwrite(hn, sizeof(float), n_embd, fh);
            tok = 0;
            for (int v = 1; v < n_vocab; v++) if (lg[v] > lg[tok]) tok = v;
            h.assign(hn, hn + n_embd);
        }
        fclose(fl);
        fclose(fh);
        FILE * fi = fopen((std::string(argv[3]) + ".draft.ids").c_str(), "w");
        fprintf(fi, "%s\n", fed.c_str());
        fclose(fi);
        printf("%s.draft.*: %d assistant steps\n", argv[3], steps);
        llama_free(cd);
        llama_model_free(md);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
