# Run commands behind the benchmarks

The commands in these pages are the saved server command lines of the runs behind the README's performance tables, with local paths
reduced to the file names of the linked Hugging Face repos (directories, ports, API keys and the host
binding removed; `--port 8090` added). All requests use temperature 0.

- `--prefix-cache-states 0` is a measurement setting: prefix caching is on by default, and the benches turn it
  off so that repeated prompts are not served from the cache. Drop it for normal use.
- **Deterministic client:** 8 fixed prompts (thinking off and on), `ignore_eos`, 256 tokens each, C requests in
  flight, 2 rounds unless noted; the number is the median round's total tok/s (per request at C=1). At 16 users
  the 8 prompts are sent twice, so 16 requests are in flight; every engine gets the same requests.
- **Prefill:** one request per prompt size, tok/s to the first token.
- **llama-benchy:** `--pp 512 --tg 256 --no-cache --exact-tg`, concurrency 1 4 16 (or 1 4), 2-3 runs; tg tok/s.
- MTP heads (`--spec-draft`) are the `mtp-*.gguf` files in the same Hugging Face repo as the model.

[llama.cpp and vLLM next to Legion](other-engines.md): the other engines' commands for every model.

## DGX Spark

- [Qwen3.8-Flash-Next UD-Q4_K_XL, 2x DGX Spark (tensor parallel)](spark/qwen38-flash-next-ud-q4-k-xl-2x-tp.md)
- [Qwen3.8-Flash-Next UD-Q4_K_XL, 1x DGX Spark](spark/qwen38-flash-next-ud-q4-k-xl-1x.md)
- [Qwen3.8-Flash-Next NVFP4, 1x DGX Spark](spark/qwen38-flash-next-nvfp4-1x.md)
- [Qwen3.8-Flash-Next NVFP4, 2x DGX Spark (tensor parallel)](spark/qwen38-flash-next-nvfp4-2x-tp.md)
- [GLM-5.3-Flash Maya-S v2 (IQ2_XXS), 1x DGX Spark](spark/glm-53-flash-maya-s-v2-iq2-xxs-1x.md)
- [Qwen3.8-27B UD-Q4_K_XL, 1x DGX Spark](spark/qwen38-27b-ud-q4-k-xl-1x.md)
- [Qwen3.8-27B NVFP4, 1x DGX Spark](spark/qwen38-27b-nvfp4-1x.md)
- [Gemma 4 31B QAT UD-Q4_K_XL, 2x DGX Spark (tensor parallel)](spark/gemma-4-31b-qat-ud-q4-k-xl-2x-tp.md)
- [Gemma 4 31B QAT UD-Q4_K_XL, 1x DGX Spark](spark/gemma-4-31b-qat-ud-q4-k-xl-1x.md)
- [Gemma 4 31B NVFP4 (RedHatAI), 1x DGX Spark](spark/gemma-4-31b-nvfp4-1x.md)
- [Nemotron 3.5 Bolt E4B and 74B-A9B UD-Q4_K_XL, 1x DGX Spark](spark/nemotron-35-bolt-1x.md)

## IBM AC922

Measured on a 6.12 kernel with the [power9-nvidia-nvlink2](https://github.com/eelgaev/power9-nvidia-nvlink2) patches (NVLink2, ATS and unified memory
on current kernels), the recommended setup; see [ac922.md](../ac922.md) for the machine's setup.

- [Qwen3.8-Flash-Next UD-Q4_K_XL, IBM AC922 4x V100 SXM2 16 GB](ac922/qwen38-flash-next-ud-q4-k-xl-4x.md)
- [Qwen3.8-Flash-Next UD-Q4_K_XL, IBM AC922 2x V100 SXM2 16 GB](ac922/qwen38-flash-next-ud-q4-k-xl-2x.md)
- [GLM-5.3-Flash Maya-S v2 (IQ2_XXS), IBM AC922 4x V100 SXM2 16 GB](ac922/glm-53-flash-maya-s-v2-iq2-xxs-4x.md)
- [GLM-5.3-Flash UD-Q4_K_XL, IBM AC922 4x V100 SXM2 16 GB](ac922/glm-53-flash-ud-q4-k-xl-4x.md)
- [Gemma 4 31B QAT UD-Q4_K_XL, IBM AC922 2x V100 SXM2 16 GB](ac922/gemma-4-31b-qat-ud-q4-k-xl-2x.md)
- [Qwen3.8-27B UD-Q4_K_XL, IBM AC922 2x V100 SXM2 16 GB](ac922/qwen38-27b-ud-q4-k-xl-2x.md)
