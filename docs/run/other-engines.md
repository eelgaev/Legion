# llama.cpp and vLLM next to Legion

[Back to the performance tables](../../README.md#performance). How the runs were measured: [README.md](README.md).

The commands behind the llama.cpp and vLLM rows of the README's tables, model by model, next to Legion's. Every
engine is measured by the same client on the same machine: the deterministic client at 1, 4 and 16 users
(8 fixed prompts, temperature 0, `ignore_eos`, 256 tokens, 2 rounds; per-request decode at 1 user, total at 4
and 16) and one prefill request per prompt size. Prefix caching is off in all three (`--prefix-cache-states 0`,
`--no-cache-prompt`, `--no-enable-prefix-caching`) and all three serve 16 sequences.

- **llama.cpp:** b12062 (upstream `29c720e`, 2026-11-02), a stock CUDA build (`GGML_CUDA=ON`; AC922: `CMAKE_CUDA_ARCHITECTURES=70`,
  `GGML_CUDA_FORCE_MMQ=ON`, NCCL; DGX Spark: `sm_121a`). Its MTP drafters are the standalone `mtp-*.gguf` files
  of the same Hugging Face repo; the `*-shared-*` variants have no token embedding and do not load in llama.cpp.
  On the AC922 the flags and environment are those of a tuned daily-use AC922 setup (unified memory, P2P, tensor
  split, NUMA binding), changed only to 16 slots and no prompt cache.
- **vLLM:** 0.33.0 on the DGX Spark, the same safetensors checkpoints and DFlash 2 speculators as Legion.
- **Prefill with llama.cpp and Gemma:** Gemma's first generated token carries no visible text, so the client
  cannot time it; the numbers are llama.cpp's own `prompt eval time` for the same requests.

AC922 runs bind the process to the GPUs' NUMA node (`numactl --cpunodebind=8 --membind=8` for GPUs 2 and 3).

## Qwen3.8-27B

Legion, UD-Q4_K_XL GGUF, 1x DGX Spark: [qwen38-27b-ud-q4-k-xl-1x.md](spark/qwen38-27b-ud-q4-k-xl-1x.md).

Legion, UD-Q4_K_XL GGUF, 2x V100: [qwen38-27b-ud-q4-k-xl-2x.md](ac922/qwen38-27b-ud-q4-k-xl-2x.md).

llama.cpp, 1x DGX Spark:

```sh
llama-server -m Qwen3.8-27B-UD-Q4_K_XL.gguf --port 8090 --jinja \
  -ngl 99 -fa on -np 16 -kvu --no-cache-prompt -c 262144 \
  -md mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 3
```

llama.cpp, 2x V100:

```sh
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 GGML_CUDA_P2P=1 GGML_CUDA_REGISTER_HOST=1 GGML_OP_OFFLOAD_MIN_BATCH=1 GGML_CUDA_GRAPH_OPT=1 \
llama-server -m Qwen3.8-27B-UD-Q4_K_XL.gguf --port 8090 --jinja \
  -ngl 99 --keep -1 -fa on -np 16 -kvu --no-cache-prompt -c 262144 --numa numactl --poll 0 \
  -t 8 -tb 40 -Cr 80-159 --cpu-strict 1 --prio 2 --load-mode none \
  -ctk q8_0 -ctv q8_0 -sm tensor -ub 256 -b 2048 \
  -md mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 3
```

Legion NVFP4 + DFlash 2: [qwen38-27b-nvfp4-1x.md](spark/qwen38-27b-nvfp4-1x.md). vLLM, 1x DGX Spark:

```sh
vllm serve unsloth/Qwen3.8-27B-NVFP4 --port 8090 \
  --max-num-seqs 16 --max-model-len 262144 --no-enable-prefix-caching \
  --gpu-memory-utilization 0.85 --trust-remote-code \
  --speculative-config.method dflash --speculative-config.model z-lab/Qwen3.8-27B-DFlash2 \
  --speculative-config.num_speculative_tokens 7
```

## Gemma 4 31B

Legion: [1x DGX Spark](spark/gemma-4-31b-qat-ud-q4-k-xl-1x.md), [2x DGX Spark](spark/gemma-4-31b-qat-ud-q4-k-xl-2x-tp.md),
[2x V100](ac922/gemma-4-31b-qat-ud-q4-k-xl-2x.md), [NVFP4 + DFlash 2](spark/gemma-4-31b-nvfp4-1x.md).

llama.cpp, 1x DGX Spark:

```sh
llama-server -m gemma-4-31B-it-qat-UD-Q4_K_XL.gguf --port 8090 --jinja \
  -ngl 99 -fa on -np 16 -kvu --no-cache-prompt -c 262144 \
  -md mtp-gemma-4-31B-it-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 3
```

llama.cpp, 2x V100:

```sh
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 GGML_CUDA_P2P=1 GGML_CUDA_REGISTER_HOST=1 GGML_OP_OFFLOAD_MIN_BATCH=1 GGML_CUDA_GRAPH_OPT=1 \
llama-server -m gemma-4-31B-it-qat-UD-Q4_K_XL.gguf --port 8090 --jinja \
  -ngl 99 --keep -1 -fa on -np 16 -kvu --no-cache-prompt -c 262144 --numa numactl --poll 0 \
  -t 8 -tb 40 -Cr 80-159 --cpu-strict 1 --prio 2 --load-mode none \
  -ctk f16 -ctv q8_0 -sm tensor -fit off -ub 2048 -b 2048 \
  -md mtp-gemma-4-31B-it.gguf --spec-type draft-mtp --spec-draft-n-max 3
```

vLLM, 1x DGX Spark:

```sh
vllm serve RedHatAI/gemma-4-31B-it-NVFP4 --port 8090 \
  --max-num-seqs 16 --max-model-len 262144 --no-enable-prefix-caching \
  --gpu-memory-utilization 0.85 --trust-remote-code --language-model-only \
  --speculative-config.method dflash --speculative-config.model RedHatAI/gemma-4-31B-it-speculator.dflash2 \
  --speculative-config.num_speculative_tokens 7
```

## Qwen3.8-Flash-Next

Legion: [1x DGX Spark](spark/qwen38-flash-next-ud-q4-k-xl-1x.md), [2x DGX Spark](spark/qwen38-flash-next-ud-q4-k-xl-2x-tp.md),
[2x V100](ac922/qwen38-flash-next-ud-q4-k-xl-2x.md), [4x V100](ac922/qwen38-flash-next-ud-q4-k-xl-4x.md),
[NVFP4](spark/qwen38-flash-next-nvfp4-1x.md), [NVFP4 2x DGX Spark](spark/qwen38-flash-next-nvfp4-2x-tp.md).

llama.cpp, 1x DGX Spark. `--lazy-mode on` keeps the Engram table on the SSD (the default, `auto`, turns lazy loading
off on the GB10 and the weights then do not fit). Adding the drafter (`-md mtp-Qwen3.8-Flash-Next-Q8_0.gguf
--spec-type draft-mtp --spec-draft-n-max 2`) makes b12062 abort on the first request (`GGML_ASSERT(buffer)`),
so the README row is without drafting:

```sh
llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --port 8090 --jinja \
  -ngl 99 -fa on -np 16 -kvu --no-cache-prompt -c 131072 --lazy-mode on
```

llama.cpp, 2x V100, 1 user and prefill (one sequence as in the daily-use setup):

```sh
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 GGML_CUDA_P2P=1 GGML_CUDA_REGISTER_HOST=1 GGML_OP_OFFLOAD_MIN_BATCH=1 GGML_CUDA_GRAPH_OPT=1 \
CUDA_SCALE_LAUNCH_QUEUES=16x GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1 \
numactl --cpunodebind=8 --membind=8 \
llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --port 8090 --jinja \
  -ngl 99 --keep -1 -fa on -np 1 --no-cache-prompt -c 200000 --numa numactl --poll 0 --load-mode none \
  -ctk q8_0 -ctv q8_0 -t 8 -tb 40 -Cr 80-159 --cpu-strict 1 --prio 2 -ts 11,11 \
  -md mtp-Qwen3.8-Flash-Next-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2
```

llama.cpp, 2x V100, 4 users (16 sequences; `--lazy-mode on` reads the Engram table from disk on demand; GPUs 0 and 1,
so NUMA node 0):

```sh
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 GGML_CUDA_P2P=1 GGML_CUDA_REGISTER_HOST=1 GGML_OP_OFFLOAD_MIN_BATCH=1 GGML_CUDA_GRAPH_OPT=1 \
CUDA_SCALE_LAUNCH_QUEUES=16x GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1 \
numactl --cpunodebind=0 --membind=0 \
llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --port 8090 --jinja \
  -ngl 99 --keep -1 -fa on -np 16 -kvu --no-cache-prompt -c 262144 --numa numactl --poll 0 \
  -t 8 -tb 40 -Cr 0-79 --cpu-strict 1 --prio 2 --lazy-mode on -ctk q8_0 -ctv q8_0 -ts 11,11 \
  -md mtp-Qwen3.8-Flash-Next-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2
```

llama.cpp, 4x V100 (one sequence, as the daily-use 4-GPU setup):

```sh
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 GGML_CUDA_P2P=1 GGML_CUDA_REGISTER_HOST=1 GGML_OP_OFFLOAD_MIN_BATCH=1 GGML_CUDA_GRAPH_OPT=1 \
CUDA_SCALE_LAUNCH_QUEUES=16x GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1 \
llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --port 8090 --jinja \
  -ngl 99 --keep -1 -fa on -np 1 --no-cache-prompt -c 200000 --numa numactl --poll 0 --load-mode none \
  -ctk q8_0 -ctv q8_0 --lazy-mode on -sm tensor -fit off -b 512 -ub 128 -ts 0.85,1,1,1 \
  -md mtp-Qwen3.8-Flash-Next-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2
```

vLLM, 1x DGX Spark, with the checkpoint's MTP head (it loads in 0.33.0). Stock vLLM keeps the 47.7 GiB FP8 n-gram
table in pinned host memory, which with the 75.6 GiB of weights does not fit the 121 GB DGX Spark; the run used a
local benchmark-only patch that maps the table from its safetensors file and reads the rows from there (switched on
by an environment variable, no effect without it):

```sh
vllm serve nvidia/Qwen3.8-Flash-Next-NVFP4 --port 8090 \
  --max-model-len 262144 --gpu-memory-utilization 0.85 --no-enable-prefix-caching --max-num-seqs 16 \
  --limit-mm-per-prompt '{"image":0,"video":0}' --skip-mm-profiling \
  --speculative-config.method mtp --speculative-config.num_speculative_tokens 2
```

On a first start FlashInfer builds its NVFP4 MoE kernels (about 12 minutes) after the weights are loaded; with the
weights resident that build can run the DGX Spark out of host RAM, so build it once before the first start, in the
same environment:

```sh
python -c 'from flashinfer.fused_moe.core import get_cutlass_fused_moe_module as m; m("120")'
```

vLLM, 2x DGX Spark, tensor parallel 2 with the checkpoint's MTP head (the multi-node `mp` backend, no Ray; NCCL
over RoCE on the QSFP link), with torch.compile and CUDA graphs. `--gpu-memory-utilization 0.65`: 0.33.0
compiles at this setting with 27 GiB of host RAM left (0.29.0 ran out at 0.85 and at 0.65). Build FlashInfer's
MoE kernels on both Sparks first, as for one Spark.

```sh
# both ranks
export HF_HUB_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  NCCL_SOCKET_IFNAME=<qsfp-if> GLOO_SOCKET_IFNAME=<qsfp-if> TP_SOCKET_IFNAME=<qsfp-if> NCCL_IB_HCA=<roce-dev> \
  VLLM_HOST_IP=<this-rank-ip>
# rank 0
vllm serve nvidia/Qwen3.8-Flash-Next-NVFP4 --port 8090 \
  --tensor-parallel-size 2 --nnodes 2 --node-rank 0 --master-addr <rank0-ip> --master-port 29500 \
  --distributed-executor-backend mp \
  --max-num-seqs 16 --max-model-len 262144 --no-enable-prefix-caching \
  --gpu-memory-utilization 0.65 --trust-remote-code \
  --limit-mm-per-prompt '{"image":0,"video":0}' --skip-mm-profiling \
  --speculative-config.method mtp --speculative-config.num_speculative_tokens 2
# rank 1: the same with --node-rank 1 --headless (no --port)
```

llama.cpp, 2x DGX Spark: a build with `-DGGML_RPC=ON` (otherwise as above), `rpc-server` on the second Spark and
llama-server on the first with `--rpc` (layer split; the weights are sent over RPC at load):

```sh
# second Spark
rpc-server -H <rpc-ip> -p 50052
# first Spark
llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --port 8090 --jinja \
  -ngl 99 -fa on -np 16 -kvu --no-cache-prompt -c 262144 --lazy-mode on --rpc <rpc-ip>:50052
```

## GLM-5.3-Flash Maya-S v2

Legion: [1x DGX Spark](spark/glm-53-flash-maya-s-v2-iq2-xxs-1x.md), [4x V100](ac922/glm-53-flash-maya-s-v2-iq2-xxs-4x.md).

llama.cpp, 1x DGX Spark (64K context: 16 sequences at 262K do not fit next to the 96.5 GB of weights):

```sh
llama-server -m GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf --port 8090 --jinja \
  -ngl 99 -fa on -np 16 -kvu --no-cache-prompt -c 65536 \
  --spec-type draft-mtp --spec-draft-n-max 2
```

Project Maya, 4x V100: [Project Maya](https://github.com/mw00/project-maya) v1.1.51 (MIT), the engine its Maya-S
GGUFs are made for, set up by its own installer with its recommended settings (MTP drafting on, the GGUF's NextN
layer, one draft; one sequence, `--parallel 1`, its default). On ppc64le the installer's CPU check needs a one-line
fix (`/proc/cpuinfo` has no `flags` line there); nothing else was changed. Setup (builds the engine for sm_70 and
writes the expert pack, the config and the start script):

```sh
./maya.sh --setup --yes --no-start --plain --gguf-dir <gguf dir> --gpus 0,1,2,3 --no-vision \
  --context 40960 --llama-dir third_party/llama.cpp --nvcc /usr/local/cuda/bin/nvcc --host-compiler $(which g++) \
  --no-speed-calibration --drafting on
```

Server (the config the setup wrote: engine args `--glm-pack <gguf dir>/pack --max-context 40960 --expert-cache auto
--prefill auto --kv-streaming on --expert-profile data/expert-profile-glm.bin`, GPUs 0-3, draft `mtp`; no extra
environment):

```sh
.venv/bin/python -m serve.server --engine maya --config maya-maya-s-v2-iq2_xxs.json --port 8090
```

With `"--parallel", "4"` in the engine args (Maya's setting for several requests at once) every request failed on
16 GB V100s (`glm spec: the recurrent-state backup did not allocate`), so the 4-user number is the `--parallel 1`
server answering 4 clients one at a time, and 16 users were not measured.

## GLM-5.3-Flash UD-Q4_K_XL

Legion: [4x V100](ac922/glm-53-flash-ud-q4-k-xl-4x.md).

llama.cpp, 4x V100: the 16-sequence 2x V100 command above (environment and CPU flags) with the 4-GPU split of the
4x V100 command and no `numactl` binding (the 4 GPUs span both sockets).
`-sm tensor` stops at load (`LLAMA_SPLIT_MODE_TENSOR not implemented for architecture 'glm5-next'`), so the
layers are split (`-sm layer`). The MTP layer is inside the GGUF. The 141 GB of weights run from unified memory
past the GPUs' 64 GB. C=1, C=4 and prefill each in a 20-minute session of their own (load about 6.5 minutes): C=1 3.6 per request, C=4
3.4 total, prefill 8K 25 tok/s; the 32K request was half done when the session ended (25 tok/s throughout):

```sh
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 GGML_CUDA_P2P=1 GGML_CUDA_REGISTER_HOST=1 GGML_OP_OFFLOAD_MIN_BATCH=1 GGML_CUDA_GRAPH_OPT=1 \
CUDA_SCALE_LAUNCH_QUEUES=16x GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1 \
llama-server -m GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf --port 8090 --jinja \
  -ngl 99 --keep -1 -fa on -np 16 -kvu --no-cache-prompt -c 262144 --numa numactl --poll 0 \
  -t 8 -tb 40 -Cr 80-159 --cpu-strict 1 --prio 2 --load-mode none \
  -ctk q8_0 -ctv q8_0 --lazy-mode on -sm layer -fit off -b 512 -ub 128 -ts 0.85,1,1,1 \
  --spec-type draft-mtp --spec-draft-n-max 2
```
