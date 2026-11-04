# Memory bandwidth

Decode on a MoE model is bound by memory bandwidth: every step reads the weights, the routed
experts and the KV cache. Legion counts those bytes and divides them by the time the steps take,
so you can see how close a run is to the device's limit and where the bytes come from (#391).

There are two kinds of numbers:

- **Model bytes (analytic, always on).** These are the bytes the steps *had to read*, counted
  from the graphs and the routes. They are not a measurement of the hardware.
- **Hardware counters (opt-in).** These are what the GPUs and the host actually moved. Turn
  them on with `legion serve --hw-counters`. Use the profilers below for exact DRAM bytes.

## Model bytes

For each step, the executor adds up the following:

| kind | what is counted | read from |
|---|---|---|
| `dense` | Every weight tensor the graph reads, counted once per step. For the embedding table, only the rows that were looked up. | device memory |
| `experts_upper` | Experts that live inside the weights (no expert cache). Upper bound: assumes every routed choice is a different expert, capped at the expert stack. | device memory |
| `experts_device`, `experts_host` | Experts of an expert cache (`--experts cache`). For passes the route log covers (256 tokens or fewer), the distinct experts each layer routed to, split into cache slots (VRAM) and host experts read in place. For longer passes, at most one read per routed choice (capped at the number of experts), split by the share of ready slots. A gathered pass (`Fetch::Copy`) counts every pair. | VRAM / host memory |
| `staged` | Expert copies into VRAM staging buffers during prefill staging. They cross the link once; the pass then reads them from VRAM, so they are also counted in `experts_device`. | link |
| `kv`, `kv_host` | The K and V attention reads, per sequence: full layers read every position, sliding layers read `window + tokens`, and paged sparse (QSA) layers read at most `tokens x (top_k + pool)` plus the pool keys. With a KV host tier, the share of the sequence's blocks that sit in the tier is counted as `kv_host`. | VRAM / host memory |

Activations, KV writes and L2 hits are left out. The figure is the traffic the model forces,
which is the right numerator for a roofline.

Totals are kept by kind of step:

- `prefill`: steps with a sequence of more than 16 tokens.
- `decode`: decode and verify steps.
- `draft`: MTP draft passes.

They are also kept per pipeline stage, together with each stage's busy time (launch to
synchronized).

- **Device GB/s** is device-memory bytes (`dense + experts_upper + experts_device + kv`) over
  the steps' time.
- **Link GB/s** is host-link bytes (`experts_host + kv_host + staged`) over the same time.
- **Roofline share** is device GB/s over the first stage's roofline. Under `--tp N`, a stage's
  roofline is N times the device's.

Counting costs no GPU syncs and no work per op:

- Graph weight bytes are computed when a graph is built.
- KV bytes come from the sequence lengths.
- Expert bytes come from the route log the cache already reads after each pass.

### Where the numbers show up

- **`/metrics`**:
  - `legion_engine_model_bytes{phase}`, `_seconds{phase}`, `_gb_per_second{phase}` (the total over device and host memory)
  - `legion_engine_model_kind_bytes{phase,kind}`
  - `legion_engine_model_memory_gb_per_second{phase}`, `legion_engine_model_link_gb_per_second{phase}`
  - `legion_engine_model_roofline_ratio{phase}`
  - `legion_engine_roofline_memory_gb_per_second{stage}`, `legion_engine_roofline_link_gb_per_second{stage}`
  - per stage: `legion_engine_stage_model_kind_bytes{phase,stage,kind}`, `legion_engine_stage_busy_seconds{phase,stage}`, `legion_engine_stage_memory_gb_per_second{phase,stage}`, `legion_engine_stage_roofline_ratio{phase,stage}`
- **The `request done` log line**:
  - `prefill_model_gb`, `prefill_model_gb_s`, `prefill_mem_gb_s`, `prefill_mem_roofline_pct`, `prefill_link_gb_s`
  - the same fields with a `decode_` prefix. Decoding includes the draft passes.
- **The live status line**: `mem 187 GB/s (80%)`. This is device GB/s over the steps of the last 5 seconds, with its share of the roofline.
- **`legion bench --json`**: each level's `model_bytes`, per phase, with `gb`, `device_gb`, `link_gb`, `experts_gb`, `kv_gb`, `device_gb_s`, `link_gb_s` and `roofline_pct`.

## Rooflines

The rooflines are measured on the project's machines, not taken from datasheets
(`src/hw/roofline.rs`):

| device | device memory read | host link per GPU | how it was measured |
|---|---|---|---|
| Tesla V100-SXM2 16 GB (AC922, NVLink 2.0) | 894 GB/s | 68.5 GB/s | `hostbw` (2026-10-19, an AC922 with 4x V100): a read-only loop over 4 GiB of device memory, best of 5 at 8-16 CTAs per SM, 893-894 GB/s on GPUs 0 and 2. Pinned host memory on the GPU's own NUMA node read in place: 68.5 GB/s per GPU at every CTA count and block size. Two GPUs of one socket reach 133-135 GB/s together; the other socket's memory gives 46-56 GB/s. |
| NVIDIA GB10 (DGX Spark) | 233 GB/s | none (one memory pool) | `mvbench` (2026-10-19, a DGX Spark): a copy loop over device memory, 233-234 GB/s. |

To add a device:

1. Measure a read-only device-memory loop at several CTAs per SM, and in-place reads of pinned
   host memory if the device has a host link. Use the best of 5 runs.
2. Save the run (`legion bench ... --save <dir>`).
3. Add an entry to `TABLE` with the GPU name substring and a `measured` note.

Unknown devices show GB/s without a percentage.

## Hardware counters

`legion serve --hw-counters` (sampling interval `--hw-counters-interval-ms`, default 200)
starts a thread that reads NVML for each GPU named in `CUDA_VISIBLE_DEVICES`. It does nothing
on the GPUs, and without the flag no thread starts. It exports:

- `legion_gpu_nvlink_rx_bytes{gpu}` and `_tx_bytes`: NVLink data over every link, from NVML
  fields 138/139 per link, which are cumulative KiB. Host-memory reads by the GPU arrive as
  received data. Peer traffic counts too.
- `legion_gpu_pcie_rx_bytes` and `_tx_bytes`: NVML PCIe throughput samples integrated over time.
- `legion_gpu_memory_busy_seconds` and `legion_gpu_sm_busy_seconds`: NVML utilization
  integrated over time. This is the share of time the memory was being read or written, not
  bytes.
- `legion_gpu_sampled_seconds`.

The `gpu` label is the NVML index (PCI order), not the CUDA ordinal.

## `legion bench --workload bandwidth`

```sh
legion serve ... --hw-counters &
legion bench --workload bandwidth --prompt-tokens 123000 --gen-tokens 256 --concurrency 1 \
    --model <name> --save results/<date>_<model>_bandwidth
```

The workload runs these phases:

1. A warm-up.
2. A prefill of one prompt of about `--prompt-tokens` tokens, with `max_tokens` 1 (the same
   synthetic prompt as `--workload prefill`).
3. A decode phase of `--concurrency` parallel requests of the same prompt, each generating
   `--gen-tokens` tokens at temperature 0 with `ignore_eos`. Decode therefore reads the KV of
   a `--prompt-tokens` context, and with a KV host tier the tier's share of it. The prefix
   cache (when on) skips the first request's prefill; the others prefill in the decode window,
   but only decode steps and draft passes count there. The decode rate is over the time after
   the phase's first token. The server's context must hold `--concurrency` sequences of that
   length.

It scrapes `/metrics` before and after each phase. For each phase it prints the analytic model
bytes (GB, of which experts, KV and link; device GB/s and the roofline share; link GB/s), and
under them each GPU's counters for the same window. `--json` and `--save` write the same report.

The two sides measure different windows:

- The analytic GB/s divide by the steps' own time.
- The hardware counters cover the whole window, including the HTTP and tokenizer time
  between steps.

## Exact DRAM bytes on a V100: nvprof / Nsight Compute

NVML has no DRAM byte counter. To check the analytic numbers against DRAM traffic, profile a
short run. Profiling replays every kernel, so time it separately; use the profile for bytes
only.

```sh
# CUDA 12.4 on the AC922 ships both; Nsight Compute 2024.1 still lists gv100.
CUDA_VISIBLE_DEVICES=2 nvprof --metrics dram_read_bytes,dram_write_bytes --csv \
    --log-file dram.csv legion run --model <gguf> --device cuda --temperature 0 --max-tokens 8 "<prompt>"
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum --csv \
    legion run --model <gguf> --device cuda --temperature 0 --max-tokens 8 "<prompt>" > dram.csv
```

To compare with `model_bytes` for the same steps:

1. Sum `dram_read_bytes` over the kernels of one decode step.
2. Read the `model_bytes` of the same step from the request-done line, using
   `decode_model_gb` divided by the number of steps.

The DRAM bytes come out higher, because they also include activations, KV writes and the reads
of the workspace.

## Host memory on POWER9: nest counters

On the AC922, `perf` exposes the memory controllers' nest counters. The PMUs are
`nest_mba0_imc` to `nest_mba7_imc`, with the events `PM_MBA<i>_READ_BYTES` and
`PM_MBA<i>_WRITE_BYTES`. Each counter is per chip: socket 0 is read through CPU 0 and
socket 1 through CPU 80. Run as root:

```sh
EV=$(for i in 0 1 2 3 4 5 6 7; do printf "nest_mba${i}_imc/PM_MBA${i}_READ_BYTES/,"; done)
perf stat -a --per-socket -x, -e "${EV%,}" sleep 10   # while the bench runs
```

Sum the eight MBAs per socket.

The counts are not bytes, and their unit is not settled:

- A CPU calibration on the AC922 read 32 GiB from node 0 with one Python thread and counted
  1.12 G on socket 0, about 29 bytes per count.
- Under GPU reads of host experts, 64 bytes per count matches NVLink: a 95K-token prefill and
  256 decoded tokens counted 43.3 G on socket 0. At 64 bytes that is 2.77 TB, against 2.88 TB
  received over the two GPUs' NVLinks in the same window. A run at 8K tokens came to 120% of
  NVLink; the counters also see the CPU's own traffic.

Calibrate with a known read before trusting absolute figures. Differences between two runs
need no calibration.

GPU reads of host experts and of the KV tier land here: on node 8 for GPUs 2 and 3, and on
node 0 for GPUs 0 and 1. The `nest_nvlink*_imc` PMUs count NVLink flits per brick.

## Acceptance runs

Saved outside this repository:

- 2026-10-20, Qwen3.8-Flash-Next: AC922, 2 V100s,
  Qwen3.8-Flash-Next UD-Q4_K_XL, `--experts cache`, MTP k=2:
  - Prefill and decode before and after this change.
  - The bandwidth workload with `--hw-counters`.
  - Results:
    - Prefill of 95K tokens (123K with the chat template) is bound by the host link. It
      stages 2.39 TB of experts at 67 GB/s, against 2.72 TB received over NVLink, and reads
      device memory at 72 GB/s (8%).
    - Decode at 95K tokens reads device memory at 157-178 GB/s (18-20%) and the link at
      20 GB/s.
- 2026-10-20, AC922: Gemma 4 31B with MTP k=3. Two
  configurations: `--tp 2` without a KV host tier, and a layer split with one. The tier is not
  supported under `--tp` yet.
  - `--tp 2`: decode at 18K tokens reads 553 GB/s, 31% of two V100s.
  - The tier at 75K tokens:
    - Decode reads device memory at 383 GB/s (43%).
    - It reads 247 GB of KV from the tier at 31 GB/s, against 295 GB received over NVLink.
- Not run yet:
  - The GB10 run: Qwen3.8-Flash-Next, prefill 8K and decode with MTP k=4.
  - The 4-GPU AC922 run.

  Both use the same `legion bench --workload bandwidth` command.
