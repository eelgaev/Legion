# Live dashboard

A terminal dashboard of a running server: requests, throughput, link and memory bandwidth,
GPUs, speculative decoding, the KV and expert caches, the step breakdown, latency and the logs
(#1198). It runs in two ways:

```bash
# In the server's own terminal: the logs go to the dashboard's log panel.
legion serve -m model.gguf --tui

# From another terminal or machine, against a running server.
legion top                                # a local server on the default port (8090)
legion top --url http://192.0.2.10:8090
legion top --url https://192.0.2.10:8443 --insecure    # self-signed certificate
```

`q` closes the dashboard. Under `serve --tui` the server keeps running and logs to stdout again
(the status line stays off); Ctrl-C stops the server as it does without `--tui`. `--tui` needs
stdout to be a terminal: without one the server logs a warning and serves without the dashboard.

`legion top` reconnects by itself (0.5 s doubling to 10 s) and shows `disconnected: <why>`
while the server is away; `--interval-ms` (100-5000, default 250) sets how often the server
sends a snapshot.

## Keys

| Key | |
|---|---|
| `q` | quit (`serve --tui`: the server keeps running) |
| `p` | pause the screen (rates start over after it) |
| `Tab` / `Shift-Tab` | focus the next panel; in the compact layout the focused panel shows next to the request table |
| `?` | help, including why a value is n/a |
| `1`-`5` | the log level shown: ERROR ... TRACE |
| `/` | search the logs (Enter keeps the filter, Esc clears it) |
| `l` | the logs full screen |
| arrows, PgUp/PgDn, Home/End | scroll the logs |

Terminals of 120 x 46 and larger get the full layout (a row of link gauges and memory dials,
sparklines); smaller ones a compact one where the links panel has the same gauges and the
memory panel bars instead of dials. Under 36 rows the sparklines go, under 28 the step bar;
80 x 24 works.

## Access

The three routes behind the dashboard sit with every other `/v1` route: with `--api-key` set
they answer 401 without the key. `legion top` sends it from `--api-key` or `LEGION_API_KEY`:

```bash
LEGION_API_KEY=... legion top --url https://192.0.2.10:8443
```

The log stream carries the server's log lines only. Legion never logs prompt or completion
text, and the ring keeps other crates' lines at INFO and above only (their debug and trace
lines can hold request headers and bodies).

## Endpoints

| Method | Path | |
|---|---|---|
| `GET` | `/v1/stats` | one snapshot (JSON) |
| `GET` | `/v1/stats/stream?interval_ms=250` | a snapshot per interval as server-sent events (`interval_ms` 100-5000, default 250) |
| `GET` | `/v1/logs/stream?level=info` | the last 2,000 log lines, then new ones as they come, as server-sent events; `level` is a name (`warn`) or 1-5, default everything |

The snapshot (`version: 1`) holds cumulative counters, so a client turns two snapshots into
rates and a missed one loses nothing: `header` (model, quant from the file name, layout, GPUs,
context, sequences, drafter, version), `engine` (requests by state, tokens, KV blocks, prefix
and expert cache hits, speculative counts, model bytes by kind of step and by stage, step
phases), `requests` (one row per request in flight), `latency` (TTFT and ITL p50/p95 over five
minutes), `hw` (whether the hardware sampler runs, and why not), `gpus` (NVML per GPU with its
link and memory gauges) and `network` (the tensor-parallel exchange between machines). Values a
backend cannot report are `null`, never a made-up number. New fields can appear without a
version bump; a changed meaning bumps it.

## Cost

Nothing runs until someone asks. The first stats request (or `--tui`) starts an NVML sampler
(every 250 ms) and turns on the engine's few dashboard counters (per-step phase times, accepts
per draft position); 60 s after the last stats request both stop again. Without a client the
engine pays one relaxed atomic load per step (and the TP exchange one relaxed add per frame). `--hw-counters` keeps its own sampler, which the
dashboard then shares. The sampler only queries NVML; it never touches the GPUs' streams.

## Throughput and latency

- **Decode tok/s:** generated tokens per second between two snapshots; per request, the mean
  of the decoding requests' own rates.
- **Prefill tok/s:** prompt tokens over the engine's prefill time. A long prompt goes in one
  chunk per step, so prompt tokens per second of wall time jump between 0 and a spike; the
  dial keeps the last chunk's rate while a prompt is in flight. The 60 s and 10 min
  sparklines are wall-time rates.
- **ITL:** the gap between streamed text events, as in `/metrics`' inter-token histogram.
  With speculative decoding the accepted tokens of a step arrive back to back, so the p50 can
  sit near 0 ms and the p95 is closer to the step time.

## The gauges

Every link (one on a PCIe card, two per GPU on an AC922, one C2C on a GB10, plus TP net with
two Sparks) gets a title line and two solid bars, RX over TX:

```text
GPU0 NVLink CPU  / 75.0 GB/s  Legion 12.1  3 links NVLink 2
RX █████████▋▓▓▓14.5 GB/s  19%  pk 22.0░░░░░░░░░░░░░░
TX ██▏▓░░░░░░░░░░3.20 GB/s  4%  pk 6.10░░░░░░░░░░░░░░
```

(`░` stands for the empty part's dim background, which the terminal draws as a colour.)

Each bar fills to its own direction over the link's per-direction full scale, with the value
inside. Past the value a shaded "ghost" fill of the same hue, dimmer, runs on to the
direction's peak over the last 10 s (`pk`), then the empty part. The fill follows the load on a gradient from green through yellow (60%) and orange
(80%) to red (100%); terminals that announce 24-bit colour (`COLORTERM=truecolor`, or a `TERM`
such as kitty, ghostty, alacritty, foot or wezterm) get it in RGB, `*-256color` terminals in the
256-colour palette, others green under 60%, yellow under 85% and red above. Grey: no full
scale. The empty part is a dim background; the value reads dark over the filled part and in the
fill colour over the ghost and the empty part (16 colours: the ghost is the fill's colour,
dim). RX is cyan and TX magenta; the title's colour says the link's
type (NVLink green, PCIe blue, C2C red, TP net white).

RX and TX are the hardware counter's; Legion (yellow) is Legion's own bytes on the link: the model bytes the steps must move (weights
and experts streamed from host memory, host-tier KV), counted by the engine, not measured. A
link without a counter gets one bar of Legion's bytes (`L`) and `RX/TX n/a (<why>)` under it.
On narrow panels the title gets shorter instead of wrapping: the link's detail goes first, then
`Legion` becomes `L` and the name its short form (`G0 CPU / 75.0 GB/s L 12.1`), then the
unit, then the full scale; the value inside a bar drops `pk` first (the ghost fill stays),
then its unit and then its percentage. The bars of a column start in the same cell. Links go into
side-by-side columns where the width allows (a GPU's links stay together), and where they do
not fit with their titles (the compact layout with four GPUs) the title goes and each row starts
with a short name instead:

```text
G0 CPU RX  ▊8.00░░
G0 CPU TX  ▏2.00░░
```

The memory gauge is a speedometer (a Braille half dial) with a needle for the measured
bandwidth (cyan) and one for Legion's bytes (yellow), and a red tick across the arc at the
measured figure's 10 s peak (Legion's where nothing measures), also as `pk` after the figures;
the compact layout's memory panel shows the same as bars.

The same 10 s peak shows next to the throughput panel's decode and prefill tok/s
(`decode 41.2 tok/s  pk 48.0`) and as the `pk` columns after the GPU table's SM utilization
and power. A higher value takes the peak over at once and a lower one once the peak is 10 s
old; a server restart clears them.

| Gauge | Measured by | Full scale |
|---|---|---|
| PCIe | NVML PCIe throughput (TX/RX) | max generation x max width (`PCIe 4.0 x16`: 31.5 GB/s) |
| NVLink CPU | NVML NVLink data counters over the links to the CPU | links x the NVLink version's speed (NVLink 2: 25 GB/s per link) |
| NVLink peer | the same over the links to other GPUs | the same |
| C2C | GPM C2C throughput where NVML has it | NVML's C2C links x speed, or the Grace parts' 450 GB/s |
| TP net | Legion's own exchange frames (no NIC counter) | the fastest active RDMA port or network interface (sysfs) |
| Memory | GPM DRAM bandwidth utilization x peak | bus width x max memory clock, or the part's spec |

What each machine shows:

- **An AC922 (4x V100):** an NVLink CPU and an NVLink peer gauge per GPU, both measured.
  Legion's figure on the CPU link is the experts and KV it streams from host memory. No memory
  dial: V100 has no GPM, and NVML's memory utilization is the share of time memory was busy,
  not bandwidth, so the GPU panel shows it as `mem %`.
- **A DGX Spark (GB10):** one C2C gauge (full scale from NVML: 8 links x 43.1 GB/s) and a
  memory dial against 273 GB/s (LPDDR5X). Neither has a hardware counter: the GB10 has no GPM
  (`nvidia-smi dmon --gpm-metrics 1,2,10` prints `-` for graphics activity, SM utilization and
  DRAM bandwidth under a full decode), no
  C2C counter, and NVML's memory utilization reads 0% while decode moves the whole bandwidth,
  so the dashboard shows it as n/a. GB10 has no CPU DRAM PMU either, so nothing counts the
  shared memory's traffic. What the memory dial does show is Legion's own bytes: the model
  bytes per step over the step time, which sits at about 270 GB/s during decode there. The
  C2C gauge reads `RX/TX n/a (no C2C counter)` and its bar and Legion figure count
  host-tier bytes, which a single Spark does not have (it is all one memory), so they stay at
  0. VRAM is `unified`. With two Sparks a TP net gauge shows
  the exchange's bytes against the NIC's speed.
- **A PCIe card (RTX 4090 and the like):** a PCIe gauge measured by NVML, with Legion's host
  bytes; no memory dial without GPM (Hopper and newer have it).

## Why a value is n/a

- **Hardware panels:** NVML is missing or sees no GPU; the reason shows in place.
- **PCIe TX/RX:** NVML has no PCIe throughput counter on this GPU.
- **C2C TX/RX:** no GPM C2C counter; the bar shows Legion's bytes and the label says
  `no C2C counter`.
- **Memory:** no GPM. NVML's memory utilization is time busy, not bandwidth; on a GB10 it
  reads 0 under any load, so the GPU panel's `mem` column is n/a there.
- **VRAM:** unified memory has no VRAM of its own.
- **Quant:** the model's file and directory names do not say.
- **Warm sequences:** the engine does not count them yet.
- **Step bar:** the engine thread's time by phase. Attention and the TP exchange sit inside
  `step`: there are no device timers. While a long prompt prefills, a step is one prompt chunk,
  so the bar reads in seconds rather than milliseconds.
- **TP net:** Legion's own frames only; no NIC counters.
