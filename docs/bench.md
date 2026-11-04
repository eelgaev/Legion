# `legion bench`

`legion bench` is a client: it drives a running OpenAI- (or Anthropic-) compatible server,
Legion or another engine, and reports what it measured. `--save <dir>` writes the JSON, a
`meta.json` (command line, date, the server's `/health`) and a `README.md` with the table.

| `--workload` | What it measures |
|---|---|
| `bench` (default) | A concurrency sweep (`--concurrency 1,8,32,128`): throughput, per-user decode rate, TTFT, ITL. |
| `chats` | Three fixed chat prompts after a warm-up, with their texts (for output hashes). |
| `prefill` | One synthetic prompt per `--prompt-tokens N[,N...]`, `max_tokens` 1: prefill tok/s. |
| `bandwidth` | Model bytes per phase against the device roofline; Legion only (reads `/metrics`). See [bandwidth.md](bandwidth.md). |
| `mixed` | Chat latency under background long prompts (below). |

Every workload sends `temperature: 0` explicitly, so a server whose default is sampling still
runs greedy.

## `--workload mixed`

The metric of #51: chat inter-token latency while long prompts are prefilled in the
background. The target is chat p99 ITL at most 2 x the p50.

```sh
legion bench --workload mixed --url http://localhost:8090 --model <id> \
    --chat-streams 4 --long-prompt-tokens 16384 --long-every 10 --duration 120 \
    --save results/<date>_<model>_mixed
```

- **Chat class:** `--chat-streams` clients (default 4), each sending streaming chat requests
  back to back: `--chat-prompt-tokens` (default 512) in, `--chat-max-tokens` (default 256) out.
- **Long class:** prompts of `--long-prompt-tokens` (default 16384) arriving every
  `--long-every` seconds (default 10, the first one period after the start), or as a Poisson
  process with `--long-rate <per second>`; each generates `--long-max-tokens` (default 64).
  They are open-loop: a long prompt is sent on schedule even while earlier ones still run.
  `--long-every 0` (or `--long-rate 0`) runs the chats alone, the baseline to compare with.
- **Time:** `--warmup` seconds (default 10) of load, then `--duration` measured seconds
  (default 120). Requests that start in the warm-up are left out of the stats. Chat streams
  start no request after warm-up + duration; requests still running then finish and count.
- **Texts:** the `--workload prefill` format (`word123` items, about 3 tokens each), a
  different text per request so no prefix cache hits, all from `--seed` (default 0). The
  Poisson schedule follows `--seed` too, so a run is repeatable against other engines.
- **Requests:** plain OpenAI `/v1/chat/completions` streaming with `--model` and the optional
  `--api-key`, `temperature: 0`, `ignore_eos` with a fixed `max_tokens`, and
  `stream_options.include_usage`. `--priority-chat <int>` and `--priority-long <int>` set the
  request's `priority` field (lower first, as SGLang and vLLM read it); it is not sent when
  unset. Nothing reads `/metrics`, so SGLang and vLLM are measured the same way.
- **Declared lengths:** `--declared-max-tokens N` sends `max_tokens` N on every request and
  hangs up once the class's own output length has streamed (the server cancels the rest), as
  an agent asking for 32K tokens whose reply stops after a few hundred. It shows what a server
  reserves for `max_tokens` at admission (Legion's `--kv-admission`, #627). Those requests
  send no `ignore_eos` (a reply may stop earlier at its own end: past an ignored end of
  sequence a model can go on with tokens that stream no text, and the client would never hang
  up), and their output tokens are the streamed deltas, not `usage`.

The report, per class (table; `--json` prints the same as `mixed.json` in `--save`):

| Column | Meaning |
|---|---|
| ok/fail | Measured requests. |
| out tok/s | The class's output tokens over the measured window. |
| strm t/s | The median request's decode rate (tokens after the first over the time after the first). |
| TTFT p50 / p90 / p99 | Time to the first streamed token, ms. |
| ITL p50 / p90 / p99 / max | Gaps between consecutive streamed deltas, ms, over all requests of the class. |

and the line `chat ITL p99 / p50`, #51's number. A delta that carries several tokens (a
speculative decoding step) is one gap, as in `--workload bench`; the output token counts come
from the server's `usage`.
