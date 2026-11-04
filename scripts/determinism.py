#!/usr/bin/env python3
"""Byte-compare greedy output across server runs (#327). Standard library only.

  determinism.py run URL OUT.json [--key KEY] [--model NAME] [--logprobs] STEP...
      sends a fixed request sequence to a running server, one request at a time, and saves
      every reply's text (and token logprobs with --logprobs)
  determinism.py compare A.json B.json [C.json ...]
      per step, "same" or the first character where each run differs from A's (and the first
      differing logprob); exit status 1 if any text differs

STEP: chat       three chat prompts, 384 tokens each, after a 16-token warm-up
      long:N:M   one prompt of about N words-and-numbers (~1.3 tokens per N on Qwen3.8-Flash-Next:
                 long:7700 is ~10K tokens, long:94500 ~123K), M tokens out

Every request: temperature 0, seed 1, streamed. Restart the server between runs, with the same
flags. Output is byte-identical run to run with `legion serve --deterministic` (or
--prefix-cache-states 0 and, with drafts, --spec-fixed) as long as requests are sent one at a
time; see docs/serving.md, "Deterministic output".
"""
import json
import random
import sys
import time
import urllib.request

CHATS = [
    "Write a detailed 400-word explanation of how a CPU cache hierarchy works, with L1, L2, L3 "
    "and main memory.",
    "Write a Python module implementing an LRU cache class with get, put, resize and stats "
    "methods, with docstrings and type hints, then a pytest test file for it.",
    "List twenty major rivers of the world. For each, give its length, the countries it flows "
    "through and one historical fact about it.",
]


def stream(url, key, model, content, max_tokens, logprobs):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 2
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), headers)
    t0 = time.monotonic()
    first = last = None
    usage, text, lps = None, [], []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                for c in (ch.get("logprobs") or {}).get("content") or []:
                    top = [t.get("logprob") for t in c.get("top_logprobs") or []]
                    lps.append([c.get("token"), c.get("logprob")] + top)
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if piece:
                    now = time.monotonic()
                    first = first or now
                    last = now
                    text.append(piece)
    n_out = (usage or {}).get("completion_tokens") or 0
    n_in = (usage or {}).get("prompt_tokens")
    ttft = (first - t0) if first else None
    return {
        "prompt_tokens": n_in,
        "completion_tokens": n_out,
        "ttft_s": ttft,
        "prefill_tok_s": n_in / ttft if n_in and ttft else None,
        "decode_tok_s": (n_out - 1) / (last - first) if first and last and last > first and n_out > 1 else None,
        "text": "".join(text),
        "logprobs": lps,
    }


def long_prompt(n):
    rng = random.Random(n)
    words = ("the of and to in is was for on that with as by at from his her it an are were this "
             "be which or had not but").split()
    body = " ".join(rng.choice(words) + str(rng.randint(0, 999)) for _ in range(n // 3))
    return body + "\nSummarize in one line."


def run(argv):
    opts = {"--key": "", "--model": "default"}
    logprobs = False
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] in opts:
            opts[argv[i]] = argv[i + 1]
            i += 2
        elif argv[i] == "--logprobs":
            logprobs = True
            i += 1
        else:
            rest.append(argv[i])
            i += 1
    if len(rest) < 3:
        sys.exit(__doc__)
    url, out, steps = rest[0].rstrip("/"), rest[1], rest[2:]
    go = lambda content, n: stream(url, opts["--key"], opts["--model"], content, n, logprobs)
    runs = []
    for s in steps:
        if s == "chat":
            go("Hello", 16)
            for k, p in enumerate(CHATS):
                r = go(p, 384)
                r["step"] = "chat%d" % (k + 1)
                runs.append(r)
                print("%s: %d tokens, %.1f tok/s" % (r["step"], r["completion_tokens"], r["decode_tok_s"] or 0),
                      flush=True)
        elif s.startswith("long:"):
            _, n, m = s.split(":")
            r = go(long_prompt(int(n)), int(m))
            r["step"] = "long%s" % n
            runs.append(r)
            print("%s: %s prompt tokens, TTFT %.2f s (%.0f tok/s), %d out, %.1f tok/s" % (
                r["step"], r["prompt_tokens"], r["ttft_s"] or 0, r["prefill_tok_s"] or 0,
                r["completion_tokens"], r["decode_tok_s"] or 0), flush=True)
        else:
            sys.exit("unknown step " + s)
    with open(out, "w") as f:
        json.dump(runs, f, indent=1)


def compare(paths):
    if len(paths) < 2:
        sys.exit(__doc__)
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    a = runs[0]
    differ = False
    for i, r in enumerate(a):
        row = []
        for b in runs[1:]:
            t1, t2 = r["text"], b[i]["text"] if i < len(b) else ""
            if t1 == t2:
                row.append("same")
            else:
                differ = True
                k = next((j for j in range(min(len(t1), len(t2))) if t1[j] != t2[j]), min(len(t1), len(t2)))
                row.append("@%d" % k)
        for b in runs[1:]:
            la, lb = r.get("logprobs") or [], (b[i].get("logprobs") or []) if i < len(b) else []
            k = next((j for j in range(min(len(la), len(lb))) if la[j] != lb[j]), None)
            if k is not None:
                row.append("logprob@%d:%s|%s" % (k, la[k], lb[k]))
        print("%-10s %s" % (r["step"], " ".join(row)))
    return 1 if differ else 0


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "compare"):
        sys.exit(__doc__)
    if sys.argv[1] == "run":
        run(sys.argv[2:])
    else:
        sys.exit(compare(sys.argv[2:]))
