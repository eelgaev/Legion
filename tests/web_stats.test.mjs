// Run with: node --test tests/web_stats.test.mjs
// Only the pure helpers of web/js/stats.js; DOM work lives in chat.js.
import assert from "node:assert/strict";
import test from "node:test";
import { computeStats, formatStats, thinkingLabel } from "../web/js/stats.js";

const usage = { prompt_tokens: 12, completion_tokens: 8, total_tokens: 20 };

test("computeStats derives time to first token and tokens/s", () => {
  // No server timings: use client estimates.
  // TTFT = 200ms. Prompt = 12. Prefill = 12 / 0.2 = 60 tok/s.
  // Decode = 8 tokens / 1.0s = 8 tok/s.
  assert.deepEqual(computeStats(usage, 1000, 1200, 2200), {
    promptTokens: 12,
    completionTokens: 8,
    ttftMs: 200,
    tokensPerSec: 8,
    prefillPerSec: 60,
    decodePerSec: 8,
  });
});

test("computeStats uses server timings when provided", () => {
  const timings = { prompt_per_second: 1450, predicted_per_second: 31.5 };
  assert.deepEqual(computeStats(usage, 1000, 1200, 2200, timings), {
    promptTokens: 12,
    completionTokens: 8,
    ttftMs: 200,
    tokensPerSec: 8,
    prefillPerSec: 1450,
    decodePerSec: 31.5,
  });
});

test("computeStats is null without a usage chunk", () => {
  assert.equal(computeStats(null, 0, 10, 20), null);
});

test("computeStats nulls out what it cannot measure", () => {
  // No completion tokens: no rate.
  assert.equal(computeStats({ prompt_tokens: 3, completion_tokens: 0 }, 0, 5, 105).decodePerSec, null);
  // No token ever arrived: no ttft and no rate.
  assert.deepEqual(computeStats({ prompt_tokens: 3, completion_tokens: 2 }, 0, null, 105), {
    promptTokens: 3,
    completionTokens: 2,
    ttftMs: null,
    tokensPerSec: null,
    prefillPerSec: null,
    decodePerSec: null,
  });
  // First token and end in the same instant: no divide by zero.
  assert.equal(computeStats(usage, 0, 100, 100).decodePerSec, null);
});

test("formatStats shows the usage numbers", () => {
  assert.equal(
    formatStats({
      promptTokens: 1234,
      completionTokens: 8,
      ttftMs: 200,
      tokensPerSec: 31.5,
      prefillPerSec: 1450,
      decodePerSec: 31.5,
    }),
    "1,234 prompt · 8 completion · 0.2 s to first token · prefill 1,450 tok/s · decode 31.5 tok/s",
  );
});

test("formatStats shows a dash for missing measurements", () => {
  assert.equal(
    formatStats({
      promptTokens: 3,
      completionTokens: 0,
      ttftMs: null,
      tokensPerSec: null,
      prefillPerSec: null,
      decodePerSec: null,
    }),
    "3 prompt · 0 completion · — to first token · prefill — tok/s · decode — tok/s",
  );
});

test("thinkingLabel rounds to one decimal", () => {
  assert.equal(thinkingLabel(3.24), "Thought for 3.2 s");
  assert.equal(thinkingLabel(0.04), "Thought for 0 s");
});
