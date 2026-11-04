// Pure helpers for the per-message stats line and the thinking label (used by chat.js).
// All timing values are performance.now() milliseconds measured by the client.

const whole = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const oneDp = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const twoDp = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });

const round = (x, dp) => (Number.isFinite(x) ? Number(x.toFixed(dp)) : null);

/**
 * Stats for one reply, from the final usage chunk and client timing.
 * `usage` is the last chunk's `usage` object (null if it never arrived);
 * `startMs` when the request was sent, `firstTokenMs` the first reasoning or content
 * delta (null if none), `endMs` when the usage chunk arrived, `timings` optional
 * server-provided timings from the usage chunk.
 * Returns null without a usage chunk; unmeasurable numbers become null.
 */
export function computeStats(usage, startMs, firstTokenMs, endMs, timings) {
  if (!usage) return null;
  const completion = usage.completion_tokens ?? 0;
  const ttftMs = firstTokenMs === null || firstTokenMs === undefined ? null : Math.round(firstTokenMs - startMs);
  const decodeSec = ttftMs === null ? null : (endMs - firstTokenMs) / 1000;
  const tokensPerSec = decodeSec > 0 && completion > 0 ? round(completion / decodeSec, 1) : null;

  const promptTokens = usage.prompt_tokens ?? 0;
  const prefillPerSec = timings?.prompt_per_second ?? (ttftMs > 0 ? round(promptTokens / (ttftMs / 1000), 1) : null);
  const decodePerSec = timings?.predicted_per_second ?? tokensPerSec;

  return {
    promptTokens,
    completionTokens: completion,
    ttftMs,
    tokensPerSec,
    prefillPerSec,
    decodePerSec,
  };
}

/** One line: "12 prompt · 8 completion · 0.2 s to first token · prefill 1,450 tok/s · decode 31.5 tok/s". */
export function formatStats(stats) {
  const ttft = stats.ttftMs === null ? "—" : `${twoDp.format(stats.ttftMs / 1000)} s`;
  const prefill = `${stats.prefillPerSec === null ? "—" : whole.format(stats.prefillPerSec)} tok/s`;
  const decode = `${stats.decodePerSec === null ? "—" : oneDp.format(stats.decodePerSec)} tok/s`;
  return [
    `${whole.format(stats.promptTokens)} prompt`,
    `${whole.format(stats.completionTokens)} completion`,
    `${ttft} to first token`,
    `prefill ${prefill} · decode ${decode}`,
  ].join(" · ");
}

/** The collapsed thinking block's label: "Thought for 3.2 s". */
export function thinkingLabel(seconds) {
  return `Thought for ${oneDp.format(seconds)} s`;
}
