// The only file that talks to the server. Only our own API: /v1/models and /v1/chat/completions.
import { parseSse } from "./sse.js";
import { getSettings } from "./settings.js";

// The API key from the settings dialog, sent as Bearer when set (the server may require one).
function authHeader() {
  const key = getSettings().apiKey.trim();
  return key ? { authorization: `Bearer ${key}` } : {};
}

export async function listModels() {
  const res = await fetch("/v1/models", { headers: authHeader() });
  if (!res.ok) throw new Error(await errorMessage(res));
  return (await res.json()).data;
}

// Async generator of chat.completion.chunk objects. Pass an AbortSignal to stop early.
export async function* streamChat(body, signal) {
  const res = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "content-type": "application/json", ...authHeader() },
    body: JSON.stringify({ ...body, stream: true, stream_options: { include_usage: true } }),
    signal,
  });
  if (!res.ok) throw new Error(await errorMessage(res));
  yield* parseSse(res.body.pipeThrough(new TextDecoderStream()));
}

// The server's error body is {"error": {"message": ...}}.
async function errorMessage(res) {
  try {
    const body = await res.json();
    if (body?.error?.message) return body.error.message;
  } catch {
    // not JSON
  }
  return `HTTP ${res.status} ${res.statusText}`;
}
