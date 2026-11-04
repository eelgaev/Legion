// Run with: node --test tests/web_settings.test.mjs
// Only the pure helpers of web/js/settings.js; the dialog wiring needs a browser.
import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULTS,
  customBody,
  load,
  requestOptions,
  systemMessage,
  validateCustom,
  validateField,
} from "../web/js/settings.js";

test("load falls back to defaults on missing or broken storage", () => {
  assert.deepEqual(load(null), DEFAULTS);
  assert.deepEqual(load("not json"), DEFAULTS);
  assert.deepEqual(load("[]"), DEFAULTS);
  assert.deepEqual(load("7"), DEFAULTS);
});

test("load keeps known string fields and drops the rest", () => {
  const saved = load(
    JSON.stringify({ temperature: "0.7", apiKey: "k", theme: "dark", model: "m", topP: 5, bogus: true }),
  );
  assert.equal(saved.temperature, "0.7");
  assert.equal(saved.apiKey, "k");
  assert.equal(saved.theme, "dark");
  assert.equal(saved.model, "m");
  assert.equal(saved.topP, DEFAULTS.topP); // numbers are not accepted
  assert.ok(!("bogus" in saved));
});

test("load keeps a valid glow and rejects an unknown one", () => {
  assert.equal(load(JSON.stringify({ glow: "on" })).glow, "on");
  assert.equal(load(JSON.stringify({ glow: "max" })).glow, "off");
  assert.equal(load(null).glow, "off");
});

test("load keeps valid scanlines and font values and rejects unknown ones", () => {
  const ok = load(JSON.stringify({ scanlines: "on", font: "terminal" }));
  assert.equal(ok.scanlines, "on");
  assert.equal(ok.font, "terminal");
  const bad = load(JSON.stringify({ scanlines: "heavy", font: "comic" }));
  assert.equal(bad.scanlines, "off");
  assert.equal(bad.font, "system");
});

test("load rejects an unknown theme", () => {
  assert.equal(load(JSON.stringify({ theme: "neon" })).theme, "system");
});

test("validateField checks ranges and integers", () => {
  assert.equal(validateField("temperature", ""), null); // empty = not set
  assert.equal(validateField("temperature", "0.7"), null);
  assert.match(validateField("temperature", "3"), /at most 2/);
  assert.match(validateField("temperature", "abc"), /number/);
  assert.match(validateField("topK", "1.5"), /whole number/);
  assert.equal(validateField("maxTokens", "1"), null);
  assert.match(validateField("maxTokens", "0"), /at least 1/);
  assert.match(validateField("presencePenalty", "-3"), /at least -2/);
  assert.equal(validateField("seed", "42"), null);
  assert.equal(validateField("systemPrompt", "anything"), null); // not a number field
});

test("requestOptions maps snake_case params and skips unset or invalid values", () => {
  const settings = { ...DEFAULTS, temperature: "0.7", topP: " 0.9 ", topK: "40", maxTokens: "abc", seed: "" };
  assert.deepEqual(requestOptions(settings), { temperature: 0.7, top_p: 0.9, top_k: 40 });
  assert.deepEqual(requestOptions(DEFAULTS), {});
});

test("validateCustom and customBody handle the custom JSON field", () => {
  assert.equal(validateCustom(""), null);
  assert.match(validateCustom("{oops"), /JSON/);
  assert.match(validateCustom("[1,2]"), /object/);
  assert.deepEqual(customBody({ ...DEFAULTS, customJson: '{"repeat_penalty": 1.1, "stop": ["/"]}' }), {
    repeat_penalty: 1.1,
    stop: ["/"],
  });
  assert.deepEqual(customBody(DEFAULTS), {});
  assert.deepEqual(customBody({ ...DEFAULTS, customJson: "nope" }), {}); // invalid: never sent
  assert.deepEqual(customBody({ ...DEFAULTS, customJson: "0" }), {}); // valid JSON, not an object
});

test("systemMessage trims the system prompt", () => {
  assert.deepEqual(systemMessage({ ...DEFAULTS, systemPrompt: " Be terse. " }), {
    role: "system",
    content: "Be terse.",
  });
  assert.equal(systemMessage(DEFAULTS), null);
});
