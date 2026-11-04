// Settings dialog: system prompt, sampling parameters, custom JSON, model, look (theme, glow,
// scanlines, font) and API key,
// persisted in localStorage under "legion.settings". chat.js merges these into every request
// (the system prompt becomes the first `system` message); api.js sends the API key as
// `Authorization: Bearer`.
//
// The pure helpers (load, validateField, validateCustom, requestOptions, customBody,
// systemMessage) touch no DOM and are tested by `node --test tests/web_settings.test.mjs`.
import { emit, on } from "./bus.js";

const STORAGE_KEY = "legion.settings";
// Settings with a fixed set of values; load() drops anything else.
const CHOICES = {
  theme: new Set(["system", "light", "dark"]),
  glow: new Set(["off", "on"]),
  scanlines: new Set(["off", "on"]),
  font: new Set(["system", "terminal"]),
};

// Every field is stored as a string; "" means "not set" and is omitted from requests.
export const DEFAULTS = {
  systemPrompt: "",
  temperature: "",
  topP: "",
  topK: "",
  minP: "",
  maxTokens: "",
  presencePenalty: "",
  frequencyPenalty: "",
  seed: "", // "" = random
  customJson: "", // merged into the request body as-is
  model: "", // "" = the first model served by /v1/models
  theme: "system", // system | light | dark
  glow: "off", // off | on: phosphor glow on text and marks (dark theme)
  scanlines: "off", // off | on: CRT scanlines over the page
  font: "system", // system | terminal: monospace terminal lettering everywhere
  apiKey: "",
};

// Inclusive bounds for the number fields; integer: whole numbers only.
const RANGES = {
  temperature: { min: 0, max: 2 },
  topP: { min: 0, max: 1 },
  topK: { min: 0, integer: true },
  minP: { min: 0, max: 1 },
  maxTokens: { min: 1, integer: true },
  presencePenalty: { min: -2, max: 2 },
  frequencyPenalty: { min: -2, max: 2 },
  seed: { min: 0, integer: true },
};

// Settings key -> request-body parameter name.
const PARAMS = {
  temperature: "temperature",
  topP: "top_p",
  topK: "top_k",
  minP: "min_p",
  maxTokens: "max_tokens",
  presencePenalty: "presence_penalty",
  frequencyPenalty: "frequency_penalty",
  seed: "seed",
};

// Settings key -> dialog control. Each control sits in a <label> with a .field-error <small>.
const CONTROLS = {
  systemPrompt: "#s-system",
  temperature: "#s-temperature",
  topP: "#s-top-p",
  topK: "#s-top-k",
  minP: "#s-min-p",
  maxTokens: "#s-max-tokens",
  presencePenalty: "#s-presence-penalty",
  frequencyPenalty: "#s-frequency-penalty",
  seed: "#s-seed",
  customJson: "#s-custom",
  model: "#s-model",
  theme: "#s-theme",
  glow: "#s-glow",
  scanlines: "#s-scanlines",
  font: "#s-font",
  apiKey: "#s-api-key",
};

// Persisted JSON merged over the defaults; anything broken or unknown falls back.
export function load(raw) {
  const out = { ...DEFAULTS };
  let saved;
  try {
    saved = JSON.parse(raw);
  } catch {
    return out;
  }
  if (!saved || typeof saved !== "object") return out;
  for (const key of Object.keys(DEFAULTS)) {
    const value = saved[key];
    if (typeof value !== "string") continue;
    if (CHOICES[key] && !CHOICES[key].has(value)) continue;
    out[key] = value;
  }
  return out;
}

// Inline error for a number field, or null when it is empty (not set) or valid.
export function validateField(key, text) {
  const range = RANGES[key];
  if (!range) return null;
  const t = text.trim();
  if (t === "") return null;
  const n = Number(t);
  if (!Number.isFinite(n)) return "enter a number";
  if (range.integer && !Number.isInteger(n)) return "enter a whole number";
  if (range.min !== undefined && n < range.min) return `must be at least ${range.min}`;
  if (range.max !== undefined && n > range.max) return `must be at most ${range.max}`;
  return null;
}

// Inline error for the custom JSON field, or null when it is empty or a JSON object.
export function validateCustom(text) {
  const t = text.trim();
  if (t === "") return null;
  let parsed;
  try {
    parsed = JSON.parse(t);
  } catch {
    return "invalid JSON";
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return "must be a JSON object";
  return null;
}

// The sampling parameters to merge into every request; unset and invalid fields are omitted.
export function requestOptions(settings) {
  const out = {};
  for (const [key, param] of Object.entries(PARAMS)) {
    const text = (settings[key] ?? "").trim();
    if (text === "" || validateField(key, text)) continue;
    out[param] = Number(text);
  }
  return out;
}

// The parsed custom JSON object, or {} when empty or invalid (the dialog shows the error).
export function customBody(settings) {
  const t = (settings.customJson ?? "").trim();
  return t && !validateCustom(t) ? JSON.parse(t) : {};
}

// The system prompt as a first message, or null when unset.
export function systemMessage(settings) {
  const text = (settings.systemPrompt ?? "").trim();
  return text ? { role: "system", content: text } : null;
}

let current = null;
const fields = new Map(); // settings key -> { input, error }
let defaultModel = null; // first id from /v1/models

// The current settings, loaded from localStorage once per page load.
export function getSettings() {
  current ??= load(globalThis.localStorage?.getItem(STORAGE_KEY) ?? null);
  return current;
}

function persist() {
  globalThis.localStorage?.setItem(STORAGE_KEY, JSON.stringify(current));
  emit("settings", { settings: current });
}

// The look settings. Theme "system" clears the light/dark override so the stylesheet follows
// the OS; glow, scanlines and font become <html data-*> attributes that style.css keys on.
function applyLook(s) {
  const root = document.documentElement;
  root.style.colorScheme = s.theme === "system" ? "" : s.theme;
  root.dataset.glow = s.glow;
  root.dataset.scanlines = s.scanlines;
  root.dataset.font = s.font;
}

function showError(key, message) {
  const { error } = fields.get(key);
  error.textContent = message ?? "";
  error.hidden = !message;
}

// One control changed: validate, and only persist and send valid values.
function onInput(key) {
  const { input } = fields.get(key);
  const text = input.value;
  const message = key === "customJson" ? validateCustom(text) : validateField(key, text);
  showError(key, message);
  if (message) return; // invalid: keep the last valid value; requestOptions omits this one
  current[key] = text;
  persist();
  if (CHOICES[key]) applyLook(current);
  if (key === "model") emit("model", { model: text || defaultModel });
}

export function initSettings() {
  const dialog = document.querySelector("#settings-dialog");
  getSettings(); // read storage before painting the fields
  applyLook(current);

  for (const [key, selector] of Object.entries(CONTROLS)) {
    const input = document.querySelector(selector);
    fields.set(key, { input, error: input.closest("label").querySelector(".field-error") });
    input.value = current[key];
    input.addEventListener("input", () => onInput(key));
  }

  document.querySelector("#settings-button").addEventListener("click", () => dialog.showModal());
  // A click on the backdrop lands on the dialog element itself and closes it.
  dialog.addEventListener("click", (e) => e.target === dialog && dialog.close());

  document.querySelector("#settings-reset").addEventListener("click", () => {
    current = { ...DEFAULTS };
    persist();
    for (const [key, { input }] of fields) input.value = current[key];
    for (const key of fields.keys()) showError(key, null);
    applyLook(current);
    emit("model", { model: defaultModel });
  });

  on("models", ({ models }) => {
    defaultModel = models[0]?.id ?? null;
    const { input } = fields.get("model");
    input.replaceChildren(new Option("Default", ""), ...models.map((m) => new Option(m.id, m.id)));
    input.value = models.some((m) => m.id === current.model) ? current.model : "";
    emit("model", { model: input.value || defaultModel });
  });
}
