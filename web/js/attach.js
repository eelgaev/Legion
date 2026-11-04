// Image attachments for vision models: paperclip button, paste and drag-and-drop.
// The paperclip only shows while the selected model's /v1/models entry lists "vision" in
// `capabilities`. Pending images appear as thumbnails above the textarea (max 16, each
// <= 20 MiB, png/jpeg/webp/gif); on send chat.js takes them as {mime, blob, dataUrl} items
// and stores {mime, blob} on the message record (store.js), so stored bubbles keep them.
// The pure helpers are tested by node --test tests/web_attach.test.mjs.
import { on } from "./bus.js";
import { state } from "./state.js";

export const IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp", "image/gif"];
export const MAX_IMAGE_BYTES = 20 * 1024 * 1024;
export const MAX_IMAGES = 16;

const make = (tag, attrs) => Object.assign(document.createElement(tag), attrs);

// Error message for a candidate file, or null when it can be added. `pending` is how many
// images are already queued for this message.
export function validateImage(file, pending) {
  const name = file.name || "pasted image";
  if (!IMAGE_TYPES.includes(file.type)) return `${name}: unsupported image type`;
  if (file.size > MAX_IMAGE_BYTES) return `${name}: larger than 20 MiB`;
  if (pending >= MAX_IMAGES) return `up to ${MAX_IMAGES} images per message`;
  return null;
}

// Whether a /v1/models entry (see GET /v1/models) accepts images.
export function isVisionModel(model) {
  return (model?.capabilities ?? []).includes("vision");
}

// The user-message content for the API: one image_url part per {mime, dataUrl} item,
// in order, then the text part.
export function buildContent(images, text) {
  return [
    ...images.map(({ dataUrl }) => ({ type: "image_url", image_url: { url: dataUrl } })),
    { type: "text", text },
  ];
}

const readDataUrl = (blob) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });

// One message in request shape: plain {role, content}, or — when it carries images — a
// content array rebuilt from the stored blobs (map this over state.messages before sending).
export async function requestMessage(m) {
  if (!m.images?.length) return { role: m.role, content: m.content };
  const items = await Promise.all(m.images.map(async ({ mime, blob }) => ({ mime, dataUrl: await readDataUrl(blob) })));
  return { role: m.role, content: buildContent(items, m.content) };
}

// Thumbnail row for a stored user message (chat.js renders the bubbles). Each img revokes its
// object URL once loaded so re-rendering a long chat with images does not leak them.
export function imagesRow(images) {
  const row = document.createElement("div");
  row.className = "bubble-images";
  for (const { mime, blob } of images) {
    const img = make("img", { src: URL.createObjectURL(blob), alt: mime });
    img.addEventListener("load", () => URL.revokeObjectURL(img.src), { once: true });
    row.append(img);
  }
  return row;
}

let pending = []; // { mime, blob, url, cell } waiting to be sent
let vision = false; // the selected model accepts images
let bar; // #attach-bar: thumbnail strip + inline error, above the textarea
let thumbs; // #pending-images
let errorEl; // #attach-error
let attachButton; // #attach (the paperclip)

const refresh = () => bar && (bar.hidden = pending.length === 0 && errorEl.hidden);
const showError = (msg) => {
  if (!errorEl) return; // takeAttachments is safe to call even before initAttach
  errorEl.textContent = msg;
  errorEl.hidden = !msg;
  refresh();
};

function addFiles(files) {
  if (!vision) {
    showError("the selected model does not accept images");
    return;
  }
  const errors = [];
  for (const file of files) {
    const err = validateImage(file, pending.length);
    if (err) {
      errors.push(err);
      continue;
    }
    const item = { mime: file.type, blob: file, url: URL.createObjectURL(file) };
    const remove = make("button", { type: "button", textContent: "×", title: "Remove" });
    remove.addEventListener("click", () => {
      pending = pending.filter((p) => p !== item);
      URL.revokeObjectURL(item.url);
      item.cell.remove();
      refresh();
    });
    item.cell = make("div", { className: "thumb" });
    item.cell.append(make("img", { src: item.url, alt: file.name }), remove);
    thumbs.append(item.cell);
    pending.push(item);
  }
  showError(errors[0] ?? "");
}

// Hands the pending images to chat.js as {mime, blob, dataUrl} items and empties the strip.
export async function takeAttachments() {
  const items = await Promise.all(
    pending.map(async ({ mime, blob, url }) => {
      const dataUrl = await readDataUrl(blob);
      URL.revokeObjectURL(url);
      return { mime, blob, dataUrl };
    }),
  );
  pending = [];
  thumbs?.replaceChildren();
  showError("");
  return items;
}

export function initAttach() {
  const form = document.querySelector("#composer");
  const input = document.querySelector("#input");
  const fileInput = document.querySelector("#attach-input");
  bar = document.querySelector("#attach-bar");
  thumbs = document.querySelector("#pending-images");
  errorEl = document.querySelector("#attach-error");
  attachButton = document.querySelector("#attach");

  // Track the selected model's capabilities: "models" carries the full entries, "model" only
  // the id chosen in the settings dialog (main.js updates state.model before our handler).
  const models = new Map();
  const setVision = (v) => {
    vision = v;
    attachButton.hidden = !v;
  };
  on("models", ({ models: list }) => {
    models.clear();
    for (const m of list) models.set(m.id, m);
    setVision(isVisionModel(models.get(state.model)));
  });
  on("model", ({ model }) => setVision(isVisionModel(models.get(model))));

  attachButton.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    addFiles([...fileInput.files]);
    fileInput.value = ""; // re-selecting the same file fires change again
  });
  input.addEventListener("paste", (e) => {
    const files = [...(e.clipboardData?.files ?? [])];
    if (!files.length) return; // plain text pastes as usual
    e.preventDefault();
    addFiles(files);
  });
  form.addEventListener("dragover", (e) => {
    e.preventDefault();
    form.classList.add("dragover");
  });
  form.addEventListener("dragleave", (e) => {
    if (!form.contains(e.relatedTarget)) form.classList.remove("dragover");
  });
  form.addEventListener("drop", (e) => {
    e.preventDefault();
    form.classList.remove("dragover");
    addFiles([...e.dataTransfer.files]);
  });
}
