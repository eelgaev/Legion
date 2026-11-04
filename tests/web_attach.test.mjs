// Run with: node --test tests/web_attach.test.mjs
// Only the pure helpers of web/js/attach.js; FileReader, object URLs and the DOM are browser-only.
import assert from "node:assert/strict";
import test from "node:test";
import { MAX_IMAGES, buildContent, isVisionModel, validateImage } from "../web/js/attach.js";

const file = (over = {}) => ({ name: "photo.png", type: "image/png", size: 1024, ...over });

test("validateImage accepts the four supported types", () => {
  for (const type of ["image/png", "image/jpeg", "image/webp", "image/gif"]) {
    assert.equal(validateImage(file({ type }), 0), null);
  }
});

test("validateImage rejects other types", () => {
  assert.match(validateImage(file({ type: "image/svg+xml" }), 0), /unsupported image type/);
  assert.match(validateImage(file({ type: "application/pdf" }), 0), /unsupported image type/);
  assert.match(validateImage(file({ type: "" }), 0), /unsupported image type/);
});

test("validateImage rejects files over 20 MiB", () => {
  assert.equal(validateImage(file({ size: 20 * 1024 * 1024 }), 0), null);
  assert.match(validateImage(file({ size: 20 * 1024 * 1024 + 1 }), 0), /larger than 20 MiB/);
});

test("validateImage caps the pending images at 16", () => {
  assert.equal(validateImage(file(), MAX_IMAGES - 1), null);
  assert.match(validateImage(file(), MAX_IMAGES), /up to 16 images per message/);
});

test("validateImage names the file, falling back for pasted images", () => {
  assert.match(validateImage(file({ name: "notes.txt", type: "text/plain" }), 0), /^notes\.txt/);
  assert.match(validateImage({ type: "text/plain", size: 1 }, 0), /^pasted image/);
});

test("isVisionModel reads the capabilities array of a /v1/models entry", () => {
  assert.equal(isVisionModel({ id: "mock", capabilities: ["vision"] }), true);
  assert.equal(isVisionModel({ id: "mock", capabilities: [] }), false);
  assert.equal(isVisionModel({ id: "mock" }), false);
  assert.equal(isVisionModel(undefined), false);
});

test("buildContent puts image_url parts before the text part, in order", () => {
  const images = [
    { mime: "image/png", dataUrl: "data:image/png;base64,AAA=" },
    { mime: "image/jpeg", dataUrl: "data:image/jpeg;base64,BBB=" },
  ];
  assert.deepEqual(buildContent(images, "what is in these?"), [
    { type: "image_url", image_url: { url: "data:image/png;base64,AAA=" } },
    { type: "image_url", image_url: { url: "data:image/jpeg;base64,BBB=" } },
    { type: "text", text: "what is in these?" },
  ]);
});

test("buildContent without images is just the text part", () => {
  assert.deepEqual(buildContent([], "hi"), [{ type: "text", text: "hi" }]);
});
