// Run with: node --test tests/web_store.test.mjs
// Only the pure helpers of web/js/store.js; IndexedDB itself is browser-only.
import assert from "node:assert/strict";
import test from "node:test";
import { isInterruptedReply, messagesAfter, sortConversations, titleFrom } from "../web/js/store.js";

test("titleFrom keeps short text and flattens whitespace", () => {
  assert.equal(titleFrom("  hello\n  world "), "hello world");
});

test("titleFrom cuts at 40 characters", () => {
  const t = titleFrom("x".repeat(100));
  assert.equal(t, `${"x".repeat(40)}…`);
});

test("titleFrom falls back for empty text", () => {
  assert.equal(titleFrom("  "), "New chat");
});

test("sortConversations is newest first and does not mutate", () => {
  const input = [{ updated: 1 }, { updated: 3 }, { updated: 2 }];
  assert.deepEqual(sortConversations(input).map((c) => c.updated), [3, 2, 1]);
  assert.equal(input[0].updated, 1);
});

test("messagesAfter returns the tail after the given id", () => {
  const msgs = [{ id: "a" }, { id: "b" }, { id: "c" }];
  assert.deepEqual(messagesAfter(msgs, "a").map((m) => m.id), ["b", "c"]);
});

test("messagesAfter is empty for the last or an unknown id", () => {
  const msgs = [{ id: "a" }];
  assert.deepEqual(messagesAfter(msgs, "a"), []);
  assert.deepEqual(messagesAfter(msgs, "zz"), []);
});

test("messagesAfter does not mutate the input", () => {
  const msgs = [{ id: "a" }, { id: "b" }];
  messagesAfter(msgs, "a");
  assert.equal(msgs.length, 2);
});

test("isInterruptedReply only matches empty assistant placeholders", () => {
  assert.equal(isInterruptedReply({ role: "assistant", content: "", reasoning: "" }), true);
  assert.equal(isInterruptedReply({ role: "assistant", content: "partial", reasoning: "" }), false);
  assert.equal(isInterruptedReply({ role: "assistant", content: "", reasoning: "hmm" }), false);
  assert.equal(isInterruptedReply({ role: "user", content: "", reasoning: "" }), false);
});
