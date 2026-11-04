// The chat view: renders state.messages and sends the conversation on submit.
// An assistant reply gets a collapsible "Thinking" block above the answer (accumulated from
// delta.reasoning_content, expanded while reasoning streams, collapsed with a "Thought for N s"
// label when the answer starts) and a stats line below it (final usage chunk + client timing).
// Per-message actions (shown on hover): user messages can be edited inline — saving resends
// and drops everything after the edited message; assistant messages can be regenerated
// (replaces the reply), copied (raw markdown) or deleted. While a reply streams a Stop
// button aborts the request and the partial text is kept.
// Messages are persisted as they happen (store.js): the user message when sent, the assistant
// reply created when streaming starts and updated when it finishes or is stopped, so closing
// the tab mid-reply loses nothing. User messages may carry images (attach.js): stored as
// {mime, blob} on the message, sent as a content array of image_url + text parts.
import { streamChat } from "./api.js";
import { imagesRow, requestMessage, takeAttachments } from "./attach.js";
import { emit, on } from "./bus.js";
import { initMarkdown, setMarkdown, streamInto } from "./markdown.js";
import { customBody, getSettings, requestOptions, systemMessage } from "./settings.js";
import { computeStats, formatStats, thinkingLabel } from "./stats.js";
import { state } from "./state.js";
import * as store from "./store.js";

const list = document.querySelector("#messages");
const form = document.querySelector("#composer");
const input = document.querySelector("#input");
const sendButton = document.querySelector("#send");
const thinkingToggle = document.querySelector("#thinking");

// The Stop button, shown only while a reply streams (see the busy handler).
const stopButton = document.createElement("button");
stopButton.type = "button";
stopButton.id = "stop";
stopButton.textContent = "Stop";
stopButton.hidden = true;
form.insertBefore(stopButton, sendButton);

let controller = null; // AbortController of the in-flight reply, or null
const elements = new Map(); // message object -> rendered element (for the actions)

const scroll = () => {
  list.scrollTop = list.scrollHeight;
};

function actionButton(label, onclick) {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = label;
  b.addEventListener("click", onclick);
  return b;
}

// The per-message action bar: Edit on user messages; Regenerate, Copy, Delete on assistant ones.
function actionsBar(m) {
  const bar = document.createElement("div");
  bar.className = "actions";
  if (m.role === "assistant") {
    bar.append(
      actionButton("Regenerate", () => regenerate(m)),
      actionButton("Copy", (e) => copyMessage(m, e.target)),
      actionButton("Delete", () => deleteMessage(m)),
    );
  } else {
    bar.append(actionButton("Edit", () => startEdit(m)));
  }
  return bar;
}

// User messages and errors: plain text, never model output.
// A message object registers the bubble in `elements` and gets an action bar; user messages
// with images (attach.js) show their thumbnails above the text.
function bubble(role, text, message = null) {
  const el = document.createElement("div");
  el.className = `message ${role}`;
  const body = document.createElement("div");
  body.className = "text";
  body.textContent = text;
  el.append(body);
  if (message?.images?.length) el.prepend(imagesRow(message.images));
  if (message) {
    elements.set(message, el);
    el.append(actionsBar(message));
  }
  list.append(el);
  scroll();
  return el;
}

function thinkingBlock(text, label) {
  const details = document.createElement("details");
  details.className = "thinking";
  const summary = document.createElement("summary");
  summary.textContent = label;
  const body = document.createElement("div");
  body.className = "thinking-text";
  body.textContent = text; // reasoning is model output: plain text, never HTML
  details.append(summary, body);
  return details;
}

function statsLine(stats) {
  const el = document.createElement("div");
  el.className = "stats muted";
  el.textContent = formatStats(stats);
  return el;
}

// Assistant message built from a message object, so stored messages (reasoning, stats)
// render like freshly streamed ones. Returns the message element and its answer container.
function assistantBubble(m) {
  const el = document.createElement("div");
  el.className = "message assistant";
  if (m.reasoning) el.append(thinkingBlock(m.reasoning, "Thinking"));
  const content = document.createElement("div");
  content.className = "content";
  setMarkdown(content, m.content); // built as DOM nodes, never parsed as HTML
  el.append(content);
  if (m.stats) el.append(statsLine(m.stats));
  el.append(actionsBar(m));
  elements.set(m, el);
  list.append(el);
  scroll();
  return { el, content };
}

// The live thinking block of a streaming reply: open while reasoning arrives, collapsed
// with the elapsed time once the answer starts (or the reply ends).
function liveThinking(messageEl, contentEl) {
  const details = thinkingBlock("", "Thinking");
  details.open = true;
  const summary = details.querySelector("summary");
  const body = details.querySelector(".thinking-text");
  messageEl.insertBefore(details, contentEl);
  let startMs = null;
  let done = false;
  return {
    append(text) {
      if (startMs === null) startMs = performance.now();
      body.textContent += text;
      scroll();
    },
    collapse() {
      if (done) return;
      done = true;
      if (startMs !== null) summary.textContent = thinkingLabel((performance.now() - startMs) / 1000);
      details.open = false;
    },
  };
}

// Stores a new message in the open conversation, creating the conversation (and the URL
// hash) on the first one. Returns the stored message with its id.
async function saveMessage(msg, parentId) {
  if (!state.convId) {
    const conv = await store.createConversation(store.titleFrom(msg.content));
    state.convId = conv.id;
    history.replaceState(null, "", `#/c/${state.convId}`);
  }
  return store.addMessage({ ...msg, convId: state.convId, parentId });
}

// Streams the reply to the conversation so far into an assistant message (already in
// state.messages and stored) and its bubble. Stop (AbortError) keeps the partial text;
// a reply that stayed empty (error, or stopped before the first token) is dropped again.
async function streamReply(reply, el, answer) {
  const live = streamInto(answer, scroll);
  let thinking = null;
  let usage = null;
  let timings = null;
  const startMs = performance.now();
  let firstTokenMs = null;
  controller = new AbortController();
  state.busy = true;
  emit("busy", { busy: true });
  try {
    const index = state.messages.indexOf(reply);
    const settings = getSettings();
    const body = {
      model: state.model,
      // requestMessage rebuilds the content array for messages with stored images (attach.js)
      messages: await Promise.all(state.messages.slice(0, index).map(requestMessage)),
      reasoning_effort: thinkingToggle.checked ? "high" : "none",
      ...requestOptions(settings), // sampling from the settings dialog; invalid values are omitted
      ...customBody(settings), // custom JSON merges last, so it wins over everything
    };
    const system = systemMessage(settings); // the system prompt is not stored in the conversation
    if (system) body.messages.unshift(system);
    for await (const chunk of streamChat(body, controller.signal)) {
      if (chunk.usage) {
        usage = chunk.usage; // the final include_usage chunk
        timings = chunk.timings;
        continue;
      }
      const delta = chunk.choices?.[0]?.delta;
      if (!delta) continue;
      if (delta.reasoning_content) {
        if (firstTokenMs === null) firstTokenMs = performance.now();
        thinking ??= liveThinking(el, answer);
        thinking.append(delta.reasoning_content);
        reply.reasoning += delta.reasoning_content;
      }
      if (delta.content) {
        if (firstTokenMs === null) firstTokenMs = performance.now();
        thinking?.collapse(); // the answer starts: fold the thinking block away
        reply.content += delta.content;
        live.update(reply.content);
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") bubble("error", err.message); // Stop is not an error
  } finally {
    controller = null;
    live.flush();
    thinking?.collapse(); // also when the reply ends without an answer
    if (usage) {
      reply.stats = computeStats(usage, startMs, firstTokenMs, performance.now(), timings);
      if (reply.stats) el.insertBefore(statsLine(reply.stats), el.querySelector(".actions"));
      scroll(); // keep the stats line and the actions in view
    }
    if (!reply.content) {
      const i = state.messages.indexOf(reply); // the conversation may have changed meanwhile
      if (i !== -1) state.messages.splice(i, 1);
      elements.delete(reply);
      el.remove();
    }
    state.busy = false;
    emit("busy", { busy: false });
    emit("messages", { messages: state.messages });
    // Persist the finished or stopped reply; drop the placeholder when there is nothing to keep.
    if (reply.content) {
      await store.updateMessage(reply.id, {
        content: reply.content,
        reasoning: reply.reasoning,
        stats: reply.stats ?? null,
      });
    } else {
      await store.deleteMessages([reply.id]);
    }
  }
}

// Appends a fresh assistant placeholder after `parentId`, stores it, and streams its reply.
async function newReply(parentId) {
  const reply = { role: "assistant", content: "", reasoning: "" };
  state.messages.push(reply);
  const { el, content } = assistantBubble(reply);
  Object.assign(reply, await saveMessage(reply, parentId));
  await streamReply(reply, el, content);
}

async function send(text, images = []) {
  const user = { role: "user", content: text };
  if (images.length) user.images = images.map(({ mime, blob }) => ({ mime, blob }));
  const prev = state.messages.at(-1);
  state.messages.push(user);
  bubble("user", text, user);
  Object.assign(user, await saveMessage(user, prev?.id ?? null));
  emit("messages", { messages: state.messages }); // sidebar: show the new conversation now
  await newReply(user.id);
}

// Inline editor for a user message: Save & resend replaces the message and drops everything
// after it; Cancel restores the original. Enter saves, Shift+Enter a newline, Escape cancels.
function startEdit(m) {
  if (state.busy || !m.id) return;
  const el = elements.get(m);
  if (!el || el.classList.contains("editing")) return;
  const body = el.querySelector(".text");
  const editor = document.createElement("div");
  editor.className = "edit";
  const textarea = document.createElement("textarea");
  textarea.value = m.content;
  const row = document.createElement("div");
  row.className = "edit-row";
  row.append(actionButton("Save & resend", () => finish(true)), actionButton("Cancel", () => finish(false)));
  editor.append(textarea, row);
  let done = false;
  const finish = async (submit) => {
    if (done) return;
    done = true;
    const text = textarea.value.trim();
    el.classList.remove("editing");
    editor.remove();
    body.hidden = false;
    if (!submit || !text) return;
    body.textContent = text;
    await resendAfter(m, text);
  };
  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      finish(true);
    } else if (e.key === "Escape") {
      e.preventDefault();
      finish(false);
    }
  });
  el.classList.add("editing");
  body.hidden = true;
  el.insertBefore(editor, el.querySelector(".actions"));
  textarea.focus();
}

// The edited user message is updated in the store, everything after it is dropped from the
// conversation and the store, and a fresh reply is streamed.
async function resendAfter(m, text) {
  m.content = text;
  await store.updateMessage(m.id, { content: text });
  const dropped = store.messagesAfter(state.messages, m.id);
  state.messages.splice(state.messages.indexOf(m) + 1);
  for (const d of dropped) {
    elements.get(d)?.remove();
    elements.delete(d);
  }
  await store.deleteMessages(dropped.map((d) => d.id).filter(Boolean));
  await newReply(m.id);
}

// Re-runs an assistant reply in place: the message keeps its id and position, the rest of
// the conversation is untouched.
async function regenerate(m) {
  if (state.busy || !m.id) return;
  const el = elements.get(m);
  if (!el) return;
  m.content = "";
  m.reasoning = "";
  m.stats = null;
  const content = document.createElement("div");
  content.className = "content";
  el.replaceChildren(content, actionsBar(m));
  await streamReply(m, el, content);
}

async function copyMessage(m, button) {
  try {
    await navigator.clipboard.writeText(m.content); // raw markdown, as stored
    button.textContent = "Copied";
  } catch {
    button.textContent = "Failed";
  }
  setTimeout(() => {
    if (button.isConnected) button.textContent = "Copy";
  }, 1500);
}

async function deleteMessage(m) {
  if (state.busy) return;
  const i = state.messages.indexOf(m);
  if (i !== -1) state.messages.splice(i, 1);
  elements.get(m)?.remove();
  elements.delete(m);
  if (m.id) await store.deleteMessages([m.id]);
  emit("messages", { messages: state.messages });
}

function renderMessage(m) {
  if (m.role === "assistant") assistantBubble(m);
  else bubble(m.role, m.content, m);
}

export function initChat() {
  initMarkdown(list);
  on("conversation", async ({ messages }) => {
    // A reply interrupted by closing the tab was stored empty when streaming started: drop it.
    const stale = messages.filter(store.isInterruptedReply);
    if (stale.length) {
      messages = messages.filter((m) => !stale.includes(m));
      state.messages = messages;
      await store.deleteMessages(stale.map((m) => m.id));
    }
    elements.clear();
    list.replaceChildren();
    for (const m of messages) renderMessage(m);
  });
  on("busy", ({ busy }) => {
    sendButton.disabled = busy;
    stopButton.hidden = !busy;
    list.classList.toggle("busy", busy); // message actions are inert while a reply streams
    if (!busy) input.focus();
  });
  stopButton.addEventListener("click", () => controller?.abort());
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (state.busy) return;
    const text = input.value.trim();
    const images = await takeAttachments(); // clears the pending strip (attach.js)
    if (!text && images.length === 0) return;
    input.value = "";
    send(text, images);
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      form.requestSubmit();
    }
  });
}
