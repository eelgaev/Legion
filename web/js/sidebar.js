// Conversation sidebar: list, search, new/rename/delete. The open conversation id is the URL hash (#/c/<id>).
import { emit, on } from "./bus.js";
import { state } from "./state.js";
import * as store from "./store.js";

const list = document.querySelector("#conversations");
const search = document.querySelector("#search");
const newChat = document.querySelector("#new-chat");
const toggle = document.querySelector("#sidebar-toggle");

let conversations = [];

const hashId = () => location.hash.match(/^#\/c\/(.+)$/)?.[1] ?? null;

function render() {
  const q = search.value.trim().toLowerCase();
  list.replaceChildren(
    ...conversations.filter((c) => c.title.toLowerCase().includes(q)).map(row),
  );
}

function row(conv) {
  const li = document.createElement("li");
  li.className = conv.id === state.convId ? "current" : "";
  const open = document.createElement("a");
  open.href = `#/c/${conv.id}`;
  open.textContent = conv.title;
  open.title = conv.title;
  open.addEventListener("click", () => (toggle.checked = false));
  const rename = button("✎", "Rename", () => startRename(li, open, conv));
  const del = button("×", "Delete", async () => {
    if (!confirm(`Delete "${conv.title}"?`)) return;
    await store.deleteConversation(conv.id);
    if (conv.id === state.convId) location.hash = "";
    await refresh();
  });
  li.append(open, rename, del);
  return li;
}

function button(label, name, onclick) {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = label;
  b.title = b.ariaLabel = name;
  b.addEventListener("click", onclick);
  return b;
}

function startRename(li, link, conv) {
  const input = document.createElement("input");
  input.value = conv.title;
  input.ariaLabel = "Conversation title";
  let done = false;
  const finish = async (save) => {
    if (done) return;
    done = true;
    const title = input.value.trim();
    if (save && title && title !== conv.title) await store.renameConversation(conv.id, title);
    await refresh();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") finish(true);
    if (e.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => finish(true));
  link.replaceWith(input);
  input.select();
  input.focus();
  li.classList.add("editing");
}

async function refresh() {
  conversations = await store.listConversations();
  render();
}

// Shows the conversation named by the hash (or an empty new chat).
async function openFromHash() {
  const id = hashId();
  const conv = id && (await store.getConversation(id));
  state.convId = conv ? conv.id : null;
  state.messages = conv ? await store.getMessages(conv.id) : [];
  emit("conversation", { id: state.convId, messages: state.messages });
  render();
}

// Saves messages that have no id yet; creates the conversation on the first one.
async function persist() {
  let created = false;
  for (const msg of state.messages) {
    if (msg.id) continue;
    if (!state.convId) {
      const conv = await store.createConversation(store.titleFrom(msg.content));
      state.convId = conv.id;
      created = true;
    }
    const prev = state.messages[state.messages.indexOf(msg) - 1];
    Object.assign(msg, await store.addMessage({ ...msg, convId: state.convId, parentId: prev?.id ?? null }));
  }
  if (created) history.replaceState(null, "", `#/c/${state.convId}`);
  await refresh();
}

export function initSidebar() {
  search.addEventListener("input", render);
  newChat.addEventListener("click", () => {
    toggle.checked = false;
    if (location.hash) location.hash = "";
    else openFromHash();
  });
  // The narrow-screen menu button is a <label> for the checkbox: make it work from the keyboard
  // too, and let Escape close the overlay.
  document.querySelector("#sidebar-button").addEventListener("keydown", (e) => {
    if (e.key !== "Enter" && e.key !== " ") return;
    e.preventDefault();
    toggle.checked = !toggle.checked;
    if (toggle.checked) newChat.focus();
  });
  addEventListener("keydown", (e) => {
    if (e.key === "Escape" && toggle.checked) toggle.checked = false;
  });
  addEventListener("hashchange", openFromHash);
  on("messages", persist);
  on("busy", ({ busy }) => (newChat.disabled = busy));
  refresh().then(openFromHash);
}
