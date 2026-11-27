// Conversation storage in IndexedDB (database "legion"). Promise-based, no dependencies.
//   conversations {id, title, created, updated}
//   messages      {id, convId, parentId, role, content, reasoning, created, stats, images}
// parentId (the previous message's id, or null) is what lets branching come later;
// stats holds the final usage/timing numbers of a reply (null until recorded);
// images holds a user message's attachments as [{mime, blob}] (Blobs store natively;
// records written before attachments simply have no images field).
const TITLE_LENGTH = 40;

let dbPromise;

function open() {
  dbPromise ??= new Promise((resolve, reject) => {
    const req = indexedDB.open("legion", 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      db.createObjectStore("conversations", { keyPath: "id" });
      db.createObjectStore("messages", { keyPath: "id" }).createIndex("convId", "convId");
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

// Runs fn(stores) in one transaction; fn returns an IDBRequest (or nothing). Resolves with its result.
async function tx(names, mode, fn) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const t = db.transaction(names, mode);
    const stores = names.map((n) => t.objectStore(n));
    const req = fn(...stores);
    t.oncomplete = () => resolve(req?.result);
    t.onerror = t.onabort = () => reject(t.error);
  });
}

// Title of a new conversation: the first TITLE_LENGTH characters of its first user message.
export function titleFrom(text) {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > TITLE_LENGTH ? `${flat.slice(0, TITLE_LENGTH).trimEnd()}…` : flat || "New chat";
}

// Newest first.
// crypto.randomUUID only exists in secure contexts (https or localhost); the UI is
// often served over plain http to a LAN address, so fall back to getRandomValues,
// which is available everywhere, to build the same v4 shape.
function uuid() {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  const b = crypto.getRandomValues(new Uint8Array(16));
  b[6] = (b[6] & 0x0f) | 0x40;
  b[8] = (b[8] & 0x3f) | 0x80;
  const h = [...b].map((x) => x.toString(16).padStart(2, "0")).join("");
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}

export function sortConversations(list) {
  return [...list].sort((a, b) => b.updated - a.updated);
}

export async function listConversations() {
  return sortConversations(await tx(["conversations"], "readonly", (s) => s.getAll()));
}

export async function createConversation(title) {
  const now = Date.now();
  const conv = { id: uuid(), title, created: now, updated: now };
  await tx(["conversations"], "readwrite", (s) => s.add(conv));
  return conv;
}

export async function getConversation(id) {
  return tx(["conversations"], "readonly", (s) => s.get(id));
}

// Oldest first.
export async function getMessages(convId) {
  const list = await tx(["messages"], "readonly", (s) => s.index("convId").getAll(convId));
  return list.sort((a, b) => a.created - b.created);
}

// Stores a message (filling id, parentId, reasoning, created, stats, images) and bumps its conversation.
export async function addMessage(msg) {
  const full = {
    id: uuid(),
    parentId: null,
    reasoning: "",
    created: Date.now(),
    stats: null,
    images: [],
    ...msg,
  };
  await tx(["messages", "conversations"], "readwrite", (messages, convs) => {
    messages.add(full);
    const get = convs.get(full.convId);
    get.onsuccess = () => get.result && convs.put({ ...get.result, updated: Date.now() });
    return null;
  });
  return full;
}

// Merges changes into a stored message.
export async function updateMessage(id, changes) {
  await tx(["messages"], "readwrite", (s) => {
    const get = s.get(id);
    get.onsuccess = () => get.result && s.put({ ...get.result, ...changes });
    return null;
  });
}

export async function deleteMessages(ids) {
  if (ids.length === 0) return;
  await tx(["messages"], "readwrite", (s) => {
    for (const id of ids) s.delete(id);
    return null;
  });
}

// The messages stored after the one with `id` — what editing a message drops with it.
export function messagesAfter(messages, id) {
  const i = messages.findIndex((m) => m.id === id);
  return i < 0 ? [] : messages.slice(i + 1);
}

// An assistant placeholder created when streaming started but never updated (the tab was
// closed mid-reply): it carries no text, so opening the conversation drops it again.
export function isInterruptedReply(m) {
  return m.role === "assistant" && !m.content && !m.reasoning;
}

export async function renameConversation(id, title) {
  await tx(["conversations"], "readwrite", (s) => {
    const get = s.get(id);
    get.onsuccess = () => get.result && s.put({ ...get.result, title });
    return null;
  });
}

export async function deleteConversation(id) {
  await tx(["conversations", "messages"], "readwrite", (convs, messages) => {
    convs.delete(id);
    const cursor = messages.index("convId").openKeyCursor(IDBKeyRange.only(id));
    cursor.onsuccess = () => {
      if (!cursor.result) return;
      messages.delete(cursor.result.primaryKey);
      cursor.result.continue();
    };
    return null;
  });
}
