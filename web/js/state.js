// Shared app state: one plain object. Change it, then emit the matching event (see bus.js).
// Message shape: { role: "user" | "assistant", content: string, images?: [{mime, blob}] }, plus id/parentId/... once stored (store.js)
export const state = {
  model: null,     // model id from GET /v1/models
  convId: null,    // open conversation id (URL hash #/c/<id>); null until the first message is saved
  messages: [],    // the conversation sent to the server
  busy: false,     // true while a reply is streaming
};
