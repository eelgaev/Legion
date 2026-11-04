// The whole event system: modules talk through named events instead of importing each other.
//   on("name", (detail) => ...)   emit("name", detail)
// Events in use (add yours to the table in web/README.md):
//   models        {models}      /v1/models loaded
//   model         {model}       a model was chosen in the settings dialog
//   messages      {messages}    the conversation changed (state.messages)
//   conversation  {id, messages} another conversation (or a new chat) was opened
//   busy          {busy}        a reply started (true) or finished (false)
//   settings      {settings}    settings saved to localStorage (settings.js)
const target = new EventTarget();

export function on(name, handler) {
  target.addEventListener(name, (e) => handler(e.detail));
}

export function emit(name, detail) {
  target.dispatchEvent(new CustomEvent(name, { detail }));
}
