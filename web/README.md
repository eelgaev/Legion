# Legion web UI

The chat page served by `legion serve` at `/`. **No npm, no build step, no CSS compilers: edit the
files directly.** No frameworks either. `web/` is embedded in the binary (`src/server/routes/webui.rs`);
in debug builds the files are read from disk, so reload the page after editing. `/assets/<path>`
serves `web/<path>`, so `web/js/chat.js` is `/assets/js/chat.js`.

## Layout

| File | Job |
|---|---|
| `index.html` | Page skeleton. Add new UI containers here, with an `id`. Also holds the `#legion-mark` SVG symbol (the logo mark, drawn in `currentColor`) and the `#empty` section (shown by CSS while `#messages` is empty). |
| `legion.svg` | Favicon: the same mark on a dark tile (fixed colors). |
| `style.css` | All styling. Colors are variables in `@layer base`; components go in `@layer components`. The look follows `docs/legion.png`: phosphor green accent and amber "LED" dots, monospace for brand, labels, stats and code, system sans for reading text; light mode is the same accents on warm off-white. Block comments only (`//` is not a CSS comment and silently eats the next rule). |
| `js/main.js` | Entry point: loads `/v1/models`, calls each feature module's `init...()`. |
| `js/empty.js` | Empty state: fills in the served model name and sends an example prompt when one is clicked. |
| `js/api.js` | The only file that calls `fetch`. `listModels()`, `streamChat(body, signal)`; adds `Authorization: Bearer` from the settings. |
| `js/sse.js` | SSE parser (handles chunks split anywhere). `parseSse(textChunks)`. |
| `js/bus.js` | `on(name, handler)` / `emit(name, detail)`: modules talk through events. |
| `js/state.js` | One shared plain object: current model, conversation id, messages, busy flag. |
| `js/store.js` | IndexedDB `legion`: `conversations` and `messages` (with `parentId`, `stats`, `images`); promise API incl. `deleteMessages`; helpers `messagesAfter`, `isInterruptedReply`. Pure helpers are tested by `node --test tests/web_store.test.mjs`. |
| `js/sidebar.js` | Conversation list, search, new/rename/delete; the open id is the URL hash `#/c/<id>`; refreshes on `messages` (chat.js persists messages itself; persisting any id-less message is the fallback). |
| `js/chat.js` | Renders messages, sends on submit, streams the reply: collapsible thinking block (`delta.reasoning_content`), per-message stats from the final `usage` chunk, `reasoning_effort` from the Thinking toggle; merges the settings (system prompt, sampling, custom JSON) into every request. Per-message actions: Edit + resend (drops everything after), Regenerate, Copy, Delete; a Stop button aborts streaming and keeps the partial reply. Persists as it goes: the user message on send, the assistant reply created at stream start and updated on finish/stop. |
| `js/attach.js` | Image attachments for vision models: paperclip button (visible only when the selected model's `/v1/models` entry has `"vision"` in `capabilities`), paste and drag-and-drop; thumbnails above the textarea (max 16, <= 20 MiB, png/jpeg/webp/gif) with inline errors; on send hands chat.js `{mime, blob, dataUrl}` items and builds the `image_url` + `text` content array. Pure helpers tested by `node --test tests/web_attach.test.mjs`. |
| `js/settings.js` | Settings `<dialog>`: system prompt, sampling parameters, custom JSON, model, theme, glow, scanlines, font, API key; persisted in `localStorage` under `legion.settings`. Pure helpers tested by `node --test tests/web_settings.test.mjs`. |
| `js/markdown.js` | Our own markdown, no third-party code: `parse(text)` builds a plain block tree, `setMarkdown(el, text)` renders it as DOM nodes (model text is never parsed as HTML, so no sanitizer), `streamInto(el)` (re-render throttled to ~50 ms), `initMarkdown(root)` (Copy buttons). Pure parts tested by `node --test tests/web_markdown.test.mjs`. |
| `js/stats.js` | Pure helpers for the stats line and the "Thought for N s" label: `computeStats(usage, startMs, firstTokenMs, endMs)`, `formatStats(stats)`, `thinkingLabel(seconds)`. Tested by `node --test tests/web_stats.test.mjs`. |

## Conventions

- One ES module per feature in `js/` (`store.js`, `sidebar.js`, `markdown.js`, `settings.js`, ...).
  Export an `initX()` and call it from `main.js`; add your markup to `index.html` and CSS to `style.css`.
- Share data through `state.js`; announce changes with `emit()`. Events in use:
  `models {models}`, `model {model}` (chosen in the settings dialog), `messages {messages}` (the conversation changed; sidebar refreshes — chat.js persists messages itself),
  `conversation {id, messages}` (another conversation opened), `busy {busy}`, `settings {settings}` (saved to localStorage).
  Add new ones to this list.
- The UI only calls our API: `GET /v1/models` and `POST /v1/chat/completions` with `stream: true`
  and `stream_options.include_usage: true` (both added by `streamChat`).
- Model output is never parsed as HTML. Assistant text goes through `markdown.js`
  (`setMarkdown`/`streamInto`), which builds DOM nodes directly; user text and errors use
  `textContent`. The string-to-HTML DOM APIs are banned in `web/js/` (a node test enforces it),
  and `web/` contains no third-party code.
- Colors only through variables from `style.css`; no hard-coded colors in components.
- Check syntax: `for f in web/js/*.js; do node --check "$f"; done`. The Rust tests cover serving.

## Where future features go

| Feature | Goes in |
|---|---|
| Conversation storage / sidebar | `js/store.js`, `js/sidebar.js`; sidebar markup beside `#app` content |
| Markdown, math | `js/markdown.js`; used by `chat.js` when setting bubble content |
| Attachments | `js/attach.js` |
| Shortcuts, export/import | `js/shortcuts.js`, `sidebar.js` |

## Platform features we rely on

Targets current evergreen Chrome, Edge, Firefox and Safari only; no polyfills. Use these instead of
writing code:

- CSS: custom properties, `color-scheme: light dark` with `light-dark()`, native nesting, `@layer`,
  `:has()` and container queries instead of toggling classes in JS, `text-wrap: pretty`,
  `field-sizing: content` (auto-growing textarea), `scrollbar-gutter: stable`.
- HTML: `<dialog>` with `showModal()` for modals, the Popover API for menus, `<details>/<summary>`
  for collapsible blocks.
- JS: ES modules, `fetch` + `response.body.pipeThrough(new TextDecoderStream())` for streams,
  `AbortController`, `navigator.clipboard`, `structuredClone`, `Intl` for dates and numbers.
