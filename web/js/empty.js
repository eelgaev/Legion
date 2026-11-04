// The empty state (#empty in index.html): logo, the served model and example prompts. CSS shows
// it while #messages has no children, so this module only fills in the model name and sends an
// example prompt when one is clicked.
import { on } from "./bus.js";
import { state } from "./state.js";

export function initEmpty() {
  const serving = document.querySelector("#empty .serving");
  const name = document.querySelector("#empty-model");
  const input = document.querySelector("#input");
  const form = document.querySelector("#composer");

  // main.js sets state.model before these handlers run (it subscribes first).
  const show = () => {
    name.textContent = state.model ?? "";
    serving.hidden = !state.model;
  };
  on("models", show);
  on("model", show);

  document.querySelector("#examples").addEventListener("click", (e) => {
    const button = e.target.closest("button");
    if (!button || state.busy) return;
    input.value = button.textContent;
    form.requestSubmit();
  });
}
