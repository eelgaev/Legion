// Entry point: loads the model name and starts each feature module.
// Add a feature by creating web/js/<feature>.js with an init function and calling it here.
import { listModels } from "./api.js";
import { initAttach } from "./attach.js";
import { emit, on } from "./bus.js";
import { initChat } from "./chat.js";
import { initEmpty } from "./empty.js";
import { getSettings, initSettings } from "./settings.js";
import { initSidebar } from "./sidebar.js";
import { state } from "./state.js";

const modelName = document.querySelector("#model-name");
const modelChip = document.querySelector("#model-chip"); // data-state colors its LED (style.css)
const showModel = (model) => {
  state.model = model;
  modelName.textContent = model ?? "no model";
  modelChip.dataset.state = model ? "ok" : "error";
};
on("models", ({ models }) => showModel(models[0]?.id ?? null));
on("model", ({ model }) => showModel(model)); // chosen in the settings dialog

const loadModels = () =>
  listModels()
    .then((models) => emit("models", { models }))
    .catch((err) => {
      modelName.textContent = `models unavailable: ${err.message}`;
      modelChip.dataset.state = "error";
    });

initChat();
initEmpty();
initAttach();
initSettings();
initSidebar();
loadModels();

// A new API key can unlock /v1/models, so reload the list when it changes.
let lastApiKey = getSettings().apiKey;
on("settings", () => {
  const key = getSettings().apiKey;
  if (key === lastApiKey) return;
  lastApiKey = key;
  loadModels();
});
