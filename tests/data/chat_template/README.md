# Chat template fixtures

`tests/chat_template.rs` renders every case in `cases.json` with Legion's `ChatTemplate` and
compares the result byte for byte with `expected/<template>.json`, which holds what HF
transformers renders (`render_jinja_template`, the code behind
`apply_chat_template(..., tokenize=False)`) or, for `raise_exception`, the error message.

Regenerate with `generate.py` (needs `transformers` and `jinja2`); it also writes `cases.json`.

Templates (`templates/`):

| file | source |
|---|---|
| `qwen3.jinja` | `chat_template` of `Qwen/Qwen3-8B` `tokenizer_config.json` (thinking) |
| `qwen36.jinja` | `tokenizer.chat_template` read from the metadata of a Qwen3.6-35B-A3B GGUF (`qwen35moe`) |
| `gemma3.jinja` | `chat_template.jinja` of `unsloth/gemma-3-4b-it` (the official `google/gemma-3-*` repos are gated) |
| `gemma4.jinja` | `chat_template.jinja` of `google/gemma-4-31B-it` |
| `nemotron35.jinja` | `chat_template.jinja` of `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` (the same text as in llama.cpp's GGUF of it) |
| `deepseek41.jinja` | `tokenizer.chat_template` of the DeepSeek-V4.1-Flash-1017 GSQ-RCO GGUF (the same text as the `chat_template.jinja` beside it; Unsloth's DeepSeek-V4.1 template, adapted) |

Only template text is committed, never model files. `bos_token` / `eos_token` per template are listed in
`generate.py` and mirrored in the test.

Tool schemas and tool-call arguments are parsed into `serde_json` maps with `preserve_order`, so the
cases keep the client's key order inside `parameters` and `arguments`, exactly as HF sees them
(`tools_unsorted_keys` and `tool_call_unsorted_keys` use non-alphabetical keys on purpose).
