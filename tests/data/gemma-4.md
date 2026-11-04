The three `ggml-vocab-gemma-4.gguf*` fixtures are unmodified copies from
`models/` in llama.cpp revision `f3363593f457169cdfbdfe22ea399b1cc139ba18`.
Their MIT license is included in `gemma-4.LICENSE`.

The `.inp` and `.out` files contain 46 cases, run with `add_special = false`
and `parse_special = false`, as in llama.cpp's `tests/test-tokenizer-0.cpp`.
