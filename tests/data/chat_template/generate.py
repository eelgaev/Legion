#!/usr/bin/env python3
"""Regenerate cases.json and expected/*.json with HF transformers' own Jinja rendering.

    python -m venv venv && venv/bin/pip install transformers jinja2
    venv/bin/python generate.py

For every template in templates/ and every case below, this renders the conversation with
`transformers.utils.chat_template_utils.render_jinja_template` (the code behind
`apply_chat_template(..., tokenize=False)`) and records either the output or the error text of
`raise_exception`. The Rust test (tests/chat_template.rs) must reproduce these byte for byte.
"""
import json
import pathlib

from transformers.utils.chat_template_utils import render_jinja_template

HERE = pathlib.Path(__file__).parent

# name -> (bos_token, eos_token). The values are what the tokenizer of that model reports.
TEMPLATES = {
    "qwen3": ("", "<|im_end|>"),
    "qwen36": ("<|endoftext|>", "<|im_end|>"),
    "gemma3": ("<bos>", "<end_of_turn>"),
    "gemma4": ("<bos>", "<eos>"),
    "nemotron35": ("<s>", "<|im_end|>"),
    "deepseek41": ("<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"),
}

WEATHER = {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. Zürich"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    },
}
SEARCH = {
    "name": "search",
    "description": "Search the web",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results"},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Filters"},
        },
        "required": ["query"],
    },
}


def msg(role, content, reasoning=None, tool_calls=(), tool_call_id=None):
    return {
        "role": role,
        "content": content,
        "reasoning": reasoning,
        "tool_calls": [
            {"id": i, "name": n, "arguments": json.dumps(a, ensure_ascii=False)} for i, n, a in tool_calls
        ],
        "tool_call_id": tool_call_id,
    }


def case(name, messages, system=None, tools=(), add_generation_prompt=True, kwargs=None, templates=None):
    return {
        "name": name,
        "system": system,
        "messages": messages,
        "tools": list(tools),
        "add_generation_prompt": add_generation_prompt,
        "kwargs": kwargs or {},
        "templates": templates or list(TEMPLATES),
    }


MULTI = [
    msg("user", "What is 2+2?"),
    msg("assistant", "4.", reasoning="Simple arithmetic."),
    msg("user", "And times 3?"),
    msg("assistant", "12.", reasoning="4*3=12."),
    msg("user", "Thanks, and plus 1?"),
]
CALL = [
    msg("user", "Weather in Zürich?"),
    msg(
        "assistant",
        "Let me check.",
        reasoning="I should call the weather tool.",
        tool_calls=[("call_1", "get_weather", {"city": "Zürich", "unit": "celsius"})],
    ),
    msg("tool", '{"temp": 18, "sky": "clear"}', tool_call_id="call_1"),
]
PARALLEL = [
    msg("user", "Weather in Paris and news about it?"),
    msg(
        "assistant",
        "",
        tool_calls=[
            ("call_a", "get_weather", {"city": "Paris"}),
            ("call_b", "search", {"query": "Paris news", "limit": 3, "tags": ["a", "b"], "opts": {"x": [1, 2.5, None, True]}}),
        ],
    ),
    msg("tool", "sunny, 21C", tool_call_id="call_a"),
    msg("tool", "Nothing special today.", tool_call_id="call_b"),
]

# Keys deliberately not in alphabetical order, at every level: Legion must keep the client's order.
UNSORTED = {
    "name": "book_flight",
    "description": "Book a flight",
    "parameters": {
        "type": "object",
        "required": ["to", "from"],
        "properties": {
            "to": {"type": "string", "description": "Destination"},
            "from": {"type": "string", "description": "Origin"},
            "date": {"type": "string", "description": "ISO date"},
            "bags": {"type": "integer", "description": "Checked bags"},
        },
    },
}
UNSORTED_CALL = [
    msg("user", "Book me to Oslo from Zürich"),
    msg(
        "assistant",
        "",
        tool_calls=[("call_u", "book_flight", {"to": "Oslo", "from": "Zürich", "bags": 2, "date": "2026-11-02"})],
    ),
    msg("tool", '{"status": "booked"}', tool_call_id="call_u"),
]

# Gemma 3 has no tool role: its template raises on it, which error_roles_not_alternating covers.
TOOLS_OK = ["qwen3", "qwen36", "gemma4", "nemotron35", "deepseek41"]

CASES = [
    case("plain", [msg("user", "Hello there!")]),
    case("plain_no_generation_prompt", [msg("user", "Hello there!")], add_generation_prompt=False),
    case("system", [msg("user", "Hi")], system="You are a helpful assistant."),
    case("system_unicode_whitespace", [msg("user", "  padded  \n")], system="Réponds en français, s'il te plaît 日本語\n"),
    case("multi_turn_reasoning", MULTI),
    case("multi_turn_reasoning_system", MULTI, system="Be brief."),
    case("multi_turn_final_assistant", MULTI[:4], add_generation_prompt=False),
    case("enable_thinking_false", [msg("user", "Hi")], kwargs={"enable_thinking": False}),
    case("enable_thinking_true", [msg("user", "Hi")], kwargs={"enable_thinking": True}),
    case("enable_thinking_true_system", [msg("user", "Hi")], system="Sys.", kwargs={"enable_thinking": True}),
    case("preserve_thinking", MULTI, kwargs={"preserve_thinking": True}, templates=["qwen36"]),
    case("tools", [msg("user", "Weather in Zürich?")], tools=[WEATHER]),
    case("tools_system", [msg("user", "Weather in Zürich?")], system="Use tools.", tools=[WEATHER, SEARCH]),
    case("tool_call_result", CALL, tools=[WEATHER], templates=TOOLS_OK),
    case("tool_call_result_thinking_false", CALL, tools=[WEATHER], kwargs={"enable_thinking": False}, templates=TOOLS_OK),
    case("tool_call_result_thinking_true", CALL, tools=[WEATHER], kwargs={"enable_thinking": True}, templates=TOOLS_OK),
    case("tool_call_continued", CALL + [msg("assistant", "It is 18 C and clear."), msg("user", "Thanks!")], tools=[WEATHER], templates=TOOLS_OK),
    case("parallel_tool_calls", PARALLEL, tools=[WEATHER, SEARCH], system="Use tools.", templates=TOOLS_OK),
    case("parallel_tool_calls_no_generation_prompt", PARALLEL, tools=[WEATHER, SEARCH], add_generation_prompt=False, templates=TOOLS_OK),
    case("tools_unsorted_keys", UNSORTED_CALL[:1], tools=[UNSORTED], templates=TOOLS_OK),
    case("tool_call_unsorted_keys", UNSORTED_CALL, tools=[UNSORTED], templates=TOOLS_OK),
    case("inline_think_in_content", [msg("user", "q"), msg("assistant", "<think>\nhidden\n</think>\n\nanswer"), msg("user", "q2")], templates=["qwen3", "qwen36"]),
    # raise_exception paths: must surface as Err with the template's message.
    case("error_roles_not_alternating", [msg("user", "a"), msg("user", "b")], templates=["gemma3"]),
    case("error_no_messages", [], templates=["qwen36"]),
    # DeepSeek-V4.1-Flash: its own thinking switch and effort levels.
    case("thinking_mode_chat", MULTI, kwargs={"thinking_mode": "chat"}, templates=["deepseek41"]),
    case("reasoning_effort_max", [msg("user", "Hi")], kwargs={"reasoning_effort": "max"}, templates=["deepseek41"]),
    case("reasoning_effort_numeric", [msg("user", "Hi")], kwargs={"reasoning_effort": "30"}, templates=["deepseek41"]),
    case("drop_thinking_false", MULTI, kwargs={"drop_thinking": False}, templates=["deepseek41"]),
    case("error_reasoning_effort", [msg("user", "Hi")], kwargs={"reasoning_effort": "huge"}, templates=["deepseek41"]),
]


def hf_messages(system, messages):
    out = []
    if system is not None:
        out.append({"role": "system", "content": system})
    for m in messages:
        d = {"role": m["role"], "content": m["content"]}
        if m["reasoning"] is not None:
            d["reasoning_content"] = m["reasoning"]
        if m["tool_calls"]:
            d["tool_calls"] = [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {"name": c["name"], "arguments": json.loads(c["arguments"])},
                }
                for c in m["tool_calls"]
            ]
        if m["tool_call_id"] is not None:
            d["tool_call_id"] = m["tool_call_id"]
        out.append(d)
    return out


def main():
    (HERE / "cases.json").write_text(json.dumps(CASES, indent=2, ensure_ascii=False) + "\n")
    (HERE / "expected").mkdir(exist_ok=True)
    for tname, (bos, eos) in TEMPLATES.items():
        source = (HERE / "templates" / f"{tname}.jinja").read_text()
        expected = {}
        for c in CASES:
            if tname not in c["templates"]:
                continue
            tools = [{"type": "function", "function": t} for t in c["tools"]] or None
            try:
                rendered, _ = render_jinja_template(
                    [hf_messages(c["system"], c["messages"])],
                    tools=tools,
                    chat_template=source,
                    add_generation_prompt=c["add_generation_prompt"],
                    bos_token=bos,
                    eos_token=eos,
                    **c["kwargs"],
                )
                expected[c["name"]] = {"output": rendered[0]}
            except Exception as e:  # noqa: BLE001 - raise_exception surfaces as a TemplateError
                expected[c["name"]] = {"error": str(e)}
        (HERE / "expected" / f"{tname}.json").write_text(json.dumps(expected, indent=2, ensure_ascii=False) + "\n")
        print(tname, len(expected), "cases")


if __name__ == "__main__":
    main()
