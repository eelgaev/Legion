"""Official openai + anthropic SDKs against a running `legion serve` (mock engine).

Configured by run.sh through COMPAT_BASE_URL, COMPAT_AUTH_BASE_URL and COMPAT_API_KEY.
Every check runs even if an earlier one fails; the exit code is non-zero on any failure.
"""

import json
import os
import sys
import traceback

import anthropic
import openai

BASE = os.environ["COMPAT_BASE_URL"]
AUTH_BASE = os.environ["COMPAT_AUTH_BASE_URL"]
KEY = os.environ["COMPAT_API_KEY"]
MODEL = "mock"

oa = openai.OpenAI(base_url=f"{BASE}/v1", api_key="unused", max_retries=0)
an = anthropic.Anthropic(base_url=BASE, api_key="unused", max_retries=0)

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def user(text):
    return [{"role": "user", "content": text}]


def get_weather_tool():
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


@check
def openai_models():
    ids = [m.id for m in oa.models.list().data]
    assert MODEL in ids, ids
    assert oa.models.retrieve(MODEL).id == MODEL


@check
def openai_chat():
    r = oa.chat.completions.create(model=MODEL, messages=user("hello there"))
    assert r.choices[0].message.content == "You said: hello there", r
    assert r.choices[0].finish_reason == "stop"
    assert r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens > 0


@check
def openai_chat_stream_usage():
    stream = oa.chat.completions.create(
        model=MODEL,
        messages=user("stream me"),
        stream=True,
        stream_options={"include_usage": True},
    )
    text, usage, finish = "", None, None
    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        for c in chunk.choices:
            text += c.delta.content or ""
            finish = c.finish_reason or finish
    assert text == "You said: stream me", text
    assert finish == "stop", finish
    assert usage is not None and usage.completion_tokens > 0, usage


@check
def openai_chat_logprobs():
    r = oa.chat.completions.create(
        model=MODEL, messages=user("a b c"), logprobs=True, top_logprobs=2
    )
    content = r.choices[0].logprobs.content
    assert content, r
    assert len(content[0].top_logprobs) == 2, content[0]
    assert content[0].logprob <= 0


@check
def openai_chat_tool_round_trip():
    tools = [get_weather_tool()]
    r = oa.chat.completions.create(
        model=MODEL,
        messages=user('/tool get_weather {"city":"Paris"}'),
        tools=tools,
    )
    choice = r.choices[0]
    assert choice.finish_reason == "tool_calls", choice
    call = choice.message.tool_calls[0]
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"city": "Paris"}
    messages = user('/tool get_weather {"city":"Paris"}') + [
        choice.message.model_dump(exclude_none=True),
        {"role": "tool", "tool_call_id": call.id, "content": "sunny"},
    ]
    r2 = oa.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    assert r2.choices[0].finish_reason == "stop", r2


@check
def openai_chat_tool_stream():
    stream = oa.chat.completions.create(
        model=MODEL,
        messages=user('/tool get_weather {"city":"Rome"}'),
        tools=[get_weather_tool()],
        stream=True,
    )
    name, args, finish = "", "", None
    for chunk in stream:
        for c in chunk.choices:
            for tc in c.delta.tool_calls or []:
                name += tc.function.name or ""
                args += tc.function.arguments or ""
            finish = c.finish_reason or finish
    assert name == "get_weather" and json.loads(args) == {"city": "Rome"}, (name, args)
    assert finish == "tool_calls", finish


@check
def openai_reasoning_effort():
    r = oa.chat.completions.create(
        model=MODEL, messages=user("think"), reasoning_effort="low"
    )
    assert r.choices[0].message.content, r


@check
def openai_completions():
    r = oa.completions.create(model=MODEL, prompt="one two three", max_tokens=16)
    assert r.choices[0].text, r
    assert r.usage.total_tokens > 0
    s = oa.completions.create(model=MODEL, prompt="one two", max_tokens=16, stream=True)
    assert "".join(c.choices[0].text for c in s if c.choices), "empty completion stream"


@check
def openai_bad_temperature():
    try:
        oa.chat.completions.create(model=MODEL, messages=user("hi"), temperature=3)
    except openai.BadRequestError as e:
        assert e.status_code == 400
    else:
        raise AssertionError("temperature=3 was accepted")


@check
def openai_any_model_name_resolves():
    # With a single served model any name maps to it (clients hard-code model names).
    r = oa.chat.completions.create(model="gpt-4o", messages=user("hi"))
    assert r.model == MODEL, r.model


@check
def responses_create():
    r = oa.responses.create(model=MODEL, input="hello responses")
    assert r.status == "completed", r.status
    assert r.output_text == "You said: hello responses", r.output_text
    assert r.usage.total_tokens > 0


@check
def responses_tool_call():
    r = oa.responses.create(
        model=MODEL,
        input='/tool get_weather {"city":"Oslo"}',
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "description": "Weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ],
    )
    calls = [o for o in r.output if o.type == "function_call"]
    assert calls and calls[0].name == "get_weather", r.output
    assert json.loads(calls[0].arguments) == {"city": "Oslo"}


@check
def responses_stream():
    with oa.responses.stream(model=MODEL, input="stream responses") as stream:
        deltas = "".join(
            e.delta for e in stream if e.type == "response.output_text.delta"
        )
        final = stream.get_final_response()
    assert deltas == "You said: stream responses", deltas
    assert final.output_text == deltas
    assert final.status == "completed"


@check
def anthropic_create():
    m = an.messages.create(model=MODEL, max_tokens=64, messages=user("hello claude"))
    assert m.content[0].type == "text"
    assert m.content[0].text == "You said: hello claude", m.content
    assert m.stop_reason == "end_turn", m.stop_reason
    assert m.usage.output_tokens > 0


@check
def anthropic_system_and_blocks():
    m = an.messages.create(
        model=MODEL,
        max_tokens=64,
        system="be brief",
        messages=[{"role": "user", "content": [{"type": "text", "text": "blocks"}]}],
    )
    assert m.content[0].text == "You said: blocks", m.content


@check
def anthropic_stream():
    with an.messages.stream(
        model=MODEL, max_tokens=64, messages=user("stream claude")
    ) as stream:
        text = "".join(stream.text_stream)
        final = stream.get_final_message()
    assert text == "You said: stream claude", text
    assert final.content[0].text == text
    assert final.stop_reason == "end_turn"
    assert final.usage.output_tokens > 0


@check
def anthropic_thinking():
    m = an.messages.create(
        model=MODEL,
        max_tokens=2048,
        thinking={"type": "enabled", "budget_tokens": 1024},
        messages=user("think hard"),
    )
    kinds = [b.type for b in m.content]
    assert kinds[0] == "thinking" and "text" in kinds, kinds
    assert m.content[0].thinking, m.content[0]


@check
def anthropic_thinking_stream():
    with an.messages.stream(
        model=MODEL,
        max_tokens=2048,
        thinking={"type": "enabled", "budget_tokens": 1024},
        messages=user("think streaming"),
    ) as stream:
        events = list(stream)
        final = stream.get_final_message()
    deltas = {e.delta.type for e in events if e.type == "content_block_delta"}
    assert "thinking_delta" in deltas and "text_delta" in deltas, deltas
    assert [b.type for b in final.content][0] == "thinking", final.content


@check
def anthropic_tool_use():
    tool = {
        "name": "get_weather",
        "description": "Weather for a city",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
    m = an.messages.create(
        model=MODEL,
        max_tokens=256,
        tools=[tool],
        messages=user('/tool get_weather {"city":"Lima"}'),
    )
    assert m.stop_reason == "tool_use", m.stop_reason
    block = next(b for b in m.content if b.type == "tool_use")
    assert block.name == "get_weather" and block.input == {"city": "Lima"}, block
    m2 = an.messages.create(
        model=MODEL,
        max_tokens=256,
        tools=[tool],
        messages=user('/tool get_weather {"city":"Lima"}')
        + [
            {"role": "assistant", "content": [b.model_dump() for b in m.content]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": block.id, "content": "rainy"},
                ],
            },
        ],
    )
    assert m2.stop_reason == "end_turn", m2


@check
def anthropic_count_tokens():
    r = an.messages.count_tokens(model=MODEL, messages=user("count these tokens"))
    assert r.input_tokens > 0, r


@check
def anthropic_bad_request():
    try:
        an.messages.create(model=MODEL, max_tokens=64, messages=[])
    except anthropic.BadRequestError:
        pass
    else:
        raise AssertionError("empty messages were accepted")


@check
def auth():
    good_oa = openai.OpenAI(base_url=f"{AUTH_BASE}/v1", api_key=KEY, max_retries=0)
    assert good_oa.chat.completions.create(model=MODEL, messages=user("hi")).choices
    good_an = anthropic.Anthropic(base_url=AUTH_BASE, api_key=KEY, max_retries=0)
    assert good_an.messages.create(model=MODEL, max_tokens=16, messages=user("hi")).content

    bad_oa = openai.OpenAI(base_url=f"{AUTH_BASE}/v1", api_key="wrong", max_retries=0)
    try:
        bad_oa.chat.completions.create(model=MODEL, messages=user("hi"))
    except openai.AuthenticationError:
        pass
    else:
        raise AssertionError("openai: wrong key accepted")
    bad_an = anthropic.Anthropic(base_url=AUTH_BASE, api_key="wrong", max_retries=0)
    try:
        bad_an.messages.create(model=MODEL, max_tokens=16, messages=user("hi"))
    except anthropic.AuthenticationError:
        pass
    else:
        raise AssertionError("anthropic: wrong key accepted")


def main():
    failed = []
    for fn in CHECKS:
        try:
            fn()
            print(f"ok    {fn.__name__}")
        except Exception:
            failed.append(fn.__name__)
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
