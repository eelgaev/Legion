# API Reference

This page lists the HTTP endpoints supported by Legion, the request fields honoured by each, and examples of requests and responses.

Unknown fields in request bodies are accepted and ignored.

Sampling fields a request leaves out (`temperature`, `top_p`, `top_k`, `min_p`, the penalties) take the served model's own defaults, not OpenAI's: see [Sampling defaults](serving.md#sampling-defaults). A field the request sets always wins, `temperature: 0` included.

## `/v1/chat/completions`

### Fields honoured

| Field | Type | Notes |
|---|---|---|
| `model` | string | Required. |
| `messages` | array | Required. List of chat messages. |
| `max_tokens` | integer | Optional. |
| `max_completion_tokens` | integer | Optional. |
| `temperature` | float | Optional. |
| `top_p` | float | Optional. |
| `top_k` | integer | Optional. |
| `min_p` | float | Optional. |
| `seed` | integer | Optional. |
| `presence_penalty` | float | Optional. |
| `frequency_penalty` | float | Optional. |
| `repetition_penalty` | float | Optional. |
| `logprobs` | boolean | Default `false`. |
| `top_logprobs` | integer | Optional. |
| `n` | integer | Optional. |
| `stop` | string | array | Optional. |
| `stream` | boolean | Default `false`. |
| `stream_options` | object | Optional. |
| `tools` | array | Default `[]`. |
| `tool_choice` | any | Optional. |
| `reasoning_effort` | string | Optional. |
| `chat_template_kwargs` | object | Optional. |
| `response_format` | object | `text`, `json_object`, or `json_schema`; see [JSON output](#json-output). |
| `priority` | integer | Default `0`. Queue order under `--scheduling-policy priority`, lower first; see [Priority and fair share](serving.md#priority-and-fair-share). |
| `user` | string | Optional. The client under `--fair-share user`. |

### Example

**Request**
```sh
curl -s localhost:8090/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"mock","messages":[{"role":"user","content":"hi"}]}'
```

**Response**
```json
{
  "id": "chatcmpl-b157fb2735e44a20a3b8586cca34b7b0",
  "object": "chat.completion",
  "created": 1792166603,
  "model": "mock",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "You said: hi"
      },
      "finish_reason": "stop",
      "logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 1,
    "completion_tokens": 3,
    "total_tokens": 4
  },
  "timings": {
    "prompt_n": 1,
    "prompt_ms": 6.178573,
    "prompt_per_second": 161.8496698185811,
    "predicted_n": 3,
    "predicted_ms": 17.75115,
    "predicted_per_second": 169.00313500815443,
    "queue_ms": 0.031
  }
}
```

`timings.queue_ms` is the time the request waited for a batch slot before its prefill began.

### Errors

**JSON shape**
```json
{
  "error": {
    "message": "string",
    "type": "string",
    "param": "string | null",
    "code": "string | null"
  }
}
```

**Example**
```json
{
  "error": {
    "message": "Failed to deserialize the JSON body into the target type: messages: invalid type: string \"not-a-list\", expected a sequence at line 1 column 39",
    "type": "invalid_request_error",
    "param": null,
    "code": null
  }
}
```

---

## `/v1/completions`

### Fields honoured

| Field | Type | Notes |
|---|---|---|
| `model` | string | Required. |
| `prompt` | string | array | Required. |
| `max_tokens` | integer | Optional. |
| `temperature` | float | Optional. |
| `top_p` | float | Optional. |
| `top_k` | integer | Optional. |
| `min_p` | float | Optional. |
| `seed` | integer | Optional. |
| `presence_penalty` | float | Optional. |
| `frequency_penalty` | float | Optional. |
| `repetition_penalty` | float | Optional. |
| `n` | integer | Optional. |
| `stop` | string | array | Optional. |
| `stream` | boolean | Default `false`. |
| `stream_options` | object | Optional. |
| `response_format` | object | `text`, `json_object`, or `json_schema`; see [JSON output](#json-output). |
| `echo` | boolean | Default `false`. |
| `logprobs` | integer | Optional. |
| `priority` | integer | Default `0`. Queue order under `--scheduling-policy priority`, lower first. |
| `user` | string | Optional. The client under `--fair-share user`. |

### Example

**Request**
```sh
curl -s localhost:8090/v1/completions -H 'content-type: application/json' \
  -d '{"model":"mock","prompt":"hi"}'
```

**Response**
```json
{
  "id": "cmpl-759c2781ac224667864bbd09f05687b5",
  "object": "text_completion",
  "created": 1792166607,
  "model": "mock",
  "choices": [
    {
      "index": 0,
      "text": "You said: hi",
      "finish_reason": "stop",
      "logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 1,
    "completion_tokens": 3,
    "total_tokens": 4
  },
  "timings": {
    "prompt_n": 1,
    "prompt_ms": 6.1549320000000005,
    "prompt_per_second": 162.47133193348031,
    "predicted_n": 3,
    "predicted_ms": 18.309815999999998,
    "predicted_per_second": 163.84654001984512
  }
}
```

### Errors

**JSON shape**
```json
{
  "error": {
    "message": "string",
    "type": "string",
    "param": "string | null",
    "code": "string | null"
  }
}
```

**Example**
```json
{
  "error": {
    "message": "Failed to deserialize the JSON body into the target type: prompt: data did not match any variant of untagged enum OneOrMany at line 1 column 29",
    "type": "invalid_request_error",
    "param": null,
    "code": null
  }
}
```

---

## `/v1/messages`

### Fields honoured

| Field | Type | Notes |
|---|---|---|
| `model` | string | Required. |
| `max_tokens` | integer | Required. |
| `system` | string | array | Optional. |
| `messages` | array | Required. |
| `stop_sequences` | array | Default `[]`. |
| `temperature` | float | Optional. |
| `top_p` | float | Optional. |
| `top_k` | integer | Optional. |
| `stream` | boolean | Default `false`. |
| `metadata` | any | Optional. `metadata.user_id` is the client under `--fair-share user`. |
| `tools` | array | Default `[]`. |
| `tool_choice` | any | Optional. |
| `thinking` | object | Optional. |
| `priority` | integer | Default `0`. Queue order under `--scheduling-policy priority`, lower first (not an Anthropic field). |

### Example

**Request**
```sh
curl -s localhost:8090/v1/messages -H 'content-type: application/json' \
  -d '{"model":"mock","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}'
```

**Response**
```json
{
  "id": "msg_4aa122ff58574baf859d9e7fdc4564af",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "You said: hi"
    }
  ],
  "model": "mock",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 1,
    "output_tokens": 3,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0
  }
}
```

### Errors

**JSON shape**
```json
{
  "type": "string",
  "error": {
    "type": "string",
    "message": "string"
  }
}
```

**Example**
```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "Failed to deserialize the JSON body into the target type: missing field `max_tokens` at line 1 column 60"
  }
}
```

---

## `/v1/messages/count_tokens`

### Fields honoured

| Field | Type | Notes |
|---|---|---|
| `model` | string | Required. |
| `max_tokens` | integer | Optional (ignored). |
| `system` | string | array | Optional. |
| `messages` | array | Required. |
| `stop_sequences` | array | Default `[]`. |
| `temperature` | float | Optional. |
| `top_p` | float | Optional. |
| `top_k` | integer | Optional. |
| `stream` | boolean | Default `false`. |
| `metadata` | any | Optional. |
| `tools` | array | Default `[]`. |
| `tool_choice` | any | Optional. |
| `thinking` | object | Optional. |

### Example

**Request**
```sh
curl -s localhost:8090/v1/messages/count_tokens -H 'content-type: application/json' \
  -d '{"model":"mock","messages":[{"role":"user","content":"hi"}]}'
```

**Response**
```json
{
  "input_tokens": 1
}
```

### Errors

**JSON shape**
```json
{
  "type": "string",
  "error": {
    "type": "string",
    "message": "string"
  }
}
```

**Example**
```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "invalid type: string \"hi\", expected a sequence"
  }
}
```

---

## `/v1/responses`

### Fields honoured

| Field | Type | Notes |
|---|---|---|
| `model` | string | Required. |
| `input` | string | array | Required. |
| `instructions` | string | Optional. |
| `tools` | array | Default `[]`. |
| `max_output_tokens` | integer | Optional. |
| `temperature` | float | Optional. |
| `top_p` | float | Optional. |
| `stream` | boolean | Default `false`. |
| `reasoning` | object | Optional. |
| `text.format` | object | JSON output format; see [JSON output](#json-output). |
| `previous_response_id` | string | Optional. |
| `store` | boolean | Optional. |
| `priority` | integer | Default `0`. Queue order under `--scheduling-policy priority`, lower first. |
| `user` | string | Optional. The client under `--fair-share user`. |

### Example

**Request**
```sh
curl -s localhost:8090/v1/responses -H 'content-type: application/json' \
  -d '{"model":"mock","input":"hi"}'
```

**Response**
```json
{
  "id": "resp_ec126e12802b4ea9a14bd83821b98101",
  "object": "response",
  "created_at": 1792166619,
  "status": "completed",
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "max_output_tokens": null,
  "model": "mock",
  "output": [
    {
      "type": "message",
      "id": "msg_b34396ee35764340936501ad1e91c44a",
      "role": "assistant",
      "status": "completed",
      "content": [
        {
          "type": "output_text",
          "text": "You said: hi",
          "annotations": []
        }
      ]
    }
  ],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {
    "effort": null,
    "summary": null
  },
  "store": false,
  "temperature": null,
  "text": {
    "format": {
      "type": "text"
    }
  },
  "tool_choice": "auto",
  "tools": [],
  "top_p": null,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 1,
    "input_tokens_details": {
      "cached_tokens": 0
    },
    "output_tokens": 3,
    "output_tokens_details": {
      "reasoning_tokens": 0
    },
    "total_tokens": 4
  },
  "user": null,
  "metadata": {}
}
```

### Errors

**JSON shape**
```json
{
  "error": {
    "message": "string",
    "type": "string",
    "param": "string | null",
    "code": "string | null"
  }
}
```

**Example**
```json
{
  "error": {
    "message": "Failed to deserialize the JSON body into the target type: input: data did not match any variant of untagged enum ResponsesInput at line 1 column 28",
    "type": "invalid_request_error",
    "param": null,
    "code": null
  }
}
```

---

## `/v1/models`

### Fields honoured
None. This is a `GET` endpoint.

### Example

**Request**
```sh
curl -s localhost:8090/v1/models
```

**Response**
```json
{
  "object": "list",
  "data": [
    {
      "id": "mock",
      "object": "model",
      "created": 1792166596,
      "owned_by": "legion",
      "capabilities": [
        "vision"
      ]
    }
  ]
}
```

---

## `/health`

### Fields honoured
None. This is a `GET` endpoint.

### Example

**Request**
```sh
curl -s localhost:8090/health
```

**Response**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "git": "783614b",
  "models": [
    "mock"
  ]
}
```

---

## `/metrics`

### Fields honoured
None. This is a `GET` endpoint.

### Example

**Request**
```sh
curl -s localhost:8090/metrics
```

**Response**
```text
# HELP legion_http_requests HTTP requests by matched route and status class.
# TYPE legion_http_requests counter
legion_http_requests_total{route="/v1/responses",status="2xx"} 1
legion_http_requests_total{route="/v1/chat/completions",status="2xx"} 1
legion_http_requests_total{route="/v1/messages",status="2xx"} 1
legion_http_requests_total{route="/v1/models",status="2xx"} 1
legion_http_requests_total{route="/v1/messages/count_tokens",status="2xx"} 1
legion_http_requests_total{route="/health",status="2xx"} 2
legion_http_requests_total{route="/v1/completions",status="2xx"} 1
...
```

## JSON output

Chat and legacy completions accept `"response_format":{"type":"json_object"}` for a
JSON object, or a schema such as:

```json
{"response_format":{"type":"json_schema","json_schema":{"name":"answer","schema":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"boolean"}},"required":["a","b"]},"strict":true}}}
```

For `/v1/responses`, use `text.format` with `name`, `schema`, and optional `strict`
directly alongside `type`:

```json
{"text":{"format":{"type":"json_schema","name":"answer","schema":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]},"strict":true}}}
```

`{"type":"text"}` preserves normal output. `strict` is accepted and ignored: JSON Schema
is always enforced. Unknown format types, missing schemas, and unsupported schema features
(such as `pattern`) return HTTP 400 with `error.param` set to `response_format` or `text.format`,
including for streaming requests.

The Qwen3.5 engine constrains each sampled token with the grammar engine. JSON requests
turn thinking off (`thinking = Some(false)`), overriding reasoning settings and
`chat_template_kwargs.enable_thinking`, so the output is the JSON document. Speculative
drafting is disabled for these requests. Reaching `max_tokens` or the context limit can
truncate the document (`finish_reason: "length"`); user stop strings can also end it early.
The grammar's supported schema subset and nesting limits are documented in
[`src/grammar/mod.rs`](../src/grammar/mod.rs).

Tools with `"strict": true` get the same enforcement for their calls (Qwen3-Coder call format);
see [Constrained output](serving.md#constrained-output-json-and-strict-tool-calls) for what is
constrained, the supported schemas and `--grammar-threads`.

The mock only proves request plumbing: it emits `{}` for JSON object mode and object schemas
without required properties, and its usual text otherwise. The Anthropic API is unchanged.
