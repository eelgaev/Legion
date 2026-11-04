# Clients

Legion speaks the OpenAI and Anthropic wire formats, so any client that accepts a custom base URL
works. The snippets below run against `legion serve` on `localhost:8090` (mock engine); replace
`localhost` with your server's address. Where a key is shown, use your `--api-key` value, or any
non-empty string when the server runs without one.

## curl

OpenAI chat completions:

```sh
curl -s localhost:8090/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"mock","messages":[{"role":"user","content":"hi"}]}'
```

Streaming (`-N` turns off curl's buffering):

```sh
curl -sN localhost:8090/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"mock","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Anthropic Messages:

```sh
curl -s localhost:8090/v1/messages -H 'content-type: application/json' \
  -H 'x-api-key: none' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"mock","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}'
```

Streaming:

```sh
curl -sN localhost:8090/v1/messages -H 'content-type: application/json' \
  -H 'x-api-key: none' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"mock","max_tokens":64,"messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Endpoints: `GET /health`, `GET /v1/models`, `GET /v1/models/{id}`, `POST /v1/chat/completions`,
`POST /v1/completions`, `POST /v1/messages`, `POST /v1/responses` (OpenAI Responses API, stateless).

## open-webui

Point open-webui at Legion's OpenAI-compatible API:

```sh
docker run -d -p 3000:8080 -e OPENAI_API_BASE_URL=http://host:8090/v1 \
  -v open-webui:/app/backend/data ghcr.io/open-webui/open-webui:main
```

then open `http://localhost:3000`. Replace `host` with an address the container can reach (the
server's LAN address, or `host.docker.internal` with `--add-host=host.docker.internal:host-gateway`);
`localhost` inside the container is the container itself, though `--network=host` instead of `-p`
makes it work and serves open-webui on port 8080. The API key (Admin Settings → Connections →
OpenAI API) is your `--api-key`, or any non-empty value when the server is open.

## openai-python

```sh
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8090/v1", api_key="none")

reply = client.chat.completions.create(
    model="mock",
    messages=[{"role": "user", "content": "hi"}],
)
print(reply.choices[0].message.content)  # You said: hi

stream = client.chat.completions.create(
    model="mock",
    messages=[{"role": "user", "content": "hi"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

## anthropic-python

```sh
pip install anthropic
```

```python
import anthropic

client = anthropic.Anthropic(base_url="http://localhost:8090", api_key="none")

message = client.messages.create(
    model="mock",
    max_tokens=64,
    messages=[{"role": "user", "content": "hi"}],
)
print(message.content[0].text)  # You said: hi

with client.messages.stream(
    model="mock", max_tokens=64, messages=[{"role": "user", "content": "hi"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="")
```

Note the base URLs differ: the OpenAI SDK appends `/chat/completions` to `.../v1`, while the
Anthropic SDK appends `/v1/messages` to the server root.

## Claude Code

```sh
export ANTHROPIC_BASE_URL=http://localhost:8090   # no /v1; Claude Code appends /v1/messages
export ANTHROPIC_AUTH_TOKEN=none                  # sent as Authorization: Bearer; must match --api-key when set
export ANTHROPIC_MODEL=mock                       # any name works while a single model is served
claude -p "hello"                                 # You said: hello
```
