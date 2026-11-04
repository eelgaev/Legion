#!/usr/bin/env bash
# Runs the official openai and anthropic Python SDKs against `legion serve` (mock engine).
# Needs network for `pip install`; not part of `cargo test`.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

pids=()
cleanup() {
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

free_port() {
    python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'
}

wait_healthy() {
    local port=$1
    for _ in $(seq 1 100); do
        if curl -fs "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.2
    done
    echo "server on port $port did not become healthy" >&2
    return 1
}

cargo build --release
bin="$root/target/release/legion"

venv="$root/target/compat-venv"
if [ ! -x "$venv/bin/python" ]; then
    python3 -m venv "$venv"
fi
"$venv/bin/pip" install --quiet --upgrade openai anthropic

open_port="$(free_port)"
auth_port="$(free_port)"
api_key="compat-test-key"

"$bin" serve --port "$open_port" &
pids+=("$!")
"$bin" serve --port "$auth_port" --api-key "$api_key" &
pids+=("$!")
wait_healthy "$open_port"
wait_healthy "$auth_port"

COMPAT_BASE_URL="http://127.0.0.1:$open_port" \
COMPAT_AUTH_BASE_URL="http://127.0.0.1:$auth_port" \
COMPAT_API_KEY="$api_key" \
    "$venv/bin/python" "$root/tests/compat/sdk.py"
