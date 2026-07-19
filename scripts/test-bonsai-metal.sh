#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Serve + smoke-test PrismML Bonsai-27B (1-bit Q1_0 GGUF) on Apple Silicon
# via the Atlas metal backend — the Metal counterpart of
# test-qwen36-tool-image.sh. Text + tool-call coverage across all three API
# surfaces; the vision (image) tests run only when BONSAI_VISION=1 (the
# mmproj tower port).
#
# Build first:
#   ATLAS_TARGET_HW=metal ATLAS_TARGET_MODEL=bonsai-27b ATLAS_TARGET_QUANT=q1_0 \
#     cargo build --release -p spark-server --no-default-features --features metal
#
# Model dir (default ~/models/bonsai-27b-atlas) must contain:
#   config.json           MLX pack config with the `quantization` blocks stripped
#   tokenizer.json, tokenizer_config.json, vocab.json, merges.txt,
#   chat_template.jinja   (from the Bonsai-27B-mlx pack)
#   Bonsai-27B-Q1_0.gguf  (symlink into the Bonsai-demo checkout is fine)
#   Bonsai-27B-mmproj-Q8_0.gguf   (optional; vision)
#
#   ./scripts/test-bonsai-metal.sh
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models/bonsai-27b-atlas}"
SERVED_NAME="${SERVED_NAME:-bonsai-27b-atlas}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8899}"
BASE_URL="http://${HOST}:${PORT}"
SPARK_BIN="${SPARK_BIN:-./target/release/spark}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
START_SERVER="${START_SERVER:-auto}"   # auto|yes|no
READY_TIMEOUT="${READY_TIMEOUT:-300}"
BONSAI_VISION="${BONSAI_VISION:-0}"

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/atlas-bonsai-test.XXXXXX")"
SERVER_LOG="${WORKDIR}/server.log"
SERVER_PID=""
STARTED_SERVER=0

log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
ok()   { printf '\033[1;32m  ✓ %s\033[0m\n' "$*" >&2; }
warn() { printf '\033[1;33m  ! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m[FATAL] %s\033[0m\n' "$*" >&2; exit 1; }

cleanup() {
  if [[ "${STARTED_SERVER}" == "1" && -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    log "stopping server (pid ${SERVER_PID})"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

command -v curl >/dev/null || die "missing curl"
command -v python3 >/dev/null || die "missing python3"
export SERVED_NAME

server_up() { curl -fsS -m 3 "${BASE_URL}/health" >/dev/null 2>&1; }

start_server() {
  [[ -x "${SPARK_BIN}" ]] || die "spark binary not found at '${SPARK_BIN}' — build with:
    ATLAS_TARGET_HW=metal ATLAS_TARGET_MODEL=bonsai-27b ATLAS_TARGET_QUANT=q1_0 \\
      cargo build --release -p spark-server --no-default-features --features metal"
  [[ -d "${MODEL_DIR}" ]] || die "model dir '${MODEL_DIR}' missing (see header)"

  log "starting: ${SPARK_BIN} serve --model-from-path ${MODEL_DIR} (logs → ${SERVER_LOG})"
  "${SPARK_BIN}" serve --model-from-path "${MODEL_DIR}" \
    --model-name "${SERVED_NAME}" \
    --bind "${HOST}" --port "${PORT}" \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --max-num-seqs 2 --max-batch-size 2 \
    --kv-cache-dtype bf16 \
    --disable-thinking \
    >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  STARTED_SERVER=1

  log "waiting up to ${READY_TIMEOUT}s for weights to load…"
  local deadline=$(( $(date +%s) + READY_TIMEOUT ))
  while ! server_up; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -n 40 "${SERVER_LOG}" >&2 || true
      die "server exited before ready (see ${SERVER_LOG})"
    fi
    [[ $(date +%s) -lt ${deadline} ]] || { tail -n 40 "${SERVER_LOG}" >&2; die "readiness timeout"; }
    sleep 3
  done
  ok "server ready"
}

chat() {
  # Unique response file per call — chat() runs concurrently in test 9.
  local body_file="$1" raw
  raw="$(mktemp "${WORKDIR}/resp.XXXXXX")"
  curl -fsS -m 300 "${BASE_URL}/v1/chat/completions" \
    -H 'Content-Type: application/json' --data @"${body_file}" >"${raw}"
  python3 - "${raw}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
msg = (d.get("choices") or [{}])[0].get("message", {})
print((msg.get("content") or "").strip())
PY
}

case "${START_SERVER}" in
  no)   server_up || die "START_SERVER=no but nothing on ${BASE_URL}" ;;
  yes)  start_server ;;
  auto) if server_up; then ok "reusing server on ${PORT}"; else start_server; fi ;;
  *)    die "START_SERVER must be auto|yes|no" ;;
esac

TOTAL=0; PASS=0
t() { TOTAL=$((TOTAL+1)); }

# 1) text smoke
t; log "test 1 — text smoke (/v1/chat/completions)"
cat >"${WORKDIR}/req_smoke.json" <<JSON
{"model":"${SERVED_NAME}","temperature":0,"max_tokens":16,
 "messages":[{"role":"user","content":"Reply with exactly one word: pong"}]}
JSON
SMOKE="$(chat "${WORKDIR}/req_smoke.json")"
printf '    model said: %q\n' "${SMOKE}" >&2
if grep -qi pong <<<"${SMOKE}"; then ok "smoke passed"; PASS=$((PASS+1)); else warn "unexpected reply"; fi

# 2) factual coherence
t; log "test 2 — coherence"
cat >"${WORKDIR}/req_coh.json" <<JSON
{"model":"${SERVED_NAME}","temperature":0,"max_tokens":48,
 "messages":[{"role":"user","content":"What is the capital of France? One short sentence."}]}
JSON
COH="$(chat "${WORKDIR}/req_coh.json")"
printf '    model said: %q\n' "${COH}" >&2
if grep -qi paris <<<"${COH}"; then ok "coherence passed (saw 'Paris')"; PASS=$((PASS+1)); else warn "no 'Paris'"; fi

# 3) tool call → OpenAI tool_calls via the qwen3_coder XML grammar
t; log "test 3 — tool call (XML grammar → tool_calls)"
cat >"${WORKDIR}/req_tool.json" <<JSON
{"model":"${SERVED_NAME}","temperature":0,"max_tokens":128,
 "tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
 "messages":[{"role":"user","content":"What is the weather in Paris right now? Use the tool."}]}
JSON
curl -fsS -m 300 "${BASE_URL}/v1/chat/completions" -H 'Content-Type: application/json' \
  --data @"${WORKDIR}/req_tool.json" >"${WORKDIR}/tool.json"
if python3 - "${WORKDIR}/tool.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
c = d["choices"][0]
assert c["finish_reason"] == "tool_calls", c["finish_reason"]
tc = c["message"]["tool_calls"][0]["function"]
assert tc["name"] == "get_weather", tc
args = json.loads(tc["arguments"])
assert "paris" in args.get("city", "").lower(), args
print(f"    tool_call: {tc['name']}({tc['arguments']})")
PY
then ok "tool call passed"; PASS=$((PASS+1)); else warn "tool call failed — see ${WORKDIR}/tool.json"; fi

# 4) tool RESULT round-trip (multi-turn with tool response)
t; log "test 4 — tool result round-trip"
cat >"${WORKDIR}/req_toolres.json" <<JSON
{"model":"${SERVED_NAME}","temperature":0,"max_tokens":48,
 "tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
 "messages":[
  {"role":"user","content":"What is the weather in Paris? Use the tool, then answer in one sentence."},
  {"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},
  {"role":"tool","tool_call_id":"call_1","content":"Sunny, 24 degrees Celsius"}]}
JSON
TOOLRES="$(chat "${WORKDIR}/req_toolres.json")"
printf '    model said: %q\n' "${TOOLRES}" >&2
if grep -qiE "sunny|24" <<<"${TOOLRES}"; then ok "tool round-trip passed"; PASS=$((PASS+1)); else warn "answer ignored tool result"; fi

# 5) /v1/messages non-streaming (usage accounting)
t; log "test 5 — /v1/messages non-streaming"
cat >"${WORKDIR}/req_anthropic.json" <<JSON
{"model":"${SERVED_NAME}","max_tokens":32,"temperature":0,
 "messages":[{"role":"user","content":"Reply with exactly one word: pong"}]}
JSON
curl -fsS -m 300 "${BASE_URL}/v1/messages" -H 'Content-Type: application/json' \
  --data @"${WORKDIR}/req_anthropic.json" >"${WORKDIR}/anthropic.json"
if python3 - "${WORKDIR}/anthropic.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["type"] == "message" and d["role"] == "assistant", d
text = "".join(b.get("text", "") for b in d["content"] if b["type"] == "text")
assert "pong" in text.lower(), f"unexpected: {text!r}"
u = d["usage"]
assert u["input_tokens"] > 0 and u["output_tokens"] > 0, u
print(f"    input_tokens={u['input_tokens']} output_tokens={u['output_tokens']}")
PY
then ok "/v1/messages passed"; PASS=$((PASS+1)); else warn "/v1/messages failed"; fi

# 6) /v1/messages streaming (framing + usage)
t; log "test 6 — /v1/messages streaming"
cat >"${WORKDIR}/req_anthropic_stream.json" <<JSON
{"model":"${SERVED_NAME}","max_tokens":32,"temperature":0,"stream":true,
 "messages":[{"role":"user","content":"Reply with exactly one word: pong"}]}
JSON
curl -fsSN -m 300 "${BASE_URL}/v1/messages" -H 'Content-Type: application/json' \
  --data @"${WORKDIR}/req_anthropic_stream.json" >"${WORKDIR}/anthropic_stream.txt"
if python3 - "${WORKDIR}/anthropic_stream.txt" <<'PY'
import json, sys
events = []
for line in open(sys.argv[1]):
    line = line.strip()
    if line.startswith("event:"):
        events.append({"name": line.split(":", 1)[1].strip()})
    elif line.startswith("data:") and events:
        try:
            events[-1]["data"] = json.loads(line.split(":", 1)[1].strip())
        except json.JSONDecodeError:
            pass
names = [e["name"] for e in events]
assert names[0] == "message_start", names
assert "content_block_delta" in names, names
assert names[-2:] == ["message_delta", "message_stop"], names[-4:]
md = next(e for e in reversed(events) if e["name"] == "message_delta")
u = md["data"]["usage"]
assert u["input_tokens"] > 0 and u["output_tokens"] > 0, u
print(f"    events={len(events)} usage={u}")
PY
then ok "/v1/messages streaming passed"; PASS=$((PASS+1)); else warn "streaming failed"; fi

# 7) count_tokens consistency
t; log "test 7 — /v1/messages/count_tokens"
curl -fsS -m 60 "${BASE_URL}/v1/messages/count_tokens" -H 'Content-Type: application/json' \
  --data @"${WORKDIR}/req_anthropic.json" >"${WORKDIR}/count.json"
if python3 - "${WORKDIR}/count.json" "${WORKDIR}/anthropic.json" <<'PY'
import json, sys
count = json.load(open(sys.argv[1]))["input_tokens"]
served = json.load(open(sys.argv[2]))["usage"]["input_tokens"]
assert count > 0, count
drift = abs(count - served) / max(served, 1)
assert drift <= 0.10, f"count {count} vs served {served}: {drift:.0%}"
print(f"    count_tokens={count} served={served}")
PY
then ok "count_tokens passed"; PASS=$((PASS+1)); else warn "count_tokens failed"; fi

# 8) /v1/responses streaming smoke
t; log "test 8 — /v1/responses streaming"
cat >"${WORKDIR}/req_responses.json" <<JSON
{"model":"${SERVED_NAME}","temperature":0,"max_output_tokens":32,"stream":true,
 "input":"Reply with exactly one word: pong"}
JSON
curl -fsSN -m 300 "${BASE_URL}/v1/responses" -H 'Content-Type: application/json' \
  --data @"${WORKDIR}/req_responses.json" >"${WORKDIR}/responses_stream.txt"
if python3 - "${WORKDIR}/responses_stream.txt" <<'PY'
import json, sys
completed = None
for line in open(sys.argv[1]):
    line = line.strip()
    if line.startswith("data:"):
        try:
            d = json.loads(line.split(":", 1)[1].strip())
        except json.JSONDecodeError:
            continue
        if d.get("type") == "response.completed":
            completed = d
assert completed is not None, "no response.completed"
usage = completed["response"].get("usage") or {}
assert usage.get("input_tokens", 0) > 0, usage
print(f"    completed, usage={usage}")
PY
then ok "/v1/responses passed"; PASS=$((PASS+1)); else warn "/v1/responses failed"; fi

# 9) concurrency isolation (2 sequences)
t; log "test 9 — 2 concurrent sequences"
cat >"${WORKDIR}/req_a.json" <<JSON
{"model":"${SERVED_NAME}","temperature":0,"max_tokens":12,
 "messages":[{"role":"user","content":"Say APPLE and nothing else"}]}
JSON
cat >"${WORKDIR}/req_b.json" <<JSON
{"model":"${SERVED_NAME}","temperature":0,"max_tokens":12,
 "messages":[{"role":"user","content":"Say BANANA and nothing else"}]}
JSON
chat "${WORKDIR}/req_a.json" >"${WORKDIR}/out_a.txt" &
PID_A=$!
B="$(chat "${WORKDIR}/req_b.json")"
wait "${PID_A}" || true
A="$(cat "${WORKDIR}/out_a.txt")"
printf '    A=%q B=%q\n' "${A}" "${B}" >&2
if grep -qi apple <<<"${A}" && grep -qi banana <<<"${B}"; then
  ok "concurrency passed"; PASS=$((PASS+1))
else warn "concurrency outputs wrong"; fi

# ── vision tests (BONSAI_VISION=1; needs the mmproj sidecar loaded) ──
if [[ "${BONSAI_VISION}" == "1" ]]; then
  make_image_data_url() {
    local png="${WORKDIR}/test.png"
    python3 - "$png" <<'PY'
import sys, zlib, struct
w = h = 128
raw = bytearray()
for _ in range(h):
    raw.append(0)
    raw.extend((220, 30, 30) * w)
def chunk(tag, data):
    c = tag + data
    return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
png = b"\x89PNG\r\n\x1a\n"
png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
png += chunk(b"IEND", b"")
open(sys.argv[1], "wb").write(png)
PY
    printf 'data:image/png;base64,%s' "$(base64 < "$png" | tr -d '\n')"
  }
  DATA_URL="$(make_image_data_url)"; export DATA_URL

  t; log "test 10 — image on a USER message"
  python3 - <<PY
import json, os
body = {"model": os.environ["SERVED_NAME"], "temperature": 0, "max_tokens": 48,
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": os.environ["DATA_URL"]}},
    {"type": "text", "text": "What is the dominant color in this image? One word."}]}]}
json.dump(body, open("${WORKDIR}/req_img.json", "w"))
PY
  IMG="$(chat "${WORKDIR}/req_img.json")"
  printf '    model said: %q\n' "${IMG}" >&2
  if grep -qi red <<<"${IMG}"; then ok "vision passed (saw 'red')"; PASS=$((PASS+1)); else warn "vision failed"; fi
fi

log "──────────────────────────────"
log "RESULT: ${PASS}/${TOTAL} tests passed (artifacts: ${WORKDIR})"
[[ "${PASS}" == "${TOTAL}" ]] || exit 1
