#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Decode/prefill tokens-per-second benchmark for Bonsai-27B Q1_0 on the
# Atlas metal backend. Starts the server if one isn't already listening
# (same conventions as test-bonsai-metal.sh), then runs
# bench/bonsai_metal_tps.py against it.
#
#   ./scripts/bench-bonsai-metal.sh                 # decode benchmark
#   MAX_TOKENS=256 RUNS=5 ./scripts/bench-bonsai-metal.sh
#   PROMPT_TOKENS=1000 ./scripts/bench-bonsai-metal.sh   # prefill-heavy
#   MLX_REF=1 ./scripts/bench-bonsai-metal.sh       # also run the MLX
#                                                   # reference from Bonsai-demo
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models/bonsai-27b-atlas}"
SERVED_NAME="${SERVED_NAME:-bonsai-27b-atlas}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8899}"
BASE_URL="http://${HOST}:${PORT}"
SPARK_BIN="${SPARK_BIN:-./target/release/spark}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
READY_TIMEOUT="${READY_TIMEOUT:-300}"
MAX_TOKENS="${MAX_TOKENS:-128}"
RUNS="${RUNS:-3}"
PROMPT_TOKENS="${PROMPT_TOKENS:-0}"
MLX_REF="${MLX_REF:-0}"
BONSAI_DEMO_DIR="${BONSAI_DEMO_DIR:-$HOME/Developer/src/github.com/PrisimML-Eng/Bonsai-demo}"

SERVER_LOG="$(mktemp "${TMPDIR:-/tmp}/atlas-bonsai-bench.XXXXXX")"
SERVER_PID=""
STARTED_SERVER=0

log() { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { printf '\033[1;31m[FATAL] %s\033[0m\n' "$*" >&2; exit 1; }

cleanup() {
  if [[ "${STARTED_SERVER}" == "1" && -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    log "stopping server (pid ${SERVER_PID})"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

server_up() { curl -fsS -m 3 "${BASE_URL}/health" >/dev/null 2>&1; }

if ! server_up; then
  [[ -x "${SPARK_BIN}" ]] || die "spark binary not found at '${SPARK_BIN}' — build with:
    ATLAS_TARGET_HW=metal ATLAS_TARGET_MODEL=bonsai-27b ATLAS_TARGET_QUANT=q1_0 \\
      cargo build --release -p spark-server --no-default-features --features metal"
  [[ -d "${MODEL_DIR}" ]] || die "model dir '${MODEL_DIR}' missing"
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
  deadline=$(( $(date +%s) + READY_TIMEOUT ))
  while ! server_up; do
    kill -0 "${SERVER_PID}" 2>/dev/null || { tail -n 40 "${SERVER_LOG}" >&2; die "server exited early"; }
    [[ $(date +%s) -lt ${deadline} ]] || { tail -n 40 "${SERVER_LOG}" >&2; die "readiness timeout"; }
    sleep 3
  done
else
  log "reusing server on ${BASE_URL}"
fi

log "atlas benchmark (max_tokens=${MAX_TOKENS}, runs=${RUNS}, prompt_tokens=${PROMPT_TOKENS})"
python3 "$(dirname "$0")/../bench/bonsai_metal_tps.py" \
  --base-url "${BASE_URL}" --model "${SERVED_NAME}" \
  --max-tokens "${MAX_TOKENS}" --runs "${RUNS}" \
  --prompt-tokens "${PROMPT_TOKENS}"

if [[ "${MLX_REF}" == "1" ]]; then
  [[ -d "${BONSAI_DEMO_DIR}" ]] || die "Bonsai-demo not found at ${BONSAI_DEMO_DIR}"
  log "MLX reference run (BONSAI_FAMILY=bonsai BONSAI_MODEL=27B, same device)…"
  ( cd "${BONSAI_DEMO_DIR}" && \
    BONSAI_FAMILY=bonsai BONSAI_MODEL=27B ./scripts/run_mlx.sh \
      -p "Write a detailed, flowing essay about the history of ocean navigation. Do not use lists; write continuous prose." \
      -n "${MAX_TOKENS}" )
fi
