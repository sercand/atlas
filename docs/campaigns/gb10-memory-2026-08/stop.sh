#!/usr/bin/env bash
# Stop by PORT OWNER, then wait for the GPU memory to actually come back.
PID=$(ss -lptnH "sport = :36200" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
[ -n "$PID" ] && kill "$PID"
for _ in $(seq 1 30); do ss -lptnH "sport = :36200" 2>/dev/null | grep -q . || break; sleep 1; done
if [ -n "$PID" ]; then
  for _ in $(seq 1 60); do
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -qx "$PID" || break
    sleep 2
  done
fi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tail -n +2
