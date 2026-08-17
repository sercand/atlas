#!/usr/bin/env bash
# Stop whatever is serving :36200, by PORT OWNER rather than by process name.
# `pkill -x spark` silently misses a binary copied to another name (the baseline
# copy is `spark-baseline-...`), which left the old server up, made the new one
# fail preflight on "only 3.94 GB free", and quietly re-measured the OLD binary.
PID=$(ss -lptnH "sport = :36200" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
if [ -n "$PID" ]; then kill "$PID"; fi
for _ in $(seq 1 30); do
  ss -lptnH "sport = :36200" 2>/dev/null | grep -q . || break
  sleep 1
done
# Then wait for the GPU memory to ACTUALLY come back. The listener closes long
# before teardown finishes sweeping ~1500 allocations, and starting the next
# server too early makes it die in preflight on "only 4.3 GB is free" — which,
# if the old server is still up, silently re-measures the OLD binary.
if [ -n "$PID" ]; then
  for _ in $(seq 1 60); do
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
      | grep -qx "$PID" || break
    sleep 2
  done
fi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tail -n +2
