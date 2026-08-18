#!/usr/bin/env bash
# OOM WATCHDOG. Kills the spark server if system available memory falls below
# a floor, so a mis-sized KV pool degrades to a dead server instead of a box
# that needs a power cycle. $1 = floor in GB (default 10).
FLOOR=${1:-10}
while true; do
  AVAIL=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  PID=$(ss -lptnH "sport = :36200" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -z "$PID" ] && { sleep 2; continue; }
  if [ "$AVAIL" -lt "$FLOOR" ]; then
    echo "$(date +%T) WATCHDOG: MemAvailable ${AVAIL}GB < ${FLOOR}GB — killing spark pid $PID"
    kill -9 "$PID"
    exit 1
  fi
  sleep 2
done
