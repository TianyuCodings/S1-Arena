#!/bin/bash
# Run a command; every 2 s sum the RSS of its process tree; kill it if above LIMIT_GB; log the peak.
# usage: watchdog.sh LIMIT_GB LOGFILE cmd args...
LIMIT_KB=$(( $1 * 1024 * 1024 )); LOG=$2; shift 2
"$@" >> "$LOG" 2>&1 &
PID=$!
PEAK=0
tree_rss() { local total=0; for p in $(pstree -p $1 2>/dev/null | grep -o '([0-9]\+)' | tr -d '()'); do r=$(ps -o rss= -p $p 2>/dev/null); total=$((total + ${r:-0})); done; echo $total; }
while kill -0 $PID 2>/dev/null; do
  RSS=$(tree_rss $PID)
  [ "$RSS" -gt "$PEAK" ] && PEAK=$RSS
  if [ "$RSS" -gt "$LIMIT_KB" ]; then echo "WATCHDOG: RSS ${RSS} kB > limit, killing" >> "$LOG"; pkill -KILL -P $PID; kill -KILL $PID; fi
  sleep 2
done
wait $PID; STATUS=$?
echo "WATCHDOG: exit $STATUS, peak tree RSS $((PEAK / 1024)) MB" >> "$LOG"
