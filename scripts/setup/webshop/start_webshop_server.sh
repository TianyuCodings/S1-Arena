#!/bin/bash
# Start the WebShop server for S1-Arena's loop/web_shopping environment: WebShop's text environment in its own
# Python 3.8 venv, on 127.0.0.1:3000. Loading takes about 2 minutes and ~18 GB of RAM; the log is logs/server.log
# ("ready on ..." when it serves). A guard kills it if its resident memory passes 24 GB.
source $HOME/s1arena_env/webshop/env.sh
if curl -s --max-time 5 http://127.0.0.1:3000/info > /dev/null; then echo "a server already answers on 127.0.0.1:3000"; exit 0; fi
if pgrep -f web_shopping_server.py > /dev/null; then echo "a server is still loading (see $JV/logs/server.log)"; exit 0; fi
echo "=== $(date -u +%FT%TZ) start" >> $JV/logs/server.log
nohup $JV/bin/rss_guard.sh 24 $JV/logs/server.log $JV/venv_webshop/bin/python -u \
  $JV/S1-Arena/s1arena/envs/loop/web_shopping_server.py --webshop $JV/WebShop --data $JV/data --port 3000 > /dev/null 2>&1 &
echo "loading (guard pid $!); wait for 'ready on' in $JV/logs/server.log"
