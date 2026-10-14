#!/bin/bash
# Stop the WebShop server started by start_webshop_server.sh (and its memory guard).
pkill -f "rss_guard.sh 24 .*web_shopping_server.py"; pkill -f "web_shopping_server.py --webshop" && echo stopped || echo "not running"
