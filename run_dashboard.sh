#!/usr/bin/env bash
# Starts the dashboard (static pages + prediction API) on http://localhost:8765 and opens it. Stop with Ctrl+C.
# First start takes ~30 s while the feature table loads. Usage: bash run_dashboard.sh [port]
cd "$(dirname "$0")"
PORT="${1:-8765}"
( until curl -s -o /dev/null "http://localhost:$PORT/api/options"; do sleep 2; done; open "http://localhost:$PORT/" 2>/dev/null || xdg-open "http://localhost:$PORT/" 2>/dev/null ) &
exec .venv/bin/python dashboard/server.py "$PORT"
