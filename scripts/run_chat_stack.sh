#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
LOG_DIR="$BACKEND_DIR/logs"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"

cleanup() {
  if [[ -n "${API_PID:-}" ]]; then
    kill "$API_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${WEB_PID:-}" ]]; then
    kill "$WEB_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "Missing virtualenv at .venv. Create it first."
  exit 1
fi

mkdir -p "$LOG_DIR"

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "Installing React dependencies..."
  (cd "$FRONTEND_DIR" && npm install)
fi

echo "Starting API server on :$API_PORT ..."
(cd "$BACKEND_DIR" && "$ROOT_DIR/.venv/bin/python" -m uvicorn api_server:app --host 0.0.0.0 --port "$API_PORT") >"$LOG_DIR/api_server.out.log" 2>&1 &
API_PID=$!

echo "Starting React dev server on :$WEB_PORT ..."
(cd "$FRONTEND_DIR" && npm run dev -- --host 0.0.0.0 --port "$WEB_PORT") >"$LOG_DIR/react_app.out.log" 2>&1 &
WEB_PID=$!

echo "Waiting for services..."
sleep 3

echo "API:  http://localhost:$API_PORT/health"
echo "Chat: http://localhost:$WEB_PORT"
echo "Logs:"
echo "  $LOG_DIR/api_server.out.log"
echo "  $LOG_DIR/react_app.out.log"
echo ""
echo "Press Ctrl+C to stop both."

wait "$API_PID" "$WEB_PID"
