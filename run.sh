#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo "  👁️  DRISHYA: Autonomous Retinal Screening Platform            "
echo "  Starting Backend API Service + Web UI Dashboard                "
echo "=================================================================="

# ── 1. Discover Python & Virtualenv ───────────────────────────────────────────
PYTHON_EXEC=""
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_EXEC="$SCRIPT_DIR/.venv/bin/python"
elif [ -f "$SCRIPT_DIR/../drishya_v2/.venv/bin/python" ]; then
    PYTHON_EXEC="$SCRIPT_DIR/../drishya_v2/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_EXEC="$(command -v python3)"
else
    echo "[-] Error: Python executable not found. Please initialize virtualenv."
    exit 1
fi

export PATH="$(dirname "$PYTHON_EXEC"):$PATH"
echo "[+] Using Python runtime: $PYTHON_EXEC"

# ── 2. Discover Node & NPM ───────────────────────────────────────────────────
if ! command -v npm &>/dev/null; then
    echo "[-] Error: npm not found in PATH. Please run inside nix develop."
    exit 1
fi
echo "[+] Using Node runtime: $(node --version) | NPM: $(npm --version)"

# ── 3. Directory & Port Checks ────────────────────────────────────────────────
mkdir -p "$SCRIPT_DIR/backend/outputs"
mkdir -p "$SCRIPT_DIR/reports"

BACKEND_PORT=8000
FRONTEND_PORT=5173

# Check for port collisions
if ss -tuln 2>/dev/null | grep -q ":${BACKEND_PORT} "; then
    echo "[-] Warning: Port $BACKEND_PORT is already in use. Checking PID..."
    fuser -k "${BACKEND_PORT}/tcp" 2>/dev/null || true
    sleep 1
fi

if ss -tuln 2>/dev/null | grep -q ":${FRONTEND_PORT} "; then
    echo "[-] Warning: Port $FRONTEND_PORT is already in use. Checking PID..."
    fuser -k "${FRONTEND_PORT}/tcp" 2>/dev/null || true
    sleep 1
fi

# ── 4. Process Tracking & Graceful Shutdown ──────────────────────────────────
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    echo ""
    echo "=================================================================="
    echo "  Shutting down DRISHYA services...                               "
    echo "=================================================================="
    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "[*] Stopping FastAPI backend (PID: $BACKEND_PID)..."
        kill -TERM "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "$FRONTEND_PID" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo "[*] Stopping Vite frontend (PID: $FRONTEND_PID)..."
        kill -TERM "$FRONTEND_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    echo "[✓] All DRISHYA services stopped."
}

trap cleanup INT TERM EXIT

# ── 5. Launch FastAPI Backend ────────────────────────────────────────────────
echo ""
echo "[1/2] Launching FastAPI AI Inference Backend..."
"$PYTHON_EXEC" -m uvicorn backend.main:app --host 0.0.0.0 --port "$BACKEND_PORT" &
BACKEND_PID=$!

# Wait for backend healthcheck probe
echo "[*] Waiting for backend to become ready on http://127.0.0.1:${BACKEND_PORT}..."
MAX_RETRIES=25
RETRY_COUNT=0
BACKEND_READY=0

while [ "$RETRY_COUNT" -lt "$MAX_RETRIES" ]; do
    if curl -s -f "http://127.0.0.1:${BACKEND_PORT}/api/health" &>/dev/null; then
        BACKEND_READY=1
        break
    fi
    sleep 0.5
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ "$BACKEND_READY" -eq 1 ]; then
    echo "[✓] FastAPI backend is live and healthy!"
else
    echo "[-] Warning: Healthcheck timed out. Backend may still be initializing weights."
fi

# ── 6. Launch Vite Frontend UI ───────────────────────────────────────────────
echo ""
echo "[2/2] Launching Vite Frontend UI..."
npm --prefix "$SCRIPT_DIR/ui" run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT" &
FRONTEND_PID=$!

sleep 1

# ── 7. Platform Status Dashboard ─────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "  🚀 DRISHYA PLATFORM IS RUNNING LOCALLY                          "
echo "=================================================================="
echo "  🌐 Web Dashboard:    http://localhost:${FRONTEND_PORT}"
echo "  ⚙️  Backend API:      http://localhost:${BACKEND_PORT}"
echo "  📖 Swagger Docs:     http://localhost:${BACKEND_PORT}/docs"
echo "  🩺 Health Endpoint:  http://localhost:${BACKEND_PORT}/api/health"
echo "=================================================================="
echo "  Modes Available in Dashboard:"
echo "    • Health Worker Mode: Rapid offline screening & auto 1-page PDF"
echo "    • Judge / Specialist Mode: Grad-CAM++ lesion inspection & metrics"
echo ""
echo "  Press Ctrl+C to shut down all services gracefully."
echo "=================================================================="

# Keep process alive and forward signals
wait "$BACKEND_PID" "$FRONTEND_PID"
