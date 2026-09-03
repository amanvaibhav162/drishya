#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# DRISHYA — One-command startup script
# Builds the React UI and starts the unified FastAPI server on port 8000.
# Usage: ./start.sh
# ──────────────────────────────────────────────────────────────────────────────

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║        दृष्य  DRISHYA AI Engine          ║"
echo "  ║   Rural Diabetic Retinopathy Screening   ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""

# ── 1. Activate virtual environment ──────────────────────────────────────────
if [ -d ".venv" ]; then
    echo "🐍 Activating Python virtual environment..."
    source .venv/bin/activate
else
    echo "⚠️  No .venv found. Make sure dependencies are installed:"
    echo "   python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# ── 2. Check model checkpoint exists ─────────────────────────────────────────
if [ ! -f "models/student_mtl_lcnet_best.pth" ]; then
    echo "⚠️  Model checkpoint not found at models/student_mtl_lcnet_best.pth"
    echo "   Place the trained .pth file in the models/ directory."
    exit 1
fi

# ── 3. Build the React UI ────────────────────────────────────────────────────
echo "⚙️  Building production UI..."
cd ui
npm run build --silent 2>&1
cd ..
echo "✅ UI built successfully → ui/dist/"

# ── 4. Start FastAPI server ──────────────────────────────────────────────────
PORT=${PORT:-8000}
echo ""
echo "🚀 Starting DRISHYA on http://localhost:${PORT}"
echo "   API docs → http://localhost:${PORT}/docs"
echo "   Press Ctrl+C to stop."
echo ""

python -m uvicorn backend.main:app --host 0.0.0.0 --port "$PORT"
