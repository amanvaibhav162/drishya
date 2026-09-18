#!/usr/bin/env bash
# Quick entrypoint for DRISHYA SOTA Training CLI Wizard

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [ -x "/workspace/envs/ml/bin/python" ]; then
    PYTHON_CMD="/workspace/envs/ml/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON_CMD="${VIRTUAL_ENV}/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

exec "$PYTHON_CMD" "$PROJECT_ROOT/training/interactive_launcher.py" "$@"
