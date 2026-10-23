#!/bin/bash
# Start the HOOPS AI Similarity Search Server on headless Linux (Ubuntu 22.04).
# Usage:
#   ./start_server.sh [--port 8000] [--host 0.0.0.0] [--reload]
#
# Set HOOPS_AI_VENV to override the HOOPS AI venv path:
#   HOOPS_AI_VENV=/custom/path/.venv ./start_server.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOPS_AI_VENV="${HOOPS_AI_VENV:-/var/HOOPS_AI/V1.1/.venv}"
PYTHON="$HOOPS_AI_VENV/bin/python"
DISPLAY_NUM=":99"

# Verify venv exists
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "Set HOOPS_AI_VENV to the HOOPS AI virtual environment path."
    exit 1
fi

# Start Xvfb virtual display if not already running
if ! pgrep -x Xvfb > /dev/null; then
    echo "Starting Xvfb virtual display on $DISPLAY_NUM ..."
    Xvfb "$DISPLAY_NUM" -screen 0 1280x960x24 > /dev/null 2>&1 &
    sleep 1
else
    echo "Xvfb already running."
fi

export DISPLAY="$DISPLAY_NUM"

echo "Starting HOOPS AI Similarity Search Server..."
echo "  venv   : $HOOPS_AI_VENV"
echo "  script : $SCRIPT_DIR/main.py"
echo ""

exec "$PYTHON" "$SCRIPT_DIR/main.py" "$@"
