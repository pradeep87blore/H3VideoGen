#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  H3 Video Gen — Linux / macOS launcher
#  Creates .venv, installs deps when needed, ensures .env, starts UI,
#  opens http://127.0.0.1:7860 when the server is ready.
# ============================================================================

cd "$(dirname "$0")"

HOST="127.0.0.1"
PORT="7860"
URL="http://${HOST}:${PORT}"
VENV_PY=".venv/bin/python"
MARKER=".venv/.deps_installed"

echo ""
echo "  ========================================"
echo "   H3 Video Gen"
echo "  ========================================"
echo ""

# ---- Find system Python ----
PY=""
if command -v python3 &>/dev/null; then
    PY="python3"
elif command -v python &>/dev/null; then
    PY="python"
fi

if [ -z "$PY" ]; then
    echo "[ERROR] Python 3 not found. Install Python 3.11+ and ensure it is on PATH."
    exit 1
fi

# ---- Virtual environment ----
if [ ! -f "$VENV_PY" ]; then
    echo "[1/4] Creating virtual environment .venv ..."
    $PY -m venv .venv
    [ -f "$MARKER" ] && rm -f "$MARKER"
else
    echo "[1/4] Virtual environment already present."
fi

if [ ! -f "$VENV_PY" ]; then
    echo "[ERROR] Expected $VENV_PY after venv create."
    exit 1
fi

# ---- Dependencies ----
if [ ! -f "requirements.txt" ]; then
    echo "[ERROR] requirements.txt not found."
    exit 1
fi

if [ ! -f "$MARKER" ]; then
    echo "[2/4] Installing Python packages (first run may take a few minutes)..."
    "$VENV_PY" -m pip install --upgrade pip
    "$VENV_PY" -m pip install -r requirements.txt
    echo "installed" > "$MARKER"
else
    echo "[2/4] Dependencies already installed."
    echo "      Delete .venv/.deps_installed to force a reinstall."
fi

# ---- .env ----
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "[3/4] Creating .env from .env.example ..."
        cp .env.example .env
        echo ""
        echo " [WARNING] Set GEMINI_API_KEY in .env for full director/critic features."
        echo ""
    else
        echo "[3/4] No .env or .env.example — continuing with defaults."
    fi
else
    echo "[3/4] .env found."
fi

# ---- Bootstrap (AI tools / models) ----
echo "[2.5/4] Checking AI tools / models (installs if missing)..."
if "$VENV_PY" run.py bootstrap; then
    echo "      Prerequisites look ready."
else
    echo "[WARNING] Some prerequisites are still missing — Generate may fail until they finish."
    echo "          Re-run: .venv/bin/python run.py bootstrap"
fi

# ---- Port already in use? ----
if ss -ltn 2>/dev/null | grep -q ":${PORT} " || \
   netstat -ltn 2>/dev/null | grep -q ":${PORT} "; then
    echo "[note] Port $PORT is already listening — opening browser to existing server."
    xdg-open "$URL" 2>/dev/null || open "$URL" 2>/dev/null || true
    exit 0
fi

echo "[4/4] Starting server on $URL"
echo "      Press Ctrl+C to stop."
echo ""

# Open browser once server is ready (background)
(
    for i in $(seq 1 90); do
        if curl -sf "${URL}/api/health" >/dev/null 2>&1; then
            xdg-open "$URL" 2>/dev/null || open "$URL" 2>/dev/null || true
            break
        fi
        sleep 1
    done
) &

"$VENV_PY" run.py serve
