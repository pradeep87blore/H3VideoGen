#!/usr/bin/env bash
set -e

MODELS_DIR="${OLLAMA_MODELS:-/models/ollama}"
mkdir -p "$MODELS_DIR"

# Start Ollama briefly to pull models on first run
if [ "${SKIP_OLLAMA_PULL:-}" != "1" ] && [ ! -f "$MODELS_DIR/.models_pulled" ]; then
    echo "[entrypoint] First run — pulling Ollama models..."
    OLLAMA_HOST="127.0.0.1:11434" OLLAMA_MODELS="$MODELS_DIR" ollama serve &
    OLLAMA_PID=$!
    sleep 5

    ollama pull llama3.2 || true
    ollama pull qwen2.5vl || true

    kill $OLLAMA_PID 2>/dev/null || true
    wait $OLLAMA_PID 2>/dev/null || true
    touch "$MODELS_DIR/.models_pulled"
    echo "[entrypoint] Models pulled."
fi

# Hand off to supervisord
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/h3gpu.conf
