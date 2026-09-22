#!/usr/bin/env bash
# Start SGLang with an open model and expose its native HTTP endpoints
# (/tokenize, /v1/score, /v1/chat/completions) on port 30000.
#
# Keep this process running while you use decide.py / benchmark.py.
# The first launch downloads the model from Hugging Face; later launches
# reuse the local cache.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"

echo "Launching SGLang server for ${MODEL} on ${HOST}:${PORT} ..."
exec python -m sglang.launch_server \
  --model-path "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}"
