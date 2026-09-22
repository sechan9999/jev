#!/usr/bin/env bash
# Start SGLang with an open model and expose its native HTTP endpoints
# (/tokenize, /v1/score, /v1/chat/completions) on port 30000.
#
# Keep this process running while you use decide.py / benchmark.py.
# The first launch downloads the model from Hugging Face; later launches
# reuse the local cache.
#
# Extra flags are forwarded to launch_server, e.g.:
#   bash run_server.sh --disable-cuda-graph --mem-fraction-static 0.7
# On WSL / a box without the CUDA toolkit (no nvcc), --disable-cuda-graph
# avoids a graph-capture failure; on a small GPU, lower --mem-fraction-static.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"

echo "Launching SGLang server for ${MODEL} on ${HOST}:${PORT} ..."
exec python -m sglang.launch_server \
  --model-path "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  "$@"
