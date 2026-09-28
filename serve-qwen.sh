#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LLAMA="$ROOT/llama.cpp/build/bin/Release"
MODEL="$ROOT/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf"

exec "$LLAMA/llama-server.exe" \
  -m "$MODEL" \
  --host 127.0.0.1 \
  --port 8080 \
  -ngl all \
  --cpu-moe \
  --load-mode none \
  -c 262144 \
  -b 16384 \
  -ub 1472 \
  --cache-ram 10240 \
  --ctx-checkpoints 8 \
  --checkpoint-min-step 2048 \
  -fa on \
  -ctk q8_0 \
  -ctv q8_0 \
  -t 16 \
  -tb 16 \
  --fit off \
  --reasoning on \
  --reasoning-format deepseek \
  --parallel 1 \
  --reasoning-effort high \
  --reasoning-budget 8096 \
  --cont-batching \
  --metrics
