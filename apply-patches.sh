#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA="$ROOT/llama.cpp"
PATCH="$ROOT/hybrid-checkpoint.patch"

if git -C "$LLAMA" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
    echo "Hybrid checkpoint patch already applied."
    exit 0
fi

echo "Checking hybrid checkpoint patch..."
git -C "$LLAMA" apply --check "$PATCH"

echo "Applying hybrid checkpoint patch..."
git -C "$LLAMA" apply "$PATCH"

echo "Done."

