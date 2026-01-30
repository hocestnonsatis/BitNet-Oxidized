#!/usr/bin/env bash
# Install repo githooks into .git/hooks so pre-commit (cargo fmt) runs without setting core.hooksPath.
set -e
ROOT="$(git rev-parse --show-toplevel)"
HOOK_SRC="$ROOT/.githooks/pre-commit"
HOOK_DST="$ROOT/.git/hooks/pre-commit"
if [ ! -f "$HOOK_SRC" ]; then
  echo "Missing $HOOK_SRC" >&2
  exit 1
fi
cp "$HOOK_SRC" "$HOOK_DST"
chmod +x "$HOOK_DST"
echo "Installed pre-commit hook to .git/hooks/pre-commit (cargo fmt on commit)"
