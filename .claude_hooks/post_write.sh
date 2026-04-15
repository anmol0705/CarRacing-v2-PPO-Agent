#!/bin/bash
FILE="$1"
if [[ -z "$FILE" ]]; then exit 0; fi

# Auto-format
black "$FILE" --quiet 2>/dev/null || true

# Auto-verify
MODULE=$(basename "$FILE" .py)
VERIFY="$(dirname "$FILE")/../tests/verify_${MODULE}.py"
if [ -f "$VERIFY" ]; then
  echo "=== AUTO-VERIFY: $MODULE ==="
  python "$VERIFY" && echo "HOOK PASS: $MODULE" || echo "HOOK FAIL: $MODULE"
fi
