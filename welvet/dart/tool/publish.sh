#!/usr/bin/env bash
# Publish welvet to pub.dev (https://pub.dev/packages/welvet).
#
# Prerequisites:
#   1. flutter pub login   (one-time — opens browser OAuth)
#   2. bash tool/copy_native.sh --all   (binaries must be present for a full release)
#
# Usage:
#   bash tool/publish.sh           # dry-run (recommended first)
#   bash tool/publish.sh --publish # actually publish
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DRY=true
if [[ "${1:-}" == "--publish" ]]; then
  DRY=false
fi

echo "=== welvet publish ==="
flutter pub get
flutter test

if $DRY; then
  echo ""
  echo "Dry run (no upload). To publish:"
  echo "  bash tool/publish.sh --publish"
  flutter pub publish --dry-run
else
  echo ""
  echo "Publishing to pub.dev..."
  flutter pub publish
fi
