#!/usr/bin/env bash
# Publish welvet federated packages to pub.dev (impl first, then main welvet).
#
# Prerequisites:
#   flutter pub login   (one-time)
#
# Usage:
#   bash tool/publish_all.sh              # dry-run all packages
#   bash tool/publish_all.sh --publish    # upload all (auto-confirm)
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PUBLISH_FLAG=""
if [[ "${1:-}" == "--publish" ]]; then
  PUBLISH_FLAG="--publish"
fi

IMPL_PKGS=(welvet_linux welvet_windows welvet_android welvet_apple)

for pkg in "${IMPL_PKGS[@]}"; do
  bash "$ROOT/tool/publish_impl.sh" "$pkg" $PUBLISH_FLAG
done

bash "$ROOT/tool/publish.sh" $PUBLISH_FLAG
