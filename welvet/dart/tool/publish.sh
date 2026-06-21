#!/usr/bin/env bash
# Publish welvet to pub.dev (https://pub.dev/packages/welvet).
#
# Native binaries are NOT committed to git. This script:
#   1. Runs copy_native.sh in the source tree (locals only).
#   2. Copies the package to a temp dir OUTSIDE the git repo.
#   3. Optionally strips platform natives to stay under pub.dev's 100 MB limit.
#   4. Tests + publishes with auto-confirm (yes).
#
# Prerequisites:
#   flutter pub login   (one-time)
#
# Usage:
#   bash tool/publish.sh                    # dry-run, desktop slice (default)
#   bash tool/publish.sh --slice macos      # macos only
#   bash tool/publish.sh --publish          # upload desktop (mac + linux + windows)
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SLICE="desktop"
DRY=true
PUBLISH=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --publish) PUBLISH=true; DRY=false; shift ;;
    --slice) SLICE="${2:?--slice requires macos|desktop|mobile|all}"; shift 2 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *) echo "Unknown: $1" >&2; exit 1 ;;
  esac
done

copy_flags_for_slice() {
  case "$SLICE" in
    macos) echo "--macos" ;;
    desktop) echo "--macos --linux --windows" ;;
    mobile) echo "--android --ios" ;;
    all) echo "--all" ;;
    *) echo "Unknown slice: $SLICE (use macos|desktop|mobile|all)" >&2; exit 1 ;;
  esac
}

strip_slice() {
  local stage="$1"
  case "$SLICE" in
    macos)
      rm -rf "$stage/native/android" "$stage/native/linux_amd64" \
        "$stage/native/windows_amd64" "$stage/ios/Welvet.xcframework"
      ;;
    desktop)
      rm -rf "$stage/native/android" "$stage/ios/Welvet.xcframework"
      ;;
    mobile)
      rm -rf "$stage/native/linux_amd64" "$stage/native/macos_universal" \
        "$stage/native/windows_amd64" "$stage/macos/Frameworks"
      ;;
    all) ;;
  esac
}

stage_package() {
  local stage="$1"
  echo "Slice: $SLICE"
  echo "copy_native in source tree..."
  # shellcheck disable=SC2046
  (cd "$ROOT" && bash tool/copy_native.sh $(copy_flags_for_slice))

  echo "Staging package → $stage"
  mkdir -p "$stage"
  rsync -a \
    --exclude '.dart_tool/' \
    --exclude 'build/' \
    --exclude '.packages' \
    --exclude '.flutter-plugins' \
    --exclude '.flutter-plugins-dependencies' \
    "$ROOT/" "$stage/"

  cat >"$stage/.gitignore" <<'EOF'
.dart_tool/
build/
.DS_Store
*.tmp
EOF

  strip_slice "$stage"
}

run_in_stage() {
  local stage="$1"
  shift
  (cd "$stage" && "$@")
}

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/welvet-publish.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT

echo "=== welvet publish ==="
stage_package "$STAGE"

echo "Running tests in staging tree..."
run_in_stage "$STAGE" flutter pub get
run_in_stage "$STAGE" flutter test

if $DRY; then
  echo ""
  echo "Dry run (slice=$SLICE). Natives are staged from disk, not from git."
  echo "To publish: bash tool/publish.sh --publish --slice $SLICE"
  run_in_stage "$STAGE" flutter pub publish --dry-run
else
  echo ""
  echo "Publishing slice=$SLICE to pub.dev (auto-confirm)..."
  run_in_stage "$STAGE" bash -c 'yes | flutter pub publish'
fi
