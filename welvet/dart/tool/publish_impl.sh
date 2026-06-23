#!/usr/bin/env bash
# Stage and publish a federated welvet implementation package (linux/windows/android/apple).
#
# Usage:
#   bash tool/publish_impl.sh welvet_linux           # dry-run
#   bash tool/publish_impl.sh welvet_linux --publish # upload to pub.dev
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG_NAME="${1:?package name: welvet_linux|welvet_windows|welvet_android|welvet_apple}"
DRY=true
PUBLISH=false

shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --publish) PUBLISH=true; DRY=false; shift ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *) echo "Unknown: $1" >&2; exit 1 ;;
  esac
done

PKG_DIR="$ROOT/packages/$PKG_NAME"
if [[ ! -d "$PKG_DIR" ]]; then
  echo "Missing $PKG_DIR" >&2
  exit 1
fi

copy_flags_for_pkg() {
  case "$PKG_NAME" in
    welvet_linux) echo "--linux" ;;
    welvet_windows) echo "--windows" ;;
    welvet_android) echo "--android" ;;
    welvet_apple) echo "--ios" ;;
    *) echo "Unknown package: $PKG_NAME" >&2; exit 1 ;;
  esac
}

# Enforce platform-only natives in the staged tarball (allowlist per impl package).
strip_impl_natives() {
  local stage="$1"
  case "$PKG_NAME" in
    welvet_linux)
      if [[ -d "$stage/native" ]]; then
        for d in "$stage/native"/*; do
          [[ -e "$d" ]] || continue
          base="${d##*/}"
          [[ "$base" == linux_amd64 || "$base" == linux_arm64" ]] || rm -rf "$d"
        done
      fi
      ;;
    welvet_windows)
      if [[ -d "$stage/native" ]]; then
        for d in "$stage/native"/*; do
          [[ -e "$d" ]] || continue
          base="${d##*/}"
          [[ "$base" == windows_amd64 || "$base" == windows_arm64" ]] || rm -rf "$d"
        done
      fi
      ;;
    welvet_android)
      rm -rf "$stage/native/linux_amd64" "$stage/native/linux_arm64" \
        "$stage/native/windows_amd64" "$stage/native/windows_arm64" \
        "$stage/native/macos_universal" "$stage/native/macos_arm64" \
        "$stage/native/macos_amd64"
      if [[ -d "$stage/native/android" ]]; then
        for d in "$stage/native/android"/*; do
          [[ -e "$d" ]] || continue
          base="${d##*/}"
          [[ "$base" == arm64-v8a || "$base" == x86_64" ]] || rm -rf "$d"
        done
      fi
      if [[ -d "$stage/native" ]]; then
        for d in "$stage/native"/*; do
          [[ -e "$d" ]] || continue
          base="${d##*/}"
          [[ "$base" == android" ]] || rm -rf "$d"
        done
      fi
      ;;
    welvet_apple)
      rm -rf "$stage/native"
      ;;
  esac
}

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/welvet-$PKG_NAME-publish.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT

echo "=== $PKG_NAME publish ==="
# shellcheck disable=SC2046
(cd "$ROOT" && bash tool/copy_native.sh $(copy_flags_for_pkg))

echo "Staging → $STAGE"
mkdir -p "$STAGE"
rsync -a \
  --exclude '.dart_tool/' \
  --exclude 'build/' \
  --exclude '.packages' \
  --exclude '.flutter-plugins' \
  --exclude '.flutter-plugins-dependencies' \
  --exclude 'android/local.properties' \
  --exclude 'android/*.iml' \
  "$PKG_DIR/" "$STAGE/"

strip_impl_natives "$STAGE"

cat >"$STAGE/.gitignore" <<'EOF'
.dart_tool/
build/
.DS_Store
*.tmp
EOF

(cd "$STAGE" && flutter pub get)

if $DRY; then
  echo "Dry run for $PKG_NAME"
  (cd "$STAGE" && flutter pub publish --dry-run)
else
  echo "Publishing $PKG_NAME to pub.dev..."
  (cd "$STAGE" && bash -c 'yes | flutter pub publish')
fi
