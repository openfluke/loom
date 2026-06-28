#!/usr/bin/env bash
# Publish main welvet to pub.dev (https://pub.dev/packages/welvet).
#
# Dart API only. Platform natives ship in welvet_linux, welvet_windows,
# welvet_android, welvet_apple (iOS + macOS).
#
# Usage:
#   bash tool/publish.sh                    # dry-run
#   bash tool/publish.sh --publish          # upload
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRY=true
PUBLISH=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --publish) PUBLISH=true; DRY=false; shift ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *) echo "Unknown: $1" >&2; exit 1 ;;
  esac
done

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/welvet-publish.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT

echo "=== welvet (main) publish ==="

echo "Running tests in source tree (pubspec_overrides for local impl packages)..."
(cd "$ROOT" && flutter pub get && flutter test)

echo "Staging package → $STAGE"
mkdir -p "$STAGE"
rsync -a \
  --exclude '.dart_tool/' \
  --exclude 'build/' \
  --exclude '.packages' \
  --exclude '.flutter-plugins' \
  --exclude '.flutter-plugins-dependencies' \
  --exclude 'packages/' \
  --exclude 'pubspec_overrides.yaml' \
  --exclude 'android/' \
  --exclude 'ios/' \
  --exclude 'linux/' \
  --exclude 'windows/' \
  --exclude 'src/' \
  --exclude 'macos/' \
  --exclude 'native/' \
  "$ROOT/" "$STAGE/"

cat >"$STAGE/.gitignore" <<'EOF'
.dart_tool/
build/
.DS_Store
*.tmp
EOF

if $DRY; then
  echo ""
  echo "Dry run (tarball size; --skip-validation until impl packages are on pub.dev)"
  (cd "$STAGE" && flutter pub publish --dry-run --skip-validation)
else
  echo ""
  echo "Publishing welvet to pub.dev (auto-confirm)..."
  (cd "$STAGE" && flutter pub get && bash -c 'yes | flutter pub publish')
fi
