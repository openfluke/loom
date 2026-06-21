#!/usr/bin/env bash
# Copy pre-built loom/welvet/cabi artifacts into loom/welvet/dart/native/.
#
# Usage:
#   bash tool/copy_native.sh              # host platforms
#   bash tool/copy_native.sh --all
#   bash tool/copy_native.sh --linux --macos
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST="${LOOM_CABI_DIST:-$ROOT/../../cabi/internal/build/dist}"
NAT="$ROOT/native"
SOUL_NAT="${ROOT}/../../../soulglitch/native"

RUN_MACOS=false RUN_LINUX=false RUN_ANDROID=false RUN_WINDOWS=false RUN_IOS=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) RUN_MACOS=true RUN_LINUX=true RUN_ANDROID=true RUN_WINDOWS=true RUN_IOS=true; shift ;;
    --macos) RUN_MACOS=true; shift ;;
    --linux) RUN_LINUX=true; shift ;;
    --android) RUN_ANDROID=true; shift ;;
    --windows) RUN_WINDOWS=true; shift ;;
    --ios) RUN_IOS=true; shift ;;
    -h|--help)
      head -12 "$0" | tail -10
      exit 0
      ;;
    *) echo "Unknown: $1" >&2; exit 1 ;;
  esac
done

if ! $RUN_MACOS && ! $RUN_LINUX && ! $RUN_ANDROID && ! $RUN_WINDOWS && ! $RUN_IOS; then
  case "$(uname -s)" in
    Darwin) RUN_MACOS=true RUN_IOS=true ;;
    Linux) RUN_LINUX=true ;;
    MINGW*|MSYS*|CYGWIN*) RUN_WINDOWS=true ;;
    *) RUN_MACOS=true RUN_LINUX=true RUN_ANDROID=true RUN_WINDOWS=true RUN_IOS=true ;;
  esac
fi

cp_file() {
  local src="$1" dst="$2"
  mkdir -p "$(dirname "$dst")"
  cp -f "$src" "$dst"
  echo "  ✓ ${dst#$ROOT/}"
}

fallback() {
  local soul="$1" dst="$2"
  if [[ -f "$soul" ]]; then
    cp_file "$soul" "$dst"
    return 0
  fi
  return 1
}

echo "copy_native → $NAT"
echo "  DIST: $DIST"

if $RUN_LINUX; then
  if [[ -f "$DIST/linux_amd64/welvet.so" ]]; then
    cp_file "$DIST/linux_amd64/welvet.so" "$NAT/linux_amd64/libwelvet.so"
  else
    fallback "$SOUL_NAT/linux/libwelvet.so" "$NAT/linux_amd64/libwelvet.so" || \
      echo "  ✗ linux_amd64 (build cabi or copy soulglitch/native/linux)"
  fi
  if [[ -f "$DIST/linux_arm64/welvet.so" ]]; then
    cp_file "$DIST/linux_arm64/welvet.so" "$NAT/linux_arm64/libwelvet.so"
  fi
fi

if $RUN_MACOS; then
  mac_src=""
  for c in macos_universal/welvet.dylib macos_arm64/welvet.dylib; do
    [[ -f "$DIST/$c" ]] && mac_src="$DIST/$c" && break
  done
  if [[ -n "$mac_src" ]]; then
    cp_file "$mac_src" "$NAT/macos_universal/libwelvet.dylib"
    cp_file "$mac_src" "$ROOT/macos/Frameworks/libwelvet.dylib"
  else
    fallback "$SOUL_NAT/macos/libwelvet.dylib" "$NAT/macos_universal/libwelvet.dylib" && \
      cp_file "$NAT/macos_universal/libwelvet.dylib" "$ROOT/macos/Frameworks/libwelvet.dylib" || \
      echo "  ✗ macOS dylib"
  fi
fi

if $RUN_WINDOWS; then
  if [[ -f "$DIST/windows_amd64/welvet.dll" ]]; then
    cp_file "$DIST/windows_amd64/welvet.dll" "$NAT/windows_amd64/welvet.dll"
  else
    fallback "$SOUL_NAT/windows/welvet.dll" "$NAT/windows_amd64/welvet.dll" || \
      echo "  ✗ windows_amd64"
  fi
  if [[ -f "$DIST/windows_arm64/welvet.dll" ]]; then
    cp_file "$DIST/windows_arm64/welvet.dll" "$NAT/windows_arm64/welvet.dll"
  fi
fi

if $RUN_ANDROID; then
  a64="$DIST/android_arm64/welvet.so"
  x64="$DIST/android_x86_64/welvet.so"
  if [[ -f "$a64" ]]; then
    cp_file "$a64" "$NAT/android/arm64-v8a/libwelvet.so"
  else
    fallback "$SOUL_NAT/android/arm64-v8a/libwelvet.so" "$NAT/android/arm64-v8a/libwelvet.so" || \
      echo "  ✗ android arm64-v8a"
  fi
  if [[ -f "$x64" ]]; then
    cp_file "$x64" "$NAT/android/x86_64/libwelvet.so"
  else
    fallback "$SOUL_NAT/android/x86_64/libwelvet.so" "$NAT/android/x86_64/libwelvet.so" || \
      echo "  ✗ android x86_64"
  fi
fi

if $RUN_IOS; then
  src_xcf="$DIST/ios_xcframework/Welvet.xcframework"
  soul_xcf="$SOUL_NAT/ios/Welvet.xcframework"
  if [[ -d "$src_xcf" ]]; then
    rm -rf "$ROOT/ios/Welvet.xcframework"
    cp -R "$src_xcf" "$ROOT/ios/Welvet.xcframework"
    echo "  ✓ ios/Welvet.xcframework (from dist)"
  elif [[ -d "$soul_xcf" ]]; then
    rm -rf "$ROOT/ios/Welvet.xcframework"
    cp -R "$soul_xcf" "$ROOT/ios/Welvet.xcframework"
    echo "  ✓ ios/Welvet.xcframework (from soulglitch)"
  else
    echo "  ✗ iOS Welvet.xcframework"
  fi
fi

echo "Done."
