#!/usr/bin/env bash
# Copy pre-built loom/welvet/cabi artifacts into federated package native/ trees.
#
# Main `welvet` is Dart-only; all platform natives ship in packages/*.
#
# Usage:
#   bash tool/copy_native.sh              # host platforms
#   bash tool/copy_native.sh --all
#   bash tool/copy_native.sh --linux --macos
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST="${LOOM_CABI_DIST:-$ROOT/../cabi/internal/build/dist}"
SOUL_NAT="${ROOT}/../../../soulglitch/native"

PKG_LINUX="$ROOT/packages/welvet_linux"
PKG_WINDOWS="$ROOT/packages/welvet_windows"
PKG_ANDROID="$ROOT/packages/welvet_android"
PKG_APPLE="$ROOT/packages/welvet_apple"

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
      head -14 "$0" | tail -12
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

# Keep only listed subdirs under a native/ tree (drops stale cross-platform copies).
prune_native_dir() {
  local dir="$1"
  shift
  local keep=("$@")
  if [[ ! -d "$dir" ]]; then return 0; fi
  for entry in "$dir"/*; do
    [[ -e "$entry" ]] || continue
    local base="${entry##*/}"
    local ok=false
    for k in "${keep[@]}"; do
      if [[ "$base" == "$k" ]]; then ok=true; break; fi
    done
    if ! $ok; then
      rm -rf "$entry"
      echo "  − pruned ${entry#$ROOT/}"
    fi
  done
}

echo "copy_native (federated)"
echo "  DIST: $DIST"

if $RUN_LINUX; then
  nat="$PKG_LINUX/native"
  if [[ -f "$DIST/linux_amd64/welvet.so" ]]; then
    cp_file "$DIST/linux_amd64/welvet.so" "$nat/linux_amd64/libwelvet.so"
  else
    fallback "$SOUL_NAT/linux/libwelvet.so" "$nat/linux_amd64/libwelvet.so" || \
      echo "  ✗ linux_amd64 (build cabi or copy soulglitch/native/linux)"
  fi
  if [[ -f "$DIST/linux_arm64/welvet.so" ]]; then
    cp_file "$DIST/linux_arm64/welvet.so" "$nat/linux_arm64/libwelvet.so"
  fi
  prune_native_dir "$nat" linux_amd64 linux_arm64
fi

if $RUN_MACOS; then
  nat="$PKG_APPLE/native"
  mac_src=""
  for c in macos_universal/welvet.dylib macos_arm64/welvet.dylib; do
    [[ -f "$DIST/$c" ]] && mac_src="$DIST/$c" && break
  done
  if [[ -n "$mac_src" ]]; then
    cp_file "$mac_src" "$nat/macos_universal/libwelvet.dylib"
    cp_file "$mac_src" "$PKG_APPLE/macos/Frameworks/libwelvet.dylib"
    install_name_tool -id "@rpath/libwelvet.dylib" "$PKG_APPLE/macos/Frameworks/libwelvet.dylib"
  else
    fallback "$SOUL_NAT/macos/libwelvet.dylib" "$nat/macos_universal/libwelvet.dylib" && \
      cp_file "$nat/macos_universal/libwelvet.dylib" "$PKG_APPLE/macos/Frameworks/libwelvet.dylib" && \
      install_name_tool -id "@rpath/libwelvet.dylib" "$PKG_APPLE/macos/Frameworks/libwelvet.dylib" || \
      echo "  ✗ macOS dylib"
  fi
  prune_native_dir "$nat" macos_universal macos_arm64 macos_amd64
fi

if $RUN_WINDOWS; then
  nat="$PKG_WINDOWS/native"
  if [[ -f "$DIST/windows_amd64/welvet.dll" ]]; then
    cp_file "$DIST/windows_amd64/welvet.dll" "$nat/windows_amd64/welvet.dll"
  else
    fallback "$SOUL_NAT/windows/welvet.dll" "$nat/windows_amd64/welvet.dll" || \
      echo "  ✗ windows_amd64"
  fi
  if [[ -f "$DIST/windows_arm64/welvet.dll" ]]; then
    cp_file "$DIST/windows_arm64/welvet.dll" "$nat/windows_arm64/welvet.dll"
  fi
  prune_native_dir "$nat" windows_amd64 windows_arm64
fi

if $RUN_ANDROID; then
  nat="$PKG_ANDROID/native"
  a64="$DIST/android_arm64/welvet.so"
  x64="$DIST/android_x86_64/welvet.so"
  if [[ -f "$a64" ]]; then
    cp_file "$a64" "$nat/android/arm64-v8a/libwelvet.so"
  else
    fallback "$SOUL_NAT/android/arm64-v8a/libwelvet.so" "$nat/android/arm64-v8a/libwelvet.so" || \
      echo "  ✗ android arm64-v8a"
  fi
  if [[ -f "$x64" ]]; then
    cp_file "$x64" "$nat/android/x86_64/libwelvet.so"
  else
    fallback "$SOUL_NAT/android/x86_64/libwelvet.so" "$nat/android/x86_64/libwelvet.so" || \
      echo "  ✗ android x86_64"
  fi
  prune_native_dir "$nat" android
  prune_native_dir "$nat/android" arm64-v8a x86_64
fi

if $RUN_IOS; then
  dst_xcf="$PKG_APPLE/ios/Welvet.xcframework"
  src_xcf="$DIST/ios_xcframework/Welvet.xcframework"
  soul_xcf="$SOUL_NAT/ios/Welvet.xcframework"
  if [[ -d "$src_xcf" ]]; then
    rm -rf "$dst_xcf"
    cp -R "$src_xcf" "$dst_xcf"
    echo "  ✓ packages/welvet_apple/ios/Welvet.xcframework (from dist)"
  elif [[ -d "$soul_xcf" ]]; then
    rm -rf "$dst_xcf"
    cp -R "$soul_xcf" "$dst_xcf"
    echo "  ✓ packages/welvet_apple/ios/Welvet.xcframework (from soulglitch)"
  else
    echo "  ✗ iOS Welvet.xcframework"
  fi
fi

echo "Done."
