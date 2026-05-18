#!/usr/bin/env bash
# welvet C-ABI build script — Linux / macOS
#
# Usage (64-bit targets only; iOS simulator slices are not built):
#   ./build_unix.sh                          # native platform + arch
#   ./build_unix.sh all                      # every target
#   ./build_unix.sh linux amd64
#   ./build_unix.sh linux arm64
#   ./build_unix.sh darwin amd64
#   ./build_unix.sh darwin arm64
#   ./build_unix.sh darwin universal         # fat binary (macOS only)
#   ./build_unix.sh windows amd64
#   ./build_unix.sh windows arm64
#   ./build_unix.sh android arm64            # NDK: ANDROID_HOME/ndk or ANDROID_NDK_HOME
#   ./build_unix.sh android x86_64
#   ./build_unix.sh ios arm64                # device (macOS only)
#   ./build_unix.sh ios xcframework          # device-only XCFramework (macOS only)
#
# Options:
#   --clean    Remove dist/ before building
#   --test     Run C verification after build (native only)
#
# Environment (optional):
#   LLVM_MINGW_HOME       llvm-mingw unpack root (Windows/arm64 only; not added to PATH)
#   BUILD_TOOLCHAIN_PATH  extra bin dirs, colon-separated
#   .build_env            local file in this dir (sourced if present)
#   ANDROID_NDK_HOME   Explicit NDK root (must contain toolchains/llvm/prebuilt/)
#   ANDROID_HOME / ANDROID_SDK_ROOT
#                      If ANDROID_NDK_HOME is unset, newest $ROOT/ndk/<version> is used
#   HOMEBREW_PREFIX    Used with Homebrew android-ndk layout under share/android-ndk
#
# Cross-compiling to Linux / Windows/arm64 from macOS (install once):
#   brew tap messense/macos-cross-toolchains
#   brew install x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu
#   brew install mingw-w64
#   # Windows/arm64: Homebrew mingw-w64 has no aarch64 GCC — add llvm-mingw bin/ to PATH, e.g.:
#   #   https://github.com/mstorsjo/llvm-mingw/releases (macOS universal tarball)

set -e
cd "$(dirname "$0")"

# ── Cross-compiler PATH (builder uses exec.LookPath; does not load your ~/.zshrc) ──
_prepend_path() {
  local d="$1"
  [ -n "$d" ] && [ -d "$d" ] || return 0
  case ":$PATH:" in
    *":$d:"*) ;;
    *) export PATH="$d:$PATH" ;;
  esac
}

# Optional: copy .build_env.example → .build_env and set LLVM_MINGW_HOME / BUILD_TOOLCHAIN_PATH
if [ -f ./.build_env ]; then
  set -a
  # shellcheck disable=SC1091
  source ./.build_env
  set +a
fi

if command -v brew >/dev/null 2>&1; then
  _prepend_path "$(brew --prefix 2>/dev/null)/bin"
fi
_prepend_path "/opt/homebrew/bin"
_prepend_path "/usr/local/bin"

# llvm-mingw is NOT prepended to PATH — its "clang" would break macOS/Linux/Android builds.
# Set LLVM_MINGW_HOME in .build_env; builder.go resolves …/bin/aarch64-w64-mingw32-clang directly.
if [ -z "${LLVM_MINGW_HOME:-}" ] && [ -d "$HOME/llvm-mingw/bin" ]; then
  export LLVM_MINGW_HOME="$HOME/llvm-mingw"
fi

if [ -n "${BUILD_TOOLCHAIN_PATH:-}" ]; then
  IFS=':' read -r -a _btp <<< "$BUILD_TOOLCHAIN_PATH"
  for _d in "${_btp[@]}"; do
    _prepend_path "$_d"
  done
fi

echo "Cross-compilers:"
for _c in x86_64-linux-gnu-gcc aarch64-linux-gnu-gcc x86_64-w64-mingw32-gcc; do
  if command -v "$_c" >/dev/null 2>&1; then
    echo "  ✓ $(command -v "$_c")"
  fi
done
if [ -n "${LLVM_MINGW_HOME:-}" ] && [ -x "$LLVM_MINGW_HOME/bin/aarch64-w64-mingw32-clang" ]; then
  echo "  ✓ $LLVM_MINGW_HOME/bin/aarch64-w64-mingw32-clang (Windows/arm64)"
fi

# Export ANDROID_NDK_HOME when Android Studio installed the NDK but only ANDROID_HOME is set.
if [ -z "${ANDROID_NDK_HOME:-}" ]; then
  _sdk="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$HOME/Library/Android/sdk}}"
  if [ -d "$_sdk/ndk" ]; then
    _pick=""
    for _name in $(ls -1 "$_sdk/ndk" 2>/dev/null | LC_ALL=C sort -V); do
      _d="$_sdk/ndk/$_name"
      [ -d "$_d/toolchains/llvm/prebuilt" ] || continue
      _pick="$_d"
    done
    if [ -n "$_pick" ]; then
      export ANDROID_NDK_HOME="$_pick"
    fi
  fi
fi
if [ -z "${ANDROID_NDK_HOME:-}" ] && [ -d "$HOME/Library/Android/sdk/ndk-bundle/toolchains/llvm/prebuilt" ]; then
  export ANDROID_NDK_HOME="$HOME/Library/Android/sdk/ndk-bundle"
fi
for _root in "/opt/homebrew/share/android-ndk" "/usr/local/share/android-ndk"; do
  if [ -z "${ANDROID_NDK_HOME:-}" ] && [ -d "$_root/toolchains/llvm/prebuilt" ]; then
    export ANDROID_NDK_HOME="$_root"
    break
  fi
done
if [ -z "${ANDROID_NDK_HOME:-}" ] && command -v brew >/dev/null 2>&1; then
  _bp="$(brew --prefix android-ndk 2>/dev/null || true)"
  if [ -n "$_bp" ]; then
    for _try in "$_bp/share/android-ndk" "$_bp"; do
      if [ -d "$_try/toolchains/llvm/prebuilt" ]; then
        export ANDROID_NDK_HOME="$_try"
        break
      fi
    done
  fi
fi

TARGET_OS=""
TARGET_ARCH=""
EXTRA_FLAGS=""

for arg in "$@"; do
  case "$arg" in
    --clean) EXTRA_FLAGS="$EXTRA_FLAGS -clean" ;;
    --test)  EXTRA_FLAGS="$EXTRA_FLAGS -test"  ;;
    all)     TARGET_OS="all" ;;
    linux|darwin|windows|android|ios)
      TARGET_OS="$arg" ;;
    amd64|arm64|x86_64|universal|xcframework)
      TARGET_ARCH="$arg" ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

# Auto-detect native platform if nothing specified
if [ -z "$TARGET_OS" ]; then
  case "$(uname -s)" in
    Linux)  TARGET_OS="linux"  ;;
    Darwin) TARGET_OS="darwin" ;;
    MINGW*|MSYS*|CYGWIN*) TARGET_OS="windows" ;;
    *) echo "Cannot auto-detect OS"; exit 1 ;;
  esac
fi

if [ -z "$TARGET_ARCH" ] && [ "$TARGET_OS" != "all" ]; then
  case "$(uname -m)" in
    x86_64|amd64)  TARGET_ARCH="amd64"  ;;
    aarch64|arm64) TARGET_ARCH="arm64"  ;;
    *) echo "Cannot auto-detect 64-bit arch from: $(uname -m)"; exit 1 ;;
  esac
fi

if [ "$TARGET_OS" = "all" ]; then
  echo "Building all platforms..."
  go run builder.go -os all $EXTRA_FLAGS
else
  echo "Building $TARGET_OS $TARGET_ARCH..."
  go run builder.go -os "$TARGET_OS" -arch "$TARGET_ARCH" $EXTRA_FLAGS
fi

# Mirror to Python source
echo "Mirroring to Python source..."
mkdir -p ../../../python/src/welvet
cp -rv dist/* ../../../python/src/welvet/
echo "Mirror complete."
