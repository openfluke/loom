#!/usr/bin/env bash
# welvet C-ABI build script — Linux / macOS
#
# Usage:
#   ./build_unix.sh                          # native platform + arch
#   ./build_unix.sh all                      # every target
#   ./build_unix.sh linux amd64
#   ./build_unix.sh linux arm64
#   ./build_unix.sh linux armv7
#   ./build_unix.sh linux x86
#   ./build_unix.sh darwin amd64
#   ./build_unix.sh darwin arm64
#   ./build_unix.sh darwin universal         # fat binary (macOS only)
#   ./build_unix.sh windows amd64
#   ./build_unix.sh windows arm64
#   ./build_unix.sh windows x86
#   ./build_unix.sh android arm64            # requires ANDROID_NDK_HOME
#   ./build_unix.sh android armv7
#   ./build_unix.sh android x86_64
#   ./build_unix.sh android x86
#   ./build_unix.sh ios arm64                # device  (macOS only)
#   ./build_unix.sh ios sim_amd64            # x86_64 simulator
#   ./build_unix.sh ios sim_arm64            # ARM64 simulator (M1+)
#   ./build_unix.sh ios xcframework          # full XCFramework (macOS only)
#
# Options:
#   --clean    Remove dist/ before building
#   --test     Run C verification after build (native only)

set -e
cd "$(dirname "$0")"

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
    amd64|arm64|arm|386|armv7|x86|x86_64|universal|xcframework|sim_amd64|sim_arm64)
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
    armv7l)        TARGET_ARCH="armv7"  ;;
    i686|i386)     TARGET_ARCH="386"    ;;
    *) echo "Cannot auto-detect arch"; exit 1 ;;
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
