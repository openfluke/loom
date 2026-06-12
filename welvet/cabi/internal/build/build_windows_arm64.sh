#!/usr/bin/env bash
# Native Windows ARM64 — the only script you need (WSL / Linux).
#
#   ./build_windows_arm64.sh              # lib + sync + lucy + welvet
#   ./build_windows_arm64.sh --skip-lib   # GNU lib already in webgpu/
#   ./build_windows_arm64.sh --rebuild-lib
#   ./build_windows_arm64.sh --lucy-only
#   ./build_windows_arm64.sh --welvet-only
#
# See README_WINDOWS_ARM64.md

set -euo pipefail
cd "$(dirname "$0")"

LOOM_ROOT="$(cd ../../../.. && pwd)"
LUCY_DIR="$LOOM_ROOT/lucy"

export LLVM_MINGW_HOME="${LLVM_MINGW_HOME:-/opt/llvm-mingw}"
if [ ! -x "$LLVM_MINGW_HOME/bin/aarch64-w64-mingw32-clang" ] && [ -x /mnt/c/llvm-mingw/bin/aarch64-w64-mingw32-clang ]; then
  LLVM_MINGW_HOME="/mnt/c/llvm-mingw"
  export LLVM_MINGW_HOME
fi

WEBGPU_ROOT=""
for _c in "$LOOM_ROOT/../webgpu" "$LOOM_ROOT/webgpu"; do
  if [ -d "$_c/wgpu/lib" ]; then
    WEBGPU_ROOT="$(cd "$_c" && pwd)"
    break
  fi
done

SKIP_LIB=0
REBUILD_LIB=0
BUILD_LUCY=1
BUILD_WELVET=1

for arg in "$@"; do
  case "$arg" in
    --skip-lib) SKIP_LIB=1 ;;
    --rebuild-lib) REBUILD_LIB=1 ;;
    --lucy-only) BUILD_WELVET=0 ;;
    --welvet-only) BUILD_LUCY=0 ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

webgpu_mod_version() {
  (cd "$LOOM_ROOT" && go list -m -f '{{.Version}}' github.com/openfluke/webgpu)
}

sync_webgpu_lib() {
  local arch=arm64
  local local_lib="$WEBGPU_ROOT/wgpu/lib/windows/${arch}/libwgpu_native.a"
  local webgpu_ver mod_lib

  if [ -z "$WEBGPU_ROOT" ] || [ ! -f "$local_lib" ]; then
    echo "ERROR: missing $local_lib" >&2
    exit 1
  fi
  webgpu_ver="$(webgpu_mod_version)"
  mod_lib="$HOME/go/pkg/mod/github.com/openfluke/webgpu@${webgpu_ver}/wgpu/lib/windows/${arch}/libwgpu_native.a"
  if [ ! -f "$mod_lib" ]; then
    echo "==> go mod download (webgpu ${webgpu_ver})"
    (cd "$LOOM_ROOT" && go mod download "github.com/openfluke/webgpu@${webgpu_ver}")
  fi
  if [ ! -f "$mod_lib" ]; then
    echo "ERROR: module cache missing $mod_lib" >&2
    exit 1
  fi
  chmod u+w "$mod_lib" 2>/dev/null || true
  cp -a "$local_lib" "$mod_lib"
  echo "Synced → $mod_lib"
}

build_gnu_lib() {
  local clang="$LLVM_MINGW_HOME/bin/aarch64-w64-mingw32-clang"
  local local_lib="$WEBGPU_ROOT/wgpu/lib/windows/arm64/libwgpu_native.a"
  local msvc_backup="${local_lib%.a}.msvc.a"
  local lib modlib webgpu_ver

  if [ -z "$WEBGPU_ROOT" ]; then
    echo "ERROR: webgpu repo not found next to loom" >&2
    exit 1
  fi
  if [ ! -x "$clang" ]; then
    echo "ERROR: llvm-mingw not found at $LLVM_MINGW_HOME" >&2
    exit 1
  fi

  rustup toolchain install 1.91.1
  rustup target add aarch64-pc-windows-gnullvm --toolchain 1.91.1
  export RUSTUP_TOOLCHAIN=1.91.1
  export CC_AARCH64_PC_WINDOWS_GNULLVM="$clang"
  export CXX_AARCH64_PC_WINDOWS_GNULLVM="$LLVM_MINGW_HOME/bin/aarch64-w64-mingw32-clang++"
  export CARGO_TARGET_AARCH64_PC_WINDOWS_GNULLVM_LINKER="$clang"
  if [ -f /usr/lib64/libclang.so ]; then
    export LIBCLANG_PATH=/usr/lib64
  elif [ -f /usr/lib/x86_64-linux-gnu/libclang.so ]; then
    export LIBCLANG_PATH=/usr/lib/x86_64-linux-gnu
  elif [ -f /usr/lib/aarch64-linux-gnu/libclang.so ]; then
    export LIBCLANG_PATH=/usr/lib/aarch64-linux-gnu
  else
    echo "Install clang for bindgen (dnf install clang / apt install libclang-dev)" >&2
    exit 1
  fi
  export BINDGEN_EXTRA_CLANG_ARGS_aarch64_pc_windows_gnullvm="--target=aarch64-w64-windows-gnu"

  rm -rf /tmp/wgpu-native-v29-build
  git clone --depth 1 --branch v29.0.0.0 --recursive \
    https://github.com/gfx-rs/wgpu-native.git /tmp/wgpu-native-v29-build
  (cd /tmp/wgpu-native-v29-build && cargo build --release --target aarch64-pc-windows-gnullvm)

  lib=/tmp/wgpu-native-v29-build/target/aarch64-pc-windows-gnullvm/release/libwgpu_native.a
  mkdir -p "$(dirname "$local_lib")"
  if [ -f "$local_lib" ] && [ ! -f "$msvc_backup" ]; then
    cp -a "$local_lib" "$msvc_backup"
  fi
  cp -a "$lib" "$local_lib"
  webgpu_ver="$(webgpu_mod_version)"
  modlib="$HOME/go/pkg/mod/github.com/openfluke/webgpu@${webgpu_ver}/wgpu/lib/windows/arm64/libwgpu_native.a"
  if [ -f "$modlib" ]; then
    chmod u+w "$modlib" 2>/dev/null || true
    cp -a "$lib" "$modlib"
  fi
  ls -lh "$local_lib"
}

build_lucy() {
  local outdir="$LUCY_DIR/dist/windows_arm64"
  mkdir -p "$outdir"
  export GOOS=windows GOARCH=arm64 CGO_ENABLED=1
  export CC="$LLVM_MINGW_HOME/bin/aarch64-w64-mingw32-clang"
  export CXX="$LLVM_MINGW_HOME/bin/aarch64-w64-mingw32-clang++"
  export CGO_LDFLAGS="-loleaut32 -lole32 -luuid"
  echo "==> lucy.exe"
  (cd "$LUCY_DIR" && go build -o "$outdir/lucy.exe" .)
  for dll in libunwind.dll; do
    for d in "$LLVM_MINGW_HOME/aarch64-w64-mingw32/bin" "$LLVM_MINGW_HOME/bin"; do
      if [ -f "$d/$dll" ]; then
        cp -a "$d/$dll" "$outdir/"
        break
      fi
    done
  done
  echo "  $outdir/lucy.exe"
}

GNU_LIB="${WEBGPU_ROOT:+$WEBGPU_ROOT/}wgpu/lib/windows/arm64/libwgpu_native.a"

if [ "$REBUILD_LIB" = 1 ]; then
  echo "==> Rebuilding GNU libwgpu_native.a (~10 min)..."
  build_gnu_lib
elif [ "$SKIP_LIB" = 0 ]; then
  if [ -z "$WEBGPU_ROOT" ] || [ ! -f "$GNU_LIB" ]; then
    echo "==> No GNU lib in webgpu — building (~10 min)..."
    build_gnu_lib
  else
    echo "==> Using $GNU_LIB"
  fi
else
  if [ ! -f "$GNU_LIB" ]; then
    echo "ERROR: --skip-lib but missing $GNU_LIB" >&2
    exit 1
  fi
fi

echo "==> Sync webgpu → Go module cache"
sync_webgpu_lib

if [ "$BUILD_LUCY" = 1 ]; then
  build_lucy
fi

if [ "$BUILD_WELVET" = 1 ]; then
  echo "==> welvet.dll"
  go run builder.go -os windows -arch arm64
  mkdir -p ../../../python/src/welvet
  cp -rv dist/* ../../../python/src/welvet/
fi

echo ""
echo "Done."
[ "$BUILD_LUCY" = 1 ] && echo "  $LUCY_DIR/dist/windows_arm64/lucy.exe"
[ "$BUILD_WELVET" = 1 ] && echo "  dist/windows_arm64/welvet.dll"
