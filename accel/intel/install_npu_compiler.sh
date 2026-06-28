#!/usr/bin/env bash
# Install Intel NPU compiler + firmware libs locally (OpenVINO 2025.4 needs these).
#
# Fedora's intel-npu-driver RPM ships libze_intel_npu.so only.
# OpenVINO 2025.4 NPU plugin uses Compiler-in-Driver → libnpu_driver_compiler.so
#
# Usage: ./install_npu_compiler.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEPS="${ROOT}/deps/npu-driver"
VERSION="${NPU_DRIVER_VERSION:-v1.32.0}"
ASSET="linux-npu-driver-${VERSION}.20260402-23905121947-ubuntu2404.tar.gz"
URL="https://github.com/intel/linux-npu-driver/releases/download/${VERSION}/${ASSET}"
STAGE="${DEPS}/stage"
ROOTFS="${DEPS}/root"

mkdir -p "$DEPS"

if [[ ! -f "${DEPS}/${ASSET}" ]]; then
  echo "Downloading ${URL} ..."
  curl -fL --retry 3 -o "${DEPS}/${ASSET}" "$URL"
fi

rm -rf "$STAGE" "$ROOTFS"
mkdir -p "$STAGE" "$ROOTFS"

tar -xf "${DEPS}/${ASSET}" -C "$STAGE"

shopt -s nullglob
debs=("$STAGE"/intel-driver-compiler-npu_*.deb "$STAGE"/intel-fw-npu_*.deb)
if ((${#debs[@]} == 0)); then
  debs=("$STAGE"/linux-npu-driver-*/intel-driver-compiler-npu_*.deb "$STAGE"/linux-npu-driver-*/intel-fw-npu_*.deb)
fi
if ((${#debs[@]} == 0)); then
  echo "No compiler/fw .deb found in ${ASSET} — check NPU_DRIVER_VERSION." >&2
  exit 1
fi

for deb in "${debs[@]}"; do
  echo "Extracting $(basename "$deb") ..."
  work="${STAGE}/extract-$(basename "$deb")"
  mkdir -p "$work"
  (cd "$work" && ar x "$deb")
  data_tar=( "$work"/data.tar.* )
  tar -xf "${data_tar[0]}" -C "$ROOTFS"
done

if [[ ! -f "${ROOTFS}/usr/lib/x86_64-linux-gnu/libnpu_driver_compiler.so" ]]; then
  echo "libnpu_driver_compiler.so not found after extract." >&2
  exit 1
fi

LIBDIR="${ROOTFS}/usr/lib/x86_64-linux-gnu"
FWDIR="${ROOTFS}/lib/firmware/updates/intel/vpu"

# Patch setup_env.sh (preserve OpenVINO block if present)
OV_BLOCK=""
if [[ -f "${ROOT}/setup_env.sh" ]]; then
  OV_BLOCK="$(grep -E '^export INTEL_OPENVINO_DIR=' "${ROOT}/setup_env.sh" || true)"
fi

cat > "${ROOT}/setup_env.sh" <<EOF
#!/usr/bin/env bash
${OV_BLOCK}
if [[ -n "\${INTEL_OPENVINO_DIR:-}" && -f "\${INTEL_OPENVINO_DIR}/setupvars.sh" ]]; then
  # shellcheck disable=SC1091
  source "\${INTEL_OPENVINO_DIR}/setupvars.sh"
fi
export INTEL_NPU_LIBDIR="${LIBDIR}"
export INTEL_NPU_FWDIR="${FWDIR}"
export LD_LIBRARY_PATH="\${INTEL_NPU_LIBDIR}:/usr/lib64:\${LD_LIBRARY_PATH:-}"
if [[ -d "\${INTEL_NPU_FWDIR}" ]]; then
  export FW_SEARCH_PATH="\${INTEL_NPU_FWDIR}"
fi
return 0 2>/dev/null || true
EOF
chmod +x "${ROOT}/setup_env.sh"

echo
echo "Installed NPU compiler libs to ${LIBDIR}"
echo "Updated ${ROOT}/setup_env.sh"
echo
echo "Also ensure (once):"
echo "  sudo dnf install intel-npu-driver oneapi-level-zero"
echo "  sudo gpasswd -a \"\$USER\" render   # then log out/in or: newgrp render"
echo
echo "Then:"
echo "  cd ${ROOT} && source ./setup_env.sh && ./run_bench.sh"
