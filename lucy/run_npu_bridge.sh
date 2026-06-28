#!/usr/bin/env bash
# Run Lucy [9] Intel NPU bridge with OpenVINO + NPU libs on LD_LIBRARY_PATH.
set -euo pipefail
LOOM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set +u
# shellcheck disable=SC1091
source "${LOOM_ROOT}/accel/intel/setup_env.sh"
set -u
cd "$(dirname "$0")"
export CGO_ENABLED=1
exec go run . "$@"
