#!/usr/bin/env bash
# Run Lucy [9] Intel NPU bridge with OpenVINO + NPU libs on LD_LIBRARY_PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
set +u
# shellcheck disable=SC1091
source "${ROOT}/npu/intel/example/setup_env.sh"
set -u
cd "$(dirname "$0")"
export CGO_ENABLED=1
exec go run . "$@"
