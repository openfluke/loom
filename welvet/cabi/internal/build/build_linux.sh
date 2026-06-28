#!/usr/bin/env bash
# Linux-only welvet C-ABI build (amd64 / arm64).
# Wrapper around build_unix.sh → builder.go
# Output: dist/linux_<arch>/welvet.so + welvet.h
#
# Usage:
#   ./build_linux.sh              # native linux arch
#   ./build_linux.sh amd64
#   ./build_linux.sh arm64
#   ./build_linux.sh all          # amd64 + arm64
#   ./build_linux.sh --clean all
#   ./build_linux.sh --test amd64
#
# Cross-compile (Fedora):
#   sudo dnf install gcc-aarch64-linux-gnu   # amd64 host → arm64
#   sudo dnf install gcc-x86_64-linux-gnu    # arm64 host → amd64

set -euo pipefail
cd "$(dirname "$0")"

EXTRA=()
ARCHES=()

host_arch="$(uname -m)"
case "$host_arch" in
	x86_64 | amd64) native=amd64 ;;
	aarch64 | arm64) native=arm64 ;;
	*) native=amd64 ;;
esac

usage() {
	echo "usage: $0 [--clean] [--test] [all | amd64 | arm64]..." >&2
	exit 1
}

for arg in "$@"; do
	case "$arg" in
		--clean | --test) EXTRA+=("$arg") ;;
		all) ARCHES=(amd64 arm64) ;;
		amd64 | arm64)
			if [[ ${#ARCHES[@]} -eq 2 ]]; then
				:
			else
				ARCHES+=("$arg")
			fi
			;;
		*) usage ;;
	esac
done

if [[ ${#ARCHES[@]} -eq 0 ]]; then
	ARCHES=("$native")
fi

first=1
for arch in "${ARCHES[@]}"; do
	if [[ $first -eq 1 ]]; then
		./build_unix.sh "${EXTRA[@]}" linux "$arch"
		first=0
	else
		# --clean only on first arch so "all" keeps both outputs
		rest=()
		for flag in "${EXTRA[@]}"; do
			[[ "$flag" == --clean ]] || rest+=("$flag")
		done
		./build_unix.sh "${rest[@]}" linux "$arch"
	fi
done
