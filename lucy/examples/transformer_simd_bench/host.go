package tfsimdbench

import (
	"runtime"

	"github.com/openfluke/loom/poly/simd"
)

func simdKind() string {
	if !simd.SimdEnabled() {
		return "none"
	}
	switch runtime.GOARCH {
	case "amd64":
		return "AVX2/FMA"
	case "arm64":
		return "NEON (asm dot kernel)"
	default:
		return "generic"
	}
}

func currentHostInfo() hostInfo {
	return hostInfo{
		Arch:     runtime.GOARCH,
		NumCPU:   runtime.NumCPU(),
		SimdKind: simdKind(),
	}
}
