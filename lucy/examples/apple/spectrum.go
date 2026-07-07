package apple

import "math"

// driftSpectrum matches seven-layer bucket labels for output drift.
type driftSpectrum int

const (
	specExact driftSpectrum = iota
	specIndustry
	specLowBit
	specDrift
	specHeavyDrift
	specBroken
	specFatal
)

func (s driftSpectrum) String() string {
	switch s {
	case specExact:
		return "💎 EXACT"
	case specIndustry:
		return "✅ INDUS"
	case specLowBit:
		return "🟨 LOWBIT"
	case specDrift:
		return "🟠 DRIFT"
	case specHeavyDrift:
		return "🟤 H-DRIFT"
	case specBroken:
		return "❌ BROKE"
	default:
		return "💀 FATAL"
	}
}

func classifyDrift(diff, tol float64) driftSpectrum {
	if math.IsNaN(diff) || math.IsInf(diff, 0) {
		return specFatal
	}
	if diff == 0 {
		return specExact
	}
	if diff <= tol {
		return specIndustry
	}
	if diff <= tol*10 {
		return specLowBit
	}
	if diff <= 0.1 {
		return specDrift
	}
	if diff <= 1.0 {
		return specHeavyDrift
	}
	return specBroken
}

func inferDriftTolerance(dtypeLabel string) float64 {
	switch dtypeLabel {
	case "FP32":
		return 1e-7
	case "FP16":
		return 1e-5
	case "INT16", "INT8", "INT4":
		return 1e-3
	default:
		return 1e-7
	}
}
