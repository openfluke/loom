package poly

import (
	"fmt"
	"math"
	"sort"
)

// EnsembleMatch represents a pair of models that complement each other.
type EnsembleMatch struct {
	ModelA   string
	ModelB   string
	Coverage float64 // Combined coverage (0.0 - 1.0)
	Overlap  float64 // Percentage of samples both got right
}

// ModelPerformance holds the correctness mask for a specific model.
type ModelPerformance struct {
	ModelID string
	// Mask[i] is true if the model correctly handled sample i.
	Mask []bool
}

// MajorityVote performs hard-voting across multiple model outputs (class indices).
func MajorityVote(outputs [][]int) []int {
	if len(outputs) == 0 { return nil }
	numSamples := len(outputs[0])
	numModels := len(outputs)
	final := make([]int, numSamples)

	for s := 0; s < numSamples; s++ {
		counts := make(map[int]int)
		for m := 0; m < numModels; m++ {
			counts[outputs[m][s]]++
		}
		
		maxCount := -1
		winner := -1
		for val, count := range counts {
			if count > maxCount {
				maxCount = count
				winner = val
			}
		}
		final[s] = winner
	}
	return final
}

// PerformanceSimilarity calculates cosine similarity between two model masks.
func PerformanceSimilarity(mA, mB ModelPerformance) float64 {
	if len(mA.Mask) != len(mB.Mask) || len(mA.Mask) == 0 { return 0 }
	
	dot := 0.0
	normA := 0.0
	normB := 0.0
	
	for i := range mA.Mask {
		valA := 0.0; if mA.Mask[i] { valA = 1.0 }
		valB := 0.0; if mB.Mask[i] { valB = 1.0 }
		
		dot += valA * valB
		normA += valA * valA
		normB += valB * valB
	}
	
	if normA == 0 || normB == 0 { return 0 }
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// FindComplementaryMatches identifies pairs of models whose combined coverage is maximized.
func FindComplementaryMatches(models []ModelPerformance, minCoverage float64) []EnsembleMatch {
	var matches []EnsembleMatch

	for i := 0; i < len(models); i++ {
		for j := i + 1; j < len(models); j++ {
			mA := models[i]
			mB := models[j]

			total := len(mA.Mask)
			if total != len(mB.Mask) || total == 0 { continue }

			covered := 0
			overlap := 0

			for k := 0; k < total; k++ {
				if mA.Mask[k] || mB.Mask[k] { covered++ }
				if mA.Mask[k] && mB.Mask[k] { overlap++ }
			}

			cov := float64(covered) / float64(total)
			ovl := float64(overlap) / float64(total)

			if cov >= minCoverage {
				matches = append(matches, EnsembleMatch{
					ModelA:   mA.ModelID,
					ModelB:   mB.ModelID,
					Coverage: cov,
					Overlap:  ovl,
				})
			}
		}
	}

	sort.Slice(matches, func(i, j int) bool {
		if matches[i].Coverage != matches[j].Coverage {
			return matches[i].Coverage > matches[j].Coverage
		}
		return matches[i].Overlap < matches[j].Overlap
	})

	return matches
}

// PrintEnsembleReport generates a human-readable summary of the best matches.
func PrintEnsembleReport(matches []EnsembleMatch, topN int) {
	fmt.Println("\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557")
	fmt.Println("\u2551           POLY-ENSEMBLE DISCOVERY REPORT            \u2551")
	fmt.Println("\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563")
	fmt.Println("\u2551 Pair                        | Coverage | Overlap  | Stat  \u2551")
	fmt.Println("\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563")

	for i, m := range matches {
		if i >= topN { break }
		status := "OK"
		if m.Coverage >= 0.99 { status = "\u2B50" }
		pair := fmt.Sprintf("%s + %s", m.ModelA, m.ModelB)
		if len(pair) > 28 { pair = pair[:25] + "..." }
		fmt.Printf("\u2551 %-28s | %7.1f%% | %7.1f%% | %-5s \u2551\n", pair, m.Coverage*100, m.Overlap*100, status)
	}
	fmt.Println("\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569")
}
