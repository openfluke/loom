package nn

import (
	"fmt"
	"sort"
)

// EnsembleMatch represents a pair of models that complement each other
type EnsembleMatch struct {
	ModelA   string
	ModelB   string
	Coverage float64 // Combined coverage (0.0 - 1.0)
	Overlap  float64 // Percentage of samples both got right
}

// ModelPerformance holds the correctness mask for a specific model
type ModelPerformance struct {
	ModelID string
	// Mask is a comprehensive boolean vector where Mask[i] is true if the model
	// correctly handled sample i.
	Mask []bool
}

// FindComplementaryMatches identifies pairs of models whose combined coverage
// is maximized (ideally 100%).
// models: List of model performances
// minCoverage: Minimum combined coverage to report (e.g., 0.95 for 95%)
func FindComplementaryMatches(models []ModelPerformance, minCoverage float64) []EnsembleMatch {
	var matches []EnsembleMatch

	for i := 0; i < len(models); i++ {
		for j := i + 1; j < len(models); j++ {
			mA := models[i]
			mB := models[j]

			// Reset counters
			totalSamples := len(mA.Mask)
			if totalSamples != len(mB.Mask) {
				continue // Should not happen if data is consistent
			}
			if totalSamples == 0 {
				continue
			}

			coveredCount := 0
			overlapCount := 0

			for k := 0; k < totalSamples; k++ {
				// Combined coverage: A OR B
				if mA.Mask[k] || mB.Mask[k] {
					coveredCount++
				}
				// Overlap: A AND B
				if mA.Mask[k] && mB.Mask[k] {
					overlapCount++
				}
			}

			coverage := float64(coveredCount) / float64(totalSamples)
			overlap := float64(overlapCount) / float64(totalSamples)

			if coverage >= minCoverage {
				matches = append(matches, EnsembleMatch{
					ModelA:   mA.ModelID,
					ModelB:   mB.ModelID,
					Coverage: coverage,
					Overlap:  overlap,
				})
			}
		}
	}

	// Sort by coverage (descending), then overlap (ascending - less overlap is better for diversity)
	sort.Slice(matches, func(i, j int) bool {
		if matches[i].Coverage != matches[j].Coverage {
			return matches[i].Coverage > matches[j].Coverage
		}
		return matches[i].Overlap < matches[j].Overlap
	})

	return matches
}

// ConvertMaskToFloat converts a boolean mask to a float vector (0.0/1.0)
// Suitable for clustering input.
func ConvertMaskToFloat(mask []bool) []float32 {
	vec := make([]float32, len(mask))
	for i, v := range mask {
		if v {
			vec[i] = 1.0
		} else {
			vec[i] = 0.0
		}
	}
	return vec
}

// PrintEnsembleReport generates a human-readable summary of the best matches
func PrintEnsembleReport(matches []EnsembleMatch, topN int) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                       ENSEMBLE DISCOVERY REPORT                                  ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║ Pair                                     | Coverage | Overlap  | Recommendation  ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════╣")

	count := 0
	for _, m := range matches {
		if count >= topN {
			break
		}

		rec := ""
		if m.Coverage >= 0.999 {
			rec = "🌟 PERFECT"
		} else if m.Coverage > 0.90 {
			rec = "✅ Excellent"
		} else {
			rec = "⚠️ Good"
		}

		pairStr := fmt.Sprintf("%s + %s", truncate(m.ModelA, 18), truncate(m.ModelB, 18))
		fmt.Printf("║ %-40s | %7.1f%% | %7.1f%% | %-15s ║\n",
			pairStr, m.Coverage*100, m.Overlap*100, rec)
		count++
	}
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════╝")
}

func truncate(s string, l int) string {
	if len(s) > l {
		return s[:l-3] + "..."
	}
	return s
}
