package fivelayer

import "fmt"

func markOK(ok bool) string {
	if ok {
		return "PASS"
	}
	return "FAIL"
}

// DTypeRow is one numerical-type result (filled during the run, printed after all training).
type DTypeRow struct {
	DType        string
	LossInit     float64
	LossFinal    float64
	BeforeBucket string
	AfterBucket  string
	BeforeOK     bool
	AfterOK      bool
	NativeOK     bool
	Learned      bool
	OverallOK    bool
	Err          string
}

// LayerSummary aggregates one layer-type run (21 dtypes).
type LayerSummary struct {
	Name         string
	Passed       int
	Failed       int
	Rows         []DTypeRow
	LayerPassed  bool
}

var sessionLayers []LayerSummary

// ResetSummaries clears accumulated results (call at start of menu session).
func ResetSummaries() {
	sessionLayers = nil
}

// RegisterLayerSummary records a completed layer run for the final manifest.
func RegisterLayerSummary(name string, passed, failed int, rows []DTypeRow) {
	sessionLayers = append(sessionLayers, LayerSummary{
		Name:        name,
		Passed:      passed,
		Failed:      failed,
		Rows:        rows,
		LayerPassed: failed == 0,
	})
}

// PrintDTypeResultsTable prints the per-dtype table after all training for one layer.
func PrintDTypeResultsTable(layerName string, rows []DTypeRow) {
	fmt.Printf("\n╔══════════════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║  %s — results (all %d numerical types)                              ║\n", layerName, len(rows))
	fmt.Printf("╚══════════════════════════════════════════════════════════════════════╝\n\n")

	fmt.Printf("| %-10s | %-10s | %-10s | %-12s | %-12s | %-6s | %-6s | %-6s | %-7s | %-7s |\n",
		"DType", "Loss[0]", "Loss[N]", "Before", "After", "B-OK", "A-OK", "Learn", "Native", "OK")
	fmt.Println("|------------|------------|------------|--------------|--------------|--------|--------|--------|---------|---------|")

	for _, r := range rows {
		if r.Err != "" {
			fmt.Printf("| %-10s | %-10s | %-10s | %-12s | %-12s | %-6s | %-6s | %-6s | %-7s | %-7s |\n",
				r.DType, "ERR", "ERR", r.Err, "", "", "", "", "", markOK(false))
			continue
		}
		fmt.Printf("| %-10s | %-10.4e | %-10.4e | %-12s | %-12s | %-6s | %-6s | %-6s | %-7s | %-7s |\n",
			r.DType, r.LossInit, r.LossFinal,
			r.BeforeBucket, r.AfterBucket,
			markOK(r.BeforeOK), markOK(r.AfterOK), markOK(r.Learned),
			markOK(r.NativeOK), markOK(r.OverallOK))
	}

	passed, failed := 0, 0
	for _, r := range rows {
		if r.OverallOK {
			passed++
		} else {
			failed++
		}
	}
	fmt.Printf("\n► %s: %d passed, %d failed (of %d dtypes)\n", layerName, passed, failed, len(rows))
}

// PrintGlobalManifest prints the master pass/fail table for the whole session.
func PrintGlobalManifest() {
	if len(sessionLayers) == 0 {
		return
	}

	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  FIVE-LAYER SESSION MANIFEST — all layers × 21 numerical types        ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Printf("| %-12s | %-8s | %-8s | %-8s | %-8s |\n", "Layer", "Passed", "Failed", "Total", "OK")
	fmt.Println("|--------------|----------|----------|----------|----------|")

	totalPass, totalFail, layersPass, layersFail := 0, 0, 0, 0
	for _, ls := range sessionLayers {
		total := ls.Passed + ls.Failed
		totalPass += ls.Passed
		totalFail += ls.Failed
		ok := markOK(ls.LayerPassed)
		if ls.LayerPassed {
			layersPass++
		} else {
			layersFail++
		}
		fmt.Printf("| %-12s | %-8d | %-8d | %-8d | %-8s |\n",
			ls.Name, ls.Passed, ls.Failed, total, ok)
	}

	fmt.Println("|--------------|----------|----------|----------|----------|")
	fmt.Printf("| %-12s | %-8d | %-8d | %-8d | %-8s |\n",
		"TOTAL", totalPass, totalFail, totalPass+totalFail, markOK(totalFail == 0))
	fmt.Printf("\n► Layers: %d passed, %d failed (of %d layer types)\n", layersPass, layersFail, len(sessionLayers))
	fmt.Printf("► Dtype checks: %d passed, %d failed (of %d)\n", totalPass, totalFail, totalPass+totalFail)
}
