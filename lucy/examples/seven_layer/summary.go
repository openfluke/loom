package sevenlayer

import "fmt"

func markOK(ok bool) string {
	if ok {
		return "PASS"
	}
	return "FAIL"
}

func formatParityDiff(d float64) string {
	if d == 0 {
		return "0"
	}
	return fmt.Sprintf("%.2e", d)
}

// DTypeRow is one numerical-type result for the seven-layer CPU suite.
type DTypeRow struct {
	DType        string
	LossInit     float64
	LossFinal    float64
	LossSC       float64
	LossSimd     float64
	BeforeBucket string
	AfterBucket  string
	BeforeOK     bool
	AfterOK      bool
	NativeOK     bool
	Learned      bool
	OverallOK    bool
	Err          string

	FwdSCMC    float64
	BwdSCMC    float64
	FwdSCMCOK  bool
	BwdSCMCOK  bool
	DetOK      bool

	TrainSCDur   string
	TrainMCDur   string
	TrainSimdDur string
	TrainSCSps   float64
	TrainMCSps   float64
	TrainSimdSps float64
	TrainSimdOK  bool

	FwdSCDur string
	FwdMCDur string
	BwdSCDur string
	BwdMCDur string

	FwdSimdDur   string
	FwdSimdPct   string
	FwdTiledSimd float64
	SimdOK       bool

	BwdSimdDur   string
	BwdSimdPct   string
	BwdTiledSimd float64
	BwdSimdOK    bool

	MemHeap           string
	MemSys            string
	MemHeapTrain      string
	WeightBytes       string
	Checkpoint        string
	EntityCheckpoint  string
	EntityBeforeOK    bool
	EntityAfterOK     bool
	EntityNativeOK    bool
	ReloadFwdDiff     float64
	ReloadLossDelta   float64
	TrainedLoss       float64
	ReloadedLoss      float64
}

type LayerSummary struct {
	Name        string
	Passed      int
	Failed      int
	Rows        []DTypeRow
	LayerPassed bool
}

var sessionLayers []LayerSummary

func ResetSummaries() { sessionLayers = nil }

func RegisterLayerSummary(name string, passed, failed int, rows []DTypeRow) {
	sessionLayers = append(sessionLayers, LayerSummary{
		Name: name, Passed: passed, Failed: failed, Rows: rows, LayerPassed: failed == 0,
	})
}

func PrintDTypeResultsTable(layerName string, rows []DTypeRow, simdLayer bool) {
	fmt.Printf("\n╔══════════════════════════════════════════════════════════════════════╗\n")
	if simdLayer {
		fmt.Printf("║  %s — correctness + SC/MC/SIMD (all %d numerical types)              ║\n", layerName, len(rows))
	} else {
		fmt.Printf("║  %s — correctness (all %d numerical types)                          ║\n", layerName, len(rows))
	}
	fmt.Printf("╚══════════════════════════════════════════════════════════════════════╝\n\n")

	if simdLayer {
		fmt.Printf("| %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-6s | %-6s | %-6s | %-6s | %-6s | %-6s | %-6s |\n",
			"DType", "Loss[0]", "Loss SC", "Loss MC", "Loss SIMD", "Bwd SC↔SIMD", "B-OK", "A-OK", "Learn", "TrnSIMD", "F-SIMD", "B-SIMD", "OK")
		fmt.Println("|------------|------------|------------|------------|------------|------------|--------|--------|--------|--------|--------|--------|--------|")
	} else {
		fmt.Printf("| %-10s | %-10s | %-10s | %-12s | %-12s | %-6s | %-6s | %-6s | %-7s | %-7s | %-6s |\n",
			"DType", "Loss[0]", "Loss[N]", "Before", "After", "B-OK", "A-OK", "Learn", "Native", "OK", "Det")
		fmt.Println("|------------|------------|------------|--------------|--------------|--------|--------|--------|---------|---------|--------|")
	}

	for _, r := range rows {
		if r.Err != "" {
			fmt.Printf("| %-10s | ERR %-6s |\n", r.DType, r.Err)
			continue
		}
		if simdLayer {
			fmt.Printf("| %-10s | %-10.4e | %-10.4e | %-10.4e | %-10.4e | %-10s | %-6s | %-6s | %-6s | %-6s | %-6s | %-6s | %-6s |\n",
				r.DType, r.LossInit, r.LossSC, r.LossFinal, r.LossSimd,
				formatParityDiff(r.BwdTiledSimd),
				markOK(r.BeforeOK), markOK(r.AfterOK), markOK(r.Learned),
				markOK(r.TrainSimdOK), markOK(r.SimdOK), markOK(r.BwdSimdOK), markOK(r.OverallOK))
		} else {
			fmt.Printf("| %-10s | %-10.4e | %-10.4e | %-12s | %-12s | %-6s | %-6s | %-6s | %-7s | %-7s | %-6s |\n",
				r.DType, r.LossInit, r.LossFinal, r.BeforeBucket, r.AfterBucket,
				markOK(r.BeforeOK), markOK(r.AfterOK), markOK(r.Learned),
				markOK(r.NativeOK), markOK(r.OverallOK), markOK(r.DetOK))
		}
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

func PrintTimingTable(layerName string, rows []DTypeRow, simdLayer bool) {
	fmt.Printf("\n╔══════════════════════════════════════════════════════════════════════╗\n")
	if simdLayer {
		fmt.Printf("║  %s — training SC · MC · SIMD (%d epochs)                              ║\n", layerName, trainEpochs)
	} else {
		fmt.Printf("║  %s — training SC · MC (%d epochs)                                     ║\n", layerName, trainEpochs)
	}
	fmt.Printf("╚══════════════════════════════════════════════════════════════════════╝\n\n")

	if simdLayer {
		fmt.Printf("| %-10s | %-11s | %-10s | %-11s | %-10s | %-11s | %-10s |\n",
			"DType", "SC time", "SC s/s", "MC time", "MC s/s", "SIMD time", "SIMD s/s")
		fmt.Println("|------------|-------------|----------|-------------|----------|-------------|----------|")
	} else {
		fmt.Printf("| %-10s | %-12s | %-10s | %-12s | %-10s |\n",
			"DType", "SC time", "SC samp/s", "MC time", "MC samp/s")
		fmt.Println("|------------|--------------|------------|--------------|------------|")
	}
	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		if simdLayer {
			fmt.Printf("| %-10s | %-11s | %-10.0f | %-11s | %-10.0f | %-11s | %-10.0f |\n",
				r.DType, r.TrainSCDur, r.TrainSCSps, r.TrainMCDur, r.TrainMCSps, r.TrainSimdDur, r.TrainSimdSps)
		} else {
			fmt.Printf("| %-10s | %-12s | %-10.0f | %-12s | %-10.0f |\n",
				r.DType, r.TrainSCDur, r.TrainSCSps, r.TrainMCDur, r.TrainMCSps)
		}
	}
}

func PrintForwardBackwardTimingTable(layerName string, rows []DTypeRow, simdLayer bool) {
	fmt.Printf("\n╔══════════════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║  %s — fwd/bwd timing SC·MC·SIMD (avg %d passes)                         ║\n", layerName, activeBenchIters)
	fmt.Printf("╚══════════════════════════════════════════════════════════════════════╝\n\n")

	if simdLayer {
		fmt.Printf("| %-10s | %-9s | %-9s | %-9s | %-9s | %-9s | %-9s | %-10s | %-10s |\n",
			"DType", "Fwd SC", "Fwd MC", "Fwd SIMD", "Bwd SC", "Bwd MC", "Bwd SIMD", "Fwd SIMD×", "Bwd SIMD×")
		fmt.Println("|------------|-----------|-----------|-----------|-----------|-----------|-----------|------------|------------|")
	} else {
		fmt.Printf("| %-10s | %-10s | %-10s | %-10s | %-10s |\n",
			"DType", "Fwd SC", "Fwd MC", "Bwd SC", "Bwd MC")
		fmt.Println("|------------|------------|------------|------------|------------|")
	}
	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		if simdLayer {
			fmt.Printf("| %-10s | %-9s | %-9s | %-9s | %-9s | %-9s | %-9s | %-10s | %-10s |\n",
				r.DType, r.FwdSCDur, r.FwdMCDur, r.FwdSimdDur, r.BwdSCDur, r.BwdMCDur, r.BwdSimdDur,
				r.FwdSimdPct, r.BwdSimdPct)
		} else {
			fmt.Printf("| %-10s | %-10s | %-10s | %-10s | %-10s |\n",
				r.DType, r.FwdSCDur, r.FwdMCDur, r.BwdSCDur, r.BwdMCDur)
		}
	}
}

func PrintSCMCSimdParityTable(layerName string, rows []DTypeRow, simdLayer bool) {
	fmt.Printf("\n╔══════════════════════════════════════════════════════════════════════╗\n")
	if simdLayer {
		fmt.Printf("║  %s — parity SC ↔ MC ↔ SIMD (all numerical types)                     ║\n", layerName)
		fmt.Println("║  Fwd: dot_tile .s | Bwd SIMD: saxpy/dot .s (all seven layer types)           ║")
	} else {
		fmt.Printf("║  %s — parity SC ↔ MC (all numerical types)                            ║\n", layerName)
	}
	fmt.Printf("╚══════════════════════════════════════════════════════════════════════╝\n\n")

	if simdLayer {
		fmt.Printf("| %-10s | %-10s | %-10s | %-10s | %-10s | %-7s | %-7s | %-7s | %-7s | %-6s |\n",
			"DType", "Fwd SC↔MC", "Fwd SC↔SIMD", "Bwd SC↔MC", "Bwd SC↔SIMD",
			"F SC/MC", "F SIMD", "B SC/MC", "B SIMD", "Det")
		fmt.Println("|------------|------------|------------|------------|------------|---------|---------|---------|---------|--------|")
	} else {
		fmt.Printf("| %-10s | %-10s | %-10s | %-8s |\n", "DType", "Fwd SC↔MC", "Bwd SC↔MC", "Det OK")
		fmt.Println("|------------|------------|------------|----------|")
	}
	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		if simdLayer {
			fmt.Printf("| %-10s | %-10s | %-10s | %-10s | %-10s | %-7s | %-7s | %-7s | %-7s | %-6s |\n",
				r.DType,
				formatParityDiff(r.FwdSCMC), formatParityDiff(r.FwdTiledSimd),
				formatParityDiff(r.BwdSCMC), formatParityDiff(r.BwdTiledSimd),
				markOK(r.FwdSCMCOK), markOK(r.SimdOK), markOK(r.BwdSCMCOK), markOK(r.BwdSimdOK), markOK(r.DetOK))
		} else {
			fmt.Printf("| %-10s | %-10s | %-10s | %-8s |\n",
				r.DType, formatParityDiff(r.FwdSCMC), formatParityDiff(r.BwdSCMC), markOK(r.DetOK))
		}
	}
}

func PrintDeterminismTable(layerName string, rows []DTypeRow, simdLayer bool) {
	PrintSCMCSimdParityTable(layerName, rows, simdLayer)
}

func PrintMemoryTable(layerName string, rows []DTypeRow) {
	fmt.Printf("\n╔══════════════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║  %s — memory & weight footprint                                       ║\n", layerName)
	fmt.Printf("╚══════════════════════════════════════════════════════════════════════╝\n\n")
	fmt.Printf("| %-10s | %-10s | %-10s | %-10s | %-10s | %-12s | %-12s |\n",
		"DType", "Heap", "Sys", "Heap+train", "Weights", "JSON ckpt", ".entity ckpt")
	fmt.Println("|------------|------------|------------|------------|------------|--------------|--------------|")
	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		fmt.Printf("| %-10s | %-10s | %-10s | %-10s | %-10s | %-12s | %-12s |\n",
			r.DType, r.MemHeap, r.MemSys, r.MemHeapTrain, r.WeightBytes, r.Checkpoint, r.EntityCheckpoint)
	}
}

func PrintTrainedReloadTable(layerName string, rows []DTypeRow, simdLayer bool) {
	fmt.Printf("\n╔══════════════════════════════════════════════════════════════════════╗\n")
	if simdLayer {
		fmt.Printf("║  %s — checkpoint save/reload (MC train + SIMD train loss)              ║\n", layerName)
	} else {
		fmt.Printf("║  %s — checkpoint save/reload (after MC train)                          ║\n", layerName)
	}
	fmt.Printf("╚══════════════════════════════════════════════════════════════════════╝\n\n")
	if simdLayer {
		fmt.Printf("| %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-8s | %-8s | %-8s | %-8s |\n",
			"DType", "Loss MC", "Loss reload", "Loss SIMD", "|Δloss|", "|Δfwd|", "JSON", "Native", "ENTITY", "E-Native")
		fmt.Println("|------------|------------|------------|------------|------------|------------|--------|--------|--------|--------|")
	} else {
		fmt.Printf("| %-10s | %-10s | %-10s | %-10s | %-10s | %-8s | %-8s | %-8s | %-8s |\n",
			"DType", "Loss train", "Loss reload", "|Δloss|", "|Δfwd|", "JSON", "Native", "ENTITY", "E-Native")
		fmt.Println("|------------|------------|------------|------------|------------|--------|--------|--------|--------|")
	}
	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		if simdLayer {
			fmt.Printf("| %-10s | %-10.4e | %-10.4e | %-10.4e | %-10.2e | %-10.2e | %-8s | %-8s | %-8s | %-8s |\n",
				r.DType, r.TrainedLoss, r.ReloadedLoss, r.LossSimd, r.ReloadLossDelta, r.ReloadFwdDiff,
				markOK(r.AfterOK), markOK(r.NativeOK), markOK(r.EntityAfterOK), markOK(r.EntityNativeOK))
		} else {
			fmt.Printf("| %-10s | %-10.4e | %-10.4e | %-10.2e | %-10.2e | %-8s | %-8s | %-8s | %-8s |\n",
				r.DType, r.TrainedLoss, r.ReloadedLoss, r.ReloadLossDelta, r.ReloadFwdDiff,
				markOK(r.AfterOK), markOK(r.NativeOK), markOK(r.EntityAfterOK), markOK(r.EntityNativeOK))
		}
	}
}

func PrintGlobalManifest() {
	if len(sessionLayers) == 0 {
		return
	}
	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  SEVEN-LAYER MANIFEST — CPU SC/MC/SIMD × 21 numerical types           ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Printf("| %-12s | %-8s | %-8s | %-8s | %-8s |\n", "Layer", "Passed", "Failed", "Total", "OK")
	fmt.Println("|--------------|----------|----------|----------|----------|")
	totalPass, totalFail, layersPass, layersFail := 0, 0, 0, 0
	for _, ls := range sessionLayers {
		total := ls.Passed + ls.Failed
		totalPass += ls.Passed
		totalFail += ls.Failed
		if ls.LayerPassed {
			layersPass++
		} else {
			layersFail++
		}
		fmt.Printf("| %-12s | %-8d | %-8d | %-8d | %-8s |\n",
			ls.Name, ls.Passed, ls.Failed, total, markOK(ls.LayerPassed))
	}
	fmt.Println("|--------------|----------|----------|----------|----------|")
	fmt.Printf("| %-12s | %-8d | %-8d | %-8d | %-8s |\n",
		"TOTAL", totalPass, totalFail, totalPass+totalFail, markOK(totalFail == 0))
	fmt.Printf("\n► Layers: %d passed, %d failed | Dtype checks: %d passed, %d failed\n",
		layersPass, layersFail, totalPass, totalFail)
}
