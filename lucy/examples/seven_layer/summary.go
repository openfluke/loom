package sevenlayer

import "fmt"

func markOK(ok bool) string {
	if ok {
		return "PASS"
	}
	return "FAIL"
}

// DTypeRow is one numerical-type result for the seven-layer CPU suite.
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

	FwdSCMC  float64
	BwdSCMC  float64
	FwdGoAsm float64
	DetOK    bool

	AsmCapable bool
	AsmUsed    bool
	AsmOK      bool

	TrainSCDur string
	TrainMCDur string
	TrainSCSps float64
	TrainMCSps float64

	FwdSCDur string
	FwdMCDur string
	BwdSCDur string
	BwdMCDur string

	MemHeap      string
	MemSys       string
	MemHeapTrain string
	WeightBytes  string
	Checkpoint        string
	EntityCheckpoint  string
	EntityBeforeOK    bool
	EntityAfterOK     bool
	EntityNativeOK    bool
	ReloadFwdDiff float64
	ReloadLossDelta float64
	TrainedLoss   float64
	ReloadedLoss  float64
}

// LayerSummary aggregates one layer-type run (21 dtypes).
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

func PrintDTypeResultsTable(layerName string, rows []DTypeRow) {
	fmt.Printf("\n╔══════════════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║  %s — correctness (all %d numerical types)                          ║\n", layerName, len(rows))
	fmt.Printf("╚══════════════════════════════════════════════════════════════════════╝\n\n")

	fmt.Printf("| %-10s | %-10s | %-10s | %-12s | %-12s | %-6s | %-6s | %-6s | %-7s | %-7s | %-6s | %-6s |\n",
		"DType", "Loss[0]", "Loss[N]", "Before", "After", "B-OK", "A-OK", "Learn", "Native", "OK", "Det", "ASM")
	fmt.Println("|------------|------------|------------|--------------|--------------|--------|--------|--------|---------|---------|--------|--------|")

	for _, r := range rows {
		if r.Err != "" {
			fmt.Printf("| %-10s | %-10s | %-10s | %-12s | %-12s | %-6s | %-6s | %-6s | %-7s | %-7s | %-6s | %-6s |\n",
				r.DType, "ERR", "ERR", r.Err, "", "", "", "", "", markOK(false), "", "")
			continue
		}
		asmCol := "N/A"
		if r.AsmCapable {
			asmCol = markOK(r.AsmOK)
		} else if r.AsmUsed {
			asmCol = "skip"
		}
		fmt.Printf("| %-10s | %-10.4e | %-10.4e | %-12s | %-12s | %-6s | %-6s | %-6s | %-7s | %-7s | %-6s | %-6s |\n",
			r.DType, r.LossInit, r.LossFinal,
			r.BeforeBucket, r.AfterBucket,
			markOK(r.BeforeOK), markOK(r.AfterOK), markOK(r.Learned),
			markOK(r.NativeOK), markOK(r.OverallOK), markOK(r.DetOK), asmCol)
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

func PrintTimingTable(layerName string, rows []DTypeRow) {
	fmt.Printf("\n╔══════════════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║  %s — CPU SC vs MC training timing                                    ║\n", layerName)
	fmt.Printf("╚══════════════════════════════════════════════════════════════════════╝\n\n")

	fmt.Printf("| %-10s | %-12s | %-10s | %-12s | %-10s |\n",
		"DType", "SC time", "SC samp/s", "MC time", "MC samp/s")
	fmt.Println("|------------|--------------|------------|--------------|------------|")

	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		fmt.Printf("| %-10s | %-12s | %-10.0f | %-12s | %-10.0f |\n",
			r.DType, r.TrainSCDur, r.TrainSCSps, r.TrainMCDur, r.TrainMCSps)
	}
}

func PrintForwardBackwardTimingTable(layerName string, rows []DTypeRow) {
	fmt.Printf("\n╔══════════════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║  %s — forward / backward timing (avg of %d passes, CPU Go)              ║\n", layerName, activeBenchIters)
	fmt.Printf("╚══════════════════════════════════════════════════════════════════════╝\n\n")

	fmt.Printf("| %-10s | %-10s | %-10s | %-10s | %-10s |\n",
		"DType", "Fwd SC", "Fwd MC", "Bwd SC", "Bwd MC")
	fmt.Println("|------------|------------|------------|------------|------------|")

	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		fmt.Printf("| %-10s | %-10s | %-10s | %-10s | %-10s |\n",
			r.DType, r.FwdSCDur, r.FwdMCDur, r.BwdSCDur, r.BwdMCDur)
	}
}

func PrintMemoryTable(layerName string, rows []DTypeRow) {
	fmt.Printf("\n╔══════════════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║  %s — memory & weight footprint (Go runtime + network)               ║\n", layerName)
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

func PrintTrainedReloadTable(layerName string, rows []DTypeRow) {
	fmt.Printf("\n╔══════════════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║  %s — trained checkpoint save/reload (JSON + .entity, after MC train)  ║\n", layerName)
	fmt.Printf("╚══════════════════════════════════════════════════════════════════════╝\n\n")
	fmt.Println("  Verifies: serialize trained net → deserialize → forward/loss/native match in-memory model.")

	fmt.Printf("| %-10s | %-10s | %-10s | %-10s | %-10s | %-8s | %-8s | %-8s | %-8s |\n",
		"DType", "Loss train", "Loss reload", "|Δloss|", "|Δfwd|", "JSON", "Native", "ENTITY", "E-Native")
	fmt.Println("|------------|------------|------------|------------|------------|--------|--------|--------|--------|")

	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		fmt.Printf("| %-10s | %-10.4e | %-10.4e | %-10.2e | %-10.2e | %-8s | %-8s | %-8s | %-8s |\n",
			r.DType, r.TrainedLoss, r.ReloadedLoss, r.ReloadLossDelta, r.ReloadFwdDiff,
			markOK(r.AfterOK), markOK(r.NativeOK), markOK(r.EntityAfterOK), markOK(r.EntityNativeOK))
	}
}

func PrintDeterminismTable(layerName string, rows []DTypeRow) {
	fmt.Printf("\n╔══════════════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║  %s — CPU determinism (SC↔MC, Go↔ASM on Dense)                       ║\n", layerName)
	fmt.Printf("╚══════════════════════════════════════════════════════════════════════╝\n\n")

	fmt.Printf("| %-10s | %-10s | %-10s | %-10s | %-8s |\n",
		"DType", "Fwd SC↔MC", "Bwd SC↔MC", "Go↔ASM", "Det OK")
	fmt.Println("|------------|------------|------------|------------|----------|")

	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		fmt.Printf("| %-10s | %-10.2e | %-10.2e | %-10.2e | %-8s |\n",
			r.DType, r.FwdSCMC, r.BwdSCMC, r.FwdGoAsm, markOK(r.DetOK))
	}
}

func PrintGlobalManifest() {
	if len(sessionLayers) == 0 {
		return
	}
	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  SEVEN-LAYER SESSION MANIFEST — CPU SC/MC × 21 numerical types       ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Println()
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
	fmt.Printf("\n► Layers: %d passed, %d failed (of %d layer types)\n", layersPass, layersFail, len(sessionLayers))
	fmt.Printf("► Dtype checks: %d passed, %d failed (of %d)\n", totalPass, totalFail, totalPass+totalFail)
}
