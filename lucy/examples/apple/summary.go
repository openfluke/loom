package apple

import "fmt"

func speedupRatio(baselineMs, accelMs float64) string {
	if accelMs <= 0 {
		return "—"
	}
	r := baselineMs / accelMs
	if r >= 10 {
		return fmt.Sprintf("%.0f×", r)
	}
	if r >= 1 {
		return fmt.Sprintf("%.1f×", r)
	}
	return fmt.Sprintf("%.2f×", r)
}

var sessionDispatch []dispatchRow

func resetDispatchSession() { sessionDispatch = nil }

func registerDispatchRows(rows []dispatchRow) {
	sessionDispatch = append(sessionDispatch, rows...)
}

func printAccelTimingTable(sizeName string, rows []dispatchRow, hasGPU bool) {
	fmt.Printf("\n╔══════════════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║  %s — forward timing (median ms after one SyncToAccel)              ║\n", sizeName)
	fmt.Printf("╚══════════════════════════════════════════════════════════════════════╝\n\n")

	if hasGPU {
		fmt.Printf("| %-14s | %-6s | %-10s | %-10s | %-10s | %-8s | %-8s | %-10s | %-10s |\n",
			"Layer", "DType", "Loom CPU", "Apple CPU", "Metal GPU", "Spd CPU", "Spd GPU", "Compile C", "Compile G")
		fmt.Println("|--------------|--------|------------|------------|------------|----------|----------|------------|------------|")
	} else {
		fmt.Printf("| %-14s | %-6s | %-10s | %-10s | %-8s | %-10s |\n",
			"Layer", "DType", "Loom CPU", "Apple CPU", "Spd CPU", "Compile C")
		fmt.Println("|--------------|--------|------------|------------|----------|------------|")
	}

	for _, r := range rows {
		if r.Note != "" && r.Note != "OK" {
			if hasGPU {
				fmt.Printf("| %-14s | %-6s | %-10s | %-10s | %-10s | %-8s | %-8s | %-10s | %-10s |\n",
					r.Layer, r.DType, "ERR", "ERR", "ERR", "—", "—", "—", "—")
			} else {
				fmt.Printf("| %-14s | %-6s | %-10s | %-10s | %-8s | %-10s |\n",
					r.Layer, r.DType, "ERR", "ERR", "—", "—")
			}
			continue
		}
		compC := fmt.Sprintf("%.2f", r.CompileCPU)
		if r.CompileCPU <= 0 {
			compC = "—"
		}
		if hasGPU {
			compG := fmt.Sprintf("%.2f", r.CompileGPU)
			if r.CompileGPU <= 0 {
				compG = "—"
			}
			fmt.Printf("| %-14s | %-6s | %-10.3f | %-10.3f | %-10.3f | %-8s | %-8s | %-10s | %-10s |\n",
				r.Layer, r.DType, r.LoomMs, r.AppleCPUMs, r.MetalGPUMs,
				speedupRatio(r.LoomMs, r.AppleCPUMs), speedupRatio(r.LoomMs, r.MetalGPUMs),
				compC, compG)
		} else {
			fmt.Printf("| %-14s | %-6s | %-10.3f | %-10.3f | %-8s | %-10s |\n",
				r.Layer, r.DType, r.LoomMs, r.AppleCPUMs,
				speedupRatio(r.LoomMs, r.AppleCPUMs), compC)
		}
	}
	fmt.Println()
	fmt.Println("  Spd = Loom÷Apple (higher = Apple faster). Compile = one-time SyncToAccel ms.")
}

func printAccelDeterminismTable(sizeName string, rows []dispatchRow, hasGPU bool) {
	fmt.Printf("\n╔══════════════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║  %s — drift spectrum (seven-style: Loom↔Apple + Metal repeat)        ║\n", sizeName)
	fmt.Printf("╚══════════════════════════════════════════════════════════════════════╝\n\n")

	if hasGPU {
		fmt.Printf("| %-14s | %-6s | %-10s | %-12s | %-10s | %-12s | %-10s | %-12s | %-10s | %-12s |\n",
			"Layer", "DType", "Loom↔ACPU", "Par CPU", "Loom↔GPU", "Par GPU", "ACPU 1↔2", "Id CPU", "GPU 1↔2", "Id GPU")
		fmt.Println("|--------------|--------|------------|------------|------------|------------|------------|------------|------------|------------|")
	} else {
		fmt.Printf("| %-14s | %-6s | %-10s | %-12s | %-10s | %-12s |\n",
			"Layer", "DType", "Loom↔ACPU", "Par CPU", "ACPU 1↔2", "Id CPU")
		fmt.Println("|--------------|--------|------------|------------|------------|------------|")
	}

	for _, r := range rows {
		if r.Note != "" && r.Note != "OK" {
			continue
		}
		if hasGPU {
			fmt.Printf("| %-14s | %-6s | %-10.2e | %-12s | %-10.2e | %-12s | %-10.2e | %-12s | %-10.2e | %-12s |\n",
				r.Layer, r.DType,
				r.CPUDrift, r.ParityCPUSpec.String(),
				r.GPUDrift, r.ParityGPUSpec.String(),
				r.InferCPUDrift, r.InferCPUSpec.String(),
				r.InferGPUDrift, r.InferGPUSpec.String())
		} else {
			fmt.Printf("| %-14s | %-6s | %-10.2e | %-12s | %-10.2e | %-12s |\n",
				r.Layer, r.DType,
				r.CPUDrift, r.ParityCPUSpec.String(),
				r.InferCPUDrift, r.InferCPUSpec.String())
		}
	}
	fmt.Println()
	fmt.Println("  Par = Loom vs Apple parity bucket. Id = plugin infer repeat-forward bucket.")
	fmt.Println("  💎 EXACT · ✅ INDUS · 🟨 LOWBIT · 🟠 DRIFT · 🟤 H-DRIFT · ❌ BROKE · 💀 FATAL")
}

func printDispatchManifest(hasGPU bool) {
	if len(sessionDispatch) == 0 {
		return
	}
	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  APPLE DISPATCH MANIFEST — Loom CPU vs Apple CPU/GPU                 ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	type counts struct{ pass, fail int }
	tally := func(pred func(dispatchRow) bool) counts {
		var c counts
		for _, r := range sessionDispatch {
			if r.Note != "" && r.Note != "OK" {
				c.fail++
				continue
			}
			if pred(r) {
				c.pass++
			} else {
				c.fail++
			}
		}
		return c
	}
	fasterCPU := tally(func(r dispatchRow) bool { return r.AppleCPUMs > 0 && r.LoomMs/r.AppleCPUMs >= 1.0 })
	fasterGPU := counts{}
	if hasGPU {
		fasterGPU = tally(func(r dispatchRow) bool { return r.MetalGPUMs > 0 && r.LoomMs/r.MetalGPUMs >= 1.0 })
	}

	total := len(sessionDispatch)
	fmt.Printf("| %-30s | %-8s | %-8s | %-8s |\n", "Check", "Pass", "Fail", "Total")
	fmt.Println("|--------------------------------|----------|----------|----------|")
	printManifestRow("Apple faster than Loom (CPU)", fasterCPU.pass, fasterCPU.fail, total)
	if hasGPU {
		printManifestRow("Apple faster than Loom (GPU)", fasterGPU.pass, fasterGPU.fail, total)
	}
	fmt.Println()
	printSpectrumHistogram("Loom↔Apple parity (CPU)", func(r dispatchRow) driftSpectrum { return r.ParityCPUSpec })
	if hasGPU {
		printSpectrumHistogram("Loom↔Apple parity (GPU)", func(r dispatchRow) driftSpectrum { return r.ParityGPUSpec })
	}
	printSpectrumHistogram("Apple infer repeat (CPU)", func(r dispatchRow) driftSpectrum { return r.InferCPUSpec })
	if hasGPU {
		printSpectrumHistogram("Apple infer repeat (GPU)", func(r dispatchRow) driftSpectrum { return r.InferGPUSpec })
	}
	fmt.Println()
	fmt.Printf("► %d layer×dtype cells exercised via DispatchLayer (SyncToAccel once per device).\n", total)
}

func printSpectrumHistogram(label string, pick func(dispatchRow) driftSpectrum) {
	bins := make([]int, int(specFatal)+1)
	ok := 0
	for _, r := range sessionDispatch {
		if r.Note != "" && r.Note != "OK" {
			continue
		}
		s := pick(r)
		if int(s) >= 0 && int(s) <= int(specFatal) {
			bins[int(s)]++
		}
		if s <= specIndustry {
			ok++
		}
	}
	total := 0
	for _, n := range bins {
		total += n
	}
	fmt.Printf("  %s (%d/%d ≤ INDUS):\n", label, ok, total)
	for s := driftSpectrum(0); s <= specFatal; s++ {
		if bins[int(s)] == 0 {
			continue
		}
		fmt.Printf("    %-12s %d\n", s.String(), bins[int(s)])
	}
}

func printManifestRow(label string, pass, fail, total int) {
	fmt.Printf("| %-30s | %-8d | %-8d | %-8d |\n", label, pass, fail, total)
}
