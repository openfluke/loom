package ninelayer

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

func printAccelTimingTable(sizeName string, rows []dispatchRow, hasNPU bool) {
	fmt.Printf("\n╔══════════════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║  %s — forward timing (median ms after one SyncToAccel)              ║\n", sizeName)
	fmt.Printf("╚══════════════════════════════════════════════════════════════════════╝\n\n")

	if hasNPU {
		fmt.Printf("| %-14s | %-6s | %-10s | %-10s | %-10s | %-8s | %-8s | %-10s | %-10s |\n",
			"Layer", "DType", "Loom CPU", "Intel CPU", "Intel NPU", "Spd CPU", "Spd NPU", "Compile C", "Compile N")
		fmt.Println("|--------------|--------|------------|------------|------------|----------|----------|------------|------------|")
	} else {
		fmt.Printf("| %-14s | %-6s | %-10s | %-10s | %-8s | %-10s |\n",
			"Layer", "DType", "Loom CPU", "Intel CPU", "Spd CPU", "Compile C")
		fmt.Println("|--------------|--------|------------|------------|----------|------------|")
	}

	for _, r := range rows {
		if r.Note != "" && r.Note != "OK" {
			if hasNPU {
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
		if hasNPU {
			compN := fmt.Sprintf("%.2f", r.CompileNPU)
			if r.CompileNPU <= 0 {
				compN = "—"
			}
			fmt.Printf("| %-14s | %-6s | %-10.3f | %-10.3f | %-10.3f | %-8s | %-8s | %-10s | %-10s |\n",
				r.Layer, r.DType, r.LoomMs, r.IntelCPUMs, r.IntelNPUMs,
				speedupRatio(r.LoomMs, r.IntelCPUMs), speedupRatio(r.LoomMs, r.IntelNPUMs),
				compC, compN)
		} else {
			fmt.Printf("| %-14s | %-6s | %-10.3f | %-10.3f | %-8s | %-10s |\n",
				r.Layer, r.DType, r.LoomMs, r.IntelCPUMs,
				speedupRatio(r.LoomMs, r.IntelCPUMs), compC)
		}
	}
	fmt.Println()
	fmt.Println("  Spd = Loom÷Intel (higher = Intel faster). Compile = one-time SyncToAccel ms.")
}

func printAccelDeterminismTable(sizeName string, rows []dispatchRow, hasNPU bool) {
	fmt.Printf("\n╔══════════════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║  %s — drift spectrum (seven-style: Loom↔Intel + Intel repeat)         ║\n", sizeName)
	fmt.Printf("╚══════════════════════════════════════════════════════════════════════╝\n\n")

	if hasNPU {
		fmt.Printf("| %-14s | %-6s | %-10s | %-12s | %-10s | %-12s | %-10s | %-12s | %-10s | %-12s |\n",
			"Layer", "DType", "Loom↔ICPU", "Par CPU", "Loom↔INPU", "Par NPU", "ICPU 1↔2", "Id CPU", "INPU 1↔2", "Id NPU")
		fmt.Println("|--------------|--------|------------|------------|------------|------------|------------|------------|------------|------------|")
	} else {
		fmt.Printf("| %-14s | %-6s | %-10s | %-12s | %-10s | %-12s |\n",
			"Layer", "DType", "Loom↔ICPU", "Par CPU", "ICPU 1↔2", "Id CPU")
		fmt.Println("|--------------|--------|------------|------------|------------|------------|")
	}

	for _, r := range rows {
		if r.Note != "" && r.Note != "OK" {
			continue
		}
		if hasNPU {
			fmt.Printf("| %-14s | %-6s | %-10.2e | %-12s | %-10.2e | %-12s | %-10.2e | %-12s | %-10.2e | %-12s |\n",
				r.Layer, r.DType,
				r.CPUDrift, r.ParityCPUSpec.String(),
				r.NPUDrift, r.ParityNPUSpec.String(),
				r.InferCPUDrift, r.InferCPUSpec.String(),
				r.InferNPUDrift, r.InferNPUSpec.String())
		} else {
			fmt.Printf("| %-14s | %-6s | %-10.2e | %-12s | %-10.2e | %-12s |\n",
				r.Layer, r.DType,
				r.CPUDrift, r.ParityCPUSpec.String(),
				r.InferCPUDrift, r.InferCPUSpec.String())
		}
	}
	fmt.Println()
	fmt.Println("  Par = Loom vs Intel parity bucket. Id = Intel infer repeat-forward bucket.")
	fmt.Println("  💎 EXACT · ✅ INDUS · 🟨 LOWBIT · 🟠 DRIFT · 🟤 H-DRIFT · ❌ BROKE · 💀 FATAL")
}

func printDispatchManifest(hasNPU bool) {
	if len(sessionDispatch) == 0 {
		return
	}
	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  NINE-LAYER DISPATCH MANIFEST — Loom CPU vs Intel CPU/NPU            ║")
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
	fasterCPU := tally(func(r dispatchRow) bool { return r.IntelCPUMs > 0 && r.LoomMs/r.IntelCPUMs >= 1.0 })
	fasterNPU := counts{}
	if hasNPU {
		fasterNPU = tally(func(r dispatchRow) bool { return r.IntelNPUMs > 0 && r.LoomMs/r.IntelNPUMs >= 1.0 })
	}

	total := len(sessionDispatch)
	fmt.Printf("| %-28s | %-8s | %-8s | %-8s |\n", "Check", "Pass", "Fail", "Total")
	fmt.Println("|------------------------------|----------|----------|----------|")
	printManifestRow("Intel faster than Loom (CPU)", fasterCPU.pass, fasterCPU.fail, total)
	if hasNPU {
		printManifestRow("Intel faster than Loom (NPU)", fasterNPU.pass, fasterNPU.fail, total)
	}
	fmt.Println()
	printSpectrumHistogram("Loom↔Intel parity (CPU)", func(r dispatchRow) driftSpectrum { return r.ParityCPUSpec })
	if hasNPU {
		printSpectrumHistogram("Loom↔Intel parity (NPU)", func(r dispatchRow) driftSpectrum { return r.ParityNPUSpec })
	}
	printSpectrumHistogram("Intel infer repeat (CPU)", func(r dispatchRow) driftSpectrum { return r.InferCPUSpec })
	if hasNPU {
		printSpectrumHistogram("Intel infer repeat (NPU)", func(r dispatchRow) driftSpectrum { return r.InferNPUSpec })
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
	fmt.Printf("| %-28s | %-8d | %-8d | %-8d |\n", label, pass, fail, total)
}
