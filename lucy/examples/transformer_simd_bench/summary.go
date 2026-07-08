package tfsimdbench

import "fmt"

func printModelTable(spec ModelSpec, results []BenchResult) {
	fmt.Printf("\n   ┌─ %s decode summary ───────────────────────────────────────┐\n", spec.ShortName)
	fmt.Printf("   %-14s %12s %12s %10s %-9s\n", "profile", "decode t/s", "prefill t/s", "speedup", "parity")
	fmt.Printf("   %-14s %12s %12s %10s %-9s\n", "-------", "----------", "-----------", "-------", "------")
	for _, r := range results {
		if r.Status != "OK" {
			fmt.Printf("   %-14s %12s %12s %10s %-9s (%s)\n", r.Profile, "-", "-", "-", "-", r.Status)
			continue
		}
		fmt.Printf("   %-14s %12.2f %12.2f %9.2fx %-9s\n",
			r.Profile, r.DecodeTokS, r.PrefillTokS, r.SpeedupVsSC, r.ParityVsSC)
	}
	fmt.Println("   └──────────────────────────────────────────────────────────────┘")
}

func printGlobalTable(all []BenchResult) {
	fmt.Println("\n════════════════════ OVERALL: decode tok/s (speedup vs cpu_sc) ════════════════════")
	fmt.Printf("%-14s %-14s %12s %10s %-9s %s\n",
		"model", "profile", "decode t/s", "speedup", "parity", "notes")
	fmt.Println("---------------------------------------------------------------------------------------")
	for _, r := range all {
		if r.Status != "OK" {
			fmt.Printf("%-14s %-14s %12s %10s %-9s %s\n",
				r.Model, r.Profile, "-", "-", "-", r.Status)
			continue
		}
		note := ""
		if r.Simd && r.SimdApplied && r.Model == "bitnet" {
			note = "AVX2 ternary matvec"
		}
		fmt.Printf("%-14s %-14s %12.2f %9.2fx %-9s %s\n",
			r.Model, r.Profile, r.DecodeTokS, r.SpeedupVsSC, r.ParityVsSC, note)
	}
	fmt.Println("---------------------------------------------------------------------------------------")
	fmt.Println("Legend: speedup = decode tok/s ÷ cpu_sc decode tok/s · parity = greedy reply vs cpu_sc")
}
