package tfsimdbench

import (
	"bufio"
	"fmt"
	"strconv"
	"strings"
)

// RunMenu is Lucy [11]. Benchmarks CPU transformer decode throughput across
// scalar/SIMD × single-/multi-core for the quantized .entity checkpoints.
func RunMenu(reader *bufio.Reader) {
	printBanner()

	fmt.Println("  [0] Run full matrix — all models × cpu_sc/cpu_mc/cpu_simd_sc/cpu_simd_mc [default]")
	for i, m := range TargetModels {
		status := "missing"
		if entityExists(m.RepoID) {
			status = "ready"
		}
		fmt.Printf("  [%d] Run %-14s (%s)\n", i+1, m.ShortName, status)
	}
	fmt.Println("  [l] List target models and .entity status")
	fmt.Print("Choice [0]: ")

	line, _ := reader.ReadString('\n')
	line = strings.TrimSpace(line)
	if line == "" {
		line = "0"
	}

	switch line {
	case "0":
		cleanup := BeginSession()
		defer cleanup()
		RunFullMatrix()
	case "l", "L":
		listModelStatus()
	default:
		idx, err := strconv.Atoi(line)
		if err != nil || idx < 1 || idx > len(TargetModels) {
			fmt.Println("Invalid selection.")
			return
		}
		cleanup := BeginSession()
		defer cleanup()
		RunSingleModel(TargetModels[idx-1])
	}
}

func printBanner() {
	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  [11] Transformer SIMD bench — .entity CPU decode throughput          ║")
	fmt.Println("║  Models: bitnet · qwen3-0.6b · smol2-135m                            ║")
	fmt.Println("║  Matrix: cpu_sc · cpu_mc · cpu_simd_sc · cpu_simd_mc                 ║")
	fmt.Printf("║  Log: %s/%s          ║\n", OutputDir, LogFile)
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Println("  Fixed prompt, greedy decode. Measures Plan 9 SIMD (MHA/SwiGLU) vs multi-core")
	fmt.Println("  tiling. Reports decode/prefill tok/s, speedup vs cpu_sc, and output parity.")
	fmt.Println()
}

func listModelStatus() {
	fmt.Println("Target models and .entity status:")
	for i, m := range TargetModels {
		status := "missing"
		if entityExists(m.RepoID) {
			status = "ready"
		}
		fmt.Printf("  [%d] %-14s  %-7s  %s\n", i+1, m.ShortName, status, entityPath(m.RepoID))
	}
}
