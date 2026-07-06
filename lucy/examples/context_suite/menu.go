package contextsuite

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// RunMenu is Lucy [10]. Automated long-context / multi-prompt .entity inference tests.
func RunMenu(reader *bufio.Reader) {
	printBanner()

	if strings.TrimSpace(os.Getenv("LOOM_CONTEXT_SUITE_AUTO")) == "1" {
		_ = os.Unsetenv("LOOM_CONTEXT_SUITE_AUTO")
		runAuto()
		return
	}

	fmt.Println("  [0] Run full matrix — all models × cpu/gpu × sc/mc × all scenarios [default]")
	fmt.Println("  [1] Run scenarios only (cpu_mc, first available model)")
	fmt.Println("  [2] List target models and .entity status")
	fmt.Print("Choice [0]: ")

	line, _ := reader.ReadString('\n')
	line = strings.TrimSpace(line)
	if line == "" {
		line = "0"
	}

	cleanup := BeginSession()
	defer cleanup()

	switch line {
	case "0":
		RunFullMatrix()
	case "1":
		runQuickSmoke()
	case "2":
		listModelStatus()
	default:
		fmt.Println("Invalid selection.")
	}
}

// RunAuto is non-interactive entry (LOOM_CONTEXT_SUITE=1 on lucy startup).
func RunAuto() {
	printBanner()
	cleanup := BeginSession()
	defer cleanup()
	if strings.TrimSpace(os.Getenv("LOOM_CONTEXT_SUITE_SMOKE")) == "1" {
		_ = os.Unsetenv("LOOM_CONTEXT_SUITE_SMOKE")
		runQuickSmoke()
		return
	}
	RunFullMatrix()
}

func runAuto() {
	cleanup := BeginSession()
	defer cleanup()
	RunFullMatrix()
}

func printBanner() {
	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  [10] Context suite — long context / multi-prompt (.entity)          ║")
	fmt.Println("║  Models: bitnet · qwen3-0.6b · smol2-135m                            ║")
	fmt.Println("║  Exec: cpu_sc · cpu_mc · gpu_sc · gpu_mc                             ║")
	fmt.Printf("║  Log: %s/%s                          ║\n", OutputDir, LogFile)
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Println("  Probes MaxSeqLen=512 KV cache limits via long prefill + multi-turn chat.")
	fmt.Println("  Saves every generation to outputs/ for manual review (gitignored).")
	fmt.Println()
}

func listModelStatus() {
	fmt.Println("Target models and .entity status:")
	for i, m := range TargetModels {
		status := "missing"
		if entityExists(m.RepoID) {
			status = "ready"
		}
		fmt.Printf("  [%d] %-14s  %s  %s\n", i+1, m.ShortName, status, entityPath(m.RepoID))
	}
	snap, err := resolveSnapshotDir(TargetModels[0].RepoID)
	if err == nil {
		fmt.Printf("\n  HF cache example: %s\n", snap)
	}
}

func runQuickSmoke() {
	models := availableModels()
	if len(models) == 0 {
		fmt.Println("❌ No .entity checkpoints found.")
		listModelStatus()
		return
	}
	spec := models[0]
	for _, m := range models {
		if m.ShortName == "smol2-135m" {
			spec = m
			break
		}
	}
	prof := ExecProfile{Name: "cpu_mc", UseGPU: false, MultiCore: true}
	scenarios := AllScenarios[:2]
	fmt.Printf("Quick smoke: %s / %s / %d scenarios\n", spec.ShortName, prof.Name, len(scenarios))
	runModelProfile(spec, prof, scenarios)
}

// ParseModelFilter returns model indices from comma-separated input (1-based).
func ParseModelFilter(s string) []int {
	var out []int
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		idx, err := strconv.Atoi(part)
		if err != nil || idx < 1 || idx > len(TargetModels) {
			continue
		}
		out = append(out, idx-1)
	}
	return out
}
