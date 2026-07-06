package contextsuite

import (
	"fmt"
	"time"
)

type suiteSummary struct {
	StartedAt  time.Time `json:"started_at"`
	FinishedAt time.Time `json:"finished_at"`
	Total      int       `json:"total"`
	Pass       int       `json:"pass"`
	RecallMiss int       `json:"recall_miss"`
	Overflow   int       `json:"overflow_ok"`
	OverMax    int       `json:"over_max"`
	Empty      int       `json:"empty"`
	LoadFail   int       `json:"load_fail"`
	Skipped    int       `json:"skipped"`
	Cells      []CellResult `json:"cells"`
}

var globalSummary suiteSummary

func ResetSummary() {
	globalSummary = suiteSummary{StartedAt: time.Now()}
}

func recordCell(c CellResult) {
	globalSummary.Cells = append(globalSummary.Cells, c)
	globalSummary.Total++
	switch c.Status {
	case "PASS":
		globalSummary.Pass++
	case "RECALL_MISS", "OVERFLOW_MISS":
		globalSummary.RecallMiss++
	case "OVERFLOW_OK":
		globalSummary.Overflow++
	case "OVER_MAX":
		globalSummary.OverMax++
	case "EMPTY":
		globalSummary.Empty++
	case "LOAD_FAIL", "SKIP":
		if c.Status == "LOAD_FAIL" {
			globalSummary.LoadFail++
		} else {
			globalSummary.Skipped++
		}
	}
}

func PrintSummary() {
	if globalSummary.FinishedAt.IsZero() {
		globalSummary.FinishedAt = time.Now()
	}
	s := globalSummary
	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  Context suite summary — long context / multi-prompt (.entity)       ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Printf("  Total cells:     %d\n", s.Total)
	fmt.Printf("  ✅ PASS:          %d\n", s.Pass)
	fmt.Printf("  🟡 OVERFLOW_OK:   %d  (expected overflow / context-bound behaviour)\n", s.Overflow)
	fmt.Printf("  🟠 RECALL_MISS:   %d  (needle not found — may indicate context loss)\n", s.RecallMiss)
	fmt.Printf("  🔴 OVER_MAX:      %d  (prompt exceeded MaxSeqLen=512)\n", s.OverMax)
	fmt.Printf("  ⚪ EMPTY:         %d\n", s.Empty)
	fmt.Printf("  ❌ LOAD_FAIL:     %d\n", s.LoadFail)
	fmt.Printf("  ⏭  SKIPPED:       %d\n", s.Skipped)
	fmt.Printf("  Duration:        %v\n", s.FinishedAt.Sub(s.StartedAt).Round(time.Millisecond))

	if s.RecallMiss > 0 || s.OverMax > 0 {
		fmt.Println("\n  Context window notes:")
		fmt.Println("  • MaxSeqLen=512 caps KV cache — older turns drop out of attention.")
		fmt.Println("  • overflow_probe and multi_turn_recall are designed to surface this.")
		fmt.Println("  • Compare saved outputs under lucy_testing_output/context_suite/outputs/")
	}
}
