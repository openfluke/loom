package adaptationsuite

import (
	"fmt"
	"math"
	"sort"
	"time"
)

// PrintLayerSummary prints a comparison table for one layer × dtype matrix.
func PrintLayerSummary(layerName, dtype string, results []*Result) {
	fmt.Printf("\n  ┌─ %s · %s — adaptation paths (score = thr×avail×acc/10k) ─────────\n", layerName, dtype)
	fmt.Printf("  │ %-22s │ %7s │ %8s │ %6s │ %6s │ %6s │ %6s │ %6s\n",
		"Path", "Score", "Thr/s", "Avail%", "AvgAcc", "Ph1", "Ph2", "Ph3")
	fmt.Println("  ├────────────────────────┼─────────┼──────────┼────────┼────────┼────────┼────────┼────────")
	for _, r := range sortResults(results) {
		if r.Err != "" {
			fmt.Printf("  │ %-22s │ ERR: %s\n", r.Path.Label(), truncate(r.Err, 40))
			continue
		}
		fmt.Printf("  │ %-22s │ %7.0f │ %8.0f │ %5.1f%% │ %5.1f%% │ %5.1f%% │ %5.1f%% │ %5.1f%%\n",
			r.Path.Label(), r.Score, r.Throughput, r.Availability, r.AvgAccuracy,
			r.Phase1Avg, r.Phase2Avg, r.Phase3Avg)
	}
	fmt.Println("  └────────────────────────┴─────────┴──────────┴────────┴────────┴────────┴────────┴────────")
}

// PrintDtypeWinners prints best path per metric for one layer.
func PrintDtypeWinners(layerName string, byDtype map[string][]*Result) {
	fmt.Printf("\n  ┌─ %s — best path per dtype (by Score) ─────────────────────────────\n", layerName)
	fmt.Printf("  │ %-10s │ %-28s │ %7s │ %5s │ %5s │ %5s\n", "DType", "Winner", "Score", "Ph1", "Ph2", "Ph3")
	fmt.Println("  ├──────────┼──────────────────────────────┼─────────┼───────┼───────┼───────┤")
	dtypes := sortedKeys(byDtype)
	for _, dt := range dtypes {
		best := bestByScore(byDtype[dt])
		if best == nil || best.Err != "" {
			fmt.Printf("  │ %-10s │ %-28s │\n", dt, "ERR")
			continue
		}
		fmt.Printf("  │ %-10s │ %-28s │ %7.0f │ %5.1f │ %5.1f │ %5.1f\n",
			dt, best.Path.Label(), best.Score, best.Phase1Avg, best.Phase2Avg, best.Phase3Avg)
	}
	fmt.Println("  └──────────┴──────────────────────────────┴─────────┴───────┴───────┴───────┘")
}

// PrintParadigmSimdTable compares QAT vs Nat and SC vs SIMD aggregates.
func PrintParadigmSimdTable(layerName string, results []*Result) {
	fmt.Printf("\n  ┌─ %s — paradigm × SIMD mean score ─────────────────────────────────\n", layerName)
	fmt.Printf("  │ %-10s │ %10s %10s │ %10s %10s │ delta Nat-QAT │ delta SIMD-SC\n", "DType", "QAT/SC", "QAT/SIMD", "Nat/SC", "Nat/SIMD")
	fmt.Println("  ├──────────┼──────────────────────┼──────────────────────┼──────────────┼──────────────")

	byDtype := groupByDtype(results)
	for _, dt := range sortedKeys(byDtype) {
		rs := byDtype[dt]
		qsc := meanScore(find(rs, ParadigmQAT, false))
		qsi := meanScore(find(rs, ParadigmQAT, true))
		nsc := meanScore(find(rs, ParadigmNative, false))
		nsi := meanScore(find(rs, ParadigmNative, true))
		fmt.Printf("  │ %-10s │ %10.0f %10.0f │ %10.0f %10.0f │ %+12.0f │ %+12.0f\n",
			dt, qsc, qsi, nsc, nsi, nsc-qsc, nsi-qsc)
	}
	fmt.Println("  └──────────┴──────────────────────┴──────────────────────┴──────────────┴──────────────")
}

// PrintModeRanking ranks update modes averaged across all paths for a layer.
func PrintModeRanking(layerName string, results []*Result) {
	fmt.Printf("\n  ┌─ %s — update mode ranking (mean score, all dtypes/paths) ───────\n", layerName)
	type modeScore struct {
		mode  UpdateMode
		score float64
		n     int
	}
	totals := make(map[UpdateMode]modeScore)
	for _, r := range results {
		if r.Err != "" {
			continue
		}
		ms := totals[r.Path.Mode]
		ms.mode = r.Path.Mode
		ms.score += r.Score
		ms.n++
		totals[r.Path.Mode] = ms
	}
	var ranked []modeScore
	for _, ms := range totals {
		if ms.n > 0 {
			ms.score /= float64(ms.n)
		}
		ranked = append(ranked, ms)
	}
	sort.Slice(ranked, func(i, j int) bool { return ranked[i].score > ranked[j].score })
	for i, ms := range ranked {
		fmt.Printf("  │ #%d %-16s  mean score %.0f  (%d runs)\n", i+1, ms.mode.String(), ms.score, ms.n)
	}
	fmt.Println("  └──────────────────────────────────────────────────────────────────────")
}

// PrintSessionManifest prints the global session summary.
func PrintSessionManifest(layers []LayerSummary) {
	if len(layers) == 0 {
		return
	}
	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  [17] Mid-stream adaptation — session manifest                        ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	totalRuns, totalErr := 0, 0
	for _, ls := range layers {
		ok := ls.Errors == 0
		tag := "PASS"
		if !ok {
			tag = "FAIL"
		}
		fmt.Printf("  %-12s %4s  dtypes %3d  paths×dtypes %4d  errors %3d  %s\n",
			ls.Name, ls.Grid, ls.DTypes, ls.TotalRuns, ls.Errors, tag)
		totalRuns += ls.TotalRuns
		totalErr += ls.Errors
	}
	fmt.Printf("\n  Total runs: %d · errors: %d\n", totalRuns, totalErr)
}

// LayerSummary aggregates one layer's adaptation session.
type LayerSummary struct {
	Name      string
	Grid      string
	DTypes    int
	TotalRuns int
	Errors    int
	Duration  time.Duration
}

func groupByDtype(results []*Result) map[string][]*Result {
	m := make(map[string][]*Result)
	for _, r := range results {
		m[r.DType] = append(m[r.DType], r)
	}
	return m
}

func find(results []*Result, par Paradigm, simd bool) []*Result {
	var out []*Result
	for _, r := range results {
		if r.Path.Paradigm == par && r.Path.SIMD == simd {
			out = append(out, r)
		}
	}
	return out
}

func meanScore(results []*Result) float64 {
	if len(results) == 0 {
		return 0
	}
	sum := 0.0
	n := 0
	for _, r := range results {
		if r.Err != "" {
			continue
		}
		sum += r.Score
		n++
	}
	if n == 0 {
		return 0
	}
	v := sum / float64(n)
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0
	}
	return v
}

func bestByScore(results []*Result) *Result {
	var best *Result
	for _, r := range results {
		if r.Err != "" {
			continue
		}
		if best == nil || r.Score > best.Score {
			best = r
		}
	}
	return best
}

func sortResults(results []*Result) []*Result {
	out := append([]*Result(nil), results...)
	sort.Slice(out, func(i, j int) bool {
		if out[i].Path.Paradigm != out[j].Path.Paradigm {
			return out[i].Path.Paradigm < out[j].Path.Paradigm
		}
		if out[i].Path.SIMD != out[j].Path.SIMD {
			return !out[i].Path.SIMD
		}
		return out[i].Path.Mode < out[j].Path.Mode
	})
	return out
}

func sortedKeys(m map[string][]*Result) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-3] + "..."
}
