package sevenlayer

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	adaptationsuite "github.com/openfluke/loom/lucy/examples/adaptation_suite"
	"github.com/openfluke/loom/poly"
)

const AdaptationLogFile = "adaptation_suite.txt"

var adaptSessionLayers []adaptationsuite.LayerSummary
var adaptActiveGrids []GridSpec

var adaptMenuEntries = []crossPathLayerEntry{
	{"Dense", poly.LayerDense, buildDenseSuite},
	{"SwiGLU", poly.LayerSwiGLU, buildSwiGLUNativeSuite},
	{"MHA", poly.LayerMultiHeadAttention, buildMHANativeSuite},
	{"CNN1", poly.LayerCNN1, buildCNN1NativeSuite},
	{"CNN2", poly.LayerCNN2, buildCNN2NativeSuite},
	{"CNN3", poly.LayerCNN3, buildCNN3NativeSuite},
	{"RNN", poly.LayerRNN, buildRNNNativeSuite},
	{"LSTM", poly.LayerLSTM, buildLSTMNativeSuite},
	{"Embedding", poly.LayerEmbedding, buildEmbeddingNativeSuite},
	{"Residual", poly.LayerResidual, buildResidualNativeSuite},
}

func BeginAdaptationSession() func() {
	adaptSessionLayers = nil
	_ = os.MkdirAll(OutputDir, 0o755)
	logPath := filepath.Join(OutputDir, AdaptationLogFile)
	_ = os.Remove(logPath)
	logFile, err := os.Create(logPath)
	if err != nil {
		fmt.Printf("Warning: could not create %s: %v\n", logPath, err)
		return func() {}
	}
	r, w, err := os.Pipe()
	if err != nil {
		_ = logFile.Close()
		return func() {}
	}
	orig := os.Stdout
	os.Stdout = w
	done := make(chan struct{})
	go func() {
		mw := io.MultiWriter(orig, logFile)
		buf := make([]byte, 4096)
		for {
			n, e := r.Read(buf)
			if n > 0 {
				_, _ = mw.Write(buf[:n])
			}
			if e != nil {
				break
			}
		}
		close(done)
	}()
	return func() {
		_ = w.Close()
		<-done
		_ = r.Close()
		_ = logFile.Close()
		os.Stdout = orig
		adaptationsuite.PrintSessionManifest(adaptSessionLayers)
		fmt.Printf("\n📄 Adaptation log: %s\n", logPath)
	}
}

// RunAdaptationMenu is Lucy [17]: mid-stream adaptation across layers, dtypes, QAT/Nat, SIMD, update modes.
func RunAdaptationMenu(reader *bufio.Reader) {
	defer BeginAdaptationSession()()

	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  [17] Mid-stream adaptation suite — full path matrix                  ║")
	fmt.Println("║  Log: lucy_testing_output/adaptation_suite.txt                        ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Println("  Phase flip A → B → A' · 6 update modes · QAT/Nat · SC/SIMD")
	fmt.Println("  Recreates [2] dense benchmark, extended to every layer × dtype")
	fmt.Println()
	fmt.Println("  Steps per path:")
	fmt.Println("    [1] 150 — quick smoke")
	fmt.Println("    [2] 450 — default")
	fmt.Println("    [3] 900 — longer signal")
	fmt.Print("  Steps [2]: ")
	stepLine, _ := reader.ReadString('\n')
	steps := adaptStepsFromChoice(strings.TrimSpace(stepLine))

	fmt.Println()
	fmt.Println("  Grid:")
	fmt.Println("    [1] 1³ — 7-layer stack (default)")
	fmt.Println("    [2] 2³ — 56-layer stack")
	fmt.Print("  Grid [1]: ")
	gridLine, _ := reader.ReadString('\n')
	adaptActiveGrids = adaptGridsFromChoice(strings.TrimSpace(gridLine))

	fmt.Println()
	fmt.Println("  Layer type:")
	fmt.Println("    [0] Run all layer types")
	for i, e := range adaptMenuEntries {
		fmt.Printf("    [%d] %s\n", i+1, e.name)
	}
	fmt.Print("  Choice [1]: ")

	line, _ := reader.ReadString('\n')
	line = strings.TrimSpace(line)
	if line == "" {
		line = "1"
	}

	cfg := adaptationsuite.DefaultConfig()
	cfg.Steps = steps
	cfg.Windows = steps / 50
	if cfg.Windows < 3 {
		cfg.Windows = 3
	}

	if line == "0" {
		for _, e := range adaptMenuEntries {
			runAdaptationEntry(e, cfg)
		}
		return
	}

	idx, err := strconv.Atoi(line)
	if err != nil || idx < 1 || idx > len(adaptMenuEntries) {
		fmt.Println("Invalid selection.")
		return
	}
	runAdaptationEntry(adaptMenuEntries[idx-1], cfg)
}

func adaptStepsFromChoice(choice string) int {
	switch choice {
	case "1":
		return 150
	case "3":
		return 900
	default:
		return 450
	}
}

func adaptGridsFromChoice(choice string) []GridSpec {
	switch choice {
	case "2":
		return []GridSpec{StandardGrids[1]}
	default:
		return []GridSpec{StandardGrids[0]}
	}
}

func runAdaptationEntry(e crossPathLayerEntry, cfg adaptationsuite.Config) {
	for _, g := range adaptActiveGrids {
		fmt.Printf("\n▶ Adaptation %s · %s (%d-layer stack) · %d steps/path …\n",
			e.name, g, g.StackLayers(), cfg.Steps)
		runAdaptationLayerSuite(e.build(g), e.primary, e.name, g, cfg)
	}
}

func runAdaptationLayerSuite(s LayerSuite, primary poly.LayerType, layerName string, g GridSpec, cfg adaptationsuite.Config) {
	scenario := adaptationsuite.RegressionScenario(s.MakeInput)
	scenario.Primary = primary

	var allResults []*adaptationsuite.Result
	errors := 0
	layerStart := time.Now()

	for _, tc := range allDTypes {
		fmt.Printf("  · %-10s ", tc.name)

		net, err := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
		if err != nil {
			fmt.Println("BUILD ERR")
			errors++
			continue
		}
		applyDType(net, tc)
		prepareTrainingNet(net, tc.dtype)
		finalizeTrainingNet(net, tc)

		wire, err := poly.SerializeNetwork(net)
		if err != nil {
			fmt.Println("SERIALIZE ERR")
			errors++
			continue
		}

		nativeOK := poly.IsLayerNativeExactDType(tc.dtype)
		paths := adaptationsuite.AllPaths(nativeOK)
		var dtypeResults []*adaptationsuite.Result
		dtypeErrors := 0

		for _, path := range paths {
			r := adaptationsuite.Run(wire, path, scenario, cfg)
			if r == nil {
				r = &adaptationsuite.Result{Path: path, Err: "nil result"}
				errors++
				dtypeErrors++
			}
			r.DType = tc.name
			r.Layer = layerName
			dtypeResults = append(dtypeResults, r)
			allResults = append(allResults, r)
			if r.Err != "" {
				dtypeErrors++
				errors++
			}
		}

		if dtypeErrors > 0 {
			fmt.Printf("FAIL %d/%d paths\n", len(paths)-dtypeErrors, len(paths))
		} else {
			best := bestAdaptScore(dtypeResults)
			if best == nil {
				fmt.Printf("PASS %2d paths  (no scored winner)\n", len(paths))
			} else {
				fmt.Printf("PASS %2d paths  best=%s score=%.0f\n", len(paths), best.Path.Label(), best.Score)
			}
		}

		adaptationsuite.PrintLayerSummary(layerName, tc.name, dtypeResults)
	}

	adaptationsuite.PrintDtypeWinners(layerName, groupAdaptByDtype(allResults))
	adaptationsuite.PrintParadigmSimdTable(layerName, allResults)
	adaptationsuite.PrintModeRanking(layerName, allResults)

	fmt.Printf("\n  %s adaptation · %s: %d runs · %d errors · %s\n",
		layerName, g, len(allResults), errors, time.Since(layerStart).Round(time.Millisecond))

	adaptSessionLayers = append(adaptSessionLayers, adaptationsuite.LayerSummary{
		Name:      layerName,
		Grid:      g.String(),
		DTypes:    len(allDTypes),
		TotalRuns: len(allResults),
		Errors:    errors,
		Duration:  time.Since(layerStart),
	})
}

func bestAdaptScore(results []*adaptationsuite.Result) *adaptationsuite.Result {
	var best *adaptationsuite.Result
	for _, r := range results {
		if r.Err != "" {
			continue
		}
		if best == nil || r.Score > best.Score {
			best = r
		}
	}
	if best == nil && len(results) > 0 {
		return results[0]
	}
	return best
}

func groupAdaptByDtype(results []*adaptationsuite.Result) map[string][]*adaptationsuite.Result {
	m := make(map[string][]*adaptationsuite.Result)
	for _, r := range results {
		m[r.DType] = append(m[r.DType], r)
	}
	return m
}
