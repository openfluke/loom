package sevenlayer

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/openfluke/loom/poly"
)

const TweenNativeLogFile = "tween_native_layers.txt"

var tweenNativeActiveGrids []GridSpec
var tweenSessionLayers []tweenLayerSummary
var tweenSessionTally testTally

type tweenLayerSummary struct {
	Name        string
	Grid        GridSpec
	Passed      int
	Failed      int
	Rows        []tweenRow
	TestsTotal  int
	TestsPassed int
}

type tweenRow struct {
	DType string
	Err   string

	NatFwdDur     string
	NatFwdSimdDur string
	NatBwdDur     string
	NatBwdSimdDur string
	natFwd        time.Duration
	natFwdSimd    time.Duration
	natBwd        time.Duration
	natBwdSimd    time.Duration

	NatFwdOK       bool
	NatFwdSimdOK   bool
	NatFwdParity   float64
	NatBwdOK       bool
	NatBwdSimdOK   bool
	NatBwdParity   float64
	NativePathOK   bool

	LossInit          float64
	LossFinalNative   float64
	LossFinalNatSimd  float64
	TrainNativeDur    string
	TrainNatSimdDur   string
	trainNative       time.Duration
	trainNatSimd      time.Duration
	TrainNativeOK     bool
	TrainNatSimdOK    bool

	fwdPair, bwdPair, trainPair pairCmp
	fwdWinner, bwdWinner, trainWinner string
	fwdWinRatio, bwdWinRatio, trainWinRatio string

	TestsTotal  int
	TestsPassed int
	OverallOK   bool
}

var tweenMenuEntries = []crossPathLayerEntry{
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

func ResetTweenSummaries() {
	tweenSessionLayers = nil
	tweenSessionTally = *newTestTally()
}

func BeginTweenNativeSession() func() {
	ResetTweenSummaries()
	_ = os.MkdirAll(OutputDir, 0o755)
	logPath := filepath.Join(OutputDir, TweenNativeLogFile)
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
		printTweenGlobalManifest()
		fmt.Printf("\n📄 Tween native log: %s\n", logPath)
	}
}

// RunTweenNativeMenu is Lucy [16]: native tween SC vs native-SIMD on all layers × dtypes.
func RunTweenNativeMenu(reader *bufio.Reader) {
	defer BeginTweenNativeSession()()

	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  [16] Tween native suite — native SC vs native-SIMD (target prop)     ║")
	fmt.Println("║  Log: lucy_testing_output/tween_native_layers.txt                     ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Println("  tween.go native MAC · chain-rule tween fwd/bwd/train")
	fmt.Println("  Compare: native forward/backward/train vs native-SIMD")
	fmt.Println()
	fmt.Println("  Grid:")
	fmt.Println("    [1] 1³ — 7-layer stack (fast)")
	fmt.Println("    [2] 2³ — 56-layer stack (default)")
	fmt.Println("    [3] 3³ — 189-layer stack (best perf signal)")
	fmt.Println("    [4] All grids (1³ + 2³ + 3³)")
	fmt.Print("  Grid [2]: ")
	gridLine, _ := reader.ReadString('\n')
	tweenNativeActiveGrids = tweenGridsFromChoice(strings.TrimSpace(gridLine))
	for _, g := range tweenNativeActiveGrids {
		fmt.Printf("  → %s: %d cells × %d layers/cell = %d stack · %d tween epochs\n",
			g, g.Cells(), sevenLayersPerCell, g.StackLayers(), trainEpochsForGrid(g))
	}
	fmt.Println()
	fmt.Println("  Layer type:")
	fmt.Println("    [0] Run all layer types")
	for i, e := range tweenMenuEntries {
		fmt.Printf("    [%d] %s\n", i+1, e.name)
	}
	fmt.Print("  Choice [1]: ")

	line, _ := reader.ReadString('\n')
	line = strings.TrimSpace(line)
	if line == "" {
		line = "1"
	}

	if line == "0" {
		for _, e := range tweenMenuEntries {
			runTweenEntry(e)
		}
		return
	}

	idx, err := strconv.Atoi(line)
	if err != nil || idx < 1 || idx > len(tweenMenuEntries) {
		fmt.Println("Invalid selection.")
		return
	}
	runTweenEntry(tweenMenuEntries[idx-1])
}

func tweenGridsFromChoice(choice string) []GridSpec {
	if choice == "" {
		choice = "2"
	}
	switch choice {
	case "1":
		return []GridSpec{StandardGrids[0]}
	case "3":
		return []GridSpec{StandardGrids[2]}
	case "4":
		return append([]GridSpec(nil), StandardGrids...)
	default:
		return []GridSpec{StandardGrids[1]}
	}
}

func runTweenEntry(e crossPathLayerEntry) {
	for _, g := range tweenNativeActiveGrids {
		fmt.Printf("\n▶ Tween native %s · %s (%d-layer stack) …\n", e.name, g, g.StackLayers())
		if e.primary == poly.LayerCNN3 && g.Cells() > 1 {
			d, h, w := cnn3Spatial(g)
			fmt.Printf("  (CNN3 spatial %d×%d×%d on %s)\n", d, h, w, g)
		}
		runTweenLayerSuite(e.build(g), e.primary)
	}
}

func defaultTweenConfig() *poly.TweenConfig {
	cfg := poly.DefaultTweenConfig()
	cfg.UseChainRule = true
	return cfg
}

func tweenLossFromState(state *poly.TweenState[float32], target *poly.Tensor[float32]) float64 {
	outIdx := state.TotalLayers
	out := state.ForwardActs[outIdx]
	if out == nil {
		return math.NaN()
	}
	return poly.CalculateLoss(out, target, "mse")
}

func captureTweenForward(net *poly.VolumetricNetwork, input *poly.Tensor[float32], simd bool) (out []float32, dur time.Duration) {
	configureNativeNetForTween(net)
	setSimdForward(net, simd)
	cfg := defaultTweenConfig()
	state := poly.NewTweenState[float32](net, cfg)
	for i := 0; i < 3; i++ {
		resetNetwork(net)
		_ = poly.TweenForward(net, state, input)
	}
	var total time.Duration
	var last []float32
	for i := 0; i < activeBenchIters; i++ {
		resetNetwork(net)
		t0 := time.Now()
		post := poly.TweenForward(net, state, input)
		total += time.Since(t0)
		if post != nil {
			last = append([]float32(nil), post.Data...)
		}
	}
	if simd {
		setSimdForward(net, false)
	}
	return last, total / time.Duration(activeBenchIters)
}

func captureTweenBackward(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32], simd bool) (gradIn []float32, dur time.Duration) {
	configureNativeNetForTween(net)
	setSimdForward(net, simd)
	cfg := defaultTweenConfig()
	state := poly.NewTweenState[float32](net, cfg)
	for i := 0; i < 3; i++ {
		resetNetwork(net)
		_ = poly.TweenForward(net, state, input)
		poly.TweenBackward(net, state, target)
	}
	var total time.Duration
	var last []float32
	for i := 0; i < activeBenchIters; i++ {
		resetNetwork(net)
		_ = poly.TweenForward(net, state, input)
		t0 := time.Now()
		poly.TweenBackward(net, state, target)
		total += time.Since(t0)
		if state.Gradients[0] != nil {
			last = append([]float32(nil), state.Gradients[0].Data...)
		}
	}
	if simd {
		setSimdForward(net, false)
	}
	return last, total / time.Duration(activeBenchIters)
}

func configureNativeNetForTween(net *poly.VolumetricNetwork) {
	wireLayerTree(net)
	net.UseExactDType = true
}

func trainTweenNative(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32], tc dtypeCase, simd bool, epochs int) (lossInit, lossFinal float64, dur time.Duration, err error) {
	applyDType(net, tc)
	configureNativeNetForTween(net)
	prepareTrainingNet(net, tc.dtype)
	finalizeTrainingNet(net, tc)
	setSimdForward(net, simd)

	cfg := defaultTweenConfig()
	state := poly.NewTweenState[float32](net, cfg)
	lr := trainingLearningRate(tc.dtype)

	resetNetwork(net)
	_ = poly.TweenForward(net, state, input)
	lossInit = tweenLossFromState(state, target)

	t0 := time.Now()
	for e := 0; e < epochs; e++ {
		resetNetwork(net)
		_ = poly.TweenForward(net, state, input)
		poly.TweenBackward(net, state, target)
		state.CalculateLinkBudgets()
		poly.ApplyTweenGaps(net, state, lr)
	}
	dur = time.Since(t0)

	resetNetwork(net)
	_ = poly.TweenForward(net, state, input)
	lossFinal = tweenLossFromState(state, target)
	if simd {
		setSimdForward(net, false)
	}
	return lossInit, lossFinal, dur, nil
}

func runTweenLayerSuite(s LayerSuite, primary poly.LayerType) {
	activeBenchIters = benchItersForGrid(s.Grid)
	epochs := trainEpochsForGrid(s.Grid)
	requiresLearn := layerRequiresLearn(primary)

	fmt.Printf("\n  ┌─ %s Tween native · %s · %d-layer stack ─────────────────────────\n",
		s.Name, s.Grid, s.Grid.StackLayers())
	fmt.Printf("  │ Native SC vs Nat-SIMD · chain-rule tween · %d epochs\n", epochs)

	var rows []tweenRow
	layerTally := newTestTally()
	passed, failed := 0, 0

	for _, tc := range allDTypes {
		fmt.Printf("  · %-10s ", tc.name)
		row := tweenRow{DType: tc.name}
		tally := newTestTally()

		if !poly.IsLayerNativeExactDType(tc.dtype) {
			row.Err = "NO-NATIVE"
			rows = append(rows, row)
			failed++
			fmt.Println("SKIP (no native path)")
			continue
		}

		net, err := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
		if err != nil {
			row.Err = "BUILD"
			rows = append(rows, row)
			failed++
			fmt.Println("BUILD ERR")
			continue
		}
		applyDType(net, tc)
		prepareTrainingNet(net, tc.dtype)
		finalizeTrainingNet(net, tc)
		input := s.MakeInput()
		target := s.MakeTarget(net, input)

		configureNativeNet(net, tc)
		row.NativePathOK = verifyNativePath(net, primary, tc)
		tally.record("native.path", row.NativePathOK)

		outNat, durNat := captureTweenForward(net, input, false)
		row.natFwd = durNat
		row.NatFwdDur = formatDur(durNat)
		row.NatFwdOK = len(outNat) > 0 && tensorFinite(outNat)
		tally.record("tween.fwd.native", row.NatFwdOK)

		outNatSimd, durNatSimd := captureTweenForward(net, input, true)
		row.natFwdSimd = durNatSimd
		row.NatFwdSimdDur = formatDur(durNatSimd)
		row.NatFwdSimdOK = len(outNatSimd) > 0 && tensorFinite(outNatSimd)
		row.NatFwdParity = maxAbsDiff(outNat, outNatSimd)
		tally.record("tween.fwd.native.simd", row.NatFwdSimdOK)

		gradNat, durBwdNat := captureTweenBackward(net, input, target, false)
		row.natBwd = durBwdNat
		row.NatBwdDur = formatDur(durBwdNat)
		row.NatBwdOK = len(gradNat) > 0 && tensorFinite(gradNat)
		tally.record("tween.bwd.native", row.NatBwdOK)

		gradNatSimd, durBwdNatSimd := captureTweenBackward(net, input, target, true)
		row.natBwdSimd = durBwdNatSimd
		row.NatBwdSimdDur = formatDur(durBwdNatSimd)
		row.NatBwdSimdOK = len(gradNatSimd) > 0 && tensorFinite(gradNatSimd)
		row.NatBwdParity = maxAbsDiff(gradNat, gradNatSimd)
		tally.record("tween.bwd.native.simd", row.NatBwdSimdOK)

		netNat, _ := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
		lossInitNat, lossFinalNat, durTrainNat, err := trainTweenNative(netNat, input, target, tc, false, epochs)
		if err != nil {
			row.Err = "TRAIN-NATIVE"
			rows = append(rows, row)
			failed++
			fmt.Println("TRAIN NATIVE ERR")
			continue
		}
		row.LossInit = lossInitNat
		row.LossFinalNative = lossFinalNat
		row.trainNative = durTrainNat
		row.TrainNativeDur = formatDur(durTrainNat)
		row.TrainNativeOK = lossFiniteOK(lossInitNat, lossFinalNat, requiresLearn) &&
			(!requiresLearn || trainingOK(lossInitNat, lossFinalNat, tc.dtype))
		tally.record("tween.train.native", row.TrainNativeOK)

		netNatSimd, _ := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
		lossInitSimd, lossFinalNatSimd, durTrainNatSimd, err := trainTweenNative(netNatSimd, input, target, tc, true, epochs)
		if err != nil {
			row.Err = "TRAIN-NAT-SIMD"
			rows = append(rows, row)
			failed++
			fmt.Println("TRAIN NAT-SIMD ERR")
			continue
		}
		if row.LossInit == 0 || math.IsNaN(row.LossInit) {
			row.LossInit = lossInitSimd
		}
		row.LossFinalNatSimd = lossFinalNatSimd
		row.trainNatSimd = durTrainNatSimd
		row.TrainNatSimdDur = formatDur(durTrainNatSimd)
		row.TrainNatSimdOK = lossFiniteOK(lossInitSimd, lossFinalNatSimd, requiresLearn) &&
			(!requiresLearn || trainingOK(lossInitSimd, lossFinalNatSimd, tc.dtype))
		tally.record("tween.train.native.simd", row.TrainNatSimdOK)

		computeTweenComparisons(&row)

		row.TestsTotal = tally.total
		row.TestsPassed = tally.passed
		row.OverallOK = row.TestsPassed == row.TestsTotal && row.Err == ""
		rows = append(rows, row)
		layerTally.merge(tally)

		if row.OverallOK {
			passed++
			fmt.Printf("PASS  %d/%d  fwd %s %s  bwd %s %s  train %s %s\n",
				row.TestsPassed, row.TestsTotal,
				row.fwdWinner, row.fwdWinRatio,
				row.bwdWinner, row.bwdWinRatio,
				row.trainWinner, row.trainWinRatio)
		} else {
			failed++
			fmt.Printf("FAIL  %d/%d tests  err=%s\n", row.TestsPassed, row.TestsTotal, row.Err)
		}
	}

	printTweenTimingTable(s.Name, rows)
	printTweenComparisonTable(s.Name, rows)
	printTweenTrainTable(s.Name, rows, epochs)
	printTweenTestTally(s.Name, layerTally)
	fmt.Printf("\n  %s Tween native · %s: %d passed · %d failed (of %d dtypes) · tests %d/%d\n",
		s.Name, s.Grid, passed, failed, len(rows), layerTally.passed, layerTally.total)

	tweenSessionLayers = append(tweenSessionLayers, tweenLayerSummary{
		Name: s.Name, Grid: s.Grid, Passed: passed, Failed: failed, Rows: rows,
		TestsTotal: layerTally.total, TestsPassed: layerTally.passed,
	})
	tweenSessionTally.merge(layerTally)
}

func computeTweenComparisons(row *tweenRow) {
	row.fwdPair = makePair("Nat-f", row.natFwd, row.natFwdSimd, "NatS-f")
	row.bwdPair = makePair("Nat-b", row.natBwd, row.natBwdSimd, "NatS-b")
	row.trainPair = makePair("Nat", row.trainNative, row.trainNatSimd, "NatS")

	natFwd := namedDur{"Nat-f", row.natFwd}
	natSimdFwd := namedDur{"NatS-f", row.natFwdSimd}
	row.fwdWinner, row.fwdWinRatio, _ = paradigmWinner(natFwd, natSimdFwd)

	natBwd := namedDur{"Nat-b", row.natBwd}
	natSimdBwd := namedDur{"NatS-b", row.natBwdSimd}
	row.bwdWinner, row.bwdWinRatio, _ = paradigmWinner(natBwd, natSimdBwd)

	natTrain := namedDur{"Nat", row.trainNative}
	natSimdTrain := namedDur{"NatS", row.trainNatSimd}
	row.trainWinner, row.trainWinRatio, _ = paradigmWinner(natTrain, natSimdTrain)
}

func printTweenTimingTable(layerName string, rows []tweenRow) {
	fmt.Printf("\n  ┌─ %s Tween — raw timing (fwd / bwd) ───────────────────────────────\n", layerName)
	fmt.Printf("  │ %-10s │ %-10s %-10s │ %-10s %-10s\n",
		"DType", "Nat-f", "NatS-f", "Nat-b", "NatS-b")
	fmt.Println("  ├──────────┼──────────┬──────────┼──────────┬──────────")
	for _, r := range rows {
		if r.Err != "" {
			fmt.Printf("  │ %-10s │ ERR %s\n", r.DType, r.Err)
			continue
		}
		fmt.Printf("  │ %-10s │ %-10s %-10s │ %-10s %-10s\n",
			r.DType, r.NatFwdDur, r.NatFwdSimdDur, r.NatBwdDur, r.NatBwdSimdDur)
	}
	fmt.Println("  └──────────┴──────────┴──────────┴──────────┴──────────")
}

func printTweenComparisonTable(layerName string, rows []tweenRow) {
	fmt.Printf("\n  ┌─ %s Tween — native SC vs native-SIMD ─────────────────────────────\n", layerName)
	fmt.Printf("  │ %-10s │ %-14s %-5s %-7s │ fwd parity │ bwd parity (informational)\n", "DType", "fwd winner", "×", "faster")
	fmt.Println("  ├──────────┼──────────────────────────────┼────────────┼────────────")
	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		fmt.Printf("  │ %-10s │ %-14s %-5s %-7s │ %10.4g │ %10.4g\n",
			r.DType, r.fwdWinner, r.fwdWinRatio, "", r.NatFwdParity, r.NatBwdParity)
	}
	fmt.Println("  └──────────┴──────────────────────────────┴────────────┴────────────")
}

func printTweenTrainTable(layerName string, rows []tweenRow, epochs int) {
	fmt.Printf("\n  ┌─ %s Tween train (%d epochs) — loss init→final ─────────────────────\n", layerName, epochs)
	fmt.Printf("  │ %-10s │ %8s %8s %8s │ %-10s %-10s │ train winner\n",
		"DType", "init", "Nat", "NatS", "Nat dur", "NatS dur")
	fmt.Println("  ├──────────┼──────────┬──────────┼──────────┬──────────┼────────────")
	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		fmt.Printf("  │ %-10s │ %8.4f %8.4f %8.4f │ %-10s %-10s │ %s %s\n",
			r.DType, r.LossInit, r.LossFinalNative, r.LossFinalNatSimd,
			r.TrainNativeDur, r.TrainNatSimdDur, r.trainWinner, r.trainWinRatio)
	}
	fmt.Println("  └──────────┴──────────┴──────────┴──────────┴──────────┴────────────")
}

func printTweenTestTally(layerName string, tally *testTally) {
	fmt.Printf("\n  ┌─ %s Tween test tally ─────────────────────────────────────────────\n", layerName)
	cats := []string{
		"native.path", "tween.fwd.native", "tween.fwd.native.simd",
		"tween.bwd.native", "tween.bwd.native.simd",
		"tween.train.native", "tween.train.native.simd",
	}
	for _, cat := range cats {
		if v, ok := tally.byCat[cat]; ok {
			fmt.Printf("  │ %-30s %3d / %3d\n", cat, v[0], v[1])
		}
	}
	fmt.Printf("  │ TOTAL (gated)                 %3d / %3d\n", tally.passed, tally.total)
	fmt.Println("  └──────────────────────────────────────────────────────────────────────")
}

func printTweenGlobalManifest() {
	if len(tweenSessionLayers) == 0 {
		return
	}
	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  [16] Tween native — session manifest                               ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	sessionPassed, sessionFailed := 0, 0
	for _, ls := range tweenSessionLayers {
		ok := ls.Failed == 0
		tag := "PASS"
		if !ok {
			tag = "FAIL"
		}
		fmt.Printf("  %-12s %4s  dtypes %3d/%3d  tests %4d/%4d  %s\n",
			ls.Name, ls.Grid, ls.Passed, ls.Passed+ls.Failed, ls.TestsPassed, ls.TestsTotal, tag)
		sessionPassed += ls.Passed
		sessionFailed += ls.Failed
	}
	fmt.Printf("\n  Session dtypes: %d passed · %d failed\n", sessionPassed, sessionFailed)
	fmt.Printf("  Session tests:  %d passed · %d failed (of %d checks)\n",
		tweenSessionTally.passed, tweenSessionTally.total-tweenSessionTally.passed, tweenSessionTally.total)
}
