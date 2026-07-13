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

const CrossPathLogFile = "cross_path_layers.txt"

const crossPathTrainEpochs = 30

type testTally struct {
	total   int
	passed  int
	byCat   map[string][2]int // [passed, total]
}

func newTestTally() *testTally {
	return &testTally{byCat: make(map[string][2]int)}
}

func (t *testTally) record(cat string, ok bool) {
	t.total++
	if ok {
		t.passed++
	}
	v := t.byCat[cat]
	v[1]++
	if ok {
		v[0]++
	}
	t.byCat[cat] = v
}

func (t *testTally) merge(o *testTally) {
	if o == nil {
		return
	}
	t.total += o.total
	t.passed += o.passed
	for k, v := range o.byCat {
		cur := t.byCat[k]
		cur[0] += v[0]
		cur[1] += v[1]
		t.byCat[k] = cur
	}
}

// BeginCrossPathSession tees stdout to lucy_testing_output/cross_path_layers.txt.
func BeginCrossPathSession() func() {
	ResetCrossPathSummaries()
	_ = os.MkdirAll(OutputDir, 0o755)
	logPath := filepath.Join(OutputDir, CrossPathLogFile)
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
		PrintCrossPathGlobalManifest()
		fmt.Printf("\n📄 Cross-path log: %s\n", logPath)
	}
}

type crossPathLayerEntry struct {
	name string
	run  func()
}

var crossPathMenuEntries = []crossPathLayerEntry{
	{"Dense", runCrossPathDense},
	{"SwiGLU", runCrossPathSwiGLU},
	{"MHA", runCrossPathMHA},
	{"CNN1", runCrossPathCNN1},
	{"CNN2", runCrossPathCNN2},
	{"CNN3", runCrossPathCNN3},
	{"RNN", runCrossPathRNN},
	{"LSTM", runCrossPathLSTM},
	{"Embedding", runCrossPathEmbedding},
	{"Residual", runCrossPathResidual},
}

type crossPathRow struct {
	DType string
	Err   string

	// Tiled (layer.go / GetActive FP32 dequant)
	FwdSCDur, FwdMCDur, FwdSimdDur string
	BwdSCDur, BwdMCDur, BwdSimdDur string
	FwdSCMC, BwdSCMC               float64
	FwdTiledSimd, BwdTiledSimd     float64
	FwdSCOK, FwdMCOK, FwdSimdOK    bool
	BwdSCOK, BwdMCOK, BwdSimdOK    bool
	FwdSCMCOK, BwdSCMCOK           bool
	TiledSimdOK, TiledBwdSimdOK    bool

	// Native exact (*_native.go)
	NativeApplicable bool
	NativePathOK     bool
	NatFwdDur        string
	NatBwdDur        string
	NatFwdSimdDur    string
	NatBwdSimdDur    string
	NatFwdOK         bool
	NatBwdOK         bool
	NatFwdSimdOK       bool
	NatBwdSimdFinite   bool
	NatFwdSimdParity   float64
	NatBwdSimdParity   float64
	NatSimdOK          bool
	NatBwdSimdParityOK bool

	// Cross-path drift (informational)
	CrossFwdSCNative float64
	CrossFwdSimdNat  float64

	// Training
	LossInit       float64
	LossFinalSC    float64
	LossFinalMC    float64
	LossFinalSimd  float64
	LossFinalNative float64
	TrainSCDur     string
	TrainMCDur     string
	TrainSimdDur   string
	TrainNativeDur string
	TrainSCOK      bool
	TrainMCOK      bool
	TrainSimdOK    bool
	TrainNativeOK  bool

	TestsTotal  int
	TestsPassed int
	OverallOK   bool
}

type crossPathLayerSummary struct {
	Name        string
	Passed      int
	Failed      int
	Rows        []crossPathRow
	TestsTotal  int
	TestsPassed int
}

var crossPathSessionLayers []crossPathLayerSummary
var crossPathSessionTally testTally

func ResetCrossPathSummaries() {
	crossPathSessionLayers = nil
	crossPathSessionTally = *newTestTally()
}

func configureTiledNet(net *poly.VolumetricNetwork) {
	wireLayerTree(net)
	net.UseExactDType = false
}

func verifyNativePath(net *poly.VolumetricNetwork, primary poly.LayerType, tc dtypeCase) bool {
	nativeLayers := 0
	for i := range net.Layers {
		if poly.LayerUsesNativeExact(&net.Layers[i]) {
			nativeLayers++
		}
	}
	ok := nativeLayers == len(net.Layers)
	if poly.IsDenseTrueNativeDType(tc.dtype) && primary == poly.LayerDense {
		trueNative := 0
		for i := range net.Layers {
			if poly.DenseUsesTrueNative(&net.Layers[i]) {
				trueNative++
			}
		}
		ok = ok && trueNative == len(net.Layers)
	}
	return ok
}

// RunCrossPathMenu is Lucy [15]: SC/MC/SIMD (tiled) vs native / native-SIMD on all layers.
func RunCrossPathMenu(reader *bufio.Reader) {
	defer BeginCrossPathSession()()

	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  [15] Cross-path CPU suite — SC/MC/SIMD vs native vs native-SIMD      ║")
	fmt.Println("║  Log: lucy_testing_output/cross_path_layers.txt                        ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Println("  Tiled: layer.go GetActive FP32 dequant · Native: *_native.go MAC rules")
	fmt.Println("  Grid 1³ · 7 layers/cell · 30 train epochs · all 21 dtypes per layer")
	fmt.Println()
	fmt.Println("  [0] Run all layer types")
	for i, e := range crossPathMenuEntries {
		fmt.Printf("  [%d] %s\n", i+1, e.name)
	}
	fmt.Print("Choice [1]: ")

	line, _ := reader.ReadString('\n')
	line = strings.TrimSpace(line)
	if line == "" {
		line = "1"
	}

	if line == "0" {
		for _, e := range crossPathMenuEntries {
			fmt.Printf("\n▶ Cross-path %s …\n", e.name)
			e.run()
		}
		return
	}

	idx, err := strconv.Atoi(line)
	if err != nil || idx < 1 || idx > len(crossPathMenuEntries) {
		fmt.Println("Invalid selection.")
		return
	}
	crossPathMenuEntries[idx-1].run()
}

func runCrossPathDense()     { runCrossPathLayerSuite(buildDenseSuite(grid1()), poly.LayerDense) }
func runCrossPathSwiGLU()    { runCrossPathLayerSuite(buildSwiGLUNativeSuite(grid1()), poly.LayerSwiGLU) }
func runCrossPathMHA()       { runCrossPathLayerSuite(buildMHANativeSuite(grid1()), poly.LayerMultiHeadAttention) }
func runCrossPathCNN1()      { runCrossPathLayerSuite(buildCNN1NativeSuite(grid1()), poly.LayerCNN1) }
func runCrossPathCNN2()      { runCrossPathLayerSuite(buildCNN2NativeSuite(grid1()), poly.LayerCNN2) }
func runCrossPathCNN3()      { runCrossPathLayerSuite(buildCNN3NativeSuite(grid1()), poly.LayerCNN3) }
func runCrossPathRNN()       { runCrossPathLayerSuite(buildRNNNativeSuite(grid1()), poly.LayerRNN) }
func runCrossPathLSTM()      { runCrossPathLayerSuite(buildLSTMNativeSuite(grid1()), poly.LayerLSTM) }
func runCrossPathEmbedding() { runCrossPathLayerSuite(buildEmbeddingNativeSuite(grid1()), poly.LayerEmbedding) }
func runCrossPathResidual()  { runCrossPathLayerSuite(buildResidualNativeSuite(grid1()), poly.LayerResidual) }

func runCrossPathLayerSuite(s LayerSuite, primary poly.LayerType) {
	activeBenchIters = benchItersForGrid(s.Grid)
	simdLayer := poly.Plan9SimdForwardForLayer(primary)
	requiresLearn := layerRequiresLearn(primary)
	epochs := crossPathTrainEpochs

	fmt.Printf("\n  ┌─ %s cross-path · %s ─────────────────────────────────────────\n", s.Name, s.Grid)
	fmt.Printf("  │ Paths: SC · MC · SIMD (tiled) · native · native-SIMD\n")

	var rows []crossPathRow
	layerTally := newTestTally()
	passed, failed := 0, 0

	for _, tc := range allDTypes {
		fmt.Printf("  · %-10s ", tc.name)
		row := crossPathRow{DType: tc.name, NativeApplicable: poly.IsLayerNativeExactDType(tc.dtype)}
		tally := newTestTally()

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

		detTol := tc.tolerance
		if detTol < 1e-10 {
			detTol = 1e-10
		}
		if primary == poly.LayerMultiHeadAttention && detTol < 1e-4 {
			detTol = 1e-4
		}
		simdTol := simdParityTol(primary, tc)
		var tiledFwdSimdOut []float32

		// ── Tiled SC / MC / SIMD ─────────────────────────────────────────
		configureTiledNet(net)

		fwdSC := captureForward(net, input, false)
		row.FwdSCDur = formatDur(fwdSC.dur)
		row.FwdSCOK = len(fwdSC.out) > 0 && tensorFinite(fwdSC.out)
		tally.record("tiled.fwd.sc", row.FwdSCOK)

		fwdMC := captureForward(net, input, true)
		row.FwdMCDur = formatDur(fwdMC.dur)
		row.FwdMCOK = len(fwdMC.out) > 0 && tensorFinite(fwdMC.out)
		tally.record("tiled.fwd.mc", row.FwdMCOK)

		row.FwdSCMC = maxAbsDiff(fwdSC.out, fwdMC.out)
		row.FwdSCMCOK = row.FwdSCMC <= detTol
		tally.record("tiled.parity.fwd.sc_mc", row.FwdSCMCOK)

		if simdLayer {
			resetNetwork(net)
			fwdSimd := captureForwardSimd(net, input, true)
			row.FwdSimdDur = formatDur(fwdSimd.dur)
			row.FwdSimdOK = len(fwdSimd.out) > 0 && tensorFinite(fwdSimd.out)
			tally.record("tiled.fwd.simd", row.FwdSimdOK)

			row.FwdTiledSimd = maxAbsDiff(fwdSC.out, fwdSimd.out)
			row.TiledSimdOK = row.FwdTiledSimd <= simdTol
			tally.record("tiled.parity.fwd.sc_simd", row.TiledSimdOK)
			tiledFwdSimdOut = fwdSimd.out
		}

		bwdSC := captureBackward(net, input, target, false)
		row.BwdSCDur = formatDur(bwdSC.dur)
		row.BwdSCOK = len(bwdSC.dx) > 0 && tensorFinite(bwdSC.dx)
		if primary != poly.LayerResidual {
			row.BwdSCOK = row.BwdSCOK && len(bwdSC.dw) > 0 && tensorFinite(bwdSC.dw)
		}
		tally.record("tiled.bwd.sc", row.BwdSCOK)

		bwdMC := captureBackward(net, input, target, true)
		row.BwdMCDur = formatDur(bwdMC.dur)
		row.BwdMCOK = len(bwdMC.dx) > 0 && tensorFinite(bwdMC.dx)
		if primary != poly.LayerResidual {
			row.BwdMCOK = row.BwdMCOK && len(bwdMC.dw) > 0 && tensorFinite(bwdMC.dw)
		}
		tally.record("tiled.bwd.mc", row.BwdMCOK)

		row.BwdSCMC = maxAbsDiff(append(bwdSC.dx, bwdSC.dw...), append(bwdMC.dx, bwdMC.dw...))
		row.BwdSCMCOK = row.BwdSCMC <= detTol*10
		tally.record("tiled.parity.bwd.sc_mc", row.BwdSCMCOK)

		if simdLayer {
			resetNetwork(net)
			bwdSimd := captureBackwardSimd(net, input, target, true)
			row.BwdSimdDur = formatDur(bwdSimd.dur)
			row.BwdSimdOK = len(bwdSimd.dx) > 0 && tensorFinite(bwdSimd.dx)
			if primary != poly.LayerResidual {
				row.BwdSimdOK = row.BwdSimdOK && len(bwdSimd.dw) > 0 && tensorFinite(bwdSimd.dw)
			}
			tally.record("tiled.bwd.simd", row.BwdSimdOK)

			row.BwdTiledSimd = maxAbsDiff(append(bwdSC.dx, bwdSC.dw...), append(bwdSimd.dx, bwdSimd.dw...))
			row.TiledBwdSimdOK = row.BwdTiledSimd <= simdTol*10
			tally.record("tiled.parity.bwd.sc_simd", row.TiledBwdSimdOK)
		}

		row.LossInit = forwardLoss(net, input, target)

		// ── Native / native-SIMD ─────────────────────────────────────────
		if row.NativeApplicable {
			configureNativeNet(net, tc)
			row.NativePathOK = verifyNativePath(net, primary, tc)
			tally.record("native.path", row.NativePathOK)

			setCPUMode(net, false)
			setSimdForward(net, false)
			fwdNat := captureForward(net, input, false)
			row.NatFwdDur = formatDur(fwdNat.dur)
			row.NatFwdOK = len(fwdNat.out) > 0 && tensorFinite(fwdNat.out)
			tally.record("native.fwd", row.NatFwdOK)

			bwdNat := captureBackward(net, input, target, false)
			row.NatBwdDur = formatDur(bwdNat.dur)
			row.NatBwdOK = len(bwdNat.dx) > 0 && tensorFinite(bwdNat.dx)
			if primary != poly.LayerResidual {
				row.NatBwdOK = row.NatBwdOK && len(bwdNat.dw) > 0 && tensorFinite(bwdNat.dw)
			}
			tally.record("native.bwd", row.NatBwdOK)

			row.CrossFwdSCNative = maxAbsDiff(fwdSC.out, fwdNat.out)

			if simdLayer {
				resetNetwork(net)
				fwdNatSimd := captureForwardSimd(net, input, true)
				row.NatFwdSimdDur = formatDur(fwdNatSimd.dur)
				row.NatFwdSimdOK = len(fwdNatSimd.out) > 0 && tensorFinite(fwdNatSimd.out)
				tally.record("native.fwd.simd", row.NatFwdSimdOK)

				row.NatFwdSimdParity = maxAbsDiff(fwdNat.out, fwdNatSimd.out)
				row.NatSimdOK = row.NatFwdSimdParity <= simdTol

				resetNetwork(net)
				bwdNatSimd := captureBackwardSimd(net, input, target, true)
				row.NatBwdSimdDur = formatDur(bwdNatSimd.dur)
				row.NatBwdSimdFinite = len(bwdNatSimd.dx) > 0 && tensorFinite(bwdNatSimd.dx)
				if primary != poly.LayerResidual {
					row.NatBwdSimdFinite = row.NatBwdSimdFinite && len(bwdNatSimd.dw) > 0 && tensorFinite(bwdNatSimd.dw)
				}
				tally.record("native.bwd.simd", row.NatBwdSimdFinite)

				row.NatBwdSimdParity = maxAbsDiff(append(bwdNat.dx, bwdNat.dw...), append(bwdNatSimd.dx, bwdNatSimd.dw...))
				row.NatBwdSimdParityOK = row.NatBwdSimdParity <= simdTol*10

				if len(tiledFwdSimdOut) > 0 {
					row.CrossFwdSimdNat = maxAbsDiff(tiledFwdSimdOut, fwdNatSimd.out)
				}
			}
		}

		// ── Training (fresh nets per path) ───────────────────────────────
		netSC, _ := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
		applyDType(netSC, tc)
		configureTrainingNet(netSC, tc, primary)
		netSC.ReleaseFP32MasterWhenIdle = true
		resSC, durSC, err := trainCPU(netSC, input, target, poly.TrainingModeCPUSC, tc, primary, epochs)
		if err != nil {
			row.Err = "TRAIN-SC"
			rows = append(rows, row)
			failed++
			fmt.Println("TRAIN SC ERR")
			continue
		}
		row.TrainSCDur = formatDur(durSC)
		row.LossFinalSC = resSC.FinalLoss
		if len(resSC.LossHistory) > 0 {
			row.LossFinalSC = resSC.LossHistory[len(resSC.LossHistory)-1]
		}
		lossInitSC := resSC.LossHistory[0]
		if row.LossInit == 0 || math.IsNaN(row.LossInit) {
			row.LossInit = lossInitSC
		}
		row.TrainSCOK = lossFiniteOK(lossInitSC, row.LossFinalSC, requiresLearn) &&
			(!requiresLearn || trainingOK(lossInitSC, row.LossFinalSC, tc.dtype))
		tally.record("train.sc", row.TrainSCOK)

		netMC, _ := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
		applyDType(netMC, tc)
		configureTrainingNet(netMC, tc, primary)
		netMC.ReleaseFP32MasterWhenIdle = true
		resMC, durMC, err := trainCPU(netMC, input, target, poly.TrainingModeCPUMC, tc, primary, epochs)
		if err != nil {
			row.Err = "TRAIN-MC"
			rows = append(rows, row)
			failed++
			fmt.Println("TRAIN MC ERR")
			continue
		}
		row.TrainMCDur = formatDur(durMC)
		row.LossFinalMC = resMC.FinalLoss
		if len(resMC.LossHistory) > 0 {
			row.LossFinalMC = resMC.LossHistory[len(resMC.LossHistory)-1]
		}
		lossInitMC := resMC.LossHistory[0]
		row.TrainMCOK = lossFiniteOK(lossInitMC, row.LossFinalMC, requiresLearn) &&
			(!requiresLearn || trainingOK(lossInitMC, row.LossFinalMC, tc.dtype))
		tally.record("train.mc", row.TrainMCOK)

		if simdLayer {
			netSimd, _ := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
			applyDType(netSimd, tc)
			configureTrainingNet(netSimd, tc, primary)
			netSimd.ReleaseFP32MasterWhenIdle = true
			resSimd, durSimd, err := trainCPU(netSimd, input, target, poly.TrainingModeCPUSimd, tc, primary, epochs)
			if err != nil {
				row.Err = "TRAIN-SIMD"
				rows = append(rows, row)
				failed++
				fmt.Println("TRAIN SIMD ERR")
				continue
			}
			row.TrainSimdDur = formatDur(durSimd)
			row.LossFinalSimd = resSimd.FinalLoss
			if len(resSimd.LossHistory) > 0 {
				row.LossFinalSimd = resSimd.LossHistory[len(resSimd.LossHistory)-1]
			}
			lossInitSimd := resSimd.LossHistory[0]
			row.TrainSimdOK = lossFiniteOK(lossInitSimd, row.LossFinalSimd, requiresLearn) &&
				(!requiresLearn || trainingOK(lossInitSimd, row.LossFinalSimd, tc.dtype))
			tally.record("train.simd", row.TrainSimdOK)
		}

		if row.NativeApplicable {
			netNat, _ := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
			applyDType(netNat, tc)
			configureNativeNet(netNat, tc)
			prepareTrainingNet(netNat, tc.dtype)
			finalizeTrainingNet(netNat, tc)
			netNat.ReleaseFP32MasterWhenIdle = true
			cfg := poly.DefaultTrainingConfig()
			cfg.Epochs = epochs
			cfg.LearningRate = trainingLearningRate(tc.dtype)
			cfg.GradientClip = 1.0
			cfg.Mode = poly.TrainingModeCPUSC
			cfg.Verbose = false
			t0 := time.Now()
			resNat, err := poly.Train(netNat, []poly.TrainingBatch[float32]{{Input: input, Target: target}}, cfg)
			row.TrainNativeDur = formatDur(time.Since(t0))
			if err != nil {
				row.Err = "TRAIN-NAT"
				rows = append(rows, row)
				failed++
				fmt.Println("TRAIN NAT ERR")
				continue
			}
			row.LossFinalNative = resNat.FinalLoss
			if len(resNat.LossHistory) > 0 {
				row.LossFinalNative = resNat.LossHistory[len(resNat.LossHistory)-1]
			}
			lossInitNat := resNat.LossHistory[0]
			row.TrainNativeOK = lossFiniteOK(lossInitNat, row.LossFinalNative, requiresLearn) &&
				(!requiresLearn || trainingOK(lossInitNat, row.LossFinalNative, tc.dtype))
			tally.record("train.native", row.TrainNativeOK)
		}

		row.TestsTotal = tally.total
		row.TestsPassed = tally.passed
		row.OverallOK = row.TestsPassed == row.TestsTotal && row.Err == ""
		rows = append(rows, row)
		layerTally.merge(tally)

		if row.OverallOK {
			passed++
			fmt.Printf("PASS  %d/%d tests  tiled fwd SC/MC/SIMD=%s/%s/%s  native=%s/%s  train SC/MC/SIMD/Nat=%s/%s/%s/%s\n",
				row.TestsPassed, row.TestsTotal,
				row.FwdSCDur, row.FwdMCDur, dashIfEmpty(row.FwdSimdDur),
				dashIfEmpty(row.NatFwdDur), dashIfEmpty(row.NatFwdSimdDur),
				markOK(row.TrainSCOK), markOK(row.TrainMCOK), markOK(row.TrainSimdOK), markOK(row.TrainNativeOK))
		} else {
			failed++
			fmt.Printf("FAIL  %d/%d tests  err=%s\n", row.TestsPassed, row.TestsTotal, row.Err)
		}
	}

	printCrossPathTimingTable(s.Name, rows, simdLayer)
	printCrossPathParityTable(s.Name, rows, simdLayer)
	printCrossPathTrainTable(s.Name, rows, simdLayer)
	printCrossPathTestTally(s.Name, rows, layerTally)
	fmt.Printf("\n  %s cross-path: %d passed · %d failed (of %d dtypes) · tests %d/%d\n",
		s.Name, passed, failed, len(rows), layerTally.passed, layerTally.total)

	crossPathSessionLayers = append(crossPathSessionLayers, crossPathLayerSummary{
		Name: s.Name, Passed: passed, Failed: failed, Rows: rows,
		TestsTotal: layerTally.total, TestsPassed: layerTally.passed,
	})
	crossPathSessionTally.merge(layerTally)
}

func dashIfEmpty(s string) string {
	if s == "" {
		return "—"
	}
	return s
}

func printCrossPathTimingTable(layerName string, rows []crossPathRow, simdLayer bool) {
	fmt.Printf("\n  ┌─ %s timing (fwd / bwd) ─────────────────────────────────────────\n", layerName)
	if simdLayer {
		fmt.Printf("  │ %-10s │ %-7s %-7s %-7s │ %-7s %-7s %-7s │ %-7s %-7s │ %-7s %-7s\n",
			"DType", "SC-f", "MC-f", "SIMD-f", "SC-b", "MC-b", "SIMD-b", "Nat-f", "NatS-f", "Nat-b", "NatS-b")
	} else {
		fmt.Printf("  │ %-10s │ %-7s %-7s │ %-7s %-7s │ %-7s\n",
			"DType", "SC-f", "MC-f", "SC-b", "MC-b", "Nat-f")
	}
	fmt.Println("  ├──────────────────────────────────────────────────────────────────────")
	for _, r := range rows {
		if r.Err != "" {
			fmt.Printf("  │ %-10s │ ERR %s\n", r.DType, r.Err)
			continue
		}
		if simdLayer {
			fmt.Printf("  │ %-10s │ %-7s %-7s %-7s │ %-7s %-7s %-7s │ %-7s %-7s │ %-7s %-7s\n",
				r.DType, r.FwdSCDur, r.FwdMCDur, dashIfEmpty(r.FwdSimdDur),
				r.BwdSCDur, r.BwdMCDur, dashIfEmpty(r.BwdSimdDur),
				dashIfEmpty(r.NatFwdDur), dashIfEmpty(r.NatFwdSimdDur),
				dashIfEmpty(r.NatBwdDur), dashIfEmpty(r.NatBwdSimdDur))
		} else {
			fmt.Printf("  │ %-10s │ %-7s %-7s │ %-7s %-7s │ %-7s\n",
				r.DType, r.FwdSCDur, r.FwdMCDur, r.BwdSCDur, r.BwdMCDur, dashIfEmpty(r.NatFwdDur))
		}
	}
	fmt.Println("  └──────────────────────────────────────────────────────────────────────")
}

func printCrossPathParityTable(layerName string, rows []crossPathRow, simdLayer bool) {
	fmt.Printf("\n  ┌─ %s parity ─────────────────────────────────────────────────────\n", layerName)
	fmt.Println("  │ (native↔native-SIMD and SC↔native are informational — different math paths)")
	fmt.Printf("  │ %-10s │ %-10s %-10s %-10s %-10s │ %-10s %-10s %-10s\n",
		"DType", "F SC↔MC", "F SC↔SIMD", "B SC↔MC", "B SC↔SIMD", "Nat F↔SIMD", "Nat B↔SIMD", "SC↔Nat")
	fmt.Println("  ├──────────────────────────────────────────────────────────────────────")
	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		fmt.Printf("  │ %-10s │ %-10s %-10s %-10s %-10s │ %-10s %-10s %.2e\n",
			r.DType,
			formatParityDiff(r.FwdSCMC), formatParityDiff(r.FwdTiledSimd),
			formatParityDiff(r.BwdSCMC), formatParityDiff(r.BwdTiledSimd),
			formatParityDiff(r.NatFwdSimdParity), formatParityDiff(r.NatBwdSimdParity),
			r.CrossFwdSCNative)
	}
	fmt.Println("  └──────────────────────────────────────────────────────────────────────")
}

func printCrossPathTrainTable(layerName string, rows []crossPathRow, simdLayer bool) {
	fmt.Printf("\n  ┌─ %s training (%d epochs) ───────────────────────────────────────\n", layerName, crossPathTrainEpochs)
	fmt.Printf("  │ %-10s │ %8s %8s %8s %8s %8s │ %-6s %-6s %-6s %-6s\n",
		"DType", "Loss₀", "SC", "MC", "SIMD", "Native", "SC", "MC", "SIMD", "Nat")
	fmt.Println("  ├──────────────────────────────────────────────────────────────────────")
	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		fmt.Printf("  │ %-10s │ %8.4f %8.4f %8.4f %8.4f %8.4f │ %-6s %-6s %-6s %-6s\n",
			r.DType, r.LossInit, r.LossFinalSC, r.LossFinalMC, r.LossFinalSimd, r.LossFinalNative,
			markOK(r.TrainSCOK), markOK(r.TrainMCOK), markOK(r.TrainSimdOK), markOK(r.TrainNativeOK))
	}
	fmt.Println("  └──────────────────────────────────────────────────────────────────────")
}

func printCrossPathTestTally(layerName string, rows []crossPathRow, t *testTally) {
	fmt.Printf("\n  ┌─ %s test tally ─────────────────────────────────────────────────\n", layerName)
	cats := []string{
		"tiled.fwd.sc", "tiled.fwd.mc", "tiled.fwd.simd",
		"tiled.bwd.sc", "tiled.bwd.mc", "tiled.bwd.simd",
		"tiled.parity.fwd.sc_mc", "tiled.parity.fwd.sc_simd",
		"tiled.parity.bwd.sc_mc", "tiled.parity.bwd.sc_simd",
		"native.path", "native.fwd", "native.bwd",
		"native.fwd.simd", "native.bwd.simd",
		"train.sc", "train.mc", "train.simd", "train.native",
	}
	for _, cat := range cats {
		v, ok := t.byCat[cat]
		if !ok || v[1] == 0 {
			continue
		}
		fmt.Printf("  │ %-28s %4d / %4d\n", cat, v[0], v[1])
	}
	fmt.Printf("  │ %-28s %4d / %4d\n", "TOTAL (gated)", t.passed, t.total)
	infoFwd, infoBwd, infoNat := 0, 0, 0
	for _, r := range rows {
		if r.Err != "" || !r.NativeApplicable {
			continue
		}
		infoNat++
		if r.NatSimdOK {
			infoFwd++
		}
		if r.NatBwdSimdParityOK {
			infoBwd++
		}
	}
	if infoNat > 0 {
		fmt.Printf("  │ %-28s %4d / %4d  (informational)\n", "native.parity.fwd.simd", infoFwd, infoNat)
		fmt.Printf("  │ %-28s %4d / %4d  (informational)\n", "native.parity.bwd.simd", infoBwd, infoNat)
	}
	fmt.Println("  └──────────────────────────────────────────────────────────────────────")
}

// PrintCrossPathGlobalManifest prints session-wide summary after [0] or single layer.
func PrintCrossPathGlobalManifest() {
	if len(crossPathSessionLayers) == 0 {
		return
	}
	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  [15] Cross-path global manifest                                      ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")

	dtypePass, dtypeFail := 0, 0
	for _, ls := range crossPathSessionLayers {
		dtypePass += ls.Passed
		dtypeFail += ls.Failed
		status := "PASS"
		if ls.Failed > 0 {
			status = "FAIL"
		}
		fmt.Printf("  %-12s  dtypes %3d/%3d  tests %5d/%5d  %s\n",
			ls.Name, ls.Passed, ls.Passed+ls.Failed, ls.TestsPassed, ls.TestsTotal, status)
	}
	fmt.Printf("\n  Session dtypes: %d passed · %d failed (of %d rows)\n",
		dtypePass, dtypeFail, dtypePass+dtypeFail)
	fmt.Printf("  Session tests:  %d passed · %d failed (of %d checks)\n",
		crossPathSessionTally.passed, crossPathSessionTally.total-crossPathSessionTally.passed, crossPathSessionTally.total)

	fmt.Println("\n  Category breakdown:")
	cats := []string{
		"tiled.fwd.sc", "tiled.fwd.mc", "tiled.fwd.simd",
		"tiled.bwd.sc", "tiled.bwd.mc", "tiled.bwd.simd",
		"tiled.parity.fwd.sc_mc", "tiled.parity.fwd.sc_simd",
		"tiled.parity.bwd.sc_mc", "tiled.parity.bwd.sc_simd",
		"native.path", "native.fwd", "native.bwd",
		"native.fwd.simd", "native.bwd.simd",
		"train.sc", "train.mc", "train.simd", "train.native",
	}
	for _, cat := range cats {
		v, ok := crossPathSessionTally.byCat[cat]
		if !ok || v[1] == 0 {
			continue
		}
		fmt.Printf("    %-28s %5d / %5d\n", cat, v[0], v[1])
	}
}
