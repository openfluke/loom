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
	fwdSC, fwdMC, fwdSimd                   time.Duration
	bwdSC, bwdMC, bwdSimd                   time.Duration
	trainSC, trainMC, trainSimd, trainNative time.Duration

	// Pairwise speed comparisons (computed after all timings + train)
	qatFwdSCSimd, qatFwdMCSimd pairCmp
	qatBwdSCSimd, qatBwdMCSimd pairCmp
	natFwdPair, natBwdPair     pairCmp
	trainSCSimd, trainMCSimd   pairCmp
	fwdWinner, bwdWinner, trainWinner string
	fwdWinRatio, bwdWinRatio, trainWinRatio string
	fwdWinFaster, bwdWinFaster, trainWinFaster string
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
	natFwd, natBwd, natFwdSimd, natBwdSimd time.Duration
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
	fmt.Println("  Tiled (QAT): layer.go GetActive FP32 dequant · Native: *_native.go MAC rules")
	fmt.Println("  Comparisons: QAT SC/MC→SIMD · Native→Native-SIMD · best-QAT vs best-Native")
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
		row.fwdSC = fwdSC.dur
		row.FwdSCDur = formatDur(fwdSC.dur)
		row.FwdSCOK = len(fwdSC.out) > 0 && tensorFinite(fwdSC.out)
		tally.record("tiled.fwd.sc", row.FwdSCOK)

		fwdMC := captureForward(net, input, true)
		row.fwdMC = fwdMC.dur
		row.FwdMCDur = formatDur(fwdMC.dur)
		row.FwdMCOK = len(fwdMC.out) > 0 && tensorFinite(fwdMC.out)
		tally.record("tiled.fwd.mc", row.FwdMCOK)

		row.FwdSCMC = maxAbsDiff(fwdSC.out, fwdMC.out)
		row.FwdSCMCOK = row.FwdSCMC <= detTol
		tally.record("tiled.parity.fwd.sc_mc", row.FwdSCMCOK)

		if simdLayer {
			resetNetwork(net)
			fwdSimd := captureForwardSimd(net, input, true)
			row.fwdSimd = fwdSimd.dur
			row.FwdSimdDur = formatDur(fwdSimd.dur)
			row.FwdSimdOK = len(fwdSimd.out) > 0 && tensorFinite(fwdSimd.out)
			tally.record("tiled.fwd.simd", row.FwdSimdOK)

			row.FwdTiledSimd = maxAbsDiff(fwdSC.out, fwdSimd.out)
			row.TiledSimdOK = row.FwdTiledSimd <= simdTol
			tally.record("tiled.parity.fwd.sc_simd", row.TiledSimdOK)
			tiledFwdSimdOut = fwdSimd.out
		}

		bwdSC := captureBackward(net, input, target, false)
		row.bwdSC = bwdSC.dur
		row.BwdSCDur = formatDur(bwdSC.dur)
		row.BwdSCOK = len(bwdSC.dx) > 0 && tensorFinite(bwdSC.dx)
		if primary != poly.LayerResidual {
			row.BwdSCOK = row.BwdSCOK && len(bwdSC.dw) > 0 && tensorFinite(bwdSC.dw)
		}
		tally.record("tiled.bwd.sc", row.BwdSCOK)

		bwdMC := captureBackward(net, input, target, true)
		row.bwdMC = bwdMC.dur
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
			row.bwdSimd = bwdSimd.dur
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
			row.natFwd = fwdNat.dur
			row.NatFwdDur = formatDur(fwdNat.dur)
			row.NatFwdOK = len(fwdNat.out) > 0 && tensorFinite(fwdNat.out)
			tally.record("native.fwd", row.NatFwdOK)

			bwdNat := captureBackward(net, input, target, false)
			row.natBwd = bwdNat.dur
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
				row.natFwdSimd = fwdNatSimd.dur
				row.NatFwdSimdDur = formatDur(fwdNatSimd.dur)
				row.NatFwdSimdOK = len(fwdNatSimd.out) > 0 && tensorFinite(fwdNatSimd.out)
				tally.record("native.fwd.simd", row.NatFwdSimdOK)

				row.NatFwdSimdParity = maxAbsDiff(fwdNat.out, fwdNatSimd.out)
				row.NatSimdOK = row.NatFwdSimdParity <= simdTol

				resetNetwork(net)
				bwdNatSimd := captureBackwardSimd(net, input, target, true)
				row.natBwdSimd = bwdNatSimd.dur
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
		row.trainSC = durSC
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
		row.trainMC = durMC
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
			row.trainSimd = durSimd
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
			row.trainNative = time.Since(t0)
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

		computeCrossPathComparisons(&row, simdLayer)

		row.TestsTotal = tally.total
		row.TestsPassed = tally.passed
		row.OverallOK = row.TestsPassed == row.TestsTotal && row.Err == ""
		rows = append(rows, row)
		layerTally.merge(tally)

		if row.OverallOK {
			passed++
			fmt.Printf("PASS  %d/%d  QAT-f SC→SIMD %s %s  Nat-f %s %s  best-fwd %s %s  train %s %s\n",
				row.TestsPassed, row.TestsTotal,
				row.qatFwdSCSimd.ratio(), row.qatFwdSCSimd.fasterPct(),
				row.natFwdPair.ratio(), row.natFwdPair.fasterPct(),
				row.fwdWinner, row.fwdWinRatio,
				row.trainWinner, row.trainWinRatio)
		} else {
			failed++
			fmt.Printf("FAIL  %d/%d tests  err=%s\n", row.TestsPassed, row.TestsTotal, row.Err)
		}
	}

	printCrossPathTimingTable(s.Name, rows, simdLayer)
	printCrossPathComparisonTable(s.Name, rows, simdLayer)
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

type namedDur struct {
	name string
	d    time.Duration
}

type pairCmp struct {
	from, to namedDur
	valid    bool
}

func makePair(fromName string, from, to time.Duration, toName string) pairCmp {
	if from <= 0 || to <= 0 {
		return pairCmp{}
	}
	return pairCmp{
		from:  namedDur{fromName, from},
		to:    namedDur{toName, to},
		valid: true,
	}
}

func (p pairCmp) ratio() string {
	if !p.valid || p.from.d <= 0 || p.to.d <= 0 || p.from.d == p.to.d {
		return "—"
	}
	return fmt.Sprintf("%.1f×", float64(p.from.d)/float64(p.to.d))
}

func (p pairCmp) fasterPct() string {
	if !p.valid || p.from.d <= 0 || p.to.d <= 0 || p.from.d == p.to.d {
		return "—"
	}
	pct := float64(p.from.d-p.to.d) / float64(p.from.d) * 100
	if pct >= 0.5 {
		return fmt.Sprintf("%.0f%%", pct)
	}
	if pct <= -0.5 {
		return fmt.Sprintf("%.0f%% slower", -pct)
	}
	return "≈0%"
}

func (p pairCmp) label() string {
	if !p.valid {
		return "—"
	}
	return fmt.Sprintf("%s %s→%s %s", p.from.name, formatDur(p.from.d), p.to.name, formatDur(p.to.d))
}

func fastestNamed(paths []namedDur) namedDur {
	var best namedDur
	for _, p := range paths {
		if p.d <= 0 {
			continue
		}
		if best.d == 0 || p.d < best.d {
			best = p
		}
	}
	return best
}

func paradigmWinner(qat, nat namedDur) (winner, ratio, faster string) {
	if qat.d <= 0 || nat.d <= 0 {
		return "—", "—", "—"
	}
	if qat.d < nat.d {
		return fmt.Sprintf("QAT %s", qat.name), fmt.Sprintf("%.1f×", float64(nat.d)/float64(qat.d)), formatSimdSpeedup(nat.d, qat.d)
	}
	if nat.d < qat.d {
		return fmt.Sprintf("Nat %s", nat.name), fmt.Sprintf("%.1f×", float64(qat.d)/float64(nat.d)), formatSimdSpeedup(qat.d, nat.d)
	}
	return "tie", "1.0×", "≈0%"
}

func computeCrossPathComparisons(row *crossPathRow, simdLayer bool) {
	if simdLayer && row.fwdSimd > 0 {
		row.qatFwdSCSimd = makePair("SC-f", row.fwdSC, row.fwdSimd, "SIMD-f")
		row.qatFwdMCSimd = makePair("MC-f", row.fwdMC, row.fwdSimd, "SIMD-f")
		row.qatBwdSCSimd = makePair("SC-b", row.bwdSC, row.bwdSimd, "SIMD-b")
		row.qatBwdMCSimd = makePair("MC-b", row.bwdMC, row.bwdSimd, "SIMD-b")
		row.trainSCSimd = makePair("SC", row.trainSC, row.trainSimd, "SIMD")
		row.trainMCSimd = makePair("MC", row.trainMC, row.trainSimd, "SIMD")
	}
	if row.NativeApplicable && simdLayer && row.natFwdSimd > 0 {
		row.natFwdPair = makePair("Nat-f", row.natFwd, row.natFwdSimd, "NatS-f")
	}
	if row.NativeApplicable && simdLayer && row.natBwdSimd > 0 {
		row.natBwdPair = makePair("Nat-b", row.natBwd, row.natBwdSimd, "NatS-b")
	}

	qatFwd := fastestNamed([]namedDur{{"SC-f", row.fwdSC}, {"MC-f", row.fwdMC}, {"SIMD-f", row.fwdSimd}})
	qatBwd := fastestNamed([]namedDur{{"SC-b", row.bwdSC}, {"MC-b", row.bwdMC}, {"SIMD-b", row.bwdSimd}})
	qatTrain := fastestNamed([]namedDur{{"SC", row.trainSC}, {"MC", row.trainMC}, {"SIMD", row.trainSimd}})
	natFwd := fastestNamed([]namedDur{{"Nat-f", row.natFwd}, {"NatS-f", row.natFwdSimd}})
	natBwd := fastestNamed([]namedDur{{"Nat-b", row.natBwd}, {"NatS-b", row.natBwdSimd}})
	natTrain := namedDur{"Nat", row.trainNative}

	row.fwdWinner, row.fwdWinRatio, row.fwdWinFaster = paradigmWinner(qatFwd, natFwd)
	row.bwdWinner, row.bwdWinRatio, row.bwdWinFaster = paradigmWinner(qatBwd, natBwd)
	if row.NativeApplicable && row.trainNative > 0 {
		row.trainWinner, row.trainWinRatio, row.trainWinFaster = paradigmWinner(qatTrain, natTrain)
	} else {
		row.trainWinner, row.trainWinRatio, row.trainWinFaster = "n/a", "—", "—"
	}
}

func printCrossPathTimingTable(layerName string, rows []crossPathRow, simdLayer bool) {
	fmt.Printf("\n  ┌─ %s raw timing per path ─────────────────────────────────────────\n", layerName)
	if simdLayer {
		fmt.Printf("  │ %-10s │ %-8s %-8s %-8s │ %-8s %-8s %-8s │ %-8s %-8s │ %-8s %-8s │ %-8s %-8s %-8s %-8s\n",
			"DType", "SC-f", "MC-f", "SIMD-f", "SC-b", "MC-b", "SIMD-b", "Nat-f", "NatS-f", "Nat-b", "NatS-b",
			"Trn-SC", "Trn-MC", "Trn-SIMD", "Trn-Nat")
	} else {
		fmt.Printf("  │ %-10s │ %-8s %-8s │ %-8s %-8s │ %-8s │ %-8s %-8s %-8s\n",
			"DType", "SC-f", "MC-f", "SC-b", "MC-b", "Nat-f", "Trn-SC", "Trn-MC", "Trn-Nat")
	}
	fmt.Println("  ├──────────┼────────┬────────┬────────┼────────┬────────┬────────┼────────┬────────┼────────┬────────┬────────┬────────┬────────")
	for _, r := range rows {
		if r.Err != "" {
			fmt.Printf("  │ %-10s │ ERR %s\n", r.DType, r.Err)
			continue
		}
		if simdLayer {
			fmt.Printf("  │ %-10s │ %-8s %-8s %-8s │ %-8s %-8s %-8s │ %-8s %-8s │ %-8s %-8s │ %-8s %-8s %-8s %-8s\n",
				r.DType, r.FwdSCDur, r.FwdMCDur, dashIfEmpty(r.FwdSimdDur),
				r.BwdSCDur, r.BwdMCDur, dashIfEmpty(r.BwdSimdDur),
				dashIfEmpty(r.NatFwdDur), dashIfEmpty(r.NatFwdSimdDur),
				dashIfEmpty(r.NatBwdDur), dashIfEmpty(r.NatBwdSimdDur),
				r.TrainSCDur, r.TrainMCDur, dashIfEmpty(r.TrainSimdDur), dashIfEmpty(r.TrainNativeDur))
		} else {
			fmt.Printf("  │ %-10s │ %-8s %-8s │ %-8s %-8s │ %-8s │ %-8s %-8s %-8s\n",
				r.DType, r.FwdSCDur, r.FwdMCDur, r.BwdSCDur, r.BwdMCDur, dashIfEmpty(r.NatFwdDur),
				r.TrainSCDur, r.TrainMCDur, dashIfEmpty(r.TrainNativeDur))
		}
	}
	fmt.Println("  └──────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────")
}

func printCrossPathComparisonTable(layerName string, rows []crossPathRow, simdLayer bool) {
	if !simdLayer {
		return
	}
	fmt.Printf("\n  ┌─ %s comparisons (explicit pairs — same paradigm only) ───────────\n", layerName)
	fmt.Println("  │ QAT = tiled GetActive FP32 · Nat = UseExactDType *_native.go")
	fmt.Printf("  │ %-10s │ %-28s %-5s %-7s │ %-28s %-5s %-7s │ %-14s %-5s %-7s\n",
		"DType", "QAT fwd SC→SIMD", "×", "faster", "Nat fwd→SIMD", "×", "faster", "best fwd", "×", "faster")
	fmt.Println("  ├──────────┼────────────────────────────────────┼────────────────────────────────────┼────────────────────────")
	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		fmt.Printf("  │ %-10s │ %-28s %-5s %-7s │ %-28s %-5s %-7s │ %-14s %-5s %-7s\n",
			r.DType, r.qatFwdSCSimd.label(), r.qatFwdSCSimd.ratio(), r.qatFwdSCSimd.fasterPct(),
			r.natFwdPair.label(), r.natFwdPair.ratio(), r.natFwdPair.fasterPct(),
			r.fwdWinner, r.fwdWinRatio, r.fwdWinFaster)
	}
	fmt.Println("  ├──────────┼────────────────────────────────────┼────────────────────────────────────┼────────────────────────")
	fmt.Printf("  │ %-10s │ %-28s %-5s %-7s │ %-28s %-5s %-7s │ %-14s %-5s %-7s\n",
		"DType", "QAT bwd SC→SIMD", "×", "faster", "Nat bwd→SIMD", "×", "faster", "best bwd", "×", "faster")
	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		fmt.Printf("  │ %-10s │ %-28s %-5s %-7s │ %-28s %-5s %-7s │ %-14s %-5s %-7s\n",
			r.DType, r.qatBwdSCSimd.label(), r.qatBwdSCSimd.ratio(), r.qatBwdSCSimd.fasterPct(),
			r.natBwdPair.label(), r.natBwdPair.ratio(), r.natBwdPair.fasterPct(),
			r.bwdWinner, r.bwdWinRatio, r.bwdWinFaster)
	}
	fmt.Println("  ├──────────┼────────────────────────────────────┼────────────────────────────────────┼────────────────────────")
	fmt.Printf("  │ %-10s │ %-28s %-5s %-7s │ %-28s %-5s %-7s │ %-14s %-5s %-7s\n",
		"DType", "QAT train MC→SIMD", "×", "faster", "QAT train SC→SIMD", "×", "faster", "best train", "×", "faster")
	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		fmt.Printf("  │ %-10s │ %-28s %-5s %-7s │ %-28s %-5s %-7s │ %-14s %-5s %-7s\n",
			r.DType, r.trainMCSimd.label(), r.trainMCSimd.ratio(), r.trainMCSimd.fasterPct(),
			r.trainSCSimd.label(), r.trainSCSimd.ratio(), r.trainSCSimd.fasterPct(),
			r.trainWinner, r.trainWinRatio, r.trainWinFaster)
	}
	fmt.Println("  └──────────┴────────────────────────────────────┴────────────────────────────────────┴────────────────────────")
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
