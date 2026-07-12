package sevenlayer

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/openfluke/loom/poly"
)

const DenseNativeLogFile = "dense_native.txt"

// BeginDenseNativeSession tees stdout to lucy_testing_output/dense_native.txt.
func BeginDenseNativeSession() func() {
	_ = os.MkdirAll(OutputDir, 0o755)
	logPath := filepath.Join(OutputDir, DenseNativeLogFile)
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
		fmt.Printf("\n📄 Dense native log: %s\n", logPath)
	}
}

type denseNativeRow struct {
	DType     string
	NativeOK  bool
	FwdOK     bool
	BwdOK     bool
	TrainOK   bool
	LossInit  float64
	LossFinal float64
	FwdDur    string
	BwdDur    string
	TrainDur  string
	Err       string
}

// RunDenseNativeMenu is Lucy [14]: dense-only native-exact forward/backward/train × 21 dtypes.
func RunDenseNativeMenu(reader *bufio.Reader) {
	defer BeginDenseNativeSession()()

	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  [14] Dense true-native — integer MAC + in-place int weight update      ║")
	fmt.Println("║  Log: lucy_testing_output/dense_native.txt                             ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Println("  Int8/Int4/Ternary/etc: pure integer forward · backward · SGD (no FP32 master)")
	fmt.Println("  Other dtypes: native-exact MAC rules (no GetActive bulk dequant)")
	fmt.Println("  Grid 1³ · 7 Dense/cell · 30 train epochs per dtype")
	fmt.Println()

	g := GridSpec{Depth: 1, Rows: 1, Cols: 1}
	s := buildDenseSuite(g)
	epochs := 30

	var rows []denseNativeRow
	passed, failed := 0, 0

	for _, tc := range allDTypes {
		if !poly.IsDenseNativeExactDType(tc.dtype) {
			continue
		}
		fmt.Printf("  · %-10s ", tc.name)
		row := denseNativeRow{DType: tc.name}

		net, err := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
		if err != nil {
			row.Err = "BUILD"
			rows = append(rows, row)
			failed++
			fmt.Println("BUILD ERR")
			continue
		}
		applyDType(net, tc)
		configureDenseNativeNet(net, tc)
		prepareTrainingNet(net, tc.dtype)
		finalizeTrainingNet(net, tc)

		nativeLayers := 0
		trueNativeLayers := 0
		for i := range net.Layers {
			if poly.DenseUsesNativeExact(&net.Layers[i]) {
				nativeLayers++
			}
			if poly.DenseUsesTrueNative(&net.Layers[i]) {
				trueNativeLayers++
			}
		}
		row.NativeOK = nativeLayers == len(net.Layers)
		if poly.IsDenseTrueNativeDType(tc.dtype) {
			row.NativeOK = row.NativeOK && trueNativeLayers == len(net.Layers)
		}
		if !row.NativeOK {
			row.Err = "PATH"
			rows = append(rows, row)
			failed++
			fmt.Println("not native-exact")
			continue
		}

		input := s.MakeInput()
		target := s.MakeTarget(net, input)
		setCPUMode(net, false)
		setSimdForward(net, false)

		t0 := time.Now()
		fwd := captureForward(net, input, false)
		row.FwdDur = formatDur(time.Since(t0))
		row.FwdOK = len(fwd.out) > 0 && tensorFinite(fwd.out)

		t0 = time.Now()
		bwd := captureBackward(net, input, target, false)
		row.BwdDur = formatDur(time.Since(t0))
		row.BwdOK = len(bwd.dx) > 0 && len(bwd.dw) > 0 && tensorFinite(bwd.dx) && tensorFinite(bwd.dw)

		net.ReleaseFP32MasterWhenIdle = true
		cfg := poly.DefaultTrainingConfig()
		cfg.Epochs = epochs
		cfg.LearningRate = trainingLearningRate(tc.dtype)
		cfg.GradientClip = 1.0
		cfg.Mode = poly.TrainingModeCPUSC
		cfg.Verbose = false
		t0 = time.Now()
		res, err := poly.Train(net, []poly.TrainingBatch[float32]{{Input: input, Target: target}}, cfg)
		row.TrainDur = formatDur(time.Since(t0))
		if err != nil {
			row.Err = "TRAIN"
			rows = append(rows, row)
			failed++
			fmt.Println("TRAIN ERR")
			continue
		}
		row.LossInit = res.LossHistory[0]
		row.LossFinal = res.FinalLoss
		if len(res.LossHistory) > 0 {
			row.LossFinal = res.LossHistory[len(res.LossHistory)-1]
		}
		row.TrainOK = lossFiniteOK(row.LossInit, row.LossFinal, true) && trainingOK(row.LossInit, row.LossFinal, tc.dtype)

		ok := row.NativeOK && row.FwdOK && row.BwdOK && row.TrainOK
		rows = append(rows, row)
		if ok {
			passed++
			fmt.Printf("PASS  fwd %s bwd %s loss %.4f→%.4f  train %s\n",
				row.FwdDur, row.BwdDur, row.LossInit, row.LossFinal, row.TrainDur)
		} else {
			failed++
			fmt.Printf("FAIL  fwd=%v bwd=%v train=%v  loss %.4f→%.4f\n",
				row.FwdOK, row.BwdOK, row.TrainOK, row.LossInit, row.LossFinal)
		}
	}

	printDenseNativeTable(rows)
	fmt.Printf("\n  Dense native-exact: %d passed · %d failed (of %d dtypes)\n", passed, failed, len(rows))
	_ = reader
}

func configureDenseNativeNet(net *poly.VolumetricNetwork, tc dtypeCase) {
	wireLayerTree(net)
	net.UseExactDType = poly.IsDenseNativeExactDType(tc.dtype)
}

func buildDenseSuite(g GridSpec) LayerSuite {
	dims := denseEndpoints(g)
	acts := []string{"LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR"}
	return LayerSuite{
		Name:          "Dense",
		Grid:          g,
		PrimaryType:   poly.LayerDense,
		CheckpointTag: "dense_native" + gridCheckpointSuffix(g),
		Banner:        fmt.Sprintf("  Grid %s · native-exact dense", g),
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "loom-dense-native", g)
			first := true
			forEachGridCell(g, func(z, y, x int) {
				for i := 0; i < sevenLayersPerCell; i++ {
					appendLayerJSON(&b, &first, fmt.Sprintf(
						`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"DENSE","activation":"%s","dtype":"%s","input_height":%d,"output_height":%d}`,
						z, y, x, i, acts[i], jsonDType, dims[i], dims[i+1],
					))
				}
			})
			b.WriteString(`]}`)
			return []byte(b.String())
		},
		MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, dims[0]) },
		MakeTarget: sinTarget,
	}
}

func printDenseNativeTable(rows []denseNativeRow) {
	fmt.Println()
	fmt.Println("  ┌─ Dense native-exact · 1³ ───────────────────────────────────────────")
	fmt.Printf("  │ %-10s %5s %5s %5s %5s %8s %8s %8s %s\n",
		"DType", "Path", "Fwd", "Bwd", "Train", "Loss₀", "Lossₙ", "Time", "Err")
	fmt.Println("  ├──────────────────────────────────────────────────────────────────────")
	for _, r := range rows {
		fmt.Printf("  │ %-10s %5s %5s %5s %5s %8.4f %8.4f %8s %s\n",
			r.DType,
			markOK(r.NativeOK), markOK(r.FwdOK), markOK(r.BwdOK), markOK(r.TrainOK),
			r.LossInit, r.LossFinal, r.TrainDur, r.Err)
	}
	fmt.Println("  └──────────────────────────────────────────────────────────────────────")
}

func tensorFinite(data []float32) bool {
	for _, v := range data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return false
		}
	}
	return true
}
