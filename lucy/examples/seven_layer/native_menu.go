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

const NativeLogFile = "native_layers.txt"

// BeginNativeSession tees stdout to lucy_testing_output/native_layers.txt.
func BeginNativeSession() func() {
	_ = os.MkdirAll(OutputDir, 0o755)
	logPath := filepath.Join(OutputDir, NativeLogFile)
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
		fmt.Printf("\n📄 Native layer log: %s\n", logPath)
	}
}

type nativeLayerEntry struct {
	name string
	run  func(*bufio.Reader)
}

var nativeMenuEntries = []nativeLayerEntry{
	{"Dense", runNativeDense},
	{"SwiGLU", runNativeSwiGLU},
	{"MHA", runNativeMHA},
	{"CNN1", runNativeCNN1},
	{"CNN2", runNativeCNN2},
	{"CNN3", runNativeCNN3},
	{"RNN", runNativeRNN},
	{"LSTM", runNativeLSTM},
	{"Embedding", runNativeEmbedding},
	{"Residual", runNativeResidual},
}

type nativeRow struct {
	DType      string
	NativeOK   bool
	FwdOK      bool
	BwdOK      bool
	TrainOK    bool
	LossInit   float64
	LossFinal  float64
	FwdDur     string
	BwdDur     string
	TrainDur   string
	FwdSimdDur string
	BwdSimdDur string
	FwdSimdPct string
	BwdSimdPct string
	SimdOK     bool
	Err        string
}

// RunNativeMenu is Lucy [14]: per-layer native forward/backward/train × 21 dtypes.
func RunNativeMenu(reader *bufio.Reader) {
	defer BeginNativeSession()()

	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  [14] Native layer suite — forward + backward in storage dtype         ║")
	fmt.Println("║  Log: lucy_testing_output/native_layers.txt                            ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Println("  layer.go = GetActive FP32 dequant · *_native.go = GetNative MAC rules")
	fmt.Println("  Grid 1³ · 7 layers/cell · 30 train epochs per dtype")
	fmt.Println()
	fmt.Println("  [0] Run all layer types")
	for i, e := range nativeMenuEntries {
		fmt.Printf("  [%d] %s\n", i+1, e.name)
	}
	fmt.Print("Choice [1]: ")

	line, _ := reader.ReadString('\n')
	line = strings.TrimSpace(line)
	if line == "" {
		line = "1"
	}

	if line == "0" {
		for _, e := range nativeMenuEntries {
			fmt.Printf("\n▶ Native %s …\n", e.name)
			e.run(reader)
		}
		return
	}

	idx, err := strconv.Atoi(line)
	if err != nil || idx < 1 || idx > len(nativeMenuEntries) {
		fmt.Println("Invalid selection.")
		return
	}
	nativeMenuEntries[idx-1].run(reader)
}

func runNativeDense(reader *bufio.Reader)     { runNativeLayerSuite(buildDenseSuite(grid1()), poly.LayerDense, reader) }
func runNativeSwiGLU(reader *bufio.Reader)    { runNativeLayerSuite(buildSwiGLUNativeSuite(grid1()), poly.LayerSwiGLU, reader) }
func runNativeMHA(reader *bufio.Reader)       { runNativeLayerSuite(buildMHANativeSuite(grid1()), poly.LayerMultiHeadAttention, reader) }
func runNativeCNN1(reader *bufio.Reader)      { runNativeLayerSuite(buildCNN1NativeSuite(grid1()), poly.LayerCNN1, reader) }
func runNativeCNN2(reader *bufio.Reader)      { runNativeLayerSuite(buildCNN2NativeSuite(grid1()), poly.LayerCNN2, reader) }
func runNativeCNN3(reader *bufio.Reader)      { runNativeLayerSuite(buildCNN3NativeSuite(grid1()), poly.LayerCNN3, reader) }
func runNativeRNN(reader *bufio.Reader)      { runNativeLayerSuite(buildRNNNativeSuite(grid1()), poly.LayerRNN, reader) }
func runNativeLSTM(reader *bufio.Reader)     { runNativeLayerSuite(buildLSTMNativeSuite(grid1()), poly.LayerLSTM, reader) }
func runNativeEmbedding(reader *bufio.Reader) { runNativeLayerSuite(buildEmbeddingNativeSuite(grid1()), poly.LayerEmbedding, reader) }
func runNativeResidual(reader *bufio.Reader)  { runNativeLayerSuite(buildResidualNativeSuite(grid1()), poly.LayerResidual, reader) }

func grid1() GridSpec { return GridSpec{Depth: 1, Rows: 1, Cols: 1} }

func configureNativeNet(net *poly.VolumetricNetwork, tc dtypeCase) {
	wireLayerTree(net)
	net.UseExactDType = poly.IsLayerNativeExactDType(tc.dtype)
}

func runNativeLayerSuite(s LayerSuite, primary poly.LayerType, reader *bufio.Reader) {
	_ = reader
	epochs := 30
	fmt.Printf("\n  ┌─ %s native · %s ───────────────────────────────────────────\n", s.Name, s.Grid)

	var rows []nativeRow
	passed, failed := 0, 0

	for _, tc := range allDTypes {
		if !poly.IsLayerNativeExactDType(tc.dtype) {
			continue
		}
		fmt.Printf("  · %-10s ", tc.name)
		row := nativeRow{DType: tc.name}

		net, err := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
		if err != nil {
			row.Err = "BUILD"
			rows = append(rows, row)
			failed++
			fmt.Println("BUILD ERR")
			continue
		}
		applyDType(net, tc)
		configureNativeNet(net, tc)
		prepareTrainingNet(net, tc.dtype)
		finalizeTrainingNet(net, tc)

		nativeLayers := 0
		for i := range net.Layers {
			if poly.LayerUsesNativeExact(&net.Layers[i]) {
				nativeLayers++
			}
		}
		row.NativeOK = nativeLayers == len(net.Layers)
		if poly.IsDenseTrueNativeDType(tc.dtype) && primary == poly.LayerDense {
			trueNative := 0
			for i := range net.Layers {
				if poly.DenseUsesTrueNative(&net.Layers[i]) {
					trueNative++
				}
			}
			row.NativeOK = row.NativeOK && trueNative == len(net.Layers)
		}
		if !row.NativeOK {
			row.Err = "PATH"
			rows = append(rows, row)
			failed++
			fmt.Println("not native")
			continue
		}

		input := s.MakeInput()
		target := s.MakeTarget(net, input)
		setCPUMode(net, false)
		setSimdForward(net, false)

		fwd := captureForward(net, input, false)
		row.FwdDur = formatDur(fwd.dur)
		row.FwdOK = len(fwd.out) > 0 && tensorFinite(fwd.out)

		bwd := captureBackward(net, input, target, false)
		row.BwdDur = formatDur(bwd.dur)
		row.BwdOK = len(bwd.dx) > 0 && tensorFinite(bwd.dx)
		if primary != poly.LayerResidual {
			row.BwdOK = row.BwdOK && len(bwd.dw) > 0 && tensorFinite(bwd.dw)
		}

		if primary == poly.LayerMultiHeadAttention || primary == poly.LayerDense || primary == poly.LayerSwiGLU || primary == poly.LayerCNN1 || primary == poly.LayerCNN2 || primary == poly.LayerCNN3 || primary == poly.LayerRNN || primary == poly.LayerLSTM || primary == poly.LayerEmbedding || primary == poly.LayerResidual {
			if poly.Plan9SimdForwardForLayer(primary) {
				resetNetwork(net)
				fwdSimd := captureForwardSimd(net, input, true)
				row.FwdSimdDur = formatDur(fwdSimd.dur)
				row.FwdSimdPct = formatSimdSpeedup(fwd.dur, fwdSimd.dur)
				row.SimdOK = len(fwdSimd.out) > 0 && tensorFinite(fwdSimd.out)

				resetNetwork(net)
				bwdSimd := captureBackwardSimd(net, input, target, true)
				row.BwdSimdDur = formatDur(bwdSimd.dur)
				row.BwdSimdPct = formatSimdSpeedup(bwd.dur, bwdSimd.dur)
				row.SimdOK = row.SimdOK && len(bwdSimd.dx) > 0 && tensorFinite(bwdSimd.dx)
				if primary != poly.LayerResidual {
					row.SimdOK = row.SimdOK && len(bwdSimd.dw) > 0 && tensorFinite(bwdSimd.dw)
				}
			}
		}

		net.ReleaseFP32MasterWhenIdle = true
		cfg := poly.DefaultTrainingConfig()
		cfg.Epochs = epochs
		cfg.LearningRate = trainingLearningRate(tc.dtype)
		cfg.GradientClip = 1.0
		cfg.Mode = poly.TrainingModeCPUSC
		cfg.Verbose = false
		t0 := time.Now()
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
		requiresLearn := layerRequiresLearn(primary)
		row.TrainOK = lossFiniteOK(row.LossInit, row.LossFinal, requiresLearn) && trainingOK(row.LossInit, row.LossFinal, tc.dtype)

		ok := row.NativeOK && row.FwdOK && row.BwdOK && row.TrainOK
		rows = append(rows, row)
		if ok {
			passed++
			if row.FwdSimdDur != "" {
				fmt.Printf("PASS  fwd %s bwd %s simd fwd %s (%s) bwd %s (%s) loss %.4f→%.4f  train %s\n",
					row.FwdDur, row.BwdDur, row.FwdSimdDur, row.FwdSimdPct, row.BwdSimdDur, row.BwdSimdPct,
					row.LossInit, row.LossFinal, row.TrainDur)
			} else {
				fmt.Printf("PASS  fwd %s bwd %s loss %.4f→%.4f  train %s\n",
					row.FwdDur, row.BwdDur, row.LossInit, row.LossFinal, row.TrainDur)
			}
		} else {
			failed++
			fmt.Printf("FAIL  fwd=%v bwd=%v train=%v  loss %.4f→%.4f\n",
				row.FwdOK, row.BwdOK, row.TrainOK, row.LossInit, row.LossFinal)
		}
	}

	printNativeTable(s.Name, rows)
	fmt.Printf("\n  %s native: %d passed · %d failed (of %d dtypes)\n", s.Name, passed, failed, len(rows))
}

func buildSwiGLUNativeSuite(g GridSpec) LayerSuite {
	dims := swigluEndpoints(g)
	return LayerSuite{
		Name:        "SwiGLU",
		Grid:        g,
		PrimaryType: poly.LayerSwiGLU,
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "loom-swiglu-native", g)
			first := true
			forEachGridCell(g, func(z, y, x int) {
				for i := 0; i < sevenLayersPerCell; i++ {
					appendLayerJSON(&b, &first, fmt.Sprintf(
						`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"SWIGLU","activation":"RELU","dtype":"%s","input_height":%d,"output_height":%d}`,
						z, y, x, i, jsonDType, dims[i], dims[i+1],
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

func buildMHANativeSuite(g GridSpec) LayerSuite {
	m := mhaShapeFor(g)
	return LayerSuite{
		Name:        "MHA",
		Grid:        g,
		PrimaryType: poly.LayerMultiHeadAttention,
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "loom-mha-native", g)
			first := true
			forEachGridCell(g, func(z, y, x int) {
				for i := 0; i < sevenLayersPerCell; i++ {
					appendLayerJSON(&b, &first, fmt.Sprintf(
						`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"MHA","activation":"RELU","dtype":"%s","d_model":%d,"num_heads":%d,"seq_length":%d}`,
						z, y, x, i, jsonDType, m.dModel, m.heads, m.seq,
					))
				}
			})
			b.WriteString(`]}`)
			return []byte(b.String())
		},
		MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, m.seq, m.dModel) },
		MakeTarget: sinTarget,
	}
}

func buildCNN1NativeSuite(g GridSpec) LayerSuite {
	ch := cnnChannelEndpoints(g)
	sp := cnnSpatial(g)
	return LayerSuite{
		Name:        "CNN1",
		Grid:        g,
		PrimaryType: poly.LayerCNN1,
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "loom-cnn1-native", g)
			first := true
			forEachGridCell(g, func(z, y, x int) {
				for i := 0; i < sevenLayersPerCell; i++ {
					appendLayerJSON(&b, &first, fmt.Sprintf(
						`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"CNN1","activation":"RELU","dtype":"%s","input_channels":%d,"filters":%d,"input_height":%d,"output_height":%d,"kernel_size":3,"stride":1,"padding":1}`,
						z, y, x, i, jsonDType, ch[i], ch[i+1], sp, sp,
					))
				}
			})
			b.WriteString(`]}`)
			return []byte(b.String())
		},
		MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, ch[0], sp) },
		MakeTarget: sinTarget,
	}
}

func buildCNN2NativeSuite(g GridSpec) LayerSuite {
	ch := cnnChannelEndpoints(g)
	sp := cnnSpatial(g)
	return LayerSuite{
		Name:        "CNN2",
		Grid:        g,
		PrimaryType: poly.LayerCNN2,
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "loom-cnn2-native", g)
			first := true
			forEachGridCell(g, func(z, y, x int) {
				for i := 0; i < sevenLayersPerCell; i++ {
					appendLayerJSON(&b, &first, fmt.Sprintf(
						`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"CNN2","activation":"RELU","dtype":"%s","input_channels":%d,"filters":%d,"input_height":%d,"input_width":%d,"output_height":%d,"output_width":%d,"kernel_size":3,"stride":1,"padding":1}`,
						z, y, x, i, jsonDType, ch[i], ch[i+1], sp, sp, sp, sp,
					))
				}
			})
			b.WriteString(`]}`)
			return []byte(b.String())
		},
		MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, ch[0], sp, sp) },
		MakeTarget: sinTarget,
	}
}

func buildCNN3NativeSuite(g GridSpec) LayerSuite {
	ch := cnn3ChannelEndpoints(g)
	d, h, w := cnn3Spatial(g)
	return LayerSuite{
		Name:        "CNN3",
		Grid:        g,
		PrimaryType: poly.LayerCNN3,
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "loom-cnn3-native", g)
			first := true
			forEachGridCell(g, func(z, y, x int) {
				for i := 0; i < sevenLayersPerCell; i++ {
					appendLayerJSON(&b, &first, fmt.Sprintf(
						`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"CNN3","activation":"RELU","dtype":"%s","input_channels":%d,"filters":%d,"input_depth":%d,"input_height":%d,"input_width":%d,"output_depth":%d,"output_height":%d,"output_width":%d,"kernel_size":3,"stride":1,"padding":1}`,
						z, y, x, i, jsonDType, ch[i], ch[i+1], d, h, w, d, h, w,
					))
				}
			})
			b.WriteString(`]}`)
			return []byte(b.String())
		},
		MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, ch[0], d, h, w) },
		MakeTarget: sinTarget,
	}
}

func buildRNNNativeSuite(g GridSpec) LayerSuite {
	dims := rnnEndpoints(g)
	return LayerSuite{
		Name:        "RNN",
		Grid:        g,
		PrimaryType: poly.LayerRNN,
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "loom-rnn-native", g)
			first := true
			seq := 4
			forEachGridCell(g, func(z, y, x int) {
				for i := 0; i < sevenLayersPerCell; i++ {
					appendLayerJSON(&b, &first, fmt.Sprintf(
						`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"RNN","activation":"TANH","dtype":"%s","input_height":%d,"output_height":%d,"seq_length":%d}`,
						z, y, x, i, jsonDType, dims[i], dims[i+1], seq,
					))
				}
			})
			b.WriteString(`]}`)
			return []byte(b.String())
		},
		MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, 4, dims[0]) },
		MakeTarget: sinTarget,
	}
}

func buildLSTMNativeSuite(g GridSpec) LayerSuite {
	dims := rnnEndpoints(g)
	return LayerSuite{
		Name:        "LSTM",
		Grid:        g,
		PrimaryType: poly.LayerLSTM,
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "loom-lstm-native", g)
			first := true
			seq := 4
			forEachGridCell(g, func(z, y, x int) {
				for i := 0; i < sevenLayersPerCell; i++ {
					appendLayerJSON(&b, &first, fmt.Sprintf(
						`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"LSTM","activation":"TANH","dtype":"%s","input_height":%d,"output_height":%d,"seq_length":%d}`,
						z, y, x, i, jsonDType, dims[i], dims[i+1], seq,
					))
				}
			})
			b.WriteString(`]}`)
			return []byte(b.String())
		},
		MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, 4, dims[0]) },
		MakeTarget: sinTarget,
	}
}

func buildEmbeddingNativeSuite(g GridSpec) LayerSuite {
	vocab := embeddingVocab(g)
	seq := embeddingSeqLen(g)
	acts := []string{"RELU", "RELU", "RELU", "RELU", "RELU", "SIGMOID"}
	return LayerSuite{
		Name:        "Embedding",
		Grid:        g,
		PrimaryType: poly.LayerEmbedding,
		BuildJSON: func(jsonDType string) []byte {
			dims := embeddingDims(g)
			denseOnly := flatEndpoints(dims[len(dims)-1])
			var b strings.Builder
			writeNetworkHeader(&b, "loom-embedding-native", g)
			first := true
			forEachGridCell(g, func(z, y, x int) {
				if isStackOrigin(z, y, x) {
					appendLayerJSON(&b, &first, fmt.Sprintf(
						`{"z":%d,"y":%d,"x":%d,"l":0,"type":"EMBEDDING","dtype":"%s","vocab_size":%d,"embedding_dim":%d}`,
						z, y, x, jsonDType, vocab, dims[0],
					))
					for i := 0; i < len(dims)-1; i++ {
						appendLayerJSON(&b, &first, fmt.Sprintf(
							`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"DENSE","activation":"%s","dtype":"%s","input_height":%d,"output_height":%d}`,
							z, y, x, i+1, acts[i], jsonDType, dims[i], dims[i+1],
						))
					}
					return
				}
				for i := 0; i < sevenLayersPerCell; i++ {
					appendLayerJSON(&b, &first, fmt.Sprintf(
						`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"DENSE","activation":"%s","dtype":"%s","input_height":%d,"output_height":%d}`,
						z, y, x, i, acts[i%len(acts)], jsonDType, denseOnly[i], denseOnly[i+1],
					))
				}
			})
			b.WriteString(`]}`)
			return []byte(b.String())
		},
		MakeInput: func() *poly.Tensor[float32] {
			t := poly.NewTensor[float32](seq, 1)
			for i := range t.Data {
				t.Data[i] = float32(i % vocab)
			}
			return t
		},
		MakeTarget: sinTarget,
	}
}

func buildResidualNativeSuite(g GridSpec) LayerSuite {
	dim := residualDim(g)
	return LayerSuite{
		Name:        "Residual",
		Grid:        g,
		PrimaryType: poly.LayerResidual,
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "loom-residual-native", g)
			first := true
			forEachGridCell(g, func(z, y, x int) {
				for i := 0; i < sevenLayersPerCell; i++ {
					appendLayerJSON(&b, &first, fmt.Sprintf(
						`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"RESIDUAL","dtype":"%s","input_height":%d,"output_height":%d}`,
						z, y, x, i, jsonDType, dim, dim,
					))
				}
			})
			b.WriteString(`]}`)
			return []byte(b.String())
		},
		MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, dim) },
		MakeTarget: sinTarget,
	}
}

func printNativeTable(layerName string, rows []nativeRow) {
	fmt.Println()
	fmt.Printf("  ┌─ %s native · 1³ ───────────────────────────────────────────\n", layerName)
	hasSimd := false
	for _, r := range rows {
		if r.FwdSimdDur != "" {
			hasSimd = true
			break
		}
	}
	if hasSimd {
		fmt.Printf("  │ %-10s %5s %5s %5s %5s %8s %8s %8s %8s %8s %s\n",
			"DType", "Path", "Fwd", "Bwd", "Train", "Loss₀", "Lossₙ", "Fwd", "Bwd", "Time", "Err")
		fmt.Println("  │                                                          SIMD fwd / bwd speedup")
	} else {
		fmt.Printf("  │ %-10s %5s %5s %5s %5s %8s %8s %8s %s\n",
			"DType", "Path", "Fwd", "Bwd", "Train", "Loss₀", "Lossₙ", "Time", "Err")
	}
	fmt.Println("  ├──────────────────────────────────────────────────────────────────────")
	for _, r := range rows {
		if hasSimd {
			simdFwd := r.FwdSimdDur
			if r.FwdSimdPct != "" {
				simdFwd += " " + r.FwdSimdPct
			}
			simdBwd := r.BwdSimdDur
			if r.BwdSimdPct != "" {
				simdBwd += " " + r.BwdSimdPct
			}
			fmt.Printf("  │ %-10s %5s %5s %5s %5s %8.4f %8.4f %8s %8s %8s %s\n",
				r.DType,
				markOK(r.NativeOK), markOK(r.FwdOK), markOK(r.BwdOK), markOK(r.TrainOK),
				r.LossInit, r.LossFinal, simdFwd, simdBwd, r.TrainDur, r.Err)
		} else {
			fmt.Printf("  │ %-10s %5s %5s %5s %5s %8.4f %8.4f %8s %s\n",
				r.DType,
				markOK(r.NativeOK), markOK(r.FwdOK), markOK(r.BwdOK), markOK(r.TrainOK),
				r.LossInit, r.LossFinal, r.TrainDur, r.Err)
		}
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

// RunDenseNativeMenu forwards to RunNativeMenu (legacy name).
func RunDenseNativeMenu(reader *bufio.Reader) {
	RunNativeMenu(reader)
}

// buildDenseSuite for native menu (1³ dense stack).
func buildDenseSuite(g GridSpec) LayerSuite {
	dims := denseEndpoints(g)
	acts := []string{"LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR"}
	return LayerSuite{
		Name:        "Dense",
		Grid:        g,
		PrimaryType: poly.LayerDense,
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
