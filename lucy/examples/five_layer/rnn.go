// Five-layer RNN example for Loom (poly).
//
// STANDALONE (copy into your repo):
//   1. Copy this file; in go.mod: require github.com/openfluke/loom (or your poly path)
//   2. Change `package fivelayer` → `package main`
//   3. Rename `RunRNN` → `main`  (or call runRNNExample() from main)
//   4. go run rnn.go
//
// Lucy menu [6] calls RunRNN(); output is teed to lucy_testing_output/five_layer.txt
package fivelayer

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/openfluke/loom/poly"
)

const (
	rnnTrainEpochs  = 50
	rnnLearningRate = float32(0.01)
	rnnGPUMode      = poly.TrainingModeGPUSC
	rnnOutputDir    = "lucy_testing_output"
)

// ── Numerical types (all 21) ─────────────────────────────────────────────────

type rnnDtypeCase struct {
	name, jsonName string
	dtype          poly.DType
	scale          float32
	tolerance      float64
}

var rnnAllDTypes = []rnnDtypeCase{
	{"Float64", "FLOAT64", poly.DTypeFloat64, 1.0, 1e-3},
	{"Float32", "FLOAT32", poly.DTypeFloat32, 1.0, 1e-5},
	{"Float16", "FLOAT16", poly.DTypeFloat16, 1.0, 1e-3},
	{"BFloat16", "BFLOAT16", poly.DTypeBFloat16, 1.0, 1e-3},
	{"FP8-E4M3", "FP8E4M3", poly.DTypeFP8E4M3, 0.01, 1e-3},
	{"FP8-E5M2", "FP8E5M2", poly.DTypeFP8E5M2, 0.01, 1e-3},
	{"Int64", "INT64", poly.DTypeInt64, 0.01, 1e-3},
	{"Uint64", "UINT64", poly.DTypeUint64, 0.01, 1e-3},
	{"Int32", "INT32", poly.DTypeInt32, 0.01, 1e-3},
	{"Uint32", "UINT32", poly.DTypeUint32, 0.01, 1e-3},
	{"Int16", "INT16", poly.DTypeInt16, 0.01, 1e-3},
	{"Uint16", "UINT16", poly.DTypeUint16, 0.01, 1e-3},
	{"Int8", "INT8", poly.DTypeInt8, 0.01, 1e-3},
	{"Uint8", "UINT8", poly.DTypeUint8, 0.01, 1e-3},
	{"Int4", "INT4", poly.DTypeInt4, 0.01, 1e-3},
	{"Uint4", "UINT4", poly.DTypeUint4, 0.01, 1e-3},
	{"FP4", "FP4", poly.DTypeFP4, 0.01, 1e-3},
	{"Int2", "INT2", poly.DTypeInt2, 0.01, 1e-3},
	{"Uint2", "UINT2", poly.DTypeUint2, 0.01, 1e-3},
	{"Ternary", "TERNARY", poly.DTypeTernary, 0.1, 1e-3},
	{"Binary", "BINARY", poly.DTypeBinary, 0.1, 1e-3},
}

// ── Deviation buckets (save/reload parity) ───────────────────────────────────

type rnnSpectrum int

const (
	rnnSpecExact rnnSpectrum = iota
	rnnSpecIndustry
	rnnSpecLowBit
	rnnSpecDrift
	rnnSpecHeavyDrift
	rnnSpecBroken
	rnnSpecFatal
)

func (s rnnSpectrum) String() string {
	switch s {
	case rnnSpecExact:
		return "💎 EXACT"
	case rnnSpecIndustry:
		return "✅ INDUS"
	case rnnSpecLowBit:
		return "🟨 LOWBIT"
	case rnnSpecDrift:
		return "🟠 DRIFT"
	case rnnSpecHeavyDrift:
		return "🟤 H-DRIFT"
	case rnnSpecBroken:
		return "❌ BROKE"
	default:
		return "💀 FATAL"
	}
}

// ── Step 1: Build JSON network spec (5 flat RNN layers, no SEQUENTIAL) ───

func rnnNetworkJSON(jsonDType string) []byte {
	dims := []int{16, 32, 32, 32, 32, 16}
	var b strings.Builder
	b.WriteString(fmt.Sprintf(
		`{"id":"loom-five-rnn","depth":1,"rows":1,"cols":1,"layers_per_cell":%d,"layers":[`,
		len(dims)-1,
	))
	for i := 0; i < len(dims)-1; i++ {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(fmt.Sprintf(
			`{"z":0,"y":0,"x":0,"l":%d,"type":"RNN","activation":"TANH","dtype":"%s","input_height":%d,"output_height":%d,"seq_length":8}`,
			i, jsonDType, dims[i], dims[i+1],
		))
	}
	b.WriteString(`]}`)
	return []byte(b.String())
}

// ── Step 2: Create network from JSON ─────────────────────────────────────────

func rnnCreateNetwork(jsonDType string) (*poly.VolumetricNetwork, error) {
	return poly.BuildNetworkFromJSON(rnnNetworkJSON(jsonDType))
}

// ── Input / target tensors ───────────────────────────────────────────────────

func rnnMakeInput() *poly.Tensor[float32] {
	t := poly.NewTensor[float32](1, 8, 16)
	for i := range t.Data {
		t.Data[i] = 0.2 * float32(math.Sin(float64(i)*0.11+0.3))
	}
	return t
}

func rnnMakeTarget(net *poly.VolumetricNetwork, input *poly.Tensor[float32]) *poly.Tensor[float32] {
	out, _, _ := poly.ForwardPolymorphic(net, input)
	tgt := poly.NewTensor[float32](out.Shape...)
	for i := range tgt.Data {
		tgt.Data[i] = 0.5 + 0.3*float32(math.Sin(float64(i)*0.17))
	}
	return tgt
}

// ── Morph all layers to numerical type ─────────────────────────────────────────

func rnnApplyDType(net *poly.VolumetricNetwork, tc rnnDtypeCase) {
	for i := range net.Layers {
		l := &net.Layers[i]
		l.DType = tc.dtype
		if l.WeightStore == nil {
			continue
		}
		l.WeightStore.InvalidateVersions()
		if tc.scale != 1.0 {
			l.WeightStore.Scale = tc.scale
		}
		l.WeightStore.Morph(tc.dtype)
		l.SyncToCPU()
	}
}

// ── Step 3: Sync to GPU ──────────────────────────────────────────────────────

func rnnSyncGPU(net *poly.VolumetricNetwork) error {
	return poly.ConfigureNetworkForMode(net, rnnGPUMode)
}

// ── Save / reload + deviation bucket ─────────────────────────────────────────

type rnnSavePhase string

const (
	rnnPhaseBefore rnnSavePhase = "BEFORE_TRAIN"
	rnnPhaseAfter  rnnSavePhase = "AFTER_TRAIN"
)

type rnnSaveResult struct {
	phase            rnnSavePhase
	forwardDiff      float64
	weightDiff       float64
	lossDelta        float64
	bucket           rnnSpectrum
	nativeOK         bool
	pass             bool
	err              string
}

func rnnCheckSaveReload(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32], tc rnnDtypeCase, phase rnnSavePhase, refLoss float64) rnnSaveResult {
	r := rnnSaveResult{phase: phase}
	out0, _, _ := poly.ForwardPolymorphic(net, input)
	baseline := append([]float32(nil), out0.Data...)

	wire, err := poly.SerializeNetwork(net)
	if err != nil {
		r.err = err.Error()
		r.bucket = rnnSpecFatal
		return r
	}
	reloaded, err := poly.DeserializeNetwork(wire)
	if err != nil {
		r.err = err.Error()
		r.bucket = rnnSpecFatal
		return r
	}
	if err := poly.ConfigureNetworkForMode(reloaded, rnnGPUMode); err != nil {
		r.err = err.Error()
		r.bucket = rnnSpecFatal
		return r
	}
	out1, _, _ := poly.ForwardPolymorphic(reloaded, input)
	r.forwardDiff = rnnMaxAbsDiff(baseline, out1.Data)
	r.bucket = rnnSpectrumMark(r.forwardDiff, tc.tolerance, out1.Data, baseline)
	r.weightDiff = rnnMaxWeightDiff(net, reloaded)
	outR, _, _ := poly.ForwardPolymorphic(reloaded, input)
	r.lossDelta = math.Abs(poly.CalculateLoss(outR, target, "mse") - refLoss)

	r.nativeOK = true
	if net.Layers[0].WeightStore != nil {
		b64, scale, native, fileErr := poly.LayerPersistenceFromJSON(wire, 0)
		l2 := reloaded.GetLayer(0, 0, 0, 0)
		if fileErr != nil || !native || b64 == "" || l2 == nil || l2.WeightStore == nil {
			r.nativeOK = false
		} else {
			decoded, decErr := poly.DecodeNativeWeights(b64, l2.DType)
			loaded := l2.WeightStore.Versions[tc.dtype]
			r.nativeOK = decErr == nil && loaded != nil && l2.WeightStore.Scale == scale &&
				poly.NativeWeightsEncoded(decoded, loaded, tc.dtype)
		}
	}
	r.pass = r.forwardDiff <= tc.tolerance && r.weightDiff <= tc.tolerance*10 &&
		r.bucket <= rnnSpecLowBit && r.nativeOK && r.err == ""
	return r
}

// ── Step 4: Train on GPU ─────────────────────────────────────────────────────

func rnnTrain(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32]) (*poly.TrainingResult, error) {
	return poly.Train(net, []poly.TrainingBatch[float32]{{Input: input, Target: target}}, &poly.TrainingConfig{
		Epochs:       rnnTrainEpochs,
		LearningRate: rnnLearningRate,
		Mode:         rnnGPUMode,
		Verbose:      false,
		LossType:     "mse",
	})
}

// ── Step 5: Save checkpoint to disk ────────────────────────────────────────────

func rnnSaveCheckpoint(net *poly.VolumetricNetwork, dtypeName string) string {
	wire, err := poly.SerializeNetwork(net)
	if err != nil {
		return ""
	}
	_ = os.MkdirAll(rnnOutputDir, 0o755)
	path := filepath.Join(rnnOutputDir, "rnn_"+dtypeName+".json")
	_ = os.WriteFile(path, wire, 0o644)
	return path
}

// ── Run all numerical types ────────────────────────────────────────────────────

func runRNNExample() bool {
	fmt.Println()
	fmt.Println("══════════════════════════════════════════════════════════════════════")
	fmt.Println("  Loom five-layer RNN — JSON · GPU · train · save · reload")
	fmt.Println("  5 flat layers in layers_per_cell (no SEQUENTIAL)")
	fmt.Println("══════════════════════════════════════════════════════════════════════")
	fmt.Printf("  Running %d numerical types (%d epochs GPU each, quiet)…\n", len(rnnAllDTypes), rnnTrainEpochs)

	var rows []DTypeRow
	passed, failed := 0, 0

	for _, tc := range rnnAllDTypes {
		fmt.Printf("  · %-10s ", tc.name)
		row := DTypeRow{DType: tc.name}

		net, err := rnnCreateNetwork(tc.jsonName)
		if err != nil {
			row.Err = "BUILD"
			rows = append(rows, row)
			failed++
			fmt.Println("BUILD ERR")
			continue
		}
		rnnApplyDType(net, tc)
		input := rnnMakeInput()
		target := rnnMakeTarget(net, input)

		if err := rnnSyncGPU(net); err != nil {
			row.Err = "GPU"
			rows = append(rows, row)
			failed++
			fmt.Println("GPU ERR")
			continue
		}

		lossBefore := rnnForwardLoss(net, input, target)
		before := rnnCheckSaveReload(net, input, target, tc, rnnPhaseBefore, lossBefore)
		row.BeforeBucket = before.bucket.String()
		row.BeforeOK = before.pass
		row.NativeOK = before.nativeOK

		res, err := rnnTrain(net, input, target)
		if err != nil {
			row.Err = "TRAIN"
			rows = append(rows, row)
			failed++
			fmt.Println("TRAIN ERR")
			continue
		}
		_ = poly.SyncWeightsFromGPU(net)
		lossInit := res.LossHistory[0]
		lossAfter := res.FinalLoss
		if len(res.LossHistory) > 0 {
			lossAfter = res.LossHistory[len(res.LossHistory)-1]
		}
		row.LossInit = lossInit
		row.LossFinal = lossAfter

		_ = rnnSaveCheckpoint(net, tc.name)
		after := rnnCheckSaveReload(net, input, target, tc, rnnPhaseAfter, lossAfter)
		row.AfterBucket = after.bucket.String()
		row.AfterOK = after.pass
		if !after.nativeOK {
			row.NativeOK = false
		}

		row.Learned = rnnTrainingOK(lossInit, lossAfter, tc.dtype)
		row.OverallOK = row.BeforeOK && row.AfterOK && row.Learned
		rows = append(rows, row)

		if row.OverallOK {
			passed++
			fmt.Printf("PASS  loss %.4e→%.4e\n", lossInit, lossAfter)
		} else {
			failed++
			fmt.Printf("FAIL  loss %.4e→%.4e learn=%s save=%s\n",
				lossInit, lossAfter, markOK(row.Learned), markOK(row.BeforeOK && row.AfterOK))
		}
	}

	PrintDTypeResultsTable("RNN", rows)
	RegisterLayerSummary("RNN", passed, failed, rows)
	return failed == 0
}

// RunRNN is called from Lucy menu [6].
func RunRNN() bool { return runRNNExample() }

// ── helpers ────────────────────────────────────────────────────────────────────

func rnnForwardLoss(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32]) float64 {
	out, _, _ := poly.ForwardPolymorphic(net, input)
	return poly.CalculateLoss(out, target, "mse")
}

func rnnPrintSaveRow(dtype string, r rnnSaveResult) {
	if r.err != "" {
		fmt.Printf("| %-10s | %-13s | ERR      |          | 💀 FATAL     |            | FAIL    |\n", dtype, r.phase)
		return
	}
	fmt.Printf("| %-10s | %-13s | %-8.2e | %-8.2e | %-12s | %-12s | %-7s |\n",
		dtype, r.phase, r.forwardDiff, r.weightDiff, r.bucket.String(), rnnMark(r.nativeOK), rnnMark(r.pass))
}

func rnnSpectrumMark(diff, tol float64, actual, baseline []float32) rnnSpectrum {
	if math.IsNaN(diff) || (math.IsInf(diff, 0) && !rnnHasSignal(baseline)) {
		return rnnSpecFatal
	}
	if math.IsInf(diff, 0) {
		return rnnSpecHeavyDrift
	}
	as, bs := rnnHasSignal(actual), rnnHasSignal(baseline)
	if !as && !bs {
		return rnnSpecExact
	}
	if !as || !bs {
		return rnnSpecHeavyDrift
	}
	if diff == 0 {
		return rnnSpecExact
	}
	if diff <= tol {
		return rnnSpecIndustry
	}
	if diff <= tol*10 {
		return rnnSpecLowBit
	}
	if diff <= 0.1 {
		return rnnSpecDrift
	}
	return rnnSpecHeavyDrift
}

func rnnHasSignal(d []float32) bool {
	for _, v := range d {
		if v != 0 && !math.IsNaN(float64(v)) {
			return true
		}
	}
	return false
}

func rnnMaxAbsDiff(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var m float64
	for i := 0; i < n; i++ {
		if v := math.Abs(float64(a[i] - b[i])); v > m {
			m = v
		}
	}
	return m
}

func rnnMaxWeightDiff(a, b *poly.VolumetricNetwork) float64 {
	var m float64
	for i := range a.Layers {
		wa, wb := a.Layers[i].WeightStore, b.Layers[i].WeightStore
		if wa != nil && wb != nil {
			if d := rnnMaxAbsDiff(wa.Master, wb.Master); d > m {
				m = d
			}
		}
	}
	return m
}

func rnnTrainingOK(init, final float64, dt poly.DType) bool {
	if math.IsNaN(init) || math.IsNaN(final) || math.IsInf(init, 0) || math.IsInf(final, 0) {
		return false
	}
	if init < 0.01 {
		return final <= init*2+1e-3
	}
	return final < init*0.99
}

func rnnMark(ok bool) string {
	if ok {
		return "PASS"
	}
	return "FAIL"
}
