// Five-layer SwiGLU example for Loom (poly).
//
// STANDALONE (copy into your repo):
//   1. Copy this file; in go.mod: require github.com/openfluke/loom (or your poly path)
//   2. Change `package fivelayer` → `package main`
//   3. Rename `RunSwiglu` → `main`  (or call runSwigluExample() from main)
//   4. go run swiglu.go
//
// Lucy menu [6] calls RunSwiglu(); output is teed to lucy_testing_output/five_layer.txt
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
	swigluTrainEpochs  = 50
	swigluLearningRate = float32(0.01)
	swigluGPUMode      = poly.TrainingModeGPUSC
	swigluOutputDir    = "lucy_testing_output"
)

// ── Numerical types (all 21) ─────────────────────────────────────────────────

type swigluDtypeCase struct {
	name, jsonName string
	dtype          poly.DType
	scale          float32
	tolerance      float64
}

var swigluAllDTypes = []swigluDtypeCase{
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

type swigluSpectrum int

const (
	swigluSpecExact swigluSpectrum = iota
	swigluSpecIndustry
	swigluSpecLowBit
	swigluSpecDrift
	swigluSpecHeavyDrift
	swigluSpecBroken
	swigluSpecFatal
)

func (s swigluSpectrum) String() string {
	switch s {
	case swigluSpecExact:
		return "💎 EXACT"
	case swigluSpecIndustry:
		return "✅ INDUS"
	case swigluSpecLowBit:
		return "🟨 LOWBIT"
	case swigluSpecDrift:
		return "🟠 DRIFT"
	case swigluSpecHeavyDrift:
		return "🟤 H-DRIFT"
	case swigluSpecBroken:
		return "❌ BROKE"
	default:
		return "💀 FATAL"
	}
}

// ── Step 1: Build JSON network spec (5 flat SWIGLU layers, no SEQUENTIAL) ───

func swigluNetworkJSON(jsonDType string) []byte {
	dims := []int{64, 64, 64, 64, 64, 32}
	var b strings.Builder
	b.WriteString(fmt.Sprintf(
		`{"id":"loom-five-swiglu","depth":1,"rows":1,"cols":1,"layers_per_cell":%d,"layers":[`,
		len(dims)-1,
	))
	for i := 0; i < len(dims)-1; i++ {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(fmt.Sprintf(
			`{"z":0,"y":0,"x":0,"l":%d,"type":"SWIGLU","activation":"RELU","dtype":"%s","input_height":%d,"output_height":%d}`,
			i, jsonDType, dims[i], dims[i+1],
		))
	}
	b.WriteString(`]}`)
	return []byte(b.String())
}

// ── Step 2: Create network from JSON ─────────────────────────────────────────

func swigluCreateNetwork(jsonDType string) (*poly.VolumetricNetwork, error) {
	return poly.BuildNetworkFromJSON(swigluNetworkJSON(jsonDType))
}

// ── Input / target tensors ───────────────────────────────────────────────────

func swigluMakeInput() *poly.Tensor[float32] {
	t := poly.NewTensor[float32](8, 64)
	for i := range t.Data {
		t.Data[i] = 0.2 * float32(math.Sin(float64(i)*0.11+0.3))
	}
	return t
}

func swigluMakeTarget(net *poly.VolumetricNetwork, input *poly.Tensor[float32]) *poly.Tensor[float32] {
	out, _, _ := poly.ForwardPolymorphic(net, input)
	tgt := poly.NewTensor[float32](out.Shape...)
	for i := range tgt.Data {
		tgt.Data[i] = 0.5 + 0.3*float32(math.Sin(float64(i)*0.17))
	}
	return tgt
}

// ── Morph all layers to numerical type ─────────────────────────────────────────

func swigluApplyDType(net *poly.VolumetricNetwork, tc swigluDtypeCase) {
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

func swigluSyncGPU(net *poly.VolumetricNetwork) error {
	return poly.ConfigureNetworkForMode(net, swigluGPUMode)
}

// ── Save / reload + deviation bucket ─────────────────────────────────────────

type swigluSavePhase string

const (
	swigluPhaseBefore swigluSavePhase = "BEFORE_TRAIN"
	swigluPhaseAfter  swigluSavePhase = "AFTER_TRAIN"
)

type swigluSaveResult struct {
	phase            swigluSavePhase
	forwardDiff      float64
	weightDiff       float64
	lossDelta        float64
	bucket           swigluSpectrum
	nativeOK         bool
	pass             bool
	err              string
}

func swigluCheckSaveReload(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32], tc swigluDtypeCase, phase swigluSavePhase, refLoss float64) swigluSaveResult {
	r := swigluSaveResult{phase: phase}
	out0, _, _ := poly.ForwardPolymorphic(net, input)
	baseline := append([]float32(nil), out0.Data...)

	wire, err := poly.SerializeNetwork(net)
	if err != nil {
		r.err = err.Error()
		r.bucket = swigluSpecFatal
		return r
	}
	reloaded, err := poly.DeserializeNetwork(wire)
	if err != nil {
		r.err = err.Error()
		r.bucket = swigluSpecFatal
		return r
	}
	if err := poly.ConfigureNetworkForMode(reloaded, swigluGPUMode); err != nil {
		r.err = err.Error()
		r.bucket = swigluSpecFatal
		return r
	}
	out1, _, _ := poly.ForwardPolymorphic(reloaded, input)
	r.forwardDiff = swigluMaxAbsDiff(baseline, out1.Data)
	r.bucket = swigluSpectrumMark(r.forwardDiff, tc.tolerance, out1.Data, baseline)
	r.weightDiff = swigluMaxWeightDiff(net, reloaded)
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
		r.bucket <= swigluSpecLowBit && r.nativeOK && r.err == ""
	return r
}

// ── Step 4: Train on GPU ─────────────────────────────────────────────────────

func swigluTrain(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32]) (*poly.TrainingResult, error) {
	return poly.Train(net, []poly.TrainingBatch[float32]{{Input: input, Target: target}}, &poly.TrainingConfig{
		Epochs:       swigluTrainEpochs,
		LearningRate: swigluLearningRate,
		Mode:         swigluGPUMode,
		Verbose:      false,
		LossType:     "mse",
	})
}

// ── Step 5: Save checkpoint to disk ────────────────────────────────────────────

func swigluSaveCheckpoint(net *poly.VolumetricNetwork, dtypeName string) string {
	wire, err := poly.SerializeNetwork(net)
	if err != nil {
		return ""
	}
	_ = os.MkdirAll(swigluOutputDir, 0o755)
	path := filepath.Join(swigluOutputDir, "swiglu_"+dtypeName+".json")
	_ = os.WriteFile(path, wire, 0o644)
	return path
}

// ── Run all numerical types ────────────────────────────────────────────────────

func runSwigluExample() bool {
	fmt.Println()
	fmt.Println("══════════════════════════════════════════════════════════════════════")
	fmt.Println("  Loom five-layer SwiGLU — JSON · GPU · train · save · reload")
	fmt.Println("  5 flat layers in layers_per_cell (no SEQUENTIAL)")
	fmt.Println("══════════════════════════════════════════════════════════════════════")
	fmt.Printf("  Running %d numerical types (%d epochs GPU each, quiet)…\n", len(swigluAllDTypes), swigluTrainEpochs)

	var rows []DTypeRow
	passed, failed := 0, 0

	for _, tc := range swigluAllDTypes {
		fmt.Printf("  · %-10s ", tc.name)
		row := DTypeRow{DType: tc.name}

		net, err := swigluCreateNetwork(tc.jsonName)
		if err != nil {
			row.Err = "BUILD"
			rows = append(rows, row)
			failed++
			fmt.Println("BUILD ERR")
			continue
		}
		swigluApplyDType(net, tc)
		input := swigluMakeInput()
		target := swigluMakeTarget(net, input)

		if err := swigluSyncGPU(net); err != nil {
			row.Err = "GPU"
			rows = append(rows, row)
			failed++
			fmt.Println("GPU ERR")
			continue
		}

		lossBefore := swigluForwardLoss(net, input, target)
		before := swigluCheckSaveReload(net, input, target, tc, swigluPhaseBefore, lossBefore)
		row.BeforeBucket = before.bucket.String()
		row.BeforeOK = before.pass
		row.NativeOK = before.nativeOK

		res, err := swigluTrain(net, input, target)
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

		_ = swigluSaveCheckpoint(net, tc.name)
		after := swigluCheckSaveReload(net, input, target, tc, swigluPhaseAfter, lossAfter)
		row.AfterBucket = after.bucket.String()
		row.AfterOK = after.pass
		if !after.nativeOK {
			row.NativeOK = false
		}

		row.Learned = swigluTrainingOK(lossInit, lossAfter, tc.dtype)
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

	PrintDTypeResultsTable("SwiGLU", rows)
	RegisterLayerSummary("SwiGLU", passed, failed, rows)
	return failed == 0
}

// RunSwiglu is called from Lucy menu [6].
func RunSwiglu() bool { return runSwigluExample() }

// RunSwiGLU is an alias for the Lucy menu.
func RunSwiGLU() bool { return runSwigluExample() }

// ── helpers ────────────────────────────────────────────────────────────────────

func swigluForwardLoss(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32]) float64 {
	out, _, _ := poly.ForwardPolymorphic(net, input)
	return poly.CalculateLoss(out, target, "mse")
}

func swigluPrintSaveRow(dtype string, r swigluSaveResult) {
	if r.err != "" {
		fmt.Printf("| %-10s | %-13s | ERR      |          | 💀 FATAL     |            | FAIL    |\n", dtype, r.phase)
		return
	}
	fmt.Printf("| %-10s | %-13s | %-8.2e | %-8.2e | %-12s | %-12s | %-7s |\n",
		dtype, r.phase, r.forwardDiff, r.weightDiff, r.bucket.String(), swigluMark(r.nativeOK), swigluMark(r.pass))
}

func swigluSpectrumMark(diff, tol float64, actual, baseline []float32) swigluSpectrum {
	if math.IsNaN(diff) || (math.IsInf(diff, 0) && !swigluHasSignal(baseline)) {
		return swigluSpecFatal
	}
	if math.IsInf(diff, 0) {
		return swigluSpecHeavyDrift
	}
	as, bs := swigluHasSignal(actual), swigluHasSignal(baseline)
	if !as && !bs {
		return swigluSpecExact
	}
	if !as || !bs {
		return swigluSpecHeavyDrift
	}
	if diff == 0 {
		return swigluSpecExact
	}
	if diff <= tol {
		return swigluSpecIndustry
	}
	if diff <= tol*10 {
		return swigluSpecLowBit
	}
	if diff <= 0.1 {
		return swigluSpecDrift
	}
	return swigluSpecHeavyDrift
}

func swigluHasSignal(d []float32) bool {
	for _, v := range d {
		if v != 0 && !math.IsNaN(float64(v)) {
			return true
		}
	}
	return false
}

func swigluMaxAbsDiff(a, b []float32) float64 {
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

func swigluMaxWeightDiff(a, b *poly.VolumetricNetwork) float64 {
	var m float64
	for i := range a.Layers {
		wa, wb := a.Layers[i].WeightStore, b.Layers[i].WeightStore
		if wa != nil && wb != nil {
			if d := swigluMaxAbsDiff(wa.Master, wb.Master); d > m {
				m = d
			}
		}
	}
	return m
}

func swigluTrainingOK(init, final float64, dt poly.DType) bool {
	if math.IsNaN(init) || math.IsNaN(final) || math.IsInf(init, 0) || math.IsInf(final, 0) {
		return false
	}
	if init < 0.01 {
		return final <= init*2+1e-3
	}
	return final < init*0.99
}

func swigluMark(ok bool) string {
	if ok {
		return "PASS"
	}
	return "FAIL"
}
