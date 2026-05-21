// Five-layer CNN3 example for Loom (poly).
//
// STANDALONE (copy into your repo):
//   1. Copy this file; in go.mod: require github.com/openfluke/loom (or your poly path)
//   2. Change `package fivelayer` → `package main`
//   3. Rename `RunCNN3` → `main`  (or call runCNN3Example() from main)
//   4. go run cnn3.go
//
// Lucy menu [6] calls RunCNN3(); output is teed to lucy_testing_output/five_layer.txt
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
	cnn3TrainEpochs  = 50
	cnn3LearningRate = float32(0.01)
	cnn3GPUMode      = poly.TrainingModeGPUSC
	cnn3OutputDir    = "lucy_testing_output"
)

// ── Numerical types (all 21) ─────────────────────────────────────────────────

type cnn3DtypeCase struct {
	name, jsonName string
	dtype          poly.DType
	scale          float32
	tolerance      float64
}

var cnn3AllDTypes = []cnn3DtypeCase{
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

type cnn3Spectrum int

const (
	cnn3SpecExact cnn3Spectrum = iota
	cnn3SpecIndustry
	cnn3SpecLowBit
	cnn3SpecDrift
	cnn3SpecHeavyDrift
	cnn3SpecBroken
	cnn3SpecFatal
)

func (s cnn3Spectrum) String() string {
	switch s {
	case cnn3SpecExact:
		return "💎 EXACT"
	case cnn3SpecIndustry:
		return "✅ INDUS"
	case cnn3SpecLowBit:
		return "🟨 LOWBIT"
	case cnn3SpecDrift:
		return "🟠 DRIFT"
	case cnn3SpecHeavyDrift:
		return "🟤 H-DRIFT"
	case cnn3SpecBroken:
		return "❌ BROKE"
	default:
		return "💀 FATAL"
	}
}

// ── Step 1: Build JSON network spec (5 flat CNN3 layers, no SEQUENTIAL) ───

func cnn3NetworkJSON(jsonDType string) []byte {
	ch := []int{2, 4, 4, 8, 8, 8}
	var b strings.Builder
	b.WriteString(fmt.Sprintf(
		`{"id":"loom-five-cnn3","depth":1,"rows":1,"cols":1,"layers_per_cell":%d,"layers":[`,
		len(ch)-1,
	))
	for i := 0; i < len(ch)-1; i++ {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(fmt.Sprintf(
			`{"z":0,"y":0,"x":0,"l":%d,"type":"CNN3","activation":"RELU","dtype":"%s","input_channels":%d,"filters":%d,"input_height":%d,"output_height":%d,"input_depth":%d,"input_height":%d,"input_width":%d,"output_depth":%d,"output_height":%d,"output_width":%d,"kernel_size":3,"stride":1,"padding":1}`,
			i, jsonDType, ch[i], ch[i+1], 16, 16, 16, 16, 16, 16, 16, 16,
		))
	}
	b.WriteString(`]}`)
	return []byte(b.String())
}

// ── Step 2: Create network from JSON ─────────────────────────────────────────

func cnn3CreateNetwork(jsonDType string) (*poly.VolumetricNetwork, error) {
	return poly.BuildNetworkFromJSON(cnn3NetworkJSON(jsonDType))
}

// ── Input / target tensors ───────────────────────────────────────────────────

func cnn3MakeInput() *poly.Tensor[float32] {
	t := poly.NewTensor[float32](2, 2, 16, 16, 16)
	for i := range t.Data {
		t.Data[i] = 0.2 * float32(math.Sin(float64(i)*0.11+0.3))
	}
	return t
}

func cnn3MakeTarget(net *poly.VolumetricNetwork, input *poly.Tensor[float32]) *poly.Tensor[float32] {
	out, _, _ := poly.ForwardPolymorphic(net, input)
	tgt := poly.NewTensor[float32](out.Shape...)
	for i := range tgt.Data {
		tgt.Data[i] = 0.5 + 0.3*float32(math.Sin(float64(i)*0.17))
	}
	return tgt
}

// ── Morph all layers to numerical type ─────────────────────────────────────────

func cnn3ApplyDType(net *poly.VolumetricNetwork, tc cnn3DtypeCase) {
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

func cnn3SyncGPU(net *poly.VolumetricNetwork) error {
	return poly.ConfigureNetworkForMode(net, cnn3GPUMode)
}

// ── Save / reload + deviation bucket ─────────────────────────────────────────

type cnn3SavePhase string

const (
	cnn3PhaseBefore cnn3SavePhase = "BEFORE_TRAIN"
	cnn3PhaseAfter  cnn3SavePhase = "AFTER_TRAIN"
)

type cnn3SaveResult struct {
	phase            cnn3SavePhase
	forwardDiff      float64
	weightDiff       float64
	lossDelta        float64
	bucket           cnn3Spectrum
	nativeOK         bool
	pass             bool
	err              string
}

func cnn3CheckSaveReload(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32], tc cnn3DtypeCase, phase cnn3SavePhase, refLoss float64) cnn3SaveResult {
	r := cnn3SaveResult{phase: phase}
	out0, _, _ := poly.ForwardPolymorphic(net, input)
	baseline := append([]float32(nil), out0.Data...)

	wire, err := poly.SerializeNetwork(net)
	if err != nil {
		r.err = err.Error()
		r.bucket = cnn3SpecFatal
		return r
	}
	reloaded, err := poly.DeserializeNetwork(wire)
	if err != nil {
		r.err = err.Error()
		r.bucket = cnn3SpecFatal
		return r
	}
	if err := poly.ConfigureNetworkForMode(reloaded, cnn3GPUMode); err != nil {
		r.err = err.Error()
		r.bucket = cnn3SpecFatal
		return r
	}
	out1, _, _ := poly.ForwardPolymorphic(reloaded, input)
	r.forwardDiff = cnn3MaxAbsDiff(baseline, out1.Data)
	r.bucket = cnn3SpectrumMark(r.forwardDiff, tc.tolerance, out1.Data, baseline)
	r.weightDiff = cnn3MaxWeightDiff(net, reloaded)
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
		r.bucket <= cnn3SpecLowBit && r.nativeOK && r.err == ""
	return r
}

// ── Step 4: Train on GPU ─────────────────────────────────────────────────────

func cnn3Train(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32]) (*poly.TrainingResult, error) {
	return poly.Train(net, []poly.TrainingBatch[float32]{{Input: input, Target: target}}, &poly.TrainingConfig{
		Epochs:       cnn3TrainEpochs,
		LearningRate: cnn3LearningRate,
		Mode:         cnn3GPUMode,
		Verbose:      false,
		LossType:     "mse",
	})
}

// ── Step 5: Save checkpoint to disk ────────────────────────────────────────────

func cnn3SaveCheckpoint(net *poly.VolumetricNetwork, dtypeName string) string {
	wire, err := poly.SerializeNetwork(net)
	if err != nil {
		return ""
	}
	_ = os.MkdirAll(cnn3OutputDir, 0o755)
	path := filepath.Join(cnn3OutputDir, "cnn3_"+dtypeName+".json")
	_ = os.WriteFile(path, wire, 0o644)
	return path
}

// ── Run all numerical types ────────────────────────────────────────────────────

func runCNN3Example() bool {
	fmt.Println()
	fmt.Println("══════════════════════════════════════════════════════════════════════")
	fmt.Println("  Loom five-layer CNN3 — JSON · GPU · train · save · reload")
	fmt.Println("  5 flat layers in layers_per_cell (no SEQUENTIAL)")
	fmt.Println("══════════════════════════════════════════════════════════════════════")
	fmt.Printf("  Running %d numerical types (%d epochs GPU each, quiet)…\n", len(cnn3AllDTypes), cnn3TrainEpochs)

	var rows []DTypeRow
	passed, failed := 0, 0

	for _, tc := range cnn3AllDTypes {
		fmt.Printf("  · %-10s ", tc.name)
		row := DTypeRow{DType: tc.name}

		net, err := cnn3CreateNetwork(tc.jsonName)
		if err != nil {
			row.Err = "BUILD"
			rows = append(rows, row)
			failed++
			fmt.Println("BUILD ERR")
			continue
		}
		cnn3ApplyDType(net, tc)
		input := cnn3MakeInput()
		target := cnn3MakeTarget(net, input)

		if err := cnn3SyncGPU(net); err != nil {
			row.Err = "GPU"
			rows = append(rows, row)
			failed++
			fmt.Println("GPU ERR")
			continue
		}

		lossBefore := cnn3ForwardLoss(net, input, target)
		before := cnn3CheckSaveReload(net, input, target, tc, cnn3PhaseBefore, lossBefore)
		row.BeforeBucket = before.bucket.String()
		row.BeforeOK = before.pass
		row.NativeOK = before.nativeOK

		res, err := cnn3Train(net, input, target)
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

		_ = cnn3SaveCheckpoint(net, tc.name)
		after := cnn3CheckSaveReload(net, input, target, tc, cnn3PhaseAfter, lossAfter)
		row.AfterBucket = after.bucket.String()
		row.AfterOK = after.pass
		if !after.nativeOK {
			row.NativeOK = false
		}

		row.Learned = cnn3TrainingOK(lossInit, lossAfter, tc.dtype)
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

	PrintDTypeResultsTable("CNN3", rows)
	RegisterLayerSummary("CNN3", passed, failed, rows)
	return failed == 0
}

// RunCNN3 is called from Lucy menu [6].
func RunCNN3() bool { return runCNN3Example() }

// ── helpers ────────────────────────────────────────────────────────────────────

func cnn3ForwardLoss(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32]) float64 {
	out, _, _ := poly.ForwardPolymorphic(net, input)
	return poly.CalculateLoss(out, target, "mse")
}

func cnn3PrintSaveRow(dtype string, r cnn3SaveResult) {
	if r.err != "" {
		fmt.Printf("| %-10s | %-13s | ERR      |          | 💀 FATAL     |            | FAIL    |\n", dtype, r.phase)
		return
	}
	fmt.Printf("| %-10s | %-13s | %-8.2e | %-8.2e | %-12s | %-12s | %-7s |\n",
		dtype, r.phase, r.forwardDiff, r.weightDiff, r.bucket.String(), cnn3Mark(r.nativeOK), cnn3Mark(r.pass))
}

func cnn3SpectrumMark(diff, tol float64, actual, baseline []float32) cnn3Spectrum {
	if math.IsNaN(diff) || (math.IsInf(diff, 0) && !cnn3HasSignal(baseline)) {
		return cnn3SpecFatal
	}
	if math.IsInf(diff, 0) {
		return cnn3SpecHeavyDrift
	}
	as, bs := cnn3HasSignal(actual), cnn3HasSignal(baseline)
	if !as && !bs {
		return cnn3SpecExact
	}
	if !as || !bs {
		return cnn3SpecHeavyDrift
	}
	if diff == 0 {
		return cnn3SpecExact
	}
	if diff <= tol {
		return cnn3SpecIndustry
	}
	if diff <= tol*10 {
		return cnn3SpecLowBit
	}
	if diff <= 0.1 {
		return cnn3SpecDrift
	}
	return cnn3SpecHeavyDrift
}

func cnn3HasSignal(d []float32) bool {
	for _, v := range d {
		if v != 0 && !math.IsNaN(float64(v)) {
			return true
		}
	}
	return false
}

func cnn3MaxAbsDiff(a, b []float32) float64 {
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

func cnn3MaxWeightDiff(a, b *poly.VolumetricNetwork) float64 {
	var m float64
	for i := range a.Layers {
		wa, wb := a.Layers[i].WeightStore, b.Layers[i].WeightStore
		if wa != nil && wb != nil {
			if d := cnn3MaxAbsDiff(wa.Master, wb.Master); d > m {
				m = d
			}
		}
	}
	return m
}

func cnn3TrainingOK(init, final float64, dt poly.DType) bool {
	if math.IsNaN(init) || math.IsNaN(final) || math.IsInf(init, 0) || math.IsInf(final, 0) {
		return false
	}
	if init < 0.01 {
		return final <= init*2+1e-3
	}
	return final < init*0.99
}

func cnn3Mark(ok bool) string {
	if ok {
		return "PASS"
	}
	return "FAIL"
}
