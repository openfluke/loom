// Five-layer CNN2 example for Loom (poly).
//
// STANDALONE (copy into your repo):
//   1. Copy this file; in go.mod: require github.com/openfluke/loom (or your poly path)
//   2. Change `package fivelayer` → `package main`
//   3. Rename `RunCNN2` → `main`  (or call runCNN2Example() from main)
//   4. go run cnn2.go
//
// Lucy menu [6] calls RunCNN2(); output is teed to lucy_testing_output/five_layer.txt
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
	cnn2TrainEpochs  = 50
	cnn2LearningRate = float32(0.01)
	cnn2GPUMode      = poly.TrainingModeGPUSC
	cnn2OutputDir    = "lucy_testing_output"
)

// ── Numerical types (all 21) ─────────────────────────────────────────────────

type cnn2DtypeCase struct {
	name, jsonName string
	dtype          poly.DType
	scale          float32
	tolerance      float64
}

var cnn2AllDTypes = []cnn2DtypeCase{
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

type cnn2Spectrum int

const (
	cnn2SpecExact cnn2Spectrum = iota
	cnn2SpecIndustry
	cnn2SpecLowBit
	cnn2SpecDrift
	cnn2SpecHeavyDrift
	cnn2SpecBroken
	cnn2SpecFatal
)

func (s cnn2Spectrum) String() string {
	switch s {
	case cnn2SpecExact:
		return "💎 EXACT"
	case cnn2SpecIndustry:
		return "✅ INDUS"
	case cnn2SpecLowBit:
		return "🟨 LOWBIT"
	case cnn2SpecDrift:
		return "🟠 DRIFT"
	case cnn2SpecHeavyDrift:
		return "🟤 H-DRIFT"
	case cnn2SpecBroken:
		return "❌ BROKE"
	default:
		return "💀 FATAL"
	}
}

// ── Step 1: Build JSON network spec (5 flat CNN2 layers, no SEQUENTIAL) ───

func cnn2NetworkJSON(jsonDType string) []byte {
	ch := []int{3, 8, 8, 16, 16, 16}
	var b strings.Builder
	b.WriteString(fmt.Sprintf(
		`{"id":"loom-five-cnn2","depth":1,"rows":1,"cols":1,"layers_per_cell":%d,"layers":[`,
		len(ch)-1,
	))
	for i := 0; i < len(ch)-1; i++ {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(fmt.Sprintf(
			`{"z":0,"y":0,"x":0,"l":%d,"type":"CNN2","activation":"RELU","dtype":"%s","input_channels":%d,"filters":%d,"input_height":%d,"output_height":%d,"input_width":%d,"output_width":%d,"kernel_size":3,"stride":1,"padding":1}`,
			i, jsonDType, ch[i], ch[i+1], 32, 32, 32, 32,
		))
	}
	b.WriteString(`]}`)
	return []byte(b.String())
}

// ── Step 2: Create network from JSON ─────────────────────────────────────────

func cnn2CreateNetwork(jsonDType string) (*poly.VolumetricNetwork, error) {
	return poly.BuildNetworkFromJSON(cnn2NetworkJSON(jsonDType))
}

// ── Input / target tensors ───────────────────────────────────────────────────

func cnn2MakeInput() *poly.Tensor[float32] {
	t := poly.NewTensor[float32](4, 3, 32, 32)
	for i := range t.Data {
		t.Data[i] = 0.2 * float32(math.Sin(float64(i)*0.11+0.3))
	}
	return t
}

func cnn2MakeTarget(net *poly.VolumetricNetwork, input *poly.Tensor[float32]) *poly.Tensor[float32] {
	out, _, _ := poly.ForwardPolymorphic(net, input)
	tgt := poly.NewTensor[float32](out.Shape...)
	for i := range tgt.Data {
		tgt.Data[i] = 0.5 + 0.3*float32(math.Sin(float64(i)*0.17))
	}
	return tgt
}

// ── Morph all layers to numerical type ─────────────────────────────────────────

func cnn2ApplyDType(net *poly.VolumetricNetwork, tc cnn2DtypeCase) {
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

func cnn2SyncGPU(net *poly.VolumetricNetwork) error {
	return poly.ConfigureNetworkForMode(net, cnn2GPUMode)
}

// ── Save / reload + deviation bucket ─────────────────────────────────────────

type cnn2SavePhase string

const (
	cnn2PhaseBefore cnn2SavePhase = "BEFORE_TRAIN"
	cnn2PhaseAfter  cnn2SavePhase = "AFTER_TRAIN"
)

type cnn2SaveResult struct {
	phase            cnn2SavePhase
	forwardDiff      float64
	weightDiff       float64
	lossDelta        float64
	bucket           cnn2Spectrum
	nativeOK         bool
	pass             bool
	err              string
}

func cnn2CheckSaveReload(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32], tc cnn2DtypeCase, phase cnn2SavePhase, refLoss float64) cnn2SaveResult {
	r := cnn2SaveResult{phase: phase}
	out0, _, _ := poly.ForwardPolymorphic(net, input)
	baseline := append([]float32(nil), out0.Data...)

	wire, err := poly.SerializeNetwork(net)
	if err != nil {
		r.err = err.Error()
		r.bucket = cnn2SpecFatal
		return r
	}
	reloaded, err := poly.DeserializeNetwork(wire)
	if err != nil {
		r.err = err.Error()
		r.bucket = cnn2SpecFatal
		return r
	}
	if err := poly.ConfigureNetworkForMode(reloaded, cnn2GPUMode); err != nil {
		r.err = err.Error()
		r.bucket = cnn2SpecFatal
		return r
	}
	out1, _, _ := poly.ForwardPolymorphic(reloaded, input)
	r.forwardDiff = cnn2MaxAbsDiff(baseline, out1.Data)
	r.bucket = cnn2SpectrumMark(r.forwardDiff, tc.tolerance, out1.Data, baseline)
	r.weightDiff = cnn2MaxWeightDiff(net, reloaded)
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
		r.bucket <= cnn2SpecLowBit && r.nativeOK && r.err == ""
	return r
}

// ── Step 4: Train on GPU ─────────────────────────────────────────────────────

func cnn2Train(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32]) (*poly.TrainingResult, error) {
	return poly.Train(net, []poly.TrainingBatch[float32]{{Input: input, Target: target}}, &poly.TrainingConfig{
		Epochs:       cnn2TrainEpochs,
		LearningRate: cnn2LearningRate,
		Mode:         cnn2GPUMode,
		Verbose:      false,
		LossType:     "mse",
	})
}

// ── Step 5: Save checkpoint to disk ────────────────────────────────────────────

func cnn2SaveCheckpoint(net *poly.VolumetricNetwork, dtypeName string) string {
	wire, err := poly.SerializeNetwork(net)
	if err != nil {
		return ""
	}
	_ = os.MkdirAll(cnn2OutputDir, 0o755)
	path := filepath.Join(cnn2OutputDir, "cnn2_"+dtypeName+".json")
	_ = os.WriteFile(path, wire, 0o644)
	return path
}

// ── Run all numerical types ────────────────────────────────────────────────────

func runCNN2Example() bool {
	fmt.Println()
	fmt.Println("══════════════════════════════════════════════════════════════════════")
	fmt.Println("  Loom five-layer CNN2 — JSON · GPU · train · save · reload")
	fmt.Println("  5 flat layers in layers_per_cell (no SEQUENTIAL)")
	fmt.Println("══════════════════════════════════════════════════════════════════════")
	fmt.Printf("  Running %d numerical types (%d epochs GPU each, quiet)…\n", len(cnn2AllDTypes), cnn2TrainEpochs)

	var rows []DTypeRow
	passed, failed := 0, 0

	for _, tc := range cnn2AllDTypes {
		fmt.Printf("  · %-10s ", tc.name)
		row := DTypeRow{DType: tc.name}

		net, err := cnn2CreateNetwork(tc.jsonName)
		if err != nil {
			row.Err = "BUILD"
			rows = append(rows, row)
			failed++
			fmt.Println("BUILD ERR")
			continue
		}
		cnn2ApplyDType(net, tc)
		input := cnn2MakeInput()
		target := cnn2MakeTarget(net, input)

		if err := cnn2SyncGPU(net); err != nil {
			row.Err = "GPU"
			rows = append(rows, row)
			failed++
			fmt.Println("GPU ERR")
			continue
		}

		lossBefore := cnn2ForwardLoss(net, input, target)
		before := cnn2CheckSaveReload(net, input, target, tc, cnn2PhaseBefore, lossBefore)
		row.BeforeBucket = before.bucket.String()
		row.BeforeOK = before.pass
		row.NativeOK = before.nativeOK

		res, err := cnn2Train(net, input, target)
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

		_ = cnn2SaveCheckpoint(net, tc.name)
		after := cnn2CheckSaveReload(net, input, target, tc, cnn2PhaseAfter, lossAfter)
		row.AfterBucket = after.bucket.String()
		row.AfterOK = after.pass
		if !after.nativeOK {
			row.NativeOK = false
		}

		row.Learned = cnn2TrainingOK(lossInit, lossAfter, tc.dtype)
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

	PrintDTypeResultsTable("CNN2", rows)
	RegisterLayerSummary("CNN2", passed, failed, rows)
	return failed == 0
}

// RunCNN2 is called from Lucy menu [6].
func RunCNN2() bool { return runCNN2Example() }

// ── helpers ────────────────────────────────────────────────────────────────────

func cnn2ForwardLoss(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32]) float64 {
	out, _, _ := poly.ForwardPolymorphic(net, input)
	return poly.CalculateLoss(out, target, "mse")
}

func cnn2PrintSaveRow(dtype string, r cnn2SaveResult) {
	if r.err != "" {
		fmt.Printf("| %-10s | %-13s | ERR      |          | 💀 FATAL     |            | FAIL    |\n", dtype, r.phase)
		return
	}
	fmt.Printf("| %-10s | %-13s | %-8.2e | %-8.2e | %-12s | %-12s | %-7s |\n",
		dtype, r.phase, r.forwardDiff, r.weightDiff, r.bucket.String(), cnn2Mark(r.nativeOK), cnn2Mark(r.pass))
}

func cnn2SpectrumMark(diff, tol float64, actual, baseline []float32) cnn2Spectrum {
	if math.IsNaN(diff) || (math.IsInf(diff, 0) && !cnn2HasSignal(baseline)) {
		return cnn2SpecFatal
	}
	if math.IsInf(diff, 0) {
		return cnn2SpecHeavyDrift
	}
	as, bs := cnn2HasSignal(actual), cnn2HasSignal(baseline)
	if !as && !bs {
		return cnn2SpecExact
	}
	if !as || !bs {
		return cnn2SpecHeavyDrift
	}
	if diff == 0 {
		return cnn2SpecExact
	}
	if diff <= tol {
		return cnn2SpecIndustry
	}
	if diff <= tol*10 {
		return cnn2SpecLowBit
	}
	if diff <= 0.1 {
		return cnn2SpecDrift
	}
	return cnn2SpecHeavyDrift
}

func cnn2HasSignal(d []float32) bool {
	for _, v := range d {
		if v != 0 && !math.IsNaN(float64(v)) {
			return true
		}
	}
	return false
}

func cnn2MaxAbsDiff(a, b []float32) float64 {
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

func cnn2MaxWeightDiff(a, b *poly.VolumetricNetwork) float64 {
	var m float64
	for i := range a.Layers {
		wa, wb := a.Layers[i].WeightStore, b.Layers[i].WeightStore
		if wa != nil && wb != nil {
			if d := cnn2MaxAbsDiff(wa.Master, wb.Master); d > m {
				m = d
			}
		}
	}
	return m
}

func cnn2TrainingOK(init, final float64, dt poly.DType) bool {
	if math.IsNaN(init) || math.IsNaN(final) || math.IsInf(init, 0) || math.IsInf(final, 0) {
		return false
	}
	if init < 0.01 {
		return final <= init*2+1e-3
	}
	return final < init*0.99
}

func cnn2Mark(ok bool) string {
	if ok {
		return "PASS"
	}
	return "FAIL"
}
