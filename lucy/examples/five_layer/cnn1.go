// Five-layer CNN1 example for Loom (poly).
//
// STANDALONE (copy into your repo):
//   1. Copy this file; in go.mod: require github.com/openfluke/loom (or your poly path)
//   2. Change `package fivelayer` → `package main`
//   3. Rename `RunCNN1` → `main`  (or call runCNN1Example() from main)
//   4. go run cnn1.go
//
// Lucy menu [6] calls RunCNN1(); output is teed to lucy_testing_output/five_layer.txt
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
	cnn1TrainEpochs  = 50
	cnn1LearningRate = float32(0.01)
	cnn1GPUMode      = poly.TrainingModeGPUSC
	cnn1OutputDir    = "lucy_testing_output"
)

// ── Numerical types (all 21) ─────────────────────────────────────────────────

type cnn1DtypeCase struct {
	name, jsonName string
	dtype          poly.DType
	scale          float32
	tolerance      float64
}

var cnn1AllDTypes = []cnn1DtypeCase{
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

type cnn1Spectrum int

const (
	cnn1SpecExact cnn1Spectrum = iota
	cnn1SpecIndustry
	cnn1SpecLowBit
	cnn1SpecDrift
	cnn1SpecHeavyDrift
	cnn1SpecBroken
	cnn1SpecFatal
)

func (s cnn1Spectrum) String() string {
	switch s {
	case cnn1SpecExact:
		return "💎 EXACT"
	case cnn1SpecIndustry:
		return "✅ INDUS"
	case cnn1SpecLowBit:
		return "🟨 LOWBIT"
	case cnn1SpecDrift:
		return "🟠 DRIFT"
	case cnn1SpecHeavyDrift:
		return "🟤 H-DRIFT"
	case cnn1SpecBroken:
		return "❌ BROKE"
	default:
		return "💀 FATAL"
	}
}

// ── Step 1: Build JSON network spec (5 flat CNN1 layers, no SEQUENTIAL) ───

func cnn1NetworkJSON(jsonDType string) []byte {
	ch := []int{3, 8, 8, 8, 16, 16}
	var b strings.Builder
	b.WriteString(fmt.Sprintf(
		`{"id":"loom-five-cnn1","depth":1,"rows":1,"cols":1,"layers_per_cell":%d,"layers":[`,
		len(ch)-1,
	))
	for i := 0; i < len(ch)-1; i++ {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(fmt.Sprintf(
			`{"z":0,"y":0,"x":0,"l":%d,"type":"CNN1","activation":"RELU","dtype":"%s","input_channels":%d,"filters":%d,"input_height":%d,"output_height":%d,"kernel_size":3,"stride":1,"padding":1}`,
			i, jsonDType, ch[i], ch[i+1], 32, 32,
		))
	}
	b.WriteString(`]}`)
	return []byte(b.String())
}

// ── Step 2: Create network from JSON ─────────────────────────────────────────

func cnn1CreateNetwork(jsonDType string) (*poly.VolumetricNetwork, error) {
	return poly.BuildNetworkFromJSON(cnn1NetworkJSON(jsonDType))
}

// ── Input / target tensors ───────────────────────────────────────────────────

func cnn1MakeInput() *poly.Tensor[float32] {
	t := poly.NewTensor[float32](4, 3, 32)
	for i := range t.Data {
		t.Data[i] = 0.2 * float32(math.Sin(float64(i)*0.11+0.3))
	}
	return t
}

func cnn1MakeTarget(net *poly.VolumetricNetwork, input *poly.Tensor[float32]) *poly.Tensor[float32] {
	out, _, _ := poly.ForwardPolymorphic(net, input)
	tgt := poly.NewTensor[float32](out.Shape...)
	for i := range tgt.Data {
		tgt.Data[i] = 0.5 + 0.3*float32(math.Sin(float64(i)*0.17))
	}
	return tgt
}

// ── Morph all layers to numerical type ─────────────────────────────────────────

func cnn1ApplyDType(net *poly.VolumetricNetwork, tc cnn1DtypeCase) {
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

func cnn1SyncGPU(net *poly.VolumetricNetwork) error {
	return poly.ConfigureNetworkForMode(net, cnn1GPUMode)
}

// ── Save / reload + deviation bucket ─────────────────────────────────────────

type cnn1SavePhase string

const (
	cnn1PhaseBefore cnn1SavePhase = "BEFORE_TRAIN"
	cnn1PhaseAfter  cnn1SavePhase = "AFTER_TRAIN"
)

type cnn1SaveResult struct {
	phase            cnn1SavePhase
	forwardDiff      float64
	weightDiff       float64
	lossDelta        float64
	bucket           cnn1Spectrum
	nativeOK         bool
	pass             bool
	err              string
}

func cnn1CheckSaveReload(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32], tc cnn1DtypeCase, phase cnn1SavePhase, refLoss float64) cnn1SaveResult {
	r := cnn1SaveResult{phase: phase}
	out0, _, _ := poly.ForwardPolymorphic(net, input)
	baseline := append([]float32(nil), out0.Data...)

	wire, err := poly.SerializeNetwork(net)
	if err != nil {
		r.err = err.Error()
		r.bucket = cnn1SpecFatal
		return r
	}
	reloaded, err := poly.DeserializeNetwork(wire)
	if err != nil {
		r.err = err.Error()
		r.bucket = cnn1SpecFatal
		return r
	}
	if err := poly.ConfigureNetworkForMode(reloaded, cnn1GPUMode); err != nil {
		r.err = err.Error()
		r.bucket = cnn1SpecFatal
		return r
	}
	out1, _, _ := poly.ForwardPolymorphic(reloaded, input)
	r.forwardDiff = cnn1MaxAbsDiff(baseline, out1.Data)
	r.bucket = cnn1SpectrumMark(r.forwardDiff, tc.tolerance, out1.Data, baseline)
	r.weightDiff = cnn1MaxWeightDiff(net, reloaded)
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
		r.bucket <= cnn1SpecLowBit && r.nativeOK && r.err == ""
	return r
}

// ── Step 4: Train on GPU ─────────────────────────────────────────────────────

func cnn1Train(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32]) (*poly.TrainingResult, error) {
	return poly.Train(net, []poly.TrainingBatch[float32]{{Input: input, Target: target}}, &poly.TrainingConfig{
		Epochs:       cnn1TrainEpochs,
		LearningRate: cnn1LearningRate,
		Mode:         cnn1GPUMode,
		Verbose:      false,
		LossType:     "mse",
	})
}

// ── Step 5: Save checkpoint to disk ────────────────────────────────────────────

func cnn1SaveCheckpoint(net *poly.VolumetricNetwork, dtypeName string) string {
	wire, err := poly.SerializeNetwork(net)
	if err != nil {
		return ""
	}
	_ = os.MkdirAll(cnn1OutputDir, 0o755)
	path := filepath.Join(cnn1OutputDir, "cnn1_"+dtypeName+".json")
	_ = os.WriteFile(path, wire, 0o644)
	return path
}

// ── Run all numerical types ────────────────────────────────────────────────────

func runCNN1Example() bool {
	fmt.Println()
	fmt.Println("══════════════════════════════════════════════════════════════════════")
	fmt.Println("  Loom five-layer CNN1 — JSON · GPU · train · save · reload")
	fmt.Println("  5 flat layers in layers_per_cell (no SEQUENTIAL)")
	fmt.Println("══════════════════════════════════════════════════════════════════════")
	fmt.Printf("  Running %d numerical types (%d epochs GPU each, quiet)…\n", len(cnn1AllDTypes), cnn1TrainEpochs)

	var rows []DTypeRow
	passed, failed := 0, 0

	for _, tc := range cnn1AllDTypes {
		fmt.Printf("  · %-10s ", tc.name)
		row := DTypeRow{DType: tc.name}

		net, err := cnn1CreateNetwork(tc.jsonName)
		if err != nil {
			row.Err = "BUILD"
			rows = append(rows, row)
			failed++
			fmt.Println("BUILD ERR")
			continue
		}
		cnn1ApplyDType(net, tc)
		input := cnn1MakeInput()
		target := cnn1MakeTarget(net, input)

		if err := cnn1SyncGPU(net); err != nil {
			row.Err = "GPU"
			rows = append(rows, row)
			failed++
			fmt.Println("GPU ERR")
			continue
		}

		lossBefore := cnn1ForwardLoss(net, input, target)
		before := cnn1CheckSaveReload(net, input, target, tc, cnn1PhaseBefore, lossBefore)
		row.BeforeBucket = before.bucket.String()
		row.BeforeOK = before.pass
		row.NativeOK = before.nativeOK

		res, err := cnn1Train(net, input, target)
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

		_ = cnn1SaveCheckpoint(net, tc.name)
		after := cnn1CheckSaveReload(net, input, target, tc, cnn1PhaseAfter, lossAfter)
		row.AfterBucket = after.bucket.String()
		row.AfterOK = after.pass
		if !after.nativeOK {
			row.NativeOK = false
		}

		row.Learned = cnn1TrainingOK(lossInit, lossAfter, tc.dtype)
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

	PrintDTypeResultsTable("CNN1", rows)
	RegisterLayerSummary("CNN1", passed, failed, rows)
	return failed == 0
}

// RunCNN1 is called from Lucy menu [6].
func RunCNN1() bool { return runCNN1Example() }

// ── helpers ────────────────────────────────────────────────────────────────────

func cnn1ForwardLoss(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32]) float64 {
	out, _, _ := poly.ForwardPolymorphic(net, input)
	return poly.CalculateLoss(out, target, "mse")
}

func cnn1PrintSaveRow(dtype string, r cnn1SaveResult) {
	if r.err != "" {
		fmt.Printf("| %-10s | %-13s | ERR      |          | 💀 FATAL     |            | FAIL    |\n", dtype, r.phase)
		return
	}
	fmt.Printf("| %-10s | %-13s | %-8.2e | %-8.2e | %-12s | %-12s | %-7s |\n",
		dtype, r.phase, r.forwardDiff, r.weightDiff, r.bucket.String(), cnn1Mark(r.nativeOK), cnn1Mark(r.pass))
}

func cnn1SpectrumMark(diff, tol float64, actual, baseline []float32) cnn1Spectrum {
	if math.IsNaN(diff) || (math.IsInf(diff, 0) && !cnn1HasSignal(baseline)) {
		return cnn1SpecFatal
	}
	if math.IsInf(diff, 0) {
		return cnn1SpecHeavyDrift
	}
	as, bs := cnn1HasSignal(actual), cnn1HasSignal(baseline)
	if !as && !bs {
		return cnn1SpecExact
	}
	if !as || !bs {
		return cnn1SpecHeavyDrift
	}
	if diff == 0 {
		return cnn1SpecExact
	}
	if diff <= tol {
		return cnn1SpecIndustry
	}
	if diff <= tol*10 {
		return cnn1SpecLowBit
	}
	if diff <= 0.1 {
		return cnn1SpecDrift
	}
	return cnn1SpecHeavyDrift
}

func cnn1HasSignal(d []float32) bool {
	for _, v := range d {
		if v != 0 && !math.IsNaN(float64(v)) {
			return true
		}
	}
	return false
}

func cnn1MaxAbsDiff(a, b []float32) float64 {
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

func cnn1MaxWeightDiff(a, b *poly.VolumetricNetwork) float64 {
	var m float64
	for i := range a.Layers {
		wa, wb := a.Layers[i].WeightStore, b.Layers[i].WeightStore
		if wa != nil && wb != nil {
			if d := cnn1MaxAbsDiff(wa.Master, wb.Master); d > m {
				m = d
			}
		}
	}
	return m
}

func cnn1TrainingOK(init, final float64, dt poly.DType) bool {
	if math.IsNaN(init) || math.IsNaN(final) || math.IsInf(init, 0) || math.IsInf(final, 0) {
		return false
	}
	if init < 0.01 {
		return final <= init*2+1e-3
	}
	return final < init*0.99
}

func cnn1Mark(ok bool) string {
	if ok {
		return "PASS"
	}
	return "FAIL"
}
