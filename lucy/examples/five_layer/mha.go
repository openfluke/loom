// Five-layer MHA example for Loom (poly).
//
// STANDALONE (copy into your repo):
//   1. Copy this file; in go.mod: require github.com/openfluke/loom (or your poly path)
//   2. Change `package fivelayer` → `package main`
//   3. Rename `RunMHA` → `main`  (or call runMHAExample() from main)
//   4. go run mha.go
//
// Lucy menu [6] calls RunMHA(); output is teed to lucy_testing_output/five_layer.txt
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
	mhaTrainEpochs  = 50
	mhaLearningRate = float32(0.01)
	mhaGPUMode      = poly.TrainingModeGPUSC
	mhaOutputDir    = "lucy_testing_output"
)

// ── Numerical types (all 21) ─────────────────────────────────────────────────

type mhaDtypeCase struct {
	name, jsonName string
	dtype          poly.DType
	scale          float32
	tolerance      float64
}

var mhaAllDTypes = []mhaDtypeCase{
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

type mhaSpectrum int

const (
	mhaSpecExact mhaSpectrum = iota
	mhaSpecIndustry
	mhaSpecLowBit
	mhaSpecDrift
	mhaSpecHeavyDrift
	mhaSpecBroken
	mhaSpecFatal
)

func (s mhaSpectrum) String() string {
	switch s {
	case mhaSpecExact:
		return "💎 EXACT"
	case mhaSpecIndustry:
		return "✅ INDUS"
	case mhaSpecLowBit:
		return "🟨 LOWBIT"
	case mhaSpecDrift:
		return "🟠 DRIFT"
	case mhaSpecHeavyDrift:
		return "🟤 H-DRIFT"
	case mhaSpecBroken:
		return "❌ BROKE"
	default:
		return "💀 FATAL"
	}
}

// ── Step 1: Build JSON network spec (5 flat MHA layers, no SEQUENTIAL) ───

func mhaNetworkJSON(jsonDType string) []byte {
	var b strings.Builder
	b.WriteString(`{"id":"loom-five-mha","depth":1,"rows":1,"cols":1,"layers_per_cell":5,"layers":[`)
	for i := 0; i < 5; i++ {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(fmt.Sprintf(
			`{"z":0,"y":0,"x":0,"l":%d,"type":"MHA","activation":"RELU","dtype":"%s","d_model":64,"num_heads":4,"seq_length":8}`,
			i, jsonDType,
		))
	}
	b.WriteString(`]}`)
	return []byte(b.String())
}

// ── Step 2: Create network from JSON ─────────────────────────────────────────

func mhaCreateNetwork(jsonDType string) (*poly.VolumetricNetwork, error) {
	return poly.BuildNetworkFromJSON(mhaNetworkJSON(jsonDType))
}

// ── Input / target tensors ───────────────────────────────────────────────────

func mhaMakeInput() *poly.Tensor[float32] {
	t := poly.NewTensor[float32](1, 8, 64)
	for i := range t.Data {
		t.Data[i] = 0.15 * float32(math.Sin(float64(i)*0.09+0.2))
	}
	return t
}

func mhaMakeTarget(net *poly.VolumetricNetwork, input *poly.Tensor[float32]) *poly.Tensor[float32] {
	out, _, _ := poly.ForwardPolymorphic(net, input)
	tgt := poly.NewTensor[float32](out.Shape...)
	for i := range tgt.Data {
		tgt.Data[i] = 0.5 + 0.3*float32(math.Sin(float64(i)*0.17))
	}
	return tgt
}

// ── Morph all layers to numerical type ─────────────────────────────────────────

func mhaApplyDType(net *poly.VolumetricNetwork, tc mhaDtypeCase) {
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

func mhaSyncGPU(net *poly.VolumetricNetwork) error {
	return poly.ConfigureNetworkForMode(net, mhaGPUMode)
}

// ── Save / reload + deviation bucket ─────────────────────────────────────────

type mhaSavePhase string

const (
	mhaPhaseBefore mhaSavePhase = "BEFORE_TRAIN"
	mhaPhaseAfter  mhaSavePhase = "AFTER_TRAIN"
)

type mhaSaveResult struct {
	phase            mhaSavePhase
	forwardDiff      float64
	weightDiff       float64
	lossDelta        float64
	bucket           mhaSpectrum
	nativeOK         bool
	pass             bool
	err              string
}

func mhaCheckSaveReload(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32], tc mhaDtypeCase, phase mhaSavePhase, refLoss float64) mhaSaveResult {
	r := mhaSaveResult{phase: phase}
	out0, _, _ := poly.ForwardPolymorphic(net, input)
	baseline := append([]float32(nil), out0.Data...)

	wire, err := poly.SerializeNetwork(net)
	if err != nil {
		r.err = err.Error()
		r.bucket = mhaSpecFatal
		return r
	}
	reloaded, err := poly.DeserializeNetwork(wire)
	if err != nil {
		r.err = err.Error()
		r.bucket = mhaSpecFatal
		return r
	}
	if err := poly.ConfigureNetworkForMode(reloaded, mhaGPUMode); err != nil {
		r.err = err.Error()
		r.bucket = mhaSpecFatal
		return r
	}
	out1, _, _ := poly.ForwardPolymorphic(reloaded, input)
	r.forwardDiff = mhaMaxAbsDiff(baseline, out1.Data)
	r.bucket = mhaSpectrumMark(r.forwardDiff, tc.tolerance, out1.Data, baseline)
	r.weightDiff = mhaMaxWeightDiff(net, reloaded)
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
		r.bucket <= mhaSpecLowBit && r.nativeOK && r.err == ""
	return r
}

// ── Step 4: Train on GPU ─────────────────────────────────────────────────────

func mhaTrain(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32]) (*poly.TrainingResult, error) {
	return poly.Train(net, []poly.TrainingBatch[float32]{{Input: input, Target: target}}, &poly.TrainingConfig{
		Epochs:       mhaTrainEpochs,
		LearningRate: mhaLearningRate,
		Mode:         mhaGPUMode,
		Verbose:      false,
		LossType:     "mse",
	})
}

// ── Step 5: Save checkpoint to disk ────────────────────────────────────────────

func mhaSaveCheckpoint(net *poly.VolumetricNetwork, dtypeName string) string {
	wire, err := poly.SerializeNetwork(net)
	if err != nil {
		return ""
	}
	_ = os.MkdirAll(mhaOutputDir, 0o755)
	path := filepath.Join(mhaOutputDir, "mha_"+dtypeName+".json")
	_ = os.WriteFile(path, wire, 0o644)
	return path
}

// ── Run all numerical types ────────────────────────────────────────────────────

func runMHAExample() bool {
	fmt.Println()
	fmt.Println("══════════════════════════════════════════════════════════════════════")
	fmt.Println("  Loom five-layer MHA — JSON · GPU · train · save · reload")
	fmt.Println("  5 flat layers in layers_per_cell (no SEQUENTIAL)")
	fmt.Println("══════════════════════════════════════════════════════════════════════")
	fmt.Printf("  Running %d numerical types (%d epochs GPU each, quiet)…\n", len(mhaAllDTypes), mhaTrainEpochs)

	var rows []DTypeRow
	passed, failed := 0, 0

	for _, tc := range mhaAllDTypes {
		fmt.Printf("  · %-10s ", tc.name)
		row := DTypeRow{DType: tc.name}

		net, err := mhaCreateNetwork(tc.jsonName)
		if err != nil {
			row.Err = "BUILD"
			rows = append(rows, row)
			failed++
			fmt.Println("BUILD ERR")
			continue
		}
		mhaApplyDType(net, tc)
		input := mhaMakeInput()
		target := mhaMakeTarget(net, input)

		if err := mhaSyncGPU(net); err != nil {
			row.Err = "GPU"
			rows = append(rows, row)
			failed++
			fmt.Println("GPU ERR")
			continue
		}

		lossBefore := mhaForwardLoss(net, input, target)
		before := mhaCheckSaveReload(net, input, target, tc, mhaPhaseBefore, lossBefore)
		row.BeforeBucket = before.bucket.String()
		row.BeforeOK = before.pass
		row.NativeOK = before.nativeOK

		res, err := mhaTrain(net, input, target)
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

		_ = mhaSaveCheckpoint(net, tc.name)
		after := mhaCheckSaveReload(net, input, target, tc, mhaPhaseAfter, lossAfter)
		row.AfterBucket = after.bucket.String()
		row.AfterOK = after.pass
		if !after.nativeOK {
			row.NativeOK = false
		}

		row.Learned = mhaTrainingOK(lossInit, lossAfter, tc.dtype)
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

	PrintDTypeResultsTable("MHA", rows)
	RegisterLayerSummary("MHA", passed, failed, rows)
	return failed == 0
}

// RunMHA is called from Lucy menu [6].
func RunMHA() bool { return runMHAExample() }

// ── helpers ────────────────────────────────────────────────────────────────────

func mhaForwardLoss(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32]) float64 {
	out, _, _ := poly.ForwardPolymorphic(net, input)
	return poly.CalculateLoss(out, target, "mse")
}

func mhaPrintSaveRow(dtype string, r mhaSaveResult) {
	if r.err != "" {
		fmt.Printf("| %-10s | %-13s | ERR      |          | 💀 FATAL     |            | FAIL    |\n", dtype, r.phase)
		return
	}
	fmt.Printf("| %-10s | %-13s | %-8.2e | %-8.2e | %-12s | %-12s | %-7s |\n",
		dtype, r.phase, r.forwardDiff, r.weightDiff, r.bucket.String(), mhaMark(r.nativeOK), mhaMark(r.pass))
}

func mhaSpectrumMark(diff, tol float64, actual, baseline []float32) mhaSpectrum {
	if math.IsNaN(diff) || (math.IsInf(diff, 0) && !mhaHasSignal(baseline)) {
		return mhaSpecFatal
	}
	if math.IsInf(diff, 0) {
		return mhaSpecHeavyDrift
	}
	as, bs := mhaHasSignal(actual), mhaHasSignal(baseline)
	if !as && !bs {
		return mhaSpecExact
	}
	if !as || !bs {
		return mhaSpecHeavyDrift
	}
	if diff == 0 {
		return mhaSpecExact
	}
	if diff <= tol {
		return mhaSpecIndustry
	}
	if diff <= tol*10 {
		return mhaSpecLowBit
	}
	if diff <= 0.1 {
		return mhaSpecDrift
	}
	return mhaSpecHeavyDrift
}

func mhaHasSignal(d []float32) bool {
	for _, v := range d {
		if v != 0 && !math.IsNaN(float64(v)) {
			return true
		}
	}
	return false
}

func mhaMaxAbsDiff(a, b []float32) float64 {
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

func mhaMaxWeightDiff(a, b *poly.VolumetricNetwork) float64 {
	var m float64
	for i := range a.Layers {
		wa, wb := a.Layers[i].WeightStore, b.Layers[i].WeightStore
		if wa != nil && wb != nil {
			if d := mhaMaxAbsDiff(wa.Master, wb.Master); d > m {
				m = d
			}
		}
	}
	return m
}

func mhaTrainingOK(init, final float64, dt poly.DType) bool {
	if math.IsNaN(init) || math.IsNaN(final) || math.IsInf(init, 0) || math.IsInf(final, 0) {
		return false
	}
	if init < 0.01 {
		return final <= init*2+1e-3
	}
	return final < init*0.99
}

func mhaMark(ok bool) string {
	if ok {
		return "PASS"
	}
	return "FAIL"
}
