// Package sevenlayer runs Lucy menu [7]: 7-deep JSON networks, CPU SC/MC parity,
// programmatic ASM (Dense forward only), train, save/reload, and timing.
package sevenlayer

import (
	"fmt"
	"math"
	"time"

	"github.com/openfluke/loom/poly"
	"github.com/openfluke/loom/poly/asm"
)

const (
	trainEpochs  = 50
	learningRate = float32(0.05)
	numLayers    = 7
)

type dtypeCase struct {
	name, jsonName string
	dtype          poly.DType
	scale          float32
	tolerance      float64
}

var allDTypes = []dtypeCase{
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

type spectrum int

const (
	specExact spectrum = iota
	specIndustry
	specLowBit
	specDrift
	specHeavyDrift
	specBroken
	specFatal
)

func (s spectrum) String() string {
	switch s {
	case specExact:
		return "💎 EXACT"
	case specIndustry:
		return "✅ INDUS"
	case specLowBit:
		return "🟨 LOWBIT"
	case specDrift:
		return "🟠 DRIFT"
	case specHeavyDrift:
		return "🟤 H-DRIFT"
	case specBroken:
		return "❌ BROKE"
	default:
		return "💀 FATAL"
	}
}

// AsmStatus describes assembly support for a layer type on this build.
type AsmStatus struct {
	ForwardCapable  bool
	BackwardCapable bool
	RuntimeEnabled  bool
	Note            string
}

func platformAsmEnabled() bool { return asm.Enabled() }

func layerAsmStatus(layerType poly.LayerType) AsmStatus {
	switch layerType {
	case poly.LayerDense:
		if !platformAsmEnabled() {
			return AsmStatus{
				ForwardCapable: false, BackwardCapable: false,
				Note: "ASM unavailable on this GOARCH (need amd64/arm64)",
			}
		}
		return AsmStatus{
			ForwardCapable: true, BackwardCapable: false,
			Note: "Dense forward ASM only; backward not implemented",
		}
	default:
		return AsmStatus{
			ForwardCapable: false, BackwardCapable: false,
			Note: "ASM not implemented for this layer type",
		}
	}
}

// SetNetworkAsm toggles UseAsmForward on the network and each Dense layer (API, not JSON).
func SetNetworkAsm(net *poly.VolumetricNetwork, on bool) {
	net.UseAsmForward = on
	for i := range net.Layers {
		l := &net.Layers[i]
		if l.Type == poly.LayerDense {
			l.UseAsmForward = on
		}
	}
}

func applyDType(net *poly.VolumetricNetwork, tc dtypeCase) {
	for i := range net.Layers {
		applyDTypeLayer(&net.Layers[i], tc)
	}
}

func applyDTypeLayer(l *poly.VolumetricLayer, tc dtypeCase) {
	l.DType = tc.dtype
	if l.WeightStore != nil {
		l.WeightStore.InvalidateVersions()
		if tc.scale != 1.0 {
			l.WeightStore.Scale = tc.scale
		}
		l.WeightStore.Morph(tc.dtype)
		if l.Network != nil {
			l.SyncToCPU()
		}
	}
	for i := range l.ParallelBranches {
		applyDTypeLayer(&l.ParallelBranches[i], tc)
	}
	for i := range l.SequentialLayers {
		applyDTypeLayer(&l.SequentialLayers[i], tc)
	}
	if l.MetaObservedLayer != nil {
		applyDTypeLayer(l.MetaObservedLayer, tc)
	}
}

// wireLayerTree sets Network on nested layers after DeserializeNetwork.
// prepareTrainingNet scales flat-cell weights so deep stacks (7 layers/cell) get usable gradients.
func prepareTrainingNet(net *poly.VolumetricNetwork, dt poly.DType) {
	if net == nil || net.LayersPerCell <= 1 {
		return
	}
	scale := float32(math.Sqrt(float64(net.LayersPerCell)))
	switch dt {
	case poly.DTypeUint64, poly.DTypeUint32, poly.DTypeUint16, poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		return // unsigned quant: keep JSON init; scaling destabilizes training
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary, poly.DTypeFP4,
		poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		scale = 1.5
	case poly.DTypeInt64, poly.DTypeInt32, poly.DTypeInt16:
		scale = 1.5
	default:
		// float32/float64/float16/bfloat16: full depth scaling
	}
	for i := range net.Layers {
		prepareTrainingLayer(&net.Layers[i], scale)
	}
}

func trainingLearningRate(dt poly.DType) float32 {
	if isQuantIntegerDType(dt) {
		return 0.01
	}
	switch dt {
	case poly.DTypeFP8E4M3, poly.DTypeFP8E5M2, poly.DTypeFP4:
		return 0.01
	default:
		return learningRate
	}
}

func prepareTrainingLayer(l *poly.VolumetricLayer, scale float32) {
	if l.WeightStore != nil && scale != 1 {
		for j := range l.WeightStore.Master {
			l.WeightStore.Master[j] *= scale
		}
		l.WeightStore.InvalidateVersions()
	}
	for i := range l.ParallelBranches {
		prepareTrainingLayer(&l.ParallelBranches[i], scale)
	}
	for i := range l.SequentialLayers {
		prepareTrainingLayer(&l.SequentialLayers[i], scale)
	}
	if l.MetaObservedLayer != nil {
		prepareTrainingLayer(l.MetaObservedLayer, scale)
	}
}

func wireLayerTree(net *poly.VolumetricNetwork) {
	for i := range net.Layers {
		wireLayer(&net.Layers[i], net)
	}
}

func wireLayer(l *poly.VolumetricLayer, net *poly.VolumetricNetwork) {
	l.Network = net
	for i := range l.ParallelBranches {
		wireLayer(&l.ParallelBranches[i], net)
	}
	for i := range l.SequentialLayers {
		wireLayer(&l.SequentialLayers[i], net)
	}
	if l.MetaObservedLayer != nil {
		wireLayer(l.MetaObservedLayer, net)
	}
}

func setCPUMode(net *poly.VolumetricNetwork, multiCore, useAsm bool) {
	net.UseGPU = false
	net.EnableMultiCoreTiling = multiCore
	SetNetworkAsm(net, useAsm)
	for i := range net.Layers {
		l := &net.Layers[i]
		l.UseTiling = true
		l.EnableMultiCoreTiling = multiCore
	}
}

func resetNetwork(net *poly.VolumetricNetwork) {
	for i := range net.Layers {
		net.Layers[i].ResetState()
	}
}

type forwardCapture struct {
	out []float32
	dur time.Duration
}

func captureForward(net *poly.VolumetricNetwork, input *poly.Tensor[float32], multiCore, useAsm bool) forwardCapture {
	setCPUMode(net, multiCore, useAsm)
	resetNetwork(net)
	t0 := time.Now()
	out, _, _ := poly.ForwardPolymorphic(net, input)
	return forwardCapture{out: append([]float32(nil), out.Data...), dur: time.Since(t0)}
}

type backwardCapture struct {
	dx, dw []float32
	dur    time.Duration
}

func captureBackward(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32], multiCore bool) backwardCapture {
	setCPUMode(net, multiCore, false)
	resetNetwork(net)

	histIn := make([]*poly.Tensor[float32], len(net.Layers))
	histPre := make([]*poly.Tensor[float32], len(net.Layers))
	curr := input
	for i := range net.Layers {
		l := &net.Layers[i]
		if l.IsDisabled {
			continue
		}
		histIn[i] = curr
		pre, post := poly.DispatchLayer(l, curr, nil)
		histPre[i] = pre
		curr = post
	}
	gradOut := poly.ComputeLossGradient(curr, target, "mse")

	t0 := time.Now()
	_, layerGrads, _ := poly.BackwardPolymorphic(net, gradOut, histIn, histPre)
	dur := time.Since(t0)

	var dx, dw []float32
	if len(layerGrads) > 0 && layerGrads[0][0] != nil {
		dx = append([]float32(nil), layerGrads[0][0].Data...)
	}
	for _, g := range layerGrads {
		if g[1] != nil {
			dw = append(dw, g[1].Data...)
		}
	}
	return backwardCapture{dx: dx, dw: dw, dur: dur}
}

func maxAbsDiff(a, b []float32) float64 {
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

func maxWeightDiff(a, b *poly.VolumetricNetwork) float64 {
	var m float64
	for i := range a.Layers {
		if d := maxLayerWeightDiff(&a.Layers[i], &b.Layers[i]); d > m {
			m = d
		}
	}
	return m
}

func maxLayerWeightDiff(a, b *poly.VolumetricLayer) float64 {
	var m float64
	if a.WeightStore != nil && b.WeightStore != nil {
		if d := maxAbsDiff(a.WeightStore.Master, b.WeightStore.Master); d > m {
			m = d
		}
	}
	for i := range a.ParallelBranches {
		if d := maxLayerWeightDiff(&a.ParallelBranches[i], &b.ParallelBranches[i]); d > m {
			m = d
		}
	}
	for i := range a.SequentialLayers {
		if d := maxLayerWeightDiff(&a.SequentialLayers[i], &b.SequentialLayers[i]); d > m {
			m = d
		}
	}
	return m
}

func spectrumMark(diff, tol float64, actual, baseline []float32) spectrum {
	if math.IsNaN(diff) || math.IsInf(diff, 0) {
		return specFatal
	}
	if diff == 0 {
		return specExact
	}
	if diff <= tol {
		return specIndustry
	}
	if diff <= tol*10 {
		return specLowBit
	}
	if diff <= 0.1 {
		return specDrift
	}
	return specHeavyDrift
}

func isQuantIntegerDType(dt poly.DType) bool {
	switch dt {
	case poly.DTypeInt64, poly.DTypeInt32, poly.DTypeInt16, poly.DTypeInt8, poly.DTypeInt4, poly.DTypeInt2,
		poly.DTypeUint64, poly.DTypeUint32, poly.DTypeUint16, poly.DTypeUint8,
		poly.DTypeUint4, poly.DTypeUint2, poly.DTypeBinary, poly.DTypeTernary, poly.DTypeFP4,
		poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		return true
	default:
		return false
	}
}

// trainingOK matches lucy/testing loss criteria (short CPU runs, quant tolerance bands).
func trainingOK(lossInit, lossFinal float64, dtype poly.DType) bool {
	if math.IsNaN(lossInit) || math.IsNaN(lossFinal) ||
		math.IsInf(lossInit, 0) || math.IsInf(lossFinal, 0) {
		return false
	}
	if lossInit > 1e-3 && (lossFinal > lossInit*50 || lossFinal > 1e10) {
		return false
	}
	if lossInit < 0.01 {
		if lossFinal <= lossInit*2.0+1e-3 {
			return true
		}
		return isQuantIntegerDType(dtype) && lossFinal < 1.0
	}
	if isQuantIntegerDType(dtype) {
		band := 0.15
		switch dtype {
		case poly.DTypeUint64, poly.DTypeUint32, poly.DTypeUint16, poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
			band = 0.22
		}
		if lossFinal <= lossInit*(1.0+band)+1e-3 {
			return true
		}
		rel := math.Abs(lossFinal-lossInit) / (math.Abs(lossInit) + 1e-9)
		return rel <= band
	}
	return lossFinal < lossInit*0.99
}

func formatDur(d time.Duration) string {
	if d < time.Second {
		return fmt.Sprintf("%dms", d.Milliseconds())
	}
	return fmt.Sprintf("%.3fs", d.Seconds())
}

func samplesPerSec(d time.Duration, epochs int) float64 {
	if d <= 0 {
		return 0
	}
	return float64(epochs) / d.Seconds()
}
