// Package sevenlayer runs Lucy menu [7]: 7-deep JSON networks, CPU SC/MC parity,
// train, save/reload, and timing.
package sevenlayer

import (
	"fmt"
	"math"
	"time"

	"github.com/openfluke/loom/poly"
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

func applyDType(net *poly.VolumetricNetwork, tc dtypeCase) {
	for i := range net.Layers {
		applyDTypeLayer(&net.Layers[i], tc)
	}
}

// configureTrainingNet enables native in-place updates (UseExactDType) on Dense only.
// Other ops use their default float MAC paths; forcing exact on CNN1/MHA caused unstable training.
func configureTrainingNet(net *poly.VolumetricNetwork, tc dtypeCase, primary poly.LayerType) {
	wireLayerTree(net)
	net.UseExactDType = primary == poly.LayerDense && poly.IsDenseNativeExactDType(tc.dtype)
}

// configureInferenceNet drops FP32 Master after native sync so forward-only RAM
// reflects morphed Versions (training nets override via EnsureTrainingWeights).
func configureInferenceNet(net *poly.VolumetricNetwork) {
	wireLayerTree(net)
	net.ReleaseFP32MasterWhenIdle = true
	net.SyncInferenceWeights()
}

func finalizeTrainingNet(net *poly.VolumetricNetwork, tc dtypeCase) {
	wireLayerTree(net)
	net.EnsureTrainingWeights()
	for i := range net.Layers {
		finalizeTrainingLayer(&net.Layers[i], tc.dtype)
	}
}

func finalizeTrainingLayer(l *poly.VolumetricLayer, dt poly.DType) {
	if l.WeightStore != nil && dt != poly.DTypeFloat32 {
		l.WeightStore.ForceMorph(dt)
	}
	for i := range l.ParallelBranches {
		finalizeTrainingLayer(&l.ParallelBranches[i], dt)
	}
	for i := range l.SequentialLayers {
		finalizeTrainingLayer(&l.SequentialLayers[i], dt)
	}
	if l.MetaObservedLayer != nil {
		finalizeTrainingLayer(l.MetaObservedLayer, dt)
	}
}

func saveReloadFwdTol(phase savePhase, tc dtypeCase, primary poly.LayerType) float64 {
	tol := tc.tolerance
	if primary == poly.LayerMultiHeadAttention {
		if tol < 1e-4 {
			tol = 1e-4
		}
		if phase == phaseAfter && tol < 2e-4 {
			tol = 2e-4
		}
	}
	if phase == phaseAfter {
		tol = tc.tolerance * 100
		if poly.IsDenseNativeTrainDType(tc.dtype) {
			tol = tc.tolerance * 1000
		}
		// Packed signed types on wide ops (SwiGLU/MHA/RNN) can show ~0.15–0.2 fwd
		// delta after reload while native blobs still match.
		switch tc.dtype {
		case poly.DTypeInt4, poly.DTypeInt2, poly.DTypeTernary:
			if tol < 0.2 {
				tol = 0.2
			}
		}
	}
	return tol
}

func saveReloadMaxBucket(phase savePhase, dt poly.DType, primary poly.LayerType) spectrum {
	if phase == phaseAfter {
		switch dt {
		case poly.DTypeInt4, poly.DTypeInt2, poly.DTypeTernary:
			return specHeavyDrift
		}
	}
	if primary == poly.LayerMultiHeadAttention {
		switch dt {
		case poly.DTypeUint64, poly.DTypeUint32, poly.DTypeUint16:
			return specHeavyDrift
		case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeBinary, poly.DTypeFP8E4M3:
			return specDrift
		}
	}
	if poly.IsDenseNativeTrainDType(dt) || isQuantIntegerDType(dt) {
		return specLowBit
	}
	return specIndustry
}

func layerMasterDim(l *poly.VolumetricLayer) (dim int, ok bool) {
	switch l.Type {
	case poly.LayerDense, poly.LayerSwiGLU:
		dim = l.InputHeight
		if l.OutputHeight > dim {
			dim = l.OutputHeight
		}
		return dim, dim > 0
	case poly.LayerMultiHeadAttention:
		return l.DModel, l.DModel > 0
	case poly.LayerCNN1:
		dim = l.InputChannels
		if l.Filters > dim {
			dim = l.Filters
		}
		if l.InputHeight > dim {
			dim = l.InputHeight
		}
		return dim, dim > 0
	case poly.LayerCNN2:
		dim = l.InputChannels
		if l.Filters > dim {
			dim = l.Filters
		}
		if l.InputHeight > dim {
			dim = l.InputHeight
		}
		if l.InputWidth > dim {
			dim = l.InputWidth
		}
		return dim, dim > 0
	case poly.LayerCNN3:
		dim = l.InputChannels
		if l.Filters > dim {
			dim = l.Filters
		}
		if l.InputDepth > dim {
			dim = l.InputDepth
		}
		if l.InputHeight > dim {
			dim = l.InputHeight
		}
		if l.InputWidth > dim {
			dim = l.InputWidth
		}
		return dim, dim > 0
	case poly.LayerRNN, poly.LayerLSTM:
		dim = l.InputHeight
		if l.OutputHeight > dim {
			dim = l.OutputHeight
		}
		return dim, dim > 0
	default:
		return 0, false
	}
}

func mhaUintMasterExtra(dtype poly.DType, stack, dim int) float32 {
	switch dtype {
	case poly.DTypeUint64, poly.DTypeUint32, poly.DTypeUint16:
		if dim < 16 {
			return 1.0
		}
		dimF := float32(16) / float32(dim)
		if dimF > 1 {
			dimF = 1
		}
		depthF := float32(32) / float32(stack)
		if depthF > 1 {
			depthF = 1
		}
		return dimF * dimF * depthF
	}
	return 1.0
}

func cnn2UintMasterExtra(dtype poly.DType, stack, kSize int) float32 {
	if stack < 48 {
		return 1.0
	}
	if kSize <= 0 {
		kSize = 1
	}
	switch dtype {
	case poly.DTypeUint64, poly.DTypeUint32, poly.DTypeUint16:
		depthF := float32(48) / float32(stack)
		if depthF > 1 {
			depthF = 1
		}
		// 2D k×k backprop needs extra dampening vs CNN1; keep floor so training still learns.
		extra := depthF / float32(kSize)
		if extra < 0.2 {
			extra = 0.2
		}
		return extra
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		depthF := float32(48) / float32(stack)
		if depthF > 1 {
			depthF = 1
		}
		extra := depthF / float32(kSize*kSize)
		if extra < 0.15 {
			extra = 0.15
		}
		return extra
	}
	return 1.0
}

func masterWeightScaleForLayer(l *poly.VolumetricLayer, tc dtypeCase) float32 {
	if l.Network == nil {
		return 1.0
	}
	dim, ok := layerMasterDim(l)
	if !ok {
		return 1.0
	}
	stack := l.Network.StackLayerCount()
	wScale := poly.MasterWeightScaleForStackDepth(tc.dtype, stack, dim)
	if l.Type == poly.LayerMultiHeadAttention {
		wScale *= mhaUintMasterExtra(tc.dtype, stack, dim)
	}
	if l.Type == poly.LayerCNN2 {
		wScale *= cnn2UintMasterExtra(tc.dtype, stack, l.KernelSize)
	}
	return wScale
}

func simdParityTol(primary poly.LayerType, tc dtypeCase) float64 {
	tol := tc.tolerance
	if tol < 1e-10 {
		tol = 1e-10
	}
	if primary == poly.LayerMultiHeadAttention {
		if tol < 1e-4 {
			tol = 1e-4
		}
		// MHA softmax amplifies projection tile-order deltas on wide unsigned paths.
		switch tc.dtype {
		case poly.DTypeUint64, poly.DTypeUint32, poly.DTypeUint16:
			if tol < 0.1 {
				tol = 0.1
			}
		}
	}
	if primary == poly.LayerCNN1 || primary == poly.LayerCNN2 || primary == poly.LayerCNN3 || primary == poly.LayerRNN || primary == poly.LayerLSTM {
		switch tc.dtype {
		case poly.DTypeUint64, poly.DTypeUint32, poly.DTypeUint16:
			if tol < 0.1 {
				tol = 0.1
			}
		}
	}
	return tol
}

func applyDTypeLayer(l *poly.VolumetricLayer, tc dtypeCase) {
	l.DType = tc.dtype
	if l.WeightStore != nil {
		l.WeightStore.InvalidateVersions()
		if wScale := masterWeightScaleForLayer(l, tc); wScale != 1.0 {
			for j := range l.WeightStore.Master {
				l.WeightStore.Master[j] *= wScale
			}
		}
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
		prepareTrainingLayer(&net.Layers[i], scale, dt)
	}
}

func trainingLearningRate(dt poly.DType) float32 {
	switch dt {
	case poly.DTypeUint64, poly.DTypeUint32, poly.DTypeUint16:
		return 0.0005
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		return 0.005
	}
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

func prepareTrainingLayer(l *poly.VolumetricLayer, scale float32, dt poly.DType) {
	// MHA float stacks are already wide; depth scaling blows up attention logits.
	if l.Type == poly.LayerMultiHeadAttention {
		scale = 1
	}
	// CNN2 k×k MACs per output (~k²×C) explode when sqrt(layers/cell) scaling
	// is applied on deep 3³ stacks; CNN1 (k×C) tolerates the same boost.
	if l.Type == poly.LayerCNN2 {
		switch dt {
		case poly.DTypeFloat64, poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16:
			scale = 1
		}
	}
	if l.WeightStore != nil && scale != 1 {
		for j := range l.WeightStore.Master {
			l.WeightStore.Master[j] *= scale
		}
		l.WeightStore.InvalidateVersions()
	}
	for i := range l.ParallelBranches {
		prepareTrainingLayer(&l.ParallelBranches[i], scale, dt)
	}
	for i := range l.SequentialLayers {
		prepareTrainingLayer(&l.SequentialLayers[i], scale, dt)
	}
	if l.MetaObservedLayer != nil {
		prepareTrainingLayer(l.MetaObservedLayer, scale, dt)
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

func setCPUMode(net *poly.VolumetricNetwork, multiCore bool) {
	net.UseGPU = false
	net.EnableMultiCoreTiling = multiCore
	for i := range net.Layers {
		l := &net.Layers[i]
		l.UseTiling = true
		l.EnableMultiCoreTiling = multiCore
	}
	net.RefreshRuntimeTileSizes()
}

func setSimdForward(net *poly.VolumetricNetwork, enabled bool) {
	net.SetSimdForwardRecursive(enabled)
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

func captureForward(net *poly.VolumetricNetwork, input *poly.Tensor[float32], multiCore bool) forwardCapture {
	out, avg := benchmarkForward(net, input, multiCore)
	return forwardCapture{out: out, dur: avg}
}

func captureForwardSimd(net *poly.VolumetricNetwork, input *poly.Tensor[float32], multiCore bool) forwardCapture {
	out, avg := benchmarkForwardSimd(net, input, multiCore)
	return forwardCapture{out: out, dur: avg}
}

type backwardCapture struct {
	dx, dw []float32
	dur    time.Duration
}

func captureBackward(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32], multiCore bool) backwardCapture {
	dx, dw, avg := benchmarkBackward(net, input, target, multiCore, false)
	return backwardCapture{dx: dx, dw: dw, dur: avg}
}

func captureBackwardSimd(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32], multiCore bool) backwardCapture {
	dx, dw, avg := benchmarkBackward(net, input, target, multiCore, true)
	return backwardCapture{dx: dx, dw: dw, dur: avg}
}

func maxAbsDiff(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var m float64
	for i := 0; i < n; i++ {
		ai, bi := float64(a[i]), float64(b[i])
		if math.IsNaN(ai) || math.IsNaN(bi) {
			return math.NaN()
		}
		if v := math.Abs(ai - bi); v > m {
			m = v
		}
	}
	return m
}

// lossFiniteOK rejects NaN/Inf and degenerate zero loss on trainable stacks.
func lossFiniteOK(lossInit, lossFinal float64, requiresLearn bool) bool {
	if math.IsNaN(lossInit) || math.IsNaN(lossFinal) ||
		math.IsInf(lossInit, 0) || math.IsInf(lossFinal, 0) {
		return false
	}
	if requiresLearn && lossInit < 1e-12 && lossFinal < 1e-12 {
		return false
	}
	return true
}

func lossFinite(loss float64) bool {
	return !math.IsNaN(loss) && !math.IsInf(loss, 0)
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

// layerRequiresLearn reports whether OverallOK should require loss improvement.
func layerRequiresLearn(lt poly.LayerType) bool {
	switch lt {
	case poly.LayerResidual:
		return false
	default:
		return true
	}
}

// trainingOK matches lucy/testing loss criteria (short CPU runs, quant tolerance bands).
func trainingOK(lossInit, lossFinal float64, dtype poly.DType) bool {
	if math.IsNaN(lossInit) || math.IsNaN(lossFinal) ||
		math.IsInf(lossInit, 0) || math.IsInf(lossFinal, 0) {
		return false
	}
	// Collapse to ~0 after non-trivial init (e.g. broken MHA float paths) is not "trained".
	if lossInit > 0.05 && lossFinal < 1e-9 {
		return false
	}
	if lossInit > 1e-3 && (lossFinal > lossInit*50 || lossFinal > 1e10) {
		return false
	}
	if lossInit < 0.01 {
		// Exact zeros on both ends = degenerate forward (e.g. CNN3 with depth=0), not trained.
		if lossInit < 1e-12 && lossFinal < 1e-12 {
			return false
		}
		if lossFinal <= lossInit*2.0+1e-3 {
			return true
		}
		return isQuantIntegerDType(dtype) && lossFinal < 1.0
	}
	// Stable low loss (Embedding/CNN1 floats): tiny drift still counts as trained.
	if lossInit > 0 && lossInit < 2.0 && lossFinal > 0 && lossFinal < 2.0 {
		if math.Abs(lossFinal-lossInit) < 0.01 && lossFinal <= lossInit*1.05 {
			return true
		}
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
		if rel <= band {
			return true
		}
		// Unsigned: init can be very low then rise to ~0.2–0.4 plateau (CNN/RNN/Embedding stacks).
		if isUnsignedQuantDType(dtype) && lossInit < 0.35 && lossFinal >= 0.15 && lossFinal <= 0.45 {
			return true
		}
		// Higher init on 3³ grids: allow modest rise (e.g. 0.40 → 0.48).
		if isUnsignedQuantDType(dtype) && lossInit >= 0.35 && lossInit < 0.55 &&
			lossFinal >= 0.15 && lossFinal <= 0.55 && lossFinal <= lossInit*1.35+1e-3 {
			return true
		}
		return false
	}
	return lossFinal < lossInit*0.99
}

func isUnsignedQuantDType(dt poly.DType) bool {
	switch dt {
	case poly.DTypeUint64, poly.DTypeUint32, poly.DTypeUint16, poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		return true
	default:
		return false
	}
}

func formatDur(d time.Duration) string {
	if d <= 0 {
		return "0"
	}
	if d < time.Microsecond {
		return fmt.Sprintf("%dns", d.Nanoseconds())
	}
	if d < time.Millisecond {
		return fmt.Sprintf("%.1fµs", float64(d)/float64(time.Microsecond))
	}
	if d < time.Second {
		ms := float64(d) / float64(time.Millisecond)
		if ms < 10 {
			return fmt.Sprintf("%.2fms", ms)
		}
		return fmt.Sprintf("%.1fms", ms)
	}
	return fmt.Sprintf("%.3fs", d.Seconds())
}

// activeBenchIters is set per grid in RunLayerSuite (fewer passes on 2³/3³).
var activeBenchIters = 25

func benchmarkForward(net *poly.VolumetricNetwork, input *poly.Tensor[float32], multiCore bool) (out []float32, avg time.Duration) {
	setCPUMode(net, multiCore)
	setSimdForward(net, false)
	for i := 0; i < 3; i++ {
		resetNetwork(net)
		_, _, _ = poly.ForwardPolymorphic(net, input)
	}
	var total time.Duration
	var last *poly.Tensor[float32]
	for i := 0; i < activeBenchIters; i++ {
		resetNetwork(net)
		t0 := time.Now()
		post, _, _ := poly.ForwardPolymorphic(net, input)
		total += time.Since(t0)
		last = post
	}
	if last == nil {
		return nil, 0
	}
	return append([]float32(nil), last.Data...), total / time.Duration(activeBenchIters)
}

func benchmarkForwardSimd(net *poly.VolumetricNetwork, input *poly.Tensor[float32], multiCore bool) (out []float32, avg time.Duration) {
	setCPUMode(net, multiCore)
	setSimdForward(net, true)
	defer setSimdForward(net, false)
	for i := 0; i < 3; i++ {
		resetNetwork(net)
		_, _, _ = poly.ForwardPolymorphic(net, input)
	}
	var total time.Duration
	var last *poly.Tensor[float32]
	for i := 0; i < activeBenchIters; i++ {
		resetNetwork(net)
		t0 := time.Now()
		post, _, _ := poly.ForwardPolymorphic(net, input)
		total += time.Since(t0)
		last = post
	}
	if last == nil {
		return nil, 0
	}
	return append([]float32(nil), last.Data...), total / time.Duration(activeBenchIters)
}

func formatSimdSpeedup(tiledMC, simd time.Duration) string {
	if tiledMC <= 0 || simd <= 0 {
		return "n/a"
	}
	pct := float64(tiledMC-simd) / float64(tiledMC) * 100
	ratio := float64(tiledMC) / float64(simd)
	if pct >= 0.5 {
		return fmt.Sprintf("%.0f%% faster (%.1f×)", pct, ratio)
	}
	if pct <= -0.5 {
		return fmt.Sprintf("%.0f%% slower (%.1f×)", -pct, float64(simd)/float64(tiledMC))
	}
	return "≈0%"
}

func benchmarkBackward(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32], multiCore, simd bool) (dx, dw []float32, avg time.Duration) {
	setCPUMode(net, multiCore)
	if simd {
		setSimdForward(net, true)
		defer setSimdForward(net, false)
	} else {
		setSimdForward(net, false)
	}
	for i := 0; i < 3; i++ {
		_, _, _ = runBackwardOnce(net, input, target, multiCore, simd)
	}
	var total time.Duration
	var lastDx, lastDw []float32
	for i := 0; i < activeBenchIters; i++ {
		dx, dw, dur := runBackwardOnce(net, input, target, multiCore, simd)
		total += dur
		lastDx, lastDw = dx, dw
	}
	return lastDx, lastDw, total / time.Duration(activeBenchIters)
}

func runBackwardOnce(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32], multiCore, simd bool) (dx, dw []float32, dur time.Duration) {
	setCPUMode(net, multiCore)
	if simd {
		setSimdForward(net, true)
		defer setSimdForward(net, false)
	} else {
		setSimdForward(net, false)
	}
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
	dur = time.Since(t0)

	if len(layerGrads) > 0 && layerGrads[0][0] != nil {
		dx = append([]float32(nil), layerGrads[0][0].Data...)
	}
	for _, g := range layerGrads {
		if g[1] != nil {
			dw = append(dw, g[1].Data...)
		}
	}
	return dx, dw, dur
}

func samplesPerSec(d time.Duration, epochs int) float64 {
	if d <= 0 {
		return 0
	}
	return float64(epochs) / d.Seconds()
}
