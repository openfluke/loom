package poly

// dense_native.go — native dtype dense when UseExactDType is set.
//
// Integer dtypes (Int8, Int4, Ternary, …): int MAC + in-place integer SGD.
// Other dtypes: per-dtype MAC via GetNative (no bulk GetActive FP32 buffer).
// Default FP32-dequant matmul lives in dense.go.

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
)

// DenseExactCache holds integer-native activations for true exact training.
type DenseExactCache struct {
	InputI8        []int8
	InputU8        []uint8  // uint-native SIMD: quantized activations as uint8
	PreI8          []int8   // MHA: attn output int8 before O-proj; else pre-activation
	PostI8         []int8
	QF64           []float64 // MHA: Q after RoPE (for attention backward)
	QI8            []int8    // MHA: quantized Q after RoPE
	KI8            []int8    // MHA: int8 KV cache (batch*msl*kvDim)
	VI8            []int8    // MHA: int8 V cache (batch*msl*kvDim)
	WeightsUpdated bool
}

// IsDenseNativeExactDType reports dtypes that use dense_native instead of dense.go.
func IsDenseNativeExactDType(dtype DType) bool {
	switch dtype {
	case DTypeFloat64, DTypeFloat32:
		return true
	default:
		return isDenseNativeTrainDType(dtype)
	}
}

// IsDenseTrueNativeDType reports dtypes with fully integer forward/backward/update.
func IsDenseTrueNativeDType(dtype DType) bool {
	switch dtype {
	case DTypeInt8, DTypeInt4, DTypeInt2, DTypeTernary, DTypeBinary,
		DTypeUint8, DTypeUint4, DTypeUint2:
		return true
	default:
		return false
	}
}

// DenseUsesNativeExact reports whether a dense layer uses dense_native.go.
func DenseUsesNativeExact(layer *VolumetricLayer) bool {
	return useDenseNativeExact(layer)
}

// DenseUsesTrueNative reports whether a dense layer uses the integer-native path.
func DenseUsesTrueNative(layer *VolumetricLayer) bool {
	return useDenseTrueNative(layer)
}

func useDenseNativeExact(layer *VolumetricLayer) bool {
	return layer != nil &&
		layer.Network != nil &&
		layer.Network.UseExactDType &&
		layer.Type == LayerDense &&
		layer.WeightStore != nil &&
		IsDenseNativeExactDType(layer.DType)
}

func useDenseTrueNative(layer *VolumetricLayer) bool {
	return layer != nil &&
		layer.Network != nil &&
		layer.Network.UseExactDType &&
		layer.Type == LayerDense &&
		layer.WeightStore != nil &&
		IsDenseTrueNativeDType(layer.DType)
}

// DenseForwardNativeExact runs dense forward in storage dtype rules.
func DenseForwardNativeExact[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	in, ok := any(input).(*Tensor[float32])
	if !ok {
		return DenseForwardTiled(layer, input)
	}
	var preF, postF *Tensor[float32]
	if layerUseSimdForward(layer) {
		if pre, post, simdOK := tryDenseForwardNativeSimd(layer, in); simdOK {
			preF, postF = pre, post
		}
	}
	if preF == nil {
		if useDenseTrueNative(layer) {
			preF, postF = denseForwardIntegerNative(layer, in)
		} else {
			preF, postF = denseForwardNativeMAC(layer, in)
		}
	}
	pre, post, ok2 := nativeTensorsAs[T](preF, postF)
	if !ok2 {
		return DenseForwardTiled(layer, input)
	}
	return pre, post
}

// DenseBackwardNativeExact runs dense backward in storage dtype rules.
func DenseBackwardNativeExact[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	in, okIn := any(input).(*Tensor[float32])
	goT, okGO := any(gradOutput).(*Tensor[float32])
	preF, okPre := any(preAct).(*Tensor[float32])
	if !okIn || !okGO || !okPre {
		return DenseBackwardTiled(layer, gradOutput, input, preAct)
	}
	var giF, gwF *Tensor[float32]
	if layerUseSimdForward(layer) {
		if gi, gw, simdOK := tryDenseBackwardNativeSimd(layer, goT, in, preF); simdOK {
			giF, gwF = gi, gw
		}
	}
	if giF == nil {
		if useDenseTrueNative(layer) {
			giF, gwF = denseBackwardIntegerNative(layer, goT, in)
		} else {
			giF, gwF = denseBackwardNativeMAC(layer, goT, in, preF)
		}
	}
	gi, okGI := nativeTensorAs[T](giF)
	gw, okGW := nativeTensorAs[T](gwF)
	if !okGI || !okGW {
		return DenseBackwardTiled(layer, gradOutput, input, preAct)
	}
	return gi, gw
}

func nativeTensorsAs[T Numeric](pre, post *Tensor[float32]) (*Tensor[T], *Tensor[T], bool) {
	var zero T
	if _, isF32 := any(zero).(float32); isF32 {
		return any(pre).(*Tensor[T]), any(post).(*Tensor[T]), true
	}
	preAct := NewTensor[T](pre.Shape...)
	postAct := NewTensor[T](post.Shape...)
	for i := range pre.Data {
		preAct.Data[i] = T(pre.Data[i])
		postAct.Data[i] = T(post.Data[i])
	}
	return preAct, postAct, true
}

func nativeTensorAs[T Numeric](t *Tensor[float32]) (*Tensor[T], bool) {
	if t == nil {
		return nil, false
	}
	var zero T
	if _, isF32 := any(zero).(float32); isF32 {
		return any(t).(*Tensor[T]), true
	}
	out := NewTensor[T](t.Shape...)
	for i := range t.Data {
		out.Data[i] = T(t.Data[i])
	}
	return out, true
}

// --- Integer-native (Int8 / Int4 / Ternary / …) --------------------------------

func ensureDenseExactCache(layer *VolumetricLayer, batch, inSz, outSz int) *DenseExactCache {
	if layer.ExactDense == nil {
		layer.ExactDense = &DenseExactCache{}
	}
	c := layer.ExactDense
	needIn := batch * inSz
	needOut := batch * outSz
	if cap(c.InputI8) < needIn {
		c.InputI8 = make([]int8, needIn)
	} else {
		c.InputI8 = c.InputI8[:needIn]
	}
	if cap(c.InputU8) < needIn {
		c.InputU8 = make([]uint8, needIn)
	} else {
		c.InputU8 = c.InputU8[:needIn]
	}
	if cap(c.PreI8) < needOut {
		c.PreI8 = make([]int8, needOut)
		c.PostI8 = make([]int8, needOut)
	} else {
		c.PreI8 = c.PreI8[:needOut]
		c.PostI8 = c.PostI8[:needOut]
	}
	return c
}

func clampI8(v int32) int8 {
	if v > 127 {
		return 127
	}
	if v < -128 {
		return -128
	}
	return int8(v)
}

func clampU8(v int32) uint8 {
	if v > 255 {
		return 255
	}
	if v < 0 {
		return 0
	}
	return uint8(v)
}

func quantizeRowF32ToI8(row []float32, scale float32) []int8 {
	out := make([]int8, len(row))
	if scale == 0 {
		scale = 1
	}
	for i, v := range row {
		out[i] = clampI8(int32(math.Round(float64(v) / float64(scale))))
	}
	return out
}

func quantizeRowF32ToU8(row []float32, scale float32) []uint8 {
	out := make([]uint8, len(row))
	if scale == 0 {
		scale = 1
	}
	for i, v := range row {
		q := int32(math.Round(float64(v) / float64(scale)))
		out[i] = clampU8(q)
	}
	return out
}

func dequantI8Row(codes []int8, scale float32, out []float32) {
	if scale == 0 {
		scale = 1
	}
	for i, c := range codes {
		out[i] = float32(c) * scale
	}
}

func dequantU8Row(codes []uint8, scale float32, out []float32) {
	if scale == 0 {
		scale = 1
	}
	for i, c := range codes {
		out[i] = float32(c) * scale
	}
}

func trueNativeWeightI8(dtype DType, code uint8) int8 {
	switch dtype {
	case DTypeTernary:
		v := int8(code)
		if v > 1 {
			return 1
		}
		if v < -1 {
			return -1
		}
		return v
	case DTypeBinary:
		if code == 0 || code == 255 {
			return -1
		}
		return 1
	case DTypeInt4:
		v := int8(code)
		if v > 7 {
			return 7
		}
		if v < -8 {
			return -8
		}
		return v
	case DTypeInt2:
		v := int8(code)
		if v > 1 {
			return 1
		}
		if v < -2 {
			return -2
		}
		return v
	default:
		return int8(code)
	}
}

func nativeWeightsI8(ws *WeightStore, dtype DType) []int8 {
	return ws.NativeSimdI8Weights(dtype)
}

func nativeWeightsU8(ws *WeightStore, dtype DType) []uint8 {
	return ws.NativeSimdU8Weights(dtype)
}

func learningRateToBitShift(lr float32) uint {
	if lr <= 0 {
		return 10
	}
	shift := int(math.Round(-math.Log2(float64(lr))))
	if shift < 4 {
		shift = 4
	}
	if shift > 16 {
		shift = 16
	}
	return uint(shift)
}

// TrueInt8DenseForward: int8 weights × int8 activations → int8 pre-activation.
func TrueInt8DenseForward(weights, inputs []int8, batch, inSz, outSz int, pre, post []int8, act ActivationType) {
	for b := 0; b < batch; b++ {
		inOff := b * inSz
		outOff := b * outSz
		for o := 0; o < outSz; o++ {
			acc := int8DotRowAcc(weights, inputs[inOff:inOff+inSz], o*inSz, inSz)
			pre[outOff+o] = clampI8(acc >> 8)
			if act == ActivationReLU && pre[outOff+o] <= 0 {
				post[outOff+o] = 0
			} else {
				post[outOff+o] = pre[outOff+o]
			}
		}
	}
}

// TrueUint8DenseForward: uint8 weights × uint8 activations → int8 pre-activation.
func TrueUint8DenseForward(weights, inputs []uint8, batch, inSz, outSz int, pre, post []int8) {
	for b := 0; b < batch; b++ {
		inOff := b * inSz
		outOff := b * outSz
		for o := 0; o < outSz; o++ {
			acc := uint8DotRowAcc(weights, inputs[inOff:inOff+inSz], o*inSz, inSz)
			pre[outOff+o] = int8(clampU8(acc >> 8))
			post[outOff+o] = pre[outOff+o]
		}
	}
}

// TrueInt8DenseBackwardUpdate: integer backward + stochastic SGD on int8 weights.
func TrueInt8DenseBackwardUpdate(
	weights []int8,
	inputs []int8,
	gradOutput []int8,
	batch, inSz, outSz int,
	lrBitShift uint,
	act ActivationType,
	preI8 []int8,
) ([]int8, []int8) {
	gradInput := make([]int8, batch*inSz)
	gradInputAcc := make([]int32, batch*inSz)
	gradWeights := make([]int32, outSz*inSz)

	gradPre := make([]int8, batch*outSz)
	for i := range gradOutput {
		g := gradOutput[i]
		if act == ActivationReLU && preI8[i] <= 0 {
			g = 0
		}
		gradPre[i] = g
	}

	for b := 0; b < batch; b++ {
		inOff := b * inSz
		outOff := b * outSz
		for o := 0; o < outSz; o++ {
			gOut := int32(gradPre[outOff+o])
			rowOff := o * inSz
			inRow := inputs[inOff : inOff+inSz]
			int8AccumWeightGrad(gradWeights, weights, inRow, gOut, rowOff, inSz)
			int8AccumInputGrad(gradInputAcc[inOff:inOff+inSz], weights, gOut, rowOff, inSz)
		}
	}
	for i, acc := range gradInputAcc {
		gradInput[i] = clampI8(acc)
	}

	mask := int32((1 << lrBitShift) - 1)
	for i := range weights {
		scaledGrad := gradWeights[i] >> lrBitShift
		remainder := gradWeights[i] & mask
		if remainder > rand.Int31n(1<<lrBitShift) {
			scaledGrad++
		}
		next := int32(weights[i]) - scaledGrad
		weights[i] = clampI8(next)
	}
	return gradInput, weights
}

func trueUint8DenseBackwardUpdate(weights []uint8, inputs, gradOutput []uint8, batch, inSz, outSz int, lrBitShift uint) ([]uint8, []uint8) {
	gradInputAcc := make([]int32, batch*inSz)
	gradWeights := make([]int32, outSz*inSz)

	for b := 0; b < batch; b++ {
		inOff := b * inSz
		outOff := b * outSz
		for o := 0; o < outSz; o++ {
			gOut := int32(gradOutput[outOff+o])
			rowOff := o * inSz
			inRow := inputs[inOff : inOff+inSz]
			uint8AccumWeightGrad(gradWeights, inRow, gOut, rowOff, inSz)
			uint8AccumInputGrad(gradInputAcc[inOff:inOff+inSz], weights, gOut, rowOff, inSz)
		}
	}
	gradInput := make([]uint8, batch*inSz)
	for i, acc := range gradInputAcc {
		gradInput[i] = clampU8(acc)
	}
	mask := int32((1 << lrBitShift) - 1)
	for i := range weights {
		scaledGrad := gradWeights[i] >> lrBitShift
		if (gradWeights[i] & mask) > rand.Int31n(1<<lrBitShift) {
			scaledGrad++
		}
		next := int32(weights[i]) - scaledGrad
		weights[i] = clampU8(next)
	}
	return gradInput, weights
}

func denseForwardIntegerNative(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	ws := layer.WeightStore
	ws.Morph(layer.DType)
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}

	batch := input.Shape[0]
	inSz := layer.InputHeight
	outSz := layer.OutputHeight
	cache := ensureDenseExactCache(layer, batch, inSz, outSz)

	for b := 0; b < batch; b++ {
		row := input.Data[b*inSz : (b+1)*inSz]
		codes := quantizeRowF32ToI8(row, scale)
		copy(cache.InputI8[b*inSz:(b+1)*inSz], codes)
	}

	preAct = NewTensor[float32](batch, outSz)
	postAct = NewTensor[float32](batch, outSz)

	switch layer.DType {
	case DTypeUint8, DTypeUint4, DTypeUint2:
		w := nativeWeightsU8(ws, layer.DType)
		if w == nil {
			return denseForwardNativeMAC(layer, input)
		}
		inU8 := make([]uint8, len(cache.InputI8))
		for i, c := range cache.InputI8 {
			inU8[i] = uint8(clampU8(int32(c)))
		}
		TrueUint8DenseForward(w, inU8, batch, inSz, outSz, cache.PreI8, cache.PostI8)
	default:
		w := nativeWeightsI8(ws, layer.DType)
		if w == nil {
			return denseForwardNativeMAC(layer, input)
		}
		TrueInt8DenseForward(w, cache.InputI8, batch, inSz, outSz, cache.PreI8, cache.PostI8, layer.Activation)
	}

	dequantI8Row(cache.PostI8, scale, postAct.Data)
	copy(preAct.Data, postAct.Data)
	if layer.Activation == ActivationLinear {
		copy(cache.PreI8, cache.PostI8)
	}
	return preAct, postAct
}

func denseBackwardIntegerNative(layer *VolumetricLayer, gradOutput, input *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	cache := layer.ExactDense
	if cache == nil || len(cache.InputI8) == 0 {
		return denseBackwardNativeMAC(layer, gradOutput, input, gradOutput)
	}

	ws := layer.WeightStore
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	batch := input.Shape[0]
	inSz := layer.InputHeight
	outSz := layer.OutputHeight
	lrShift := learningRateToBitShift(layer.Network.ExactNativeLR)

	gradOutI8 := quantizeRowF32ToI8(gradOutput.Data, scale)

	gradInput = NewTensor[float32](batch, inSz)
	gradWeights = NewTensor[float32](outSz, inSz)

	switch layer.DType {
	case DTypeUint8, DTypeUint4, DTypeUint2:
		w := nativeWeightsU8(ws, layer.DType)
		inU8 := make([]uint8, len(cache.InputI8))
		for i, c := range cache.InputI8 {
			inU8[i] = uint8(clampU8(int32(c)))
		}
		gOutU8 := make([]uint8, len(gradOutI8))
		for i, c := range gradOutI8 {
			gOutU8[i] = uint8(clampU8(int32(c)))
		}
		giU8, wU8 := trueUint8DenseBackwardUpdate(w, inU8, gOutU8, batch, inSz, outSz, lrShift)
		ws.Versions[layer.DType] = wU8
		dequantU8Row(giU8, scale, gradInput.Data)
	default:
		w := nativeWeightsI8(ws, layer.DType)
		giI8, wI8 := TrueInt8DenseBackwardUpdate(
			w, cache.InputI8, gradOutI8,
			batch, inSz, outSz,
			lrShift, layer.Activation, cache.PreI8,
		)
		raw := make([]uint8, len(wI8))
		for i, v := range wI8 {
			raw[i] = uint8(v)
		}
		ws.Versions[layer.DType] = raw
		ws.Master = nil
		dequantI8Row(giI8, scale, gradInput.Data)
	}

	cache.WeightsUpdated = true
	ws.GPUWeights = make(map[DType]any)
	if ws.CPUPacked != nil {
		delete(ws.CPUPacked, layer.DType)
	}
	ws.invalidateNativeSimdCache(layer.DType)
	return gradInput, gradWeights
}

// --- Per-dtype native MAC (Float16, Int64, FP8, …) ----------------------------

func denseForwardNativeMAC(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	ws := layer.WeightStore
	ws.Morph(layer.DType)

	batchSize := input.Shape[0]
	inputSize := layer.InputHeight
	if inputSize <= 0 {
		inputSize = input.Shape[len(input.Shape)-1]
	}
	outputSize := layer.OutputHeight

	preAct = NewTensor[float32](batchSize, outputSize)
	postAct = NewTensor[float32](batchSize, outputSize)

	if layer.EnableMultiCoreTiling && outputSize > 8 {
		denseNativeForwardParallel(layer, input, preAct)
	} else {
		denseNativeForwardSerial(layer, input, preAct)
	}
	for i := range postAct.Data {
		postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
	}
	return preAct, postAct
}

func denseNativeForwardSerial(layer *VolumetricLayer, input, preAct *Tensor[float32]) {
	batch := input.Shape[0]
	inSz := layer.InputHeight
	outSz := layer.OutputHeight
	for b := 0; b < batch; b++ {
		inRow := input.Data[b*inSz : (b+1)*inSz]
		outRow := preAct.Data[b*outSz : (b+1)*outSz]
		for o := 0; o < outSz; o++ {
			outRow[o] = denseNativeDotForward(layer, inRow, o*inSz, inSz)
		}
	}
}

func denseNativeForwardParallel(layer *VolumetricLayer, input, preAct *Tensor[float32]) {
	batch := input.Shape[0]
	inSz := layer.InputHeight
	outSz := layer.OutputHeight
	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	sem := make(chan struct{}, numCPUs)
	for b := 0; b < batch; b++ {
		for oTile := 0; oTile < outSz; oTile += 16 {
			sem <- struct{}{}
			wg.Add(1)
			go func(b, oTile int) {
				defer func() { <-sem; wg.Done() }()
				oEnd := oTile + 16
				if oEnd > outSz {
					oEnd = outSz
				}
				inRow := input.Data[b*inSz : (b+1)*inSz]
				outRow := preAct.Data[b*outSz : (b+1)*outSz]
				for o := oTile; o < oEnd; o++ {
					outRow[o] = denseNativeDotForward(layer, inRow, o*inSz, inSz)
				}
			}(b, oTile)
		}
	}
	wg.Wait()
}

func denseNativeDotForward(layer *VolumetricLayer, input []float32, rowOff, inSz int) float32 {
	ws := layer.WeightStore
	dt := layer.DType
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	native := ws.GetNative(dt)
	if native == nil {
		return 0
	}

	switch dt {
	case DTypeFloat64:
		w := native.([]float64)
		var sum float64
		for i := 0; i < inSz; i++ {
			sum += w[rowOff+i] * float64(input[i])
		}
		return float32(sum)
	case DTypeFloat32:
		w := ws.Master
		var sum float32
		for i := 0; i < inSz; i++ {
			sum += w[rowOff+i] * input[i]
		}
		return sum
	case DTypeFloat16:
		w := native.([]uint16)
		sum := float32(0)
		for i := 0; i < inSz; i++ {
			wv := float16ToFloat32(w[rowOff+i])
			prod := float16ToFloat32(float32ToFloat16(wv * input[i]))
			sum = float16ToFloat32(float32ToFloat16(sum + prod))
		}
		return sum
	case DTypeBFloat16:
		w := native.([]uint16)
		sum := float32(0)
		for i := 0; i < inSz; i++ {
			wv := bfloat16ToFloat32(w[rowOff+i])
			prod := bfloat16ToFloat32(float32ToBFloat16(wv * input[i]))
			sum = bfloat16ToFloat32(float32ToBFloat16(sum + prod))
		}
		return sum
	case DTypeFP8E4M3:
		w := native.([]uint8)
		sum := float32(0)
		for i := 0; i < inSz; i++ {
			wv := e4m3ToFloat32(w[rowOff+i]) * scale
			p := e4m3ToFloat32(float32ToE4M3(wv*input[i] / scale)) * scale
			sum = e4m3ToFloat32(float32ToE4M3(sum/scale+p/scale)) * scale
		}
		return sum
	case DTypeFP8E5M2:
		w := native.([]uint8)
		sum := float32(0)
		for i := 0; i < inSz; i++ {
			wv := e5m2ToFloat32(w[rowOff+i]) * scale
			p := e5m2ToFloat32(float32ToE5M2(wv*input[i] / scale)) * scale
			sum = e5m2ToFloat32(float32ToE5M2(sum/scale+p/scale)) * scale
		}
		return sum
	case DTypeInt64:
		w := native.([]int64)
		var acc int64
		for i := 0; i < inSz; i++ {
			acc += w[rowOff+i] * int64(math.Round(float64(input[i])/float64(scale)))
		}
		return float32(acc) * scale * scale
	case DTypeUint64:
		w := native.([]uint64)
		var acc uint64
		for i := 0; i < inSz; i++ {
			xq := uint64(math.Max(0, math.Round(float64(input[i])/float64(scale))))
			acc += w[rowOff+i] * xq
		}
		return float32(acc) * scale * scale
	case DTypeInt32:
		w := native.([]int32)
		var acc int64
		for i := 0; i < inSz; i++ {
			acc += int64(w[rowOff+i]) * int64(math.Round(float64(input[i])/float64(scale)))
		}
		return float32(acc) * scale * scale
	case DTypeUint32:
		w := native.([]uint32)
		var acc int64
		for i := 0; i < inSz; i++ {
			xq := int64(math.Max(0, math.Round(float64(input[i])/float64(scale))))
			acc += int64(w[rowOff+i]) * xq
		}
		return float32(acc) * scale * scale
	case DTypeInt16:
		w := native.([]int16)
		var acc int64
		for i := 0; i < inSz; i++ {
			acc += int64(w[rowOff+i]) * int64(math.Round(float64(input[i])/float64(scale)))
		}
		return float32(acc) * scale * scale
	case DTypeUint16:
		w := native.([]uint16)
		var acc int64
		for i := 0; i < inSz; i++ {
			xq := int64(math.Max(0, math.Round(float64(input[i])/float64(scale))))
			acc += int64(w[rowOff+i]) * xq
		}
		return float32(acc) * scale * scale
	case DTypeFP4:
		w := native.([]uint8)
		var acc float64
		for i := 0; i < inSz; i++ {
			wv := fp4CodeToFloat32(w[rowOff+i], scale)
			acc += float64(wv) * float64(input[i])
		}
		return float32(acc)
	default:
		codes, ok := nativeU8WeightsView(native)
		if !ok {
			return 0
		}
		var acc int64
		for i := 0; i < inSz; i++ {
			wv := denseNativeSignedU8Weight(dt, codes[rowOff+i], scale)
			xq := int64(math.Round(float64(input[i]) / float64(scale)))
			acc += int64(wv) * xq
		}
		return float32(acc) * scale * scale
	}
}

func denseNativeSignedU8Weight(dt DType, code uint8, scale float32) int32 {
	switch dt {
	case DTypeUint8, DTypeUint4, DTypeUint2:
		return int32(code)
	case DTypeFP4:
		return int32(math.Round(float64(fp4CodeToFloat32(code, scale) / scale)))
	default:
		return int32(int8(code))
	}
}

func denseBackwardNativeMAC(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	ws := layer.WeightStore
	ws.Morph(layer.DType)

	batchSize := input.Shape[0]
	inputSize := layer.InputHeight
	outputSize := layer.OutputHeight

	gradInput = NewTensor[float32](batchSize, inputSize)
	gradWeights = NewTensor[float32](outputSize, inputSize)
	gradPre := denseGradPreAct(gradOutput, preAct, layer.Activation)

	if layer.EnableMultiCoreTiling && outputSize > 8 {
		denseNativeBackwardParallel(layer, gradInput, gradWeights, input, gradPre, batchSize, inputSize, outputSize)
	} else {
		denseNativeBackwardSerial(layer, gradInput, gradWeights, input, gradPre, batchSize, inputSize, outputSize)
	}
	return gradInput, gradWeights
}

func denseNativeBackwardSerial(layer *VolumetricLayer, gradInput, gradWeights *Tensor[float32], input *Tensor[float32], gradPre []float64, batch, inSz, outSz int) {
	gwAcc := make([]float64, len(gradWeights.Data))
	giAcc := make([]float64, len(gradInput.Data))

	for b := 0; b < batch; b++ {
		inRow := input.Data[b*inSz : (b+1)*inSz]
		outOff := b * outSz
		for o := 0; o < outSz; o++ {
			g := gradPre[outOff+o]
			rowOff := o * inSz
			for i := 0; i < inSz; i++ {
				gwAcc[rowOff+i] += denseNativeGradWTerm(layer, inRow[i], g)
				giAcc[b*inSz+i] += denseNativeGradXTerm(layer, rowOff+i, g)
			}
		}
	}
	for i := range gradWeights.Data {
		gradWeights.Data[i] = float32(gwAcc[i])
	}
	for i := range gradInput.Data {
		gradInput.Data[i] = float32(giAcc[i])
	}
}

func denseNativeBackwardParallel(layer *VolumetricLayer, gradInput, gradWeights *Tensor[float32], input *Tensor[float32], gradPre []float64, batch, inSz, outSz int) {
	gwAcc := make([]float64, len(gradWeights.Data))
	giAcc := make([]float64, len(gradInput.Data))
	var mu sync.Mutex
	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	sem := make(chan struct{}, numCPUs)

	for o := 0; o < outSz; o++ {
		sem <- struct{}{}
		wg.Add(1)
		go func(o int) {
			defer func() { <-sem; wg.Done() }()
			rowOff := o * inSz
			localGW := make([]float64, inSz)
			for b := 0; b < batch; b++ {
				g := gradPre[b*outSz+o]
				inRow := input.Data[b*inSz : (b+1)*inSz]
				for i := 0; i < inSz; i++ {
					localGW[i] += denseNativeGradWTerm(layer, inRow[i], g)
				}
			}
			mu.Lock()
			for i := 0; i < inSz; i++ {
				gwAcc[rowOff+i] += localGW[i]
			}
			mu.Unlock()
		}(o)
	}
	wg.Wait()

	for b := 0; b < batch; b++ {
		sem <- struct{}{}
		wg.Add(1)
		go func(b int) {
			defer func() { <-sem; wg.Done() }()
			localGI := make([]float64, inSz)
			inOff := b * inSz
			outOff := b * outSz
			for o := 0; o < outSz; o++ {
				g := gradPre[outOff+o]
				rowOff := o * inSz
				for i := 0; i < inSz; i++ {
					localGI[i] += denseNativeGradXTerm(layer, rowOff+i, g)
				}
			}
			mu.Lock()
			for i := 0; i < inSz; i++ {
				giAcc[inOff+i] += localGI[i]
			}
			mu.Unlock()
		}(b)
	}
	wg.Wait()

	for i := range gradWeights.Data {
		gradWeights.Data[i] = float32(gwAcc[i])
	}
	for i := range gradInput.Data {
		gradInput.Data[i] = float32(giAcc[i])
	}
}

func denseNativeGradWTerm(layer *VolumetricLayer, inputVal float32, gradPre float64) float64 {
	dt := layer.DType
	scale := layer.WeightStore.Scale
	if scale == 0 {
		scale = 1
	}
	switch dt {
	case DTypeFloat16, DTypeBFloat16, DTypeFP8E4M3, DTypeFP8E5M2:
		return float64(inputVal) * gradPre
	case DTypeInt64, DTypeInt32, DTypeInt16, DTypeUint64, DTypeUint32, DTypeUint16:
		xq := math.Round(float64(inputVal) / float64(scale))
		return xq * gradPre
	default:
		if isDenseNativeQuantU8(dt) {
			xq := math.Round(float64(inputVal) / float64(scale))
			return xq * gradPre
		}
		return float64(inputVal) * gradPre
	}
}

func denseNativeGradXTerm(layer *VolumetricLayer, weightIdx int, gradPre float64) float64 {
	ws := layer.WeightStore
	dt := layer.DType
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	native := ws.GetNative(dt)
	if native == nil {
		return 0
	}

	switch dt {
	case DTypeFloat64:
		w := native.([]float64)
		return w[weightIdx] * gradPre
	case DTypeFloat32:
		return float64(ws.Master[weightIdx]) * gradPre
	case DTypeFloat16:
		w := native.([]uint16)
		wv := float16ToFloat32(w[weightIdx])
		return float64(wv) * gradPre
	case DTypeBFloat16:
		w := native.([]uint16)
		wv := bfloat16ToFloat32(w[weightIdx])
		return float64(wv) * gradPre
	case DTypeFP8E4M3:
		w := native.([]uint8)
		wv := e4m3ToFloat32(w[weightIdx]) * scale
		return float64(wv) * gradPre
	case DTypeFP8E5M2:
		w := native.([]uint8)
		wv := e5m2ToFloat32(w[weightIdx]) * scale
		return float64(wv) * gradPre
	case DTypeInt64:
		w := native.([]int64)
		return float64(w[weightIdx]) * float64(scale) * gradPre
	case DTypeUint64:
		w := native.([]uint64)
		return float64(w[weightIdx]) * float64(scale) * gradPre
	case DTypeInt32:
		w := native.([]int32)
		return float64(w[weightIdx]) * float64(scale) * gradPre
	case DTypeUint32:
		w := native.([]uint32)
		return float64(w[weightIdx]) * float64(scale) * gradPre
	case DTypeInt16:
		w := native.([]int16)
		return float64(w[weightIdx]) * float64(scale) * gradPre
	case DTypeUint16:
		w := native.([]uint16)
		return float64(w[weightIdx]) * float64(scale) * gradPre
	case DTypeFP4:
		w := native.([]uint8)
		wv := fp4CodeToFloat32(w[weightIdx], scale)
		return float64(wv) * gradPre
	default:
		codes, ok := nativeU8WeightsView(native)
		if !ok {
			return 0
		}
		wv := denseNativeSignedU8Weight(dt, codes[weightIdx], scale)
		return float64(wv) * float64(scale) * gradPre
	}
}

func isDenseNativeQuantU8(dt DType) bool {
	switch dt {
	case DTypeInt8, DTypeUint8, DTypeInt4, DTypeUint4, DTypeInt2, DTypeUint2, DTypeTernary, DTypeBinary:
		return true
	default:
		return false
	}
}
