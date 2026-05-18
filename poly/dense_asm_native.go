package poly

import (
	"math"

	"github.com/openfluke/loom/poly/asm/matmul"
)

// denseForwardAsmNative runs dtype-native integer matmul in asm (no FP inside the dot).
// Morphed weights in WeightStore are used directly (one quant per byte/word) — not
// repacked into uint32 bitstreams for the hot path.
func denseForwardAsmNative[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	batchSize := input.Shape[0]
	inputSize := layer.InputHeight
	outputSize := layer.OutputHeight
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 32
	}

	scale := layer.WeightStore.Scale
	if scale == 0 {
		scale = 1
	}

	preAct = NewTensor[T](batchSize, outputSize)
	postAct = NewTensor[T](batchSize, outputSize)

	switch layer.DType {
	case DTypeInt16, DTypeUint16:
		denseForwardAsmNativeI16(layer, input, preAct, batchSize, inputSize, outputSize, tileSize, scale)
	case DTypeInt32, DTypeUint32:
		denseForwardAsmNativeI32(layer, input, preAct, batchSize, inputSize, outputSize, tileSize, scale)
	case DTypeInt64:
		denseForwardAsmNativeI64(layer, input, preAct, batchSize, inputSize, outputSize, tileSize, scale)
	case DTypeUint64:
		denseForwardAsmNativeU64(layer, input, preAct, batchSize, inputSize, outputSize, tileSize, scale)
	default:
		denseForwardAsmMorphU8(layer, input, preAct, batchSize, inputSize, outputSize, tileSize, scale)
	}

	for i := range postAct.Data {
		postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
	}
	return preAct, postAct
}

// denseForwardAsmMorphU8 handles dtypes morphed to []uint8 (Int8/Int4/Binary/Ternary/…).
func denseForwardAsmMorphU8[T Numeric](
	layer *VolumetricLayer,
	input *Tensor[T],
	preAct *Tensor[T],
	batch, inDim, outDim, tileSize int,
	scale float32,
) {
	wRaw := morphWeightsU8(layer.WeightStore, layer.DType)
	if wRaw == nil {
		denseForwardAsmFloat(layer, input, preAct, batch, inDim, outDim, tileSize)
		return
	}

	qIn := make([]uint8, len(input.Data))
	quantizeInputNative(layer.DType, input.Data, scale, qIn)
	qOut := make([]uint8, len(preAct.Data))

	signed := dtypeNativeSignedDot(layer.DType)
	mc := layer.EnableMultiCoreTiling
	matmul.ForwardNativeU8(qOut, qIn, wRaw, batch, inDim, outDim, mc, tileSize, signed)
	dequantMorphU8ToTensor(preAct.Data, qOut, scale*scale, signed)
}

func morphWeightsU8(ws *WeightStore, dtype DType) []uint8 {
	if ws == nil {
		return nil
	}
	ws.Morph(dtype)
	switch w := ws.Versions[dtype].(type) {
	case []uint8:
		return w
	case []int8:
		out := make([]uint8, len(w))
		for i, v := range w {
			out[i] = uint8(v)
		}
		return out
	default:
		return nil
	}
}

func dtypeNativeSignedDot(dtype DType) bool {
	switch dtype {
	case DTypeUint8, DTypeUint4, DTypeUint2, DTypeFP4:
		return false
	default:
		return true
	}
}

func quantizeInputNative[T Numeric](dtype DType, in []T, scale float32, out []uint8) {
	if scale == 0 {
		scale = 1
	}
	for i, v := range in {
		out[i] = nativeQuantValue(dtype, float32(v)/scale)
	}
}

func denseForwardAsmFloat[T Numeric](layer *VolumetricLayer, input *Tensor[T], preAct *Tensor[T], batch, inDim, outDim, tileSize int) {
	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)
	denseForwardAsmTyped(layer, input, preAct, wData, batch, inDim, outDim, tileSize)
}

func dequantMorphU8ToTensor[T Numeric](dst []T, q []uint8, scale float32, signed bool) {
	for i := range dst {
		var v float32
		if signed {
			v = float32(int8(q[i])) * scale
		} else {
			v = float32(q[i]) * scale
		}
		dst[i] = T(v)
	}
}

func isNativeQuantDType(dt DType) bool {
	switch dt {
	case DTypeInt8, DTypeUint8,
		DTypeInt16, DTypeUint16, DTypeInt32, DTypeUint32,
		DTypeInt64, DTypeUint64,
		DTypeInt4, DTypeUint4, DTypeFP4,
		DTypeInt2, DTypeUint2, DTypeTernary, DTypeBinary:
		return true
	default:
		return false
	}
}

func denseForwardAsmNativeI16[T Numeric](
	layer *VolumetricLayer,
	input *Tensor[T],
	preAct *Tensor[T],
	batch, inDim, outDim, tileSize int,
	scale float32,
) {
	layer.WeightStore.Morph(layer.DType)
	w16, ok := layer.WeightStore.Versions[layer.DType].([]int16)
	if !ok {
		if wU, ok2 := layer.WeightStore.Versions[layer.DType].([]uint16); ok2 {
			denseForwardAsmNativeU16(layer, input, preAct, batch, inDim, outDim, tileSize, scale, wU)
			return
		}
		denseForwardAsmFloat(layer, input, preAct, batch, inDim, outDim, tileSize)
		return
	}
	qIn := make([]int16, len(input.Data))
	qOut := make([]int16, len(preAct.Data))
	quantizeInputToInt16(input.Data, scale, qIn)
	mc := layer.EnableMultiCoreTiling
	matmul.ForwardNativeI16(qOut, qIn, w16, batch, inDim, outDim, mc, tileSize)
	dequantToTensorInt(preAct.Data, qOut, scale*scale)
}

func denseForwardAsmNativeU16[T Numeric](
	layer *VolumetricLayer,
	input *Tensor[T],
	preAct *Tensor[T],
	batch, inDim, outDim, tileSize int,
	scale float32,
	wU []uint16,
) {
	qIn := make([]uint16, len(input.Data))
	qOut := make([]uint16, len(preAct.Data))
	quantizeInputToUint16(input.Data, scale, qIn)
	mc := layer.EnableMultiCoreTiling
	matmul.ForwardNativeU16(qOut, qIn, wU, batch, inDim, outDim, mc, tileSize)
	dequantToTensorUint(preAct.Data, qOut, scale*scale)
}

func denseForwardAsmNativeI32[T Numeric](
	layer *VolumetricLayer,
	input *Tensor[T],
	preAct *Tensor[T],
	batch, inDim, outDim, tileSize int,
	scale float32,
) {
	layer.WeightStore.Morph(layer.DType)
	w32, ok := layer.WeightStore.Versions[layer.DType].([]int32)
	if !ok {
		if wU, ok2 := layer.WeightStore.Versions[layer.DType].([]uint32); ok2 {
			qIn := make([]uint32, len(input.Data))
			qOut := make([]uint32, len(preAct.Data))
			quantizeInputToUint32(input.Data, scale, qIn)
			mc := layer.EnableMultiCoreTiling
			matmul.ForwardNativeU32(qOut, qIn, wU, batch, inDim, outDim, mc, tileSize)
			dequantToTensorU32(preAct.Data, qOut, scale*scale)
			return
		}
		denseForwardAsmFloat(layer, input, preAct, batch, inDim, outDim, tileSize)
		return
	}
	qIn := make([]int32, len(input.Data))
	qOut := make([]int32, len(preAct.Data))
	quantizeInputToInt32(input.Data, scale, qIn)
	mc := layer.EnableMultiCoreTiling
	matmul.ForwardNativeI32(qOut, qIn, w32, batch, inDim, outDim, mc, tileSize)
	dequantToTensorInt32(preAct.Data, qOut, scale*scale)
}

func denseForwardAsmNativeI64[T Numeric](
	layer *VolumetricLayer,
	input *Tensor[T],
	preAct *Tensor[T],
	batch, inDim, outDim, tileSize int,
	scale float32,
) {
	layer.WeightStore.Morph(layer.DType)
	w64, ok := layer.WeightStore.Versions[layer.DType].([]int64)
	if !ok {
		denseForwardAsmFloat(layer, input, preAct, batch, inDim, outDim, tileSize)
		return
	}
	qIn := make([]int64, len(input.Data))
	qOut := make([]int64, len(preAct.Data))
	quantizeInputToInt64(input.Data, scale, qIn)
	mc := layer.EnableMultiCoreTiling
	matmul.ForwardNativeI64(qOut, qIn, w64, batch, inDim, outDim, mc, tileSize)
	dequantToTensorI64(preAct.Data, qOut, scale*scale)
}

func denseForwardAsmNativeU64[T Numeric](
	layer *VolumetricLayer,
	input *Tensor[T],
	preAct *Tensor[T],
	batch, inDim, outDim, tileSize int,
	scale float32,
) {
	layer.WeightStore.Morph(layer.DType)
	w64, ok := layer.WeightStore.Versions[layer.DType].([]uint64)
	if !ok {
		denseForwardAsmFloat(layer, input, preAct, batch, inDim, outDim, tileSize)
		return
	}
	qIn := make([]uint64, len(input.Data))
	qOut := make([]uint64, len(preAct.Data))
	quantizeInputToUint64(input.Data, scale, qIn)
	mc := layer.EnableMultiCoreTiling
	matmul.ForwardNativeU64(qOut, qIn, w64, batch, inDim, outDim, mc, tileSize)
	dequantToTensorU64(preAct.Data, qOut, scale*scale)
}

func quantizeInputToInt16[T Numeric](in []T, scale float32, out []int16) {
	if scale == 0 {
		scale = 1
	}
	for i, v := range in {
		out[i] = int16(math.Round(float64(v) / float64(scale)))
	}
}

func quantizeInputToInt32[T Numeric](in []T, scale float32, out []int32) {
	if scale == 0 {
		scale = 1
	}
	for i, v := range in {
		out[i] = int32(math.Round(float64(v) / float64(scale)))
	}
}

func quantizeInputToInt64[T Numeric](in []T, scale float32, out []int64) {
	if scale == 0 {
		scale = 1
	}
	for i, v := range in {
		out[i] = int64(math.Round(float64(v) / float64(scale)))
	}
}

func quantizeInputToUint16[T Numeric](in []T, scale float32, out []uint16) {
	if scale == 0 {
		scale = 1
	}
	for i, v := range in {
		q := int(math.Round(float64(v) / float64(scale)))
		if q < 0 {
			q = 0
		}
		out[i] = uint16(q)
	}
}

func quantizeInputToUint32[T Numeric](in []T, scale float32, out []uint32) {
	if scale == 0 {
		scale = 1
	}
	for i, v := range in {
		q := int(math.Round(float64(v) / float64(scale)))
		if q < 0 {
			q = 0
		}
		out[i] = uint32(q)
	}
}

func quantizeInputToUint64[T Numeric](in []T, scale float32, out []uint64) {
	if scale == 0 {
		scale = 1
	}
	for i, v := range in {
		q := int(math.Round(float64(v) / float64(scale)))
		if q < 0 {
			q = 0
		}
		out[i] = uint64(q)
	}
}

func dequantToTensorInt[T Numeric](dst []T, q []int16, scale float32) {
	for i := range dst {
		dst[i] = T(float32(q[i]) * scale)
	}
}

func dequantToTensorInt32[T Numeric](dst []T, q []int32, scale float32) {
	for i := range dst {
		dst[i] = T(float32(q[i]) * scale)
	}
}

func dequantToTensorI64[T Numeric](dst []T, q []int64, scale float32) {
	for i := range dst {
		dst[i] = T(float32(q[i]) * scale)
	}
}

func dequantToTensorUint[T Numeric](dst []T, q []uint16, scale float32) {
	for i := range dst {
		dst[i] = T(float32(q[i]) * scale)
	}
}

func dequantToTensorU32[T Numeric](dst []T, q []uint32, scale float32) {
	for i := range dst {
		dst[i] = T(float32(q[i]) * scale)
	}
}

func dequantToTensorU64[T Numeric](dst []T, q []uint64, scale float32) {
	for i := range dst {
		dst[i] = T(float32(q[i]) * scale)
	}
}
