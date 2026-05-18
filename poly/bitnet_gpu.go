package poly

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

func bitNetPackedMatrixScale(ws *WeightStore, offset int) float32 {
	if ws == nil || ws.CPUPacked == nil {
		return 1.0
	}
	if matrix, ok := ws.CPUPacked[bitNetPackedKey(offset)].(*BitNetTernaryMatrix); ok && matrix != nil && matrix.Scale != 0 {
		return matrix.Scale
	}
	return ws.bitNetPackedScale(offset)
}

func bitNetGPUScaleValue(ws *WeightStore, key DType, offset int) float32 {
	if ws != nil && ws.GPUScaleValues != nil {
		if scale, ok := ws.GPUScaleValues[key]; ok && scale != 0 {
			return scale
		}
	}
	return bitNetPackedMatrixScale(ws, offset)
}

func (l *VolumetricLayer) syncBitNetPackedMatrix(ctx *WGPUContext, gpuKey DType, offset, rows, cols int, label string) error {
	if l == nil || l.WeightStore == nil {
		return fmt.Errorf("%s: missing weight store", label)
	}
	matrix, ok := l.WeightStore.GetBitNetTernaryMatrix(offset, rows, cols)
	if !ok || matrix == nil || len(matrix.Words) == 0 {
		return fmt.Errorf("%s: missing BitNet packed matrix offset=%d shape=%dx%d", label, offset, rows, cols)
	}
	buf, err := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    label + " BitNet ternary",
		Contents: wgpu.ToBytes(matrix.Words),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}
	l.WeightStore.GPUWeights[gpuKey] = buf
	if l.WeightStore.GPUScaleValues == nil {
		l.WeightStore.GPUScaleValues = make(map[DType]float32)
	}
	l.WeightStore.GPUScaleValues[gpuKey] = matrix.Scale
	return nil
}

func (l *VolumetricLayer) syncBitNetPackedDense(ctx *WGPUContext) error {
	return l.syncBitNetPackedMatrix(ctx, DTypeTernary, 0, l.OutputHeight, l.InputHeight, "Dense")
}

func (l *VolumetricLayer) syncBitNetPackedMHA(ctx *WGPUContext) error {
	dModel := l.DModel
	qDim := l.QueryDim
	if qDim == 0 {
		qDim = l.NumHeads * l.HeadDim
	}
	kvDim := l.NumKVHeads * l.HeadDim
	if kvDim == 0 {
		kvDim = dModel
	}
	qwStart := 0
	kwStart := qwStart + qDim*dModel
	vwStart := kwStart + kvDim*dModel
	owStart := vwStart + kvDim*dModel
	if err := l.syncBitNetPackedMatrix(ctx, WeightMHAQuery, qwStart, qDim, dModel, "MHA Q"); err != nil {
		return err
	}
	if err := l.syncBitNetPackedMatrix(ctx, WeightMHAKey, kwStart, kvDim, dModel, "MHA K"); err != nil {
		return err
	}
	if err := l.syncBitNetPackedMatrix(ctx, WeightMHAValue, vwStart, kvDim, dModel, "MHA V"); err != nil {
		return err
	}
	if err := l.syncBitNetPackedMatrix(ctx, WeightMHAProjection, owStart, dModel, qDim, "MHA O"); err != nil {
		return err
	}
	if len(l.InnerNormWeight) > 0 {
		buf, err := ctx.CreatePersistentBuffer(l.InnerNormWeight, "MHA BitNet inner norm")
		if err != nil {
			return err
		}
		l.WeightStore.GPUWeights[WeightMHAInnerNorm] = buf
	}
	return nil
}

func (l *VolumetricLayer) syncBitNetPackedSwiGLU(ctx *WGPUContext) error {
	inputSize, intermediateSize := l.InputHeight, l.OutputHeight
	wSize := inputSize * intermediateSize
	gateWStart := 0
	upWStart := wSize
	downWStart := 2 * wSize
	if err := l.syncBitNetPackedMatrix(ctx, DType(100), gateWStart, intermediateSize, inputSize, "SwiGLU Gate"); err != nil {
		return err
	}
	if err := l.syncBitNetPackedMatrix(ctx, DType(101), upWStart, intermediateSize, inputSize, "SwiGLU Up"); err != nil {
		return err
	}
	if err := l.syncBitNetPackedMatrix(ctx, DType(102), downWStart, inputSize, intermediateSize, "SwiGLU Down"); err != nil {
		return err
	}
	if len(l.InnerNormWeight) > 0 {
		buf, err := ctx.CreatePersistentBuffer(l.InnerNormWeight, "SwiGLU BitNet inner norm")
		if err != nil {
			return err
		}
		l.WeightStore.GPUWeights[WeightSwiGLUInnerNorm] = buf
	}
	return nil
}

func bitNetPackedBiasBuffer(l *VolumetricLayer, key DType) *wgpu.Buffer {
	if l == nil || l.WeightStore == nil {
		return nil
	}
	buf, _ := l.WeightStore.GPUWeights[key].(*wgpu.Buffer)
	return buf
}

func bitNetQWords(inputSize int) int {
	return (inputSize + 3) / 4
}

func bitNetQuantizeActivationGPU(ctx *WGPUContext, label string, batchSize, inputSize int, inputBuf *wgpu.Buffer) (*wgpu.Buffer, *wgpu.Buffer, error) {
	qBytes := uint64(batchSize * bitNetQWords(inputSize) * 4)
	scaleBytes := uint64(batchSize * 4)
	qBuf := ctx.GetActivationBuffer(label+"_q8pack", qBytes, wgpu.BufferUsageStorage)
	scaleBuf := ctx.GetActivationBuffer(label+"_q8scale", scaleBytes, wgpu.BufferUsageStorage)
	if err := ctx.DispatchBitNetQuantizeActivation(batchSize, inputSize, inputBuf, qBuf, scaleBuf); err != nil {
		return nil, nil, err
	}
	return qBuf, scaleBuf, nil
}
