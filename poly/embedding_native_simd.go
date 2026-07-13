package poly

import (
	"math"
	"math/rand"
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

// embedding_native_simd.go — native-exact embedding SIMD: MAC dtypes via parallel lookup; integers via parallel gather/scatter.

func tryEmbeddingForwardNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() {
		return nil, nil, false
	}
	if useEmbeddingTrueNative(layer) {
		if embeddingTrueNativeUsesU8(layer.DType) {
			return embeddingForwardUIntegerNativeSimd(layer, input)
		}
		return embeddingForwardIntegerNativeSimd(layer, input)
	}
	return embeddingForwardNativeMACSimd(layer, input)
}

func tryEmbeddingBackwardNativeSimd(layer *VolumetricLayer, gradOutput, input *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() {
		return nil, nil, false
	}
	if useEmbeddingTrueNative(layer) {
		if embeddingTrueNativeUsesU8(layer.DType) {
			return embeddingBackwardUIntegerNativeSimd(layer, gradOutput, input)
		}
		return embeddingBackwardIntegerNativeSimd(layer, gradOutput, input)
	}
	return embeddingBackwardNativeMACSimd(layer, gradOutput, input)
}

func embeddingForwardNativeMACSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	ws := layer.WeightStore
	count := ws.WeightCount(layer.DType)
	if count <= 0 {
		count = layer.VocabSize * embeddingDim(layer)
	}
	if count <= 0 {
		return nil, nil, false
	}
	wData := ws.NativeSimdF32Weights(layer.DType)
	if wData == nil {
		return nil, nil, false
	}
	preAct, postAct = embeddingForwardSimdF32WithWeights(layer, input, wData)
	return preAct, postAct, true
}

func embeddingBackwardNativeMACSimd(layer *VolumetricLayer, gradOutput, input *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	gradInput, gradWeights = embeddingBackwardSimdF32WithWeights(layer, gradOutput, input)
	return gradInput, gradWeights, true
}

func embeddingForwardIntegerNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	ws := layer.WeightStore
	ws.Morph(layer.DType)
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	w := ws.NativeSimdI8Weights(layer.DType)
	if w == nil {
		return nil, nil, false
	}

	vocabSize := layer.VocabSize
	embDim := embeddingDim(layer)
	seqLen := embeddingTokenCount(input)
	tileSize := layer.GetCPUSimdTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 32
	}

	outShape := append([]int{}, input.Shape...)
	outShape = append(outShape, embDim)
	preAct = NewTensor[float32](outShape...)
	postAct = NewTensor[float32](outShape...)

	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	sem := make(chan struct{}, numCPUs)

	for iTile := 0; iTile < seqLen; iTile += tileSize {
		sem <- struct{}{}
		wg.Add(1)
		go func(iTile int) {
			defer func() { <-sem; wg.Done() }()
			iEnd := iTile + tileSize
			if iEnd > seqLen {
				iEnd = seqLen
			}
			for i := iTile; i < iEnd; i++ {
				tokenID := int(input.Data[i])
				if tokenID < 0 || tokenID >= vocabSize {
					continue
				}
				rowBase := tokenID * embDim
				outBase := i * embDim
				for j := 0; j < embDim; j++ {
					if rowBase+j >= len(w) {
						break
					}
					preAct.Data[outBase+j] = float32(w[rowBase+j]) * scale
				}
			}
		}(iTile)
	}
	wg.Wait()

	for i := range preAct.Data {
		postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
	}
	return preAct, postAct, true
}

func embeddingForwardUIntegerNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	ws := layer.WeightStore
	ws.Morph(layer.DType)
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	w := ws.NativeSimdU8Weights(layer.DType)
	if w == nil {
		return nil, nil, false
	}

	vocabSize := layer.VocabSize
	embDim := embeddingDim(layer)
	seqLen := embeddingTokenCount(input)
	tileSize := layer.GetCPUSimdTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 32
	}

	outShape := append([]int{}, input.Shape...)
	outShape = append(outShape, embDim)
	preAct = NewTensor[float32](outShape...)
	postAct = NewTensor[float32](outShape...)

	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	sem := make(chan struct{}, numCPUs)

	for iTile := 0; iTile < seqLen; iTile += tileSize {
		sem <- struct{}{}
		wg.Add(1)
		go func(iTile int) {
			defer func() { <-sem; wg.Done() }()
			iEnd := iTile + tileSize
			if iEnd > seqLen {
				iEnd = seqLen
			}
			for i := iTile; i < iEnd; i++ {
				tokenID := int(input.Data[i])
				if tokenID < 0 || tokenID >= vocabSize {
					continue
				}
				rowBase := tokenID * embDim
				outBase := i * embDim
				for j := 0; j < embDim; j++ {
					if rowBase+j >= len(w) {
						break
					}
					preAct.Data[outBase+j] = float32(w[rowBase+j]) * scale
				}
			}
		}(iTile)
	}
	wg.Wait()

	for i := range preAct.Data {
		postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
	}
	return preAct, postAct, true
}

func embeddingBackwardIntegerNativeSimd(layer *VolumetricLayer, gradOutput, input *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	ws := layer.WeightStore
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	w := ws.NativeSimdI8Weights(layer.DType)
	if w == nil {
		return nil, nil, false
	}

	vocabSize := layer.VocabSize
	embDim := embeddingDim(layer)
	gradStride := embeddingGradStride(gradOutput, embDim)
	seqLen := embeddingTokenCount(input)
	wCount := len(w)
	tileSize := layer.GetCPUSimdTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 32
	}

	gradInput = NewTensor[float32](input.Shape...)
	gradWeights = NewTensor[float32](wCount)
	gradW := make([]int32, wCount)

	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	sem := make(chan struct{}, numCPUs)
	numTiles := (seqLen + tileSize - 1) / tileSize
	localGradW := make([][]int32, numTiles)
	for k := range localGradW {
		localGradW[k] = make([]int32, wCount)
	}

	for k := 0; k < numTiles; k++ {
		iTile := k * tileSize
		sem <- struct{}{}
		wg.Add(1)
		go func(k, iTile int) {
			defer func() { <-sem; wg.Done() }()
			iEnd := iTile + tileSize
			if iEnd > seqLen {
				iEnd = seqLen
			}
			acc := localGradW[k]
			for i := iTile; i < iEnd; i++ {
				tokenID := int(input.Data[i])
				if tokenID < 0 || tokenID >= vocabSize {
					continue
				}
				rowBase := tokenID * embDim
				outBase := embeddingOutBase(i, gradStride)
				if outBase+embDim > len(gradOutput.Data) {
					continue
				}
				for j := 0; j < embDim; j++ {
					idx := rowBase + j
					if idx >= wCount {
						break
					}
					g := int32(math.Round(float64(gradOutput.Data[outBase+j]) / float64(scale)))
					acc[idx] += g
				}
			}
		}(k, iTile)
	}
	wg.Wait()

	for k := range localGradW {
		for i, v := range localGradW[k] {
			gradW[i] += v
		}
	}

	lrShift := learningRateToBitShift(layer.Network.ExactNativeLR)
	applyStochasticNativeI8Update(layer.DType, w, gradW, lrShift)
	publishInt8Weights(ws, layer.DType, w)
	markLayerNativeWeightsUpdated(layer, ws, layer.DType, ws.Versions[layer.DType].([]uint8))
	return gradInput, gradWeights, true
}

func embeddingBackwardUIntegerNativeSimd(layer *VolumetricLayer, gradOutput, input *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	ws := layer.WeightStore
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	w := ws.NativeSimdU8Weights(layer.DType)
	if w == nil {
		return nil, nil, false
	}

	vocabSize := layer.VocabSize
	embDim := embeddingDim(layer)
	gradStride := embeddingGradStride(gradOutput, embDim)
	seqLen := embeddingTokenCount(input)
	wCount := len(w)
	tileSize := layer.GetCPUSimdTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 32
	}

	gradInput = NewTensor[float32](input.Shape...)
	gradWeights = NewTensor[float32](wCount)
	gradW := make([]int32, wCount)

	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	sem := make(chan struct{}, numCPUs)
	numTiles := (seqLen + tileSize - 1) / tileSize
	localGradW := make([][]int32, numTiles)
	for k := range localGradW {
		localGradW[k] = make([]int32, wCount)
	}

	for k := 0; k < numTiles; k++ {
		iTile := k * tileSize
		sem <- struct{}{}
		wg.Add(1)
		go func(k, iTile int) {
			defer func() { <-sem; wg.Done() }()
			iEnd := iTile + tileSize
			if iEnd > seqLen {
				iEnd = seqLen
			}
			acc := localGradW[k]
			for i := iTile; i < iEnd; i++ {
				tokenID := int(input.Data[i])
				if tokenID < 0 || tokenID >= vocabSize {
					continue
				}
				rowBase := tokenID * embDim
				outBase := embeddingOutBase(i, gradStride)
				if outBase+embDim > len(gradOutput.Data) {
					continue
				}
				for j := 0; j < embDim; j++ {
					idx := rowBase + j
					if idx >= wCount {
						break
					}
					g := int32(math.Round(float64(gradOutput.Data[outBase+j]) / float64(scale)))
					acc[idx] += g
				}
			}
		}(k, iTile)
	}
	wg.Wait()

	for k := range localGradW {
		for i, v := range localGradW[k] {
			gradW[i] += v
		}
	}

	lrShift := learningRateToBitShift(layer.Network.ExactNativeLR)
	mask := int32((1 << lrShift) - 1)
	for i := range w {
		scaledGrad := gradW[i] >> lrShift
		if (gradW[i] & mask) > rand.Int31n(1<<lrShift) {
			scaledGrad++
		}
		w[i] = clampNativeU8Weight(layer.DType, clampU8(int32(w[i])-scaledGrad))
	}
	ws.Versions[layer.DType] = w
	ws.Master = nil
	if layer.ExactDense != nil {
		layer.ExactDense.WeightsUpdated = true
	}
	ws.GPUWeights = make(map[DType]any)
	if ws.CPUPacked != nil {
		delete(ws.CPUPacked, layer.DType)
	}
	ws.invalidateNativeSimdCache(layer.DType)
	markLayerNativeWeightsUpdated(layer, ws, layer.DType, ws.Versions[layer.DType].([]uint8))
	return gradInput, gradWeights, true
}
