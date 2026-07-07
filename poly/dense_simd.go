package poly

import (
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

func tryDenseForwardSimd[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T], ok bool) {
	in, ok := any(input).(*Tensor[float32])
	if !ok {
		return nil, nil, false
	}
	preF, postF := denseForwardSimdF32(layer, in)
	return simdTensorsAs[T](preF, postF)
}

func simdTensorsAs[T Numeric](pre, post *Tensor[float32]) (*Tensor[T], *Tensor[T], bool) {
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

// denseLayerSimdViable reports whether this layer is wide enough for SIMD dots.
func denseLayerSimdViable(layer *VolumetricLayer) bool {
	if layer == nil {
		return false
	}
	minDim := DenseSimdMinDim()
	in := layer.InputHeight
	out := layer.OutputHeight
	if in < minDim && out < minDim {
		return false
	}
	return true
}

func denseForwardSimdF32(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	layer.EnsureRuntimeTileSizes()

	if usePackedTernaryCPU(layer) {
		pre, post := DenseForwardPackedTernaryCPU(layer, input)
		return pre, post
	}
	if !denseLayerSimdViable(layer) {
		pre, post := DenseForwardTiled(layer, input)
		return pre, post
	}

	batchSize := input.Shape[0]
	inputSize := layer.InputHeight
	outputSize := layer.OutputHeight
	tileSize := layer.GetCPUSimdTileSize(layer.DType)
	tileSize = capDenseTileToLayer(tileSize, inputSize, outputSize)

	preAct = NewTensor[float32](batchSize, outputSize)
	postAct = NewTensor[float32](batchSize, outputSize)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[float32](weights)

	useParallel := layer.EnableMultiCoreTiling && outputSize > tileSize
	if useParallel {
		denseSimdForwardParallel(input, preAct, wData, batchSize, inputSize, outputSize, tileSize)
	} else {
		denseSimdForwardSerial(input, preAct, wData, batchSize, inputSize, outputSize, tileSize)
	}

	for i := range postAct.Data {
		postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
	}
	return preAct, postAct
}

func denseSimdForwardSerial(input, preAct *Tensor[float32], weights []float32, batch, inputSize, outputSize, tileSize int) {
	for oTile := 0; oTile < outputSize; oTile += tileSize {
		oEnd := oTile + tileSize
		if oEnd > outputSize {
			oEnd = outputSize
		}
		for iTile := 0; iTile < inputSize; iTile += tileSize {
			iEnd := iTile + tileSize
			if iEnd > inputSize {
				iEnd = inputSize
			}
			for b := 0; b < batch; b++ {
				rowBase := b * inputSize
				outBase := b * outputSize
				for o := oTile; o < oEnd; o++ {
					sum := 0.0
					if iTile > 0 {
						sum = float64(preAct.Data[outBase+o])
					}
					rowOff := o * inputSize
					sum = simd.DotTile(
						input.Data[rowBase:rowBase+inputSize],
						weights[rowOff:rowOff+inputSize],
						iTile, iEnd, sum,
					)
					preAct.Data[outBase+o] = float32(sum)
				}
			}
		}
	}
}

func denseSimdForwardParallel(input, preAct *Tensor[float32], weights []float32, batch, inputSize, outputSize, tileSize int) {
	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	sem := make(chan struct{}, numCPUs)

	for oTile := 0; oTile < outputSize; oTile += tileSize {
		sem <- struct{}{}
		wg.Add(1)
		go func(oTile int) {
			defer func() { <-sem; wg.Done() }()
			oEnd := oTile + tileSize
			if oEnd > outputSize {
				oEnd = outputSize
			}
			for iTile := 0; iTile < inputSize; iTile += tileSize {
				iEnd := iTile + tileSize
				if iEnd > inputSize {
					iEnd = inputSize
				}
				for b := 0; b < batch; b++ {
					rowBase := b * inputSize
					outBase := b * outputSize
					for o := oTile; o < oEnd; o++ {
						sum := 0.0
						if iTile > 0 {
							sum = float64(preAct.Data[outBase+o])
						}
						rowOff := o * inputSize
						sum = simd.DotTile(
							input.Data[rowBase:rowBase+inputSize],
							weights[rowOff:rowOff+inputSize],
							iTile, iEnd, sum,
						)
						preAct.Data[outBase+o] = float32(sum)
					}
				}
			}
		}(oTile)
	}
	wg.Wait()
}
