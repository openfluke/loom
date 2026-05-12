package poly

import (
	"runtime"
	"sync"
)

// DenseForwardPolymorphic performs a forward pass through a dense layer.
func DenseForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	return DenseForwardTiled(layer, input)
}

// DenseBackwardPolymorphic calculates gradients for the dense layer.
func DenseBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	return DenseBackwardTiled(layer, gradOutput, input, preAct)
}

// DenseForwardTiled performs a tiled forward pass for the dense layer (multi-core).
func DenseForwardTiled[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if usePackedTernaryCPU(layer) {
		return DenseForwardPackedTernaryCPU(layer, input)
	}

	batchSize := input.Shape[0]
	outputSize := layer.OutputHeight
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 32
	}

	preAct = NewTensor[T](batchSize, outputSize)
	postAct = NewTensor[T](batchSize, outputSize)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)

	denseForwardTiledParallel(layer, input, preAct, wData, tileSize)

	for i := range postAct.Data {
		postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
	}
	return preAct, postAct
}

func DenseForwardPackedTernaryCPU[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	batchSize := input.Shape[0]
	inputSize := layer.InputHeight
	outputSize := layer.OutputHeight
	matrix, ok := layer.WeightStore.GetBitNetTernaryMatrix(0, outputSize, inputSize)
	if !ok {
		exact := layer.Network.UseExactDType
		layer.Network.UseExactDType = false
		pre, post := DenseForwardTiled(layer, input)
		layer.Network.UseExactDType = exact
		return pre, post
	}

	preAct = NewTensor[T](batchSize, outputSize)
	postAct = NewTensor[T](batchSize, outputSize)
	tmp := make([]float64, outputSize)
	xq := make([]int8, inputSize)
	for b := 0; b < batchSize; b++ {
		row := input.Data[b*inputSize : (b+1)*inputSize]
		var activationMax float32
		xq, activationMax = bitNetQuantizeActivationNumeric(row, xq)
		if !bitNetTernaryMatVecQuantized(matrix, xq, activationMax, tmp) {
			exact := layer.Network.UseExactDType
			layer.Network.UseExactDType = false
			pre, post := DenseForwardTiled(layer, input)
			layer.Network.UseExactDType = exact
			return pre, post
		}
		for o := 0; o < outputSize; o++ {
			v := T(tmp[o])
			preAct.Data[b*outputSize+o] = v
			postAct.Data[b*outputSize+o] = Activate(v, layer.Activation)
		}
	}
	return preAct, postAct
}

func denseForwardTiledParallel[T Numeric](layer *VolumetricLayer, input *Tensor[T], preAct *Tensor[T], weights []T, tileSize int) {
	batchSize := input.Shape[0]
	inputSize := layer.InputHeight
	outputSize := layer.OutputHeight
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
				for b := 0; b < batchSize; b++ {
					for o := oTile; o < oEnd; o++ {
						var sum float64
						if iTile > 0 {
							sum = float64(preAct.Data[b*outputSize+o])
						}
						rowOff := o * inputSize
						for i := iTile; i < iEnd; i++ {
							sum += float64(input.Data[b*inputSize+i]) * float64(weights[rowOff+i])
						}
						preAct.Data[b*outputSize+o] = T(sum)
					}
				}
			}
		}(oTile)
	}
	wg.Wait()
}

// DenseBackwardTiled performs a tiled backward pass for the dense layer (multi-core).
func DenseBackwardTiled[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	batchSize := input.Shape[0]
	inputSize := layer.InputHeight
	outputSize := layer.OutputHeight
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 32
	}

	gradInput = NewTensor[T](batchSize, inputSize)
	gradWeights = NewTensor[T](outputSize, inputSize)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)

	gradPre := make([]float64, batchSize*outputSize)
	for i := 0; i < len(gradOutput.Data); i++ {
		gradPre[i] = float64(gradOutput.Data[i]) * float64(ActivateDerivative(preAct.Data[i], layer.Activation))
	}

	gi64 := make([]float64, len(gradInput.Data))
	gw64 := make([]float64, len(gradWeights.Data))
	var mu sync.Mutex

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
			localGW := make([]float64, (oEnd-oTile)*inputSize)
			for b := 0; b < batchSize; b++ {
				for o := oTile; o < oEnd; o++ {
					g := gradPre[b*outputSize+o]
					for i := 0; i < inputSize; i++ {
						localGW[(o-oTile)*inputSize+i] += float64(input.Data[b*inputSize+i]) * g
					}
				}
			}
			mu.Lock()
			for o := oTile; o < oEnd; o++ {
				for i := 0; i < inputSize; i++ {
					gw64[o*inputSize+i] += localGW[(o-oTile)*inputSize+i]
				}
			}
			mu.Unlock()
		}(oTile)
	}
	wg.Wait()

	for b := 0; b < batchSize; b++ {
		sem <- struct{}{}
		wg.Add(1)
		go func(b int) {
			defer func() { <-sem; wg.Done() }()
			localGI := make([]float64, inputSize)
			for o := 0; o < outputSize; o++ {
				g := gradPre[b*outputSize+o]
				for i := 0; i < inputSize; i++ {
					localGI[i] += float64(wData[o*inputSize+i]) * g
				}
			}
			mu.Lock()
			for i := 0; i < inputSize; i++ {
				gi64[b*inputSize+i] += localGI[i]
			}
			mu.Unlock()
		}(b)
	}
	wg.Wait()

	for i := range gradInput.Data {
		gradInput.Data[i] = T(gi64[i])
	}
	for i := range gradWeights.Data {
		gradWeights.Data[i] = T(gw64[i])
	}
	return gradInput, gradWeights
}

// DenseGPUTileSizes returns the SC and MC GPU tile sizes for Dense kernels.
func DenseGPUTileSizes(ctx *WGPUContext, dtype DType) (scTile, mcTile int) {
	if ctx == nil {
		return 32, 64 // default fallbacks
	}
	return DenseGPUTileSizesFromContext(ctx, dtype)
}
