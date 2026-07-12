package poly

import (
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

// DenseForwardPolymorphic performs a forward pass through a dense layer.
func DenseForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if layerUseSimdForward(layer) && simd.SimdEnabled() {
		if pre, post, ok := tryDenseForwardSimd(layer, input); ok {
			return pre, post
		}
	}
	return DenseForwardTiled(layer, input)
}

// DenseBackwardPolymorphic calculates gradients for the dense layer.
func DenseBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	if layerUseSimdForward(layer) && simd.SimdEnabled() {
		if gi, gw, ok := tryDenseBackwardSimd(layer, gradOutput, input, preAct); ok {
			return gi, gw
		}
	}
	return DenseBackwardTiled(layer, gradOutput, input, preAct)
}

// DenseForwardTiled performs a tiled forward pass for the dense layer (multi-core).
func DenseForwardTiled[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	layer.EnsureRuntimeTileSizes()

	if usePackedTernaryCPU(layer) {
		return DenseForwardPackedTernaryCPU(layer, input)
	}

	batchSize := input.Shape[0]
	inputSize := layer.InputHeight
	if inputSize <= 0 {
		inputSize = input.Shape[len(input.Shape)-1]
	}
	outputSize := layer.OutputHeight
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 32
	}
	tileSize = capDenseTileToLayer(tileSize, inputSize, outputSize)

	preAct = NewTensor[T](batchSize, outputSize)
	postAct = NewTensor[T](batchSize, outputSize)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)

	if layer.EnableMultiCoreTiling {
		denseForwardTiledParallel(layer, input, preAct, wData, tileSize)
	} else {
		denseForwardTiledSerial(layer, input, preAct, wData, tileSize)
	}

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

func denseForwardTiledSerial[T Numeric](layer *VolumetricLayer, input *Tensor[T], preAct *Tensor[T], weights []T, tileSize int) {
	batchSize := input.Shape[0]
	inputSize := layer.InputHeight
	outputSize := layer.OutputHeight

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
	}
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

// DenseBackwardTiled performs a tiled backward pass for the dense layer.
func DenseBackwardTiled[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	layer.EnsureRuntimeTileSizes()

	batchSize := input.Shape[0]
	inputSize := layer.InputHeight
	outputSize := layer.OutputHeight
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 32
	}
	tileSize = capDenseTileToLayer(tileSize, inputSize, outputSize)

	gradInput = NewTensor[T](batchSize, inputSize)
	gradWeights = NewTensor[T](outputSize, inputSize)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)

	gradPre := denseGradPreAct(gradOutput, preAct, layer.Activation)

	if layer.EnableMultiCoreTiling {
		denseBackwardTiledParallel(gradInput, gradWeights, input, wData, gradPre, batchSize, inputSize, outputSize, tileSize)
	} else {
		denseBackwardTiledSerial(gradInput, gradWeights, input, wData, gradPre, batchSize, inputSize, outputSize, tileSize)
	}
	return gradInput, gradWeights
}

func denseGradPreAct[T Numeric](gradOutput, preAct *Tensor[T], activation ActivationType) []float64 {
	gradPre := make([]float64, len(gradOutput.Data))
	for i := 0; i < len(gradOutput.Data); i++ {
		gradPre[i] = float64(gradOutput.Data[i]) * float64(ActivateDerivative(preAct.Data[i], activation))
	}
	return gradPre
}

// denseBackwardDWLocal accumulates ∂L/∂W into localGW for output rows [oTile,oEnd).
func denseBackwardDWLocal[T Numeric](localGW []float64, input *Tensor[T], gradPre []float64, batchSize, inputSize, outputSize, oTile, oEnd int) {
	for b := 0; b < batchSize; b++ {
		inOff := b * inputSize
		outOff := b * outputSize
		for o := oTile; o < oEnd; o++ {
			g := gradPre[outOff+o]
			lo := (o - oTile) * inputSize
			for i := 0; i < inputSize; i++ {
				localGW[lo+i] += float64(input.Data[inOff+i]) * g
			}
		}
	}
}

// denseBackwardDXLocal accumulates ∂L/∂X into localGI for batch index b.
func denseBackwardDXLocal[T Numeric](localGI []float64, wData []T, gradPre []float64, b, inputSize, outputSize int) {
	outOff := b * outputSize
	for o := 0; o < outputSize; o++ {
		g := gradPre[outOff+o]
		rowOff := o * inputSize
		for i := 0; i < inputSize; i++ {
			localGI[i] += float64(wData[rowOff+i]) * g
		}
	}
}

func denseBackwardTiledSerial[T Numeric](gradInput, gradWeights *Tensor[T], input *Tensor[T], wData []T, gradPre []float64, batchSize, inputSize, outputSize, tileSize int) {
	gi64 := make([]float64, len(gradInput.Data))
	gw64 := make([]float64, len(gradWeights.Data))

	for oTile := 0; oTile < outputSize; oTile += tileSize {
		oEnd := oTile + tileSize
		if oEnd > outputSize {
			oEnd = outputSize
		}
		localGW := make([]float64, (oEnd-oTile)*inputSize)
		denseBackwardDWLocal(localGW, input, gradPre, batchSize, inputSize, outputSize, oTile, oEnd)
		for o := oTile; o < oEnd; o++ {
			rowOff := o * inputSize
			lo := (o - oTile) * inputSize
			for i := 0; i < inputSize; i++ {
				gw64[rowOff+i] += localGW[lo+i]
			}
		}
	}

	for b := 0; b < batchSize; b++ {
		localGI := make([]float64, inputSize)
		denseBackwardDXLocal(localGI, wData, gradPre, b, inputSize, outputSize)
		inOff := b * inputSize
		for i := 0; i < inputSize; i++ {
			gi64[inOff+i] += localGI[i]
		}
	}

	for i := range gradInput.Data {
		gradInput.Data[i] = T(gi64[i])
	}
	for i := range gradWeights.Data {
		gradWeights.Data[i] = T(gw64[i])
	}
}

func denseBackwardTiledParallel[T Numeric](gradInput, gradWeights *Tensor[T], input *Tensor[T], wData []T, gradPre []float64, batchSize, inputSize, outputSize, tileSize int) {
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
			denseBackwardDWLocal(localGW, input, gradPre, batchSize, inputSize, outputSize, oTile, oEnd)
			mu.Lock()
			for o := oTile; o < oEnd; o++ {
				rowOff := o * inputSize
				lo := (o - oTile) * inputSize
				for i := 0; i < inputSize; i++ {
					gw64[rowOff+i] += localGW[lo+i]
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
			denseBackwardDXLocal(localGI, wData, gradPre, b, inputSize, outputSize)
			mu.Lock()
			inOff := b * inputSize
			for i := 0; i < inputSize; i++ {
				gi64[inOff+i] += localGI[i]
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
}

// DenseGPUTileSizes returns the SC and MC GPU tile sizes for Dense kernels.
func DenseGPUTileSizes(ctx *WGPUContext, dtype DType) (scTile, mcTile int) {
	if ctx == nil {
		return 32, 64 // default fallbacks
	}
	return DenseGPUTileSizesFromContext(ctx, dtype)
}
