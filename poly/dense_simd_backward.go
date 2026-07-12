package poly

import (
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

func tryDenseBackwardSimd[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T], ok bool) {
	in, okIn := any(input).(*Tensor[float32])
	goT, okGO := any(gradOutput).(*Tensor[float32])
	preF, okPre := any(preAct).(*Tensor[float32])
	if !okIn || !okGO || !okPre {
		return nil, nil, false
	}
	giF, gwF := denseBackwardSimdF32(layer, goT, in, preF)
	gi, okGI := simdTensorAsBackward[T](giF)
	gw, okGW := simdTensorAsBackward[T](gwF)
	if !okGI || !okGW {
		return nil, nil, false
	}
	return gi, gw, true
}

func simdTensorAsBackward[T Numeric](t *Tensor[float32]) (*Tensor[T], bool) {
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

func denseBackwardSimdF32(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	layer.EnsureRuntimeTileSizes()

	batchSize := input.Shape[0]
	inputSize := layer.InputHeight
	outputSize := layer.OutputHeight
	tileSize := layer.GetCPUSimdTileSize(layer.DType)
	tileSize = capDenseTileToLayer(tileSize, inputSize, outputSize)

	gradInput = NewTensor[float32](batchSize, inputSize)
	gradWeights = NewTensor[float32](outputSize, inputSize)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[float32](weights)

	gradPre := denseGradPreAct(gradOutput, preAct, layer.Activation)

	if layer.EnableMultiCoreTiling {
		denseSimdBackwardParallel(gradInput, gradWeights, input, wData, gradPre, batchSize, inputSize, outputSize, tileSize)
	} else {
		denseSimdBackwardSerial(gradInput, gradWeights, input, wData, gradPre, batchSize, inputSize, outputSize, tileSize)
	}
	return gradInput, gradWeights
}

// denseBackwardDWLocalSimd: localGW[o-oTile,:] += g * input[b,:] via saxpy (AVX2/NEON).
func denseBackwardDWLocalSimd(localGW []float64, input *Tensor[float32], gradPre []float64, batchSize, inputSize, outputSize, oTile, oEnd int) {
	for b := 0; b < batchSize; b++ {
		inOff := b * inputSize
		outOff := b * outputSize
		for o := oTile; o < oEnd; o++ {
			g := gradPre[outOff+o]
			lo := (o - oTile) * inputSize
			simd.SaxpyF32AccF64(localGW[lo:lo+inputSize], g, input.Data[inOff:inOff+inputSize], inputSize)
		}
	}
}

// denseBackwardDXLocalSimd: localGI += g * weights[o,:] via saxpy (AVX2/NEON).
func denseBackwardDXLocalSimd(localGI []float64, wData []float32, gradPre []float64, b, inputSize, outputSize int) {
	outOff := b * outputSize
	for o := 0; o < outputSize; o++ {
		g := gradPre[outOff+o]
		rowOff := o * inputSize
		simd.SaxpyF32AccF64(localGI, g, wData[rowOff:rowOff+inputSize], inputSize)
	}
}

func denseSimdBackwardSerial(gradInput, gradWeights *Tensor[float32], input *Tensor[float32], wData []float32, gradPre []float64, batchSize, inputSize, outputSize, tileSize int) {
	gi64 := make([]float64, len(gradInput.Data))
	gw64 := make([]float64, len(gradWeights.Data))

	// Phase 1: ∂L/∂W — output tiles (serial), same structure as denseBackwardTiledSerial.
	for oTile := 0; oTile < outputSize; oTile += tileSize {
		oEnd := oTile + tileSize
		if oEnd > outputSize {
			oEnd = outputSize
		}
		localGW := make([]float64, (oEnd-oTile)*inputSize)
		denseBackwardDWLocalSimd(localGW, input, gradPre, batchSize, inputSize, outputSize, oTile, oEnd)
		for o := oTile; o < oEnd; o++ {
			rowOff := o * inputSize
			lo := (o - oTile) * inputSize
			for i := 0; i < inputSize; i++ {
				gw64[rowOff+i] += localGW[lo+i]
			}
		}
	}

	// Phase 2: ∂L/∂X — per batch (serial), same structure as denseBackwardTiledSerial.
	for b := 0; b < batchSize; b++ {
		localGI := make([]float64, inputSize)
		denseBackwardDXLocalSimd(localGI, wData, gradPre, b, inputSize, outputSize)
		inOff := b * inputSize
		for i := 0; i < inputSize; i++ {
			gi64[inOff+i] += localGI[i]
		}
	}

	for i := range gradInput.Data {
		gradInput.Data[i] = float32(gi64[i])
	}
	for i := range gradWeights.Data {
		gradWeights.Data[i] = float32(gw64[i])
	}
}

func denseSimdBackwardParallel(gradInput, gradWeights *Tensor[float32], input *Tensor[float32], wData []float32, gradPre []float64, batchSize, inputSize, outputSize, tileSize int) {
	gi64 := make([]float64, len(gradInput.Data))
	gw64 := make([]float64, len(gradWeights.Data))
	var mu sync.Mutex

	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	sem := make(chan struct{}, numCPUs)

	// Phase 1: ∂L/∂W — parallel output tiles (same as denseBackwardTiledParallel).
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
			denseBackwardDWLocalSimd(localGW, input, gradPre, batchSize, inputSize, outputSize, oTile, oEnd)
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

	// Phase 2: ∂L/∂X — parallel per batch (same as denseBackwardTiledParallel).
	for b := 0; b < batchSize; b++ {
		sem <- struct{}{}
		wg.Add(1)
		go func(b int) {
			defer func() { <-sem; wg.Done() }()
			localGI := make([]float64, inputSize)
			denseBackwardDXLocalSimd(localGI, wData, gradPre, b, inputSize, outputSize)
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
		gradInput.Data[i] = float32(gi64[i])
	}
	for i := range gradWeights.Data {
		gradWeights.Data[i] = float32(gw64[i])
	}
}
