package poly

// rnn.go — default RNN path: GetActive FP32 dequant. Native: rnn_native.go.

import (
	"math"
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

// RNNForwardPolymorphic performs a forward pass through an RNN layer.
func RNNForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if useRNNNativeExact(layer) {
		return RNNForwardNativeExact(layer, input)
	}
	if layerUseSimdForward(layer) && simd.SimdEnabled() {
		if pre, post, ok := tryRNNForwardSimd(layer, input); ok {
			return pre, post
		}
	}
	return RNNForwardTiled(layer, input)
}

// RNNBackwardPolymorphic calculates gradients for the RNN layer using BPTT.
func RNNBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	if useRNNNativeExact(layer) {
		return RNNBackwardNativeExact(layer, gradOutput, input, preAct)
	}
	if layerUseSimdForward(layer) && simd.SimdEnabled() {
		if gi, gw, ok := tryRNNBackwardSimd(layer, gradOutput, input, preAct); ok {
			return gi, gw
		}
	}
	return RNNBackwardTiled(layer, gradOutput, input, preAct)
}

// RNNForwardTiled performs a multi-core tiled forward pass for RNN.
func RNNForwardTiled[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	return rnnForwardTiledParallel(layer, input)
}

// RNNBackwardTiled performs a multi-core tiled backward pass for RNN using BPTT.
func RNNBackwardTiled[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	return rnnBackwardTiledParallel(layer, gradOutput, input, preAct)
}

func rnnForwardTiledParallel[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	batchSize, inputSize, hiddenSize, seqLength := input.Shape[0], layer.InputHeight, layer.OutputHeight, layer.SeqLength
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 32
	}

	preAct = NewTensor[T](batchSize, seqLength, hiddenSize)
	postAct = NewTensor[T](batchSize, seqLength, hiddenSize)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)
	ihSize, hhSize := hiddenSize*inputSize, hiddenSize*hiddenSize
	wIH, wHH, bH := wData[0:ihSize], wData[ihSize:ihSize+hhSize], wData[ihSize+hhSize:]

	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	sem := make(chan struct{}, numCPUs)

	for b := 0; b < batchSize; b++ {
		sem <- struct{}{}
		wg.Add(1)
		go func(b int) {
			defer func() { <-sem; wg.Done() }()
			hPrev := make([]float64, hiddenSize)
			for t := 0; t < seqLength; t++ {
				for th := 0; th < hiddenSize; th += tileSize {
					eh := th + tileSize
					if eh > hiddenSize {
						eh = hiddenSize
					}
					for h := th; h < eh; h++ {
						sum := float64(bH[h])
						it := b*seqLength*inputSize + t*inputSize
						for ti := 0; ti < inputSize; ti += tileSize {
							ei := ti + tileSize
							if ei > inputSize {
								ei = inputSize
							}
							for i := ti; i < ei; i++ {
								sum += float64(input.Data[it+i]) * float64(wIH[h*inputSize+i])
							}
						}
						for tp := 0; tp < hiddenSize; tp += tileSize {
							ep := tp + tileSize
							if ep > hiddenSize {
								ep = hiddenSize
							}
							for hp := tp; hp < ep; hp++ {
								sum += hPrev[hp] * float64(wHH[h*hiddenSize+hp])
							}
						}
						preAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h] = T(sum)
						postAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h] = T(math.Tanh(sum))
					}
				}
				for h := 0; h < hiddenSize; h++ {
					hPrev[h] = float64(postAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h])
				}
			}
		}(b)
	}
	wg.Wait()
	return preAct, postAct
}

func rnnBackwardTiledParallel[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	batchSize, inputSize, hiddenSize, seqLength := input.Shape[0], layer.InputHeight, layer.OutputHeight, layer.SeqLength
	gradInput = NewTensor[T](batchSize, seqLength, inputSize)
	gradWeights = NewTensor[T](layer.WeightStore.WeightCount(layer.DType))
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 32
	}

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)
	ihSize, hhSize := hiddenSize*inputSize, hiddenSize*hiddenSize
	wIH, wHH := wData[0:ihSize], wData[ihSize:ihSize+hhSize]

	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	sem := make(chan struct{}, numCPUs)

	// High-precision accumulation buffers for parity
	gi64 := make([]float64, len(gradInput.Data))
	gw64 := make([]float64, len(gradWeights.Data))
	var mu sync.Mutex

	for b := 0; b < batchSize; b++ {
		sem <- struct{}{}
		wg.Add(1)
		go func(b int) {
			defer func() { <-sem; wg.Done() }()
			
			localGI := make([]float64, seqLength*inputSize)
			localGW := make([]float64, len(gradWeights.Data))
			gH := make([]float64, hiddenSize)

			for t := seqLength - 1; t >= 0; t-- {
				nextGH := make([]float64, hiddenSize)
				for h := 0; h < hiddenSize; h++ {
					hVal := math.Tanh(float64(preAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h]))
					gPre := (gH[h] + float64(gradOutput.Data[b*seqLength*hiddenSize+t*hiddenSize+h])) * (1.0 - hVal*hVal)
					
					localGW[ihSize+hhSize+h] += gPre
					it := t * inputSize
					for i := 0; i < inputSize; i++ {
						localGW[h*inputSize+i] += gPre * float64(input.Data[b*seqLength*inputSize+it+i])
						localGI[it+i] += float64(wIH[h*inputSize+i]) * gPre
					}
					for hp := 0; hp < hiddenSize; hp++ {
						hPrevVal := 0.0
						if t > 0 {
							hPrevVal = math.Tanh(float64(preAct.Data[b*seqLength*hiddenSize+(t-1)*hiddenSize+hp]))
						}
						localGW[ihSize+h*hiddenSize+hp] += gPre * hPrevVal
						nextGH[hp] += float64(wHH[h*hiddenSize+hp]) * gPre
					}
				}
				gH = nextGH
			}
			
			mu.Lock()
			startIdx := b * seqLength * inputSize
			for i := 0; i < len(localGI); i++ { gi64[startIdx+i] += localGI[i] }
			for i := 0; i < len(localGW); i++ { gw64[i] += localGW[i] }
			mu.Unlock()
		}(b)
	}
	wg.Wait()

	for i := range gradInput.Data { gradInput.Data[i] = T(gi64[i]) }
	for i := range gradWeights.Data { gradWeights.Data[i] = T(gw64[i]) }
	return gradInput, gradWeights
}
