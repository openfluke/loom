package poly

import (
	"math"
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

func tryRNNBackwardSimd[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T], ok bool) {
	in, okIn := any(input).(*Tensor[float32])
	goT, okGO := any(gradOutput).(*Tensor[float32])
	preF, okPre := any(preAct).(*Tensor[float32])
	if !okIn || !okGO || !okPre {
		return nil, nil, false
	}
	giF, gwF := rnnBackwardSimdF32(layer, goT, in, preF)
	gi, okGI := simdTensorAsBackward[T](giF)
	gw, okGW := simdTensorAsBackward[T](gwF)
	if !okGI || !okGW {
		return nil, nil, false
	}
	return gi, gw, true
}

func rnnBackwardSimdF32(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	layer.EnsureRuntimeTileSizes()

	batchSize := input.Shape[0]
	inputSize := layer.InputHeight
	hiddenSize := layer.OutputHeight
	seqLength := layer.SeqLength

	gradInput = NewTensor[float32](batchSize, seqLength, inputSize)
	gradWeights = NewTensor[float32](layer.WeightStore.WeightCount(layer.DType))

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[float32](weights)

	ihSize := hiddenSize * inputSize
	hhSize := hiddenSize * hiddenSize
	wIH := wData[0:ihSize]
	wHH := wData[ihSize : ihSize+hhSize]

	gi64 := make([]float64, len(gradInput.Data))
	gw64 := make([]float64, len(gradWeights.Data))

	rnnSimdBackwardBPTTParallel(
		gi64, gw64,
		gradOutput.Data, input.Data, preAct.Data,
		wIH, wHH,
		batchSize, inputSize, hiddenSize, seqLength, ihSize, hhSize,
	)

	for i := range gradInput.Data {
		gradInput.Data[i] = float32(gi64[i])
	}
	for i := range gradWeights.Data {
		gradWeights.Data[i] = float32(gw64[i])
	}
	return gradInput, gradWeights
}

// rnnSimdBackwardBPTTParallel mirrors rnnBackwardTiledParallel (parallel over batch, BPTT over time).
func rnnSimdBackwardBPTTParallel(
	gi64, gw64 []float64,
	gradOutput, input, preAct []float32,
	wIH, wHH []float32,
	batchSize, inputSize, hiddenSize, seqLength, ihSize, hhSize int,
) {
	var mu sync.Mutex
	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup

	for b := 0; b < batchSize; b++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(b int) {
			defer func() { <-sem; wg.Done() }()

			localGI := make([]float64, seqLength*inputSize)
			localGW := make([]float64, len(gw64))
			gH := make([]float64, hiddenSize)

			inBatch := b * seqLength * inputSize
			outBatch := b * seqLength * hiddenSize

			for t := seqLength - 1; t >= 0; t-- {
				nextGH := make([]float64, hiddenSize)
				it := t * inputSize
				inputRow := input[inBatch+it : inBatch+it+inputSize]
				giRow := localGI[it : it+inputSize]

				for h := 0; h < hiddenSize; h++ {
					preIdx := outBatch + t*hiddenSize + h
					hVal := math.Tanh(float64(preAct[preIdx]))
					gPre := (gH[h] + float64(gradOutput[preIdx])) * (1.0 - hVal*hVal)

					localGW[ihSize+hhSize+h] += gPre

					wIHRow := wIH[h*inputSize : (h+1)*inputSize]
					gwIHRow := localGW[h*inputSize : (h+1)*inputSize]
					simd.SaxpyF32AccF64(gwIHRow, gPre, inputRow, inputSize)
					simd.SaxpyF32AccF64(giRow, gPre, wIHRow, inputSize)

					for hp := 0; hp < hiddenSize; hp++ {
						hPrevVal := 0.0
						if t > 0 {
							hPrevVal = math.Tanh(float64(preAct[outBatch+(t-1)*hiddenSize+hp]))
						}
						localGW[ihSize+h*hiddenSize+hp] += gPre * hPrevVal
						nextGH[hp] += float64(wHH[h*hiddenSize+hp]) * gPre
					}
				}
				gH = nextGH
			}

			mu.Lock()
			giStart := b * seqLength * inputSize
			for i := 0; i < len(localGI); i++ {
				gi64[giStart+i] += localGI[i]
			}
			for i := 0; i < len(localGW); i++ {
				gw64[i] += localGW[i]
			}
			mu.Unlock()
		}(b)
	}
	wg.Wait()
}
