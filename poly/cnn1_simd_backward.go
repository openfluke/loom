package poly

import (
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

func tryCNN1BackwardSimd[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T], ok bool) {
	if useNativeQuantCNN1(layer) {
		return nil, nil, false
	}
	in, okIn := any(input).(*Tensor[float32])
	goT, okGO := any(gradOutput).(*Tensor[float32])
	preF, okPre := any(preAct).(*Tensor[float32])
	if !okIn || !okGO || !okPre {
		return nil, nil, false
	}
	giF, gwF := cnn1BackwardSimdF32(layer, goT, in, preF)
	gi, okGI := simdTensorAsBackward[T](giF)
	gw, okGW := simdTensorAsBackward[T](gwF)
	if !okGI || !okGW {
		return nil, nil, false
	}
	return gi, gw, true
}

func cnn1BackwardSimdF32(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[float32](weights)
	return cnn1BackwardSimdF32WithWeights(layer, gradOutput, input, preAct, wData)
}

func cnn1BackwardSimdF32WithWeights(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32], wData []float32) (gradInput, gradWeights *Tensor[float32]) {
	layer.EnsureRuntimeTileSizes()

	batchSize := input.Shape[0]
	seqLen, inC := layer.InputHeight, layer.InputChannels
	outLen, filters := layer.OutputHeight, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	kernelVol := inC * kSize

	gradInput = NewTensor[float32](batchSize, inC, seqLen)
	gradWeights = NewTensor[float32](filters, inC, kSize)

	gi64 := make([]float64, len(gradInput.Data))
	gw64 := make([]float64, len(gradWeights.Data))

	cnn1SimdBackwardDXParallel(
		gi64, gradOutput.Data, preAct.Data, wData,
		batchSize, seqLen, inC, outLen, filters, kSize, stride, padding, kernelVol,
		layer.Activation,
	)
	cnn1SimdBackwardDWParallel(
		gw64, gradOutput.Data, preAct.Data, input.Data,
		batchSize, seqLen, inC, outLen, filters, kSize, stride, padding,
		layer.Activation,
	)

	for i := range gradInput.Data {
		gradInput.Data[i] = float32(gi64[i])
	}
	for i := range gradWeights.Data {
		gradWeights.Data[i] = float32(gw64[i])
	}
	return gradInput, gradWeights
}

func cnn1SimdGradPre(gradOutput, preAct []float32, outIdx int, activation ActivationType) float64 {
	return float64(gradOutput[outIdx]) * float64(ActivateDerivative(preAct[outIdx], activation))
}

// cnn1SimdBackwardDXParallel accumulates ∂L/∂X via output-centric saxpy scatter (parallel over filters).
func cnn1SimdBackwardDXParallel(
	gi64 []float64,
	gradOutput, preAct, weights []float32,
	batchSize, seqLen, inC, outLen, filters, kSize, stride, padding, kernelVol int,
	activation ActivationType,
) {
	var mu sync.Mutex
	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup

	for f := 0; f < filters; f++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(f int) {
			defer func() { <-sem; wg.Done() }()
			localGI := make([]float64, batchSize*inC*seqLen)
			wBase := f * kernelVol
			for b := 0; b < batchSize; b++ {
				inBatch := b * inC * seqLen
				for o := 0; o < outLen; o++ {
					outIdx := b*filters*outLen + f*outLen + o
					gOut := cnn1SimdGradPre(gradOutput, preAct, outIdx, activation)
					inPos0 := o*stride - padding
					if stride == 1 && inPos0 >= 0 && inPos0+kSize <= seqLen {
						for ic := 0; ic < inC; ic++ {
							giOff := inBatch + ic*seqLen + inPos0
							wOff := wBase + ic*kSize
							simd.SaxpyF32AccF64(localGI[giOff:giOff+kSize], gOut, weights[wOff:wOff+kSize], kSize)
						}
						continue
					}
					for ic := 0; ic < inC; ic++ {
						for k := 0; k < kSize; k++ {
							inPos := o*stride + k - padding
							if inPos >= 0 && inPos < seqLen {
								giOff := inBatch + ic*seqLen + inPos
								localGI[giOff] += gOut * float64(weights[wBase+ic*kSize+k])
							}
						}
					}
				}
			}
			mu.Lock()
			for i, v := range localGI {
				gi64[i] += v
			}
			mu.Unlock()
		}(f)
	}
	wg.Wait()
}

// cnn1SimdBackwardDWParallel mirrors CNN1BackwardTiledParallel dW (parallel over filters).
func cnn1SimdBackwardDWParallel(
	gw64 []float64,
	gradOutput, preAct, input []float32,
	batchSize, seqLen, inC, outLen, filters, kSize, stride, padding int,
	activation ActivationType,
) {
	var mu sync.Mutex
	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup

	for f := 0; f < filters; f++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(f int) {
			defer func() { <-sem; wg.Done() }()
			localGW := make([]float64, inC*kSize)
			for b := 0; b < batchSize; b++ {
				inBatch := b * inC * seqLen
				for o := 0; o < outLen; o++ {
					outIdx := b*filters*outLen + f*outLen + o
					gOut := cnn1SimdGradPre(gradOutput, preAct, outIdx, activation)
					inPos0 := o*stride - padding
					if stride == 1 && inPos0 >= 0 && inPos0+kSize <= seqLen {
						for ic := 0; ic < inC; ic++ {
							inIdx := inBatch + ic*seqLen + inPos0
							simd.SaxpyF32AccF64(localGW[ic*kSize:ic*kSize+kSize], gOut, input[inIdx:inIdx+kSize], kSize)
						}
						continue
					}
					for ic := 0; ic < inC; ic++ {
						for k := 0; k < kSize; k++ {
							inPos := o*stride + k - padding
							if inPos >= 0 && inPos < seqLen {
								inIdx := inBatch + ic*seqLen + inPos
								localGW[ic*kSize+k] += gOut * float64(input[inIdx])
							}
						}
					}
				}
			}
			mu.Lock()
			base := f * inC * kSize
			for i, v := range localGW {
				gw64[base+i] += v
			}
			mu.Unlock()
		}(f)
	}
	wg.Wait()
}
