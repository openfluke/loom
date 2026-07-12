package poly

import (
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

func tryCNN2BackwardSimd[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T], ok bool) {
	in, okIn := any(input).(*Tensor[float32])
	goT, okGO := any(gradOutput).(*Tensor[float32])
	preF, okPre := any(preAct).(*Tensor[float32])
	if !okIn || !okGO || !okPre {
		return nil, nil, false
	}
	giF, gwF := cnn2BackwardSimdF32(layer, goT, in, preF)
	gi, okGI := simdTensorAsBackward[T](giF)
	gw, okGW := simdTensorAsBackward[T](gwF)
	if !okGI || !okGW {
		return nil, nil, false
	}
	return gi, gw, true
}

func cnn2BackwardSimdF32(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[float32](weights)
	return cnn2BackwardSimdF32WithWeights(layer, gradOutput, input, preAct, wData)
}

func cnn2BackwardSimdF32WithWeights(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32], wData []float32) (gradInput, gradWeights *Tensor[float32]) {
	layer.EnsureRuntimeTileSizes()

	batchSize := input.Shape[0]
	inH, inW, inC := layer.InputHeight, layer.InputWidth, layer.InputChannels
	outH, outW, filters := layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	kernelVol := inC * kSize * kSize

	gradInput = NewTensor[float32](batchSize, inC, inH, inW)
	gradWeights = NewTensor[float32](filters, inC, kSize, kSize)

	gi64 := make([]float64, len(gradInput.Data))
	gw64 := make([]float64, len(gradWeights.Data))

	cnn2SimdBackwardDXParallel(
		gi64, gradOutput.Data, preAct.Data, wData,
		batchSize, inH, inW, inC, outH, outW, filters, kSize, stride, padding, kernelVol,
		layer.Activation,
	)
	cnn2SimdBackwardDWParallel(
		gw64, gradOutput.Data, preAct.Data, input.Data,
		batchSize, inH, inW, inC, outH, outW, filters, kSize, stride, padding,
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

func cnn2SimdGradPre(gradOutput, preAct []float32, outIdx int, activation ActivationType) float64 {
	return float64(gradOutput[outIdx]) * float64(ActivateDerivative(preAct[outIdx], activation))
}

func cnn2SimdPatchFits(oh, ow, inH, inW, kSize, stride, padding int) bool {
	if stride != 1 {
		return false
	}
	kh0 := oh*stride - padding
	kw0 := ow*stride - padding
	return kh0 >= 0 && kh0+kSize <= inH && kw0 >= 0 && kw0+kSize <= inW
}

// cnn2SimdBackwardDXParallel accumulates ∂L/∂X via output-centric saxpy scatter (parallel over filters).
func cnn2SimdBackwardDXParallel(
	gi64 []float64,
	gradOutput, preAct, weights []float32,
	batchSize, inH, inW, inC, outH, outW, filters, kSize, stride, padding, kernelVol int,
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
			localGI := make([]float64, batchSize*inC*inH*inW)
			wBase := f * kernelVol
			for b := 0; b < batchSize; b++ {
				inBatch := b * inC * inH * inW
				for oh := 0; oh < outH; oh++ {
					for ow := 0; ow < outW; ow++ {
						outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
						gOut := cnn2SimdGradPre(gradOutput, preAct, outIdx, activation)
						kh0 := oh*stride - padding
						kw0 := ow*stride - padding
						if cnn2SimdPatchFits(oh, ow, inH, inW, kSize, stride, padding) {
							for ic := 0; ic < inC; ic++ {
								icInBase := inBatch + ic*inH*inW
								wIC := wBase + ic*kSize*kSize
								for kh := 0; kh < kSize; kh++ {
									giOff := icInBase + (kh0+kh)*inW + kw0
									wOff := wIC + kh*kSize
									simd.SaxpyF32AccF64(localGI[giOff:giOff+kSize], gOut, weights[wOff:wOff+kSize], kSize)
								}
							}
							continue
						}
						for ic := 0; ic < inC; ic++ {
							icInBase := inBatch + ic*inH*inW
							wIC := wBase + ic*kSize*kSize
							for kh := 0; kh < kSize; kh++ {
								ih := kh0 + kh
								for kw := 0; kw < kSize; kw++ {
									iw := kw0 + kw
									if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
										giOff := icInBase + ih*inW + iw
										localGI[giOff] += gOut * float64(weights[wIC+kh*kSize+kw])
									}
								}
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

// cnn2SimdBackwardDWParallel mirrors CNN2 tiled dW (parallel over filters).
func cnn2SimdBackwardDWParallel(
	gw64 []float64,
	gradOutput, preAct, input []float32,
	batchSize, inH, inW, inC, outH, outW, filters, kSize, stride, padding int,
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
			localGW := make([]float64, inC*kSize*kSize)
			for b := 0; b < batchSize; b++ {
				inBatch := b * inC * inH * inW
				for oh := 0; oh < outH; oh++ {
					for ow := 0; ow < outW; ow++ {
						outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
						gOut := cnn2SimdGradPre(gradOutput, preAct, outIdx, activation)
						kh0 := oh*stride - padding
						kw0 := ow*stride - padding
						if cnn2SimdPatchFits(oh, ow, inH, inW, kSize, stride, padding) {
							for ic := 0; ic < inC; ic++ {
								icInBase := inBatch + ic*inH*inW
								gwIC := ic * kSize * kSize
								for kh := 0; kh < kSize; kh++ {
									inOff := icInBase + (kh0+kh)*inW + kw0
									gwOff := gwIC + kh*kSize
									simd.SaxpyF32AccF64(localGW[gwOff:gwOff+kSize], gOut, input[inOff:inOff+kSize], kSize)
								}
							}
							continue
						}
						for ic := 0; ic < inC; ic++ {
							icInBase := inBatch + ic*inH*inW
							gwIC := ic * kSize * kSize
							for kh := 0; kh < kSize; kh++ {
								ih := kh0 + kh
								for kw := 0; kw < kSize; kw++ {
									iw := kw0 + kw
									if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
										inOff := icInBase + ih*inW + iw
										localGW[gwIC+kh*kSize+kw] += gOut * float64(input[inOff])
									}
								}
							}
						}
					}
				}
			}
			mu.Lock()
			base := f * inC * kSize * kSize
			for i, v := range localGW {
				gw64[base+i] += v
			}
			mu.Unlock()
		}(f)
	}
	wg.Wait()
}
