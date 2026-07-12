package poly

import (
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

func tryCNN3BackwardSimd[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T], ok bool) {
	in, okIn := any(input).(*Tensor[float32])
	goT, okGO := any(gradOutput).(*Tensor[float32])
	preF, okPre := any(preAct).(*Tensor[float32])
	if !okIn || !okGO || !okPre {
		return nil, nil, false
	}
	giF, gwF := cnn3BackwardSimdF32(layer, goT, in, preF)
	gi, okGI := simdTensorAsBackward[T](giF)
	gw, okGW := simdTensorAsBackward[T](gwF)
	if !okGI || !okGW {
		return nil, nil, false
	}
	return gi, gw, true
}

func cnn3BackwardSimdF32(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	layer.EnsureRuntimeTileSizes()

	batchSize := input.Shape[0]
	inD, inH, inW, inC := layer.InputDepth, layer.InputHeight, layer.InputWidth, layer.InputChannels
	outD, outH, outW, filters := layer.OutputDepth, layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	kernelVol := inC * kSize * kSize * kSize

	gradInput = NewTensor[float32](batchSize, inC, inD, inH, inW)
	gradWeights = NewTensor[float32](filters, inC, kSize, kSize, kSize)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[float32](weights)

	gi64 := make([]float64, len(gradInput.Data))
	gw64 := make([]float64, len(gradWeights.Data))

	cnn3SimdBackwardDXParallel(
		gi64, gradOutput.Data, preAct.Data, wData,
		batchSize, inD, inH, inW, inC, outD, outH, outW, filters, kSize, stride, padding, kernelVol,
		layer.Activation,
	)
	cnn3SimdBackwardDWParallel(
		gw64, gradOutput.Data, preAct.Data, input.Data,
		batchSize, inD, inH, inW, inC, outD, outH, outW, filters, kSize, stride, padding,
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

func cnn3SimdGradPre(gradOutput, preAct []float32, outIdx int, activation ActivationType) float64 {
	return float64(gradOutput[outIdx]) * float64(ActivateDerivative(preAct[outIdx], activation))
}

func cnn3SimdPatchFits(od, oh, ow, inD, inH, inW, kSize, stride, padding int) bool {
	if stride != 1 {
		return false
	}
	kd0 := od*stride - padding
	kh0 := oh*stride - padding
	kw0 := ow*stride - padding
	return kd0 >= 0 && kd0+kSize <= inD &&
		kh0 >= 0 && kh0+kSize <= inH &&
		kw0 >= 0 && kw0+kSize <= inW
}

// cnn3SimdBackwardDXParallel accumulates ∂L/∂X via output-centric saxpy scatter (parallel over filters).
func cnn3SimdBackwardDXParallel(
	gi64 []float64,
	gradOutput, preAct, weights []float32,
	batchSize, inD, inH, inW, inC, outD, outH, outW, filters, kSize, stride, padding, kernelVol int,
	activation ActivationType,
) {
	inCStride := inD * inH * inW
	inDStride := inH * inW
	inHStride := inW
	filtCStride := kSize * kSize * kSize
	filtDStride := kSize * kSize
	filtHStride := kSize
	outFStride := outD * outH * outW
	outDStride := outH * outW
	outHStride := outW

	var mu sync.Mutex
	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup

	for f := 0; f < filters; f++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(f int) {
			defer func() { <-sem; wg.Done() }()
			localGI := make([]float64, batchSize*inC*inD*inH*inW)
			wBase := f * kernelVol
			for b := 0; b < batchSize; b++ {
				inBatch := b * inC * inCStride
				for od := 0; od < outD; od++ {
					for oh := 0; oh < outH; oh++ {
						for ow := 0; ow < outW; ow++ {
							outIdx := b*filters*outFStride + f*outFStride + od*outDStride + oh*outHStride + ow
							gOut := cnn3SimdGradPre(gradOutput, preAct, outIdx, activation)
							kd0 := od*stride - padding
							kh0 := oh*stride - padding
							kw0 := ow*stride - padding
							if cnn3SimdPatchFits(od, oh, ow, inD, inH, inW, kSize, stride, padding) {
								for ic := 0; ic < inC; ic++ {
									icInBase := inBatch + ic*inCStride
									wIC := wBase + ic*filtCStride
									for kd := 0; kd < kSize; kd++ {
										idInBase := icInBase + (kd0+kd)*inDStride
										wKD := wIC + kd*filtDStride
										for kh := 0; kh < kSize; kh++ {
											giOff := idInBase + (kh0+kh)*inHStride + kw0
											wOff := wKD + kh*filtHStride
											simd.SaxpyF32AccF64(localGI[giOff:giOff+kSize], gOut, weights[wOff:wOff+kSize], kSize)
										}
									}
								}
								continue
							}
							for ic := 0; ic < inC; ic++ {
								icInBase := inBatch + ic*inCStride
								wIC := wBase + ic*filtCStride
								for kd := 0; kd < kSize; kd++ {
									id := kd0 + kd
									for kh := 0; kh < kSize; kh++ {
										ih := kh0 + kh
										for kw := 0; kw < kSize; kw++ {
											iw := kw0 + kw
											if id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW {
												giOff := icInBase + id*inDStride + ih*inHStride + iw
												localGI[giOff] += gOut * float64(weights[wIC+kd*filtDStride+kh*filtHStride+kw])
											}
										}
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

// cnn3SimdBackwardDWParallel mirrors CNN3 tiled dW (parallel over filters).
func cnn3SimdBackwardDWParallel(
	gw64 []float64,
	gradOutput, preAct, input []float32,
	batchSize, inD, inH, inW, inC, outD, outH, outW, filters, kSize, stride, padding int,
	activation ActivationType,
) {
	inCStride := inD * inH * inW
	inDStride := inH * inW
	inHStride := inW
	filtCStride := kSize * kSize * kSize
	filtDStride := kSize * kSize
	filtHStride := kSize
	outFStride := outD * outH * outW
	outDStride := outH * outW
	outHStride := outW

	var mu sync.Mutex
	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup

	for f := 0; f < filters; f++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(f int) {
			defer func() { <-sem; wg.Done() }()
			localGW := make([]float64, inC*filtCStride)
			for b := 0; b < batchSize; b++ {
				inBatch := b * inC * inCStride
				for od := 0; od < outD; od++ {
					for oh := 0; oh < outH; oh++ {
						for ow := 0; ow < outW; ow++ {
							outIdx := b*filters*outFStride + f*outFStride + od*outDStride + oh*outHStride + ow
							gOut := cnn3SimdGradPre(gradOutput, preAct, outIdx, activation)
							kd0 := od*stride - padding
							kh0 := oh*stride - padding
							kw0 := ow*stride - padding
							if cnn3SimdPatchFits(od, oh, ow, inD, inH, inW, kSize, stride, padding) {
								for ic := 0; ic < inC; ic++ {
									icInBase := inBatch + ic*inCStride
									gwIC := ic * filtCStride
									for kd := 0; kd < kSize; kd++ {
										idInBase := icInBase + (kd0+kd)*inDStride
										gwKD := gwIC + kd*filtDStride
										for kh := 0; kh < kSize; kh++ {
											inOff := idInBase + (kh0+kh)*inHStride + kw0
											gwOff := gwKD + kh*filtHStride
											simd.SaxpyF32AccF64(localGW[gwOff:gwOff+kSize], gOut, input[inOff:inOff+kSize], kSize)
										}
									}
								}
								continue
							}
							for ic := 0; ic < inC; ic++ {
								icInBase := inBatch + ic*inCStride
								gwIC := ic * filtCStride
								for kd := 0; kd < kSize; kd++ {
									id := kd0 + kd
									for kh := 0; kh < kSize; kh++ {
										ih := kh0 + kh
										for kw := 0; kw < kSize; kw++ {
											iw := kw0 + kw
											if id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW {
												inOff := icInBase + id*inDStride + ih*inHStride + iw
												localGW[gwIC+kd*filtDStride+kh*filtHStride+kw] += gOut * float64(input[inOff])
											}
										}
									}
								}
							}
						}
					}
				}
			}
			mu.Lock()
			base := f * inC * filtCStride
			for i, v := range localGW {
				gw64[base+i] += v
			}
			mu.Unlock()
		}(f)
	}
	wg.Wait()
}
