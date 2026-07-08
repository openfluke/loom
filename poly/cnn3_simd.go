package poly

import (
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

func tryCNN3ForwardSimd[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T], ok bool) {
	in, ok := any(input).(*Tensor[float32])
	if !ok {
		return nil, nil, false
	}
	preF, postF := cnn3ForwardSimdF32(layer, in)
	return simdTensorsAs[T](preF, postF)
}

func cnn3ForwardSimdF32(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	layer.EnsureRuntimeTileSizes()

	batchSize := input.Shape[0]
	inD, inH, inW, inC := layer.InputDepth, layer.InputHeight, layer.InputWidth, layer.InputChannels
	outD, outH, outW, filters := layer.OutputDepth, layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	kernelVol := inC * kSize * kSize * kSize

	preAct = NewTensor[float32](batchSize, filters, outD, outH, outW)
	postAct = NewTensor[float32](batchSize, filters, outD, outH, outW)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[float32](weights)

	useParallel := layer.EnableMultiCoreTiling && filters > 1
	if useParallel {
		cnn3SimdForwardParallel(input, preAct, postAct, wData, batchSize, inD, inH, inW, inC, outD, outH, outW, filters, kSize, stride, padding, kernelVol, layer.Activation)
	} else {
		cnn3SimdForwardSerial(input, preAct, postAct, wData, batchSize, inD, inH, inW, inC, outD, outH, outW, filters, kSize, stride, padding, kernelVol, layer.Activation)
	}
	return preAct, postAct
}

// cnn3FillPatch gathers the k×k×k×inC receptive field into a contiguous patch (zeros on padding).
func cnn3FillPatch(patch, input []float32, b, od, oh, ow, inC, inD, inH, inW, kSize, stride, padding int) {
	inBatch := b * inC * inD * inH * inW
	inCStride := inD * inH * inW
	inDStride := inH * inW
	inHStride := inW
	kVol := kSize * kSize * kSize
	kDStride := kSize * kSize
	kHStride := kSize

	kd0 := od*stride - padding
	kh0 := oh*stride - padding
	kw0 := ow*stride - padding
	for ic := 0; ic < inC; ic++ {
		icInBase := inBatch + ic*inCStride
		pBase := ic * kVol
		for kd := 0; kd < kSize; kd++ {
			id := kd0 + kd
			for kh := 0; kh < kSize; kh++ {
				ih := kh0 + kh
				for kw := 0; kw < kSize; kw++ {
					iw := kw0 + kw
					v := float32(0)
					if id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW {
						v = input[icInBase+id*inDStride+ih*inHStride+iw]
					}
					patch[pBase+kd*kDStride+kh*kHStride+kw] = v
				}
			}
		}
	}
}

func cnn3FillPatchFast(patch, input []float32, b, od, oh, ow, inC, inD, inH, inW, kSize, stride, padding int) {
	inBatch := b * inC * inD * inH * inW
	inCStride := inD * inH * inW
	inDStride := inH * inW
	inHStride := inW
	kVol := kSize * kSize * kSize
	kDStride := kSize * kSize
	kHStride := kSize

	kd0 := od*stride - padding
	kh0 := oh*stride - padding
	kw0 := ow*stride - padding
	for ic := 0; ic < inC; ic++ {
		icInBase := inBatch + ic*inCStride
		pBase := ic * kVol
		for kd := 0; kd < kSize; kd++ {
			idInBase := icInBase + (kd0+kd)*inDStride
			for kh := 0; kh < kSize; kh++ {
				copy(
					patch[pBase+kd*kDStride+kh*kHStride:pBase+kd*kDStride+(kh+1)*kHStride],
					input[idInBase+(kh0+kh)*inHStride+kw0:idInBase+(kh0+kh)*inHStride+kw0+kSize],
				)
			}
		}
	}
}

func cnn3ConvDotSimd(input, weights, patch []float32, b, f, od, oh, ow, inC, inD, inH, inW, kSize, stride, padding, kernelVol int) float64 {
	wBase := f * kernelVol
	kd0 := od*stride - padding
	kh0 := oh*stride - padding
	kw0 := ow*stride - padding
	if stride == 1 && kd0 >= 0 && kd0+kSize <= inD &&
		kh0 >= 0 && kh0+kSize <= inH &&
		kw0 >= 0 && kw0+kSize <= inW {
		cnn3FillPatchFast(patch, input, b, od, oh, ow, inC, inD, inH, inW, kSize, stride, padding)
	} else {
		cnn3FillPatch(patch, input, b, od, oh, ow, inC, inD, inH, inW, kSize, stride, padding)
	}
	return simd.DotTile(patch, weights[wBase:wBase+kernelVol], 0, kernelVol, 0)
}

func cnn3SimdForwardSerial(input, preAct, postAct *Tensor[float32], weights []float32, batch, inD, inH, inW, inC, outD, outH, outW, filters, kSize, stride, padding, kernelVol int, activation ActivationType) {
	patch := make([]float32, kernelVol)
	outFStride := outD * outH * outW
	outDStride := outH * outW
	outHStride := outW
	for b := 0; b < batch; b++ {
		for f := 0; f < filters; f++ {
			for od := 0; od < outD; od++ {
				for oh := 0; oh < outH; oh++ {
					for ow := 0; ow < outW; ow++ {
						sum := cnn3ConvDotSimd(input.Data, weights, patch, b, f, od, oh, ow, inC, inD, inH, inW, kSize, stride, padding, kernelVol)
						outIdx := b*filters*outFStride + f*outFStride + od*outDStride + oh*outHStride + ow
						preAct.Data[outIdx] = float32(sum)
						postAct.Data[outIdx] = Activate(float32(sum), activation)
					}
				}
			}
		}
	}
}

func cnn3SimdForwardParallel(input, preAct, postAct *Tensor[float32], weights []float32, batch, inD, inH, inW, inC, outD, outH, outW, filters, kSize, stride, padding, kernelVol int, activation ActivationType) {
	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup
	outFStride := outD * outH * outW
	outDStride := outH * outW
	outHStride := outW

	for f := 0; f < filters; f++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(f int) {
			defer func() { <-sem; wg.Done() }()
			patch := make([]float32, kernelVol)
			for b := 0; b < batch; b++ {
				for od := 0; od < outD; od++ {
					for oh := 0; oh < outH; oh++ {
						for ow := 0; ow < outW; ow++ {
							sum := cnn3ConvDotSimd(input.Data, weights, patch, b, f, od, oh, ow, inC, inD, inH, inW, kSize, stride, padding, kernelVol)
							outIdx := b*filters*outFStride + f*outFStride + od*outDStride + oh*outHStride + ow
							preAct.Data[outIdx] = float32(sum)
							postAct.Data[outIdx] = Activate(float32(sum), activation)
						}
					}
				}
			}
		}(f)
	}
	wg.Wait()
}
