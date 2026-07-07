package poly

import (
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

func tryCNN2ForwardSimd[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T], ok bool) {
	in, ok := any(input).(*Tensor[float32])
	if !ok {
		return nil, nil, false
	}
	preF, postF := cnn2ForwardSimdF32(layer, in)
	return simdTensorsAs[T](preF, postF)
}

func cnn2LayerSimdViable(layer *VolumetricLayer) bool {
	if layer == nil {
		return false
	}
	kSize := layer.KernelSize
	if kSize <= 0 {
		kSize = 1
	}
	kernelVol := layer.InputChannels * kSize * kSize
	return kernelVol >= CNN1SimdMinDim()
}

func cnn2ForwardSimdF32(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	layer.EnsureRuntimeTileSizes()

	if !cnn2LayerSimdViable(layer) {
		return CNN2ForwardTiled(layer, input)
	}

	batchSize := input.Shape[0]
	inH, inW, inC := layer.InputHeight, layer.InputWidth, layer.InputChannels
	outH, outW, filters := layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	kernelVol := inC * kSize * kSize

	preAct = NewTensor[float32](batchSize, filters, outH, outW)
	postAct = NewTensor[float32](batchSize, filters, outH, outW)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[float32](weights)

	useParallel := layer.EnableMultiCoreTiling && filters > 1
	if useParallel {
		cnn2SimdForwardParallel(input, preAct, postAct, wData, batchSize, inH, inW, inC, outH, outW, filters, kSize, stride, padding, kernelVol, layer.Activation)
	} else {
		cnn2SimdForwardSerial(input, preAct, postAct, wData, batchSize, inH, inW, inC, outH, outW, filters, kSize, stride, padding, kernelVol, layer.Activation)
	}
	return preAct, postAct
}

// cnn2FillPatch gathers the k×k×inC receptive field into a contiguous patch (zeros on padding).
func cnn2FillPatch(patch, input []float32, b, oh, ow, inC, inH, inW, kSize, stride, padding int) {
	inBatch := b * inC * inH * inW
	kh0 := oh*stride - padding
	kw0 := ow*stride - padding
	for ic := 0; ic < inC; ic++ {
		icInBase := inBatch + ic*inH*inW
		pBase := ic * kSize * kSize
		for kh := 0; kh < kSize; kh++ {
			ih := kh0 + kh
			for kw := 0; kw < kSize; kw++ {
				iw := kw0 + kw
				v := float32(0)
				if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
					v = input[icInBase+ih*inW+iw]
				}
				patch[pBase+kh*kSize+kw] = v
			}
		}
	}
}

func cnn2FillPatchFast(patch, input []float32, b, oh, ow, inC, inH, inW, kSize, stride, padding int) {
	inBatch := b * inC * inH * inW
	kh0 := oh*stride - padding
	kw0 := ow*stride - padding
	for ic := 0; ic < inC; ic++ {
		icInBase := inBatch + ic*inH*inW
		pBase := ic * kSize * kSize
		for kh := 0; kh < kSize; kh++ {
			copy(
				patch[pBase+kh*kSize:pBase+(kh+1)*kSize],
				input[icInBase+(kh0+kh)*inW+kw0:icInBase+(kh0+kh)*inW+kw0+kSize],
			)
		}
	}
}

func cnn2ConvDotSimd(input, weights, patch []float32, b, f, oh, ow, inC, inH, inW, kSize, stride, padding, kernelVol int) float64 {
	wBase := f * kernelVol
	kh0 := oh*stride - padding
	kw0 := ow*stride - padding
	if stride == 1 && kh0 >= 0 && kh0+kSize <= inH && kw0 >= 0 && kw0+kSize <= inW {
		cnn2FillPatchFast(patch, input, b, oh, ow, inC, inH, inW, kSize, stride, padding)
	} else {
		cnn2FillPatch(patch, input, b, oh, ow, inC, inH, inW, kSize, stride, padding)
	}
	return simd.DotTile(patch, weights[wBase:wBase+kernelVol], 0, kernelVol, 0)
}

func cnn2SimdForwardSerial(input, preAct, postAct *Tensor[float32], weights []float32, batch, inH, inW, inC, outH, outW, filters, kSize, stride, padding, kernelVol int, activation ActivationType) {
	patch := make([]float32, kernelVol)
	for b := 0; b < batch; b++ {
		for f := 0; f < filters; f++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					sum := cnn2ConvDotSimd(input.Data, weights, patch, b, f, oh, ow, inC, inH, inW, kSize, stride, padding, kernelVol)
					outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
					preAct.Data[outIdx] = float32(sum)
					postAct.Data[outIdx] = Activate(float32(sum), activation)
				}
			}
		}
	}
}

func cnn2SimdForwardParallel(input, preAct, postAct *Tensor[float32], weights []float32, batch, inH, inW, inC, outH, outW, filters, kSize, stride, padding, kernelVol int, activation ActivationType) {
	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup

	for f := 0; f < filters; f++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(f int) {
			defer func() { <-sem; wg.Done() }()
			patch := make([]float32, kernelVol)
			for b := 0; b < batch; b++ {
				for oh := 0; oh < outH; oh++ {
					for ow := 0; ow < outW; ow++ {
						sum := cnn2ConvDotSimd(input.Data, weights, patch, b, f, oh, ow, inC, inH, inW, kSize, stride, padding, kernelVol)
						outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
						preAct.Data[outIdx] = float32(sum)
						postAct.Data[outIdx] = Activate(float32(sum), activation)
					}
				}
			}
		}(f)
	}
	wg.Wait()
}
