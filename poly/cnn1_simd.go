package poly

import (
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

func tryCNN1ForwardSimd[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T], ok bool) {
	in, ok := any(input).(*Tensor[float32])
	if !ok {
		return nil, nil, false
	}
	preF, postF := cnn1ForwardSimdF32(layer, in)
	return simdTensorsAs[T](preF, postF)
}

func cnn1ForwardSimdF32(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	layer.EnsureRuntimeTileSizes()

	// Only correctness-based formats fall back; explicit SIMD is honored at any width.
	if useBitpackedCPUCNN1(layer) {
		return CNN1ForwardPackedCPU(layer, input)
	}
	if useNativeQuantCNN1(layer) {
		return CNN1ForwardTiled(layer, input)
	}

	batchSize := input.Shape[0]
	seqLen, inC := layer.InputHeight, layer.InputChannels
	outLen, filters := layer.OutputHeight, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	kernelVol := inC * kSize

	preAct = NewTensor[float32](batchSize, filters, outLen)
	postAct = NewTensor[float32](batchSize, filters, outLen)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[float32](weights)

	useParallel := layer.EnableMultiCoreTiling && filters > 1
	if useParallel {
		cnn1SimdForwardParallel(input, preAct, postAct, wData, batchSize, seqLen, inC, outLen, filters, kSize, stride, padding, kernelVol, layer.Activation)
	} else {
		cnn1SimdForwardSerial(input, preAct, postAct, wData, batchSize, seqLen, inC, outLen, filters, kSize, stride, padding, kernelVol, layer.Activation)
	}
	return preAct, postAct
}

// cnn1ConvDotSimd accumulates one output pixel. For stride=1 uses contiguous per-channel DotTile
// (no im2col patch buffer); edges fall back to scalar taps.
func cnn1ConvDotSimd(input, weights []float32, b, f, o, inC, seqLen, kSize, stride, padding, kernelVol int) float64 {
	var sum float64
	wBase := f * kernelVol
	inBatch := b * inC * seqLen
	for ic := 0; ic < inC; ic++ {
		icInBase := inBatch + ic*seqLen
		icW := weights[wBase+ic*kSize : wBase+(ic+1)*kSize]
		inPos0 := o*stride - padding
		if stride == 1 && inPos0 >= 0 && inPos0+kSize <= seqLen {
			sum = simd.DotTile(input[icInBase+inPos0:], icW, 0, kSize, sum)
			continue
		}
		for k := 0; k < kSize; k++ {
			inPos := o*stride + k - padding
			if inPos >= 0 && inPos < seqLen {
				sum += float64(input[icInBase+inPos]) * float64(icW[k])
			}
		}
	}
	return sum
}

func cnn1SimdForwardSerial(input, preAct, postAct *Tensor[float32], weights []float32, batch, seqLen, inC, outLen, filters, kSize, stride, padding, kernelVol int, activation ActivationType) {
	for b := 0; b < batch; b++ {
		for f := 0; f < filters; f++ {
			for o := 0; o < outLen; o++ {
				sum := cnn1ConvDotSimd(input.Data, weights, b, f, o, inC, seqLen, kSize, stride, padding, kernelVol)
				outIdx := b*filters*outLen + f*outLen + o
				preAct.Data[outIdx] = float32(sum)
				postAct.Data[outIdx] = Activate(float32(sum), activation)
			}
		}
	}
}

func cnn1SimdForwardParallel(input, preAct, postAct *Tensor[float32], weights []float32, batch, seqLen, inC, outLen, filters, kSize, stride, padding, kernelVol int, activation ActivationType) {
	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup

	for f := 0; f < filters; f++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(f int) {
			defer func() { <-sem; wg.Done() }()
			for b := 0; b < batch; b++ {
				for o := 0; o < outLen; o++ {
					sum := cnn1ConvDotSimd(input.Data, weights, b, f, o, inC, seqLen, kSize, stride, padding, kernelVol)
					outIdx := b*filters*outLen + f*outLen + o
					preAct.Data[outIdx] = float32(sum)
					postAct.Data[outIdx] = Activate(float32(sum), activation)
				}
			}
		}(f)
	}
	wg.Wait()
}
