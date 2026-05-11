package poly

import (
	"runtime"
	"sync"
)

func useBitpackedCPUCNN1(layer *VolumetricLayer) bool {
	if !useNativeQuantCNN1(layer) {
		return false
	}
	switch layer.DType {
	case DTypeInt4, DTypeInt2, DTypeTernary, DTypeBinary:
		return true
	default:
		return false
	}
}

func cnn1PackedCPUWords(layer *VolumetricLayer) ([]uint32, float64, bool) {
	if layer == nil || layer.WeightStore == nil {
		return nil, 1.0, false
	}
	packed, ok := layer.WeightStore.GetNativePackedCPU(layer.DType).([]uint32)
	if !ok || len(packed) == 0 {
		return nil, 1.0, false
	}
	scale := float64(layer.WeightStore.Scale)
	if scale == 0 {
		scale = 1.0
	}
	return packed, scale, true
}

func cnn1PackedDecode(dtype DType, packed []uint32, idx int) float64 {
	switch dtype {
	case DTypeInt4:
		word := packed[idx/8]
		shift := uint((idx % 8) * 4)
		code := int8((word >> shift) & 0x0F)
		if code > 7 {
			code -= 16
		}
		return float64(code)
	case DTypeInt2:
		word := packed[idx/16]
		shift := uint((idx % 16) * 2)
		code := int8((word >> shift) & 0x03)
		if code > 1 {
			code -= 4
		}
		return float64(code)
	case DTypeTernary:
		word := packed[idx/16]
		shift := uint((idx % 16) * 2)
		code := uint8((word >> shift) & 0x03)
		switch code {
		case 0:
			return -1.0
		case 2:
			return 1.0
		default:
			return 0.0
		}
	case DTypeBinary:
		word := packed[idx/32]
		shift := uint(idx % 32)
		if ((word >> shift) & 0x01) != 0 {
			return 1.0
		}
		return -1.0
	default:
		return 0.0
	}
}

func CNN1ForwardPackedCPU[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if _, _, ok := cnn1PackedCPUWords(layer); !ok {
		return CNN1ForwardTiled(layer, input)
	}
	return cnn1ForwardPackedCPUParallel(layer, input)
}

func cnn1ForwardPackedCPUParallel[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	batchSize := input.Shape[0]
	seqLen, inC := layer.InputHeight, layer.InputChannels
	outLen, filters := layer.OutputHeight, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding

	packed, scale, ok := cnn1PackedCPUWords(layer)
	if !ok {
		return CNN1ForwardTiled(layer, input)
	}

	preAct = NewTensor[T](batchSize, filters, outLen)
	postAct = NewTensor[T](batchSize, filters, outLen)

	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup
	for b := 0; b < batchSize; b++ {
		for f := 0; f < filters; f++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(b, f int) {
				defer func() { <-sem; wg.Done() }()
				wBase := f * inC * kSize
				for o := 0; o < outLen; o++ {
					var sum float64
					for ic := 0; ic < inC; ic++ {
						for k := 0; k < kSize; k++ {
							inPos := o*stride + k - padding
							if inPos >= 0 && inPos < seqLen {
								inIdx := b*inC*seqLen + ic*seqLen + inPos
								wIdx := wBase + ic*kSize + k
								sum += float64(input.Data[inIdx]) * cnn1PackedDecode(layer.DType, packed, wIdx)
							}
						}
					}
					outIdx := b*filters*outLen + f*outLen + o
					val := T(sum * scale)
					preAct.Data[outIdx] = val
					postAct.Data[outIdx] = Activate(val, layer.Activation)
				}
			}(b, f)
		}
	}
	wg.Wait()
	return preAct, postAct
}

func CNN1BackwardPackedCPU[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	if _, _, ok := cnn1PackedCPUWords(layer); !ok {
		return CNN1BackwardTiled(layer, gradOutput, input, preAct)
	}
	return cnn1BackwardPackedCPUParallel(layer, gradOutput, input, preAct)
}

func cnn1BackwardPackedCPUParallel[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	batchSize := input.Shape[0]
	seqLen, inC := layer.InputHeight, layer.InputChannels
	outLen, filters := layer.OutputHeight, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding

	packed, scale, ok := cnn1PackedCPUWords(layer)
	if !ok {
		return CNN1BackwardTiled(layer, gradOutput, input, preAct)
	}

	gradInput = NewTensor[T](batchSize, inC, seqLen)
	gradWeights = NewTensor[T](filters, inC, kSize)
	gi64 := make([]float64, len(gradInput.Data))
	gw64 := make([]float64, len(gradWeights.Data))
	var mu sync.Mutex
	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup

	for b := 0; b < batchSize; b++ {
		for ic := 0; ic < inC; ic++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(b, ic int) {
				defer func() { <-sem; wg.Done() }()
				local := make([]float64, seqLen)
				for inPos := 0; inPos < seqLen; inPos++ {
					var sum float64
					for f := 0; f < filters; f++ {
						wBase := f * inC * kSize
						for k := 0; k < kSize; k++ {
							o := inPos + padding - k
							if o >= 0 && o%stride == 0 {
								o /= stride
								if o < outLen {
									outIdx := b*filters*outLen + f*outLen + o
									gOut := float64(gradOutput.Data[outIdx]) * float64(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
									wIdx := wBase + ic*kSize + k
									sum += gOut * cnn1PackedDecode(layer.DType, packed, wIdx)
								}
							}
						}
					}
					local[inPos] = sum * scale
				}
				mu.Lock()
				for pos, v := range local {
					gi64[b*inC*seqLen+ic*seqLen+pos] += v
				}
				mu.Unlock()
			}(b, ic)
		}
	}
	wg.Wait()

	for f := 0; f < filters; f++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(f int) {
			defer func() { <-sem; wg.Done() }()
			localGW := make([]float64, inC*kSize)
			for b := 0; b < batchSize; b++ {
				for o := 0; o < outLen; o++ {
					outIdx := b*filters*outLen + f*outLen + o
					gOut := float64(gradOutput.Data[outIdx]) * float64(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
					for ic := 0; ic < inC; ic++ {
						for k := 0; k < kSize; k++ {
							inPos := o*stride + k - padding
							if inPos >= 0 && inPos < seqLen {
								inIdx := b*inC*seqLen + ic*seqLen + inPos
								localGW[ic*kSize+k] += gOut * float64(input.Data[inIdx])
							}
						}
					}
				}
			}
			mu.Lock()
			for idx, v := range localGW {
				gw64[f*inC*kSize+idx] += v
			}
			mu.Unlock()
		}(f)
	}
	wg.Wait()

	for i := range gradInput.Data {
		gradInput.Data[i] = T(gi64[i])
	}
	for i := range gradWeights.Data {
		gradWeights.Data[i] = T(gw64[i])
	}
	return gradInput, gradWeights
}
