package poly

// residual_native.go — residual add in native-exact mode (no weights).

func useResidualNativeExact(layer *VolumetricLayer) bool {
	return layer != nil &&
		layer.Network != nil &&
		layer.Network.UseExactDType &&
		layer.Type == LayerResidual
}

func ResidualForwardNativeExact[T Numeric](layer *VolumetricLayer, input, skip *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if skip == nil || len(skip.Data) != len(input.Data) {
		return input, input.Clone()
	}
	in, okIn := any(input).(*Tensor[float32])
	sk, okSk := any(skip).(*Tensor[float32])
	if !okIn || !okSk {
		return ResidualForwardTiled(layer, input, skip)
	}
	var outF *Tensor[float32]
	if layerUseSimdForward(layer) {
		if out, simdOK := tryResidualForwardNativeSimd(layer, in, sk); simdOK {
			outF = out
		}
	}
	if outF == nil {
		outF = residualForwardNative(in, sk)
	}
	out, okOut := nativeTensorAs[T](outF)
	if !okOut {
		return ResidualForwardTiled(layer, input, skip)
	}
	preAct = &Tensor[T]{Nested: []*Tensor[T]{skip}}
	return preAct, out
}

func ResidualBackwardNativeExact[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	goT, ok := any(gradOutput).(*Tensor[float32])
	if !ok {
		return ResidualBackwardTiled(layer, gradOutput, input, preAct)
	}
	var giF *Tensor[float32]
	if layerUseSimdForward(layer) {
		if gi, simdOK := tryResidualBackwardNativeSimd(layer, goT); simdOK {
			giF = gi
		}
	}
	if giF == nil {
		giF = residualBackwardNative(goT)
	}
	gi, okGI := nativeTensorAs[T](giF)
	if !okGI {
		return ResidualBackwardTiled(layer, gradOutput, input, preAct)
	}
	var gradSkip *Tensor[T]
	if preAct != nil && len(preAct.Nested) > 0 && preAct.Nested[0] != nil {
		gradSkip = gradOutput.Clone()
	}
	gradWeights = &Tensor[T]{Nested: []*Tensor[T]{gradSkip}}
	return gi, gradWeights
}

func residualForwardNative(input, skip *Tensor[float32]) *Tensor[float32] {
	output := NewTensor[float32](input.Shape...)
	for i := range output.Data {
		output.Data[i] = input.Data[i] + skip.Data[i]
	}
	return output
}

func residualBackwardNative(gradOutput *Tensor[float32]) *Tensor[float32] {
	return gradOutput.Clone()
}
