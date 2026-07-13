package poly

// residual_simd.go — SIMD residual forward for non-native-exact polymorphic path.

func tryResidualForwardSimd[T Numeric](layer *VolumetricLayer, input, skip *Tensor[T]) (preAct, postAct *Tensor[T], ok bool) {
	in, okIn := any(input).(*Tensor[float32])
	sk, okSk := any(skip).(*Tensor[float32])
	if !okIn || !okSk {
		return nil, nil, false
	}
	outF := residualForwardSimdF32(layer, in, sk)
	out, okOut := simdTensorAsBackward[T](outF)
	if !okOut {
		return nil, nil, false
	}
	preAct = &Tensor[T]{Nested: []*Tensor[T]{skip}}
	return preAct, out, true
}
