package poly

import (
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

// ResidualForwardPolymorphic adds a residual connection: output = input + skip.
func ResidualForwardPolymorphic[T Numeric](layer *VolumetricLayer, input, skip *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if useResidualNativeExact(layer) {
		return ResidualForwardNativeExact(layer, input, skip)
	}
	if skip == nil || len(skip.Data) != len(input.Data) {
		return input, input.Clone()
	}
	if layerUseSimdForward(layer) && simd.SimdEnabled() {
		if pre, post, ok := tryResidualForwardSimd(layer, input, skip); ok {
			return pre, post
		}
	}
	return ResidualForwardTiled(layer, input, skip)
}

// ResidualBackwardPolymorphic computes gradients for Residual layer.
func ResidualBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	if useResidualNativeExact(layer) {
		return ResidualBackwardNativeExact(layer, gradOutput, input, preAct)
	}
	return ResidualBackwardTiled(layer, gradOutput, input, preAct)
}

// ResidualForwardTiled performs a tiled forward pass for Residual (multi-core).
func ResidualForwardTiled[T Numeric](layer *VolumetricLayer, input, skip *Tensor[T]) (preAct, postAct *Tensor[T]) {
	output := NewTensor[T](input.Shape...)
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 1024
	}

	residualForwardTiledParallel(input.Data, skip.Data, output.Data, tileSize)

	preAct = &Tensor[T]{
		Nested: []*Tensor[T]{skip},
	}
	return preAct, output
}

// residualForwardTiledParallel adds input+skip in parallel tiles across all CPU cores.
func residualForwardTiledParallel[T Numeric](input, skip, output []T, tileSize int) {
	n := len(input)
	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	sem := make(chan struct{}, numCPUs)

	for t := 0; t < n; t += tileSize {
		sem <- struct{}{}
		wg.Add(1)
		go func(t int) {
			defer func() { <-sem; wg.Done() }()
			end := t + tileSize
			if end > n {
				end = n
			}
			for i := t; i < end; i++ {
				output[i] = input[i] + skip[i]
			}
		}(t)
	}
	wg.Wait()
}

// ResidualBackwardTiled performs a tiled backward pass for Residual.
func ResidualBackwardTiled[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	gradInput = gradOutput.Clone()
	
	// gradWeights for Residual is used here to return gradSkip
	var gradSkip *Tensor[T]
	if preAct != nil && len(preAct.Nested) > 0 {
		gradSkip = gradOutput.Clone()
	}

	gradWeights = &Tensor[T]{
		Nested: []*Tensor[T]{gradSkip},
	}
	
	return gradInput, gradWeights
}
