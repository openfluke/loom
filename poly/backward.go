package poly

import (
	"time"
)

// DispatchLayerBackward acts as the universal routing hub for gradients.
// This handles the backward pass metamorphosis for various layer types.
func DispatchLayerBackward[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	switch layer.Type {
	case LayerDense:
		return DenseBackwardPolymorphic(layer, gradOutput, input, preAct)
	case LayerMultiHeadAttention:
		// Stubs for future stage 2/3 kernels
		return gradOutput, nil
	case LayerSwiGLU:
		return gradOutput, nil
	case LayerRMSNorm:
		return gradOutput, nil
	default:
		return gradOutput, nil
	}
}

// BackwardPolymorphic executes a full backward pass through the 3D grid.
// It propagates gradients from the output back to the input, accumulating weight gradients.
func BackwardPolymorphic[T Numeric](n *VolumetricNetwork, gradOutput *Tensor[T], inputs, preActs []*Tensor[T]) (gradInput *Tensor[T], layerGradients [][2]*Tensor[T], layerTimes []time.Duration) {
	currentGrad := gradOutput
	layerGradients = make([][2]*Tensor[T], len(n.Layers))
	layerTimes = make([]time.Duration, len(n.Layers))

	// Walk backwards through the 3D grid
	for z := n.Depth - 1; z >= 0; z-- {
		for y := n.Rows - 1; y >= 0; y-- {
			for x := n.Cols - 1; x >= 0; x-- {
				for l := n.LayersPerCell - 1; l >= 0; l-- {
					idx := n.GetIndex(z, y, x, l)
					layer := &n.Layers[idx]
					if layer.IsDisabled {
						continue
					}

					// Fetch historical state for this layer
					input := inputs[idx]
					preAct := preActs[idx]

					// DISPATCH BACKWARD
					lStart := time.Now()
					gIn, gW := DispatchLayerBackward(layer, currentGrad, input, preAct)
					layerTimes[idx] = time.Since(lStart)
					currentGrad = gIn
					layerGradients[idx] = [2]*Tensor[T]{gIn, gW}
				}
			}
		}
	}

	return currentGrad, layerGradients, layerTimes
}
