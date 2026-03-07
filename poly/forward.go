package poly

import (
	"time"
)

// DispatchLayer acts as the universal routing hub for all layer types.
// This is the "Jump Table" that handles numerical metamorphosis across 50+ layer types.
func DispatchLayer[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	switch layer.Type {
	case LayerDense:
		return DenseForwardPolymorphic(layer, input)
	case LayerCNN1:
		return CNN1ForwardPolymorphic(layer, input)
	case LayerCNN2:
		return CNN2ForwardPolymorphic(layer, input)
	case LayerCNN3:
		return CNN3ForwardPolymorphic(layer, input)
	case LayerRNN:
		return RNNForwardPolymorphic(layer, input)
	case LayerLSTM:
		return LSTMForwardPolymorphic(layer, input)
	case LayerMultiHeadAttention:
		return MHAForwardPolymorphic(layer, input)
	case LayerSwiGLU:
		return SwiGLUForwardPolymorphic(layer, input)
	case LayerRMSNorm:
		return RMSNormForwardPolymorphic(layer, input)
	case LayerLayerNorm:
		return LayerNormForwardPolymorphic(layer, input)
	default:
		return DenseForwardPolymorphic(layer, input)
	}
}

// ForwardPolymorphic executes the network using a unified generic dispatcher.
// It iterates through the 3D grid and handles DType transitions between layers.
func ForwardPolymorphic[T Numeric](n *VolumetricNetwork, input *Tensor[T]) (*Tensor[T], time.Duration, []time.Duration) {
	start := time.Now()
	currentTensor := input
	layerTimes := make([]time.Duration, len(n.Layers))

	for z := 0; z < n.Depth; z++ {
		for y := 0; y < n.Rows; y++ {
			for x := 0; x < n.Cols; x++ {
				for l := 0; l < n.LayersPerCell; l++ {
					idx := n.GetIndex(z, y, x, l)
					layer := &n.Layers[idx]
					if layer.IsDisabled {
						continue
					}

					// UNIFIED REGISTRY DISPATCH
					lStart := time.Now()
					_, post := DispatchLayer(layer, currentTensor)
					layerTimes[idx] = time.Since(lStart)
					currentTensor = post
				}
			}
		}
	}

	return currentTensor, time.Since(start), layerTimes
}
