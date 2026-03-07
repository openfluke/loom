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
	case LayerCNN1:
		return CNN1BackwardPolymorphic(layer, gradOutput, input, preAct)
	case LayerCNN2:
		return CNN2BackwardPolymorphic(layer, gradOutput, input, preAct)
	case LayerCNN3:
		return CNN3BackwardPolymorphic(layer, gradOutput, input, preAct)
	case LayerRNN:
		return RNNBackwardPolymorphic(layer, gradOutput, input, preAct)
	case LayerLSTM:
		return LSTMBackwardPolymorphic(layer, gradOutput, input, preAct)
	case LayerMultiHeadAttention:
		return MHABackwardPolymorphic(layer, gradOutput, input, preAct)
	case LayerSwiGLU:
		return SwiGLUBackwardPolymorphic(layer, gradOutput, input, preAct)
	case LayerRMSNorm:
		return RMSNormBackwardPolymorphic(layer, gradOutput, input, preAct)
	case LayerLayerNorm:
		return LayerNormBackwardPolymorphic(layer, gradOutput, input, preAct)
	case LayerConvTransposed1D:
		return ConvTransposed1DBackwardPolymorphic(layer, gradOutput, input, preAct)
	case LayerConvTransposed2D:
		return ConvTransposed2DBackwardPolymorphic(layer, gradOutput, input, preAct)
	case LayerConvTransposed3D:
		return ConvTransposed3DBackwardPolymorphic(layer, gradOutput, input, preAct)
	case LayerEmbedding:
		return EmbeddingBackwardPolymorphic(layer, gradOutput, input, preAct)
	case LayerKMeans:
		return KMeansBackwardPolymorphic(layer, gradOutput, input, preAct)
	case LayerSoftmax:
		return SoftmaxBackwardPolymorphic(layer, gradOutput, input, preAct)
	case LayerParallel:
		return ParallelBackwardPolymorphic(layer, gradOutput, input, preAct)
	case LayerSequential:
		return SequentialBackwardPolymorphic(layer, gradOutput, input, preAct)
	default:
		return DenseBackwardPolymorphic(layer, gradOutput, input, preAct)
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
