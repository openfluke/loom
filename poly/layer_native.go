package poly

// layer_native.go — shared native dtype routing. Layer-specific paths live in *_native.go.

// IsLayerNativeExactDType reports dtypes that can use native layer paths.
func IsLayerNativeExactDType(dtype DType) bool {
	return IsDenseNativeExactDType(dtype)
}

func layerSupportsNativeExact(lt LayerType) bool {
	switch lt {
	case LayerDense, LayerSwiGLU, LayerMultiHeadAttention,
		LayerCNN1, LayerCNN2, LayerCNN3,
		LayerRNN, LayerLSTM, LayerEmbedding, LayerResidual:
		return true
	default:
		return false
	}
}

func useLayerNativeExact(layer *VolumetricLayer) bool {
	return layer != nil &&
		layer.Network != nil &&
		layer.Network.UseExactDType &&
		layer.WeightStore != nil &&
		layerSupportsNativeExact(layer.Type) &&
		IsLayerNativeExactDType(layer.DType)
}

// LayerUsesNativeExact reports whether a layer runs the native path for its type.
func LayerUsesNativeExact(layer *VolumetricLayer) bool {
	if layer == nil {
		return false
	}
	switch layer.Type {
	case LayerDense:
		return DenseUsesNativeExact(layer)
	case LayerResidual:
		return layer.Network != nil && layer.Network.UseExactDType
	default:
		return useLayerNativeExact(layer)
	}
}
