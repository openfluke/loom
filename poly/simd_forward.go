package poly

import "github.com/openfluke/loom/poly/simd"

// Plan9SimdEnabled reports whether AVX2/NEON tile dots are linked for this GOARCH.
func Plan9SimdEnabled() bool {
	return simd.SimdEnabled()
}

func layerUseSimdForward(layer *VolumetricLayer) bool {
	if layer == nil {
		return false
	}
	if layer.UseSimdForward {
		return true
	}
	if layer.Network != nil && layer.Network.UseSimdForward {
		return true
	}
	return false
}

// Plan9SimdForwardForLayer reports whether this layer type has a Plan 9 SIMD CPU forward.
func Plan9SimdForwardForLayer(t LayerType) bool {
	if !simd.SimdEnabled() {
		return false
	}
	switch t {
	case LayerDense, LayerSwiGLU:
		return true
	default:
		return false
	}
}

// LayerSupportsSimdForward lists layer types with Plan 9 SIMD forward (when linked).
func LayerSupportsSimdForward(t LayerType) bool {
	switch t {
	case LayerDense, LayerSwiGLU:
		return true
	default:
		return false
	}
}

// SetSimdForward enables Plan 9 SIMD forward (Dense, SwiGLU) on the network and every layer.
func (n *VolumetricNetwork) SetSimdForward(enabled bool) {
	if n == nil {
		return
	}
	n.UseSimdForward = enabled
	for i := range n.Layers {
		n.Layers[i].UseSimdForward = enabled
	}
}

func setSimdForwardTree(layer *VolumetricLayer, enabled bool) {
	if layer == nil {
		return
	}
	layer.UseSimdForward = enabled
	for i := range layer.ParallelBranches {
		setSimdForwardTree(&layer.ParallelBranches[i], enabled)
	}
	for i := range layer.SequentialLayers {
		setSimdForwardTree(&layer.SequentialLayers[i], enabled)
	}
	if layer.MetaObservedLayer != nil {
		setSimdForwardTree(layer.MetaObservedLayer, enabled)
	}
}

// SetSimdForwardRecursive enables SIMD forward on the network and nested sub-layers.
func (n *VolumetricNetwork) SetSimdForwardRecursive(enabled bool) {
	if n == nil {
		return
	}
	n.UseSimdForward = enabled
	for i := range n.Layers {
		n.Layers[i].UseSimdForward = enabled
		setSimdForwardTree(&n.Layers[i], enabled)
	}
}
