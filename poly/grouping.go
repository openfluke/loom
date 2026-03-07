package poly

import (
	"fmt"
	"strings"
)

// DetectedTensor represents a tensor found in a model file.
type DetectedTensor struct {
	Name    string
	Shape   []int
	DType   string
	InSize  int
	OutSize int
	CanLoad bool
}

// GroupRelatedTensors identifies groups of tensors that belong to the same complex layer.
func GroupRelatedTensors(detected []DetectedTensor) map[string][]DetectedTensor {
	groups := make(map[string][]DetectedTensor)

	for _, d := range detected {
		if d.CanLoad { continue }
		parts := strings.Split(d.Name, ".")
		if len(parts) < 3 { continue }

		// Logic to find layer grouping prefix
		// Usually: model.layers.X.self_attn
		prefix := ""
		for i, p := range parts {
			if strings.HasPrefix(p, "layer") || strings.EqualFold(p, "layers") {
				if i+2 < len(parts) {
					prefix = strings.Join(parts[:i+3], ".")
					break
				}
			}
		}

		if prefix != "" {
			groups[prefix] = append(groups[prefix], d)
		}
	}

	return groups
}

// ReconstructSwiGLULayer builds a SwiGLU layer from gated MLP tensors.
func ReconstructSwiGLULayer(name string, tensors []DetectedTensor, dModel int) (*VolumetricLayer, error) {
	var gate, up, down bool
	for _, t := range tensors {
		n := strings.ToLower(t.Name)
		if strings.Contains(n, "gate") || strings.Contains(n, "w1") { gate = true }
		if strings.Contains(n, "up") || strings.Contains(n, "w3") { up = true }
		if strings.Contains(n, "down") || strings.Contains(n, "w2") { down = true }
	}

	if gate && up && down {
		fmt.Printf("Reconstructing SwiGLU layer: %s\n", name)
		return &VolumetricLayer{
			Type:         LayerSwiGLU,
			InputHeight:  dModel,
			OutputHeight: dModel,
		}, nil
	}
	return nil, fmt.Errorf("missing SwiGLU components in %s", name)
}

// ReconstructRMSNormLayer builds an RMSNorm layer.
func ReconstructRMSNormLayer(name string, tensors []DetectedTensor, dModel int) (*VolumetricLayer, error) {
	for _, t := range tensors {
		if strings.Contains(strings.ToLower(t.Name), "norm") || strings.Contains(strings.ToLower(t.Name), "weight") {
			fmt.Printf("Reconstructing RMSNorm layer: %s\n", name)
			return &VolumetricLayer{
				Type:        LayerRMSNorm,
				InputHeight: dModel,
			}, nil
		}
	}
	return nil, fmt.Errorf("missing RMSNorm components in %s", name)
}

// ReconstructMHALayer attempts to build a VolumetricLayer of type MultiHeadAttention from grouped tensors.
func ReconstructMHALayer(name string, tensors []DetectedTensor, dModel int, numHeads int) (*VolumetricLayer, error) {
	// Look for Q, K, V, O components
	var q, k, v, o bool
	for _, t := range tensors {
		n := strings.ToLower(t.Name)
		if strings.Contains(n, "q_proj") || strings.Contains(n, "query") { q = true }
		if strings.Contains(n, "k_proj") || strings.Contains(n, "key") { k = true }
		if strings.Contains(n, "v_proj") || strings.Contains(n, "value") { v = true }
		if strings.Contains(n, "o_proj") || strings.Contains(n, "out_proj") { o = true }
	}

	if q && k && v && o {
		fmt.Printf("Reconstructing MHA layer: %s\n", name)
		return &VolumetricLayer{
			Type:       LayerMultiHeadAttention,
			DModel:     dModel,
			NumHeads:   numHeads,
			NumKVHeads: numHeads,
			HeadDim:    dModel / numHeads,
			SeqLength:  1,
		}, nil
	}

	return nil, fmt.Errorf("missing components for MHA layer in group %s", name)
}

// ReconstructCNNLayer attempts to build a VolumetricLayer of type CNN from grouped tensors.
func ReconstructCNNLayer(name string, tensors []DetectedTensor, ltype LayerType) (*VolumetricLayer, error) {
	var weight bool
	for _, t := range tensors {
		n := strings.ToLower(t.Name)
		if strings.Contains(n, "weight") { weight = true }
	}

	if weight {
		fmt.Printf("Reconstructing CNN layer: %s\n", name)
		return &VolumetricLayer{
			Type: ltype,
			// Details like filters/ksize would be extracted from tensor shapes in a real loader
		}, nil
	}
	return nil, fmt.Errorf("missing CNN components in %s", name)
}

// ReconstructLayerNormLayer builds a LayerNorm layer.
func ReconstructLayerNormLayer(name string, tensors []DetectedTensor, dModel int) (*VolumetricLayer, error) {
	for _, t := range tensors {
		n := strings.ToLower(t.Name)
		if strings.Contains(n, "weight") || strings.Contains(n, "gamma") {
			fmt.Printf("Reconstructing LayerNorm layer: %s\n", name)
			return &VolumetricLayer{
				Type:        LayerLayerNorm,
				InputHeight: dModel,
			}, nil
		}
	}
	return nil, fmt.Errorf("missing LayerNorm components in %s", name)
}
