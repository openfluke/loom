package poly

import (
	"fmt"
	"strings"
)

// PrefixWeightMapper handles mapping tensors with potentially complex prefixes
type PrefixWeightMapper struct {
	Patterns map[string][]string
}

// NewPrefixWeightMapper creates a default mapper for common LLM architectures
func NewPrefixWeightMapper() *PrefixWeightMapper {
	return &PrefixWeightMapper{
		Patterns: map[string][]string{
			"embeddings": {
				"model.embed_tokens.weight",
				"transformer.wte.weight",
				"embeddings.weight",
				"embed_tokens.weight",
			},
			"final_norm": {
				"model.norm.weight",
				"transformer.ln_f.weight",
				"ln_f.weight",
				"norm.weight",
			},
			"lm_head": {
				"lm_head.weight",
				"output.weight",
			},
		},
	}
}

// MapWeights finds weights for specific roles in the provided tensor map, handling generic prefixes
func (m *PrefixWeightMapper) MapWeights(tensors map[string][]float32) (embeddings, lmHead, finalNorm []float32, hasFinalNorm bool) {
	embeddings = m.Find(tensors, "embeddings")
	finalNorm = m.Find(tensors, "final_norm")
	hasFinalNorm = (finalNorm != nil)

	lmHead = m.Find(tensors, "lm_head")
	if lmHead == nil {
		fmt.Printf("ℹ️  No separate lm_head found, assuming tied weights (using embeddings).\n")
		lmHead = embeddings
	}

	return
}

// Find searches for a tensor based on the patterns registered for a role
func (m *PrefixWeightMapper) Find(tensors map[string][]float32, role string) []float32 {
	patterns, ok := m.Patterns[role]
	if !ok {
		return nil
	}

	for _, pattern := range patterns {
		// Exact match first
		if t, ok := tensors[pattern]; ok {
			fmt.Printf("  ✓ Loaded %s: %d values (role: %s)\n", pattern, len(t), role)
			return t
		}

		// Suffix match
		for k, v := range tensors {
			if strings.HasSuffix(k, pattern) {
				fmt.Printf("  ✓ Loaded %s (via suffix: %s): %d values (role: %s)\n", k, pattern, len(v), role)
				return v
			}
		}
	}

	return nil
}

// LoadWithPrefixes loads weights into a VolumetricNetwork by interpreting layer indices and prefixes
func LoadWithPrefixes(net *VolumetricNetwork, tensors map[string][]float32) error {
	for k, v := range tensors {
		parts := strings.Split(k, ".")
		layerIdx := -1
		for i, p := range parts {
			if p == "layers" || p == "h" {
				if i+1 < len(parts) {
					fmt.Sscanf(parts[i+1], "%d", &layerIdx)
					break
				}
			}
		}

		if layerIdx == -1 {
			continue
		}

		// Map to 4 slots per block
		// 0: input_layernorm
		// 1: self_attn
		// 2: post_attention_layernorm
		// 3: mlp

		slot := -1
		subRole := ""

		if strings.Contains(k, "input_layernorm") || strings.Contains(k, "ln_1") {
			slot = 0
		} else if strings.Contains(k, "self_attn") || strings.Contains(k, "attn") {
			slot = 1
			if strings.Contains(k, "q_proj") { subRole = "q" }
			if strings.Contains(k, "k_proj") { subRole = "k" }
			if strings.Contains(k, "v_proj") { subRole = "v" }
			if strings.Contains(k, "o_proj") { subRole = "o" }
		} else if strings.Contains(k, "post_attention_layernorm") || strings.Contains(k, "ln_2") {
			slot = 2
		} else if strings.Contains(k, "mlp") {
			slot = 3
			if strings.Contains(k, "gate_proj") || strings.Contains(k, "w1") { subRole = "g" }
			if strings.Contains(k, "up_proj") || strings.Contains(k, "w3") { subRole = "u" }
			if strings.Contains(k, "down_proj") || strings.Contains(k, "w2") { subRole = "d" }
		}

		if slot == -1 {
			continue
		}

		layer := net.GetLayer(0, 0, 0, layerIdx*4 + slot)
		if layer == nil {
			continue
		}

		if subRole == "" {
			// Norm or similar (single weight)
			copyWeights(layer, "w", v)
		} else {
			copyWeights(layer, subRole, v)
		}
	}

	fmt.Printf("✅ Finished loading weights with prefixes.\n")
	return nil
}

func copyWeights(layer *VolumetricLayer, role string, data []float32) {
	if layer.WeightStore == nil {
		return
	}
	
	dModel := layer.DModel
	if dModel == 0 { dModel = layer.InputHeight }
	kvDim := layer.NumKVHeads * layer.HeadDim
	if kvDim == 0 { kvDim = dModel }
	intermediateSize := layer.OutputHeight

	offset := -1
	switch layer.Type {
	case LayerMultiHeadAttention:
		switch role {
		case "q": offset = 0
		case "k": offset = dModel * dModel
		case "v": offset = dModel * (dModel + kvDim)
		case "o": offset = dModel * (dModel + 2 * kvDim)
		}
	case LayerSwiGLU:
		switch role {
		case "g": offset = 0
		case "u": offset = dModel * intermediateSize
		case "d": offset = 2 * dModel * intermediateSize
		}
	case LayerRMSNorm:
		if role == "w" { offset = 0 }
	}

	if offset != -1 && offset+len(data) <= len(layer.WeightStore.Master) {
		copy(layer.WeightStore.Master[offset:], data)
		// Noisy logging:
		// fmt.Printf("    - Matched role %s (offset %d)\n", role, offset)
	}
}
