package nn

import (
	"fmt"
	//"math"
	"sort"
	"strconv"
	"strings"
)

// groupRelatedTensors identifies groups of tensors that belong to the same complex layer
// e.g., MultiHeadAttention (Q, K, V, O) or SwiGLU (Gate, Up, Down)
func groupRelatedTensors(detected []DetectedTensor) map[string][]DetectedTensor {
	groups := make(map[string][]DetectedTensor)
	
	for _, d := range detected {
		// Only group tensors that haven't been loaded yet
		if d.CanLoad {
			continue // Already handled
		}

		// Parse name to find layer prefix
		// Format usually: model.layers.0.self_attn.q_proj.weight
		parts := strings.Split(d.Name, ".")
		if len(parts) < 3 {
			continue
		}

		// Find numeral index to identify layer block
		layerIdx := -1
		prefixEnd := 0
		
		for i, part := range parts {
			if _, err := strconv.Atoi(part); err == nil {
				layerIdx = i
				prefixEnd = i
				break
			}
		}

		if layerIdx == -1 {
			continue // Not part of a numbered layer block
		}

		// Group key is the prefix up to the layer index + component
		// e.g. "model.layers.0.self_attn"
		// We need to look a bit deeper to distinguish MHA vs MLP
		
		// Common patterns:
		// MHA: ...self_attn.q_proj
		// MLP: ...mlp.gate_proj
		
		if prefixEnd+2 < len(parts) {
			// Check for attention key
			// e.g. "model.layers.0.self_attn"
			subBlock := parts[prefixEnd+1]
			
			key := strings.Join(parts[:prefixEnd+2], ".")
			
			// Detect MHA block
			if strings.Contains(subBlock, "attn") || strings.Contains(subBlock, "attention") {
				groups[key] = append(groups[key], d)
			}
			
			// Detect MLP/SwiGLU block
			if strings.Contains(subBlock, "mlp") || strings.Contains(subBlock, "ffn") {
				groups[key] = append(groups[key], d)
			}
		}
	}
	
	return groups
}

// processMultiHeadAttentionGroups converts grouped tensors into LayerMultiHeadAttention configs
func (n *Network) processMultiHeadAttentionGroups(groups map[string][]DetectedTensor, tensors map[string]TensorWithShape, config GenericModelConfig) {
	// Sort keys for deterministic order
	keys := make([]string, 0, len(groups))
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	
	for _, key := range keys {
		group := groups[key]
		
		// Check if this group looks like MHA
		// Needs Q, K, V, O weights
		var q, k, v, o DetectedTensor
		var qFound, kFound, vFound, oFound bool
		
		for _, t := range group {
			name := strings.ToLower(t.Name)
			if strings.Contains(name, "q_proj") || strings.Contains(name, "query") {
				q = t
				qFound = true
			} else if strings.Contains(name, "k_proj") || strings.Contains(name, "key") {
				k = t
				kFound = true
			} else if strings.Contains(name, "v_proj") || strings.Contains(name, "value") {
				v = t
				vFound = true
			} else if strings.Contains(name, "o_proj") || strings.Contains(name, "out_proj") || strings.Contains(name, "output") {
				o = t
				oFound = true
			}
		}
		
		if qFound && kFound && vFound && oFound {
			// Found logical MHA block!
			fmt.Printf("Building MHA Layer from group: %s\n", key)
			
			// Get actual tensor values
			qT := tensors[q.Name]
			kT := tensors[k.Name]
			vT := tensors[v.Name]
			oT := tensors[o.Name]
			
			// Look for biases
			qBias := findMatchingBias(tensors, q.Name)
			kBias := findMatchingBias(tensors, k.Name)
			vBias := findMatchingBias(tensors, v.Name)
			oBias := findMatchingBias(tensors, o.Name)
			
			// Make sure biases exist (create zero bias if nil)
			if qBias == nil { qBias = make([]float32, q.OutSize) }
			if kBias == nil { kBias = make([]float32, k.OutSize) }
			if vBias == nil { vBias = make([]float32, v.OutSize) }
			if oBias == nil { oBias = make([]float32, o.OutSize) }
			
			// Determine MHA params from shapes/config
			hiddenSize := q.InSize
			numHeads := config.NumHeads
			numKVHeads := config.NumKVHeads
			if numHeads == 0 { numHeads = 12 } // guess
			if numKVHeads == 0 { numKVHeads = numHeads }
			
			headDim := hiddenSize / numHeads
			if headDim == 0 { headDim = 64 } // reasonable default for Llama-like
			
			n.Layers = append(n.Layers, LayerConfig{
				Type: LayerMultiHeadAttention,
				
				// Weights
				QWeights: qT.Values,
				KWeights: kT.Values,
				VWeights: vT.Values,
				OutputWeight: oT.Values,
				
				// Biases
				QBias: qBias,
				KBias: kBias,
				VBias: vBias,
				OutputBias: oBias,
				
				// Config
				DModel:     hiddenSize,
				NumHeads:   numHeads,
				NumKVHeads: numKVHeads,
				HeadDim:    headDim,
			})
		}
	}
}
