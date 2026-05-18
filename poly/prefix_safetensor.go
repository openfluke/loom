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

// MapWeightsFromStored decodes global tensors (embeddings, lm_head, final norm) from raw payloads.
func (m *PrefixWeightMapper) MapWeightsFromStored(tensors map[string]HFStoredTensor) (embeddings, lmHead, finalNorm []float32, hasFinalNorm bool) {
	embeddings = m.findStoredAsFloat32(tensors, "embeddings")
	finalNorm = m.findStoredAsFloat32(tensors, "final_norm")
	hasFinalNorm = (finalNorm != nil)

	lmHead = m.findStoredAsFloat32(tensors, "lm_head")
	if lmHead == nil {
		fmt.Printf("ℹ️  No separate lm_head found, assuming tied weights (using embeddings).\n")
		lmHead = embeddings
	}

	return
}

func (m *PrefixWeightMapper) findStoredAsFloat32(tensors map[string]HFStoredTensor, role string) []float32 {
	patterns, ok := m.Patterns[role]
	if !ok {
		return nil
	}

	for _, pattern := range patterns {
		if t, ok := tensors[pattern]; ok {
			out, err := decodeHFStoredTensorToFloat32(t, pattern)
			if err != nil {
				fmt.Printf("⚠️  skipping %s: %v\n", pattern, err)
				return nil
			}
			fmt.Printf("  ✓ Loaded %s: %d values (role: %s, dtype=%s)\n", pattern, len(out), role, t.DType)
			return out
		}

		for k, t := range tensors {
			if strings.HasSuffix(k, pattern) {
				out, err := decodeHFStoredTensorToFloat32(t, k)
				if err != nil {
					fmt.Printf("⚠️  skipping %s: %v\n", k, err)
					return nil
				}
				fmt.Printf("  ✓ Loaded %s (via suffix: %s): %d values (role: %s, dtype=%s)\n", k, pattern, len(out), role, t.DType)
				return out
			}
		}
	}

	return nil
}

func decodeHFStoredTensorToFloat32(t HFStoredTensor, nameForErr string) ([]float32, error) {
	n := StoredTensorNumElements(t)
	if n <= 0 {
		return nil, fmt.Errorf("tensor %s: empty shape", nameForErr)
	}
	want, err := safetensorPayloadByteLength(t.DType, n)
	if err != nil {
		return nil, fmt.Errorf("tensor %s: %w", nameForErr, err)
	}
	if len(t.Bytes) != want {
		return nil, fmt.Errorf("tensor %s: byte len %d != expected %d for dtype=%s n=%d", nameForErr, len(t.Bytes), want, t.DType, n)
	}
	out := make([]float32, n)
	if err := DecodeTensorBytesInto(out, t.Bytes, t.DType); err != nil {
		return nil, fmt.Errorf("tensor %s: %w", nameForErr, err)
	}
	return out, nil
}

// LoadWithPrefixes loads weights into a VolumetricNetwork by interpreting layer indices and prefixes
func LoadWithPrefixes(net *VolumetricNetwork, tensors map[string][]float32) error {
	for k, v := range tensors {
		layerIdx, slot, subRole, ok := decodeHFPrefixRouting(k)
		if !ok {
			continue
		}

		layer := net.GetLayer(0, 0, 0, layerIdx*4+slot)
		if layer == nil {
			continue
		}

		if subRole == "" {
			copyWeights(layer, "w", v)
		} else {
			copyWeights(layer, subRole, v)
		}
	}

	fmt.Printf("✅ Finished loading weights with prefixes.\n")
	return nil
}

// LoadWithPrefixesFromHFStored maps HF decoder tensors from raw safetensors payloads (HFStoredTensor)
// without expanding them to []float32 first: U8 microsoft BitLinear slabs go straight into CPUPacked;
// BF16/F32 dense weights decode directly into WeightStore.Master.
func LoadWithPrefixesFromHFStored(net *VolumetricNetwork, tensors map[string]HFStoredTensor) error {
	for k, t := range tensors {
		layerIdx, slot, subRole, ok := decodeHFPrefixRouting(k)
		if !ok {
			continue
		}

		layer := net.GetLayer(0, 0, 0, layerIdx*4+slot)
		if layer == nil {
			continue
		}

		if subRole == "" {
			copyWeightsFromHFStored(layer, "w", t)
		} else {
			copyWeightsFromHFStored(layer, subRole, t)
		}
	}

	fmt.Printf("✅ Finished loading weights with prefixes.\n")
	return nil
}

func decodeHFPrefixRouting(k string) (layerIdx int, slot int, subRole string, ok bool) {
	layerIdx = -1
	parts := strings.Split(k, ".")
	for i, p := range parts {
		if p == "layers" || p == "h" {
			if i+1 < len(parts) {
				fmt.Sscanf(parts[i+1], "%d", &layerIdx)
				break
			}
		}
	}

	if layerIdx < 0 {
		return 0, 0, "", false
	}

	slot = -1
	subRole = ""

	if strings.Contains(k, "inner_attn_ln") || strings.Contains(k, "attn_sub_norm") {
		slot = 1
		subRole = "inner_norm"
	} else if strings.Contains(k, "ffn_layernorm") || strings.Contains(k, "ffn_sub_norm") {
		slot = 3
		subRole = "inner_norm"
	} else if strings.Contains(k, "input_layernorm") || strings.Contains(k, "ln_1") {
		slot = 0
	} else if strings.Contains(k, "self_attn") || strings.Contains(k, "attn") {
		slot = 1
		isScale := strings.Contains(k, "weight_scale")
		if strings.Contains(k, "q_proj") {
			subRole = "q"
		}
		if strings.Contains(k, "k_proj") {
			subRole = "k"
		}
		if strings.Contains(k, "v_proj") {
			subRole = "v"
		}
		if strings.Contains(k, "o_proj") {
			subRole = "o"
		}
		if isScale && subRole != "" {
			subRole += "_scale"
		}
		if strings.Contains(k, "q_norm") {
			subRole = "qn"
		}
		if strings.Contains(k, "k_norm") {
			subRole = "kn"
		}
	} else if strings.Contains(k, "post_attention_layernorm") || strings.Contains(k, "ln_2") {
		slot = 2
	} else if strings.Contains(k, "mlp") {
		slot = 3
		isScale := strings.Contains(k, "weight_scale")
		if strings.Contains(k, "gate_proj") || strings.Contains(k, "w1") {
			subRole = "g"
		}
		if strings.Contains(k, "up_proj") || strings.Contains(k, "w3") {
			subRole = "u"
		}
		if strings.Contains(k, "down_proj") || strings.Contains(k, "w2") {
			subRole = "d"
		}
		if isScale && subRole != "" {
			subRole += "_scale"
		}
	}

	if slot < 0 {
		return 0, 0, "", false
	}
	return layerIdx, slot, subRole, true
}

// copyWeights maps one HF tensor into a decoder sub-layer. BitLinear-style
// projections may arrive either as dense float weights (copied into Master and
// later packed by PrepareDecoderBlockBitNetTernaryCPU) or in the microsoft/
// bitnet-b1.58 offline packed layout accepted by SetMicrosoftBitNetPackedMatrix
// (no dense FP32 slab in Master for that matrix).
func copyWeights(layer *VolumetricLayer, role string, data []float32) {
	if layer.WeightStore == nil {
		return
	}

	dModel := layer.DModel
	if dModel == 0 {
		dModel = layer.InputHeight
	}
	qDim := layer.QueryDim
	if qDim == 0 {
		if layer.NumHeads > 0 && layer.HeadDim > 0 {
			qDim = layer.NumHeads * layer.HeadDim
		} else {
			qDim = dModel
		}
	}
	kvDim := layer.NumKVHeads * layer.HeadDim
	if kvDim == 0 {
		kvDim = dModel
	}
	intermediateSize := layer.OutputHeight

	offset := -1
	scaleOffset := -1
	rows := 0
	cols := 0
	switch layer.Type {
	case LayerMultiHeadAttention:
		switch role {
		case "q":
			offset = 0
			rows, cols = qDim, dModel
		case "q_scale":
			scaleOffset = 0
		case "k":
			offset = qDim * dModel
			rows, cols = kvDim, dModel
		case "k_scale":
			scaleOffset = qDim * dModel
		case "v":
			offset = qDim*dModel + kvDim*dModel
			rows, cols = kvDim, dModel
		case "v_scale":
			scaleOffset = qDim*dModel + kvDim*dModel
		case "o":
			offset = qDim*dModel + 2*kvDim*dModel
			rows, cols = dModel, qDim
		case "o_scale":
			scaleOffset = qDim*dModel + 2*kvDim*dModel
		case "qn":
			layer.QNormWeight = append([]float32(nil), data...)
		case "kn":
			layer.KNormWeight = append([]float32(nil), data...)
		case "inner_norm":
			layer.InnerNormWeight = append([]float32(nil), data...)
		}
	case LayerSwiGLU:
		switch role {
		case "g":
			offset = 0
			rows, cols = intermediateSize, dModel
		case "g_scale":
			scaleOffset = 0
		case "u":
			offset = dModel * intermediateSize
			rows, cols = intermediateSize, dModel
		case "u_scale":
			scaleOffset = dModel * intermediateSize
		case "d":
			offset = 2 * dModel * intermediateSize
			rows, cols = dModel, intermediateSize
		case "d_scale":
			scaleOffset = 2 * dModel * intermediateSize
		case "inner_norm":
			layer.InnerNormWeight = append([]float32(nil), data...)
		}
	case LayerRMSNorm:
		if role == "w" {
			offset = 0
		}
	}

	if scaleOffset != -1 && len(data) > 0 {
		layer.WeightStore.SetBitNetPackedScale(scaleOffset, data[0])
		return
	}

	if offset != -1 && rows > 0 && cols > 0 && len(data)*4 == rows*cols {
		if layer.WeightStore.SetMicrosoftBitNetPackedMatrix(offset, rows, cols, data) {
			return
		}
	}

	if offset != -1 && offset+len(data) <= len(layer.WeightStore.Master) {
		copy(layer.WeightStore.Master[offset:], data)
		// Noisy logging:
		// fmt.Printf("    - Matched role %s (offset %d)\n", role, offset)
	}
}

func copyWeightsFromHFStored(layer *VolumetricLayer, role string, t HFStoredTensor) {
	if layer.WeightStore == nil {
		return
	}

	dModel := layer.DModel
	if dModel == 0 {
		dModel = layer.InputHeight
	}
	qDim := layer.QueryDim
	if qDim == 0 {
		if layer.NumHeads > 0 && layer.HeadDim > 0 {
			qDim = layer.NumHeads * layer.HeadDim
		} else {
			qDim = dModel
		}
	}
	kvDim := layer.NumKVHeads * layer.HeadDim
	if kvDim == 0 {
		kvDim = dModel
	}
	intermediateSize := layer.OutputHeight

	offset := -1
	scaleOffset := -1
	rows := 0
	cols := 0
	switch layer.Type {
	case LayerMultiHeadAttention:
		switch role {
		case "q":
			offset = 0
			rows, cols = qDim, dModel
		case "q_scale":
			scaleOffset = 0
		case "k":
			offset = qDim * dModel
			rows, cols = kvDim, dModel
		case "k_scale":
			scaleOffset = qDim * dModel
		case "v":
			offset = qDim*dModel + kvDim*dModel
			rows, cols = kvDim, dModel
		case "v_scale":
			scaleOffset = qDim*dModel + kvDim*dModel
		case "o":
			offset = qDim*dModel + 2*kvDim*dModel
			rows, cols = dModel, qDim
		case "o_scale":
			scaleOffset = qDim*dModel + 2*kvDim*dModel
		case "qn":
			if vec, err := decodeHFStoredTensorToFloat32(t, "qn"); err == nil {
				layer.QNormWeight = append([]float32(nil), vec...)
			}
			return
		case "kn":
			if vec, err := decodeHFStoredTensorToFloat32(t, "kn"); err == nil {
				layer.KNormWeight = append([]float32(nil), vec...)
			}
			return
		case "inner_norm":
			if vec, err := decodeHFStoredTensorToFloat32(t, "inner_norm"); err == nil {
				layer.InnerNormWeight = append([]float32(nil), vec...)
			}
			return
		}
	case LayerSwiGLU:
		switch role {
		case "g":
			offset = 0
			rows, cols = intermediateSize, dModel
		case "g_scale":
			scaleOffset = 0
		case "u":
			offset = dModel * intermediateSize
			rows, cols = intermediateSize, dModel
		case "u_scale":
			scaleOffset = dModel * intermediateSize
		case "d":
			offset = 2 * dModel * intermediateSize
			rows, cols = dModel, intermediateSize
		case "d_scale":
			scaleOffset = 2 * dModel * intermediateSize
		case "inner_norm":
			if vec, err := decodeHFStoredTensorToFloat32(t, "inner_norm"); err == nil {
				layer.InnerNormWeight = append([]float32(nil), vec...)
			}
			return
		}
	case LayerRMSNorm:
		if role == "w" {
			offset = 0
		}
	}

	if scaleOffset != -1 && len(t.Bytes) > 0 {
		n := StoredTensorNumElements(t)
		if n >= 1 {
			buf := make([]float32, n)
			if err := DecodeTensorBytesInto(buf, t.Bytes, t.DType); err == nil {
				layer.WeightStore.SetBitNetPackedScale(scaleOffset, buf[0])
			}
		}
		return
	}

	// Microsoft offline packed: (rows/4)*cols bytes, 2 bits per weight lane.
	if offset != -1 && rows > 0 && cols > 0 && rows%4 == 0 && len(t.Bytes)*4 == rows*cols {
		if layer.WeightStore.SetMicrosoftBitNetPackedMatrixBytes(offset, rows, cols, t.Bytes) {
			return
		}
	}

	// Same bit layout as above but stored as F32 file dtype (rows*cols bytes total).
	if offset != -1 && rows > 0 && cols > 0 && rows%4 == 0 && t.DType == "F32" && len(t.Bytes) == rows*cols {
		tmp := make([]float32, (rows/4)*cols)
		if err := DecodeTensorBytesInto(tmp, t.Bytes, t.DType); err == nil {
			if layer.WeightStore.SetMicrosoftBitNetPackedMatrix(offset, rows, cols, tmp) {
				return
			}
		}
	}

	if offset != -1 && rows > 0 && cols > 0 {
		ne := rows * cols
		want, err := safetensorPayloadByteLength(t.DType, ne)
		if err == nil && len(t.Bytes) == want && offset+ne <= len(layer.WeightStore.Master) {
			_ = DecodeTensorBytesInto(layer.WeightStore.Master[offset:offset+ne], t.Bytes, t.DType)
			return
		}
	}

	if layer.Type == LayerRMSNorm && role == "w" && offset == 0 {
		ne := StoredTensorNumElements(t)
		if ne > 0 && ne <= len(layer.WeightStore.Master) {
			_ = DecodeTensorBytesInto(layer.WeightStore.Master[:ne], t.Bytes, t.DType)
		}
	}
}
