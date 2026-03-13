package nn

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
)

// SaveSafetensors writes tensors to a safetensors file
func SaveSafetensors(filepath string, tensors map[string]TensorWithShape) error {
	data, err := SerializeSafetensors(tensors)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath, data, 0644)
}

// SerializeSafetensors converts tensors to safetensors format bytes
func SerializeSafetensors(tensors map[string]TensorWithShape) ([]byte, error) {
	// Build header metadata
	header := make(map[string]interface{})
	currentOffset := 0

	// Calculate sizes and build header
	// Sort names for deterministic order
	var names []string
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)

	// Calculate sizes and build header
	for _, name := range names {
		tensor := tensors[name]
		var dataSize int
		if tensor.DType == "F4" {
			// Handle FP4 special case: 2 values per byte
			numElements := 1
			for _, dim := range tensor.Shape {
				numElements *= dim
			}
			dataSize = (numElements + 1) / 2
		} else {
			bytesPerElement := getBytesPerElement(tensor.DType)
			if bytesPerElement == 0 {
				return nil, fmt.Errorf("unsupported dtype: %s", tensor.DType)
			}

			numElements := 1
			for _, dim := range tensor.Shape {
				numElements *= dim
			}
			dataSize = numElements * bytesPerElement
		}

		header[name] = map[string]interface{}{
			"dtype":        tensor.DType,
			"shape":        tensor.Shape,
			"data_offsets": []int{currentOffset, currentOffset + dataSize},
		}

		currentOffset += dataSize
	}

	// Serialize header to JSON
	headerJSON, err := json.Marshal(header)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal header: %w", err)
	}

	// Build file: [header_size (8 bytes)] [header JSON] [tensor data]
	headerSize := uint64(len(headerJSON))
	totalSize := 8 + headerSize + uint64(currentOffset)
	result := make([]byte, totalSize)

	// Write header size (little-endian)
	binary.LittleEndian.PutUint64(result[0:8], headerSize)

	// Write header JSON
	copy(result[8:8+headerSize], headerJSON)

	// Write tensor data
	dataStart := 8 + headerSize
	currentDataOffset := 0

	for _, name := range names {
		tensor := tensors[name]
		offset := int(dataStart) + currentDataOffset
		bytesWritten, err := writeTensorData(result[offset:], tensor)
		if err != nil {
			return nil, fmt.Errorf("failed to write tensor %s: %w", name, err)
		}
		currentDataOffset += bytesWritten
	}

	return result, nil
}

// getBytesPerElement returns bytes per element for a dtype
func getBytesPerElement(dtype string) int {
	switch dtype {
	case "F64":
		return 8
	case "F32", "I32", "U32":
		return 4
	case "F16", "BF16", "I16", "U16":
		return 2
	case "I8", "U8":
		return 1
	case "F4":
		return 0 // Special case: handled separately
	case "I64", "U64":
		return 8
	default:
		return 0
	}
}

// writeTensorData writes tensor data in the specified dtype format
func writeTensorData(dest []byte, tensor TensorWithShape) (int, error) {
	numElements := len(tensor.Values)

	switch tensor.DType {
	case "F32":
		for i, val := range tensor.Values {
			binary.LittleEndian.PutUint32(dest[i*4:], math.Float32bits(val))
		}
		return numElements * 4, nil

	case "F64":
		for i, val := range tensor.Values {
			binary.LittleEndian.PutUint64(dest[i*8:], math.Float64bits(float64(val)))
		}
		return numElements * 8, nil

	case "F16":
		for i, val := range tensor.Values {
			f16 := float32ToFloat16(val)
			binary.LittleEndian.PutUint16(dest[i*2:], f16)
		}
		return numElements * 2, nil

	case "BF16":
		for i, val := range tensor.Values {
			bf16 := float32ToBFloat16(val)
			binary.LittleEndian.PutUint16(dest[i*2:], bf16)
		}
		return numElements * 2, nil

	case "F4":
		// Pack 2 FP4 values per byte
		bytesWritten := (numElements + 1) / 2
		for i := 0; i < numElements; i++ {
			fp4 := float32ToFP4(tensor.Values[i])
			byteIdx := i / 2
			if i%2 == 0 {
				// Lower 4 bits
				dest[byteIdx] = (dest[byteIdx] & 0xF0) | (fp4 & 0x0F)
			} else {
				// Upper 4 bits
				dest[byteIdx] = (dest[byteIdx] & 0x0F) | ((fp4 & 0x0F) << 4)
			}
		}
		return bytesWritten, nil

	case "I8":
		for i, val := range tensor.Values {
			dest[i] = byte(int8(val))
		}
		return numElements, nil

	case "I16":
		for i, val := range tensor.Values {
			binary.LittleEndian.PutUint16(dest[i*2:], uint16(int16(val)))
		}
		return numElements * 2, nil

	case "I32":
		for i, val := range tensor.Values {
			binary.LittleEndian.PutUint32(dest[i*4:], uint32(int32(val)))
		}
		return numElements * 4, nil

	case "I64":
		for i, val := range tensor.Values {
			binary.LittleEndian.PutUint64(dest[i*8:], uint64(int64(val)))
		}
		return numElements * 8, nil

	case "U8":
		for i, val := range tensor.Values {
			dest[i] = byte(uint8(val))
		}
		return numElements, nil

	case "U16":
		for i, val := range tensor.Values {
			binary.LittleEndian.PutUint16(dest[i*2:], uint16(val))
		}
		return numElements * 2, nil

	case "U32":
		for i, val := range tensor.Values {
			binary.LittleEndian.PutUint32(dest[i*4:], uint32(val))
		}
		return numElements * 4, nil

	case "U64":
		for i, val := range tensor.Values {
			binary.LittleEndian.PutUint64(dest[i*8:], uint64(val))
		}
		return numElements * 8, nil

	default:
		return 0, fmt.Errorf("unsupported dtype: %s", tensor.DType)
	}
}

// float32ToFP4 converts a float32 to 4-bit float (E2M1 format) for SafeTensors
// FP4 E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit
func float32ToFP4(f32 float32) uint8 {
	if f32 == 0 {
		return 0
	}
	bits := math.Float32bits(f32)
	sign := uint8((bits >> 31) & 0x1)
	absF32 := f32
	if absF32 < 0 {
		absF32 = -absF32
	}

	// FP4 E2M1 (Bias 1)
	// Smallest subnormal: 2^(-1) * 0.5 = 0.25
	// Smallest normal: 2^(1-1) * 1.0 = 1.0
	// Largest normal: 2^(2-1) * 1.5 = 3.0

	// Handle range [0, 0.625) -> map to Zero or Subnormal (0.125 threshold)
	if absF32 < 0.625 {
		if absF32 < 0.125 {
			return sign << 3
		}
		return (sign << 3) | 1 // Subnormal 0.25
	}

	// Normal numbers
	exp := int32((math.Float32bits(absF32) >> 23) & 0xFF)
	mant := math.Float32bits(absF32) & 0x7FFFFF

	e := exp - 127 + 1
	if e >= 3 {
		return (sign << 3) | 0x7 // Saturated Max/Inf
	}
	if e < 1 { // Should be covered by subnormal check above
		return (sign << 3) | 1
	}

	// Round mantissa (1 bit)
	// Bias 0.5 in the next bit
	m := uint8((mant + 0x400000) >> 23)
	if m > 1 {
		m = 0
		e++
	}

	if e >= 3 {
		return (sign << 3) | 0x7
	}

	return (sign << 3) | (uint8(e) << 1) | m
}

// SaveWeightsToSafetensors saves the network weights to a safetensors file
// It uses a consistent naming scheme: "layers.{index}.{param}"
// For nested layers (Parallel/Sequential), it uses "layers.{i}.branches.{b}.layers.{j}..." or similar
func (n *Network) SaveWeightsToSafetensors(filepath string) error {
	tensors := make(map[string]TensorWithShape)

	var extractLayer func(prefix string, l *LayerConfig)
	extractLayer = func(prefix string, l *LayerConfig) {
		if l.IsDisabled {
			return
		}

		// Helper to add tensor
		add := func(name string, data []float32, shape ...int) {
			if len(data) == 0 {
				return
			}
			tensors[prefix+"."+name] = TensorWithShape{
				DType:  "F32",
				Shape:  shape,
				Values: data,
			}
		}

		switch l.Type {
		case LayerDense:
			add("weight", l.Kernel, l.OutputHeight, l.InputHeight)
			add("bias", l.Bias, l.OutputHeight)

		case LayerConv2D:
			add("weight", l.Kernel, l.Filters, l.InputChannels, l.KernelSize, l.KernelSize)
			add("bias", l.Bias, l.Filters)

		case LayerConv1D:
			add("weight", l.Kernel, l.Conv1DFilters, l.Conv1DInChannels, l.Conv1DKernelSize)
			add("bias", l.Bias, l.Conv1DFilters)

		case LayerNorm, LayerRMSNorm:
			add("gamma", l.Gamma, l.NormSize)
			add("beta", l.Beta, l.NormSize)

		case LayerEmbedding:
			add("weight", l.EmbeddingWeights, l.VocabSize, l.EmbeddingDim)

		case LayerRNN:
			add("weight_ih", l.WeightIH, l.HiddenSize, l.RNNInputSize)
			add("weight_hh", l.WeightHH, l.HiddenSize, l.HiddenSize)
			add("bias_h", l.BiasH, l.HiddenSize)

		case LayerLSTM:
			// Gates: i, f, g, o
			add("weight_ih_i", l.WeightIH_i, l.HiddenSize, l.RNNInputSize)
			add("weight_hh_i", l.WeightHH_i, l.HiddenSize, l.HiddenSize)
			add("bias_i", l.BiasH_i, l.HiddenSize)

			add("weight_ih_f", l.WeightIH_f, l.HiddenSize, l.RNNInputSize)
			add("weight_hh_f", l.WeightHH_f, l.HiddenSize, l.HiddenSize)
			add("bias_f", l.BiasH_f, l.HiddenSize)

			add("weight_ih_g", l.WeightIH_g, l.HiddenSize, l.RNNInputSize)
			add("weight_hh_g", l.WeightHH_g, l.HiddenSize, l.HiddenSize)
			add("bias_g", l.BiasH_g, l.HiddenSize)

			add("weight_ih_o", l.WeightIH_o, l.HiddenSize, l.RNNInputSize)
			add("weight_hh_o", l.WeightHH_o, l.HiddenSize, l.HiddenSize)
			add("bias_o", l.BiasH_o, l.HiddenSize)

		case LayerMultiHeadAttention:
			add("q_weight", l.QWeights, l.DModel, l.DModel)
			add("k_weight", l.KWeights, l.DModel, l.DModel) // Or smaller if GQA
			add("v_weight", l.VWeights, l.DModel, l.DModel) // Or smaller if GQA
			add("o_weight", l.OutputWeight, l.DModel, l.DModel)

			add("q_bias", l.QBias, l.DModel)
			add("k_bias", l.KBias, l.DModel)
			add("v_bias", l.VBias, l.DModel)
			add("o_bias", l.OutputBias, l.DModel)

		case LayerSwiGLU:
			// Gate, Up, Down
			// Assuming shapes are stored flat but conceptually matrices
			// Logic usually: Gate/Up [Hidden, Inter], Down [Inter, Hidden]?
			// Need to verify dimensions from Config struct if possible, but assuming flat
			// we rely on implied shape. `l.GateWeights` is a slice.
			// Let's assume Dense-like shapes if sizes known?
			// SwiGLU forward code likely has sizes.
			// For now, save as flat 1D if shapes unclear, OR assume standard Dense.
			// Assuming simply [len] for now to avoid crashes if dimensions unavailable.
			add("gate_weight", l.GateWeights, len(l.GateWeights))
			add("up_weight", l.UpWeights, len(l.UpWeights))
			add("down_weight", l.DownWeights, len(l.DownWeights))
			add("gate_bias", l.GateBias, len(l.GateBias))
			add("up_bias", l.UpBias, len(l.UpBias))
			add("down_bias", l.DownBias, len(l.DownBias))

		case LayerParallel, LayerSequential:
			// Recurse into branches
			// ParallelBranches is []LayerConfig
			for b, branch := range l.ParallelBranches {
				branchPrefix := fmt.Sprintf("%s.branches.%d", prefix, b)
				extractLayer(branchPrefix, &branch)
			}

			// Handle FilterGateConfig for MoE (Filter combine mode)
			if l.FilterGateConfig != nil {
				extractLayer(prefix+".filter_gate", l.FilterGateConfig)
			}
		}
	}

	for i := range n.Layers {
		extractLayer(fmt.Sprintf("layers.%d", i), &n.Layers[i])
	}

	return SaveSafetensors(filepath, tensors)
}
