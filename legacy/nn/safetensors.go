package nn

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"unsafe"
)

// SafetensorsHeader contains metadata about tensors in the file
type SafetensorsHeader struct {
	Tensors map[string]TensorInfo `json:"-"`
}

// TensorInfo describes a tensor's properties
type TensorInfo struct {
	DType  string `json:"dtype"`
	Shape  []int  `json:"shape"`
	Offset []int  `json:"data_offsets"`
}

// LoadSafetensors reads a safetensors file and returns tensors by name
func LoadSafetensors(filepath string) (map[string][]float32, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	// Read header size (first 8 bytes, little-endian)
	var headerSize uint64
	if err := binary.Read(file, binary.LittleEndian, &headerSize); err != nil {
		return nil, fmt.Errorf("failed to read header size: %w", err)
	}

	// Read header JSON
	headerBytes := make([]byte, headerSize)
	if _, err := file.Read(headerBytes); err != nil {
		return nil, fmt.Errorf("failed to read header: %w", err)
	}

	// Parse header
	var rawHeader map[string]interface{}
	if err := json.Unmarshal(headerBytes, &rawHeader); err != nil {
		return nil, fmt.Errorf("failed to parse header: %w", err)
	}

	// Read all tensor data (rest of file after header)
	dataStart := 8 + headerSize
	fileInfo, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("failed to stat file: %w", err)
	}
	dataSize := fileInfo.Size() - int64(dataStart)

	allData := make([]byte, dataSize)
	if _, err := file.Read(allData); err != nil {
		return nil, fmt.Errorf("failed to read data: %w", err)
	}

	// Extract tensors
	tensors := make(map[string][]float32)
	for name, value := range rawHeader {
		if name == "__metadata__" {
			continue // Skip metadata
		}

		// Parse tensor info
		infoMap, ok := value.(map[string]interface{})
		if !ok {
			continue
		}

		dtype, _ := infoMap["dtype"].(string)
		// Check supported types handled below
		// if dtype != "F32" && dtype != "F16" && dtype != "BF16" { ... }

		shapeList, _ := infoMap["shape"].([]interface{})
		offsetList, _ := infoMap["data_offsets"].([]interface{})

		shape := make([]int, len(shapeList))
		for i, v := range shapeList {
			shape[i] = int(v.(float64))
		}

		startOffset := int(offsetList[0].(float64))
		// endOffset := int(offsetList[1].(float64)) // Not needed, calculated from shape

		// Calculate number of elements
		numElements := 1
		for _, dim := range shape {
			numElements *= dim
		}

		// Convert to float32
		tensorData := make([]float32, numElements)

		if dtype == "F32" {
			// Direct conversion from bytes to float32
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*4
				bits := binary.LittleEndian.Uint32(allData[offset : offset+4])
				tensorData[i] = float32frombits(bits)
			}
		} else if dtype == "F16" {
			// Convert from float16 to float32
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*2
				f16bits := binary.LittleEndian.Uint16(allData[offset : offset+2])
				tensorData[i] = float16ToFloat32(f16bits)
			}
		} else if dtype == "BF16" {
			// Convert from bfloat16 to float32
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*2
				bf16bits := binary.LittleEndian.Uint16(allData[offset : offset+2])
				tensorData[i] = bfloat16ToFloat32(bf16bits)
			}
		} else if dtype == "F64" {
			// Convert from float64 to float32
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*8
				bits := binary.LittleEndian.Uint64(allData[offset : offset+8])
				tensorData[i] = float32(math.Float64frombits(bits))
			}
		} else if dtype == "I64" || dtype == "U64" {
			// Convert from int64/uint64 to float32
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*8
				// We treat both as uint64 bits for reading, then cast
				bits := binary.LittleEndian.Uint64(allData[offset : offset+8])
				if dtype == "I64" {
					tensorData[i] = float32(int64(bits))
				} else {
					tensorData[i] = float32(bits)
				}
			}
		} else if dtype == "I32" || dtype == "U32" {
			// Convert from int32/uint32 to float32
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*4
				bits := binary.LittleEndian.Uint32(allData[offset : offset+4])
				if dtype == "I32" {
					tensorData[i] = float32(int32(bits))
				} else {
					tensorData[i] = float32(bits)
				}
			}
		} else if dtype == "I16" || dtype == "U16" {
			// Convert from int16/uint16 to float32
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*2
				bits := binary.LittleEndian.Uint16(allData[offset : offset+2])
				if dtype == "I16" {
					tensorData[i] = float32(int16(bits))
				} else {
					tensorData[i] = float32(bits)
				}
			}
		} else if dtype == "I8" || dtype == "U8" {
			// Convert from int8/uint8 to float32
			for i := 0; i < numElements; i++ {
				offset := startOffset + i
				b := allData[offset]
				if dtype == "I8" {
					tensorData[i] = float32(int8(b))
				} else {
					tensorData[i] = float32(b)
				}
			}
		} else if dtype == "F4" {
			// 4-bit float (E2M1) packed 2 per byte
			for i := 0; i < numElements; i++ {
				offset := startOffset + i/2
				b := allData[offset]
				var nibble uint8
				if i%2 == 0 {
					nibble = b & 0x0F // Lower 4 bits
				} else {
					nibble = (b >> 4) & 0x0F // Upper 4 bits
				}
				// Decode E2M1
				tensorData[i] = fp4ToFloat32(nibble)
			}
		}

		tensors[name] = tensorData
	}

	return tensors, nil
}

// LoadSafetensorsFromBytes reads safetensors data from a byte slice and returns tensors by name
func LoadSafetensorsFromBytes(data []byte) (map[string][]float32, error) {
	if len(data) < 8 {
		return nil, fmt.Errorf("data too short: need at least 8 bytes for header size")
	}

	// Read header size (first 8 bytes, little-endian)
	headerSize := binary.LittleEndian.Uint64(data[0:8])

	if len(data) < int(8+headerSize) {
		return nil, fmt.Errorf("data too short: header size %d but only %d bytes available", headerSize, len(data)-8)
	}

	// Read header JSON
	headerBytes := data[8 : 8+headerSize]

	// Parse header
	var rawHeader map[string]interface{}
	if err := json.Unmarshal(headerBytes, &rawHeader); err != nil {
		return nil, fmt.Errorf("failed to parse header: %w", err)
	}

	// Tensor data starts after header
	allData := data[8+headerSize:]

	// Extract tensors
	tensors := make(map[string][]float32)
	for name, value := range rawHeader {
		if name == "__metadata__" {
			continue // Skip metadata
		}

		// Parse tensor info
		infoMap, ok := value.(map[string]interface{})
		if !ok {
			continue
		}

		dtype, _ := infoMap["dtype"].(string)
		// Check supported types handled below
		// if dtype != "F32" && dtype != "F16" && dtype != "BF16" { ... }

		shapeList, _ := infoMap["shape"].([]interface{})
		offsetList, _ := infoMap["data_offsets"].([]interface{})

		shape := make([]int, len(shapeList))
		for i, v := range shapeList {
			shape[i] = int(v.(float64))
		}

		startOffset := int(offsetList[0].(float64))

		// Calculate number of elements
		numElements := 1
		for _, dim := range shape {
			numElements *= dim
		}

		// Convert to float32
		tensorData := make([]float32, numElements)

		if dtype == "F32" {
			// Direct conversion from bytes to float32
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*4
				if offset+4 > len(allData) {
					return nil, fmt.Errorf("tensor %s: data out of bounds", name)
				}
				bits := binary.LittleEndian.Uint32(allData[offset : offset+4])
				tensorData[i] = float32frombits(bits)
			}
		} else if dtype == "F16" {
			// Convert from float16 to float32
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*2
				if offset+2 > len(allData) {
					return nil, fmt.Errorf("tensor %s: data out of bounds", name)
				}
				f16bits := binary.LittleEndian.Uint16(allData[offset : offset+2])
				tensorData[i] = float16ToFloat32(f16bits)
			}
		} else if dtype == "BF16" {
			// Convert from bfloat16 to float32
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*2
				if offset+2 > len(allData) {
					return nil, fmt.Errorf("tensor %s: data out of bounds", name)
				}
				bf16bits := binary.LittleEndian.Uint16(allData[offset : offset+2])
				tensorData[i] = bfloat16ToFloat32(bf16bits)
			}
		} else if dtype == "F64" {
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*8
				if offset+8 > len(allData) {
					return nil, fmt.Errorf("tensor %s: data out of bounds", name)
				}
				bits := binary.LittleEndian.Uint64(allData[offset : offset+8])
				tensorData[i] = float32(math.Float64frombits(bits))
			}
		} else if dtype == "I64" || dtype == "U64" {
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*8
				if offset+8 > len(allData) {
					return nil, fmt.Errorf("tensor %s: data out of bounds", name)
				}
				bits := binary.LittleEndian.Uint64(allData[offset : offset+8])
				if dtype == "I64" {
					tensorData[i] = float32(int64(bits))
				} else {
					tensorData[i] = float32(bits)
				}
			}
		} else if dtype == "I32" || dtype == "U32" {
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*4
				if offset+4 > len(allData) {
					return nil, fmt.Errorf("tensor %s: data out of bounds", name)
				}
				bits := binary.LittleEndian.Uint32(allData[offset : offset+4])
				if dtype == "I32" {
					tensorData[i] = float32(int32(bits))
				} else {
					tensorData[i] = float32(bits)
				}
			}
		} else if dtype == "I16" || dtype == "U16" {
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*2
				if offset+2 > len(allData) {
					return nil, fmt.Errorf("tensor %s: data out of bounds", name)
				}
				bits := binary.LittleEndian.Uint16(allData[offset : offset+2])
				if dtype == "I16" {
					tensorData[i] = float32(int16(bits))
				} else {
					tensorData[i] = float32(bits)
				}
			}
		} else if dtype == "I8" || dtype == "U8" {
			for i := 0; i < numElements; i++ {
				offset := startOffset + i
				if offset >= len(allData) {
					return nil, fmt.Errorf("tensor %s: data out of bounds", name)
				}
				b := allData[offset]
				if dtype == "I8" {
					tensorData[i] = float32(int8(b))
				} else {
					tensorData[i] = float32(b)
				}
			}
		} else if dtype == "F4" {
			for i := 0; i < numElements; i++ {
				offset := startOffset + i/2
				if offset >= len(allData) {
					return nil, fmt.Errorf("tensor %s: data out of bounds", name)
				}
				b := allData[offset]
				var nibble uint8
				if i%2 == 0 {
					nibble = b & 0x0F
				} else {
					nibble = (b >> 4) & 0x0F
				}
				tensorData[i] = fp4ToFloat32(nibble)
			}
		}

		tensors[name] = tensorData
	}

	return tensors, nil
}

// float32frombits converts uint32 bits to float32
func float32frombits(bits uint32) float32 {
	return *(*float32)(unsafe.Pointer(&bits))
}

// float16ToFloat32 converts a float16 (half precision) to float32
func float16ToFloat32(f16 uint16) float32 {
	sign := uint32((f16 >> 15) & 0x1)
	exponent := uint32((f16 >> 10) & 0x1F)
	mantissa := uint32(f16 & 0x3FF)

	var f32bits uint32
	if exponent == 0 {
		if mantissa == 0 {
			// Zero
			f32bits = sign << 31
		} else {
			// Subnormal
			exponent = 1
			for (mantissa & 0x400) == 0 {
				mantissa <<= 1
				exponent--
			}
			mantissa &= 0x3FF
			f32bits = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13)
		}
	} else if exponent == 0x1F {
		// Inf or NaN
		f32bits = (sign << 31) | (0xFF << 23) | (mantissa << 13)
	} else {
		// Normal
		f32bits = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13)
	}

	return float32frombits(f32bits)
}

// bfloat16ToFloat32 converts a bfloat16 to float32
func bfloat16ToFloat32(bf16 uint16) float32 {
	// bfloat16 is just the top 16 bits of float32
	f32bits := uint32(bf16) << 16
	return float32frombits(f32bits)
}

// fp4ToFloat32 converts a 4-bit float (FP4 E2M1) to float32
// FP4 format: 1 sign bit, 2 exponent bits, 1 mantissa bit
func fp4ToFloat32(fp4 uint8) float32 {
	sign := uint32((fp4 >> 3) & 0x1)
	exponent := uint32((fp4 >> 1) & 0x3)
	mantissa := uint32(fp4 & 0x1)

	// Handle special cases
	if exponent == 0 {
		if mantissa == 0 {
			// Zero
			return float32frombits(sign << 31)
		}
		// Subnormal: value = (-1)^sign * 2^-1 * (mantissa / 2)
		// In float32: exp = 126, mantissa shifted
		f32bits := (sign << 31) | (126 << 23) | (mantissa << 22)
		return float32frombits(f32bits)
	} else if exponent == 3 {
		// Inf or NaN
		f32bits := (sign << 31) | (0xFF << 23) | (mantissa << 22)
		return float32frombits(f32bits)
	} else {
		// Normal: value = (-1)^sign * 2^(exp-1) * (1 + mantissa/2)
		// exp-1 maps: 1->0, 2->1
		// In float32: exp_float32 = (exp-1) + 127
		exp_f32 := (exponent - 1) + 127
		f32bits := (sign << 31) | (exp_f32 << 23) | (mantissa << 22)
		return float32frombits(f32bits)
	}
}

// TensorWithShape holds tensor data along with its shape
type TensorWithShape struct {
	Values []float32
	Shape  []int
	DType  string
}

// LoadSafetensorsWithShapes loads safetensors and returns both values and shapes
// This enables proper layer type detection based on tensor dimensions
func LoadSafetensorsWithShapes(data []byte) (map[string]TensorWithShape, error) {
	if len(data) < 8 {
		return nil, fmt.Errorf("data too short: need at least 8 bytes for header size")
	}

	headerSize := binary.LittleEndian.Uint64(data[0:8])

	if len(data) < int(8+headerSize) {
		return nil, fmt.Errorf("data too short: header size %d but only %d bytes available", headerSize, len(data)-8)
	}

	headerBytes := data[8 : 8+headerSize]

	var rawHeader map[string]interface{}
	if err := json.Unmarshal(headerBytes, &rawHeader); err != nil {
		return nil, fmt.Errorf("failed to parse header: %w", err)
	}

	allData := data[8+headerSize:]

	tensors := make(map[string]TensorWithShape)
	for name, value := range rawHeader {
		if name == "__metadata__" {
			continue
		}

		infoMap, ok := value.(map[string]interface{})
		if !ok {
			continue
		}

		dtype, _ := infoMap["dtype"].(string)

		// Check if dtype is supported
		supported := dtype == "F32" || dtype == "F64" || dtype == "F16" || dtype == "BF16" || dtype == "F4" ||
			dtype == "I8" || dtype == "I16" || dtype == "I32" || dtype == "I64" ||
			dtype == "U8" || dtype == "U16" || dtype == "U32" || dtype == "U64"

		if !supported {
			fmt.Printf("Warning: skipping tensor %s with unsupported dtype %s\n", name, dtype)
			continue
		}

		shapeList, _ := infoMap["shape"].([]interface{})
		offsetList, _ := infoMap["data_offsets"].([]interface{})

		shape := make([]int, len(shapeList))
		for i, v := range shapeList {
			shape[i] = int(v.(float64))
		}

		startOffset := int(offsetList[0].(float64))

		numElements := 1
		for _, dim := range shape {
			numElements *= dim
		}

		tensorData := make([]float32, numElements)

		switch dtype {
		case "F32":
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*4
				if offset+4 > len(allData) {
					return nil, fmt.Errorf("tensor %s: data out of bounds", name)
				}
				bits := binary.LittleEndian.Uint32(allData[offset : offset+4])
				tensorData[i] = float32frombits(bits)
			}
		case "F64":
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*8
				if offset+8 > len(allData) {
					return nil, fmt.Errorf("tensor %s: data out of bounds", name)
				}
				bits := binary.LittleEndian.Uint64(allData[offset : offset+8])
				tensorData[i] = float32(math.Float64frombits(bits))
			}
		case "F16":
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*2
				if offset+2 > len(allData) {
					return nil, fmt.Errorf("tensor %s: data out of bounds", name)
				}
				f16bits := binary.LittleEndian.Uint16(allData[offset : offset+2])
				tensorData[i] = float16ToFloat32(f16bits)
			}
		case "BF16":
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*2
				if offset+2 > len(allData) {
					return nil, fmt.Errorf("tensor %s: data out of bounds", name)
				}
				bf16bits := binary.LittleEndian.Uint16(allData[offset : offset+2])
				tensorData[i] = bfloat16ToFloat32(bf16bits)
			}
		case "F4":
			// FP4: 2 values per byte (4 bits each)
			for i := 0; i < numElements; i++ {
				byteOffset := startOffset + i/2
				if byteOffset >= len(allData) {
					return nil, fmt.Errorf("tensor %s: data out of bounds", name)
				}
				b := allData[byteOffset]
				var fp4bits uint8
				if i%2 == 0 {
					fp4bits = b & 0x0F // Lower 4 bits
				} else {
					fp4bits = (b >> 4) & 0x0F // Upper 4 bits
				}
				tensorData[i] = fp4ToFloat32(fp4bits)
			}
		case "I8":
			for i := 0; i < numElements; i++ {
				offset := startOffset + i
				if offset >= len(allData) {
					return nil, fmt.Errorf("tensor %s: data out of bounds", name)
				}
				tensorData[i] = float32(int8(allData[offset]))
			}
		case "I16":
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*2
				if offset+2 > len(allData) {
					return nil, fmt.Errorf("tensor %s: data out of bounds", name)
				}
				val := int16(binary.LittleEndian.Uint16(allData[offset : offset+2]))
				tensorData[i] = float32(val)
			}
		case "I32":
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*4
				if offset+4 > len(allData) {
					return nil, fmt.Errorf("tensor %s: data out of bounds", name)
				}
				val := int32(binary.LittleEndian.Uint32(allData[offset : offset+4]))
				tensorData[i] = float32(val)
			}
		case "I64":
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*8
				if offset+8 > len(allData) {
					return nil, fmt.Errorf("tensor %s: data out of bounds", name)
				}
				val := int64(binary.LittleEndian.Uint64(allData[offset : offset+8]))
				tensorData[i] = float32(val)
			}
		case "U8":
			for i := 0; i < numElements; i++ {
				offset := startOffset + i
				if offset >= len(allData) {
					return nil, fmt.Errorf("tensor %s: data out of bounds", name)
				}
				tensorData[i] = float32(allData[offset])
			}
		case "U16":
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*2
				if offset+2 > len(allData) {
					return nil, fmt.Errorf("tensor %s: data out of bounds", name)
				}
				val := binary.LittleEndian.Uint16(allData[offset : offset+2])
				tensorData[i] = float32(val)
			}
		case "U32":
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*4
				if offset+4 > len(allData) {
					return nil, fmt.Errorf("tensor %s: data out of bounds", name)
				}
				val := binary.LittleEndian.Uint32(allData[offset : offset+4])
				tensorData[i] = float32(val)
			}
		case "U64":
			for i := 0; i < numElements; i++ {
				offset := startOffset + i*8
				if offset+8 > len(allData) {
					return nil, fmt.Errorf("tensor %s: data out of bounds", name)
				}
				val := binary.LittleEndian.Uint64(allData[offset : offset+8])
				tensorData[i] = float32(val)
			}
		}

		tensors[name] = TensorWithShape{
			Values: tensorData,
			Shape:  shape,
			DType:  dtype,
		}
	}

	return tensors, nil
}

// InferLayerType determines layer type from tensor shape
// Returns: "conv" (4D), "dense" (2D), "norm" (1D), "embedding" (2D large vocab), "unknown"
func InferLayerType(shape []int) string {
	switch len(shape) {
	case 1:
		return "norm" // 1D = bias or norm weight
	case 2:
		// 2D could be dense/linear or embedding
		// Embeddings typically have vocab_size >> hidden_size
		if shape[0] > shape[1]*4 {
			return "embedding" // [vocab_size, hidden_size]
		}
		return "dense" // [out_features, in_features]
	case 3:
		return "conv1d" // [out, in, kernel]
	case 4:
		return "conv2d" // [out, in, kernel_h, kernel_w]
	default:
		return "unknown"
	}
}

// LoadWeightsFromSafetensors loads weights from a safetensors file into the network
// It expects keys matching "layers.{index}.{param}" or recursive names
func (n *Network) LoadWeightsFromSafetensors(filepath string) error {
	tensors, err := LoadSafetensors(filepath)
	if err != nil {
		return err
	}

	var loadLayer func(prefix string, l *LayerConfig)
	loadLayer = func(prefix string, l *LayerConfig) {
		if w, ok := tensors[prefix+".weight"]; ok {
			switch l.Type {
			case LayerDense, LayerConv2D, LayerConv1D:
				l.Kernel = w
			case LayerEmbedding:
				l.EmbeddingWeights = w
			}
		}

		if b, ok := tensors[prefix+".bias"]; ok {
			l.Bias = b
		}

		if g, ok := tensors[prefix+".gamma"]; ok {
			l.Gamma = g
		}

		if b, ok := tensors[prefix+".beta"]; ok {
			l.Beta = b
		}

		// RNN/LSTM specific
		if w, ok := tensors[prefix+".weight_ih"]; ok {
			l.WeightIH = w
		}
		if w, ok := tensors[prefix+".weight_hh"]; ok {
			l.WeightHH = w
		}
		if b, ok := tensors[prefix+".bias_h"]; ok {
			l.BiasH = b
		}

		// LSTM Gates
		if w, ok := tensors[prefix+".weight_ih_i"]; ok {
			l.WeightIH_i = w
		}
		if w, ok := tensors[prefix+".weight_hh_i"]; ok {
			l.WeightHH_i = w
		}
		if b, ok := tensors[prefix+".bias_i"]; ok {
			l.BiasH_i = b
		}

		if w, ok := tensors[prefix+".weight_ih_f"]; ok {
			l.WeightIH_f = w
		}
		if w, ok := tensors[prefix+".weight_hh_f"]; ok {
			l.WeightHH_f = w
		}
		if b, ok := tensors[prefix+".bias_f"]; ok {
			l.BiasH_f = b
		}

		if w, ok := tensors[prefix+".weight_ih_g"]; ok {
			l.WeightIH_g = w
		}
		if w, ok := tensors[prefix+".weight_hh_g"]; ok {
			l.WeightHH_g = w
		}
		if b, ok := tensors[prefix+".bias_g"]; ok {
			l.BiasH_g = b
		}

		if w, ok := tensors[prefix+".weight_ih_o"]; ok {
			l.WeightIH_o = w
		}
		if w, ok := tensors[prefix+".weight_hh_o"]; ok {
			l.WeightHH_o = w
		}
		if b, ok := tensors[prefix+".bias_o"]; ok {
			l.BiasH_o = b
		}

		// MHA
		if w, ok := tensors[prefix+".q_weight"]; ok {
			l.QWeights = w
		}
		if w, ok := tensors[prefix+".k_weight"]; ok {
			l.KWeights = w
		}
		if w, ok := tensors[prefix+".v_weight"]; ok {
			l.VWeights = w
		}
		if w, ok := tensors[prefix+".o_weight"]; ok {
			l.OutputWeight = w
		}

		if b, ok := tensors[prefix+".q_bias"]; ok {
			l.QBias = b
		}
		if b, ok := tensors[prefix+".k_bias"]; ok {
			l.KBias = b
		}
		if b, ok := tensors[prefix+".v_bias"]; ok {
			l.VBias = b
		}
		if b, ok := tensors[prefix+".o_bias"]; ok {
			l.OutputBias = b
		}

		// SwiGLU
		if w, ok := tensors[prefix+".gate_weight"]; ok {
			l.GateWeights = w
		}
		if w, ok := tensors[prefix+".up_weight"]; ok {
			l.UpWeights = w
		}
		if w, ok := tensors[prefix+".down_weight"]; ok {
			l.DownWeights = w
		}
		if b, ok := tensors[prefix+".gate_bias"]; ok {
			l.GateBias = b
		}
		if b, ok := tensors[prefix+".up_bias"]; ok {
			l.UpBias = b
		}
		if b, ok := tensors[prefix+".down_bias"]; ok {
			l.DownBias = b
		}

		// Recursive for Nested Layers
		if l.Type == LayerParallel || l.Type == LayerSequential {
			for b := range l.ParallelBranches {
				branchPrefix := fmt.Sprintf("%s.branches.%d", prefix, b)
				loadLayer(branchPrefix, &l.ParallelBranches[b])
			}

			// Handle FilterGateConfig
			if l.FilterGateConfig != nil {
				loadLayer(prefix+".filter_gate", l.FilterGateConfig)
			}
		}
	}

	for i := range n.Layers {
		loadLayer(fmt.Sprintf("layers.%d", i), &n.Layers[i])
	}
	return nil
}
