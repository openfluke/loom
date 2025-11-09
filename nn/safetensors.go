package nn

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
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
		if dtype != "F32" && dtype != "F16" && dtype != "BF16" {
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
		if dtype != "F32" && dtype != "F16" && dtype != "BF16" {
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
