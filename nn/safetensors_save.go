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
	bits := math.Float32bits(f32)
	sign := uint8((bits >> 31) & 0x1)

	// Extract exponent and mantissa
	exp := int32((bits >> 23) & 0xFF)
	mant := bits & 0x7FFFFF

	// Handle special cases
	if exp == 0xFF || math.IsNaN(float64(f32)) || math.IsInf(float64(f32), 0) {
		// Inf or NaN -> max exponent
		return (sign << 3) | 0x6 | uint8((mant>>22)&0x1)
	}

	if exp == 0 || f32 == 0 {
		// Zero or subnormal
		return sign << 3
	}

	// Rebias: F32 bias=127, FP4 bias=1
	exp = exp - 127 + 1

	// Clamp to FP4 range
	var fp4Exp uint8
	if exp <= 0 {
		fp4Exp = 0 // Subnormal/zero
	} else if exp >= 3 {
		fp4Exp = 3 // Inf
	} else {
		fp4Exp = uint8(exp)
	}

	// Round mantissa to 1 bit
	fp4Mant := uint8((mant + 0x400000) >> 23)

	return (sign << 3) | (fp4Exp << 1) | fp4Mant
}
