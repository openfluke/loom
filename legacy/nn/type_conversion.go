package nn

import (
	"fmt"
	"math"
)

// =============================================================================
// Universal Type Conversion System
// =============================================================================
// This module provides comprehensive conversion between all numerical types
// supported by Loom: F32, F64, F16, BF16, F4, I8-I64, U8-U64

// NumericType represents all supported numerical data types
type NumericType string

const (
	TypeF32  NumericType = "F32"
	TypeF64  NumericType = "F64"
	TypeF16  NumericType = "F16"
	TypeBF16 NumericType = "BF16"
	TypeF4   NumericType = "F4"
	TypeI8   NumericType = "I8"
	TypeI16  NumericType = "I16"
	TypeI32  NumericType = "I32"
	TypeI64  NumericType = "I64"
	TypeU8   NumericType = "U8"
	TypeU16  NumericType = "U16"
	TypeU32  NumericType = "U32"
	TypeU64  NumericType = "U64"
)

// TypedValue holds a value with its type information
type TypedValue struct {
	Value interface{} // Can be float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64
	Type  NumericType
}

// ConvertValue converts a value from one type to another
// This is the universal conversion function
func ConvertValue(value interface{}, fromType, toType NumericType) (interface{}, error) {
	// Fast path: same type
	if fromType == toType {
		return value, nil
	}

	// Convert to intermediate float64 for universal conversion
	intermediate, err := toFloat64(value, fromType)
	if err != nil {
		return nil, err
	}

	// Convert from float64 to target type
	return fromFloat64(intermediate, toType)
}

// ConvertSlice converts an entire slice from one type to another
func ConvertSlice(values interface{}, fromType, toType NumericType) (interface{}, error) {
	// Fast path: same type
	if fromType == toType {
		return values, nil
	}

	switch fromType {
	case TypeF32:
		return convertFromF32Slice(values.([]float32), toType)
	case TypeF64:
		return convertFromF64Slice(values.([]float64), toType)
	case TypeI8:
		return convertFromI8Slice(values.([]int8), toType)
	case TypeI16:
		return convertFromI16Slice(values.([]int16), toType)
	case TypeI32:
		return convertFromI32Slice(values.([]int32), toType)
	case TypeI64:
		return convertFromI64Slice(values.([]int64), toType)
	case TypeU8:
		return convertFromU8Slice(values.([]uint8), toType)
	case TypeU16:
		return convertFromU16Slice(values.([]uint16), toType)
	case TypeU32:
		return convertFromU32Slice(values.([]uint32), toType)
	case TypeU64:
		return convertFromU64Slice(values.([]uint64), toType)
	default:
		return nil, fmt.Errorf("unsupported source type: %s", fromType)
	}
}

// =============================================================================
// To Float64 Converters (Universal Intermediate)
// =============================================================================

func toFloat64(value interface{}, fromType NumericType) (float64, error) {
	switch fromType {
	case TypeF32:
		return float64(value.(float32)), nil
	case TypeF64:
		return value.(float64), nil
	case TypeI8:
		return float64(value.(int8)), nil
	case TypeI16:
		return float64(value.(int16)), nil
	case TypeI32:
		return float64(value.(int32)), nil
	case TypeI64:
		return float64(value.(int64)), nil
	case TypeU8:
		return float64(value.(uint8)), nil
	case TypeU16:
		return float64(value.(uint16)), nil
	case TypeU32:
		return float64(value.(uint32)), nil
	case TypeU64:
		return float64(value.(uint64)), nil
	case TypeF16:
		// F16 stored as uint16
		return float64(float16ToFloat32(value.(uint16))), nil
	case TypeBF16:
		// BF16 stored as uint16
		return float64(bfloat16ToFloat32(value.(uint16))), nil
	case TypeF4:
		// F4 stored as uint8
		return float64(fp4ToFloat32(value.(uint8))), nil
	default:
		return 0, fmt.Errorf("unsupported type for conversion: %s", fromType)
	}
}

// =============================================================================
// From Float64 Converters
// =============================================================================

func fromFloat64(value float64, toType NumericType) (interface{}, error) {
	switch toType {
	case TypeF32:
		return float32(value), nil
	case TypeF64:
		return value, nil
	case TypeI8:
		// Clamp to int8 range
		if value > 127 {
			return int8(127), nil
		} else if value < -128 {
			return int8(-128), nil
		}
		return int8(value), nil
	case TypeI16:
		if value > 32767 {
			return int16(32767), nil
		} else if value < -32768 {
			return int16(-32768), nil
		}
		return int16(value), nil
	case TypeI32:
		if value > math.MaxInt32 {
			return int32(math.MaxInt32), nil
		} else if value < math.MinInt32 {
			return int32(math.MinInt32), nil
		}
		return int32(value), nil
	case TypeI64:
		return int64(value), nil
	case TypeU8:
		// Clamp to uint8 range
		if value < 0 {
			return uint8(0), nil
		} else if value > 255 {
			return uint8(255), nil
		}
		return uint8(value), nil
	case TypeU16:
		if value < 0 {
			return uint16(0), nil
		} else if value > 65535 {
			return uint16(65535), nil
		}
		return uint16(value), nil
	case TypeU32:
		if value < 0 {
			return uint32(0), nil
		} else if value > math.MaxUint32 {
			return uint32(math.MaxUint32), nil
		}
		return uint32(value), nil
	case TypeU64:
		if value < 0 {
			return uint64(0), nil
		}
		return uint64(value), nil
	case TypeF16:
		// Return as uint16 (raw bits)
		return float32ToFloat16(float32(value)), nil
	case TypeBF16:
		return float32ToBFloat16(float32(value)), nil
	case TypeF4:
		return float32ToFP4(float32(value)), nil
	default:
		return nil, fmt.Errorf("unsupported target type: %s", toType)
	}
}

// =============================================================================
// Slice Converters
// =============================================================================

func convertFromF32Slice(values []float32, toType NumericType) (interface{}, error) {
	switch toType {
	case TypeF32:
		// Same type - return copy
		result := make([]float32, len(values))
		copy(result, values)
		return result, nil
	case TypeF64:
		result := make([]float64, len(values))
		for i, v := range values {
			result[i] = float64(v)
		}
		return result, nil
	case TypeF16:
		result := make([]uint16, len(values))
		for i, v := range values {
			result[i] = float32ToFloat16(v)
		}
		return result, nil
	case TypeBF16:
		result := make([]uint16, len(values))
		for i, v := range values {
			result[i] = float32ToBFloat16(v)
		}
		return result, nil
	case TypeF4:
		result := make([]uint8, len(values))
		for i, v := range values {
			result[i] = float32ToFP4(v)
		}
		return result, nil

	case TypeI8:
		result := make([]int8, len(values))
		for i, v := range values {
			val, _ := fromFloat64(float64(v), TypeI8)
			result[i] = val.(int8)
		}
		return result, nil
	case TypeI16:
		result := make([]int16, len(values))
		for i, v := range values {
			val, _ := fromFloat64(float64(v), TypeI16)
			result[i] = val.(int16)
		}
		return result, nil
	case TypeI32:
		result := make([]int32, len(values))
		for i, v := range values {
			result[i] = int32(v)
		}
		return result, nil
	case TypeI64:
		result := make([]int64, len(values))
		for i, v := range values {
			result[i] = int64(v)
		}
		return result, nil
	case TypeU8:
		result := make([]uint8, len(values))
		for i, v := range values {
			val, _ := fromFloat64(float64(v), TypeU8)
			result[i] = val.(uint8)
		}
		return result, nil
	case TypeU16:
		result := make([]uint16, len(values))
		for i, v := range values {
			val, _ := fromFloat64(float64(v), TypeU16)
			result[i] = val.(uint16)
		}
		return result, nil
	case TypeU32:
		result := make([]uint32, len(values))
		for i, v := range values {
			val, _ := fromFloat64(float64(v), TypeU32)
			result[i] = val.(uint32)
		}
		return result, nil
	case TypeU64:
		result := make([]uint64, len(values))
		for i, v := range values {
			val, _ := fromFloat64(float64(v), TypeU64)
			result[i] = val.(uint64)
		}
		return result, nil
	default:
		return nil, fmt.Errorf("unsupported target type: %s", toType)
	}
}

func convertFromF64Slice(values []float64, toType NumericType) (interface{}, error) {
	switch toType {
	case TypeF32:
		result := make([]float32, len(values))
		for i, v := range values {
			result[i] = float32(v)
		}
		return result, nil
	case TypeI8, TypeI16, TypeI32, TypeI64, TypeU8, TypeU16, TypeU32, TypeU64:
		// Convert to F32 first then use existing converter
		f32 := make([]float32, len(values))
		for i, v := range values {
			f32[i] = float32(v)
		}
		return convertFromF32Slice(f32, toType)
	default:
		return nil, fmt.Errorf("unsupported target type: %s", toType)
	}
}

func convertFromI8Slice(values []int8, toType NumericType) (interface{}, error) {
	f32 := make([]float32, len(values))
	for i, v := range values {
		f32[i] = float32(v)
	}
	return convertFromF32Slice(f32, toType)
}

func convertFromI16Slice(values []int16, toType NumericType) (interface{}, error) {
	f32 := make([]float32, len(values))
	for i, v := range values {
		f32[i] = float32(v)
	}
	return convertFromF32Slice(f32, toType)
}

func convertFromI32Slice(values []int32, toType NumericType) (interface{}, error) {
	f32 := make([]float32, len(values))
	for i, v := range values {
		f32[i] = float32(v)
	}
	return convertFromF32Slice(f32, toType)
}

func convertFromI64Slice(values []int64, toType NumericType) (interface{}, error) {
	f32 := make([]float32, len(values))
	for i, v := range values {
		f32[i] = float32(v)
	}
	return convertFromF32Slice(f32, toType)
}

func convertFromU8Slice(values []uint8, toType NumericType) (interface{}, error) {
	f32 := make([]float32, len(values))
	for i, v := range values {
		f32[i] = float32(v)
	}
	return convertFromF32Slice(f32, toType)
}

func convertFromU16Slice(values []uint16, toType NumericType) (interface{}, error) {
	f32 := make([]float32, len(values))
	for i, v := range values {
		f32[i] = float32(v)
	}
	return convertFromF32Slice(f32, toType)
}

func convertFromU32Slice(values []uint32, toType NumericType) (interface{}, error) {
	f32 := make([]float32, len(values))
	for i, v := range values {
		f32[i] = float32(v)
	}
	return convertFromF32Slice(f32, toType)
}

func convertFromU64Slice(values []uint64, toType NumericType) (interface{}, error) {
	f32 := make([]float32, len(values))
	for i, v := range values {
		f32[i] = float32(v)
	}
	return convertFromF32Slice(f32, toType)
}

// =============================================================================
// Helper Functions
// =============================================================================

// GetTypeName returns the string name of a NumericType
func GetTypeName(t NumericType) string {
	return string(t)
}

// GetTypeSize returns the byte size of a numeric type
func GetTypeSize(t NumericType) int {
	switch t {
	case TypeF32, TypeI32, TypeU32:
		return 4
	case TypeF64, TypeI64, TypeU64:
		return 8
	case TypeF16, TypeBF16, TypeI16, TypeU16:
		return 2
	case TypeI8, TypeU8:
		return 1
	case TypeF4:
		return 0 // Special: 2 values per byte
	default:
		return 0
	}
}

// IsNumericTypeFloat returns true if the type is a floating point type
func IsNumericTypeFloat(t NumericType) bool {
	return t == TypeF32 || t == TypeF64 || t == TypeF16 || t == TypeBF16 || t == TypeF4
}

// IsNumericTypeSignedInt returns true if the type is a signed integer
func IsNumericTypeSignedInt(t NumericType) bool {
	return t == TypeI8 || t == TypeI16 || t == TypeI32 || t == TypeI64
}

// IsNumericTypeUnsigned returns true if the type is an unsigned integer
func IsNumericTypeUnsigned(t NumericType) bool {
	return t == TypeU8 || t == TypeU16 || t == TypeU32 || t == TypeU64
}

// GetTypeRange returns the min and max representable values for a type
func GetTypeRange(t NumericType) (min, max float64) {
	switch t {
	case TypeF32:
		return -math.MaxFloat32, math.MaxFloat32
	case TypeF64:
		return -math.MaxFloat64, math.MaxFloat64
	case TypeF16:
		return -65504, 65504 // Half precision range
	case TypeBF16:
		return -3.39e38, 3.39e38 // Same exponent range as F32
	case TypeF4:
		return -6, 6 // E2M1 format range
	case TypeI8:
		return -128, 127
	case TypeI16:
		return -32768, 32767
	case TypeI32:
		return math.MinInt32, math.MaxInt32
	case TypeI64:
		return math.MinInt64, math.MaxInt64
	case TypeU8:
		return 0, 255
	case TypeU16:
		return 0, 65535
	case TypeU32:
		return 0, math.MaxUint32
	case TypeU64:
		return 0, math.MaxUint64
	default:
		return 0, 0
	}
}
