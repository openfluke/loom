package nn

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
)

// =============================================================================
// Multi-Precision Model Serialization
// =============================================================================
// This module extends the base serialization to support storing and loading
// models with different numerical types (float32, float64, int8, int16, int32).

// MultiPrecisionWeights stores weights with explicit type information
type MultiPrecisionWeights struct {
	DType  string                `json:"dtype"` // "float32", "float64", "bfloat16", "float16", "float8", "float4", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "int4"
	Scale  float64               `json:"scale"` // Quantization scale (for int types)
	Layers []MultiPrecisionLayer `json:"layers"`
}

// MultiPrecisionLayer stores weights for a single layer with type-aware storage
type MultiPrecisionLayer struct {
	DType string `json:"dtype,omitempty"` // Layer-specific dtype override

	// All weights stored as base64-encoded bytes
	// The bytes are interpreted according to DType
	Kernel       string `json:"kernel,omitempty"`
	Biases       string `json:"biases,omitempty"`
	QWeights     string `json:"q_weights,omitempty"`
	KWeights     string `json:"k_weights,omitempty"`
	VWeights     string `json:"v_weights,omitempty"`
	OutputWeight string `json:"output_weight,omitempty"`
	QBias        string `json:"q_bias,omitempty"`
	KBias        string `json:"k_bias,omitempty"`
	VBias        string `json:"v_bias,omitempty"`
	OutputBias   string `json:"output_bias,omitempty"`

	// RNN weights
	WeightIH string `json:"weight_ih,omitempty"`
	WeightHH string `json:"weight_hh,omitempty"`
	BiasH    string `json:"bias_h,omitempty"`

	// LSTM weights (4 gates: input, forget, cell, output)
	WeightII string `json:"weight_ii,omitempty"` // Input gate input weights
	WeightIF string `json:"weight_if,omitempty"` // Forget gate input weights
	WeightIG string `json:"weight_ig,omitempty"` // Cell gate input weights
	WeightIO string `json:"weight_io,omitempty"` // Output gate input weights
	WeightHI string `json:"weight_hi,omitempty"` // Input gate hidden weights
	WeightHF string `json:"weight_hf,omitempty"` // Forget gate hidden weights
	WeightHG string `json:"weight_hg,omitempty"` // Cell gate hidden weights
	WeightHO string `json:"weight_ho,omitempty"` // Output gate hidden weights
	BiasI    string `json:"bias_i,omitempty"`    // Input gate bias
	BiasF    string `json:"bias_f,omitempty"`    // Forget gate bias
	BiasG    string `json:"bias_g,omitempty"`    // Cell gate bias
	BiasO    string `json:"bias_o,omitempty"`    // Output gate bias

	// Normalization weights
	Gamma string `json:"gamma,omitempty"`
	Beta  string `json:"beta,omitempty"`

	// SwiGLU weights
	GateWeights string `json:"gate_weights,omitempty"`
	UpWeights   string `json:"up_weights,omitempty"`
	DownWeights string `json:"down_weights,omitempty"`
	GateBias    string `json:"gate_bias,omitempty"`
	UpBias      string `json:"up_bias,omitempty"`
	DownBias    string `json:"down_bias,omitempty"`

	// Embedding weights
	EmbeddingWeights string `json:"embedding_weights,omitempty"`

	// Parallel layer branches (recursive)
	BranchWeights []MultiPrecisionLayer `json:"branch_weights,omitempty"`
}

// =============================================================================
// Type-Aware Encoding Functions
// =============================================================================

// encodeFloat32Slice encodes float32 slice to base64 bytes
func encodeFloat32Slice(data []float32) string {
	if len(data) == 0 {
		return ""
	}
	bytes := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(bytes[i*4:], math.Float32bits(v))
	}
	return base64.StdEncoding.EncodeToString(bytes)
}

// decodeFloat32Slice decodes base64 bytes to float32 slice
func decodeFloat32Slice(encoded string) ([]float32, error) {
	if encoded == "" {
		return nil, nil
	}
	bytes, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, err
	}
	data := make([]float32, len(bytes)/4)
	for i := range data {
		data[i] = math.Float32frombits(binary.LittleEndian.Uint32(bytes[i*4:]))
	}
	return data, nil
}

// encodeFloat64Slice encodes float64 slice to base64 bytes
func encodeFloat64Slice(data []float64) string {
	if len(data) == 0 {
		return ""
	}
	bytes := make([]byte, len(data)*8)
	for i, v := range data {
		binary.LittleEndian.PutUint64(bytes[i*8:], math.Float64bits(v))
	}
	return base64.StdEncoding.EncodeToString(bytes)
}

// decodeFloat64Slice decodes base64 bytes to float64 slice
func decodeFloat64Slice(encoded string) ([]float64, error) {
	if encoded == "" {
		return nil, nil
	}
	bytes, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, err
	}
	data := make([]float64, len(bytes)/8)
	for i := range data {
		data[i] = math.Float64frombits(binary.LittleEndian.Uint64(bytes[i*8:]))
	}
	return data, nil
}

// encodeInt8Slice encodes int8 slice to base64 bytes
func encodeInt8Slice(data []int8) string {
	if len(data) == 0 {
		return ""
	}
	bytes := make([]byte, len(data))
	for i, v := range data {
		bytes[i] = byte(v)
	}
	return base64.StdEncoding.EncodeToString(bytes)
}

// decodeInt8Slice decodes base64 bytes to int8 slice
func decodeInt8Slice(encoded string) ([]int8, error) {
	if encoded == "" {
		return nil, nil
	}
	bytes, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, err
	}
	data := make([]int8, len(bytes))
	for i, b := range bytes {
		data[i] = int8(b)
	}
	return data, nil
}

// encodeInt16Slice encodes int16 slice to base64 bytes
func encodeInt16Slice(data []int16) string {
	if len(data) == 0 {
		return ""
	}
	bytes := make([]byte, len(data)*2)
	for i, v := range data {
		binary.LittleEndian.PutUint16(bytes[i*2:], uint16(v))
	}
	return base64.StdEncoding.EncodeToString(bytes)
}

// decodeInt16Slice decodes base64 bytes to int16 slice
func decodeInt16Slice(encoded string) ([]int16, error) {
	if encoded == "" {
		return nil, nil
	}
	bytes, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, err
	}
	data := make([]int16, len(bytes)/2)
	for i := range data {
		data[i] = int16(binary.LittleEndian.Uint16(bytes[i*2:]))
	}
	return data, nil
}

// encodeInt32Slice encodes int32 slice to base64 bytes
func encodeInt32Slice(data []int32) string {
	if len(data) == 0 {
		return ""
	}
	bytes := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(bytes[i*4:], uint32(v))
	}
	return base64.StdEncoding.EncodeToString(bytes)
}

// decodeInt32Slice decodes base64 bytes to int32 slice
func decodeInt32Slice(encoded string) ([]int32, error) {
	if encoded == "" {
		return nil, nil
	}
	bytes, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, err
	}
	data := make([]int32, len(bytes)/4)
	for i := range data {
		data[i] = int32(binary.LittleEndian.Uint32(bytes[i*4:]))
	}
	return data, nil
}

// =============================================================================
// Multi-Precision Save Functions
// =============================================================================

// encodeUint8Slice encodes uint8 slice to base64 bytes
func encodeUint8Slice(data []uint8) string {
	if len(data) == 0 {
		return ""
	}
	bytes := make([]byte, len(data))
	for i, v := range data {
		bytes[i] = byte(v)
	}
	return base64.StdEncoding.EncodeToString(bytes)
}

// decodeUint8Slice decodes base64 bytes to uint8 slice
func decodeUint8Slice(encoded string) ([]uint8, error) {
	if encoded == "" {
		return nil, nil
	}
	bytes, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, err
	}
	data := make([]uint8, len(bytes))
	for i, b := range bytes {
		data[i] = uint8(b)
	}
	return data, nil
}

// encodeUint16Slice encodes uint16 slice to base64 bytes
func encodeUint16Slice(data []uint16) string {
	if len(data) == 0 {
		return ""
	}
	bytes := make([]byte, len(data)*2)
	for i, v := range data {
		binary.LittleEndian.PutUint16(bytes[i*2:], v)
	}
	return base64.StdEncoding.EncodeToString(bytes)
}

// decodeUint16Slice decodes base64 bytes to uint16 slice
func decodeUint16Slice(encoded string) ([]uint16, error) {
	if encoded == "" {
		return nil, nil
	}
	bytes, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, err
	}
	data := make([]uint16, len(bytes)/2)
	for i := range data {
		data[i] = binary.LittleEndian.Uint16(bytes[i*2:])
	}
	return data, nil
}

// encodeUint32Slice encodes uint32 slice to base64 bytes
func encodeUint32Slice(data []uint32) string {
	if len(data) == 0 {
		return ""
	}
	bytes := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(bytes[i*4:], v)
	}
	return base64.StdEncoding.EncodeToString(bytes)
}

// decodeUint32Slice decodes base64 bytes to uint32 slice
func decodeUint32Slice(encoded string) ([]uint32, error) {
	if encoded == "" {
		return nil, nil
	}
	bytes, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, err
	}
	data := make([]uint32, len(bytes)/4)
	for i := range data {
		data[i] = binary.LittleEndian.Uint32(bytes[i*4:])
	}
	return data, nil
}

// encodeUint64Slice encodes uint64 slice to base64 bytes
func encodeUint64Slice(data []uint64) string {
	if len(data) == 0 {
		return ""
	}
	bytes := make([]byte, len(data)*8)
	for i, v := range data {
		binary.LittleEndian.PutUint64(bytes[i*8:], v)
	}
	return base64.StdEncoding.EncodeToString(bytes)
}

// decodeUint64Slice decodes base64 bytes to uint64 slice
func decodeUint64Slice(encoded string) ([]uint64, error) {
	if encoded == "" {
		return nil, nil
	}
	bytes, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, err
	}
	data := make([]uint64, len(bytes)/8)
	for i := range data {
		data[i] = binary.LittleEndian.Uint64(bytes[i*8:])
	}
	return data, nil
}

// encodeFloat16Slice encodes float16 slice (stored as uint16) to base64 bytes
func encodeFloat16Slice(data []uint16) string {
	return encodeUint16Slice(data)
}

// decodeFloat16Slice decodes base64 bytes to float16 slice (stored as uint16)
func decodeFloat16Slice(encoded string) ([]uint16, error) {
	return decodeUint16Slice(encoded)
}

// float32ToFloat16 converts float32 to IEEE 754 half precision stored as uint16
func float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := (bits >> 31) & 1
	exp := int((bits>>23)&0xFF) - 127
	mantissa := bits & 0x7FFFFF

	if exp > 15 {
		return uint16((sign << 15) | 0x7C00) // Infinity
	} else if exp < -14 {
		return uint16(sign << 15) // Zero/denorm
	}

	newExp := uint16(exp + 15)
	newMant := uint16(mantissa >> 13)
	return uint16((sign << 15) | (uint32(newExp) << 10) | uint32(newMant))
}

// mpFloat16ToFloat32 converts IEEE 754 half precision (uint16) to float32
// Note: Named with mp prefix to avoid conflict with safetensors.go
func mpFloat16ToFloat32(h uint16) float32 {
	sign := uint32((h >> 15) & 1)
	exp := uint32((h >> 10) & 0x1F)
	mant := uint32(h & 0x3FF)

	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign << 31) // Zero
		}
		// Denormalized
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3FF
	} else if exp == 31 {
		// Inf or NaN
		if mant == 0 {
			return math.Float32frombits((sign << 31) | 0x7F800000)
		}
		return math.Float32frombits((sign << 31) | 0x7FC00000) // NaN
	}

	exp = exp + (127 - 15)
	mant = mant << 13
	return math.Float32frombits((sign << 31) | (exp << 23) | mant)
}

// =============================================================================
// BFloat16 (Brain Floating Point) - Google's format for ML
// =============================================================================

// float32ToBFloat16 converts float32 to bfloat16 stored as uint16
// BFloat16: 1 sign bit, 8 exponent bits, 7 mantissa bits (same exp range as float32)
func float32ToBFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	// BFloat16 is just the upper 16 bits of float32 with rounding
	// Round to nearest even
	rounding := uint32(0x00007FFF) + ((bits >> 16) & 1)
	return uint16((bits + rounding) >> 16)
}

// mpBFloat16ToFloat32 converts bfloat16 (uint16) to float32
// Note: Named with mp prefix to avoid conflict with safetensors.go
func mpBFloat16ToFloat32(bf uint16) float32 {
	// BFloat16 is just the upper 16 bits of float32
	return math.Float32frombits(uint32(bf) << 16)
}

// encodeBFloat16Slice encodes a slice as bfloat16
func encodeBFloat16Slice(data []float32) string {
	if len(data) == 0 {
		return ""
	}
	bf16 := make([]uint16, len(data))
	for i, v := range data {
		bf16[i] = float32ToBFloat16(v)
	}
	return encodeUint16Slice(bf16)
}

// decodeBFloat16Slice decodes bfloat16 to float32 slice
func decodeBFloat16Slice(encoded string) ([]float32, error) {
	bf16, err := decodeUint16Slice(encoded)
	if err != nil {
		return nil, err
	}
	result := make([]float32, len(bf16))
	for i, v := range bf16 {
		result[i] = mpBFloat16ToFloat32(v)
	}
	return result, nil
}

// =============================================================================
// Float8 (E4M3 format) - 8-bit floating point for extreme quantization
// =============================================================================

// float32ToFloat8E4M3 converts float32 to 8-bit float (E4M3 format)
// E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits (range ~±240)
func float32ToFloat8E4M3(f float32) uint8 {
	if f == 0 {
		return 0
	}

	bits := math.Float32bits(f)
	sign := (bits >> 31) & 1
	exp := int((bits>>23)&0xFF) - 127
	mantissa := bits & 0x7FFFFF

	// Bias for E4M3 is 7
	newExp := exp + 7

	// Clamp to valid range
	if newExp >= 15 {
		// Max value (not infinity in E4M3)
		return uint8((sign << 7) | 0x7E) // ±240 max
	} else if newExp <= 0 {
		return uint8(sign << 7) // Zero
	}

	newMant := uint8(mantissa >> 20) // Top 3 bits
	return uint8((sign << 7) | uint32(newExp)<<3 | uint32(newMant))
}

// float8E4M3ToFloat32 converts 8-bit float (E4M3) to float32
func float8E4M3ToFloat32(f8 uint8) float32 {
	if f8 == 0 || f8 == 0x80 {
		return 0
	}

	sign := uint32((f8 >> 7) & 1)
	exp := int((f8 >> 3) & 0x0F)
	mant := uint32(f8 & 0x07)

	// Convert from E4M3 bias (7) to float32 bias (127)
	realExp := exp - 7 + 127

	// Expand mantissa from 3 bits to 23 bits
	fullMant := mant << 20

	return math.Float32frombits((sign << 31) | (uint32(realExp) << 23) | fullMant)
}

// encodeFloat8Slice encodes a slice as 8-bit floats (E4M3)
func encodeFloat8Slice(data []float32) string {
	if len(data) == 0 {
		return ""
	}
	bytes := make([]byte, len(data))
	for i, v := range data {
		bytes[i] = float32ToFloat8E4M3(v)
	}
	return base64.StdEncoding.EncodeToString(bytes)
}

// decodeFloat8Slice decodes 8-bit floats (E4M3) to float32
func decodeFloat8Slice(encoded string) ([]float32, error) {
	if encoded == "" {
		return nil, nil
	}
	bytes, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, err
	}
	result := make([]float32, len(bytes))
	for i, b := range bytes {
		result[i] = float8E4M3ToFloat32(b)
	}
	return result, nil
}

// =============================================================================
// Float4 (4-bit quantization) - Extreme compression
// =============================================================================

// float32ToFloat4 converts float32 to 4-bit representation
// Uses simple linear quantization: 4 bits = 16 levels centered at 0
// Range: [-1, 1] mapped to [-7, 7] (value 8 reserved for zero)
func float32ToFloat4(f float32, scale float64) uint8 {
	// Normalize to [-1, 1] range using scale, then quantize to [-7, 7]
	normalized := f / float32(scale)
	if normalized > 1 {
		normalized = 1
	} else if normalized < -1 {
		normalized = -1
	}

	// Map [-1, 1] to [0, 15]
	quantized := int((normalized + 1) * 7.5)
	if quantized > 15 {
		quantized = 15
	} else if quantized < 0 {
		quantized = 0
	}
	return uint8(quantized)
}

// float4ToFloat32 converts 4-bit value back to float32
func float4ToFloat32(q uint8, scale float64) float32 {
	// Map [0, 15] back to [-1, 1], then apply scale
	normalized := (float32(q) / 7.5) - 1
	return normalized * float32(scale)
}

// encodeFloat4Slice encodes a slice as 4-bit values (2 values per byte)
func encodeFloat4Slice(data []float32, scale float64) string {
	if len(data) == 0 {
		return ""
	}
	// Pack 2 values per byte
	numBytes := (len(data) + 1) / 2
	bytes := make([]byte, numBytes)
	for i := 0; i < len(data); i += 2 {
		high := float32ToFloat4(data[i], scale)
		low := uint8(0)
		if i+1 < len(data) {
			low = float32ToFloat4(data[i+1], scale)
		}
		bytes[i/2] = (high << 4) | low
	}
	return base64.StdEncoding.EncodeToString(bytes)
}

// decodeFloat4Slice decodes 4-bit values to float32 slice
func decodeFloat4Slice(encoded string, scale float64, numElements int) ([]float32, error) {
	if encoded == "" {
		return nil, nil
	}
	bytes, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, err
	}
	result := make([]float32, numElements)
	for i := 0; i < numElements; i++ {
		byteIdx := i / 2
		if byteIdx >= len(bytes) {
			break
		}
		if i%2 == 0 {
			result[i] = float4ToFloat32((bytes[byteIdx]>>4)&0x0F, scale)
		} else {
			result[i] = float4ToFloat32(bytes[byteIdx]&0x0F, scale)
		}
	}
	return result, nil
}

// =============================================================================
// Int4 (4-bit integer quantization) - Similar to QLoRA style
// =============================================================================

// encodeInt4Slice encodes a slice as 4-bit signed integers (2 values per byte)
// Values are quantized to [-8, 7] range
func encodeInt4Slice(data []int8) string {
	if len(data) == 0 {
		return ""
	}
	// Pack 2 values per byte
	numBytes := (len(data) + 1) / 2
	bytes := make([]byte, numBytes)
	for i := 0; i < len(data); i += 2 {
		// Clamp to [-8, 7] and offset to [0, 15]
		high := clampInt8ToInt4(data[i])
		low := int8(0)
		if i+1 < len(data) {
			low = clampInt8ToInt4(data[i+1])
		}
		bytes[i/2] = uint8((int8ToUint4(high) << 4) | int8ToUint4(low))
	}
	return base64.StdEncoding.EncodeToString(bytes)
}

// decodeInt4Slice decodes 4-bit integers to int8 slice
func decodeInt4Slice(encoded string, numElements int) ([]int8, error) {
	if encoded == "" {
		return nil, nil
	}
	bytes, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, err
	}
	result := make([]int8, numElements)
	for i := 0; i < numElements; i++ {
		byteIdx := i / 2
		if byteIdx >= len(bytes) {
			break
		}
		if i%2 == 0 {
			result[i] = uint4ToInt8((bytes[byteIdx] >> 4) & 0x0F)
		} else {
			result[i] = uint4ToInt8(bytes[byteIdx] & 0x0F)
		}
	}
	return result, nil
}

// clampInt8ToInt4 clamps an int8 to [-8, 7] range
func clampInt8ToInt4(v int8) int8 {
	if v > 7 {
		return 7
	} else if v < -8 {
		return -8
	}
	return v
}

// int8ToUint4 converts signed int4 (as int8) to unsigned 4-bit value [0, 15]
func int8ToUint4(v int8) uint8 {
	return uint8((v + 8) & 0x0F)
}

// uint4ToInt8 converts unsigned 4-bit value [0, 15] to signed int4 (as int8)
func uint4ToInt8(v uint8) int8 {
	return int8(v) - 8
}

// =============================================================================
// Int64 Encoding (for completeness)
// =============================================================================

// encodeInt64Slice encodes int64 slice to base64 bytes
func encodeInt64Slice(data []int64) string {
	if len(data) == 0 {
		return ""
	}
	bytes := make([]byte, len(data)*8)
	for i, v := range data {
		binary.LittleEndian.PutUint64(bytes[i*8:], uint64(v))
	}
	return base64.StdEncoding.EncodeToString(bytes)
}

// decodeInt64Slice decodes base64 bytes to int64 slice
func decodeInt64Slice(encoded string) ([]int64, error) {
	if encoded == "" {
		return nil, nil
	}
	bytes, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, err
	}
	data := make([]int64, len(bytes)/8)
	for i := range data {
		data[i] = int64(binary.LittleEndian.Uint64(bytes[i*8:]))
	}
	return data, nil
}

// =============================================================================
// Multi-Precision Save Functions
// =============================================================================

// SaveModelWithDType saves a model with weights converted to the specified dtype
// Supported dtypes: "float32", "float64", "int8", "int16", "int32", "uint8", "uint16", "uint32", "uint64", "float16"
func (n *Network) SaveModelWithDType(modelID string, dtype string) (string, error) {
	config := NetworkConfig{
		ID:            modelID,
		BatchSize:     n.BatchSize,
		GridRows:      n.GridRows,
		GridCols:      n.GridCols,
		LayersPerCell: n.LayersPerCell,
		Layers:        []LayerDefinition{},
	}

	// Determine quantization scale for integer and uint types
	// For float types, scale is not strictly used but passed anyway
	scale := 1.0
	switch dtype {
	case "int8":
		scale = 127.0 // Map [-1, 1] to [-127, 127]
	case "int16":
		scale = 32767.0 // Map [-1, 1] to [-32767, 32767]
	case "int32":
		scale = 2147483647.0 // Full range
	case "int64":
		scale = 127.0 // Use same as int8 for neural network weights (avoids precision issues)
	case "int4":
		scale = 1.0 // int4 already handles scaling internally via 8x factor
	case "float4":
		scale = 1.0 // float4 uses internal scaling
	case "float8":
		scale = 1.0 // float8 uses internal scaling
	case "uint8":
		scale = 255.0 // Map [0, 1] to [0, 255]
	case "uint16":
		scale = 65535.0
	case "uint32":
		scale = 4294967295.0
	}

	mpWeights := MultiPrecisionWeights{
		DType:  dtype,
		Scale:  scale,
		Layers: []MultiPrecisionLayer{},
	}

	// Serialize each layer
	totalLayers := n.GridRows * n.GridCols * n.LayersPerCell
	for i := 0; i < totalLayers; i++ {
		row := i / (n.GridCols * n.LayersPerCell)
		col := (i / n.LayersPerCell) % n.GridCols
		layer := i % n.LayersPerCell

		layerConfig := n.GetLayer(row, col, layer)

		// Create layer definition (same as standard serialization)
		layerDef := LayerDefinition{
			Type:       layerTypeToString(layerConfig.Type),
			Activation: activationToString(layerConfig.Activation),
		}

		// Create multi-precision layer weights
		mpLayer := serializeLayerMultiPrecision(layerConfig, dtype, scale)

		// Add layer-specific config based on type
		switch layerConfig.Type {
		case LayerDense:
			layerDef.InputHeight = layerConfig.InputHeight
			layerDef.OutputHeight = layerConfig.OutputHeight
		case LayerConv2D:
			layerDef.InputChannels = layerConfig.InputChannels
			layerDef.Filters = layerConfig.Filters
			layerDef.KernelSize = layerConfig.KernelSize
			layerDef.Stride = layerConfig.Stride
			layerDef.Padding = layerConfig.Padding
			layerDef.InputHeight = layerConfig.InputHeight
			layerDef.InputWidth = layerConfig.InputWidth
			layerDef.OutputHeight = layerConfig.OutputHeight
			layerDef.OutputWidth = layerConfig.OutputWidth
		case LayerConv1D:
			layerDef.InputChannels = layerConfig.InputChannels
			layerDef.Filters = layerConfig.Filters
			layerDef.KernelSize = layerConfig.KernelSize
			layerDef.Stride = layerConfig.Stride
			layerDef.Padding = layerConfig.Padding
			layerDef.InputHeight = layerConfig.InputHeight
			layerDef.InputWidth = layerConfig.InputWidth
			layerDef.OutputHeight = layerConfig.OutputHeight
			layerDef.OutputWidth = layerConfig.OutputWidth
		case LayerMultiHeadAttention:
			layerDef.DModel = layerConfig.DModel
			layerDef.NumHeads = layerConfig.NumHeads
			layerDef.SeqLength = layerConfig.SeqLength
		case LayerRNN:
			layerDef.InputSize = layerConfig.RNNInputSize
			layerDef.HiddenSize = layerConfig.HiddenSize
			layerDef.SeqLength = layerConfig.SeqLength
		case LayerLSTM:
			layerDef.InputSize = layerConfig.RNNInputSize
			layerDef.HiddenSize = layerConfig.HiddenSize
			layerDef.SeqLength = layerConfig.SeqLength
		case LayerNorm:
			layerDef.NormSize = layerConfig.NormSize
			layerDef.Epsilon = layerConfig.Epsilon
		case LayerRMSNorm:
			layerDef.NormSize = layerConfig.NormSize
			layerDef.Epsilon = layerConfig.Epsilon
		case LayerSwiGLU:
			layerDef.InputHeight = layerConfig.InputHeight
			layerDef.OutputHeight = layerConfig.OutputHeight
		case LayerEmbedding:
			layerDef.VocabSize = layerConfig.VocabSize
			layerDef.EmbeddingDim = layerConfig.EmbeddingDim
		case LayerSoftmax:
			layerDef.SoftmaxVariant = softmaxTypeToString(layerConfig.SoftmaxVariant)
			layerDef.SoftmaxRows = layerConfig.SoftmaxRows
			layerDef.SoftmaxCols = layerConfig.SoftmaxCols
			layerDef.Temperature = layerConfig.Temperature
			layerDef.GumbelNoise = layerConfig.GumbelNoise
			layerDef.Mask = layerConfig.Mask
			layerDef.HierarchyLevels = layerConfig.HierarchyLevels
			layerDef.AdaptiveClusters = layerConfig.AdaptiveClusters
			layerDef.MixtureWeights = layerConfig.MixtureWeights
			layerDef.EntmaxAlpha = layerConfig.EntmaxAlpha
		case LayerResidual:
			// Residual has no specific config
		case LayerParallel:
			layerDef.CombineMode = layerConfig.CombineMode
			layerDef.Branches = serializeBranches(layerConfig.ParallelBranches)
		case LayerSequential:
			layerDef.Branches = serializeBranches(layerConfig.ParallelBranches)
		}

		config.Layers = append(config.Layers, layerDef)
		mpWeights.Layers = append(mpWeights.Layers, mpLayer)
	}

	// Create the bundle
	weightsJSON, err := json.Marshal(mpWeights)
	if err != nil {
		return "", fmt.Errorf("failed to marshal multi-precision weights: %w", err)
	}

	encodedWeights := EncodedWeights{
		Format: "multiPrecisionB64",
		Data:   base64.StdEncoding.EncodeToString(weightsJSON),
	}

	savedModel := SavedModel{
		ID:      modelID,
		Config:  config,
		Weights: encodedWeights,
	}

	bundle := ModelBundle{
		Type:    "modelhost/bundle",
		Version: 2, // Version 2 indicates multi-precision support
		Models:  []SavedModel{savedModel},
	}

	return bundle.SaveToString()
}

// serializeLayerMultiPrecision converts layer weights to multi-precision format
func serializeLayerMultiPrecision(cfg *LayerConfig, dtype string, scale float64) MultiPrecisionLayer {
	mp := MultiPrecisionLayer{DType: dtype}

	switch cfg.Type {
	case LayerDense:
		mp.Kernel = EncodeSliceWithDType(cfg.Kernel, dtype, scale)
		mp.Biases = EncodeSliceWithDType(cfg.Bias, dtype, scale)
	case LayerConv2D:
		mp.Kernel = EncodeSliceWithDType(cfg.Kernel, dtype, scale)
		mp.Biases = EncodeSliceWithDType(cfg.Bias, dtype, scale)
	case LayerMultiHeadAttention:
		mp.QWeights = EncodeSliceWithDType(cfg.QWeights, dtype, scale)
		mp.KWeights = EncodeSliceWithDType(cfg.KWeights, dtype, scale)
		mp.VWeights = EncodeSliceWithDType(cfg.VWeights, dtype, scale)
		mp.OutputWeight = EncodeSliceWithDType(cfg.OutputWeight, dtype, scale)
		mp.QBias = EncodeSliceWithDType(cfg.QBias, dtype, scale)
		mp.KBias = EncodeSliceWithDType(cfg.KBias, dtype, scale)
		mp.VBias = EncodeSliceWithDType(cfg.VBias, dtype, scale)
		mp.OutputBias = EncodeSliceWithDType(cfg.OutputBias, dtype, scale)
	case LayerNorm:
		mp.Gamma = EncodeSliceWithDType(cfg.Gamma, dtype, scale)
		mp.Beta = EncodeSliceWithDType(cfg.Beta, dtype, scale)
	case LayerRMSNorm:
		mp.Gamma = EncodeSliceWithDType(cfg.Gamma, dtype, scale)
	case LayerSwiGLU:
		mp.GateWeights = EncodeSliceWithDType(cfg.GateWeights, dtype, scale)
		mp.UpWeights = EncodeSliceWithDType(cfg.UpWeights, dtype, scale)
		mp.DownWeights = EncodeSliceWithDType(cfg.DownWeights, dtype, scale)
		mp.GateBias = EncodeSliceWithDType(cfg.GateBias, dtype, scale)
		mp.UpBias = EncodeSliceWithDType(cfg.UpBias, dtype, scale)
		mp.DownBias = EncodeSliceWithDType(cfg.DownBias, dtype, scale)
	case LayerRNN:
		mp.WeightIH = EncodeSliceWithDType(cfg.WeightIH, dtype, scale)
		mp.WeightHH = EncodeSliceWithDType(cfg.WeightHH, dtype, scale)
		mp.BiasH = EncodeSliceWithDType(cfg.BiasH, dtype, scale)
	case LayerLSTM:
		mp.WeightII = EncodeSliceWithDType(cfg.WeightIH_i, dtype, scale)
		mp.WeightIF = EncodeSliceWithDType(cfg.WeightIH_f, dtype, scale)
		mp.WeightIG = EncodeSliceWithDType(cfg.WeightIH_g, dtype, scale)
		mp.WeightIO = EncodeSliceWithDType(cfg.WeightIH_o, dtype, scale)
		mp.WeightHI = EncodeSliceWithDType(cfg.WeightHH_i, dtype, scale)
		mp.WeightHF = EncodeSliceWithDType(cfg.WeightHH_f, dtype, scale)
		mp.WeightHG = EncodeSliceWithDType(cfg.WeightHH_g, dtype, scale)
		mp.WeightHO = EncodeSliceWithDType(cfg.WeightHH_o, dtype, scale)
		mp.BiasI = EncodeSliceWithDType(cfg.BiasH_i, dtype, scale)
		mp.BiasF = EncodeSliceWithDType(cfg.BiasH_f, dtype, scale)
		mp.BiasG = EncodeSliceWithDType(cfg.BiasH_g, dtype, scale)
		mp.BiasO = EncodeSliceWithDType(cfg.BiasH_o, dtype, scale)
	case LayerConv1D:
		mp.Kernel = EncodeSliceWithDType(cfg.Kernel, dtype, scale)
		mp.Biases = EncodeSliceWithDType(cfg.Bias, dtype, scale)
	case LayerEmbedding:
		mp.EmbeddingWeights = EncodeSliceWithDType(cfg.EmbeddingWeights, dtype, scale)
	case LayerResidual:
		// Residual layers typically have no trainable weights (skip connections)
		// But store any associated weights if present
		mp.Kernel = EncodeSliceWithDType(cfg.Kernel, dtype, scale)
	case LayerSoftmax:
		// Softmax has no trainable weights
	case LayerParallel:
		for _, branch := range cfg.ParallelBranches {
			mp.BranchWeights = append(mp.BranchWeights, serializeLayerMultiPrecision(&branch, dtype, scale))
		}
	case LayerSequential:
		for _, branch := range cfg.ParallelBranches {
			mp.BranchWeights = append(mp.BranchWeights, serializeLayerMultiPrecision(&branch, dtype, scale))
		}
	}

	return mp
}

// EncodeSliceWithDType encodes a float32 slice to the target dtype
func EncodeSliceWithDType(data []float32, dtype string, scale float64) string {
	if len(data) == 0 {
		return ""
	}

	switch dtype {
	case "float64":
		f64 := make([]float64, len(data))
		for i, v := range data {
			f64[i] = float64(v)
		}
		return encodeFloat64Slice(f64)
	case "float16":
		f16 := make([]uint16, len(data))
		for i, v := range data {
			f16[i] = float32ToFloat16(v)
		}
		return encodeFloat16Slice(f16)
	case "int8":
		i8 := make([]int8, len(data))
		for i, v := range data {
			scaled := float64(v) * scale
			if scaled > 127 {
				scaled = 127
			} else if scaled < -127 {
				scaled = -127
			}
			i8[i] = int8(scaled)
		}
		return encodeInt8Slice(i8)
	case "int16":
		i16 := make([]int16, len(data))
		for i, v := range data {
			scaled := float64(v) * scale
			if scaled > 32767 {
				scaled = 32767
			} else if scaled < -32767 {
				scaled = -32767
			}
			i16[i] = int16(scaled)
		}
		return encodeInt16Slice(i16)
	case "int32":
		i32 := make([]int32, len(data))
		for i, v := range data {
			i32[i] = int32(float64(v) * scale)
		}
		return encodeInt32Slice(i32)
	case "uint8":
		u8 := make([]uint8, len(data))
		for i, v := range data {
			// Map [-1, 1] to [0, 255] roughly: (v+1)*127.5
			// Or just scale if scale is used differently.
			// Let's use the offset approach for uints
			val := (float64(v) + 1.0) * 127.5
			if val < 0 {
				val = 0
			} else if val > 255 {
				val = 255
			}
			u8[i] = uint8(val)
		}
		return encodeUint8Slice(u8)
	case "uint16":
		u16 := make([]uint16, len(data))
		for i, v := range data {
			// Map [-1, 1] to [0, 65535]
			val := (float64(v) + 1.0) * 32767.5
			if val < 0 {
				val = 0
			} else if val > 65535 {
				val = 65535
			}
			u16[i] = uint16(val)
		}
		return encodeUint16Slice(u16)
	case "uint32":
		u32 := make([]uint32, len(data))
		for i, v := range data {
			// Map [-1, 1] to [0, 2^32-1]
			val := (float64(v) + 1.0) * 2147483647.5
			if val < 0 {
				val = 0
			} else if val > 4294967295 {
				val = 4294967295
			}
			u32[i] = uint32(val)
		}
		return encodeUint32Slice(u32)
	case "uint64":
		u64 := make([]uint64, len(data))
		for i, v := range data {
			// Map [-1, 1] to [0, 2^64-1]
			// Approximating large scale
			val := (float64(v) + 1.0) * 9.223372e18
			if val < 0 {
				val = 0
			}
			// Note: float64 precision issues at this scale, but good enough for demo
			u64[i] = uint64(val)
		}
		return encodeUint64Slice(u64)

	case "bfloat16":
		return encodeBFloat16Slice(data)

	case "float8":
		return encodeFloat8Slice(data)

	case "float4":
		return encodeFloat4Slice(data, scale)

	case "int4":
		// Convert float32 to int4 via int8 intermediate with scaling
		i8 := make([]int8, len(data))
		for i, v := range data {
			scaled := v * float32(scale/8.0) // Scale for int4 range [-8, 7]
			if scaled > 7 {
				scaled = 7
			} else if scaled < -8 {
				scaled = -8
			}
			i8[i] = int8(scaled)
		}
		return encodeInt4Slice(i8)

	case "int64":
		i64 := make([]int64, len(data))
		for i, v := range data {
			i64[i] = int64(float64(v) * scale)
		}
		return encodeInt64Slice(i64)

	default: // float32
		return encodeFloat32Slice(data)
	}
}

// =============================================================================
// Multi-Precision Load Functions
// =============================================================================

// LoadModelWithDType loads a model and converts weights to the specified target dtype
// The model can be loaded regardless of its stored dtype
func LoadModelWithDType(jsonString string, modelID string, targetDType string) (*Network, string, error) {
	bundle, err := LoadBundleFromString(jsonString)
	if err != nil {
		return nil, "", err
	}

	for _, savedModel := range bundle.Models {
		if savedModel.ID == modelID {
			// Check if this is a multi-precision model
			if savedModel.Weights.Format == "multiPrecisionB64" {
				return deserializeMultiPrecisionModel(savedModel, targetDType)
			}
			// Fall back to standard deserialization
			net, err := DeserializeModel(savedModel)
			return net, "float32", err
		}
	}

	return nil, "", fmt.Errorf("model %s not found in bundle", modelID)
}

// deserializeMultiPrecisionModel deserializes a multi-precision model
func deserializeMultiPrecisionModel(saved SavedModel, targetDType string) (*Network, string, error) {
	config := saved.Config

	// Create network
	network := NewNetwork(
		config.BatchSize,
		config.GridRows,
		config.GridCols,
		config.LayersPerCell,
	)
	network.BatchSize = config.BatchSize

	// Decode weights
	weightsJSON, err := base64.StdEncoding.DecodeString(saved.Weights.Data)
	if err != nil {
		return nil, "", fmt.Errorf("failed to decode weights: %w", err)
	}

	var mpWeights MultiPrecisionWeights
	if err := json.Unmarshal(weightsJSON, &mpWeights); err != nil {
		return nil, "", fmt.Errorf("failed to unmarshal multi-precision weights: %w", err)
	}

	storedDType := mpWeights.DType
	scale := mpWeights.Scale
	if scale == 0 {
		scale = 1.0
	}

	// Deserialize each layer
	for i := 0; i < len(config.Layers) && i < len(mpWeights.Layers); i++ {
		row := i / (config.GridCols * config.LayersPerCell)
		col := (i / config.LayersPerCell) % config.GridCols
		layer := i % config.LayersPerCell

		layerDef := config.Layers[i]
		mpLayer := mpWeights.Layers[i]

		layerConfig := deserializeLayerMultiPrecision(layerDef, mpLayer, storedDType, scale)
		network.SetLayer(row, col, layer, layerConfig)
	}

	return network, storedDType, nil
}

// deserializeLayerMultiPrecision converts multi-precision layer to LayerConfig
func deserializeLayerMultiPrecision(def LayerDefinition, mp MultiPrecisionLayer, dtype string, scale float64) LayerConfig {
	cfg := LayerConfig{
		Type:       stringToLayerType(def.Type),
		Activation: stringToActivation(def.Activation),
	}

	switch cfg.Type {
	case LayerDense:
		cfg.InputHeight = def.InputHeight
		cfg.OutputHeight = def.OutputHeight
		cfg.Kernel = DecodeSliceWithDType(mp.Kernel, dtype, scale)
		cfg.Bias = DecodeSliceWithDType(mp.Biases, dtype, scale)
	case LayerConv2D:
		cfg.InputChannels = def.InputChannels
		cfg.Filters = def.Filters
		cfg.KernelSize = def.KernelSize
		cfg.Stride = def.Stride
		cfg.Padding = def.Padding
		cfg.InputHeight = def.InputHeight
		cfg.InputWidth = def.InputWidth
		cfg.OutputHeight = def.OutputHeight
		cfg.OutputWidth = def.OutputWidth
		cfg.Kernel = DecodeSliceWithDType(mp.Kernel, dtype, scale)
		cfg.Bias = DecodeSliceWithDType(mp.Biases, dtype, scale)
	case LayerMultiHeadAttention:
		cfg.DModel = def.DModel
		cfg.NumHeads = def.NumHeads
		cfg.SeqLength = def.SeqLength
		cfg.QWeights = DecodeSliceWithDType(mp.QWeights, dtype, scale)
		cfg.KWeights = DecodeSliceWithDType(mp.KWeights, dtype, scale)
		cfg.VWeights = DecodeSliceWithDType(mp.VWeights, dtype, scale)
		cfg.OutputWeight = DecodeSliceWithDType(mp.OutputWeight, dtype, scale)
		cfg.QBias = DecodeSliceWithDType(mp.QBias, dtype, scale)
		cfg.KBias = DecodeSliceWithDType(mp.KBias, dtype, scale)
		cfg.VBias = DecodeSliceWithDType(mp.VBias, dtype, scale)
		cfg.OutputBias = DecodeSliceWithDType(mp.OutputBias, dtype, scale)
	case LayerNorm:
		cfg.NormSize = def.NormSize
		cfg.Epsilon = def.Epsilon
		cfg.Gamma = DecodeSliceWithDType(mp.Gamma, dtype, scale)
		cfg.Beta = DecodeSliceWithDType(mp.Beta, dtype, scale)
	case LayerRMSNorm:
		cfg.NormSize = def.NormSize
		cfg.Epsilon = def.Epsilon
		cfg.Gamma = DecodeSliceWithDType(mp.Gamma, dtype, scale)
	case LayerSwiGLU:
		cfg.InputHeight = def.InputHeight
		cfg.OutputHeight = def.OutputHeight
		cfg.GateWeights = DecodeSliceWithDType(mp.GateWeights, dtype, scale)
		cfg.UpWeights = DecodeSliceWithDType(mp.UpWeights, dtype, scale)
		cfg.DownWeights = DecodeSliceWithDType(mp.DownWeights, dtype, scale)
		cfg.GateBias = DecodeSliceWithDType(mp.GateBias, dtype, scale)
		cfg.UpBias = DecodeSliceWithDType(mp.UpBias, dtype, scale)
		cfg.DownBias = DecodeSliceWithDType(mp.DownBias, dtype, scale)
	case LayerRNN:
		cfg.RNNInputSize = def.InputSize
		cfg.HiddenSize = def.HiddenSize
		cfg.SeqLength = def.SeqLength
		cfg.WeightIH = DecodeSliceWithDType(mp.WeightIH, dtype, scale)
		cfg.WeightHH = DecodeSliceWithDType(mp.WeightHH, dtype, scale)
		cfg.BiasH = DecodeSliceWithDType(mp.BiasH, dtype, scale)
	case LayerLSTM:
		cfg.RNNInputSize = def.InputSize
		cfg.HiddenSize = def.HiddenSize
		cfg.SeqLength = def.SeqLength
		cfg.WeightIH_i = DecodeSliceWithDType(mp.WeightII, dtype, scale)
		cfg.WeightIH_f = DecodeSliceWithDType(mp.WeightIF, dtype, scale)
		cfg.WeightIH_g = DecodeSliceWithDType(mp.WeightIG, dtype, scale)
		cfg.WeightIH_o = DecodeSliceWithDType(mp.WeightIO, dtype, scale)
		cfg.WeightHH_i = DecodeSliceWithDType(mp.WeightHI, dtype, scale)
		cfg.WeightHH_f = DecodeSliceWithDType(mp.WeightHF, dtype, scale)
		cfg.WeightHH_g = DecodeSliceWithDType(mp.WeightHG, dtype, scale)
		cfg.WeightHH_o = DecodeSliceWithDType(mp.WeightHO, dtype, scale)
		cfg.BiasH_i = DecodeSliceWithDType(mp.BiasI, dtype, scale)
		cfg.BiasH_f = DecodeSliceWithDType(mp.BiasF, dtype, scale)
		cfg.BiasH_g = DecodeSliceWithDType(mp.BiasG, dtype, scale)
		cfg.BiasH_o = DecodeSliceWithDType(mp.BiasO, dtype, scale)
	case LayerConv1D:
		cfg.InputChannels = def.InputChannels
		cfg.Filters = def.Filters
		cfg.KernelSize = def.KernelSize
		cfg.Stride = def.Stride
		cfg.Padding = def.Padding
		cfg.InputHeight = def.InputHeight
		cfg.InputWidth = def.InputWidth
		cfg.OutputHeight = def.OutputHeight
		cfg.OutputWidth = def.OutputWidth
		cfg.Kernel = DecodeSliceWithDType(mp.Kernel, dtype, scale)
		cfg.Bias = DecodeSliceWithDType(mp.Biases, dtype, scale)
	case LayerEmbedding:
		cfg.VocabSize = def.VocabSize
		cfg.EmbeddingDim = def.EmbeddingDim
		cfg.EmbeddingWeights = DecodeSliceWithDType(mp.EmbeddingWeights, dtype, scale)
	case LayerResidual:
		// Residual layers may have optional weights
		cfg.Kernel = DecodeSliceWithDType(mp.Kernel, dtype, scale)
	case LayerSoftmax:
		cfg.SoftmaxVariant = stringToSoftmaxType(def.SoftmaxVariant)
		cfg.SoftmaxRows = def.SoftmaxRows
		cfg.SoftmaxCols = def.SoftmaxCols
		cfg.Temperature = def.Temperature
		cfg.GumbelNoise = def.GumbelNoise
		cfg.Mask = def.Mask
		cfg.HierarchyLevels = def.HierarchyLevels
		cfg.AdaptiveClusters = def.AdaptiveClusters
		cfg.MixtureWeights = def.MixtureWeights
		cfg.EntmaxAlpha = def.EntmaxAlpha
	case LayerParallel:
		cfg.CombineMode = def.CombineMode
		for j, branchDef := range def.Branches {
			var branchMP MultiPrecisionLayer
			if j < len(mp.BranchWeights) {
				branchMP = mp.BranchWeights[j]
			}
			branch := deserializeLayerMultiPrecision(branchDef, branchMP, dtype, scale)
			cfg.ParallelBranches = append(cfg.ParallelBranches, branch)
		}
	case LayerSequential:
		for j, branchDef := range def.Branches {
			var branchMP MultiPrecisionLayer
			if j < len(mp.BranchWeights) {
				branchMP = mp.BranchWeights[j]
			}
			branch := deserializeLayerMultiPrecision(branchDef, branchMP, dtype, scale)
			cfg.ParallelBranches = append(cfg.ParallelBranches, branch)
		}
	}

	return cfg
}

// DecodeSliceWithDType decodes a slice from the stored dtype to float32
func DecodeSliceWithDType(encoded string, dtype string, scale float64) []float32 {
	if encoded == "" {
		return nil
	}

	switch dtype {
	case "float64":
		f64, err := decodeFloat64Slice(encoded)
		if err != nil {
			return nil
		}
		result := make([]float32, len(f64))
		for i, v := range f64 {
			result[i] = float32(v)
		}
		return result
	case "float16":
		f16, err := decodeFloat16Slice(encoded)
		if err != nil {
			return nil
		}
		result := make([]float32, len(f16))
		for i, v := range f16 {
			result[i] = float16ToFloat32(v)
		}
		return result
	case "int8":
		i8, err := decodeInt8Slice(encoded)
		if err != nil {
			return nil
		}
		result := make([]float32, len(i8))
		for i, v := range i8 {
			result[i] = float32(float64(v) / scale)
		}
		return result
	case "int16":
		i16, err := decodeInt16Slice(encoded)
		if err != nil {
			return nil
		}
		result := make([]float32, len(i16))
		for i, v := range i16 {
			result[i] = float32(float64(v) / scale)
		}
		return result
	case "int32":
		i32, err := decodeInt32Slice(encoded)
		if err != nil {
			return nil
		}
		result := make([]float32, len(i32))
		for i, v := range i32 {
			result[i] = float32(float64(v) / scale)
		}
		return result
	case "uint8":
		u8, err := decodeUint8Slice(encoded)
		if err != nil {
			return nil
		}
		result := make([]float32, len(u8))
		for i, v := range u8 {
			// Reverse: (v / 127.5) - 1.0
			result[i] = float32((float64(v) / 127.5) - 1.0)
		}
		return result
	case "uint16":
		u16, err := decodeUint16Slice(encoded)
		if err != nil {
			return nil
		}
		result := make([]float32, len(u16))
		for i, v := range u16 {
			// Reverse: (v / 32767.5) - 1.0
			result[i] = float32((float64(v) / 32767.5) - 1.0)
		}
		return result
	case "uint32":
		u32, err := decodeUint32Slice(encoded)
		if err != nil {
			return nil
		}
		result := make([]float32, len(u32))
		for i, v := range u32 {
			// Reverse
			result[i] = float32((float64(v) / 2147483647.5) - 1.0)
		}
		return result
	case "uint64":
		u64, err := decodeUint64Slice(encoded)
		if err != nil {
			return nil
		}
		result := make([]float32, len(u64))
		for i, v := range u64 {
			result[i] = float32((float64(v) / 9.223372e18) - 1.0)
		}
		return result

	case "bfloat16":
		result, err := decodeBFloat16Slice(encoded)
		if err != nil {
			return nil
		}
		return result

	case "float8":
		result, err := decodeFloat8Slice(encoded)
		if err != nil {
			return nil
		}
		return result

	case "float4":
		// Note: float4 decode needs element count which we don't have here
		// Fall back to estimating from encoded size (2 elements per byte)
		bytes, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			return nil
		}
		numElements := len(bytes) * 2
		result, err := decodeFloat4Slice(encoded, scale, numElements)
		if err != nil {
			return nil
		}
		return result

	case "int4":
		// Similar issue - estimate element count
		bytes, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			return nil
		}
		numElements := len(bytes) * 2
		i8, err := decodeInt4Slice(encoded, numElements)
		if err != nil {
			return nil
		}
		result := make([]float32, len(i8))
		for i, v := range i8 {
			result[i] = float32(v) * float32(8.0/scale) // Reverse the scaling
		}
		return result

	case "int64":
		i64, err := decodeInt64Slice(encoded)
		if err != nil {
			return nil
		}
		result := make([]float32, len(i64))
		for i, v := range i64 {
			result[i] = float32(float64(v) / scale)
		}
		return result

	default: // float32
		f32, err := decodeFloat32Slice(encoded)
		if err != nil {
			return nil
		}
		return f32
	}
}

// stringToLayerType converts string to LayerType
func stringToLayerType(s string) LayerType {
	switch s {
	case "dense":
		return LayerDense
	case "conv2d":
		return LayerConv2D
	case "conv1d":
		return LayerConv1D
	case "multi_head_attention", "mha":
		return LayerMultiHeadAttention
	case "rnn":
		return LayerRNN
	case "lstm":
		return LayerLSTM
	case "softmax":
		return LayerSoftmax
	case "layer_norm", "layernorm":
		return LayerNorm
	case "rms_norm", "rmsnorm":
		return LayerRMSNorm
	case "swiglu":
		return LayerSwiGLU
	case "embedding":
		return LayerEmbedding
	case "residual":
		return LayerResidual
	case "parallel":
		return LayerParallel
	case "sequential":
		return LayerSequential
	default:
		return LayerDense
	}
}

// =============================================================================
// Model Size Information
// =============================================================================

// ModelSizeInfo contains information about model storage size
type ModelSizeInfo struct {
	DType          string `json:"dtype"`
	TotalWeights   int    `json:"total_weights"`
	BytesPerWeight int    `json:"bytes_per_weight"`
	TotalBytes     int    `json:"total_bytes"`
	Base64Bytes    int    `json:"base64_bytes"` // After base64 encoding (~33% larger)
}

// GetModelSizeInfo returns size information for different dtypes
func (n *Network) GetModelSizeInfo() map[string]ModelSizeInfo {
	// Count total weights
	totalWeights := 0
	totalLayers := n.GridRows * n.GridCols * n.LayersPerCell
	for i := 0; i < totalLayers; i++ {
		row := i / (n.GridCols * n.LayersPerCell)
		col := (i / n.LayersPerCell) % n.GridCols
		layer := i % n.LayersPerCell
		cfg := n.GetLayer(row, col, layer)
		totalWeights += countLayerWeights(cfg)
	}

	return map[string]ModelSizeInfo{
		"float32": {
			DType:          "float32",
			TotalWeights:   totalWeights,
			BytesPerWeight: 4,
			TotalBytes:     totalWeights * 4,
			Base64Bytes:    int(float64(totalWeights*4) * 1.34),
		},
		"float64": {
			DType:          "float64",
			TotalWeights:   totalWeights,
			BytesPerWeight: 8,
			TotalBytes:     totalWeights * 8,
			Base64Bytes:    int(float64(totalWeights*8) * 1.34),
		},
		"float16": {
			DType:          "float16",
			TotalWeights:   totalWeights,
			BytesPerWeight: 2,
			TotalBytes:     totalWeights * 2,
			Base64Bytes:    int(float64(totalWeights*2) * 1.34),
		},
		"int32": {
			DType:          "int32",
			TotalWeights:   totalWeights,
			BytesPerWeight: 4,
			TotalBytes:     totalWeights * 4,
			Base64Bytes:    int(float64(totalWeights*4) * 1.34),
		},
		"int16": {
			DType:          "int16",
			TotalWeights:   totalWeights,
			BytesPerWeight: 2,
			TotalBytes:     totalWeights * 2,
			Base64Bytes:    int(float64(totalWeights*2) * 1.34),
		},
		"int8": {
			DType:          "int8",
			TotalWeights:   totalWeights,
			BytesPerWeight: 1,
			TotalBytes:     totalWeights * 1,
			Base64Bytes:    int(float64(totalWeights*1) * 1.34),
		},
		"uint32": {
			DType:          "uint32",
			TotalWeights:   totalWeights,
			BytesPerWeight: 4,
			TotalBytes:     totalWeights * 4,
			Base64Bytes:    int(float64(totalWeights*4) * 1.34),
		},
		"uint16": {
			DType:          "uint16",
			TotalWeights:   totalWeights,
			BytesPerWeight: 2,
			TotalBytes:     totalWeights * 2,
			Base64Bytes:    int(float64(totalWeights*2) * 1.34),
		},
		"uint8": {
			DType:          "uint8",
			TotalWeights:   totalWeights,
			BytesPerWeight: 1,
			TotalBytes:     totalWeights * 1,
			Base64Bytes:    int(float64(totalWeights*1) * 1.34),
		},
		"uint64": {
			DType:          "uint64",
			TotalWeights:   totalWeights,
			BytesPerWeight: 8,
			TotalBytes:     totalWeights * 8,
			Base64Bytes:    int(float64(totalWeights*8) * 1.34),
		},
	}
}

// countLayerWeights counts the number of weights in a layer
func countLayerWeights(cfg *LayerConfig) int {
	count := 0
	switch cfg.Type {
	case LayerDense:
		count = len(cfg.Kernel) + len(cfg.Bias)
	case LayerConv2D:
		count = len(cfg.Kernel) + len(cfg.Bias)
	case LayerMultiHeadAttention:
		count = len(cfg.QWeights) + len(cfg.KWeights) + len(cfg.VWeights) + len(cfg.OutputWeight)
		count += len(cfg.QBias) + len(cfg.KBias) + len(cfg.VBias) + len(cfg.OutputBias)
	case LayerNorm:
		count = len(cfg.Gamma) + len(cfg.Beta)
	case LayerRMSNorm:
		count = len(cfg.Gamma)
	case LayerSwiGLU:
		count = len(cfg.GateWeights) + len(cfg.UpWeights) + len(cfg.DownWeights)
		count += len(cfg.GateBias) + len(cfg.UpBias) + len(cfg.DownBias)
	case LayerParallel:
		for i := range cfg.ParallelBranches {
			count += countLayerWeights(&cfg.ParallelBranches[i])
		}
	case LayerSequential:
		for i := range cfg.ParallelBranches {
			count += countLayerWeights(&cfg.ParallelBranches[i])
		}
	}
	return count
}
