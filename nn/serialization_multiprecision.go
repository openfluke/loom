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
	DType  string                  `json:"dtype"`  // "float32", "float64", "int8", "int16", "int32"
	Scale  float64                 `json:"scale"`  // Quantization scale (for int types)
	Layers []MultiPrecisionLayer   `json:"layers"`
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
	WeightIH     string `json:"weight_ih,omitempty"`
	WeightHH     string `json:"weight_hh,omitempty"`
	BiasH        string `json:"bias_h,omitempty"`
	Gamma        string `json:"gamma,omitempty"`
	Beta         string `json:"beta,omitempty"`
	GateWeights  string `json:"gate_weights,omitempty"`
	UpWeights    string `json:"up_weights,omitempty"`
	DownWeights  string `json:"down_weights,omitempty"`
	GateBias     string `json:"gate_bias,omitempty"`
	UpBias       string `json:"up_bias,omitempty"`
	DownBias     string `json:"down_bias,omitempty"`

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

// SaveModelWithDType saves a model with weights converted to the specified dtype
// Supported dtypes: "float32", "float64", "int8", "int16", "int32"
func (n *Network) SaveModelWithDType(modelID string, dtype string) (string, error) {
	config := NetworkConfig{
		ID:            modelID,
		BatchSize:     n.BatchSize,
		GridRows:      n.GridRows,
		GridCols:      n.GridCols,
		LayersPerCell: n.LayersPerCell,
		Layers:        []LayerDefinition{},
	}

	// Determine quantization scale for integer types
	scale := 1.0
	switch dtype {
	case "int8":
		scale = 127.0 // Map [-1, 1] to [-127, 127]
	case "int16":
		scale = 32767.0 // Map [-1, 1] to [-32767, 32767]
	case "int32":
		scale = 2147483647.0 // Full range
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
		case LayerMultiHeadAttention:
			layerDef.DModel = layerConfig.DModel
			layerDef.NumHeads = layerConfig.NumHeads
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
		case LayerParallel:
			layerDef.CombineMode = layerConfig.CombineMode
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
		mp.Kernel = encodeSliceWithDType(cfg.Kernel, dtype, scale)
		mp.Biases = encodeSliceWithDType(cfg.Bias, dtype, scale)
	case LayerConv2D:
		mp.Kernel = encodeSliceWithDType(cfg.Kernel, dtype, scale)
		mp.Biases = encodeSliceWithDType(cfg.Bias, dtype, scale)
	case LayerMultiHeadAttention:
		mp.QWeights = encodeSliceWithDType(cfg.QWeights, dtype, scale)
		mp.KWeights = encodeSliceWithDType(cfg.KWeights, dtype, scale)
		mp.VWeights = encodeSliceWithDType(cfg.VWeights, dtype, scale)
		mp.OutputWeight = encodeSliceWithDType(cfg.OutputWeight, dtype, scale)
		mp.QBias = encodeSliceWithDType(cfg.QBias, dtype, scale)
		mp.KBias = encodeSliceWithDType(cfg.KBias, dtype, scale)
		mp.VBias = encodeSliceWithDType(cfg.VBias, dtype, scale)
		mp.OutputBias = encodeSliceWithDType(cfg.OutputBias, dtype, scale)
	case LayerNorm:
		mp.Gamma = encodeSliceWithDType(cfg.Gamma, dtype, scale)
		mp.Beta = encodeSliceWithDType(cfg.Beta, dtype, scale)
	case LayerRMSNorm:
		mp.Gamma = encodeSliceWithDType(cfg.Gamma, dtype, scale)
	case LayerSwiGLU:
		mp.GateWeights = encodeSliceWithDType(cfg.GateWeights, dtype, scale)
		mp.UpWeights = encodeSliceWithDType(cfg.UpWeights, dtype, scale)
		mp.DownWeights = encodeSliceWithDType(cfg.DownWeights, dtype, scale)
		mp.GateBias = encodeSliceWithDType(cfg.GateBias, dtype, scale)
		mp.UpBias = encodeSliceWithDType(cfg.UpBias, dtype, scale)
		mp.DownBias = encodeSliceWithDType(cfg.DownBias, dtype, scale)
	case LayerParallel:
		for _, branch := range cfg.ParallelBranches {
			mp.BranchWeights = append(mp.BranchWeights, serializeLayerMultiPrecision(&branch, dtype, scale))
		}
	}

	return mp
}

// encodeSliceWithDType encodes a float32 slice to the target dtype
func encodeSliceWithDType(data []float32, dtype string, scale float64) string {
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
		cfg.Kernel = decodeSliceWithDType(mp.Kernel, dtype, scale)
		cfg.Bias = decodeSliceWithDType(mp.Biases, dtype, scale)
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
		cfg.Kernel = decodeSliceWithDType(mp.Kernel, dtype, scale)
		cfg.Bias = decodeSliceWithDType(mp.Biases, dtype, scale)
	case LayerMultiHeadAttention:
		cfg.DModel = def.DModel
		cfg.NumHeads = def.NumHeads
		cfg.SeqLength = def.SeqLength
		cfg.QWeights = decodeSliceWithDType(mp.QWeights, dtype, scale)
		cfg.KWeights = decodeSliceWithDType(mp.KWeights, dtype, scale)
		cfg.VWeights = decodeSliceWithDType(mp.VWeights, dtype, scale)
		cfg.OutputWeight = decodeSliceWithDType(mp.OutputWeight, dtype, scale)
		cfg.QBias = decodeSliceWithDType(mp.QBias, dtype, scale)
		cfg.KBias = decodeSliceWithDType(mp.KBias, dtype, scale)
		cfg.VBias = decodeSliceWithDType(mp.VBias, dtype, scale)
		cfg.OutputBias = decodeSliceWithDType(mp.OutputBias, dtype, scale)
	case LayerNorm:
		cfg.NormSize = def.NormSize
		cfg.Epsilon = def.Epsilon
		cfg.Gamma = decodeSliceWithDType(mp.Gamma, dtype, scale)
		cfg.Beta = decodeSliceWithDType(mp.Beta, dtype, scale)
	case LayerRMSNorm:
		cfg.NormSize = def.NormSize
		cfg.Epsilon = def.Epsilon
		cfg.Gamma = decodeSliceWithDType(mp.Gamma, dtype, scale)
	case LayerSwiGLU:
		cfg.InputHeight = def.InputHeight
		cfg.OutputHeight = def.OutputHeight
		cfg.GateWeights = decodeSliceWithDType(mp.GateWeights, dtype, scale)
		cfg.UpWeights = decodeSliceWithDType(mp.UpWeights, dtype, scale)
		cfg.DownWeights = decodeSliceWithDType(mp.DownWeights, dtype, scale)
		cfg.GateBias = decodeSliceWithDType(mp.GateBias, dtype, scale)
		cfg.UpBias = decodeSliceWithDType(mp.UpBias, dtype, scale)
		cfg.DownBias = decodeSliceWithDType(mp.DownBias, dtype, scale)
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
	}

	return cfg
}

// decodeSliceWithDType decodes a slice from the stored dtype to float32
func decodeSliceWithDType(encoded string, dtype string, scale float64) []float32 {
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
	case "parallel":
		return LayerParallel
	case "embedding":
		return LayerEmbedding
	case "conv1d":
		return LayerConv1D
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
	}
	return count
}
