package poly

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
)

// PersistenceNetworkSpec represents the serializable state of a VolumetricNetwork.
type PersistenceNetworkSpec struct {
	ID            string                 `json:"id"`
	Depth         int                    `json:"depth"`
	Rows          int                    `json:"rows"`
	Cols          int                    `json:"cols"`
	LayersPerCell int                    `json:"layers_per_cell"`
	Layers        []PersistenceLayerSpec `json:"layers"`
}

// PersistenceLayerSpec represents the serializable state of a VolumetricLayer.
type PersistenceLayerSpec struct {
	Z int `json:"z"`
	Y int `json:"y"`
	X int `json:"x"`
	L int `json:"l"`

	Type       string `json:"type"`
	Activation string `json:"activation"`
	DType      string `json:"dtype"`

	InputHeight   int `json:"input_height,omitempty"`
	InputWidth    int `json:"input_width,omitempty"`
	InputDepth    int `json:"input_depth,omitempty"`
	OutputHeight  int `json:"output_height,omitempty"`
	OutputWidth   int `json:"output_width,omitempty"`
	OutputDepth   int `json:"output_depth,omitempty"`
	InputChannels int `json:"input_channels,omitempty"`
	Filters       int `json:"filters,omitempty"`
	KernelSize    int `json:"kernel_size,omitempty"`
	Stride        int `json:"stride,omitempty"`
	Padding       int `json:"padding,omitempty"`
	OutputPadding int `json:"output_padding,omitempty"`

	NumHeads     int     `json:"num_heads,omitempty"`
	NumKVHeads   int     `json:"num_kv_heads,omitempty"`
	HeadDim      int     `json:"head_dim,omitempty"`
	DModel       int     `json:"d_model,omitempty"`
	SeqLength    int     `json:"seq_length,omitempty"`
	RoPEFreqBase float64 `json:"rope_freq_base,omitempty"`

	VocabSize    int `json:"vocab_size,omitempty"`
	EmbeddingDim int `json:"embedding_dim,omitempty"`

	NumClusters       int     `json:"num_clusters,omitempty"`
	KMeansTemperature float64 `json:"kmeans_temperature,omitempty"`
	OutputMode        string  `json:"output_mode,omitempty"`

	SoftmaxType string  `json:"softmax_type,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
	SoftmaxRows int     `json:"softmax_rows,omitempty"`
	SoftmaxCols int     `json:"softmax_cols,omitempty"`
	EntmaxAlpha float64 `json:"entmax_alpha,omitempty"`
	GumbelNoise bool    `json:"gumbel_noise,omitempty"`

	// Weights
	Weights string  `json:"weights,omitempty"` // Base64 encoded weights
	Native  bool    `json:"native,omitempty"`  // True if weights are in target DType, False if Master FP32
	Scale   float32 `json:"scale,omitempty"`

	// Recursion
	ParallelBranches []PersistenceLayerSpec `json:"parallel_branches,omitempty"`
	CombineMode      string                 `json:"combine_mode,omitempty"`
	SequentialLayers []PersistenceLayerSpec `json:"sequential_layers,omitempty"`

	UseTiling  bool `json:"use_tiling,omitempty"`
	TileSize   int  `json:"tile_size,omitempty"`
	IsDisabled bool `json:"is_disabled,omitempty"`

	// Metacognition
	MetaRules         []MetaRule            `json:"meta_rules,omitempty"`
	MetaObservedLayer *PersistenceLayerSpec `json:"meta_observed_layer,omitempty"`
}

// SerializeNetwork converts a VolumetricNetwork into a JSON byte slice.
func SerializeNetwork(net *VolumetricNetwork) ([]byte, error) {
	spec := PersistenceNetworkSpec{
		ID:            "network",
		Depth:         net.Depth,
		Rows:          net.Rows,
		Cols:          net.Cols,
		LayersPerCell: net.LayersPerCell,
		Layers:        make([]PersistenceLayerSpec, 0, len(net.Layers)),
	}

	for _, l := range net.Layers {
		spec.Layers = append(spec.Layers, serializeLayer(&l))
	}

	return json.MarshalIndent(spec, "", "  ")
}

func serializeLayer(l *VolumetricLayer) PersistenceLayerSpec {
	ls := PersistenceLayerSpec{
		Z: l.Z, Y: l.Y, X: l.X, L: l.L,
		Type:       fmt.Sprintf("%v", l.Type),
		Activation: fmt.Sprintf("%v", l.Activation),
		DType:      l.DType.String(),

		InputHeight:   l.InputHeight,
		InputWidth:    l.InputWidth,
		InputDepth:    l.InputDepth,
		OutputHeight:  l.OutputHeight,
		OutputWidth:   l.OutputWidth,
		OutputDepth:   l.OutputDepth,
		InputChannels: l.InputChannels,
		Filters:       l.Filters,
		KernelSize:    l.KernelSize,
		Stride:        l.Stride,
		Padding:       l.Padding,
		OutputPadding: l.OutputPadding,

		NumHeads:     l.NumHeads,
		NumKVHeads:   l.NumKVHeads,
		HeadDim:      l.HeadDim,
		DModel:       l.DModel,
		SeqLength:    l.SeqLength,
		RoPEFreqBase: l.RoPEFreqBase,

		VocabSize:    l.VocabSize,
		EmbeddingDim: l.EmbeddingDim,

		NumClusters:       l.NumClusters,
		KMeansTemperature: l.KMeansTemperature,
		OutputMode:        l.KMeansOutputMode,

		SoftmaxType: fmt.Sprintf("%v", l.SoftmaxType),
		Temperature: l.Temperature,
		SoftmaxRows: l.SoftmaxRows,
		SoftmaxCols: l.SoftmaxCols,
		EntmaxAlpha: l.EntmaxAlpha,
		GumbelNoise: l.GumbelNoise,

		CombineMode: l.CombineMode,
		UseTiling:   l.UseTiling,
		TileSize:    l.TileSize,
		IsDisabled:  l.IsDisabled,
	}

	if l.WeightStore != nil && len(l.WeightStore.Master) > 0 {
		// Persist native weights for this layer's DType (int8 bytes, bf16 uint16, etc.).
		// Keep the live scale so save/load round-trips match what training used.
		dt := l.DType
		delete(l.WeightStore.Versions, dt)
		l.WeightStore.Morph(dt)
		ls.Scale = l.WeightStore.Scale
		active := l.WeightStore.Versions[dt]
		if active == nil {
			active = l.WeightStore.GetNative(dt)
		}
		if active != nil {
			ls.Weights = encodeNativeWeights(active, dt)
			ls.Native = true
		}
	}

	if len(l.ParallelBranches) > 0 {
		ls.ParallelBranches = make([]PersistenceLayerSpec, len(l.ParallelBranches))
		for i := range l.ParallelBranches {
			ls.ParallelBranches[i] = serializeLayer(&l.ParallelBranches[i])
		}
	}

	if len(l.SequentialLayers) > 0 {
		ls.SequentialLayers = make([]PersistenceLayerSpec, len(l.SequentialLayers))
		for i := range l.SequentialLayers {
			ls.SequentialLayers[i] = serializeLayer(&l.SequentialLayers[i])
		}
	}

	if l.MetaObservedLayer != nil {
		observed := serializeLayer(l.MetaObservedLayer)
		ls.MetaObservedLayer = &observed
	}
	ls.MetaRules = l.MetaRules

	return ls
}

// DeserializeNetwork reconstructs a VolumetricNetwork from a JSON byte slice.
func DeserializeNetwork(jsonData []byte) (*VolumetricNetwork, error) {
	var spec PersistenceNetworkSpec
	if err := json.Unmarshal(jsonData, &spec); err != nil {
		return nil, err
	}

	net := NewVolumetricNetwork(spec.Depth, spec.Rows, spec.Cols, spec.LayersPerCell)
	for _, ls := range spec.Layers {
		l := net.GetLayer(ls.Z, ls.Y, ls.X, ls.L)
		if err := applyPersistenceLayerSpec(l, ls); err != nil {
			return nil, err
		}
	}

	return net, nil
}

func applyPersistenceLayerSpec(l *VolumetricLayer, ls PersistenceLayerSpec) error {
	l.Type = ParseLayerType(ls.Type)
	l.Activation = ParseActivationType(ls.Activation)
	l.DType = ParseDType(ls.DType)

	l.InputHeight = ls.InputHeight
	l.InputWidth = ls.InputWidth
	l.InputDepth = ls.InputDepth
	l.OutputHeight = ls.OutputHeight
	l.OutputWidth = ls.OutputWidth
	l.OutputDepth = ls.OutputDepth
	l.InputChannels = ls.InputChannels
	l.Filters = ls.Filters
	l.KernelSize = ls.KernelSize
	l.Stride = ls.Stride
	l.Padding = ls.Padding
	l.OutputPadding = ls.OutputPadding

	l.NumHeads = ls.NumHeads
	l.NumKVHeads = ls.NumKVHeads
	l.HeadDim = ls.HeadDim
	l.DModel = ls.DModel
	l.SeqLength = ls.SeqLength
	l.RoPEFreqBase = ls.RoPEFreqBase

	l.VocabSize = ls.VocabSize
	l.EmbeddingDim = ls.EmbeddingDim

	l.NumClusters = ls.NumClusters
	l.KMeansTemperature = ls.KMeansTemperature
	l.KMeansOutputMode = ls.OutputMode

	l.SoftmaxType = ParseSoftmaxType(ls.SoftmaxType)
	l.Temperature = ls.Temperature
	l.SoftmaxRows = ls.SoftmaxRows
	l.SoftmaxCols = ls.SoftmaxCols
	l.EntmaxAlpha = ls.EntmaxAlpha
	l.GumbelNoise = ls.GumbelNoise

	l.CombineMode = ls.CombineMode
	l.UseTiling = ls.UseTiling
	l.TileSize = ls.TileSize
	l.IsDisabled = ls.IsDisabled

	// Initialize weights based on populated fields
	initializeWeights(l)

	if ls.Weights != "" {
		if l.WeightStore == nil {
			return fmt.Errorf("failed to initialize WeightStore for layer type %v", l.Type)
		}
		l.WeightStore.Scale = ls.Scale
		dt := l.DType

		if ls.Native {
			decoded, err := decodeNativeWeights(ls.Weights, dt)
			if err != nil {
				return err
			}
			l.WeightStore.SetLoadedWeights(dt, decoded)
		} else {
			// Legacy checkpoints: float32 master blob before per-dtype native saves.
			m, err := decodeWeights(ls.Weights)
			if err != nil {
				return err
			}
			l.WeightStore.Master = m
			l.WeightStore.Morph(dt)
		}
	}

	if len(ls.ParallelBranches) > 0 {
		l.ParallelBranches = make([]VolumetricLayer, len(ls.ParallelBranches))
		for i, bs := range ls.ParallelBranches {
			if err := applyPersistenceLayerSpec(&l.ParallelBranches[i], bs); err != nil {
				return err
			}
		}
	}

	if len(ls.SequentialLayers) > 0 {
		l.SequentialLayers = make([]VolumetricLayer, len(ls.SequentialLayers))
		for i, ss := range ls.SequentialLayers {
			if err := applyPersistenceLayerSpec(&l.SequentialLayers[i], ss); err != nil {
				return err
			}
		}
	}

	if ls.MetaObservedLayer != nil {
		l.MetaObservedLayer = new(VolumetricLayer)
		if err := applyPersistenceLayerSpec(l.MetaObservedLayer, *ls.MetaObservedLayer); err != nil {
			return err
		}
	}
	l.MetaRules = ls.MetaRules

	return nil
}

// ParseSoftmaxType converts string to SoftmaxType.
func ParseSoftmaxType(s string) SoftmaxType {
	switch s {
	case "Standard":
		return SoftmaxStandard
	case "Grid":
		return SoftmaxGrid
	case "Hierarchical":
		return SoftmaxHierarchical
	case "Temperature":
		return SoftmaxTemperature
	case "Gumbel":
		return SoftmaxGumbel
	case "Masked":
		return SoftmaxMasked
	case "Sparse":
		return SoftmaxSparse
	case "Adaptive":
		return SoftmaxAdaptive
	case "Mixture":
		return SoftmaxMixture
	case "Entmax":
		return SoftmaxEntmax
	default:
		return SoftmaxStandard
	}
}

// encodeWeights converts float32 slice to base64 string (Little Endian).
func encodeWeights(w []float32) string {
	bytes := make([]byte, len(w)*4)
	for i, v := range w {
		binary.LittleEndian.PutUint32(bytes[i*4:], math.Float32bits(v))
	}
	return base64.StdEncoding.EncodeToString(bytes)
}

// decodeWeights converts base64 string to float32 slice.
func decodeWeights(s string) ([]float32, error) {
	bytes, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return nil, err
	}
	if len(bytes)%4 != 0 {
		return nil, fmt.Errorf("invalid weight byte length: %d", len(bytes))
	}
	w := make([]float32, len(bytes)/4)
	for i := range w {
		w[i] = math.Float32frombits(binary.LittleEndian.Uint32(bytes[i*4:]))
	}
	return w, nil
}

// LayerPersistenceFromJSON reads the first layer's on-disk weight fields from serialized network JSON.
func LayerPersistenceFromJSON(jsonData []byte, layerIndex int) (weightsB64 string, scale float32, native bool, err error) {
	var spec PersistenceNetworkSpec
	if err := json.Unmarshal(jsonData, &spec); err != nil {
		return "", 0, false, err
	}
	if layerIndex < 0 || layerIndex >= len(spec.Layers) {
		return "", 0, false, fmt.Errorf("layer index %d out of range (layers=%d)", layerIndex, len(spec.Layers))
	}
	ls := spec.Layers[layerIndex]
	return ls.Weights, ls.Scale, ls.Native, nil
}

// NativeWeightsEncoded compares two native weight stores by on-disk encoding.
func NativeWeightsEncoded(a any, b any, dtype DType) bool {
	if a == nil || b == nil {
		return false
	}
	return encodeNativeWeights(a, dtype) == encodeNativeWeights(b, dtype)
}

// LayerNativePersistenceSnapshot returns the native weights blob and scale that
// SerializeNetwork would write for this layer dtype (Morph + encode, no JSON).
func LayerNativePersistenceSnapshot(ws *WeightStore, dtype DType) (weightsB64 string, scale float32, ok bool) {
	if ws == nil || len(ws.Master) == 0 {
		return "", 0, false
	}
	delete(ws.Versions, dtype)
	ws.Morph(dtype)
	scale = ws.Scale
	active := ws.Versions[dtype]
	if active == nil {
		active = ws.GetNative(dtype)
	}
	if active == nil {
		return "", scale, false
	}
	return encodeNativeWeights(active, dtype), scale, true
}

// encodeNativeWeights converts an active weight slice to base64 string.
func encodeNativeWeights(data any, dt DType) string {
	switch w := data.(type) {
	case []float64:
		buf := make([]byte, len(w)*8)
		for i, v := range w {
			binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(v))
		}
		return base64.StdEncoding.EncodeToString(buf)
	case []float32:
		if dt == DTypeBFloat16 {
			// Pack as 16-bit BFloat16 (top 16 bits of float32)
			buf := make([]byte, len(w)*2)
			for i, v := range w {
				u32 := math.Float32bits(v)
				binary.LittleEndian.PutUint16(buf[i*2:], uint16(u32>>16))
			}
			return base64.StdEncoding.EncodeToString(buf)
		} else if dt == DTypeFloat16 {
			// Pack as 16-bit IEEE Float16 (simulated truncation for now)
			buf := make([]byte, len(w)*2)
			for i, v := range w {
				// Simple truncation of mantissa for now (not true Float16 but 2 bytes)
				u32 := math.Float32bits(v)
				binary.LittleEndian.PutUint16(buf[i*2:], uint16(u32>>16))
			}
			return base64.StdEncoding.EncodeToString(buf)
		}
		return encodeWeights(w)
	case []int64:
		buf := make([]byte, len(w)*8)
		for i, v := range w {
			binary.LittleEndian.PutUint64(buf[i*8:], uint64(v))
		}
		return base64.StdEncoding.EncodeToString(buf)
	case []int32:
		buf := make([]byte, len(w)*4)
		for i, v := range w {
			binary.LittleEndian.PutUint32(buf[i*4:], uint32(v))
		}
		return base64.StdEncoding.EncodeToString(buf)
	case []int16:
		buf := make([]byte, len(w)*2)
		for i, v := range w {
			binary.LittleEndian.PutUint16(buf[i*2:], uint16(v))
		}
		return base64.StdEncoding.EncodeToString(buf)
	case []int8:
		if dt == DTypeInt4 || dt == DTypeUint4 || dt == DTypeFP4 {
			// Pack 2 weights per byte
			buf := make([]byte, (len(w)+1)/2)
			for i, v := range w {
				val := byte(v & 0x0F)
				if i%2 == 0 {
					buf[i/2] |= (val << 4)
				} else {
					buf[i/2] |= val
				}
			}
			return base64.StdEncoding.EncodeToString(buf)
		} else if dt == DTypeInt2 || dt == DTypeUint2 || dt == DTypeTernary {
			// Pack 4 weights per byte
			buf := make([]byte, (len(w)+3)/4)
			for i, v := range w {
				val := byte(v & 0x03)
				shift := uint(6 - (i%4)*2)
				buf[i/4] |= (val << shift)
			}
			return base64.StdEncoding.EncodeToString(buf)
		} else if dt == DTypeBinary {
			// Pack 8 weights per byte
			buf := make([]byte, (len(w)+7)/8)
			for i, v := range w {
				if v > 0 {
					buf[i/8] |= (1 << uint(7-(i%8)))
				}
			}
			return base64.StdEncoding.EncodeToString(buf)
		}
		buf := make([]byte, len(w))
		for i, v := range w {
			buf[i] = byte(v)
		}
		return base64.StdEncoding.EncodeToString(buf)
	case []uint64:
		buf := make([]byte, len(w)*8)
		for i, v := range w {
			binary.LittleEndian.PutUint64(buf[i*8:], v)
		}
		return base64.StdEncoding.EncodeToString(buf)
	case []uint32:
		buf := make([]byte, len(w)*4)
		for i, v := range w {
			binary.LittleEndian.PutUint32(buf[i*4:], v)
		}
		return base64.StdEncoding.EncodeToString(buf)
	case []uint16:
		buf := make([]byte, len(w)*2)
		for i, v := range w {
			binary.LittleEndian.PutUint16(buf[i*2:], v)
		}
		return base64.StdEncoding.EncodeToString(buf)
	case []uint8:
		if dt == DTypeInt4 || dt == DTypeUint4 || dt == DTypeFP4 {
			// Sub-byte native storage uses one 4-bit code per weight; persist it compactly.
			buf := make([]byte, (len(w)+1)/2)
			for i, v := range w {
				val := v & 0x0F
				if i%2 == 0 {
					buf[i/2] |= val << 4
				} else {
					buf[i/2] |= val
				}
			}
			return base64.StdEncoding.EncodeToString(buf)
		} else if dt == DTypeInt2 || dt == DTypeUint2 || dt == DTypeTernary {
			buf := make([]byte, (len(w)+3)/4)
			for i, v := range w {
				val := v & 0x03
				shift := uint(6 - (i%4)*2)
				buf[i/4] |= val << shift
			}
			return base64.StdEncoding.EncodeToString(buf)
		} else if dt == DTypeBinary {
			buf := make([]byte, (len(w)+7)/8)
			for i, v := range w {
				if int8(v) > 0 {
					buf[i/8] |= 1 << uint(7-(i%8))
				}
			}
			return base64.StdEncoding.EncodeToString(buf)
		}
		return base64.StdEncoding.EncodeToString(w)
	default:
		return ""
	}
}

// DecodeNativeWeights converts a base64 native checkpoint blob to in-memory native storage.
func DecodeNativeWeights(s string, dt DType) (any, error) {
	return decodeNativeWeights(s, dt)
}

// decodeNativeWeights converts base64 string to a slice of the given DType.
func decodeNativeWeights(s string, dt DType) (any, error) {
	bytes, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return nil, err
	}

	switch dt {
	case DTypeFloat64:
		w := make([]float64, len(bytes)/8)
		for i := range w {
			w[i] = math.Float64frombits(binary.LittleEndian.Uint64(bytes[i*8:]))
		}
		return w, nil
	case DTypeFloat32:
		return decodeWeights(s)
	case DTypeFloat16, DTypeBFloat16:
		w := make([]uint16, len(bytes)/2)
		for i := range w {
			w[i] = binary.LittleEndian.Uint16(bytes[i*2:])
		}
		return w, nil
	case DTypeInt64:
		w := make([]int64, len(bytes)/8)
		for i := range w {
			w[i] = int64(binary.LittleEndian.Uint64(bytes[i*8:]))
		}
		return w, nil
	case DTypeUint64:
		w := make([]uint64, len(bytes)/8)
		for i := range w {
			w[i] = binary.LittleEndian.Uint64(bytes[i*8:])
		}
		return w, nil
	case DTypeInt32:
		w := make([]int32, len(bytes)/4)
		for i := range w {
			w[i] = int32(binary.LittleEndian.Uint32(bytes[i*4:]))
		}
		return w, nil
	case DTypeUint32:
		w := make([]uint32, len(bytes)/4)
		for i := range w {
			w[i] = binary.LittleEndian.Uint32(bytes[i*4:])
		}
		return w, nil
	case DTypeInt16:
		w := make([]int16, len(bytes)/2)
		for i := range w {
			w[i] = int16(binary.LittleEndian.Uint16(bytes[i*2:]))
		}
		return w, nil
	case DTypeUint16:
		w := make([]uint16, len(bytes)/2)
		for i := range w {
			w[i] = binary.LittleEndian.Uint16(bytes[i*2:])
		}
		return w, nil
	case DTypeInt8:
		w := make([]int8, len(bytes))
		for i := range w {
			w[i] = int8(bytes[i])
		}
		return w, nil
	case DTypeUint8, DTypeFP8E4M3, DTypeFP8E5M2:
		w := make([]uint8, len(bytes))
		copy(w, bytes)
		return w, nil
	case DTypeInt4, DTypeUint4:
		// Unpack 2 weights per byte into native []uint8 storage.
		w := make([]uint8, len(bytes)*2)
		for i := range bytes {
			valHigh := (bytes[i] >> 4) & 0x0F
			valLow := bytes[i] & 0x0F

			sh := int8(valHigh)
			sl := int8(valLow)

			// Sign extend only if it's a signed type
			if dt == DTypeInt4 {
				if sh > 7 {
					sh -= 16
				}
				if sl > 7 {
					sl -= 16
				}
			}

			w[i*2] = uint8(sh)
			if i*2+1 < len(w) {
				w[i*2+1] = uint8(sl)
			}
		}
		return w, nil
	case DTypeFP4:
		// Unpack 2 FP4 codes per byte into []uint8 code storage.
		w := make([]uint8, len(bytes)*2)
		for i := range bytes {
			w[i*2] = (bytes[i] >> 4) & 0x0F
			if i*2+1 < len(w) {
				w[i*2+1] = bytes[i] & 0x0F
			}
		}
		return w, nil
	case DTypeInt2, DTypeUint2, DTypeTernary:
		// Unpack 4 weights per byte into native []uint8 storage.
		w := make([]uint8, len(bytes)*4)
		for i := range bytes {
			for j := 0; j < 4; j++ {
				shift := uint(6 - j*2)
				val := (bytes[i] >> shift) & 0x03
				sv := int8(val)
				// Sign extend only if it's a signed type
				if dt == DTypeInt2 || dt == DTypeTernary {
					if sv > 1 {
						sv -= 4
					}
				}
				if i*4+j < len(w) {
					w[i*4+j] = uint8(sv)
				}
			}
		}
		return w, nil
	case DTypeBinary:
		// Unpack 8 weights per byte into native []uint8 storage.
		w := make([]uint8, len(bytes)*8)
		for i := range bytes {
			for j := 0; j < 8; j++ {
				bit := (bytes[i] >> uint(7-j)) & 0x01
				val := int8(-1)
				if bit == 1 {
					val = 1
				}
				if i*8+j < len(w) {
					w[i*8+j] = uint8(val)
				}
			}
		}
		return w, nil
	default:
		return bytes, nil
	}
}
