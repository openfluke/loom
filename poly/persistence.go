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
	ID            string                   `json:"id"`
	Depth         int                      `json:"depth"`
	Rows          int                      `json:"rows"`
	Cols          int                      `json:"cols"`
	LayersPerCell int                      `json:"layers_per_cell"`
	Layers        []PersistenceLayerSpec   `json:"layers"`
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

	SoftmaxType     string  `json:"softmax_type,omitempty"`
	Temperature     float64 `json:"temperature,omitempty"`
	SoftmaxRows     int     `json:"softmax_rows,omitempty"`
	SoftmaxCols     int     `json:"softmax_cols,omitempty"`
	EntmaxAlpha     float64 `json:"entmax_alpha,omitempty"`
	GumbelNoise     bool    `json:"gumbel_noise,omitempty"`

	// Weights
	Weights string  `json:"weights,omitempty"` // Base64 encoded weights
	Native  bool    `json:"native,omitempty"`  // True if weights are in target DType, False if Master FP32
	Scale   float32 `json:"scale,omitempty"`

	// Recursion
	ParallelBranches []PersistenceLayerSpec `json:"parallel_branches,omitempty"`
	CombineMode      string                 `json:"combine_mode,omitempty"`
	SequentialLayers []PersistenceLayerSpec `json:"sequential_layers,omitempty"`
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
	}

	if l.WeightStore != nil {
		dt := l.DType
		active := l.WeightStore.GetActive(dt)
		
		// If not cached, but it's the target DType, we should morph it for saving
		if active == nil && dt != DTypeFloat32 {
			l.WeightStore.Morph(dt)
			active = l.WeightStore.GetActive(dt)
		}

		if active != nil {
			ls.Weights = encodeNativeWeights(active, dt)
			ls.Native = true
		} else if len(l.WeightStore.Master) > 0 {
			ls.Weights = encodeWeights(l.WeightStore.Master)
			ls.Native = false
		}
		ls.Scale = l.WeightStore.Scale
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

	// Initialize weights based on populated fields
	initializeWeights(l)

	if ls.Weights != "" {
		dt := DTypeFloat32
		if ls.Native {
			dt = l.DType
		}
		
		if l.WeightStore == nil {
			return fmt.Errorf("failed to initialize WeightStore for layer type %v", l.Type)
		}
		l.WeightStore.Scale = ls.Scale

		if ls.Native {
			decoded, err := decodeNativeWeights(ls.Weights, dt)
			if err != nil {
				return err
			}
			l.WeightStore.Versions[dt] = decoded
			l.WeightStore.Unpack(dt)
		} else {
			m, err := decodeWeights(ls.Weights)
			if err != nil {
				return err
			}
			l.WeightStore.Master = m
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

	return nil
}

// ParseSoftmaxType converts string to SoftmaxType.
func ParseSoftmaxType(s string) SoftmaxType {
	switch s {
	case "Standard": return SoftmaxStandard
	case "Grid": return SoftmaxGrid
	case "Hierarchical": return SoftmaxHierarchical
	case "Temperature": return SoftmaxTemperature
	case "Gumbel": return SoftmaxGumbel
	case "Masked": return SoftmaxMasked
	case "Sparse": return SoftmaxSparse
	case "Adaptive": return SoftmaxAdaptive
	case "Mixture": return SoftmaxMixture
	case "Entmax": return SoftmaxEntmax
	default: return SoftmaxStandard
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

// encodeNativeWeights converts an active weight slice to base64 string.
func encodeNativeWeights(data any, dt DType) string {
	switch w := data.(type) {
	case []float64:
		buf := make([]byte, len(w)*8)
		for i, v := range w { binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(v)) }
		return base64.StdEncoding.EncodeToString(buf)
	case []float32:
		return encodeWeights(w)
	case []int64:
		buf := make([]byte, len(w)*8)
		for i, v := range w { binary.LittleEndian.PutUint64(buf[i*8:], uint64(v)) }
		return base64.StdEncoding.EncodeToString(buf)
	case []int32:
		buf := make([]byte, len(w)*4)
		for i, v := range w { binary.LittleEndian.PutUint32(buf[i*4:], uint32(v)) }
		return base64.StdEncoding.EncodeToString(buf)
	case []int16:
		buf := make([]byte, len(w)*2)
		for i, v := range w { binary.LittleEndian.PutUint16(buf[i*2:], uint16(v)) }
		return base64.StdEncoding.EncodeToString(buf)
	case []int8:
		if dt == DTypeInt4 || dt == DTypeUint4 || dt == DTypeFP4 {
			// Pack 2 weights per byte
			buf := make([]byte, (len(w)+1)/2)
			for i, v := range w {
				val := byte(v & 0x0F)
				if i%2 == 0 { buf[i/2] |= (val << 4) } else { buf[i/2] |= val }
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
				if v > 0 { buf[i/8] |= (1 << uint(7-(i%8))) }
			}
			return base64.StdEncoding.EncodeToString(buf)
		}
		buf := make([]byte, len(w))
		for i, v := range w { buf[i] = byte(v) }
		return base64.StdEncoding.EncodeToString(buf)
	case []uint64:
		buf := make([]byte, len(w)*8)
		for i, v := range w { binary.LittleEndian.PutUint64(buf[i*8:], v) }
		return base64.StdEncoding.EncodeToString(buf)
	case []uint32:
		buf := make([]byte, len(w)*4)
		for i, v := range w { binary.LittleEndian.PutUint32(buf[i*4:], v) }
		return base64.StdEncoding.EncodeToString(buf)
	case []uint16:
		buf := make([]byte, len(w)*2)
		for i, v := range w { binary.LittleEndian.PutUint16(buf[i*2:], v) }
		return base64.StdEncoding.EncodeToString(buf)
	case []uint8:
		return base64.StdEncoding.EncodeToString(w)
	default:
		return ""
	}
}

// decodeNativeWeights converts base64 string to a slice of the given DType.
func decodeNativeWeights(s string, dt DType) (any, error) {
	bytes, err := base64.StdEncoding.DecodeString(s)
	if err != nil { return nil, err }

	switch dt {
	case DTypeFloat64:
		w := make([]float64, len(bytes)/8)
		for i := range w { w[i] = math.Float64frombits(binary.LittleEndian.Uint64(bytes[i*8:])) }
		return w, nil
	case DTypeFloat16, DTypeBFloat16:
		// Currently stored as float32 in our simulation, but 16-bit packed is future goal
		return decodeWeights(s)
	case DTypeInt64, DTypeUint64:
		w := make([]int64, len(bytes)/8)
		for i := range w { w[i] = int64(binary.LittleEndian.Uint64(bytes[i*8:])) }
		return w, nil
	case DTypeInt32, DTypeUint32:
		w := make([]int32, len(bytes)/4)
		for i := range w { w[i] = int32(binary.LittleEndian.Uint32(bytes[i*4:])) }
		return w, nil
	case DTypeInt16, DTypeUint16:
		w := make([]int16, len(bytes)/2)
		for i := range w { w[i] = int16(binary.LittleEndian.Uint16(bytes[i*2:])) }
		return w, nil
	case DTypeInt8, DTypeUint8, DTypeFP8E4M3, DTypeFP8E5M2:
		w := make([]int8, len(bytes))
		for i := range w { w[i] = int8(bytes[i]) }
		return w, nil
	case DTypeInt4, DTypeUint4, DTypeFP4:
		// Unpack 2 weights per byte into []int8
		w := make([]int8, len(bytes)*2)
		for i := range bytes {
			valHigh := (bytes[i] >> 4) & 0x0F
			valLow := bytes[i] & 0x0F
			
			// Sign extend if needed (approximate)
			sh := int8(valHigh)
			if sh > 7 { sh -= 16 }
			sl := int8(valLow)
			if sl > 7 { sl -= 16 }
			
			w[i*2] = sh
			if i*2+1 < len(w) { w[i*2+1] = sl }
		}
		return w, nil
	case DTypeInt2, DTypeUint2, DTypeTernary:
		// Unpack 4 weights per byte into []int8
		w := make([]int8, len(bytes)*4)
		for i := range bytes {
			for j := 0; j < 4; j++ {
				shift := uint(6 - j*2)
				val := (bytes[i] >> shift) & 0x03
				sv := int8(val)
				if sv > 1 { sv -= 4 }
				if i*4+j < len(w) { w[i*4+j] = sv }
			}
		}
		return w, nil
	case DTypeBinary:
		// Unpack 8 weights per byte into []int8
		w := make([]int8, len(bytes)*8)
		for i := range bytes {
			for j := 0; j < 8; j++ {
				bit := (bytes[i] >> uint(7-j)) & 0x01
				val := int8(-1)
				if bit == 1 { val = 1 }
				if i*8+j < len(w) { w[i*8+j] = val }
			}
		}
		return w, nil
	default:
		return bytes, nil
	}
}
