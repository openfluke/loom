package poly

import (
	"math/rand"
)

// WeightStore manages multiple numerical versions of the same weights.
// This is the core of "Polymorphic Layer-Morphing".
type WeightStore struct {
	Master   []float32     // Master FP32 weights (Source of Truth)
	Versions map[DType]any // Active versions (e.g., map[DTypeFP4][]byte)
	Scale    float32       // Quantization scale factor
}

// Morph converts master weights into the target DType and caches the result.
func (ws *WeightStore) Morph(dtype DType) {
	if dtype == DTypeFloat32 {
		return
	}
	
	// Check if already morphed
	if _, ok := ws.Versions[dtype]; ok {
		return
	}

	size := len(ws.Master)
	
	switch dtype {
	case DTypeFloat64:
		w := make([]float64, size)
		for i, v := range ws.Master { w[i] = float64(v) }
		ws.Versions[dtype] = w
	case DTypeFloat16, DTypeBFloat16:
		// Simulated 16-bit storage (using float32 but masked)
		w := make([]float32, size)
		for i, v := range ws.Master { w[i] = SimulatePrecision(v, dtype, ws.Scale) }
		ws.Versions[dtype] = w
	case DTypeInt64, DTypeUint64:
		w := make([]int64, size)
		for i, v := range ws.Master { w[i] = int64(v/ws.Scale) }
		ws.Versions[dtype] = w
	case DTypeInt32, DTypeUint32:
		w := make([]int32, size)
		for i, v := range ws.Master { w[i] = int32(v/ws.Scale) }
		ws.Versions[dtype] = w
	case DTypeInt16, DTypeUint16:
		w := make([]int16, size)
		for i, v := range ws.Master { w[i] = int16(v/ws.Scale) }
		ws.Versions[dtype] = w
	case DTypeInt8, DTypeUint8, DTypeFP8E4M3, DTypeFP8E5M2:
		w := make([]int8, size)
		for i, v := range ws.Master { w[i] = int8(v/ws.Scale) }
		ws.Versions[dtype] = w
	case DTypeInt4, DTypeUint4, DTypeFP4, DTypeInt2, DTypeUint2, DTypeTernary:
		// Store as unpacked []int8 in RAM for layer compatibility
		w := make([]int8, size)
		for i, v := range ws.Master { w[i] = int8(v/ws.Scale) }
		ws.Versions[dtype] = w
	case DTypeBinary:
		// Store as unpacked []int8 (-1 or 1) in RAM
		w := make([]int8, size)
		for i, v := range ws.Master {
			if v > 0 { w[i] = 1 } else { w[i] = -1 }
		}
		ws.Versions[dtype] = w
	}
}

// NewWeightStore creates a new storage for weights.
func NewWeightStore(size int) *WeightStore {
	return &WeightStore{
		Master:   AlignedFloat32(size),
		Versions: make(map[DType]any),
		Scale:    1.0,
	}
}

// Randomize fills the master weights with small random values to break symmetry.
func (ws *WeightStore) Randomize(seed int64) {
	r := rand.New(rand.NewSource(seed))
	for i := range ws.Master {
		ws.Master[i] = (r.Float32()*2 - 1) // Random values between -1.0 and 1.0
	}
	// Clear stale versions
	ws.Versions = make(map[DType]any)
}

// GetActive returns the data for the given DType if it exists.
func (ws *WeightStore) GetActive(dtype DType) any {
	return ws.Versions[dtype]
}

// SetVersion stores a converted version of weights.
func (ws *WeightStore) SetVersion(dtype DType, data any) {
	ws.Versions[dtype] = data
}

// SizeInBytes calculates the memory footprint of the currently active version.
func (ws *WeightStore) SizeInBytes(dtype DType) int {
	count := len(ws.Master)
	switch dtype {
	case DTypeFloat64, DTypeInt64, DTypeUint64:
		return count * 8
	case DTypeFloat32, DTypeInt32, DTypeUint32:
		return count * 4
	case DTypeFloat16, DTypeBFloat16, DTypeInt16, DTypeUint16:
		return count * 2
	case DTypeInt8, DTypeUint8, DTypeFP8E4M3, DTypeFP8E5M2:
		return count * 1
	case DTypeInt4, DTypeUint4, DTypeFP4, DTypeInt2, DTypeUint2, DTypeTernary:
		// Grouping sub-byte types for estimation
		if dtype == DTypeInt2 || dtype == DTypeUint2 || dtype == DTypeTernary {
			return (count + 3) / 4 // 4 weights per byte
		}
		return (count + 1) / 2 // 2 weights per byte
	case DTypeBinary:
		return (count + 7) / 8 // 8 weights per byte
	default:
		return count * 4
	}
}

// CastWeights is a universal utility to extract and cast weight slices from the polymorphic WeightStore.
// It is the "Universal Converter" that allows any layer type (Dense, CNN, MHA) to access weights
// in their required numeric type on-the-fly.
func CastWeights[T Numeric](weights any) []T {
	if weights == nil {
		return nil
	}
	switch w := weights.(type) {
	case []float64:
		return ConvertSlice[float64, T](w)
	case []float32:
		return ConvertSlice[float32, T](w)
	case []int64:
		return ConvertSlice[int64, T](w)
	case []int32:
		return ConvertSlice[int32, T](w)
	case []int16:
		return ConvertSlice[int16, T](w)
	case []int8:
		return ConvertSlice[int8, T](w)
	case []uint64:
		return ConvertSlice[uint64, T](w)
	case []uint32:
		return ConvertSlice[uint32, T](w)
	case []uint16:
		return ConvertSlice[uint16, T](w)
	case []uint8:
		return ConvertSlice[uint8, T](w)
	default:
		// Attempt to handle byte-packed versions (FP4, Binary, etc.)
		if b, ok := weights.([]byte); ok {
			res := make([]T, len(b))
			for i, v := range b {
				res[i] = T(v)
			}
			return res
		}
		return nil
	}
}

// Unpack reconstructs master weights from a bit-packed native version.
func (ws *WeightStore) Unpack(dtype DType) {
	data := ws.Versions[dtype]
	if data == nil { return }
	
	switch dtype {
	case DTypeFloat64:
		if w, ok := data.([]float64); ok {
			for i, v := range w { ws.Master[i] = float32(v) }
		}
	case DTypeFloat32, DTypeFloat16, DTypeBFloat16:
		if w, ok := data.([]float32); ok {
			copy(ws.Master, w)
		}
	case DTypeInt64, DTypeUint64, DTypeInt32, DTypeUint32, DTypeInt16, DTypeUint16, DTypeInt8, DTypeUint8, DTypeFP8E4M3, DTypeFP8E5M2, DTypeInt4, DTypeUint4, DTypeFP4, DTypeInt2, DTypeUint2, DTypeTernary, DTypeBinary:
		// These are already unpacked slices (int8/int16/etc) in Versions during I/O
		ws.Master = CastWeights[float32](data)
		for i := range ws.Master { ws.Master[i] *= ws.Scale }
	}
}

// convertSlice is a private helper for the CastWeights generic engine.
func ConvertSlice[In Numeric, Out Numeric](in []In) []Out {
	out := make([]Out, len(in))
	for i, v := range in {
		out[i] = Out(v)
	}
	return out
}

// ApplyGradients performs a simple SGD update (weight = weight - lr * gradient).
// This is the "Learning" step that mutates the actual weights in the Master store.
func (ws *WeightStore) ApplyGradients(gradWeights *Tensor[float32], lr float32) {
	if gradWeights == nil || len(gradWeights.Data) != len(ws.Master) {
		return
	}
	for i := range ws.Master {
		ws.Master[i] -= lr * gradWeights.Data[i]
	}
	// After applying gradients, previously cached low-bit versions are now STALE.
	// We clear them so the next Forward pass forces a "Metamorphosis" re-quantization.
	ws.Versions = make(map[DType]any)
}
