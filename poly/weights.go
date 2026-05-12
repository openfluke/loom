package poly

import (
	"math"
	"math/rand"

	"github.com/openfluke/webgpu/wgpu"
)

// WeightStore manages multiple numerical representations of a layer's weights.
// In the current GPU-sovereign architecture, the VRAM-resident GPUWeights act
// as the primary execution state, while the Master and Versions serve as
// secondary sync points for persistence, I/O, and CPU fallbacks.
type WeightStore struct {
	Master     []float32              // Secondary sync body (Source of Truth for Persistence/IO)
	Versions   map[DType]any          // CPU-side cached versions (e.g., map[DTypeFP4][]byte)
	CPUPacked  map[DType]any          // CPU-side packed caches for exact low-bit kernels
	GPUWeights map[DType]any          // Primary Execution Store (VRAM-resident wgpu.Buffer)
	GPUScales  map[DType]*wgpu.Buffer // VRAM-resident scales for quantized types
	Scale      float32                // Dynamic quantization scale factor
}

func isCNN1NativeQuantDType(dtype DType) bool {
	switch dtype {
	case DTypeInt8, DTypeInt4, DTypeFP4, DTypeInt2, DTypeTernary, DTypeBinary,
		DTypeFP8E4M3, DTypeFP8E5M2, DTypeUint8, DTypeUint4, DTypeUint2,
		DTypeFloat16, DTypeBFloat16, DTypeInt16:
		return true
	default:
		return false
	}
}

var fp4DecodeTable = [16]float32{
	0.0, 0.75, 1.0, 1.5,
	2.0, 3.0, float32(math.Inf(1)), float32(math.NaN()),
	float32(math.Copysign(0, -1)), -0.75, -1.0, -1.5,
	-2.0, -3.0, float32(math.Inf(-1)), float32(math.NaN()),
}

var fp4FiniteCodes = [12]uint8{0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD}

func fp4CodeToFloat32(code uint8, scale float32) float32 {
	if scale == 0 {
		scale = 1.0
	}
	return fp4DecodeTable[code&0x0F] * scale
}

func nearestFP4Code(v float32, scale float32) uint8 {
	if scale == 0 {
		scale = 1.0
	}
	unscaled := v / scale
	bestCode := uint8(0)
	bestDiff := float32(math.Inf(1))
	for _, code := range fp4FiniteCodes {
		ref := fp4DecodeTable[code&0x0F]
		diff := float32(math.Abs(float64(unscaled - ref)))
		if diff < bestDiff {
			bestDiff = diff
			bestCode = code
		}
	}
	return bestCode
}

func decodeFP4Codes(codes []uint8, scale float32) []float32 {
	out := make([]float32, len(codes))
	for i, code := range codes {
		out[i] = fp4CodeToFloat32(code, scale)
	}
	return out
}

func nativeQuantValue(dtype DType, v float32) uint8 {
	switch dtype {
	case DTypeInt8, DTypeInt4, DTypeInt2, DTypeTernary:
		q := int(math.Round(float64(v)))
		return uint8(q) // Use uint8 as the byte container
	case DTypeBinary:
		if v >= 0 {
			return 1
		}
		return 255 // -1 for int8
	case DTypeUint8:
		q := int(math.Round(float64(v)))
		if q < 0 {
			q = 0
		}
		if q > 255 {
			q = 255
		}
		return uint8(q)
	case DTypeUint4:
		q := int(math.Round(float64(v)))
		if q < 0 {
			q = 0
		}
		if q > 15 {
			q = 15
		}
		return uint8(q)
	case DTypeUint2:
		q := int(math.Round(float64(v)))
		if q < 0 {
			q = 0
		}
		if q > 3 {
			q = 3
		}
		return uint8(q)
	default:
		return uint8(math.Round(float64(v)))
	}
}

// Float32 to FP8-E4M3 (Sign=1, Exp=4, Mant=3, Bias=7)
func float32ToE4M3(v float32) uint8 {
	f := math.Float32bits(v)
	sign := uint8((f >> 31) & 0x01)
	exp := int((f >> 23) & 0xFF)
	mant := f & 0x7FFFFF

	if f == 0 || f == 0x80000000 {
		return sign << 7
	}

	// Re-bias exponent
	e := exp - 127 + 7
	var res uint8
	if e <= 0 {
		// Subnormal or zero
		return sign << 7
	} else if e >= 15 {
		// Saturation to max finite value (OCP E4M3 has no Inf)
		res = (sign << 7) | (0xF << 3) | 0x7
	} else {
		res = (sign << 7) | (uint8(e) << 3) | uint8(mant>>20)
	}
	return res
}

func e4m3ToFloat32(v uint8) float32 {
	sign := uint32(v >> 7)
	exp := uint32((v >> 3) & 0x0F)
	mant := uint32(v & 0x07)

	if exp == 0 && mant == 0 {
		return math.Float32frombits(sign << 31)
	}
	if exp == 0 {
		// Subnormal approximation
		return math.Float32frombits((sign<<31)|(127-7)<<23|(mant<<20)) * 0.125
	}

	resExp := exp + 127 - 7
	return math.Float32frombits((sign << 31) | (resExp << 23) | (mant << 20))
}

// Float32 to FP8-E5M2 (Sign=1, Exp=5, Mant=2, Bias=15)
func float32ToE5M2(v float32) uint8 {
	f := math.Float32bits(v)
	sign := uint8((f >> 31) & 0x01)
	exp := int((f >> 23) & 0xFF)
	mant := f & 0x7FFFFF

	if f == 0 || f == 0x80000000 {
		return sign << 7
	}

	e := exp - 127 + 15
	if e <= 0 {
		return sign << 7
	} else if e >= 31 {
		// Infinity
		return (sign << 7) | (0x1F << 2)
	}
	return (sign << 7) | (uint8(e) << 2) | uint8(mant>>21)
}

func e5m2ToFloat32(v uint8) float32 {
	sign := uint32(v >> 7)
	exp := uint32((v >> 2) & 0x1F)
	mant := uint32(v & 0x03)

	if exp == 0 && mant == 0 {
		return math.Float32frombits(sign << 31)
	}
	if exp == 31 {
		if mant == 0 {
			return float32(math.Inf(int(1 - 2*sign)))
		}
		return float32(math.NaN())
	}

	resExp := exp + 127 - 15
	return math.Float32frombits((sign << 31) | (resExp << 23) | (mant << 21))
}

// Float16/BFloat16 approximations
func float32ToFloat16(v float32) uint16 {
	f := math.Float32bits(v)
	sign := uint16((f >> 31) & 0x1)
	exp := int((f >> 23) & 0xFF)
	mant := f & 0x7FFFFF

	if f == 0 || f == 0x80000000 {
		return sign << 15
	}

	e := exp - 127 + 15
	if e <= 0 {
		return sign << 15
	} else if e >= 31 {
		return (sign << 15) | (0x1F << 10)
	}
	return (sign << 15) | (uint16(e) << 10) | uint16(mant>>13)
}

func float32ToBFloat16(v float32) uint16 {
	return uint16(math.Float32bits(v) >> 16)
}

// Morph prepares a CPU-side quantized version of the weights. Note that for
// active training, quantization increasingly occurs directly in VRAM via
// GPU shaders, bypassing this CPU-bound path.
func (ws *WeightStore) Morph(dtype DType) {
	ws.morph(dtype, false)
}

// ForceMorph re-quantizes from the secondary Master slice. Use this after
// externally modifying Master or restoring from a high-precision snapshot.
func (ws *WeightStore) ForceMorph(dtype DType) {
	delete(ws.Versions, dtype)
	if ws.CPUPacked != nil {
		delete(ws.CPUPacked, dtype)
	}
	ws.morph(dtype, false)
}

// InvalidateVersions clears the CPU and GPU weight caches. This forces the engine
// to re-synchronize from the Master body or re-initialize its GPU state.
func (ws *WeightStore) InvalidateVersions() {
	ws.Versions = make(map[DType]any)
	ws.CPUPacked = make(map[DType]any)
	ws.GPUWeights = make(map[DType]any)
}

func (ws *WeightStore) morph(dtype DType, force bool) {
	if dtype == DTypeFloat32 {
		return
	}

	// Return cached version unless forced
	if !force {
		if _, ok := ws.Versions[dtype]; ok {
			return
		}
	}

	size := len(ws.Master)

	// Auto-Dynamic Scaling: If scale is 1.0 (unset), find the optimal range.
	// This ensures INT8 uses the full -128 to 127 range.
	if ws.Scale == 1.0 && size > 0 {
		maxAbs := float32(0)
		sumAbs := float64(0)
		for _, v := range ws.Master {
			a := float32(math.Abs(float64(v)))
			if a > maxAbs {
				maxAbs = a
			}
			sumAbs += float64(a)
		}

		// Use the dtype's actual max representable integer as the divisor so
		// that the full weight range maps to the full quantization range without
		// clipping.  Using 127 for Int4 (max=7) would clip 94% of values.
		var maxInt float32
		switch dtype {
		case DTypeInt4, DTypeUint4:
			maxInt = 7.0
		case DTypeFP4:
			maxInt = 3.0
		case DTypeInt2, DTypeUint2, DTypeTernary:
			maxInt = 1.0
		case DTypeBinary:
			// Binary uses mean absolute value as the representative scale.
			ws.Scale = float32(sumAbs) / float32(size)
			if ws.Scale == 0 {
				ws.Scale = 1.0
			}
			goto morphSwitch
		default:
			// Int8, Uint8, FP8, Int16..Int64, Float16, BFloat16, Float64
			maxInt = 127.0
		}
		if maxAbs == 0 {
			ws.Scale = 1.0
		} else {
			ws.Scale = maxAbs / maxInt
		}
	}
morphSwitch:

	switch dtype {
	case DTypeFloat64:
		w := make([]float64, size)
		for i, v := range ws.Master {
			w[i] = float64(v)
		}
		ws.Versions[dtype] = w
	case DTypeFloat16:
		w := make([]uint16, size)
		for i, v := range ws.Master {
			w[i] = float32ToFloat16(v)
		}
		ws.Versions[dtype] = w
	case DTypeBFloat16:
		w := make([]uint16, size)
		for i, v := range ws.Master {
			w[i] = float32ToBFloat16(v)
		}
		ws.Versions[dtype] = w
	case DTypeInt64:
		w := make([]int64, size)
		for i, v := range ws.Master {
			w[i] = int64(math.Round(float64(v / ws.Scale)))
		}
		ws.Versions[dtype] = w
	case DTypeUint64:
		w := make([]uint64, size)
		for i, v := range ws.Master {
			w[i] = uint64(math.Abs(math.Round(float64(v / ws.Scale))))
		}
		ws.Versions[dtype] = w
	case DTypeInt32:
		w := make([]int32, size)
		for i, v := range ws.Master {
			w[i] = int32(math.Round(float64(v / ws.Scale)))
		}
		ws.Versions[dtype] = w
	case DTypeUint32:
		w := make([]uint32, size)
		for i, v := range ws.Master {
			w[i] = uint32(math.Abs(math.Round(float64(v / ws.Scale))))
		}
		ws.Versions[dtype] = w
	case DTypeInt16:
		w := make([]int16, size)
		for i, v := range ws.Master {
			w[i] = int16(math.Round(float64(v / ws.Scale)))
		}
		ws.Versions[dtype] = w
	case DTypeUint16:
		w := make([]uint16, size)
		for i, v := range ws.Master {
			w[i] = uint16(math.Abs(math.Round(float64(v / ws.Scale))))
		}
		ws.Versions[dtype] = w
	case DTypeFP8E4M3:
		w := make([]uint8, size)
		for i, v := range ws.Master {
			w[i] = float32ToE4M3(v / ws.Scale)
		}
		ws.Versions[dtype] = w
	case DTypeFP8E5M2:
		w := make([]uint8, size)
		for i, v := range ws.Master {
			w[i] = float32ToE5M2(v / ws.Scale)
		}
		ws.Versions[dtype] = w
	case DTypeInt8, DTypeUint8, DTypeInt4, DTypeUint4, DTypeInt2, DTypeUint2, DTypeTernary, DTypeBinary, DTypeFP4:
		if dtype == DTypeFP4 {
			w := make([]uint8, size)
			for i, v := range ws.Master {
				w[i] = nearestFP4Code(v, ws.Scale)
			}
			ws.Versions[dtype] = w
			return
		}
		w := make([]uint8, size)
		for i, v := range ws.Master {
			w[i] = nativeQuantValue(dtype, v/ws.Scale)
		}
		ws.Versions[dtype] = w
	}
}

// NewWeightStore creates a new storage for weights.
func NewWeightStore(size int) *WeightStore {
	return &WeightStore{
		Master:     AlignedFloat32(size),
		Versions:   make(map[DType]any),
		CPUPacked:  make(map[DType]any),
		GPUWeights: make(map[DType]any),
		GPUScales:  make(map[DType]*wgpu.Buffer),
		Scale:      1.0,
	}
}

// Randomize fills the master weights with small random values to break symmetry.
func (ws *WeightStore) Randomize(seed int64, scale float32) {
	r := rand.New(rand.NewSource(seed))
	for i := range ws.Master {
		ws.Master[i] = (r.Float32()*2 - 1) * scale // Random values between -scale and scale
	}
	// Clear stale versions
	ws.Versions = make(map[DType]any)
	ws.CPUPacked = make(map[DType]any)
	ws.GPUWeights = make(map[DType]any)
}

// HeRandomize initializes weights using He initialization (Kaiming Normal).
func (ws *WeightStore) HeRandomize(seed int64, inputSize int) {
	r := rand.New(rand.NewSource(seed))
	stddev := float32(math.Sqrt(2.0 / float64(inputSize)))
	for i := range ws.Master {
		ws.Master[i] = float32(r.NormFloat64()) * stddev
	}
	// Clear stale versions
	ws.Versions = make(map[DType]any)
	ws.CPUPacked = make(map[DType]any)
	ws.GPUWeights = make(map[DType]any)
}

func (ws *WeightStore) GetActive(dtype DType) any {
	if dtype == DTypeFloat32 {
		return ws.Master
	}
	v, ok := ws.Versions[dtype]
	if !ok {
		ws.Morph(dtype)
		v = ws.Versions[dtype]
	}
	if v == nil {
		return nil
	}

	switch dtype {
	case DTypeFloat64:
		return v
	case DTypeFloat16:
		if raw, ok := v.([]uint16); ok {
			f := make([]float32, len(raw))
			for i, r := range raw {
				f[i] = float16ToFloat32(r)
			}
			return f
		}
	case DTypeBFloat16:
		if raw, ok := v.([]uint16); ok {
			f := make([]float32, len(raw))
			for i, r := range raw {
				f[i] = bfloat16ToFloat32(r)
			}
			return f
		}
	case DTypeFP8E4M3:
		if raw, ok := v.([]uint8); ok {
			f := make([]float32, len(raw))
			for i, r := range raw {
				f[i] = e4m3ToFloat32(r) * ws.Scale
			}
			return f
		}
	case DTypeFP8E5M2:
		if raw, ok := v.([]uint8); ok {
			f := make([]float32, len(raw))
			for i, r := range raw {
				f[i] = e5m2ToFloat32(r) * ws.Scale
			}
			return f
		}
	case DTypeInt8, DTypeUint8, DTypeInt4, DTypeUint4, DTypeInt2, DTypeUint2, DTypeTernary, DTypeBinary, DTypeFP4:
		raw, ok := v.([]uint8)
		if !ok || len(raw) == 0 {
			return nil
		}
		out := make([]float32, len(raw))
		for i, val := range raw {
			if dtype == DTypeFP4 {
				out[i] = fp4CodeToFloat32(val, ws.Scale)
			} else if dtype == DTypeUint8 || dtype == DTypeUint4 || dtype == DTypeUint2 {
				out[i] = float32(val) * ws.Scale
			} else {
				// Signed types: treat uint8 as int8
				out[i] = float32(int8(val)) * ws.Scale
			}
		}
		return out
	}
	return v
}

// GetNative returns the raw active representation for dtype without applying
// the PTQ simulation scale. This is the direct-on-quant path used by exact
// low-bit CNN1 execution and training.
func (ws *WeightStore) GetNative(dtype DType) any {
	if dtype == DTypeFloat32 {
		return ws.Master
	}
	if _, ok := ws.Versions[dtype]; !ok {
		ws.Morph(dtype)
	}
	return ws.Versions[dtype]
}

// GetNativePackedCPU returns a CPU-side packed cache for exact low-bit kernels.
// The packed cache is derived from the current native representation rather than
// the FP32 master so it remains faithful to the exact-dtype execution path.
func (ws *WeightStore) GetNativePackedCPU(dtype DType) any {
	if ws == nil {
		return nil
	}
	if ws.CPUPacked == nil {
		ws.CPUPacked = make(map[DType]any)
	}
	if packed, ok := ws.CPUPacked[dtype]; ok {
		return packed
	}

	ws.Morph(dtype)

	var p any
	switch dtype {
	case DTypeFloat16, DTypeBFloat16, DTypeInt16, DTypeUint16:
		switch v := ws.Versions[dtype].(type) {
		case []uint16:
			p = pack16BitToU32(v)
		case []int16:
			p = pack16BitToU32(v)
		}
	case DTypeInt32, DTypeUint32:
		switch v := ws.Versions[dtype].(type) {
		case []int32:
			p = pack32BitToU32(v)
		case []uint32:
			p = pack32BitToU32(v)
		}
	case DTypeInt64, DTypeUint64:
		switch v := ws.Versions[dtype].(type) {
		case []int64:
			p = pack64BitToU32(v)
		case []uint64:
			p = pack64BitToU32(v)
		}
	case DTypeInt8, DTypeUint8, DTypeFP8E4M3, DTypeFP8E5M2:
		switch v := ws.Versions[dtype].(type) {
		case []uint8:
			p = pack8BitToU32(v)
		case []int8:
			p = packSignedBytesToU32(v)
		}
	case DTypeInt4, DTypeUint4, DTypeFP4:
		if v, ok := ws.Versions[dtype].([]uint8); ok {
			p = packUnsignedNibblesToU32(v)
		}
	case DTypeInt2, DTypeUint2, DTypeTernary:
		if v, ok := ws.Versions[dtype].([]uint8); ok {
			if dtype == DTypeTernary {
				p = packTernaryToU32(v)
			} else {
				p = packUnsigned2ToU32(v)
			}
		}
	case DTypeBinary:
		if v, ok := ws.Versions[dtype].([]uint8); ok {
			p = packBinaryToU32(v)
		}
	default:
		return nil
	}
	ws.CPUPacked[dtype] = p
	return p
}

// MorphToFloat32ForGPU returns a float32 slice that represents the master weights
// at the precision of dtype. For Float32/Float64 it returns the master slice directly.
// For all other types it morphs to that dtype and converts back to float32 with the
// scale factor applied — giving GPU kernels "quantization-simulated" weights without
// needing new shaders.  This is the PTQ simulation upload path.
func (ws *WeightStore) MorphToFloat32ForGPU(dtype DType) []float32 {
	if dtype == DTypeFloat32 || dtype == DTypeFloat64 {
		return ws.Master
	}
	ws.Morph(dtype)
	version := ws.Versions[dtype]
	if version == nil {
		return ws.Master
	}
	if dtype == DTypeFP4 {
		if codes, ok := version.([]uint8); ok {
			return decodeFP4Codes(codes, ws.Scale)
		}
	}
	raw := CastWeights[float32](version)
	if raw == nil {
		return ws.Master
	}
	out := make([]float32, len(raw))
	scale := ws.Scale
	if scale == 0 {
		scale = 1.0
	}
	for i, v := range raw {
		out[i] = v * scale
	}
	return out
}

func (ws *WeightStore) syncMasterFromNative(dtype DType) {
	data := ws.GetNative(dtype)
	if data == nil {
		return
	}
	if dtype == DTypeFP4 {
		if codes, ok := data.([]uint8); ok {
			decoded := decodeFP4Codes(codes, ws.Scale)
			limit := len(decoded)
			if limit > len(ws.Master) {
				limit = len(ws.Master)
			}
			copy(ws.Master[:limit], decoded[:limit])
		}
		return
	}
	raw := CastWeights[float32](data)
	if raw == nil {
		return
	}
	scale := ws.Scale
	if scale == 0 {
		scale = 1.0
	}
	limit := len(raw)
	if limit > len(ws.Master) {
		limit = len(ws.Master)
	}
	for i := 0; i < limit; i++ {
		ws.Master[i] = raw[i] * scale
	}
}

// SetVersion stores a converted version of weights.
func (ws *WeightStore) SetVersion(dtype DType, data any) {
	ws.Versions[dtype] = data
	if ws.CPUPacked != nil {
		delete(ws.CPUPacked, dtype)
	}
}

// Int8Slice returns a signed int8 view of the requested version.
// Some morph paths store 8-bit data as []uint8; for INT8 GPU upload we
// preserve underlying bit patterns and reinterpret via explicit conversion.
func (ws *WeightStore) Int8Slice(dtype DType) []int8 {
	v := ws.Versions[dtype]
	switch data := v.(type) {
	case []int8:
		return data
	case []uint8:
		out := make([]int8, len(data))
		for i, b := range data {
			out[i] = int8(b)
		}
		return out
	default:
		return CastWeights[int8](v)
	}
}

// SizeInBytes calculates the memory footprint of the currently active version.
func (ws *WeightStore) SizeInBytes(dtype DType) int {
	count := len(ws.Master)
	if count == 0 && dtype == DTypeTernary && len(ws.CPUPacked) > 0 {
		total := 0
		for _, packed := range ws.CPUPacked {
			if matrix, ok := packed.(*BitNetTernaryMatrix); ok && matrix != nil {
				total += len(matrix.Words) * 4
			}
		}
		if total > 0 {
			return total
		}
	}
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
		return (count + 7) / 8 // 8 weights per byte (standard estimation)
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
	if data == nil {
		return
	}

	switch dtype {
	case DTypeFloat64:
		if w, ok := data.([]float64); ok {
			for i, v := range w {
				ws.Master[i] = float32(v)
			}
		}
	case DTypeFloat32:
		if w, ok := data.([]float32); ok {
			copy(ws.Master, w)
		}
	case DTypeFloat16:
		if w, ok := data.([]uint16); ok {
			for i, v := range w {
				ws.Master[i] = float16ToFloat32(v)
			}
		}
	case DTypeBFloat16:
		if w, ok := data.([]uint16); ok {
			for i, v := range w {
				ws.Master[i] = bfloat16ToFloat32(v)
			}
		}
	case DTypeFP4:
		if w, ok := data.([]uint8); ok {
			decoded := decodeFP4Codes(w, ws.Scale)
			copy(ws.Master, decoded)
		}
	case DTypeFP8E4M3:
		if w, ok := data.([]uint8); ok {
			for i, v := range w {
				ws.Master[i] = e4m3ToFloat32(v) * ws.Scale
			}
		}
	case DTypeFP8E5M2:
		if w, ok := data.([]uint8); ok {
			for i, v := range w {
				ws.Master[i] = e5m2ToFloat32(v) * ws.Scale
			}
		}
	case DTypeInt64, DTypeUint64, DTypeInt32, DTypeUint32, DTypeInt16, DTypeUint16, DTypeUint8, DTypeUint4, DTypeUint2:
		packed := CastWeights[float32](data)
		for i := 0; i < len(ws.Master) && i < len(packed); i++ {
			ws.Master[i] = packed[i] * ws.Scale
		}
	case DTypeInt8, DTypeInt4, DTypeInt2, DTypeTernary, DTypeBinary:
		scale := ws.Scale
		if scale == 0 {
			scale = 1.0
		}
		switch packed := data.(type) {
		case []uint8:
			for i := 0; i < len(ws.Master) && i < len(packed); i++ {
				ws.Master[i] = float32(int8(packed[i])) * scale
			}
		case []int8:
			for i := 0; i < len(ws.Master) && i < len(packed); i++ {
				ws.Master[i] = float32(packed[i]) * scale
			}
		default:
			converted := CastWeights[float32](data)
			for i := 0; i < len(ws.Master) && i < len(converted); i++ {
				ws.Master[i] = converted[i] * scale
			}
		}
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
func (ws *WeightStore) ApplyGradients(gradWeights *Tensor[float32], lr float32, clipVal float32) {
	if gradWeights == nil || len(gradWeights.Data) == 0 {
		return
	}
	limit := len(gradWeights.Data)
	if len(ws.Master) < limit {
		limit = len(ws.Master)
	}

	for i := 0; i < limit; i++ {
		g := gradWeights.Data[i]
		if clipVal > 0 {
			if g > clipVal {
				g = clipVal
			}
			if g < -clipVal {
				g = -clipVal
			}
		}
		ws.Master[i] -= lr * g
	}
	// After applying gradients, previously cached low-bit versions are now STALE.
	// We clear them so the next Forward pass forces a "Metamorphosis" re-quantization.
	ws.Versions = make(map[DType]any)
	ws.CPUPacked = make(map[DType]any)
	ws.GPUWeights = make(map[DType]any)
}

func (ws *WeightStore) ApplyGradientsNative(dtype DType, gradWeights *Tensor[float32], lr float32, clipVal float32) bool {
	if !isCNN1NativeQuantDType(dtype) || gradWeights == nil || len(gradWeights.Data) == 0 {
		return false
	}

	ws.Morph(dtype)
	switch dtype {
	case DTypeFP4:
		raw, ok := ws.Versions[dtype].([]uint8)
		if !ok || len(raw) == 0 {
			return false
		}

		limit := len(gradWeights.Data)
		if len(raw) < limit {
			limit = len(raw)
		}
		for i := 0; i < limit; i++ {
			g := gradWeights.Data[i]
			if clipVal > 0 {
				if g > clipVal {
					g = clipVal
				}
				if g < -clipVal {
					g = -clipVal
				}
			}
			ws.Master[i] -= lr * g
			raw[i] = nearestFP4Code(ws.Master[i], ws.Scale)
		}
		ws.Versions[dtype] = raw
		if ws.CPUPacked != nil {
			delete(ws.CPUPacked, dtype)
		}
		ws.GPUWeights = make(map[DType]any)
		return true

	case DTypeFP8E4M3, DTypeFP8E5M2:
		raw, ok := ws.Versions[dtype].([]uint8)
		if !ok || len(raw) == 0 {
			return false
		}
		limit := len(gradWeights.Data)
		if len(raw) < limit {
			limit = len(raw)
		}
		for i := 0; i < limit; i++ {
			g := gradWeights.Data[i]
			if clipVal > 0 {
				if g > clipVal {
					g = clipVal
				}
				if g < -clipVal {
					g = -clipVal
				}
			}
			ws.Master[i] -= lr * g
			if dtype == DTypeFP8E4M3 {
				raw[i] = float32ToE4M3(ws.Master[i] / ws.Scale)
			} else {
				raw[i] = float32ToE5M2(ws.Master[i] / ws.Scale)
			}
		}
		ws.Versions[dtype] = raw
		ws.GPUWeights = make(map[DType]any)
		return true

	case DTypeFloat16, DTypeBFloat16:
		raw, ok := ws.Versions[dtype].([]uint16)
		if !ok || len(raw) == 0 {
			return false
		}
		limit := len(gradWeights.Data)
		if len(raw) < limit {
			limit = len(raw)
		}
		for i := 0; i < limit; i++ {
			g := gradWeights.Data[i]
			if clipVal > 0 {
				if g > clipVal {
					g = clipVal
				}
				if g < -clipVal {
					g = -clipVal
				}
			}
			ws.Master[i] -= lr * g
			if dtype == DTypeFloat16 {
				raw[i] = float32ToFloat16(ws.Master[i])
			} else {
				raw[i] = float32ToBFloat16(ws.Master[i])
			}
		}
		ws.Versions[dtype] = raw
		ws.GPUWeights = make(map[DType]any)
		return true

	default:
		raw, ok := ws.Versions[dtype].([]uint8)
		if !ok || len(raw) == 0 {
			return false
		}

		scale := ws.Scale
		if scale == 0 {
			scale = 1.0
		}

		limit := len(gradWeights.Data)
		if len(raw) < limit {
			limit = len(raw)
		}

		for i := 0; i < limit; i++ {
			g := gradWeights.Data[i]
			if clipVal > 0 {
				if g > clipVal {
					g = clipVal
				}
				if g < -clipVal {
					g = -clipVal
				}
			}
			ws.Master[i] -= lr * g
			raw[i] = nativeQuantValue(dtype, ws.Master[i]/scale)
		}

		ws.Versions[dtype] = raw
		if ws.CPUPacked != nil {
			delete(ws.CPUPacked, dtype)
		}
		ws.GPUWeights = make(map[DType]any)
		return true
	}
}

// ReleaseInferenceHostWeights drops CPU-side Master/Versions/CPUPacked after GPUWeights are
// populated. Intended for inference-only paths so VRAM holds the active weights.
func (ws *WeightStore) ReleaseInferenceHostWeights() {
	if ws == nil || len(ws.GPUWeights) == 0 {
		return
	}
	ws.Master = nil
	ws.Versions = nil
	ws.CPUPacked = nil
}

// Release explicitly destroys all WGPU weight and scale buffers.
func (ws *WeightStore) Release() {
	for dt, buf := range ws.GPUWeights {
		if b, ok := buf.(*wgpu.Buffer); ok && b != nil {
			b.Destroy()
		}
		delete(ws.GPUWeights, dt)
	}
	for dt, b := range ws.GPUScales {
		if b != nil {
			b.Destroy()
		}
		delete(ws.GPUScales, dt)
	}
}
