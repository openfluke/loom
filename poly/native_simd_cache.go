package poly

// native_simd_cache.go — cached weight views for native-exact SIMD (avoid per-forward decode).

type nativeSimdWeightCache struct {
	gen uint32
	f32 []float32
	i8  []int8
	u8  []uint8
}

func (ws *WeightStore) nativeSimdCaches() map[DType]*nativeSimdWeightCache {
	if ws.nativeSimdCaches_ == nil {
		ws.nativeSimdCaches_ = make(map[DType]*nativeSimdWeightCache)
	}
	return ws.nativeSimdCaches_
}

func (ws *WeightStore) invalidateNativeSimdCache(dtype DType) {
	ws.nativeSimdGen++
	if ws.nativeSimdCaches_ != nil {
		delete(ws.nativeSimdCaches_, dtype)
	}
}

func (ws *WeightStore) invalidateAllNativeSimdCache() {
	ws.nativeSimdGen++
	ws.nativeSimdCaches_ = nil
}

func (ws *WeightStore) nativeSimdCacheHit(c *nativeSimdWeightCache) bool {
	return c != nil && c.gen == ws.nativeSimdGen
}

// NativeSimdF32Weights returns cached float32 weights for native-exact SIMD MAC paths.
// Float32 uses Master directly; other dtypes decode once per Versions change.
func (ws *WeightStore) NativeSimdF32Weights(dtype DType) []float32 {
	if ws == nil {
		return nil
	}
	if dtype == DTypeFloat32 {
		return ws.Master
	}
	ws.Morph(dtype)
	src := ws.GetNative(dtype)
	if src == nil {
		return nil
	}
	caches := ws.nativeSimdCaches()
	if c := caches[dtype]; ws.nativeSimdCacheHit(c) && c.f32 != nil {
		return c.f32
	}
	f32 := buildNativeSimdF32(ws, dtype, src)
	caches[dtype] = &nativeSimdWeightCache{gen: ws.nativeSimdGen, f32: f32}
	return f32
}

func buildNativeSimdF32(ws *WeightStore, dtype DType, src any) []float32 {
	switch dtype {
	case DTypeFloat64:
		raw, ok := src.([]float64)
		if !ok {
			return nil
		}
		out := make([]float32, len(raw))
		for i, v := range raw {
			out[i] = float32(v)
		}
		return out
	case DTypeFloat16:
		raw, ok := src.([]uint16)
		if !ok {
			return nil
		}
		out := make([]float32, len(raw))
		for i, r := range raw {
			out[i] = float16ToFloat32(r)
		}
		return out
	case DTypeBFloat16:
		raw, ok := src.([]uint16)
		if !ok {
			return nil
		}
		out := make([]float32, len(raw))
		for i, r := range raw {
			out[i] = bfloat16ToFloat32(r)
		}
		return out
	case DTypeFP8E4M3:
		raw, ok := src.([]uint8)
		if !ok {
			return nil
		}
		out := make([]float32, len(raw))
		for i, r := range raw {
			out[i] = e4m3ToFloat32(r) * ws.Scale
		}
		return out
	case DTypeFP8E5M2:
		raw, ok := src.([]uint8)
		if !ok {
			return nil
		}
		out := make([]float32, len(raw))
		for i, r := range raw {
			out[i] = e5m2ToFloat32(r) * ws.Scale
		}
		return out
	case DTypeInt64:
		raw, ok := src.([]int64)
		if !ok {
			return nil
		}
		out := make([]float32, len(raw))
		for i, r := range raw {
			out[i] = float32(r) * ws.Scale
		}
		return out
	case DTypeUint64:
		raw, ok := src.([]uint64)
		if !ok {
			return nil
		}
		out := make([]float32, len(raw))
		for i, r := range raw {
			out[i] = float32(r) * ws.Scale
		}
		return out
	case DTypeInt32:
		raw, ok := src.([]int32)
		if !ok {
			return nil
		}
		out := make([]float32, len(raw))
		for i, r := range raw {
			out[i] = float32(r) * ws.Scale
		}
		return out
	case DTypeUint32:
		raw, ok := src.([]uint32)
		if !ok {
			return nil
		}
		out := make([]float32, len(raw))
		for i, r := range raw {
			out[i] = float32(r) * ws.Scale
		}
		return out
	case DTypeInt16:
		raw, ok := src.([]int16)
		if !ok {
			return nil
		}
		out := make([]float32, len(raw))
		for i, r := range raw {
			out[i] = float32(r) * ws.Scale
		}
		return out
	case DTypeUint16:
		raw, ok := src.([]uint16)
		if !ok {
			return nil
		}
		out := make([]float32, len(raw))
		for i, r := range raw {
			out[i] = float32(r) * ws.Scale
		}
		return out
	case DTypeInt8, DTypeUint8, DTypeInt4, DTypeUint4, DTypeInt2, DTypeUint2, DTypeTernary, DTypeBinary, DTypeFP4:
		codes, ok := nativeU8WeightsView(src)
		if !ok {
			return nil
		}
		out := make([]float32, len(codes))
		for i, val := range codes {
			if dtype == DTypeFP4 {
				out[i] = fp4CodeToFloat32(val, ws.Scale)
			} else if dtype == DTypeUint8 || dtype == DTypeUint4 || dtype == DTypeUint2 {
				out[i] = float32(val) * ws.Scale
			} else {
				out[i] = float32(int8(val)) * ws.Scale
			}
		}
		return out
	default:
		return CastWeights[float32](src)
	}
}

// NativeSimdI8Weights returns cached int8 weights for integer-native SIMD paths.
func (ws *WeightStore) NativeSimdI8Weights(dtype DType) []int8 {
	if ws == nil {
		return nil
	}
	ws.Morph(dtype)
	src := ws.GetNative(dtype)
	codes, ok := nativeU8WeightsView(src)
	if !ok {
		return nil
	}
	caches := ws.nativeSimdCaches()
	if c := caches[dtype]; ws.nativeSimdCacheHit(c) && c.i8 != nil {
		return c.i8
	}
	out := make([]int8, len(codes))
	for i, c := range codes {
		out[i] = trueNativeWeightI8(dtype, c)
	}
	entry := caches[dtype]
	if entry == nil || !ws.nativeSimdCacheHit(entry) {
		entry = &nativeSimdWeightCache{gen: ws.nativeSimdGen}
		caches[dtype] = entry
	}
	entry.i8 = out
	return out
}

// NativeSimdU8Weights returns cached uint8 weights for unsigned integer-native SIMD paths.
func (ws *WeightStore) NativeSimdU8Weights(dtype DType) []uint8 {
	if ws == nil {
		return nil
	}
	ws.Morph(dtype)
	src := ws.GetNative(dtype)
	codes, ok := nativeU8WeightsView(src)
	if !ok {
		return nil
	}
	caches := ws.nativeSimdCaches()
	if c := caches[dtype]; ws.nativeSimdCacheHit(c) && c.u8 != nil {
		return c.u8
	}
	out := make([]uint8, len(codes))
	switch dtype {
	case DTypeUint4:
		for i, c := range codes {
			if c > 15 {
				out[i] = 15
			} else {
				out[i] = c
			}
		}
	case DTypeUint2:
		for i, c := range codes {
			if c > 3 {
				out[i] = 3
			} else {
				out[i] = c
			}
		}
	default:
		copy(out, codes)
	}
	entry := caches[dtype]
	if entry == nil || !ws.nativeSimdCacheHit(entry) {
		entry = &nativeSimdWeightCache{gen: ws.nativeSimdGen}
		caches[dtype] = entry
	}
	entry.u8 = out
	return out
}
