package simd

// int8DotSimdForward is set by SetSimdForward / SetSimdForwardRecursive.
// Int8 dot and saxpy kernels run only when this flag and hardware SIMD are both on.
var int8DotSimdForward bool

// SetInt8DotSimdForward toggles DotI8Tile / SaxpyI8ScaleI32Acc kernels.
func SetInt8DotSimdForward(enabled bool) {
	int8DotSimdForward = enabled
}

func int8DotSimdEnabled() bool {
	return simdEnabled() && int8DotSimdForward
}

// Int8DotSimdActive reports whether native int8 vector dot/saxpy paths are on.
func Int8DotSimdActive() bool {
	return int8DotSimdEnabled()
}
