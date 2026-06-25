package poly

import (
	"runtime"
	"runtime/debug"
)

// ReleaseInferenceTransientMemory encourages the Go runtime to reclaim unused heap
// pages after inference mount steps (ReleaseInferenceHostWeights, entity decode, etc.).
// Call after each decoder block or global weight release during GPU/CPU entity mount.
func ReleaseInferenceTransientMemory() {
	runtime.GC()
	debug.FreeOSMemory()
}
