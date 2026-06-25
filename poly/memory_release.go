package poly

import (
	"runtime"
	"runtime/debug"
	"sync"
)

var memoryScavengerOnce sync.Once

// InitMemoryScavenger should run once at process/library bootstrap (ChaosInit, lucy main, etc.).
//
// RSS vs Jetsam on Darwin: by default Go returns freed heap pages with madvise(MADV_FREE),
// which keeps them mapped in your RSS until the kernel reclaims under memory pressure.
// Apple's Jetsam kills on RSS, not on "logical" free heap — so a spike during .entity mount
// can jetsam the app even after host weights are released.
//
// Fix (must be set before the Go runtime starts — os.Setenv from Go is too late):
//
//   - iOS/macOS host app: setenv("GODEBUG", "madvdontneed=1", …) in a C constructor or
//     before the first //export call (see mvp-simulation/flutter/ios/Runner/chaos_ffi_keep.c).
//   - Desktop lucy: GODEBUG=madvdontneed=1 go run .
//   - Linux: default is already MADV_DONTNEED; no env needed.
//   - Windows: madvise N/A; FreeOSMemory still helps after GC.
func InitMemoryScavenger() {
	memoryScavengerOnce.Do(func() {
		// Warm scavenger path; no-op on platforms without madvise semantics.
	})
}

// ReleaseInferenceTransientMemory forces GC and returns unused heap pages to the OS.
// Call after each decoder block or global weight release during GPU/CPU entity mount,
// and once more when mount completes.
//
// Pair with InitMemoryScavenger (madvdontneed=1) on iOS/macOS so RSS actually falls after this runs.
func ReleaseInferenceTransientMemory() {
	runtime.GC()
	debug.FreeOSMemory()
}

// AggressiveReleaseMemoryToOS runs an extra GC/scavenge pass for large transient drops
// (e.g. after full entity mount or CPU Q4 materialize). Safe on all GOOS; use sparingly.
func AggressiveReleaseMemoryToOS() {
	runtime.GC()
	debug.FreeOSMemory()
	runtime.GC()
	debug.FreeOSMemory()
}
