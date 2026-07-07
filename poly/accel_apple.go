package poly

import "github.com/openfluke/loom/poly/accel"

// DiscoverAppleAccel loads the Apple Metal/Accelerate plugin (Metal GPU via MPSGraph
// + a portable CPU reference backend as the parity anchor).
//
// The layer→bench-name mapping (intelLayerDesc), SyncToAccel, LayerWeightBytesForAccel
// and DispatchAccelForward in accel_intel.go are vendor-neutral — they route through
// Registry.PluginFor(ExecTarget), so the same machinery drives the Apple targets
// (ExecAppleCPU / ExecAppleGPU) once the registry is opened here.
func DiscoverAppleAccel(cfg accel.AccelConfig) (*accel.Registry, error) {
	return accel.DiscoverApple(cfg)
}
