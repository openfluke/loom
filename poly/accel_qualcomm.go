package poly

import "github.com/openfluke/loom/poly/accel"

// DiscoverQualcommAccel loads the Qualcomm QNN plugin (Hexagon HTP + QnnCpu reference).
//
// The layer→bench-name mapping (intelLayerDesc), SyncToAccel, LayerWeightBytesForAccel
// and DispatchAccelForward in accel_intel.go are vendor-neutral — they route through
// Registry.PluginFor(ExecTarget), so the same machinery drives Qualcomm targets
// (ExecQualcommCPU / ExecQualcommNPU) once the registry is opened here.
func DiscoverQualcommAccel(cfg accel.AccelConfig) (*accel.Registry, error) {
	return accel.DiscoverQualcomm(cfg)
}
