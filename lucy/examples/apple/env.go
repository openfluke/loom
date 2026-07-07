package apple

import "github.com/openfluke/loom/poly/accel"

// ensureAppleEnv is a no-op — Metal / MetalPerformanceShadersGraph / Accelerate ship
// with macOS, so there is no runtime search path to prepare (unlike OpenVINO or QNN).
func ensureAppleEnv() {
	_ = accel.PrepareAppleRuntime()
}
