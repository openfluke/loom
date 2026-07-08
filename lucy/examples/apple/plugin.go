package apple

import (
	"github.com/openfluke/loom/poly/accel"
)

func defaultPluginPath() string {
	return accel.DefaultApplePath()
}

// GPUReady reports whether the Apple plugin sees a Metal GPU device.
func GPUReady(path string) bool {
	return accel.AppleGPUAvailable(path)
}

// openPlugin opens the Apple plugin for device "CPU" (portable reference) or
// "GPU" (Metal / MPSGraph).
func openPlugin(device string) (accel.Plugin, error) {
	return accel.OpenApple(defaultPluginPath(), device)
}

func DefaultPluginPath() string {
	return defaultPluginPath()
}
