package ninelayer

import (
	"github.com/openfluke/loom/poly/accel"
)

func defaultPluginPath() string {
	return accel.DefaultIntelPath()
}

func NPUReady(path string) bool {
	return accel.NPUAvailable(path)
}

func openPlugin(device string) (accel.Plugin, error) {
	return accel.OpenIntel(defaultPluginPath(), device)
}
