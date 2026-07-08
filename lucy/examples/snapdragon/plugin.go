package snapdragon

import (
	"github.com/openfluke/loom/poly/accel"
)

func defaultPluginPath() string {
	return accel.DefaultQualcommPath()
}

func NPUReady(path string) bool {
	return accel.QualcommNPUAvailable(path)
}

func openPlugin(device string) (accel.Plugin, error) {
	return accel.OpenQualcomm(defaultPluginPath(), device)
}
