//go:build !linux || !cgo

package accel

import "fmt"

func defaultIntelPath() string {
	return ""
}

func intelNPUAvailable(path string) bool {
	return false
}

func openIntelPlugin(path, device string) (Plugin, error) {
	return nil, fmt.Errorf("%w (need linux + CGO_ENABLED=1)", ErrUnavailable)
}
