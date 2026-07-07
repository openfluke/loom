//go:build !windows || !cgo

package accel

import "fmt"

// PrepareQualcommRuntime is a no-op off Windows (the QNN plugin is Windows-on-Snapdragon).
func PrepareQualcommRuntime() error { return nil }

func defaultQualcommPath() string { return "" }

func qualcommNPUAvailable(path string) bool { return false }

func openQualcommPlugin(path, device string) (Plugin, error) {
	return nil, fmt.Errorf("%w (need windows/arm64 + CGO_ENABLED=1 for Qualcomm QNN)", ErrUnavailable)
}
