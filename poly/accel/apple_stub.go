//go:build !darwin || !cgo

package accel

import "fmt"

// PrepareAppleRuntime is a no-op off macOS (the Apple plugin is Metal/Accelerate).
func PrepareAppleRuntime() error { return nil }

func defaultApplePath() string { return "" }

func appleGPUAvailable(path string) bool { return false }

func openApplePlugin(path, device string) (Plugin, error) {
	return nil, fmt.Errorf("%w (need darwin + CGO_ENABLED=1 for Apple Metal/Accelerate)", ErrUnavailable)
}
