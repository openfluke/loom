//go:build !darwin && !linux && !freebsd && !openbsd

package poly

func processRSSBytes() uint64 {
	return 0
}
