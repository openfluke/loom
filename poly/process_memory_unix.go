//go:build darwin || linux || freebsd || openbsd

package poly

import (
	"runtime"
	"syscall"
)

func processRSSBytes() uint64 {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return 0
	}
	// Darwin reports bytes; Linux and BSD report kilobytes.
	if runtime.GOOS == "darwin" {
		return uint64(ru.Maxrss)
	}
	return uint64(ru.Maxrss) * 1024
}
