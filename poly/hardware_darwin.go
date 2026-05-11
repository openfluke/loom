//go:build darwin || ios

package poly

import (
	"encoding/binary"
	"fmt"
	"syscall"
	"time"
	"unsafe"
)

func getUptime() string {
	// kern.boottime returns a struct timeval
	s, err := syscall.Sysctl("kern.boottime")
	if err != nil || len(s) < 16 {
		return "Unknown (Darwin)"
	}
	
	// struct timeval is { int64_t tv_sec; int32_t tv_usec; } on 64-bit Darwin
	// But it might vary. Usually 16 bytes total.
	sec := *(*int64)(unsafe.Pointer(&[]byte(s)[0]))
	if sec == 0 {
		return "Unknown (Darwin)"
	}
	
	bootTime := time.Unix(sec, 0)
	return time.Since(bootTime).String()
}

func getCPUModel() string {
	s, err := syscall.Sysctl("machdep.cpu.brand_string")
	if err != nil {
		// Fallback for some Apple Silicon cases or different sysctl name
		s, err = syscall.Sysctl("hw.model")
		if err != nil {
			return "Unknown CPU"
		}
	}
	return s
}

func getSystemMemory() (total, free uint64) {
	// hw.memsize is a uint64
	s, err := syscall.Sysctl("hw.memsize")
	if err == nil && len(s) == 8 {
		total = binary.LittleEndian.Uint64([]byte(s))
	}

	// Free memory is complex (needs host_statistics / vm_stat)
	// For now, return total / 2 as a placeholder as before, but at least build works
	free = total / 2
	return
}

func getGPUInfo() GPUInfo {
	return GPUInfo{Model: "Unknown GPU (Apple Metal/Universal)"}
}

func getDiskUsage() []DiskInfo {
	return []DiskInfo{{Path: "/"}}
}

func sysctlUint64(name string) (uint64, error) {
	s, err := syscall.Sysctl(name)
	if err != nil {
		return 0, err
	}
	if len(s) != 8 {
		return 0, fmt.Errorf("unexpected sysctl size: %d", len(s))
	}
	return binary.LittleEndian.Uint64([]byte(s)), nil
}
