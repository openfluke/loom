package poly

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

// GetDeviceDescription returns a human-readable string of the running OS, CPU, RAM, and GPU.
func GetDeviceDescription(net *VolumetricNetwork) string {
	osName := runtime.GOOS
	cpuName := "Unknown CPU"
	ramStr := "Unknown RAM"

	if osName == "windows" {
		// Get CPU Name
		if out, err := exec.Command("wmic", "cpu", "get", "Name", "/format:value").Output(); err == nil {
			for _, line := range strings.Split(string(out), "\n") {
				line = strings.TrimSpace(line)
				if strings.HasPrefix(line, "Name=") {
					cpuName = strings.TrimPrefix(line, "Name=")
				}
			}
		}
		// Get Total RAM
		if out, err := exec.Command("wmic", "computersystem", "get", "TotalPhysicalMemory", "/format:value").Output(); err == nil {
			for _, line := range strings.Split(string(out), "\n") {
				line = strings.TrimSpace(line)
				if strings.HasPrefix(line, "TotalPhysicalMemory=") {
					if bytes, err := strconv.ParseUint(strings.TrimPrefix(line, "TotalPhysicalMemory="), 10, 64); err == nil {
						ramStr = fmt.Sprintf("%.2f GB", float64(bytes)/(1024*1024*1024))
					}
				}
			}
		}
	} else if osName == "darwin" {
		if out, err := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output(); err == nil {
			cpuName = strings.TrimSpace(string(out))
		}
		if out, err := exec.Command("sysctl", "-n", "hw.memsize").Output(); err == nil {
			if bytes, err := strconv.ParseUint(strings.TrimSpace(string(out)), 10, 64); err == nil {
				ramStr = fmt.Sprintf("%.2f GB", float64(bytes)/(1024*1024*1024))
			}
		}
	} else if osName == "linux" {
		// Attempt simple parse from /proc/cpuinfo and /proc/meminfo
		if b, err := os.ReadFile("/proc/cpuinfo"); err == nil {
			for _, line := range strings.Split(string(b), "\n") {
				if strings.HasPrefix(line, "model name") {
					parts := strings.SplitN(line, ":", 2)
					if len(parts) == 2 {
						cpuName = strings.TrimSpace(parts[1])
						break
					}
				}
			}
		}
		if b, err := os.ReadFile("/proc/meminfo"); err == nil {
			for _, line := range strings.Split(string(b), "\n") {
				if strings.HasPrefix(line, "MemTotal:") {
					fields := strings.Fields(line)
					if len(fields) >= 2 {
						if kb, err := strconv.ParseUint(fields[1], 10, 64); err == nil {
							ramStr = fmt.Sprintf("%.2f GB", float64(kb)/(1024*1024))
						}
					}
					break
				}
			}
		}
	}

	gpuStr := "None (CPU Only)"
	if net != nil && net.GPUContext != nil && net.GPUContext.Adapter != nil {
		info := net.GPUContext.Adapter.GetInfo()
		gpuStr = fmt.Sprintf("%s (%v)", info.Name, info.BackendType)
	}

	return fmt.Sprintf("OS: %s | CPU: %s | RAM: %s | GPU: %s", osName, cpuName, ramStr, gpuStr)
}
