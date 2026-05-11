//go:build linux || android

package poly

import (
	"os"
	"strconv"
	"strings"
	"time"
)

func getUptime() string {
	if b, err := os.ReadFile("/proc/uptime"); err == nil {
		fields := strings.Fields(string(b))
		if len(fields) > 0 {
			if sec, err := strconv.ParseFloat(fields[0], 64); err == nil {
				return time.Duration(sec * float64(time.Second)).String()
			}
		}
	}
	return "Unknown"
}

func getCPUModel() string {
	if b, err := os.ReadFile("/proc/cpuinfo"); err == nil {
		for _, line := range strings.Split(string(b), "\n") {
			if strings.HasPrefix(line, "model name") {
				parts := strings.SplitN(line, ":", 2)
				if len(parts) == 2 {
					return strings.TrimSpace(parts[1])
				}
			}
		}
	}
	return "Unknown CPU"
}

func getSystemMemory() (total, free uint64) {
	if b, err := os.ReadFile("/proc/meminfo"); err == nil {
		lines := strings.Split(string(b), "\n")
		for _, line := range lines {
			fields := strings.Fields(line)
			if len(fields) < 2 {
				continue
			}
			val, _ := strconv.ParseUint(fields[1], 10, 64)
			val *= 1024
			if fields[0] == "MemTotal:" {
				total = val
			}
			if fields[0] == "MemAvailable:" {
				free = val
			}
			if free == 0 && fields[0] == "MemFree:" {
				free = val
			}
		}
	}
	return
}

func getGPUInfo() GPUInfo {
	var info GPUInfo
	info.Model = "Unknown GPU"
	// Attempt to find GPU info in /sys/class/drm
	if entries, err := os.ReadDir("/sys/class/drm"); err == nil {
		for _, e := range entries {
			if strings.HasPrefix(e.Name(), "card") && !strings.Contains(e.Name(), "-") {
				// Try to get VRAM from mem_info_vram_total (AMD) or similar
				vramPath := "/sys/class/drm/" + e.Name() + "/device/mem_info_vram_total"
				if b, err := os.ReadFile(vramPath); err == nil {
					if v, err := strconv.ParseUint(strings.TrimSpace(string(b)), 10, 64); err == nil {
						info.VRAM = v
					}
				}
				// Try to get model from device/uevent or vendor/device files
				modelPath := "/sys/class/drm/" + e.Name() + "/device/uevent"
				if b, err := os.ReadFile(modelPath); err == nil {
					for _, line := range strings.Split(string(b), "\n") {
						if strings.HasPrefix(line, "PCI_ID=") {
							info.Vendor = strings.TrimPrefix(line, "PCI_ID=")
						}
					}
				}
			}
		}
	}
	return info
}

func getDiskUsage() []DiskInfo {
	var disks []DiskInfo
	if b, err := os.ReadFile("/proc/mounts"); err == nil {
		lines := strings.Split(string(b), "\n")
		for _, line := range lines {
			fields := strings.Fields(line)
			if len(fields) >= 2 && (strings.HasPrefix(fields[1], "/mnt") || fields[1] == "/" || strings.HasPrefix(fields[1], "/storage")) {
				disks = append(disks, DiskInfo{Path: fields[1]})
			}
		}
	}
	return disks
}
