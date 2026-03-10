package poly

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

// HardwareInfo stores metadata about the running system to optimize tiling.
type HardwareInfo struct {
	L1DataCacheSize int // in bytes
	L2CacheSize     int // in bytes
	L3CacheSize     int // in bytes
	NumCPU          int
}

// GetHardwareInfo attempts to detect cache sizes and CPU info.
func GetHardwareInfo() HardwareInfo {
	info := HardwareInfo{
		L1DataCacheSize: 32768, // Default 32KB
		L2CacheSize:     262144, // Default 256KB
		L3CacheSize:     8388608, // Default 8MB
		NumCPU:          runtime.NumCPU(),
	}

	if runtime.GOOS == "windows" {
		// Attempt to get L2/L3 via wmic
		out, err := exec.Command("wmic", "cpu", "get", "L2CacheSize,L3CacheSize", "/format:value").Output()
		if err == nil {
			lines := strings.Split(string(out), "\n")
			for _, line := range lines {
				line = strings.TrimSpace(line)
				if strings.HasPrefix(line, "L2CacheSize=") {
					if val, e := strconv.Atoi(strings.TrimPrefix(line, "L2CacheSize=")); e == nil {
						info.L2CacheSize = val * 1024
					}
				}
				if strings.HasPrefix(line, "L3CacheSize=") {
					if val, e := strconv.Atoi(strings.TrimPrefix(line, "L3CacheSize=")); e == nil {
						info.L3CacheSize = val * 1024
					}
				}
			}
		}
	} else if runtime.GOOS == "darwin" || runtime.GOOS == "ios" {
		// macOS & iOS sysctl
		// These keys are standard across Intel and Apple Silicon (ARM)
		keys := map[string]*int{
			"hw.l1dcachesize": &info.L1DataCacheSize,
			"hw.l2cachesize":  &info.L2CacheSize,
			"hw.l3cachesize":  &info.L3CacheSize,
		}
		for key, target := range keys {
			if out, err := exec.Command("sysctl", "-n", key).Output(); err == nil {
				if val, e := strconv.Atoi(strings.TrimSpace(string(out))); e == nil && val > 0 {
					*target = val
				}
			}
		}
	} else if runtime.GOOS == "linux" || runtime.GOOS == "android" {
		// Linux & Android /sys/devices/system/cpu/cpu0/cache/
		// We iterate through indices to find the right level and type
		for i := 0; i < 4; i++ {
			basePath := "/sys/devices/system/cpu/cpu0/cache/index" + strconv.Itoa(i) + "/"
			levelData, _ := os.ReadFile(basePath + "level")
			typeData, _ := os.ReadFile(basePath + "type")
			sizeData, _ := os.ReadFile(basePath + "size")
			
			if len(sizeData) == 0 { continue }
			
			level, _ := strconv.Atoi(strings.TrimSpace(string(levelData)))
			cacheType := strings.ToLower(strings.TrimSpace(string(typeData)))
			
			// Parse size string like "32K", "1M"
			s := strings.TrimSpace(string(sizeData))
			multiplier := 1
			if strings.HasSuffix(s, "K") {
				multiplier = 1024
				s = s[:len(s)-1]
			} else if strings.HasSuffix(s, "M") {
				multiplier = 1024 * 1024
				s = s[:len(s)-1]
			}
			val, _ := strconv.Atoi(s)
			bytes := val * multiplier
			
			if level == 1 && (cacheType == "data" || cacheType == "unified") {
				info.L1DataCacheSize = bytes
			} else if level == 2 {
				info.L2CacheSize = bytes
			} else if level == 3 {
				info.L3CacheSize = bytes
			}
		}
	}

	// Architectural Defaults Overrides if detection was partial
	if info.L1DataCacheSize <= 0 {
		if runtime.GOARCH == "arm64" {
			info.L1DataCacheSize = 65536 // Many ARM64 (M1/M2/modern phones) have 64KB L1D
		} else {
			info.L1DataCacheSize = 32768 // Standard x86
		}
	}
	return info
}

// CalculateOptimalTileSize returns a tile size that fits the working set in L1/L2.
// For MHA: Working set = TileSize * headDim * 2 * 4 (K and V tiles in float32)
func CalculateOptimalTileSize(headDim int) int {
	info := GetHardwareInfo()
	
	// We want the KV tile to fit comfortably in L1 Data Cache.
	// 32KB L1 is standard.
	// TileSize = L1Size / (headDim * 2 * 4)
	
	tileSize := info.L1DataCacheSize / (headDim * 2 * 4)
	
	// Clamp to reasonable ranges
	if tileSize < 8 { tileSize = 8 }
	if tileSize > 256 { tileSize = 256 }
	
	// Align to 8 or 16 for SIMD friendliness
	if tileSize >= 16 {
		tileSize = (tileSize / 16) * 16
	} else {
		tileSize = (tileSize / 8) * 8
	}
	
	return tileSize
}

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


