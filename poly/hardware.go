package poly

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
)

var (
	hardwareInfo     HardwareInfo
	hardwareInfoOnce sync.Once
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
	hardwareInfoOnce.Do(func() {
		hardwareInfo = HardwareInfo{
			L1DataCacheSize: 32768,   // Default 32KB
			L2CacheSize:     262144,  // Default 256KB
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
							hardwareInfo.L2CacheSize = val * 1024
						}
					}
					if strings.HasPrefix(line, "L3CacheSize=") {
						if val, e := strconv.Atoi(strings.TrimPrefix(line, "L3CacheSize=")); e == nil {
							hardwareInfo.L3CacheSize = val * 1024
						}
					}
				}
			}
		} else if runtime.GOOS == "darwin" || runtime.GOOS == "ios" {
			// macOS & iOS sysctl
			keys := map[string]*int{
				"hw.l1dcachesize": &hardwareInfo.L1DataCacheSize,
				"hw.l2cachesize":  &hardwareInfo.L2CacheSize,
				"hw.l3cachesize":  &hardwareInfo.L3CacheSize,
			}
			for key, target := range keys {
				if out, err := exec.Command("sysctl", "-n", key).Output(); err == nil {
					if val, e := strconv.Atoi(strings.TrimSpace(string(out))); e == nil && val > 0 {
						*target = val
					}
				}
			}
		} else if runtime.GOOS == "linux" || runtime.GOOS == "android" {
			for i := 0; i < 4; i++ {
				basePath := "/sys/devices/system/cpu/cpu0/cache/index" + strconv.Itoa(i) + "/"
				levelData, _ := os.ReadFile(basePath + "level")
				typeData, _ := os.ReadFile(basePath + "type")
				sizeData, _ := os.ReadFile(basePath + "size")

				if len(sizeData) == 0 {
					continue
				}

				level, _ := strconv.Atoi(strings.TrimSpace(string(levelData)))
				cacheType := strings.ToLower(strings.TrimSpace(string(typeData)))

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
					hardwareInfo.L1DataCacheSize = bytes
				} else if level == 2 {
					hardwareInfo.L2CacheSize = bytes
				} else if level == 3 {
					hardwareInfo.L3CacheSize = bytes
				}
			}
		}

		// Architectural Defaults Overrides if detection was partial
		if hardwareInfo.L1DataCacheSize <= 0 {
			if runtime.GOARCH == "arm64" {
				hardwareInfo.L1DataCacheSize = 65536 // Many ARM64 (M1/M2/modern phones) have 64KB L1D
			} else {
				hardwareInfo.L1DataCacheSize = 32768 // Standard x86
			}
		}
	})
	return hardwareInfo
}

// CalculateOptimalTileSize returns a tile size that fits the working set in L1/L2.
// For MHA: Working set = TileSize * headDim * 2 * 4 (K and V tiles in float32)
func CalculateOptimalTileSize(headDim int) int {
	info := GetHardwareInfo()

	tileSize := info.L1DataCacheSize / (headDim * 2 * 4)

	// Clamp to reasonable ranges
	if tileSize < 8 {
		tileSize = 8
	}
	if tileSize > 256 {
		tileSize = 256
	}

	// Align to 8 or 16 for SIMD friendliness
	if tileSize >= 16 {
		tileSize = (tileSize / 16) * 16
	} else {
		tileSize = (tileSize / 8) * 8
	}

	return tileSize
}

// CalculateOptimalCNN3TileSize picks a TileSize that fits the 3D local neighborhood in L1.
// Working set approx = (TileSize^3 * InChannels * 4) bytes.
func CalculateOptimalCNN3TileSize(inChannels int) int {
	info := GetHardwareInfo()
	l1 := info.L1DataCacheSize
	if l1 <= 0 {
		l1 = 65536 // Modern default (64KB)
	}

	// We want (T^3 * C * 4) < L1
	// T^3 < L1 / (4 * C)
	// T < cubert(L1 / (4 * C))
	limit := float64(l1) / float64(4*inChannels)
	tFloat := math.Pow(limit, 1.0/3.0)

	// Round up to nearest power of 2 within [8, 32].
	// Floor-to-multiple alignment loses precision (e.g. 10 → 8 on Apple M-series).
	tileSize := 8
	for _, candidate := range []int{8, 16, 32} {
		if float64(candidate) <= tFloat {
			tileSize = candidate
		}
	}

	return tileSize
}

// CalculateOptimalGPUTileSizeFromLimits derives the best GPU tiling size from raw WebGPU Limits.
//   sharedMemBytes = adapter.GetLimits().Limits.MaxComputeWorkgroupStorageSize
//   maxInvocations = adapter.GetLimits().Limits.MaxComputeInvocationsPerWorkgroup
//   headDim        = model head dimension (e.g. 64, 128)
//
// Logic: each tile row costs headDim*2*4 bytes (K+V, float32).
// We use at most half of shared mem so the driver has spill room.
// Result is clamped to [8, 64] and aligned to 8 to match the WGSL shader workgroup size.
func CalculateOptimalGPUTileSizeFromLimits(sharedMemBytes, maxInvocations uint32, headDim int) int {
	const float32Bytes = 4

	if headDim <= 0 {
		headDim = 64
	}

	bytesPerTileRow := headDim * 2 * float32Bytes
	tileSize := 32 // safe fallback

	if sharedMemBytes > 0 {
		usable := int(sharedMemBytes) / 2
		tileSize = usable / bytesPerTileRow
	}

	// Cap to workgroup invocation limit
	if maxInvocations > 0 && tileSize > int(maxInvocations) {
		tileSize = int(maxInvocations)
	}

	// Clamp and align
	if tileSize < 8  { tileSize = 8 }
	if tileSize > 64 { tileSize = 64 }
	tileSize = (tileSize / 8) * 8
	if tileSize < 8  { tileSize = 8 }

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
