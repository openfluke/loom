package poly

import (
	"math"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
)

// =============================================================================
// Hardware cache detection
// =============================================================================

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

// =============================================================================
// Bytes-per-element helper
// =============================================================================

// cnn3DTypeBytesPerElement returns the actual bytes-per-weight for a given DType
// based on how Morph stores it in RAM. Smaller types fit more into L1, allowing
// larger tile sizes and better cache reuse.
func cnn3DTypeBytesPerElement(dtype DType) float64 {
	switch dtype {
	case DTypeFloat64, DTypeInt64, DTypeUint64:
		return 8
	case DTypeFloat32, DTypeFloat16, DTypeBFloat16: // Float16/BFloat16 stored as []float32 via Morph
		return 4
	case DTypeInt32, DTypeUint32:
		return 4
	case DTypeInt16, DTypeUint16:
		return 2
	default:
		// Int8, Uint8, FP8E4M3, FP8E5M2, Int4, Uint4, FP4,
		// Int2, Uint2, Ternary, Binary — all stored as []int8 via Morph
		return 1
	}
}

// =============================================================================
// CPU tile size calculators
// =============================================================================

// CalculateOptimalTileSize returns a tile size that fits the working set in L1.
// For MHA: Working set = TileSize * headDim * 2 * bytesPerWeight (K and V tiles in RAM).
func CalculateOptimalTileSize(headDim int, dtype DType) int {
	info := GetHardwareInfo()
	bytesPerWeight := cnn3DTypeBytesPerElement(dtype)
	if bytesPerWeight < 1 {
		bytesPerWeight = 1
	}

	tileSize := info.L1DataCacheSize / int(float64(headDim*2)*bytesPerWeight)

	if tileSize < 8 {
		tileSize = 8
	}
	if tileSize > 256 {
		tileSize = 256
	}

	if tileSize >= 16 {
		tileSize = (tileSize / 16) * 16
	} else {
		tileSize = (tileSize / 8) * 8
	}

	return tileSize
}

// CalculateOptimalCNN1TileSize picks a TileSize that fits the 1D local neighborhood in L1.
// Working set approx = (TileSize * InChannels * bytesPerWeight) bytes.
func CalculateOptimalCNN1TileSize(inChannels int, dtype DType) int {
	info := GetHardwareInfo()
	l1 := info.L1DataCacheSize
	if l1 <= 0 {
		l1 = 65536
	}

	bytesPerWeight := cnn3DTypeBytesPerElement(dtype)

	// T * C * bytesPerWeight < L1  =>  T < L1 / (bytesPerWeight * C)
	limit := float64(l1) / (bytesPerWeight * float64(inChannels))

	tileSize := 8
	for _, candidate := range []int{8, 16, 32, 64, 128} {
		if float64(candidate) <= limit {
			tileSize = candidate
		}
	}

	return tileSize
}

// CalculateOptimalCNN2TileSize picks a TileSize that fits the 2D local neighborhood in L1.
// Working set approx = (TileSize^2 * InChannels * bytesPerWeight) bytes.
func CalculateOptimalCNN2TileSize(inChannels int, dtype DType) int {
	info := GetHardwareInfo()
	l1 := info.L1DataCacheSize
	if l1 <= 0 {
		l1 = 65536
	}

	bytesPerWeight := cnn3DTypeBytesPerElement(dtype)

	// T^2 * C * bytesPerWeight < L1  =>  T < sqrt(L1 / (bytesPerWeight * C))
	limit := float64(l1) / (bytesPerWeight * float64(inChannels))
	tFloat := math.Sqrt(limit)

	tileSize := 8
	for _, candidate := range []int{8, 16, 32, 64} {
		if float64(candidate) <= tFloat {
			tileSize = candidate
		}
	}

	return tileSize
}

// CalculateOptimalCNN3TileSize picks a TileSize that fits the 3D local neighborhood in L1.
// Working set approx = (TileSize^3 * InChannels * bytesPerWeight) bytes.
func CalculateOptimalCNN3TileSize(inChannels int, dtype DType) int {
	info := GetHardwareInfo()
	l1 := info.L1DataCacheSize
	if l1 <= 0 {
		l1 = 65536
	}

	bytesPerWeight := cnn3DTypeBytesPerElement(dtype)

	// T^3 * C * bytesPerWeight < L1  =>  T < cuberoot(L1 / (bytesPerWeight * C))
	limit := float64(l1) / (bytesPerWeight * float64(inChannels))
	tFloat := math.Pow(limit, 1.0/3.0)

	tileSize := 8
	for _, candidate := range []int{8, 16, 32} {
		if float64(candidate) <= tFloat {
			tileSize = candidate
		}
	}

	return tileSize
}

// CalculateOptimalDenseTileSize picks a TileSize for Dense forward/backward.
// Working set per output tile ≈ TileSize × inputSize × bytesPerWeight (weight row slice).
func CalculateOptimalDenseTileSize(inputSize int, dtype DType) int {
	info := GetHardwareInfo()
	l1 := info.L1DataCacheSize
	if l1 <= 0 {
		l1 = 32768
	}
	bytesPerWeight := cnn3DTypeBytesPerElement(dtype)
	if bytesPerWeight < 1 {
		bytesPerWeight = 1
	}
	limit := float64(l1) / (bytesPerWeight * float64(inputSize))
	tileSize := 8
	for _, candidate := range []int{8, 16, 32, 64, 128, 256} {
		if float64(candidate) <= limit {
			tileSize = candidate
		}
	}
	return tileSize
}

// CalculateOptimalSwiGLUTileSize picks a TileSize for SwiGLU sequence-tiled computation.
// Working set per sequence tile ≈ TileSize × inputSize × bytesPerWeight (gate/down row slice).
func CalculateOptimalSwiGLUTileSize(inputSize int, dtype DType) int {
	return CalculateOptimalDenseTileSize(inputSize, dtype)
}

// CalculateOptimalRNNTileSize picks a TileSize for RNN hidden-state tiling.
// Working set per tile ≈ TileSize × (inputSize + hiddenSize) × bytesPerWeight (combined weight row).
func CalculateOptimalRNNTileSize(inputSize, hiddenSize int, dtype DType) int {
	info := GetHardwareInfo()
	l1 := info.L1DataCacheSize
	if l1 <= 0 {
		l1 = 32768
	}
	bytesPerWeight := cnn3DTypeBytesPerElement(dtype)
	if bytesPerWeight < 1 {
		bytesPerWeight = 1
	}
	combined := inputSize + hiddenSize
	limit := float64(l1) / (bytesPerWeight * float64(combined))
	tileSize := 8
	for _, candidate := range []int{8, 16, 32, 64, 128, 256} {
		if float64(candidate) <= limit {
			tileSize = candidate
		}
	}
	return tileSize
}

// CalculateOptimalLSTMTileSize picks a TileSize for LSTM hidden-state tiling.
// Working set per tile ≈ TileSize × (inputSize + hiddenSize) × 4 gates × bytesPerWeight.
func CalculateOptimalLSTMTileSize(inputSize, hiddenSize int, dtype DType) int {
	info := GetHardwareInfo()
	l1 := info.L1DataCacheSize
	if l1 <= 0 {
		l1 = 32768
	}
	bytesPerWeight := cnn3DTypeBytesPerElement(dtype)
	if bytesPerWeight < 1 {
		bytesPerWeight = 1
	}
	combined := inputSize + hiddenSize
	limit := float64(l1) / (bytesPerWeight * float64(combined) * 4)
	tileSize := 8
	for _, candidate := range []int{8, 16, 32, 64, 128, 256} {
		if float64(candidate) <= limit {
			tileSize = candidate
		}
	}
	return tileSize
}

// CalculateOptimalEmbeddingTileSize picks a TileSize for Embedding sequence-tiled lookup.
// Working set per tile ≈ TileSize × embeddingDim × bytesPerWeight (embedding row slice).
func CalculateOptimalEmbeddingTileSize(embeddingDim int, dtype DType) int {
	info := GetHardwareInfo()
	l1 := info.L1DataCacheSize
	if l1 <= 0 {
		l1 = 32768
	}
	bytesPerWeight := cnn3DTypeBytesPerElement(dtype)
	if bytesPerWeight < 1 {
		bytesPerWeight = 1
	}
	limit := float64(l1) / (bytesPerWeight * float64(embeddingDim))
	tileSize := 8
	for _, candidate := range []int{8, 16, 32, 64, 128, 256} {
		if float64(candidate) <= limit {
			tileSize = candidate
		}
	}
	return tileSize
}

// CalculateOptimalResidualTileSize picks a TileSize for Residual (skip-connection add).
// Working set per tile ≈ TileSize × 3 arrays (input, skip, output) × bytesPerWeight.
// Residual is trivially parallel so large tiles are preferred.
func CalculateOptimalResidualTileSize(dtype DType) int {
	info := GetHardwareInfo()
	l1 := info.L1DataCacheSize
	if l1 <= 0 {
		l1 = 32768
	}
	bytesPerWeight := cnn3DTypeBytesPerElement(dtype)
	if bytesPerWeight < 1 {
		bytesPerWeight = 1
	}
	limit := float64(l1) / (bytesPerWeight * 3.0)
	tileSize := 64
	for _, candidate := range []int{64, 128, 256, 512, 1024, 2048} {
		if float64(candidate) <= limit {
			tileSize = candidate
		}
	}
	return tileSize
}

// =============================================================================
// GPU tile size calculators
// =============================================================================

// CalculateOptimalGPUTileSizeFromLimits derives the best GPU tiling size from raw WebGPU Limits.
//
//	sharedMemBytes = adapter.GetLimits().Limits.MaxComputeWorkgroupStorageSize
//	maxInvocations = adapter.GetLimits().Limits.MaxComputeInvocationsPerWorkgroup
//	headDim        = model head dimension (e.g. 64, 128)
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

	if maxInvocations > 0 && tileSize > int(maxInvocations) {
		tileSize = int(maxInvocations)
	}

	if tileSize < 8 {
		tileSize = 8
	}
	if tileSize > 64 {
		tileSize = 64
	}
	tileSize = (tileSize / 8) * 8
	if tileSize < 8 {
		tileSize = 8
	}

	return tileSize
}

// DenseGPUTileSizesFromContext returns the SC and MC GPU tile sizes for Dense kernels.
func DenseGPUTileSizesFromContext(ctx *WGPUContext, dtype DType) (scTile, mcTile int) {
	// Dense matmuls are often bound by memory latency for small batches,
	// and memory bandwidth for large ones. 
	// Smaller dtypes can support larger tiles.
	bytes := cnn3DTypeBytesPerElement(dtype)
	multiplier := 1.0
	if bytes < 4 {
		multiplier = 4.0 / bytes
	}
	if multiplier > 4.0 { multiplier = 4.0 }

	// For Dense SC, we prioritize being able to fit weights in shared memory if possible.
	// But usually, we only cache inputs in shared memory.
	sc := int(float64(ctx.GPUTileSize) * multiplier)
	if sc < 32 { sc = 32 }
	if sc > 128 { sc = 128 } // Capped for workgroup size limits
	sc = (sc / 32) * 32

	mc := int(float64(ctx.Limits.MaxComputeInvocationsPerWorkgroup) * multiplier)
	if mc > 256 { mc = 256 }
	mc = (mc / 64) * 64
	if mc < 64 { mc = 64 }

	return sc, mc
}

func cnnGPUTileSizesFromContext(ctx *WGPUContext, dtype DType) (scTile, mcTile int) {
	limits := ctx.Limits
	bytes := cnn3DTypeBytesPerElement(dtype)
	
	// Factor in bandwidth/cache benefits of smaller types
	multiplier := 1.0
	if bytes < 4 {
		multiplier = 4.0 / bytes // e.g. 4.0 for Int8
	}
	if multiplier > 4.0 {
		multiplier = 4.0 // Cap scaling
	}

	sc := int(float64(ctx.GPUTileSize) * multiplier)
	if sc < 32 {
		sc = 32
	}
	// Strictly cap sc to 64 to avoid exhausting workgroup shared memory
	// (64 tokens * 64 headDim * 2 buffers * 4 bytes = 32KB)
	if sc > 64 {
		sc = 64
	}
	sc = (sc / 32) * 32

	mc := int(float64(limits.MaxComputeInvocationsPerWorkgroup) / 4.0 * multiplier)
	if mc > 128 {
		mc = 128
	}
	mc = (mc / 64) * 64
	if mc < 64 {
		mc = 64
	}
	return sc, mc
}

// MHAGPUTileSizes returns SC and MC GPU tile sizes for MHA attention kernels.
func MHAGPUTileSizes(ctx *WGPUContext, headDim int, dtype DType) (scTile, mcTile int) {
	sharedMem := ctx.Limits.MaxComputeWorkgroupStorageSize
	maxInv := ctx.Limits.MaxComputeInvocationsPerWorkgroup
	
	// MHA shaders use F32 in shared memory regardless of DType (after dequantization),
	// but smaller types benefit from faster global-to-shared transfers.
	// We use the base limit calculation but allow slightly larger tiles if DType is small.
	scTile = CalculateOptimalGPUTileSizeFromLimits(sharedMem/2, maxInv, headDim)
	mcTile = CalculateOptimalGPUTileSizeFromLimits(sharedMem, maxInv, headDim)
	
	bytes := cnn3DTypeBytesPerElement(dtype)
	if bytes < 4 {
		// Conservative boost for small types
		scTile = int(float64(scTile) * 1.5)
		mcTile = int(float64(mcTile) * 1.5)
	}

	// Always align to 8 and cap at 128 for MHA
	if scTile > 64 { scTile = 64 }
	if mcTile > 128 { mcTile = 128 }

	return (scTile/8)*8, (mcTile/8)*8
}

// =============================================================================
// Per-layer per-dtype tile size getters
// =============================================================================

// allDTypes lists every numerical type for per-dtype tile population.
var allDTypes = []DType{
	DTypeFloat32, DTypeFloat64, DTypeFloat16, DTypeBFloat16,
	DTypeFP8E4M3, DTypeFP8E5M2,
	DTypeInt64, DTypeUint64, DTypeInt32, DTypeUint32,
	DTypeInt16, DTypeUint16, DTypeInt8, DTypeUint8,
	DTypeInt4, DTypeUint4, DTypeFP4,
	DTypeInt2, DTypeUint2, DTypeTernary, DTypeBinary,
}

// GetCPUTileSize returns the CPU tile size for the given dtype.
// Falls back to the legacy TileSize if no per-dtype entry exists.
func (l *VolumetricLayer) GetCPUTileSize(dtype DType) int {
	if l.CPUTileSizes != nil {
		if ts, ok := l.CPUTileSizes[dtype]; ok && ts > 0 {
			return ts
		}
	}
	if l.TileSize > 0 {
		return l.TileSize
	}
	return 8
}

// GetGPUSCTileSize returns the GPU single-core tile size for the given dtype.
func (l *VolumetricLayer) GetGPUSCTileSize(dtype DType) int {
	if l.GPUSCTileSizes != nil {
		if ts, ok := l.GPUSCTileSizes[dtype]; ok && ts > 0 {
			return ts
		}
	}
	if l.Network != nil && l.Network.GPUContext != nil {
		sc, _ := cnnGPUTileSizesFromContext(l.Network.GPUContext, dtype)
		return sc
	}
	return 64
}

// GetGPUMCTileSize returns the GPU multi-core tile size for the given dtype.
func (l *VolumetricLayer) GetGPUMCTileSize(dtype DType) int {
	if l.GPUMCTileSizes != nil {
		if ts, ok := l.GPUMCTileSizes[dtype]; ok && ts > 0 {
			return ts
		}
	}
	if l.Network != nil && l.Network.GPUContext != nil {
		_, mc := cnnGPUTileSizesFromContext(l.Network.GPUContext, dtype)
		return mc
	}
	return 256
}
