package poly

import (
	"math"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/openfluke/loom/poly/simd"
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

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// CalculateOptimalCNN1TileSizeForLayer derives a runtime CPU tile size from the
// actual layer shape plus detected L1 cache, rather than only the input channel count.
func CalculateOptimalCNN1TileSizeForLayer(l *VolumetricLayer, dtype DType) int {
	if l == nil {
		return CalculateOptimalCNN1TileSize(1, dtype)
	}

	info := GetHardwareInfo()
	l1 := info.L1DataCacheSize
	if l1 <= 0 {
		l1 = 32768
	}
	target := int(float64(l1) * 0.75)
	bytesPerWeight := cnn3DTypeBytesPerElement(dtype)
	if bytesPerWeight < 1 {
		bytesPerWeight = 1
	}

	inC := maxInt(1, l.InputChannels)
	kSize := maxInt(1, l.KernelSize)
	stride := maxInt(1, l.Stride)
	padding := maxInt(0, l.Padding)
	outLen := maxInt(1, l.OutputHeight)
	filters := maxInt(1, l.Filters)

	best := 4
	for _, candidate := range []int{4, 8, 16, 32, 64, 128} {
		outTile := minInt(candidate, outLen)
		filterTile := minInt(candidate, filters)

		inputSpan := (outTile-1)*stride + kSize + 2*padding
		inputBytes := inputSpan * inC * 4
		weightBytes := int(float64(filterTile*inC*kSize) * bytesPerWeight)
		accumBytes := outTile * filterTile * 8
		totalBytes := inputBytes + weightBytes + accumBytes

		if totalBytes <= target {
			best = candidate
		}
	}

	if best < 4 {
		best = 4
	}
	if best > outLen {
		best = outLen
	}
	return best
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
	return calculateOptimalDenseTileSize(inputSize, dtype, false)
}

func calculateOptimalDenseTileSize(inputSize int, dtype DType, simdPath bool) int {
	info := GetHardwareInfo()
	l1 := info.L1DataCacheSize
	if l1 <= 0 {
		l1 = 32768
	}
	bytesPerWeight := cnn3DTypeBytesPerElement(dtype)
	if bytesPerWeight < 1 {
		bytesPerWeight = 1
	}
	// SIMD dots keep input + weight rows hot; use ~75% of L1 like CNN1 layer sizing.
	cacheBudget := l1
	if simdPath {
		cacheBudget = int(float64(l1) * 0.75)
	}
	limit := float64(cacheBudget) / (bytesPerWeight * float64(inputSize))
	tileSize := 8
	candidates := []int{8, 16, 32, 64, 128, 256}
	for _, candidate := range candidates {
		if float64(candidate) <= limit {
			tileSize = candidate
		}
	}
	return tileSize
}

// DenseSimdMinDim is the per-layer width below which SIMD dot overhead loses to scalar tiled.
func DenseSimdMinDim() int {
	if !simd.SimdEnabled() {
		return math.MaxInt32
	}
	switch runtime.GOARCH {
	case "amd64":
		return 16
	case "arm64":
		return 8
	default:
		return math.MaxInt32
	}
}

func simdVecAlign() int {
	switch runtime.GOARCH {
	case "amd64":
		return 8
	case "arm64":
		return 4
	default:
		return 4
	}
}

// CalculateOptimalDenseSimdTileSizeForLayer picks output/input tile sizes for Dense SIMD
// (AVX2 on amd64, NEON on arm64) from L1 cache, layer dimensions, and vector width.
func CalculateOptimalDenseSimdTileSizeForLayer(l *VolumetricLayer, dtype DType) int {
	if l == nil {
		return 32
	}
	inputSize := maxInt(1, l.InputHeight)
	outputSize := maxInt(1, l.OutputHeight)
	dim := maxInt(inputSize, outputSize)

	if !simd.SimdEnabled() || dim < DenseSimdMinDim() {
		ts := CalculateOptimalDenseTileSize(inputSize, dtype)
		if ts > outputSize {
			ts = outputSize
		}
		return ts
	}

	base := calculateOptimalDenseTileSize(inputSize, dtype, true)
	align := simdVecAlign()
	minTile := align * 2
	if minTile < 16 {
		minTile = 16
	}
	if base < minTile && dim >= minTile {
		base = minTile
	}
	if dim >= 32 && base < 32 {
		info := GetHardwareInfo()
		l1 := info.L1DataCacheSize
		if l1 <= 0 {
			l1 = 32768
		}
		bytesPerWeight := cnn3DTypeBytesPerElement(dtype)
		if bytesPerWeight < 1 {
			bytesPerWeight = 1
		}
		if float64(32) <= float64(l1)*0.75/(bytesPerWeight*float64(inputSize)) {
			base = 32
		}
	}
	if base > outputSize {
		base = outputSize
	}
	if base >= align {
		base = (base / align) * align
	}
	if base <= 0 {
		base = minInt(minTile, outputSize)
	}
	if base <= 0 {
		base = 8
	}
	return base
}

// MasterWeightScaleForStackDepth shrinks master weights for deep wide stacks where
// unsigned integer dtypes (positive-only morphed weights) overflow activations.
func MasterWeightScaleForStackDepth(dtype DType, stackLayers, layerDim int) float32 {
	switch dtype {
	case DTypeUint64, DTypeUint32, DTypeUint16:
		if stackLayers < 48 || layerDim <= 16 {
			return 1.0
		}
		dimRatio := float64(16) / float64(layerDim)
		depthRatio := float64(48) / float64(stackLayers)
		if depthRatio > 1 {
			depthRatio = 1
		}
		return float32(dimRatio * dimRatio * depthRatio)
	}
	return 1.0
}

// MorphScaleForStackDepth is deprecated; use MasterWeightScaleForStackDepth.
func MorphScaleForStackDepth(dtype DType, stackLayers, layerDim int, baseScale float32) float32 {
	_ = baseScale
	return MasterWeightScaleForStackDepth(dtype, stackLayers, layerDim)
}

func capDenseTileToLayer(ts, inputSize, outputSize int) int {
	maxDim := maxInt(inputSize, outputSize)
	if maxDim <= 0 {
		return ts
	}
	if ts > maxDim {
		ts = maxDim
	}
	if ts <= 0 {
		ts = 1
	}
	return ts
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
	if multiplier > 4.0 {
		multiplier = 4.0
	}

	// For Dense SC, we prioritize being able to fit weights in shared memory if possible.
	sc := int(float64(ctx.GPUTileSize) * multiplier)
	if sc < 32 {
		sc = 32
	}
	if sc > 128 {
		sc = 128
	}
	sc = (sc / 32) * 32

	mc := int(float64(ctx.Limits.MaxComputeInvocationsPerWorkgroup) * multiplier)
	if mc > 256 {
		mc = 256
	}
	mc = (mc / 64) * 64
	if mc < 64 {
		mc = 64
	}
	return sc, mc
}

// SwiGLUGPUTileSizes returns the SC and MC tile sizes for SwiGLU tiling based on the
// GPU's auto-detected capabilities and the numerical DType.
func SwiGLUGPUTileSizes(ctx *WGPUContext, dtype DType) (scTile, mcTile int) {
	// SwiGLU tiling is similar to Dense tiling but usually involves larger intermediates.
	bytes := cnn3DTypeBytesPerElement(dtype)
	multiplier := 1.0
	if bytes < 4 {
		multiplier = 4.0 / bytes
	}
	if multiplier > 4.0 {
		multiplier = 4.0
	}

	sc := int(float64(ctx.GPUTileSize) * 4 * multiplier)
	if sc < 64 {
		sc = 64
	}
	if sc > 256 {
		sc = 256
	}
	sc = (sc / 64) * 64

	mc := int(float64(ctx.Limits.MaxComputeInvocationsPerWorkgroup))
	if mc <= 0 || mc > 256 {
		mc = 256
	}
	// Factor in multiplier for MC if possible, but usually workgroup limits are hard.
	// However, smaller types might allow larger workgroups if register pressure is lower,
	// but MaxComputeInvocationsPerWorkgroup is a static limit.
	mc = (mc / 64) * 64
	if mc < 64 {
		mc = 64
	}

	if sc >= mc {
		sc = mc / 2
		if sc < 64 {
			sc = 64
		}
	}

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

// CNN1GPUTileSizesForLayer derives GPU SC/MC tile sizes from actual layer output
// shape and adapter limits so the workgroup size matches the problem size better.
func CNN1GPUTileSizesForLayer(ctx *WGPUContext, l *VolumetricLayer, dtype DType) (scTile, mcTile int) {
	if ctx == nil || l == nil {
		return 64, 128
	}

	maxInv := int(ctx.Limits.MaxComputeInvocationsPerWorkgroup)
	if maxInv <= 0 {
		maxInv = 256
	}
	outLen := maxInt(1, l.OutputHeight)
	bytes := cnn3DTypeBytesPerElement(dtype)
	multiplier := 1.0
	if bytes < 4 {
		multiplier = 4.0 / bytes
	}
	if multiplier > 4.0 {
		multiplier = 4.0
	}

	scoreCandidate := func(candidate int, modeBias float64) float64 {
		if candidate > maxInv {
			return -1
		}
		usable := minInt(candidate, outLen)
		waste := float64(candidate) / float64(maxInt(usable, 1))
		score := float64(usable) / waste
		if outLen >= candidate {
			score *= multiplier
		} else {
			score *= math.Sqrt(multiplier)
		}
		if candidate > outLen*2 {
			score *= 0.75
		}
		return score * modeBias
	}

	bestSC, bestSCScore := 32, -1.0
	for _, candidate := range []int{32, 64, 128} {
		score := scoreCandidate(candidate, 1.0)
		if score > bestSCScore {
			bestSC = candidate
			bestSCScore = score
		}
	}

	bestMC, bestMCScore := 64, -1.0
	for _, candidate := range []int{64, 128, 256} {
		score := scoreCandidate(candidate, 1.05)
		if score > bestMCScore {
			bestMC = candidate
			bestMCScore = score
		}
	}

	if bestSC > maxInv {
		bestSC = maxInv
	}
	if bestMC > maxInv {
		bestMC = maxInv
	}
	if bestSC < 32 {
		bestSC = 32
	}
	if bestMC < 64 {
		bestMC = 64
	}
	if bestMC < bestSC {
		bestMC = bestSC
	}
	return bestSC, bestMC
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
	if scTile > 64 {
		scTile = 64
	}
	if mcTile > 128 {
		mcTile = 128
	}

	return (scTile / 8) * 8, (mcTile / 8) * 8
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

// GetCPUSimdTileSize returns the SIMD tile size for Dense forward on this layer.
func (l *VolumetricLayer) GetCPUSimdTileSize(dtype DType) int {
	if l.CPUSimdTileSizes != nil {
		if ts, ok := l.CPUSimdTileSizes[dtype]; ok && ts > 0 {
			return ts
		}
	}
	if l.Type == LayerDense {
		return CalculateOptimalDenseSimdTileSizeForLayer(l, dtype)
	}
	return l.GetCPUTileSize(dtype)
}

// EnsureRuntimeTileSizes populates per-dtype CPU/SIMD tile maps when missing.
func (l *VolumetricLayer) EnsureRuntimeTileSizes() {
	if l == nil {
		return
	}
	needRefresh := l.CPUTileSizes == nil
	if l.Type == LayerDense && l.CPUSimdTileSizes == nil {
		needRefresh = true
	}
	if needRefresh {
		l.refreshRuntimeCPUTileSizes()
	}
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

func (l *VolumetricLayer) refreshRuntimeCPUTileSizes() {
	l.CPUTileSizes = make(map[DType]int, len(allDTypes))
	if l.Type == LayerDense {
		l.CPUSimdTileSizes = make(map[DType]int, len(allDTypes))
	}
	for _, dtype := range allDTypes {
		var ts int
		switch l.Type {
		case LayerCNN1:
			ts = CalculateOptimalCNN1TileSizeForLayer(l, dtype)
		case LayerCNN2:
			ts = CalculateOptimalCNN2TileSize(l.InputChannels, dtype)
		case LayerCNN3:
			ts = CalculateOptimalCNN3TileSize(l.InputChannels, dtype)
		case LayerMultiHeadAttention:
			ts = CalculateOptimalTileSize(l.HeadDim, dtype)
		case LayerSwiGLU:
			ts = CalculateOptimalSwiGLUTileSize(l.InputHeight, dtype)
		case LayerDense:
			ts = CalculateOptimalDenseTileSize(l.InputHeight, dtype)
			ts = capDenseTileToLayer(ts, l.InputHeight, l.OutputHeight)
			simdTS := CalculateOptimalDenseSimdTileSizeForLayer(l, dtype)
			l.CPUSimdTileSizes[dtype] = capDenseTileToLayer(simdTS, l.InputHeight, l.OutputHeight)
		case LayerRNN:
			ts = CalculateOptimalRNNTileSize(l.InputHeight, l.OutputHeight, dtype)
		case LayerLSTM:
			ts = CalculateOptimalLSTMTileSize(l.InputHeight, l.OutputHeight, dtype)
		case LayerEmbedding:
			ts = CalculateOptimalEmbeddingTileSize(l.EmbeddingDim, dtype)
		case LayerResidual:
			ts = CalculateOptimalResidualTileSize(dtype)
		default:
			ts = 8
		}
		l.CPUTileSizes[dtype] = ts
	}
	if l.TileSize <= 0 {
		l.TileSize = l.GetCPUTileSize(l.DType)
	}
}

func (l *VolumetricLayer) refreshRuntimeGPUTileSizes() {
	if l.Network == nil || l.Network.GPUContext == nil {
		return
	}
	l.GPUSCTileSizes = make(map[DType]int, len(allDTypes))
	l.GPUMCTileSizes = make(map[DType]int, len(allDTypes))
	for _, dtype := range allDTypes {
		var sc, mc int
		switch l.Type {
		case LayerCNN1:
			sc, mc = CNN1GPUTileSizesForLayer(l.Network.GPUContext, l, dtype)
		case LayerMultiHeadAttention:
			sc, mc = MHAGPUTileSizes(l.Network.GPUContext, l.HeadDim, dtype)
		case LayerSwiGLU:
			sc, mc = SwiGLUGPUTileSizes(l.Network.GPUContext, dtype)
		default:
			sc, mc = cnnGPUTileSizesFromContext(l.Network.GPUContext, dtype)
		}
		l.GPUSCTileSizes[dtype] = sc
		l.GPUMCTileSizes[dtype] = mc
	}
}

func (n *VolumetricNetwork) RefreshRuntimeTileSizes() {
	var walk func(*VolumetricLayer)
	walk = func(layer *VolumetricLayer) {
		if layer == nil {
			return
		}
		layer.refreshRuntimeCPUTileSizes()
		layer.refreshRuntimeGPUTileSizes()
		for i := range layer.ParallelBranches {
			walk(&layer.ParallelBranches[i])
		}
		for i := range layer.SequentialLayers {
			walk(&layer.SequentialLayers[i])
		}
		if layer.FilterGateConfig != nil {
			walk(layer.FilterGateConfig)
		}
	}
	for i := range n.Layers {
		walk(&n.Layers[i])
	}
}
