package poly

import (
	"math"
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

const bitNetTernaryPackedKeyBase DType = 10000
const bitNetTernaryPackedScaleKeyBase DType = 20000

// BitNetTernaryMatrix is a row-major {-1, 0, +1} matrix packed as 16 weights
// per u32. Scale is applied once after the add/subtract-only dot product.
type BitNetTernaryMatrix struct {
	Rows     int
	Cols     int
	RowWords int
	Offset   int
	Scale    float32
	Words    []uint32

	// Codes holds one unsigned 2-bit code {0,1,2} per weight (ternary = code-1),
	// row-major with each row zero-padded to RowStride bytes. Built once for the
	// AVX2 MAD path so inference never unpacks 2-bit weights on the fly.
	Codes     []uint8
	RowStride int

	// PackedStride holds the 2-bit codes still packed (4/byte), row-major with each
	// row zero-padded to PackedBlocks*16 bytes (64 codes per 16-byte block). The
	// arm64 NEON kernel reads this directly, streaming 4x less weight memory than
	// Codes — the win on bandwidth-bound decode. Built lazily instead of Codes.
	PackedStride []uint8
	PackedBlocks int

	// TL1Nibbles holds microsoft/BitNet TL1 4-bit weight-pair indices (two per
	// byte, high nibble first) per row, padded to TL1PairStride bytes. Built once
	// for the LUT matvec path on arm64.
	TL1Nibbles    []uint8
	TL1PairStride int
	TL1TailCode   []uint8 // per-row code for odd final column, or 1 (=ternary 0)
}

func bitNetActivationMaxAbs[T Numeric](input []T) float32 {
	maxAbs := float32(0)
	for _, v := range input {
		a := float32(math.Abs(float64(v)))
		if a > maxAbs {
			maxAbs = a
		}
	}
	if maxAbs < 1e-5 {
		return 1e-5
	}
	return maxAbs
}

func bitNetActivationMaxAbsFloat64(input []float64) float32 {
	maxAbs := float32(0)
	for _, v := range input {
		a := float32(math.Abs(v))
		if a > maxAbs {
			maxAbs = a
		}
	}
	if maxAbs < 1e-5 {
		return 1e-5
	}
	return maxAbs
}

func bitNetQuantizedActivation[T Numeric](v T, maxAbs float32) int8 {
	scale := 127.0 / float64(maxAbs)
	q := int(math.Round(float64(v) * scale))
	if q < -128 {
		q = -128
	}
	if q > 127 {
		q = 127
	}
	return int8(q)
}

func bitNetQuantizedActivationFloat64(v float64, maxAbs float32) int8 {
	scale := 127.0 / float64(maxAbs)
	q := int(math.Round(v * scale))
	if q < -128 {
		q = -128
	}
	if q > 127 {
		q = 127
	}
	return int8(q)
}

func bitNetQuantizeActivationNumeric[T Numeric](input []T, buf []int8) ([]int8, float32) {
	if len(buf) < len(input) {
		buf = make([]int8, len(input))
	}
	activationMax := bitNetActivationMaxAbs(input)
	for i, v := range input {
		buf[i] = bitNetQuantizedActivation(v, activationMax)
	}
	return buf[:len(input)], activationMax
}

func bitNetQuantizeActivationFloat64(input []float64, buf []int8) ([]int8, float32) {
	if len(buf) < len(input) {
		buf = make([]int8, len(input))
	}
	activationMax := bitNetActivationMaxAbsFloat64(input)
	for i, v := range input {
		buf[i] = bitNetQuantizedActivationFloat64(v, activationMax)
	}
	return buf[:len(input)], activationMax
}

func bitNetTernaryScale(weights []float32) float32 {
	if len(weights) == 0 {
		return 1.0
	}
	var sumAbs float64
	for _, v := range weights {
		sumAbs += math.Abs(float64(v))
	}
	scale := float32(sumAbs / float64(len(weights)))
	if scale == 0 {
		return 1.0
	}
	return scale
}

func bitNetQuantValue(v, scale float32) uint8 {
	if scale == 0 {
		scale = 1.0
	}
	q := int(math.Round(float64(v / scale)))
	if q > 1 {
		q = 1
	}
	if q < -1 {
		q = -1
	}
	return uint8(int8(q))
}

// MorphBitNetTernary converts this weight store to the BitNet b1.58 absmean
// ternary representation. It is explicit so existing DTypeTernary behavior does
// not change for callers that rely on the older max-abs morphing path.
func (ws *WeightStore) MorphBitNetTernary() {
	if ws == nil {
		return
	}
	scale := bitNetTernaryScale(ws.Master)
	w := make([]uint8, len(ws.Master))
	for i, v := range ws.Master {
		w[i] = bitNetQuantValue(v, scale)
	}
	ws.Scale = scale
	ws.Versions[DTypeTernary] = w
	if ws.CPUPacked != nil {
		ws.CPUPacked = make(map[DType]any)
	}
	ws.GPUWeights = make(map[DType]any)
}

// MorphLayerBitNetTernary applies BitNet b1.58 ternary quantization to one
// layer and switches its execution dtype to DTypeTernary.
func MorphLayerBitNetTernary(layer *VolumetricLayer) error {
	if layer == nil || layer.WeightStore == nil {
		return nil
	}
	layer.WeightStore.MorphBitNetTernary()
	layer.DType = DTypeTernary
	return nil
}

// MorphLayerBitNetNativeTernary preserves the older unit-scale ternary path for
// experiments. BitNet HF checkpoints should use MorphLayerBitNetTernary instead
// because their BitLinear code dequantizes by absmean scale.
func MorphLayerBitNetNativeTernary(layer *VolumetricLayer) error {
	if layer == nil || layer.WeightStore == nil {
		return nil
	}
	layer.WeightStore.MorphBitNetTernary()
	if raw, ok := layer.WeightStore.Versions[DTypeTernary].([]uint8); ok {
		for i, v := range raw {
			layer.WeightStore.Master[i] = float32(int8(v))
		}
		layer.WeightStore.Scale = 1.0
		layer.WeightStore.Versions = make(map[DType]any)
		layer.WeightStore.CPUPacked = make(map[DType]any)
		layer.WeightStore.GPUWeights = make(map[DType]any)
	}
	layer.DType = DTypeTernary
	return nil
}

// MorphNetworkBitNetTernary quantizes linear/attention/MLP style weights while
// leaving normalization parameters in float32, matching common BitNet layouts.
func MorphNetworkBitNetTernary(n *VolumetricNetwork) error {
	if n == nil {
		return nil
	}
	for i := range n.Layers {
		if err := morphLayerTreeBitNetTernary(&n.Layers[i]); err != nil {
			return err
		}
	}
	return nil
}

// MorphNetworkBitNetNativeTernary preserves the older unit-scale ternary path
// for experiments. BitNet HF checkpoints should use MorphNetworkBitNetTernary.
func MorphNetworkBitNetNativeTernary(n *VolumetricNetwork) error {
	if n == nil {
		return nil
	}
	for i := range n.Layers {
		if err := morphLayerTreeBitNetNativeTernary(&n.Layers[i]); err != nil {
			return err
		}
	}
	return nil
}

func morphLayerTreeBitNetTernary(l *VolumetricLayer) error {
	if l == nil {
		return nil
	}
	switch l.Type {
	case LayerRMSNorm, LayerLayerNorm, LayerSoftmax, LayerResidual:
		// Keep scale-only/non-weight projection layers in their existing dtype.
	default:
		if err := MorphLayerBitNetTernary(l); err != nil {
			return err
		}
	}
	for i := range l.ParallelBranches {
		if err := morphLayerTreeBitNetTernary(&l.ParallelBranches[i]); err != nil {
			return err
		}
	}
	for i := range l.SequentialLayers {
		if err := morphLayerTreeBitNetTernary(&l.SequentialLayers[i]); err != nil {
			return err
		}
	}
	return nil
}

func morphLayerTreeBitNetNativeTernary(l *VolumetricLayer) error {
	if l == nil {
		return nil
	}
	switch l.Type {
	case LayerRMSNorm, LayerLayerNorm, LayerSoftmax, LayerResidual:
	default:
		if err := MorphLayerBitNetNativeTernary(l); err != nil {
			return err
		}
	}
	for i := range l.ParallelBranches {
		if err := morphLayerTreeBitNetNativeTernary(&l.ParallelBranches[i]); err != nil {
			return err
		}
	}
	for i := range l.SequentialLayers {
		if err := morphLayerTreeBitNetNativeTernary(&l.SequentialLayers[i]); err != nil {
			return err
		}
	}
	return nil
}

func usePackedTernaryCPU(layer *VolumetricLayer) bool {
	return layer != nil &&
		layer.Network != nil &&
		layer.Network.UseExactDType &&
		!layer.Network.UseGPU &&
		layer.DType == DTypeTernary &&
		layer.WeightStore != nil
}

func bitNetPackedKey(offset int) DType {
	return bitNetTernaryPackedKeyBase + DType(offset)
}

func bitNetPackedScaleKey(offset int) DType {
	return bitNetTernaryPackedScaleKeyBase + DType(offset)
}

func (ws *WeightStore) lookupBitNetTernaryPacked(offset, rows, cols int) *BitNetTernaryMatrix {
	if ws == nil || ws.CPUPacked == nil || rows <= 0 || cols <= 0 || offset < 0 {
		return nil
	}
	key := bitNetPackedKey(offset)
	if m, ok := ws.CPUPacked[key].(*BitNetTernaryMatrix); ok && m != nil &&
		m.Offset == offset && m.Rows == rows && m.Cols == cols {
		return m
	}
	for _, v := range ws.CPUPacked {
		if m, ok := v.(*BitNetTernaryMatrix); ok && m != nil &&
			m.Offset == offset && m.Rows == rows && m.Cols == cols {
			return m
		}
	}
	return nil
}

func (ws *WeightStore) GetBitNetTernaryMatrix(offset, rows, cols int) (*BitNetTernaryMatrix, bool) {
	if ws == nil || rows <= 0 || cols <= 0 || offset < 0 {
		return nil, false
	}
	total := rows * cols
	if ws.CPUPacked == nil {
		ws.CPUPacked = make(map[DType]any)
	}
	key := bitNetPackedKey(offset)
	// Prefer slabs installed directly from safetensors (microsoft/bitnet-b1.58
	// offline packed layout via SetMicrosoftBitNetPackedMatrix) — those never
	// populate Master[offset:offset+total] with a dense FP32 unfold.
	if m := ws.lookupBitNetTernaryPacked(offset, rows, cols); m != nil {
		return m, true
	}

	if raw, ok := ws.Versions[DTypeTernary].([]uint8); ok && offset+total <= len(raw) {
		scale := ws.Scale
		if scale == 0 {
			scale = 1
		}
		if m := ws.lookupBitNetTernaryPacked(offset, rows, cols); m != nil {
			return m, true
		}
		matrix, ok := packNativeTernaryToBitNetMatrix(raw[offset:offset+total], rows, cols, scale)
		if !ok {
			return nil, false
		}
		matrix.Offset = offset
		ws.CPUPacked[key] = matrix
		return matrix, true
	}

	if ws.Master == nil || offset+total > len(ws.Master) {
		return nil, false
	}
	matrix, ok := packFloat32AsBitNetTernaryMatrix(ws.Master[offset:offset+total], rows, cols)
	if !ok {
		return nil, false
	}
	matrix.Offset = offset
	ws.CPUPacked[key] = matrix
	return matrix, true
}

// PrepareNetworkBitNetTernaryCPU prepares decoder BitLinear weights for CPU
// inference: either packs from FP32 Master (HF float weights) or keeps slabs
// already decoded from microsoft/bitnet-b1.58 offline safetensors via
// SetMicrosoftBitNetPackedMatrix. Embeddings / RMSNorm stay as loaded.
func PrepareNetworkBitNetTernaryCPU(n *VolumetricNetwork) error {
	if n == nil {
		return nil
	}
	for i := range n.Layers {
		if err := prepareLayerTreeBitNetTernaryCPU(&n.Layers[i]); err != nil {
			return err
		}
	}
	return nil
}

// PrepareDecoderBlockBitNetTernaryCPU prepares one 4-layer HF decoder block
// (RMSNorm, MHA, RMSNorm, SwiGLU). Releases FP32 Master only after successful pack
// so offline-packed weights never get re-quantized from an empty Master span.
func PrepareDecoderBlockBitNetTernaryCPU(n *VolumetricNetwork, blockIdx int) error {
	if n == nil || blockIdx < 0 {
		return nil
	}
	base := blockIdx * 4
	if base >= len(n.Layers) {
		return nil
	}
	end := base + 4
	if end > len(n.Layers) {
		end = len(n.Layers)
	}
	for i := base; i < end; i++ {
		if err := prepareLayerTreeBitNetTernaryCPU(&n.Layers[i]); err != nil {
			return err
		}
	}
	return nil
}

func prepareLayerTreeBitNetTernaryCPU(l *VolumetricLayer) error {
	if l == nil {
		return nil
	}
	switch l.Type {
	case LayerDense:
		if l.WeightStore != nil {
			if _, ok := l.WeightStore.GetBitNetTernaryMatrix(0, l.OutputHeight, l.InputHeight); ok {
				l.DType = DTypeTernary
				releaseBitNetPackedProjectionMaster(l.WeightStore)
			}
		}
	case LayerMultiHeadAttention:
		prepareBitNetMHA(l)
	case LayerSwiGLU:
		prepareBitNetSwiGLU(l)
	case LayerRMSNorm, LayerLayerNorm, LayerSoftmax, LayerResidual:
		// Keep non-BitLinear layers as-is.
	default:
		if l.WeightStore != nil {
			l.DType = DTypeTernary
		}
	}
	for i := range l.ParallelBranches {
		if err := prepareLayerTreeBitNetTernaryCPU(&l.ParallelBranches[i]); err != nil {
			return err
		}
	}
	for i := range l.SequentialLayers {
		if err := prepareLayerTreeBitNetTernaryCPU(&l.SequentialLayers[i]); err != nil {
			return err
		}
	}
	return nil
}

// EnsureBitNetMHAWeights builds packed ternary matrices for MHA Q/K/V/O (CPU/GPU BitNet paths).
func EnsureBitNetMHAWeights(l *VolumetricLayer) {
	prepareBitNetMHA(l)
}

func prepareBitNetMHA(l *VolumetricLayer) {
	if l == nil || l.WeightStore == nil {
		return
	}
	dModel := l.DModel
	numHeads := l.NumHeads
	numKVHeads := l.NumKVHeads
	if numKVHeads == 0 {
		numKVHeads = numHeads
	}
	headDim := l.HeadDim
	qDim := l.QueryDim
	if qDim == 0 {
		qDim = numHeads * headDim
	}
	kvDim := numKVHeads * headDim
	qwStart := 0
	kwStart := qwStart + qDim*dModel
	vwStart := kwStart + kvDim*dModel
	owStart := vwStart + kvDim*dModel
	qbStart := owStart + dModel*qDim
	obEnd := qbStart + qDim + kvDim + kvDim + dModel
	l.DType = DTypeTernary
	l.WeightStore.GetBitNetTernaryMatrix(qwStart, qDim, dModel)
	l.WeightStore.GetBitNetTernaryMatrix(kwStart, kvDim, dModel)
	l.WeightStore.GetBitNetTernaryMatrix(vwStart, kvDim, dModel)
	l.WeightStore.GetBitNetTernaryMatrix(owStart, dModel, qDim)
	if bitNetBiasTailIsZero(l.WeightStore, qbStart, obEnd) {
		releaseBitNetPackedProjectionMaster(l.WeightStore)
	}
}

func prepareBitNetSwiGLU(l *VolumetricLayer) {
	if l == nil || l.WeightStore == nil {
		return
	}
	inputSize, intermediateSize := l.InputHeight, l.OutputHeight
	wSize := inputSize * intermediateSize
	gateWStart := 0
	upWStart := wSize
	downWStart := 2 * wSize
	gateBStart := 3 * wSize
	downBEnd := gateBStart + intermediateSize + intermediateSize + inputSize
	l.DType = DTypeTernary
	l.WeightStore.GetBitNetTernaryMatrix(gateWStart, intermediateSize, inputSize)
	l.WeightStore.GetBitNetTernaryMatrix(upWStart, intermediateSize, inputSize)
	l.WeightStore.GetBitNetTernaryMatrix(downWStart, inputSize, intermediateSize)
	if bitNetBiasTailIsZero(l.WeightStore, gateBStart, downBEnd) {
		releaseBitNetPackedProjectionMaster(l.WeightStore)
	}
}

func bitNetBiasTailIsZero(ws *WeightStore, start, end int) bool {
	if ws == nil || start < 0 || end > len(ws.Master) || start > end {
		return false
	}
	for _, v := range ws.Master[start:end] {
		if v != 0 {
			return false
		}
	}
	return true
}

func releaseBitNetPackedProjectionMaster(ws *WeightStore) {
	if ws == nil {
		return
	}
	ws.Master = nil
	ws.Versions = make(map[DType]any)
	ws.GPUWeights = make(map[DType]any)
}

func ternaryStorageToCode(v uint8) uint8 {
	switch int8(v) {
	case -1:
		return 0
	case 1:
		return 2
	default:
		return 1
	}
}

func bitNetTernaryBias(ws *WeightStore, idx int) float64 {
	if ws == nil || idx < 0 || idx >= len(ws.Master) {
		return 0
	}
	return float64(ws.Master[idx])
}

func bitNetTernaryMatVecNumeric[T Numeric](matrix *BitNetTernaryMatrix, input []T, out []float64) bool {
	if matrix == nil || len(input) < matrix.Cols || len(out) < matrix.Rows {
		return false
	}
	xq, activationMax := bitNetQuantizeActivationNumeric(input[:matrix.Cols], nil)
	return bitNetTernaryMatVecQuantized(matrix, xq, activationMax, out)
}

func bitNetTernaryMatVecQuantized(matrix *BitNetTernaryMatrix, xq []int8, activationMax float32, out []float64) bool {
	if matrix == nil || len(xq) < matrix.Cols || len(out) < matrix.Rows {
		return false
	}
	outputScale := float64(matrix.Scale) * float64(activationMax) / 127.0
	if outputScale == 0 {
		outputScale = 1.0
	}
	bitNetTernaryMatVecInt8(matrix, xq, outputScale, out)
	return true
}

func bitNetTernaryMatVecInt8(matrix *BitNetTernaryMatrix, xq []int8, outputScale float64, out []float64) {
	if BitNetTernarySimdActive() {
		// Packed-2-bit MAD kernel (default on arm64 — fastest today).
		if simd.BitNetPackedAvailable() && ensureBitNetPacked(matrix) {
			bitNetTernaryMatVecInt8PackedSimd(matrix, xq, outputScale, out)
			return
		}
		// TL1 LUT path (opt-in via SetBitNetTL1Forward): microsoft/BitNet lookup+add.
		if simd.BitNetTL1Active() && simd.BitNetTL1Available() && ensureBitNetTL1(matrix) {
			bitNetTernaryMatVecInt8TL1(matrix, xq, outputScale, out)
			return
		}
		if ensureBitNetCodes(matrix) {
			bitNetTernaryMatVecInt8Simd(matrix, xq, outputScale, out)
			return
		}
	}
	bitNetTernaryMatVecInt8Scalar(matrix, xq, outputScale, out)
}

// ensureBitNetPacked lazily builds the strided 2-bit packed weights the NEON
// packed kernel reads. Each row is the row's packed uint32 words re-laid as
// little-endian bytes and zero-padded to a whole number of 64-column (16-byte)
// blocks. Built once per matrix, then reused every token.
func ensureBitNetPacked(matrix *BitNetTernaryMatrix) bool {
	if matrix == nil || matrix.Rows <= 0 || matrix.Cols <= 0 {
		return false
	}
	if len(matrix.PackedStride) > 0 && matrix.PackedBlocks > 0 {
		return true
	}
	rowWords := matrix.RowWords
	if rowWords <= 0 {
		rowWords = (matrix.Cols + 15) / 16
	}
	if len(matrix.Words) < matrix.Rows*rowWords {
		return false
	}
	blocks := (rowWords + 3) / 4 // 4 words = 16 bytes = 64 codes per block
	strideBytes := blocks * 16
	packed := make([]uint8, matrix.Rows*strideBytes)
	for r := 0; r < matrix.Rows; r++ {
		wordBase := r * rowWords
		dstBase := r * strideBytes
		for w := 0; w < rowWords; w++ {
			word := matrix.Words[wordBase+w]
			b := dstBase + w*4
			packed[b+0] = uint8(word)
			packed[b+1] = uint8(word >> 8)
			packed[b+2] = uint8(word >> 16)
			packed[b+3] = uint8(word >> 24)
		}
		// bytes [rowWords*4 : strideBytes] stay 0; paired with zero-padded
		// activations they contribute nothing to sum(code*act) or sum(act).
	}
	matrix.PackedStride = packed
	matrix.PackedBlocks = blocks
	return true
}

// bitNetTernaryMatVecInt8PackedSimd runs the packed BitNet MAD kernel:
// dot = sum(code*act) - sum(act), since each ternary weight equals code-1.
func bitNetTernaryMatVecInt8PackedSimd(matrix *BitNetTernaryMatrix, xq []int8, outputScale float64, out []float64) {
	blocks := matrix.PackedBlocks
	strideBytes := blocks * 16
	acts := make([]int8, blocks*64) // zero-padded past Cols
	copy(acts, xq[:matrix.Cols])
	var actSum int32
	for i := 0; i < matrix.Cols; i++ {
		actSum += int32(xq[i])
	}
	packed := matrix.PackedStride

	worker := func(start, end int) {
		for r := start; r < end; r++ {
			raw := simd.BitNetTernaryPackedRowDot(packed[r*strideBytes:(r+1)*strideBytes], acts, blocks)
			out[r] = float64(raw-actSum) * outputScale
		}
	}
	bitNetRunRows(matrix.Rows, matrix.Cols, worker)
}

// ensureBitNetTL1 lazily packs each row's 2-bit codes into TL1 4-bit pair indices
// (two per byte). Odd column counts store the last code in TL1TailCode per row.
func ensureBitNetTL1(matrix *BitNetTernaryMatrix) bool {
	if matrix == nil || matrix.Rows <= 0 || matrix.Cols <= 0 {
		return false
	}
	if len(matrix.TL1Nibbles) > 0 && matrix.TL1PairStride > 0 {
		return true
	}
	rowWords := matrix.RowWords
	if rowWords <= 0 {
		rowWords = (matrix.Cols + 15) / 16
	}
	if len(matrix.Words) < matrix.Rows*rowWords {
		return false
	}
	pairCount := matrix.Cols / 2
	stride := (pairCount + 1) / 2
	nibbles := make([]uint8, matrix.Rows*stride)
	tails := make([]uint8, matrix.Rows)
	for r := 0; r < matrix.Rows; r++ {
		wordBase := r * rowWords
		dstBase := r * stride
		c := 0
		for c+1 < matrix.Cols {
			code0 := uint8((matrix.Words[wordBase+c/16] >> uint((c%16)*2)) & 0x03)
			code1 := uint8((matrix.Words[wordBase+(c+1)/16] >> uint(((c+1)%16)*2)) & 0x03)
			idx := simd.TL1IndexFromCodes(code0, code1)
			pair := c / 2
			if pair&1 == 0 {
				nibbles[dstBase+pair/2] = idx << 4
			} else {
				nibbles[dstBase+pair/2] |= idx
			}
			c += 2
		}
		if matrix.Cols&1 == 1 {
			tails[r] = uint8((matrix.Words[wordBase+c/16] >> uint((c%16)*2)) & 0x03)
		} else {
			tails[r] = 1 // ternary 0 — tail term skipped
		}
	}
	matrix.TL1Nibbles = nibbles
	matrix.TL1PairStride = stride
	matrix.TL1TailCode = tails
	return true
}

// bitNetTernaryMatVecInt8TL1 runs the microsoft/BitNet TL1 LUT matvec: QLUT is
// built once from the quantized activation vector, then rows are processed in
// 16-wide batches (Microsoft TL1 M-dimension batching). Stays single-threaded —
// on Apple silicon this beats fan-out (goroutine overhead > gain).
func bitNetTernaryMatVecInt8TL1(matrix *BitNetTernaryMatrix, xq []int8, outputScale float64, out []float64) {
	fullPairs := matrix.Cols / 2
	qlut := make([]int16, fullPairs*16)
	simd.BuildBitNetTL1QLUT(xq, matrix.Cols, fullPairs, qlut)

	var tailAct int8
	if matrix.Cols&1 == 1 {
		tailAct = xq[matrix.Cols-1]
	}
	simd.BitNetTL1MatVecBatched(
		matrix.TL1Nibbles, matrix.TL1PairStride, matrix.Rows, matrix.Cols,
		qlut, matrix.TL1TailCode, tailAct, out, outputScale,
	)
}

// ensureBitNetCodes lazily unpacks the 2-bit packed weights into one unsigned
// byte per weight (row-padded to a multiple of 32) so the AVX2 MAD kernel never
// unpacks on the fly. Built once per matrix, then cached and reused every token.
func ensureBitNetCodes(matrix *BitNetTernaryMatrix) bool {
	if matrix == nil || matrix.Rows <= 0 || matrix.Cols <= 0 {
		return false
	}
	if len(matrix.Codes) > 0 && matrix.RowStride >= matrix.Cols {
		return true
	}
	rowWords := matrix.RowWords
	if rowWords <= 0 {
		rowWords = (matrix.Cols + 15) / 16
	}
	if len(matrix.Words) < matrix.Rows*rowWords {
		return false
	}
	stride := ((matrix.Cols + 31) / 32) * 32
	codes := make([]uint8, matrix.Rows*stride)
	for r := 0; r < matrix.Rows; r++ {
		wordBase := r * rowWords
		dstBase := r * stride
		for c := 0; c < matrix.Cols; c++ {
			word := matrix.Words[wordBase+c/16]
			codes[dstBase+c] = uint8((word >> uint((c%16)*2)) & 0x03)
		}
		// padded [Cols:stride] stay 0; paired with zero-padded activations they
		// contribute nothing to code*act or the sum(act) correction.
	}
	matrix.Codes = codes
	matrix.RowStride = stride
	return true
}

// bitNetTernaryMatVecInt8Simd runs the BitNet MAD kernel: dot = sum(code*act) -
// sum(act), since each ternary weight equals code-1.
func bitNetTernaryMatVecInt8Simd(matrix *BitNetTernaryMatrix, xq []int8, outputScale float64, out []float64) {
	stride := matrix.RowStride
	acts := make([]int8, stride)
	copy(acts, xq[:matrix.Cols])
	var actSum int32
	for i := 0; i < matrix.Cols; i++ {
		actSum += int32(xq[i])
	}
	codes := matrix.Codes

	worker := func(start, end int) {
		for r := start; r < end; r++ {
			raw := simd.BitNetTernaryCodeRowDot(codes[r*stride:(r+1)*stride], acts, stride)
			out[r] = float64(raw-actSum) * outputScale
		}
	}
	bitNetRunRows(matrix.Rows, matrix.Cols, worker)
}

func bitNetTernaryMatVecInt8Scalar(matrix *BitNetTernaryMatrix, xq []int8, outputScale float64, out []float64) {
	rowWords := matrix.RowWords
	if rowWords <= 0 {
		rowWords = (matrix.Cols + 15) / 16
	}
	fullWords := matrix.Cols / 16
	tailCols := matrix.Cols % 16

	worker := func(start, end int) {
		for r := start; r < end; r++ {
			wordBase := r * rowWords
			xOff := 0
			var sum int32
			for w := 0; w < fullWords; w++ {
				sum += bitNetTernaryWordDot16(matrix.Words[wordBase+w], xq[xOff:xOff+16])
				xOff += 16
			}
			if tailCols > 0 {
				sum += bitNetTernaryWordDotTail(matrix.Words[wordBase+fullWords], xq[xOff:xOff+tailCols], tailCols)
			}
			out[r] = float64(sum) * outputScale
		}
	}
	bitNetRunRows(matrix.Rows, matrix.Cols, worker)
}

// bitNetRunRows runs a row worker single-threaded for small matrices, otherwise
// fans out across GOMAXPROCS goroutines (shared by scalar + SIMD matvec paths).
func bitNetRunRows(rows, cols int, worker func(start, end int)) {
	work := rows * cols
	if work < 262144 || rows < 4 {
		worker(0, rows)
		return
	}
	workers := runtime.GOMAXPROCS(0)
	if workers > rows {
		workers = rows
	}
	if workers <= 1 {
		worker(0, rows)
		return
	}
	chunk := (rows + workers - 1) / workers
	var wg sync.WaitGroup
	for start := 0; start < rows; start += chunk {
		end := start + chunk
		if end > rows {
			end = rows
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			worker(start, end)
		}(start, end)
	}
	wg.Wait()
}

func bitNetTernaryWordDot16(word uint32, xq []int8) int32 {
	var sum int32
	code := word & 0x03
	sum += int32(xq[0]) * (int32(code) - 1)
	code = (word >> 2) & 0x03
	sum += int32(xq[1]) * (int32(code) - 1)
	code = (word >> 4) & 0x03
	sum += int32(xq[2]) * (int32(code) - 1)
	code = (word >> 6) & 0x03
	sum += int32(xq[3]) * (int32(code) - 1)
	code = (word >> 8) & 0x03
	sum += int32(xq[4]) * (int32(code) - 1)
	code = (word >> 10) & 0x03
	sum += int32(xq[5]) * (int32(code) - 1)
	code = (word >> 12) & 0x03
	sum += int32(xq[6]) * (int32(code) - 1)
	code = (word >> 14) & 0x03
	sum += int32(xq[7]) * (int32(code) - 1)
	code = (word >> 16) & 0x03
	sum += int32(xq[8]) * (int32(code) - 1)
	code = (word >> 18) & 0x03
	sum += int32(xq[9]) * (int32(code) - 1)
	code = (word >> 20) & 0x03
	sum += int32(xq[10]) * (int32(code) - 1)
	code = (word >> 22) & 0x03
	sum += int32(xq[11]) * (int32(code) - 1)
	code = (word >> 24) & 0x03
	sum += int32(xq[12]) * (int32(code) - 1)
	code = (word >> 26) & 0x03
	sum += int32(xq[13]) * (int32(code) - 1)
	code = (word >> 28) & 0x03
	sum += int32(xq[14]) * (int32(code) - 1)
	code = (word >> 30) & 0x03
	sum += int32(xq[15]) * (int32(code) - 1)
	return sum
}

func bitNetTernaryWordDotTail(word uint32, xq []int8, n int) int32 {
	var sum int32
	for i := 0; i < n; i++ {
		code := (word >> uint(i*2)) & 0x03
		sum += int32(xq[i]) * (int32(code) - 1)
	}
	return sum
}

func bitNetTernaryMatVecFloat64(matrix *BitNetTernaryMatrix, input []float64, out []float64) bool {
	if matrix == nil || len(input) < matrix.Cols || len(out) < matrix.Rows {
		return false
	}
	xq, activationMax := bitNetQuantizeActivationFloat64(input[:matrix.Cols], nil)
	return bitNetTernaryMatVecQuantized(matrix, xq, activationMax, out)
}

func bitNetRMSNormFloat64(data []float64, eps float64) {
	bitNetRMSNormFloat64Weighted(data, nil, eps)
}

func bitNetRMSNormFloat64Weighted(data []float64, weight []float32, eps float64) {
	if len(data) == 0 {
		return
	}
	if eps <= 0 {
		eps = 1e-5
	}
	var sumSq float64
	for _, v := range data {
		sumSq += v * v
	}
	scale := 1.0 / math.Sqrt(sumSq/float64(len(data))+eps)
	for i := range data {
		w := 1.0
		if i < len(weight) {
			w = float64(weight[i])
		}
		data[i] *= scale * w
	}
}

func bitNetRMSNormTensorRow[T Numeric](data []T, eps float64) {
	bitNetRMSNormTensorRowWeighted(data, nil, eps)
}

func bitNetRMSNormTensorRowWeighted[T Numeric](data []T, weight []float32, eps float64) {
	if len(data) == 0 {
		return
	}
	if eps <= 0 {
		eps = 1e-5
	}
	var sumSq float64
	for _, v := range data {
		f := float64(v)
		sumSq += f * f
	}
	scale := 1.0 / math.Sqrt(sumSq/float64(len(data))+eps)
	for i, v := range data {
		w := 1.0
		if i < len(weight) {
			w = float64(weight[i])
		}
		data[i] = T(float64(v) * scale * w)
	}
}

// packNativeTernaryToBitNetMatrix builds a CPU matmul slab from persisted []uint8
// {-1,0,1} codes. Forward must use this path after save/reload so execution matches
// the native blob, not a re-quantize of Master*scale.
func packNativeTernaryToBitNetMatrix(native []uint8, rows, cols int, scale float32) (*BitNetTernaryMatrix, bool) {
	if rows <= 0 || cols <= 0 || rows*cols > len(native) {
		return nil, false
	}
	if scale == 0 {
		scale = 1
	}
	return &BitNetTernaryMatrix{
		Rows:     rows,
		Cols:     cols,
		RowWords: (cols + 15) / 16,
		Scale:    scale,
		Words:    packTernaryRowsToU32(native, rows, cols),
	}, true
}

func packFloat32AsBitNetTernaryMatrix(weights []float32, rows, cols int) (*BitNetTernaryMatrix, bool) {
	if rows <= 0 || cols <= 0 || rows*cols > len(weights) {
		return nil, false
	}
	total := rows * cols
	scale := bitNetTernaryScale(weights[:total])
	raw := make([]uint8, total)
	alreadyTernary := true
	for i := 0; i < total; i++ {
		v := weights[i]
		if !(math.Abs(float64(v)) < 1e-6 || math.Abs(float64(v-1)) < 1e-6 || math.Abs(float64(v+1)) < 1e-6) {
			alreadyTernary = false
		}
		raw[i] = bitNetQuantValue(v, scale)
	}
	if alreadyTernary {
		scale = 1.0
		for i := 0; i < total; i++ {
			raw[i] = bitNetQuantValue(weights[i], 1.0)
		}
	}
	return &BitNetTernaryMatrix{
		Rows:     rows,
		Cols:     cols,
		RowWords: (cols + 15) / 16,
		Scale:    scale,
		Words:    packTernaryRowsToU32(raw, rows, cols),
	}, true
}

func packTernaryRowsToU32(data []uint8, rows, cols int) []uint32 {
	rowWords := (cols + 15) / 16
	packed := make([]uint32, rows*rowWords)
	for r := 0; r < rows; r++ {
		rowOff := r * cols
		wordOff := r * rowWords
		for c := 0; c < cols; c++ {
			code := ternaryStorageToCode(data[rowOff+c])
			shift := uint((c % 16) * 2)
			packed[wordOff+c/16] |= uint32(code) << shift
		}
	}
	return packed
}

func (ws *WeightStore) SetBitNetPackedScale(offset int, scale float32) {
	if ws == nil {
		return
	}
	if scale == 0 {
		scale = 1.0
	}
	if ws.CPUPacked == nil {
		ws.CPUPacked = make(map[DType]any)
	}
	ws.CPUPacked[bitNetPackedScaleKey(offset)] = scale
	if matrix, ok := ws.CPUPacked[bitNetPackedKey(offset)].(*BitNetTernaryMatrix); ok && matrix != nil {
		matrix.Scale = scale
	}
}

func (ws *WeightStore) bitNetPackedScale(offset int) float32 {
	if ws == nil || ws.CPUPacked == nil {
		return 1.0
	}
	if scale, ok := ws.CPUPacked[bitNetPackedScaleKey(offset)].(float32); ok && scale != 0 {
		return scale
	}
	return 1.0
}

func (ws *WeightStore) SetMicrosoftBitNetPackedMatrix(offset, rows, cols int, packed []float32) bool {
	if ws == nil || rows <= 0 || cols <= 0 || offset < 0 || len(packed) == 0 {
		return false
	}
	if rows%4 != 0 || len(packed) != (rows/4)*cols {
		return false
	}
	if ws.CPUPacked == nil {
		ws.CPUPacked = make(map[DType]any)
	}
	words := packMicrosoftOfflineBitNetRowsToU32FromFloatU8Slots(packed, rows, cols)
	matrix := &BitNetTernaryMatrix{
		Rows:     rows,
		Cols:     cols,
		RowWords: (cols + 15) / 16,
		Offset:   offset,
		Scale:    ws.bitNetPackedScale(offset),
		Words:    words,
	}
	ws.CPUPacked[bitNetPackedKey(offset)] = matrix
	return true
}

// SetMicrosoftBitNetPackedMatrixBytes installs microsoft/bitnet-b1.58 offline-packed
// weights from raw U8 (or any per-cell byte payload) without an intermediate []float32.
// len(packed) must equal (rows/4)*cols and rows must be divisible by 4.
func (ws *WeightStore) SetMicrosoftBitNetPackedMatrixBytes(offset, rows, cols int, packed []byte) bool {
	if ws == nil || rows <= 0 || cols <= 0 || offset < 0 || len(packed) == 0 {
		return false
	}
	if rows%4 != 0 || len(packed) != (rows/4)*cols {
		return false
	}
	if ws.CPUPacked == nil {
		ws.CPUPacked = make(map[DType]any)
	}
	words := packMicrosoftOfflineBitNetRowsToU32FromBytes(packed, rows, cols)
	matrix := &BitNetTernaryMatrix{
		Rows:     rows,
		Cols:     cols,
		RowWords: (cols + 15) / 16,
		Offset:   offset,
		Scale:    ws.bitNetPackedScale(offset),
		Words:    words,
	}
	ws.CPUPacked[bitNetPackedKey(offset)] = matrix
	return true
}

// packMicrosoftOfflineBitNetRowsToU32FromFloatU8Slots matches HF checkpoints that decode
// U8 tensors into []float32 (values 0–255) before packing.
func packMicrosoftOfflineBitNetRowsToU32FromFloatU8Slots(packed []float32, rows, cols int) []uint32 {
	rowWords := (cols + 15) / 16
	out := make([]uint32, rows*rowWords)
	packedRows := rows / 4
	for pr := 0; pr < packedRows; pr++ {
		for c := 0; c < cols; c++ {
			b := uint8(packed[pr*cols+c])
			for lane := 0; lane < 4; lane++ {
				code := (b >> uint(lane*2)) & 0x03
				if code > 2 {
					code = 1
				}
				row := lane*packedRows + pr
				wordOff := row*rowWords + c/16
				shift := uint((c % 16) * 2)
				out[wordOff] |= uint32(code) << shift
			}
		}
	}
	return out
}

func packMicrosoftOfflineBitNetRowsToU32FromBytes(packed []byte, rows, cols int) []uint32 {
	rowWords := (cols + 15) / 16
	out := make([]uint32, rows*rowWords)
	packedRows := rows / 4
	for pr := 0; pr < packedRows; pr++ {
		for c := 0; c < cols; c++ {
			b := packed[pr*cols+c]
			for lane := 0; lane < 4; lane++ {
				code := (b >> uint(lane*2)) & 0x03
				if code > 2 {
					code = 1
				}
				row := lane*packedRows + pr
				wordOff := row*rowWords + c/16
				shift := uint((c % 16) * 2)
				out[wordOff] |= uint32(code) << shift
			}
		}
	}
	return out
}
