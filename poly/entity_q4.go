package poly

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/openfluke/webgpu/wgpu"
)

const entityBlobDTypeQ4_0 = "Q4_0"

// Q4_0 CPU-side cache: exact GPU upload bytes baked into .entity (no re-QuantizeQ4_0 on load).
func (ws *WeightStore) ensureQ4_0Maps() {
	if ws == nil {
		return
	}
	if ws.Q4_0Scales == nil {
		ws.Q4_0Scales = make(map[DType][]float32)
	}
	if ws.Q4_0Packed == nil {
		ws.Q4_0Packed = make(map[DType][]uint32)
	}
}

func (ws *WeightStore) SetQ4_0Component(key DType, scales []float32, packed []uint32) {
	ws.ensureQ4_0Maps()
	if len(scales) == 0 || len(packed) == 0 {
		return
	}
	ws.Q4_0Scales[key] = append([]float32(nil), scales...)
	ws.Q4_0Packed[key] = append([]uint32(nil), packed...)
}

func (ws *WeightStore) HasQ4_0Component(key DType) bool {
	if ws == nil {
		return false
	}
	p, ok := ws.Q4_0Packed[key]
	return ok && len(p) > 0
}

func (ws *WeightStore) HasAnyQ4_0() bool {
	if ws == nil || len(ws.Q4_0Packed) == 0 {
		return false
	}
	for _, p := range ws.Q4_0Packed {
		if len(p) > 0 {
			return true
		}
	}
	return false
}

// PackQ4_0GPU matches syncQuantizedComponent packing (32-weight blocks, 512-word align).
func PackQ4_0GPU(data []float32) (scales []float32, packed []uint32) {
	blocks := QuantizeQ4_0(data)
	numBlocks := len(blocks)
	packedSize := numBlocks * 4
	alignedSize := (packedSize + 63) &^ 63
	if alignedSize < 512 {
		alignedSize = 512
	}
	scales = make([]float32, numBlocks)
	packed = make([]uint32, alignedSize)
	for i, b := range blocks {
		scales[i] = b.Scale
		for j := 0; j < 4; j++ {
			packed[i*4+j] = uint32(b.Weights[j*4]) |
				(uint32(b.Weights[j*4+1]) << 8) |
				(uint32(b.Weights[j*4+2]) << 16) |
				(uint32(b.Weights[j*4+3]) << 24)
		}
	}
	return scales, packed
}

func encodeEntityQ4_0Blob(scales []float32, packed []uint32) []byte {
	raw := make([]byte, 8+len(scales)*4+len(packed)*4)
	binary.LittleEndian.PutUint32(raw[0:4], uint32(len(scales)))
	binary.LittleEndian.PutUint32(raw[4:8], uint32(len(packed)))
	off := 8
	for _, s := range scales {
		binary.LittleEndian.PutUint32(raw[off:off+4], math.Float32bits(s))
		off += 4
	}
	for _, w := range packed {
		binary.LittleEndian.PutUint32(raw[off:off+4], w)
		off += 4
	}
	return raw
}

func decodeEntityQ4_0Blob(raw []byte) (scales []float32, packed []uint32, err error) {
	if len(raw) < 8 {
		return nil, nil, fmt.Errorf("q4_0 blob too short")
	}
	nScales := int(binary.LittleEndian.Uint32(raw[0:4]))
	nPacked := int(binary.LittleEndian.Uint32(raw[4:8]))
	need := 8 + nScales*4 + nPacked*4
	if len(raw) < need {
		return nil, nil, fmt.Errorf("q4_0 blob truncated: have %d need %d", len(raw), need)
	}
	scales = make([]float32, nScales)
	off := 8
	for i := range scales {
		scales[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[off : off+4]))
		off += 4
	}
	packed = make([]uint32, nPacked)
	for i := range packed {
		packed[i] = binary.LittleEndian.Uint32(raw[off : off+4])
		off += 4
	}
	return scales, packed, nil
}

func appendEntityQ4_0Blob(path string, key DType, data []float32, payload *bytes.Buffer, blobs *[]EntityWeightBlob) {
	if len(data) == 0 {
		return
	}
	scales, packed := PackQ4_0GPU(data)
	raw := encodeEntityQ4_0Blob(scales, packed)
	offset := payload.Len()
	payload.Write(raw)
	*blobs = append(*blobs, EntityWeightBlob{
		Path:   fmt.Sprintf("%s.q4_0.%d", path, int(key)),
		Offset: uint64(offset),
		Length: uint64(len(raw)),
		DType:  entityBlobDTypeQ4_0,
		Native: true,
	})
}

func collectEntityQ4_0Layer(l *VolumetricLayer, path string, payload *bytes.Buffer, blobs *[]EntityWeightBlob) {
	if l.WeightStore == nil || len(l.WeightStore.Master) == 0 {
		return
	}
	switch l.Type {
	case LayerMultiHeadAttention:
		d := l.DModel
		q := l.QueryDim
		if q == 0 {
			q = d
		}
		kv := l.NumKVHeads * l.HeadDim
		qwSize := q * d
		kwSize := d * kv
		vwSize := d * kv
		owSize := d * q
		w := l.WeightStore.Master
		if len(w) < qwSize+kwSize+vwSize+owSize {
			return
		}
		appendEntityQ4_0Blob(path, WeightMHAQuery, w[0:qwSize], payload, blobs)
		appendEntityQ4_0Blob(path, WeightMHAKey, w[qwSize:qwSize+kwSize], payload, blobs)
		appendEntityQ4_0Blob(path, WeightMHAValue, w[qwSize+kwSize:qwSize+kwSize+vwSize], payload, blobs)
		appendEntityQ4_0Blob(path, WeightMHAProjection, w[qwSize+kwSize+vwSize:qwSize+kwSize+vwSize+owSize], payload, blobs)
		collectEntityMHANormBlobs(l, path, payload, blobs)
	case LayerSwiGLU:
		h, inter := l.InputHeight, l.OutputHeight
		wSize := h * inter
		w := l.WeightStore.Master
		if len(w) < 3*wSize+2*inter+h {
			return
		}
		appendEntityQ4_0Blob(path, DType(100), w[0:wSize], payload, blobs)
		appendEntityQ4_0Blob(path, DType(101), w[wSize:2*wSize], payload, blobs)
		appendEntityQ4_0Blob(path, DType(102), w[2*wSize:3*wSize], payload, blobs)
		// FP32 biases (GPU path reads these from Master tail).
		bias := append([]float32(nil), w[3*wSize:3*wSize+2*inter+h]...)
		raw := EncodeWeightsRaw(bias)
		offset := payload.Len()
		payload.Write(raw)
		*blobs = append(*blobs, EntityWeightBlob{
			Path:   path + ".biases",
			Offset: uint64(offset),
			Length: uint64(len(raw)),
			DType:  DTypeFloat32.String(),
			Native: false,
		})
	}
}

func collectEntityMHANormBlobs(l *VolumetricLayer, path string, payload *bytes.Buffer, blobs *[]EntityWeightBlob) {
	appendEntityFP32AuxBlob(path+".q_norm", l.QNormWeight, payload, blobs)
	appendEntityFP32AuxBlob(path+".k_norm", l.KNormWeight, payload, blobs)
}

func appendEntityFP32AuxBlob(path string, weights []float32, payload *bytes.Buffer, blobs *[]EntityWeightBlob) {
	if len(weights) == 0 {
		return
	}
	raw := EncodeWeightsRaw(weights)
	offset := payload.Len()
	payload.Write(raw)
	*blobs = append(*blobs, EntityWeightBlob{
		Path:   path,
		Offset: uint64(offset),
		Length: uint64(len(raw)),
		DType:  DTypeFloat32.String(),
		Native: false,
	})
}

func applyEntityMHANormBlob(l *VolumetricLayer, path string, raw []byte) error {
	weights, err := DecodeWeightsRaw(raw)
	if err != nil {
		return err
	}
	switch {
	case strings.HasSuffix(path, ".q_norm"):
		l.QNormWeight = weights
	case strings.HasSuffix(path, ".k_norm"):
		l.KNormWeight = weights
	default:
		return fmt.Errorf("unknown MHA norm blob %q", path)
	}
	return nil
}

func applyEntityQ4_0Blob(l *VolumetricLayer, path string, raw []byte, blob EntityWeightBlob) error {
	ensureLayerWeightStore(l)
	parts := strings.Split(blob.Path, ".")
	if len(parts) < 4 || parts[len(parts)-2] != "q4_0" {
		return fmt.Errorf("invalid q4_0 path %q", blob.Path)
	}
	keyInt, err := strconv.Atoi(parts[len(parts)-1])
	if err != nil {
		return fmt.Errorf("invalid q4_0 key in %q", blob.Path)
	}
	key := DType(keyInt)
	scales, packed, err := decodeEntityQ4_0Blob(raw)
	if err != nil {
		return err
	}
	l.WeightStore.SetQ4_0Component(key, scales, packed)
	l.DType = DTypeInt4
	return nil
}

func applyEntityBiasBlob(l *VolumetricLayer, raw []byte) error {
	ensureLayerWeightStore(l)
	bias, err := DecodeWeightsRaw(raw)
	if err != nil {
		return err
	}
	// Store biases only — do not allocate a zero-filled 3*wSize FP32 shell.
	// Packed Q4 CPU/GPU paths read Master as [gateB|upB|downB]; MaterializeQ4_0ForCPU
	// expands into the legacy padded Master layout when needed.
	l.WeightStore.Master = append([]float32(nil), bias...)
	return nil
}

func entityBlobIsQ4_0(b EntityWeightBlob) bool {
	return b.DType == entityBlobDTypeQ4_0 || strings.Contains(b.Path, ".q4_0.")
}

func entityBlobLayerPath(blobPath string) string {
	if i := strings.Index(blobPath, ".q4_0."); i >= 0 {
		return blobPath[:i]
	}
	if i := strings.Index(blobPath, ".bitnet_ternary."); i >= 0 {
		return blobPath[:i]
	}
	if strings.HasSuffix(blobPath, ".biases") {
		return strings.TrimSuffix(blobPath, ".biases")
	}
	if strings.HasSuffix(blobPath, ".inner_norm") {
		return strings.TrimSuffix(blobPath, ".inner_norm")
	}
	if strings.HasSuffix(blobPath, ".q_norm") {
		return strings.TrimSuffix(blobPath, ".q_norm")
	}
	if strings.HasSuffix(blobPath, ".k_norm") {
		return strings.TrimSuffix(blobPath, ".k_norm")
	}
	return blobPath
}

// DequantizeQ4_0GPUPacked expands baked .entity / GPU-upload Q4_0 blocks to FP32 weights.
func DequantizeQ4_0GPUPacked(scales []float32, packed []uint32) []float32 {
	if len(scales) == 0 || len(packed) == 0 {
		return nil
	}
	numBlocks := len(scales)
	blocks := make([]Q4_0Block, numBlocks)
	for i := 0; i < numBlocks; i++ {
		b := &blocks[i]
		b.Scale = scales[i]
		base := i * 4
		if base+3 >= len(packed) {
			break
		}
		for j := 0; j < 4; j++ {
			w := packed[base+j]
			b.Weights[j*4] = byte(w)
			b.Weights[j*4+1] = byte(w >> 8)
			b.Weights[j*4+2] = byte(w >> 16)
			b.Weights[j*4+3] = byte(w >> 24)
		}
	}
	return DequantizeQ4_0(blocks, numBlocks*32)
}

func (ws *WeightStore) dequantizeQ4_0Component(key DType) []float32 {
	if ws == nil || !ws.HasQ4_0Component(key) {
		return nil
	}
	return DequantizeQ4_0GPUPacked(ws.Q4_0Scales[key], ws.Q4_0Packed[key])
}

// MaterializeQ4_0ForCPU expands baked Q4_0 components into Master for CPU forward.
// GPU inference uses uploadQ4_0Cached; CPU tiled forward reads FP32 via GetActive/Master.
// Skipped when Network.UsePackedQ4CPU — see q4_cpu.go.
func (l *VolumetricLayer) MaterializeQ4_0ForCPU() {
	ws := l.WeightStore
	if ws == nil || !ws.HasAnyQ4_0() {
		return
	}
	if l.Network != nil && l.Network.UsePackedQ4CPU {
		return
	}
	switch l.Type {
	case LayerMultiHeadAttention:
		d := l.DModel
		q := l.QueryDim
		if q == 0 {
			q = d
		}
		kv := l.NumKVHeads * l.HeadDim
		qwSize := q * d
		kwSize := d * kv
		vwSize := d * kv
		owSize := d * q
		biasSize := q + kv + kv + d
		total := qwSize + kwSize + vwSize + owSize + biasSize
		master := make([]float32, total)
		if qW := ws.dequantizeQ4_0Component(WeightMHAQuery); len(qW) >= qwSize {
			copy(master[0:qwSize], qW[:qwSize])
		}
		off := qwSize
		if kW := ws.dequantizeQ4_0Component(WeightMHAKey); len(kW) >= kwSize {
			copy(master[off:off+kwSize], kW[:kwSize])
		}
		off += kwSize
		if vW := ws.dequantizeQ4_0Component(WeightMHAValue); len(vW) >= vwSize {
			copy(master[off:off+vwSize], vW[:vwSize])
		}
		off += vwSize
		if oW := ws.dequantizeQ4_0Component(WeightMHAProjection); len(oW) >= owSize {
			copy(master[off:off+owSize], oW[:owSize])
		}
		ws.Master = master
	case LayerSwiGLU:
		h, inter := l.InputHeight, l.OutputHeight
		wSize := h * inter
		biasTail := 2*inter + h
		total := 3*wSize + biasTail
		master := make([]float32, total)
		if gate := ws.dequantizeQ4_0Component(DType(100)); len(gate) >= wSize {
			copy(master[0:wSize], gate[:wSize])
		}
		if up := ws.dequantizeQ4_0Component(DType(101)); len(up) >= wSize {
			copy(master[wSize:2*wSize], up[:wSize])
		}
		if down := ws.dequantizeQ4_0Component(DType(102)); len(down) >= wSize {
			copy(master[2*wSize:3*wSize], down[:wSize])
		}
		switch {
		case len(ws.Master) == biasTail:
			copy(master[3*wSize:], ws.Master)
		case len(ws.Master) >= total:
			copy(master[3*wSize:], ws.Master[3*wSize:total])
		}
		ws.Master = master
	default:
		return
	}
	l.DType = DTypeFloat32
	if ws.Versions != nil {
		delete(ws.Versions, DTypeInt4)
	}
}

func (ws *WeightStore) uploadQ4_0Cached(ctx *WGPUContext, weightDType DType, label string) bool {
	if ws == nil || !ws.HasQ4_0Component(weightDType) {
		return false
	}
	scales := ws.Q4_0Scales[weightDType]
	packed := ws.Q4_0Packed[weightDType]
	sBuf, err := ctx.CreatePersistentBuffer(scales, label+" Scales")
	if err != nil {
		return false
	}
	pBuf, err := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    label + " Packed",
		Contents: wgpu.ToBytes(packed),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return false
	}
	ws.GPUScales[weightDType] = sBuf
	ws.GPUWeights[weightDType] = pBuf
	return true
}
