package poly

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"strconv"
	"strings"
)

const entityBlobDTypeBitNetTernary = "BITNET_TERNARY"

func (ws *WeightStore) hasBitNetCPUPacked() bool {
	if ws == nil || len(ws.CPUPacked) == 0 {
		return false
	}
	for _, v := range ws.CPUPacked {
		if m, ok := v.(*BitNetTernaryMatrix); ok && m != nil && len(m.Words) > 0 {
			return true
		}
	}
	return false
}

func (ws *WeightStore) HasAnyBitNetTernary() bool {
	return ws != nil && ws.hasBitNetCPUPacked()
}

func EncodeEntityBitNetTernaryBlob(m *BitNetTernaryMatrix) []byte {
	return encodeEntityBitNetTernaryBlob(m)
}

func DecodeEntityBitNetTernaryBlob(raw []byte) (*BitNetTernaryMatrix, error) {
	return decodeEntityBitNetTernaryBlob(raw)
}

func encodeEntityBitNetTernaryBlob(m *BitNetTernaryMatrix) []byte {
	if m == nil || len(m.Words) == 0 {
		return nil
	}
	raw := make([]byte, 24+len(m.Words)*4)
	binary.LittleEndian.PutUint32(raw[0:4], uint32(m.Rows))
	binary.LittleEndian.PutUint32(raw[4:8], uint32(m.Cols))
	binary.LittleEndian.PutUint32(raw[8:12], uint32(m.Offset))
	binary.LittleEndian.PutUint32(raw[12:16], math.Float32bits(m.Scale))
	binary.LittleEndian.PutUint32(raw[16:20], uint32(m.RowWords))
	binary.LittleEndian.PutUint32(raw[20:24], uint32(len(m.Words)))
	off := 24
	for _, w := range m.Words {
		binary.LittleEndian.PutUint32(raw[off:off+4], w)
		off += 4
	}
	return raw
}

func decodeEntityBitNetTernaryBlob(raw []byte) (*BitNetTernaryMatrix, error) {
	if len(raw) < 24 {
		return nil, fmt.Errorf("bitnet ternary blob too short")
	}
	rows := int(binary.LittleEndian.Uint32(raw[0:4]))
	cols := int(binary.LittleEndian.Uint32(raw[4:8]))
	offset := int(binary.LittleEndian.Uint32(raw[8:12]))
	scale := math.Float32frombits(binary.LittleEndian.Uint32(raw[12:16]))
	rowWords := int(binary.LittleEndian.Uint32(raw[16:20]))
	nWords := int(binary.LittleEndian.Uint32(raw[20:24]))
	need := 24 + nWords*4
	if len(raw) < need {
		return nil, fmt.Errorf("bitnet ternary blob truncated: have %d need %d", len(raw), need)
	}
	words := make([]uint32, nWords)
	off := 24
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(raw[off : off+4])
		off += 4
	}
	if rowWords <= 0 {
		rowWords = (cols + 15) / 16
	}
	return &BitNetTernaryMatrix{
		Rows:     rows,
		Cols:     cols,
		RowWords: rowWords,
		Offset:   offset,
		Scale:    scale,
		Words:    words,
	}, nil
}

func (ws *WeightStore) installBitNetTernaryMatrix(key DType, m *BitNetTernaryMatrix) {
	if ws == nil || m == nil || len(m.Words) == 0 {
		return
	}
	if ws.CPUPacked == nil {
		ws.CPUPacked = make(map[DType]any)
	}
	ws.CPUPacked[key] = m
	if m.Scale != 0 {
		ws.CPUPacked[bitNetPackedScaleKey(m.Offset)] = m.Scale
	}
}

func appendEntityBitNetTernaryBlob(path string, key DType, m *BitNetTernaryMatrix, payload *bytes.Buffer, blobs *[]EntityWeightBlob) {
	raw := encodeEntityBitNetTernaryBlob(m)
	if len(raw) == 0 {
		return
	}
	offset := payload.Len()
	payload.Write(raw)
	*blobs = append(*blobs, EntityWeightBlob{
		Path:   fmt.Sprintf("%s.bitnet_ternary.%d", path, int(key)),
		Offset: uint64(offset),
		Length: uint64(len(raw)),
		DType:  entityBlobDTypeBitNetTernary,
		Native: true,
	})
}

func collectEntityBitNetLayer(l *VolumetricLayer, path string, payload *bytes.Buffer, blobs *[]EntityWeightBlob) {
	if l.WeightStore == nil || !l.WeightStore.hasBitNetCPUPacked() {
		return
	}
	for key, v := range l.WeightStore.CPUPacked {
		m, ok := v.(*BitNetTernaryMatrix)
		if !ok || m == nil || len(m.Words) == 0 {
			continue
		}
		// Matrix keys are bitNetPackedKey(offset)=10000+offset (can exceed 20000 for large
		// HF layouts). Scale-only entries are float32 at bitNetPackedScaleKey — filtered above.
		appendEntityBitNetTernaryBlob(path, key, m, payload, blobs)
	}
	collectEntityBitNetAuxBlobs(l, path, payload, blobs)
}

func collectEntityBitNetAuxBlobs(l *VolumetricLayer, path string, payload *bytes.Buffer, blobs *[]EntityWeightBlob) {
	appendEntityFP32AuxBlob(path+".inner_norm", l.InnerNormWeight, payload, blobs)
	if l.Type == LayerMultiHeadAttention {
		collectEntityMHANormBlobs(l, path, payload, blobs)
	}
}

func applyEntityBitNetTernaryBlob(l *VolumetricLayer, blobPath string, raw []byte) error {
	if l.WeightStore == nil {
		return fmt.Errorf("no WeightStore for %q", blobPath)
	}
	parts := strings.Split(blobPath, ".")
	if len(parts) < 4 || parts[len(parts)-2] != "bitnet_ternary" {
		return fmt.Errorf("invalid bitnet ternary path %q", blobPath)
	}
	keyInt, err := strconv.Atoi(parts[len(parts)-1])
	if err != nil {
		return fmt.Errorf("invalid bitnet ternary key in %q", blobPath)
	}
	m, err := decodeEntityBitNetTernaryBlob(raw)
	if err != nil {
		return err
	}
	l.WeightStore.installBitNetTernaryMatrix(DType(keyInt), m)
	l.DType = DTypeTernary
	return nil
}

func applyEntityBitNetAuxBlob(l *VolumetricLayer, path string, raw []byte) error {
	weights, err := DecodeWeightsRaw(raw)
	if err != nil {
		return err
	}
	switch {
	case strings.HasSuffix(path, ".inner_norm"):
		l.InnerNormWeight = weights
	case strings.HasSuffix(path, ".q_norm"):
		l.QNormWeight = weights
	case strings.HasSuffix(path, ".k_norm"):
		l.KNormWeight = weights
	default:
		return fmt.Errorf("unknown BitNet aux blob %q", path)
	}
	return nil
}

func entityBlobIsBitNetTernary(b EntityWeightBlob) bool {
	return b.DType == entityBlobDTypeBitNetTernary || strings.Contains(b.Path, ".bitnet_ternary.")
}
