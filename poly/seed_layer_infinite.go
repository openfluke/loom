package poly

import (
	"bytes"
	"compress/flate"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

const infiniteDenseLayerFormat = "loom-infinite-dense-layer-v1"

// DefaultDenseWeightChunk is the tile size for sparse weight overrides [out, in].
var DefaultDenseWeightChunk = []int{8, 8}

// InfiniteDenseLayerManifest is one dense layer as root seed + optional sparse weight diffs.
type InfiniteDenseLayerManifest struct {
	Format    string                    `json:"format"`
	In        int                       `json:"in"`
	Out       int                       `json:"out"`
	DType     string                    `json:"dtype"`
	LayerSeed uint64                    `json:"layer_seed"`
	WeightFP  uint64                    `json:"weight_fp"`
	ChunkSize []int                     `json:"chunk_size,omitempty"`
	Overrides []DenseWeightChunkOverride `json:"overrides,omitempty"`
}

// DenseWeightChunkOverride is one weight tile that differs from He-init(layer_seed).
type DenseWeightChunkOverride struct {
	At      []int  `json:"at"`
	Shape   []int  `json:"shape"`
	Payload []byte `json:"payload"`
}

// BuildDenseLayerFromSeed materializes one dense layer from a layer seed.
func BuildDenseLayerFromSeed(layerSeed uint64, in, out int, dtype DType) (*VolumetricLayer, error) {
	if in <= 0 || out <= 0 {
		return nil, fmt.Errorf("dense layer: invalid shape %dx%d", in, out)
	}
	if dtype == 0 {
		dtype = DTypeFloat32
	}
	l := &VolumetricLayer{
		Type:         LayerDense,
		Activation:   ActivationReLU,
		InputHeight:  in,
		OutputHeight: out,
		DType:        dtype,
		WeightStore:  NewWeightStore(in * out),
	}
	InitWeightStoreHeSeeded(l.WeightStore, in, layerSeed)
	if dtype != DTypeFloat32 {
		l.WeightStore.Morph(dtype)
	}
	return l, nil
}

// EncodeInfiniteDenseLayer packs weights into the smallest manifest (seed-only or seed+sparse).
func EncodeInfiniteDenseLayer(ws *WeightStore, in, out int, dtype DType, layerSeed uint64) (*InfiniteDenseLayerManifest, error) {
	if ws == nil {
		return nil, fmt.Errorf("dense layer: nil weight store")
	}
	if dtype == 0 {
		dtype = DTypeFloat32
	}
	want, err := BuildDenseLayerFromSeed(layerSeed, in, out, dtype)
	if err != nil {
		return nil, err
	}
	m := &InfiniteDenseLayerManifest{
		Format:    infiniteDenseLayerFormat,
		In:        in,
		Out:       out,
		DType:     dtype.String(),
		LayerSeed: layerSeed,
		WeightFP:  weightStoreFingerprint(ws),
		ChunkSize: append([]int(nil), DefaultDenseWeightChunk...),
	}
	if weightStoreFingerprint(ws) == weightStoreFingerprint(want.WeightStore) {
		return m, nil
	}
	shape := []int{out, in}
	cs := append([]int(nil), DefaultDenseWeightChunk...)
	overrides := make(map[string]DenseWeightChunkOverride)
	err = foreachWeightChunk(shape, cs, func(chunkCoord, localOrigin, localShape []int) error {
		tile, err := extractWeightTile(ws.Master, in, out, localOrigin, localShape)
		if err != nil {
			return err
		}
		wantTile, err := extractWeightTile(want.WeightStore.Master, in, out, localOrigin, localShape)
		if err != nil {
			return err
		}
		if weightTileEqual(tile, wantTile) {
			delete(overrides, weightChunkKey(chunkCoord))
			return nil
		}
		raw := packFloat32s(tile)
		payload, err := compressSeedBytes(raw)
		if err != nil {
			return err
		}
		overrides[weightChunkKey(chunkCoord)] = DenseWeightChunkOverride{
			At: chunkCoord, Shape: append([]int(nil), localShape...), Payload: payload,
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	for _, o := range overrides {
		m.Overrides = append(m.Overrides, o)
	}
	return m, nil
}

// DecodeInfiniteDenseLayer rebuilds a WeightStore from a manifest.
func DecodeInfiniteDenseLayer(m *InfiniteDenseLayerManifest) (*WeightStore, error) {
	if m == nil {
		return nil, fmt.Errorf("dense layer: nil manifest")
	}
	if m.Format != "" && m.Format != infiniteDenseLayerFormat {
		return nil, fmt.Errorf("dense layer: unknown format %q", m.Format)
	}
	dtype := ParseDType(m.DType)
	if dtype == 0 {
		dtype = DTypeFloat32
	}
	layer, err := BuildDenseLayerFromSeed(m.LayerSeed, m.In, m.Out, dtype)
	if err != nil {
		return nil, err
	}
	ws := layer.WeightStore
	cs := m.ChunkSize
	if len(cs) == 0 {
		cs = DefaultDenseWeightChunk
	}
	for _, o := range m.Overrides {
		raw, err := decompressSeedBytes(o.Payload)
		if err != nil {
			return nil, err
		}
		tile, err := unpackFloat32s(raw, numelInt(o.Shape))
		if err != nil {
			return nil, err
		}
		localOrigin := make([]int, 2)
		for i := range o.At {
			localOrigin[i] = o.At[i] * cs[i]
		}
		if err := pasteWeightTile(ws.Master, m.In, m.Out, localOrigin, o.Shape, tile); err != nil {
			return nil, err
		}
	}
	if m.WeightFP != 0 && weightStoreFingerprint(ws) != m.WeightFP {
		return nil, fmt.Errorf("dense layer: weight fp mismatch got 0x%x want 0x%x", weightStoreFingerprint(ws), m.WeightFP)
	}
	if dtype != DTypeFloat32 {
		ws.Morph(dtype)
	}
	return ws, nil
}

// BuildDenseLayerFromInfiniteManifest is seed → layer (with sparse overrides applied).
func BuildDenseLayerFromInfiniteManifest(m *InfiniteDenseLayerManifest) (*VolumetricLayer, error) {
	if m == nil {
		return nil, fmt.Errorf("dense layer: nil manifest")
	}
	ws, err := DecodeInfiniteDenseLayer(m)
	if err != nil {
		return nil, err
	}
	dtype := ParseDType(m.DType)
	if dtype == 0 {
		dtype = DTypeFloat32
	}
	return &VolumetricLayer{
		Type:         LayerDense,
		Activation:   ActivationReLU,
		InputHeight:  m.In,
		OutputHeight: m.Out,
		DType:        dtype,
		WeightStore:  ws,
	}, nil
}

// ManifestFromDenseLayer extracts seeds from a layer built via He-init(layer_seed).
func ManifestFromDenseLayer(l *VolumetricLayer, layerSeed uint64) (*InfiniteDenseLayerManifest, error) {
	if l == nil || l.WeightStore == nil {
		return nil, fmt.Errorf("dense layer: nil layer")
	}
	if l.Type != LayerDense {
		return nil, fmt.Errorf("dense layer: type %v", l.Type)
	}
	dtype := l.DType
	if dtype == 0 {
		dtype = DTypeFloat32
	}
	m, err := EncodeInfiniteDenseLayer(l.WeightStore, l.InputHeight, l.OutputHeight, dtype, layerSeed)
	if err != nil {
		return nil, err
	}
	ok, err := denseLayerMatchesSeed(l, layerSeed, dtype.String())
	if err != nil {
		return nil, err
	}
	if !ok && len(m.Overrides) == 0 {
		return nil, fmt.Errorf("dense layer: weights do not match seed 0x%x", layerSeed)
	}
	return m, nil
}

// MarshalInfiniteDenseLayer JSON-encodes a layer manifest.
func MarshalInfiniteDenseLayer(m *InfiniteDenseLayerManifest) ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

// ParseInfiniteDenseLayer decodes JSON.
func ParseInfiniteDenseLayer(data []byte) (*InfiniteDenseLayerManifest, error) {
	var m InfiniteDenseLayerManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// OverrideCount returns sparse chunk count (0 = pure procedural layer seed).
func (m *InfiniteDenseLayerManifest) OverrideCount() int {
	if m == nil {
		return 0
	}
	return len(m.Overrides)
}

// WeightStoreFingerprint returns FNV-1a hash of master float32 weights.
func WeightStoreFingerprint(ws *WeightStore) uint64 {
	return weightStoreFingerprint(ws)
}

func packFloat32s(data []float32) []byte {
	buf := make([]byte, 4*len(data))
	for i, v := range data {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

func unpackFloat32s(raw []byte, n int) ([]float32, error) {
	want := 4 * n
	if len(raw) != want {
		return nil, fmt.Errorf("dense layer: payload %d bytes want %d", len(raw), want)
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out, nil
}

func compressSeedBytes(raw []byte) ([]byte, error) {
	var buf bytes.Buffer
	w, err := flate.NewWriter(&buf, flate.BestCompression)
	if err != nil {
		return nil, err
	}
	if _, err := w.Write(raw); err != nil {
		_ = w.Close()
		return nil, err
	}
	if err := w.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func decompressSeedBytes(data []byte) ([]byte, error) {
	r := flate.NewReader(bytes.NewReader(data))
	defer r.Close()
	return io.ReadAll(r)
}

func numelInt(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

func weightChunkKey(coords []int) string {
	parts := make([]string, len(coords))
	for i, c := range coords {
		parts[i] = strconv.Itoa(c)
	}
	return strings.Join(parts, ",")
}

func foreachWeightChunk(shape, chunkSize []int, fn func(chunkCoord, localOrigin, localShape []int) error) error {
	rank := len(shape)
	chunksPerDim := make([]int, rank)
	for i := range shape {
		cs := chunkSize[i]
		if cs <= 0 {
			cs = 8
		}
		chunksPerDim[i] = (shape[i] + cs - 1) / cs
	}
	chunkCoord := make([]int, rank)
	var walk func(dim int) error
	walk = func(dim int) error {
		if dim == rank {
			localOrigin := make([]int, rank)
			localShape := make([]int, rank)
			for i := range rank {
				cs := chunkSize[i]
				if cs <= 0 {
					cs = 8
				}
				localOrigin[i] = chunkCoord[i] * cs
				remain := shape[i] - localOrigin[i]
				if remain < cs {
					localShape[i] = remain
				} else {
					localShape[i] = cs
				}
			}
			return fn(append([]int(nil), chunkCoord...), localOrigin, localShape)
		}
		for c := 0; c < chunksPerDim[dim]; c++ {
			chunkCoord[dim] = c
			if err := walk(dim + 1); err != nil {
				return err
			}
		}
		return nil
	}
	return walk(0)
}

func extractWeightTile(master []float32, in, out int, localOrigin, localShape []int) ([]float32, error) {
	if len(localShape) != 2 {
		return nil, fmt.Errorf("dense layer: weight tile rank %d", len(localShape))
	}
	tile := make([]float32, localShape[0]*localShape[1])
	for o := 0; o < localShape[0]; o++ {
		for i := 0; i < localShape[1]; i++ {
			ro := localOrigin[0] + o
			col := localOrigin[1] + i
			tile[o*localShape[1]+i] = master[ro*in+col]
		}
	}
	return tile, nil
}

func pasteWeightTile(master []float32, in, out int, localOrigin []int, localShape []int, tile []float32) error {
	if len(localShape) != 2 {
		return fmt.Errorf("dense layer: weight tile rank %d", len(localShape))
	}
	if len(tile) != localShape[0]*localShape[1] {
		return fmt.Errorf("dense layer: tile size mismatch")
	}
	for o := 0; o < localShape[0]; o++ {
		for i := 0; i < localShape[1]; i++ {
			ro := localOrigin[0] + o
			col := localOrigin[1] + i
			if ro >= out || col >= in {
				continue
			}
			master[ro*in+col] = tile[o*localShape[1]+i]
		}
	}
	return nil
}

func weightTileEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}
