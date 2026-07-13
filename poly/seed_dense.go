package poly

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"strconv"
)

const denseManifestFormat = "loom-dense-manifest-v1"

// DenseLayerManifest is one dense layer seed + dtype (no weight bytes).
type DenseLayerManifest struct {
	Index     int    `json:"index"`
	Path      string `json:"path"`
	In        int    `json:"in"`
	Out       int    `json:"out"`
	LayerSeed uint64 `json:"layer_seed"`
	DType     string `json:"dtype"`
	WeightFP  uint64 `json:"weight_fp"`
}

// DenseWeightManifest is topology seed + per-layer weight seeds.
type DenseWeightManifest struct {
	Format       string               `json:"format"`
	TopologySeed uint64               `json:"topology_seed"`
	Sizes        []int                `json:"sizes"`
	Layers       []DenseLayerManifest `json:"layers"`
	NetworkFP    uint64               `json:"network_fp"`
	ForwardFP    uint64               `json:"forward_fp"`
}

// DenseTopologySeed hashes layer widths into a topology-only seed.
func DenseTopologySeed(name string, sizes []int) uint64 {
	parts := []any{"loom-dense-v1", name}
	for _, s := range sizes {
		parts = append(parts, s)
	}
	return SeedFrom(parts...)
}

// DenseLayerWeightSeed derives per-layer weight seed from topology seed.
func DenseLayerWeightSeed(topologySeed uint64, layerIndex int) uint64 {
	return DeriveLayerSeed(topologySeed, layerIndex, denseLayerPath(layerIndex))
}

func denseLayerPath(index int) string {
	return "dense." + strconv.Itoa(index)
}

// BuildDenseManifest creates per-layer seeds and fingerprints.
func BuildDenseManifest(topologySeed uint64, sizes []int, dtypes []string) (*DenseWeightManifest, error) {
	if len(sizes) < 2 {
		return nil, fmt.Errorf("dense: need at least input and output sizes")
	}
	m := &DenseWeightManifest{
		Format:       denseManifestFormat,
		TopologySeed: topologySeed,
		Sizes:        append([]int(nil), sizes...),
	}
	for i := 0; i < len(sizes)-1; i++ {
		dt := "float32"
		if i < len(dtypes) && dtypes[i] != "" {
			dt = dtypes[i]
		}
		in, out := sizes[i], sizes[i+1]
		layerSeed := DenseLayerWeightSeed(topologySeed, i)
		_, fp, err := denseInitWeights(in, out, layerSeed, dt)
		if err != nil {
			return nil, err
		}
		m.Layers = append(m.Layers, DenseLayerManifest{
			Index: i, Path: denseLayerPath(i), In: in, Out: out,
			LayerSeed: layerSeed, DType: dt, WeightFP: fp,
		})
	}
	m.NetworkFP = denseManifestNetworkFP(m.Layers)
	out, err := ForwardDenseManifest(m)
	if err != nil {
		return nil, err
	}
	m.ForwardFP = denseHashFloat64(out)
	return m, nil
}

// RebuildDenseManifest verifies seeds-only rebuild matches fingerprints.
func RebuildDenseManifest(m *DenseWeightManifest) (*DenseWeightManifest, error) {
	if m == nil {
		return nil, fmt.Errorf("dense: nil manifest")
	}
	rebuilt, err := BuildDenseManifest(m.TopologySeed, m.Sizes, denseLayerDTypes(m))
	if err != nil {
		return nil, err
	}
	if rebuilt.NetworkFP != m.NetworkFP {
		return nil, fmt.Errorf("dense: network fp mismatch got 0x%x want 0x%x", rebuilt.NetworkFP, m.NetworkFP)
	}
	if m.ForwardFP != 0 && rebuilt.ForwardFP != m.ForwardFP {
		return nil, fmt.Errorf("dense: forward fp mismatch got 0x%x want 0x%x", rebuilt.ForwardFP, m.ForwardFP)
	}
	return rebuilt, nil
}

// BuildDenseVolumetricFromManifest builds a poly VolumetricNetwork from dense manifest seeds.
func BuildDenseVolumetricFromManifest(m *DenseWeightManifest) (*VolumetricNetwork, error) {
	if m == nil || len(m.Layers) == 0 {
		return nil, fmt.Errorf("dense: empty manifest")
	}
	net := NewVolumetricNetwork(1, 1, 1, len(m.Layers))
	net.InitSeed = m.TopologySeed
	for i, layer := range m.Layers {
		l := net.GetLayer(0, 0, 0, i)
		l.Type = LayerDense
		l.Activation = ActivationReLU
		l.InputHeight = layer.In
		l.OutputHeight = layer.Out
		l.DType = ParseDType(layer.DType)
		if l.DType == 0 {
			l.DType = DTypeFloat32
		}
		l.WeightStore = NewWeightStore(layer.In * layer.Out)
		InitWeightStoreHeSeeded(l.WeightStore, layer.In, layer.LayerSeed)
		if l.DType != DTypeFloat32 {
			l.WeightStore.Morph(l.DType)
		}
	}
	return net, nil
}

// ManifestFromDenseNetwork extracts layer seeds from a seeded dense volumetric stack.
func ManifestFromDenseNetwork(net *VolumetricNetwork, topologySeed uint64, sizes []int, dtypes []string) (*DenseWeightManifest, error) {
	if net == nil {
		return nil, fmt.Errorf("dense: nil network")
	}
	if len(sizes) < 2 {
		return nil, fmt.Errorf("dense: need sizes")
	}
	m := &DenseWeightManifest{
		Format: denseManifestFormat, TopologySeed: topologySeed,
		Sizes: append([]int(nil), sizes...),
	}
	for i := 0; i < len(sizes)-1; i++ {
		dt := "float32"
		if i < len(dtypes) && dtypes[i] != "" {
			dt = dtypes[i]
		}
		l := net.GetLayer(0, 0, 0, i)
		if l.Type != LayerDense {
			return nil, fmt.Errorf("dense: layer %d type %v", i, l.Type)
		}
		seed := DenseLayerWeightSeed(topologySeed, i)
		ok, err := denseLayerMatchesSeed(l, seed, dt)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, fmt.Errorf("dense: layer %d weights do not match seed 0x%x", i, seed)
		}
		fp := weightStoreFingerprint(l.WeightStore)
		m.Layers = append(m.Layers, DenseLayerManifest{
			Index: i, Path: denseLayerPath(i), In: sizes[i], Out: sizes[i+1],
			LayerSeed: seed, DType: dt, WeightFP: fp,
		})
	}
	m.NetworkFP = denseManifestNetworkFP(m.Layers)
	in := NewTensorFromSlice(seedDemoForwardInput(sizes[0]), 1, sizes[0])
	out, _, _ := ForwardPolymorphic(net, in)
	if out != nil {
		m.ForwardFP = seedOutputHash(out.Data)
	}
	return m, nil
}

// ForwardDenseManifest forward-passes using manifest seeds (rebuilds in RAM).
func ForwardDenseManifest(m *DenseWeightManifest) ([]float64, error) {
	net, err := BuildDenseVolumetricFromManifest(m)
	if err != nil {
		return nil, err
	}
	if len(m.Sizes) == 0 {
		return nil, fmt.Errorf("dense: empty sizes")
	}
	in := NewTensorFromSlice(seedDemoForwardInput(m.Sizes[0]), 1, m.Sizes[0])
	out, _, _ := ForwardPolymorphic(net, in)
	if out == nil {
		return nil, fmt.Errorf("dense: forward nil")
	}
	fd := make([]float64, len(out.Data))
	for i, v := range out.Data {
		fd[i] = float64(v)
	}
	return fd, nil
}

// MarshalDenseManifest JSON-encodes a manifest.
func MarshalDenseManifest(m *DenseWeightManifest) ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

// ParseDenseManifest decodes JSON manifest.
func ParseDenseManifest(data []byte) (*DenseWeightManifest, error) {
	var m DenseWeightManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

func denseLayerDTypes(m *DenseWeightManifest) []string {
	out := make([]string, len(m.Layers))
	for i, l := range m.Layers {
		out[i] = l.DType
	}
	return out
}

func denseManifestNetworkFP(layers []DenseLayerManifest) uint64 {
	h := fnv.New64a()
	var buf [8]byte
	for _, l := range layers {
		binary.LittleEndian.PutUint64(buf[:], l.WeightFP)
		_, _ = h.Write(buf[:])
	}
	return h.Sum64()
}

func denseInitWeights(in, out int, seed uint64, dtype string) (any, uint64, error) {
	master := make([]float32, in*out)
	InitFloat32HeSeeded(master, in, seed)
	dt := ParseDType(dtype)
	if dt == 0 {
		dt = DTypeFloat32
	}
	if dt == DTypeFloat32 {
		return master, weightStoreFingerprint(&WeightStore{Master: master}), nil
	}
	ws := NewWeightStore(in * out)
	copy(ws.Master, master)
	ws.Morph(dt)
	fp := weightStoreFingerprint(ws)
	return ws.Master, fp, nil
}

func denseLayerMatchesSeed(l *VolumetricLayer, seed uint64, dtype string) (bool, error) {
	if l == nil || l.WeightStore == nil {
		return false, nil
	}
	_, fp, err := denseInitWeights(l.InputHeight, l.OutputHeight, seed, dtype)
	if err != nil {
		return false, err
	}
	return weightStoreFingerprint(l.WeightStore) == fp, nil
}

func denseHashFloat64(data []float64) uint64 {
	h := fnv.New64a()
	var buf [8]byte
	for _, v := range data {
		binary.LittleEndian.PutUint64(buf[:], math.Float64bits(v))
		_, _ = h.Write(buf[:])
	}
	return h.Sum64()
}
