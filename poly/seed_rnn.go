package poly

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"strconv"
)

const rnnManifestFormat = "loom-rnn-manifest-v1"

// RNNLayerManifest is one RNN layer seed + dtype (no weight bytes).
type RNNLayerManifest struct {
	Index     int    `json:"index"`
	Path      string `json:"path"`
	In        int    `json:"in"`
	Out       int    `json:"out"`
	LayerSeed uint64 `json:"layer_seed"`
	DType     string `json:"dtype"`
	WeightFP  uint64 `json:"weight_fp"`
}

// RNNWeightManifest is topology seed + per-layer weight seeds.
type RNNWeightManifest struct {
	Format       string             `json:"format"`
	TopologySeed uint64             `json:"topology_seed"`
	Sizes        []int              `json:"sizes"`
	Layers       []RNNLayerManifest `json:"layers"`
	NetworkFP    uint64             `json:"network_fp"`
	ForwardFP    uint64             `json:"forward_fp"`
}

// RNNTopologySeed hashes layer widths into a topology-only seed.
func RNNTopologySeed(name string, sizes []int) uint64 {
	parts := []any{"loom-rnn-v1", name}
	for _, s := range sizes {
		parts = append(parts, s)
	}
	return SeedFrom(parts...)
}

// RNNLayerWeightSeed derives per-layer weight seed from topology seed.
func RNNLayerWeightSeed(topologySeed uint64, layerIndex int) uint64 {
	return DeriveLayerSeed(topologySeed, layerIndex, rnnLayerPath(layerIndex))
}

func rnnLayerPath(index int) string {
	return "rnn." + strconv.Itoa(index)
}

// BuildRNNManifest creates per-layer seeds and fingerprints.
func BuildRNNManifest(topologySeed uint64, sizes []int, dtypes []string) (*RNNWeightManifest, error) {
	if len(sizes) < 2 {
		return nil, fmt.Errorf("rnn: need at least input and output sizes")
	}
	m := &RNNWeightManifest{
		Format:       rnnManifestFormat,
		TopologySeed: topologySeed,
		Sizes:        append([]int(nil), sizes...),
	}
	for i := 0; i < len(sizes)-1; i++ {
		dt := "float32"
		if i < len(dtypes) && dtypes[i] != "" {
			dt = dtypes[i]
		}
		in, out := sizes[i], sizes[i+1]
		layerSeed := RNNLayerWeightSeed(topologySeed, i)
		fp, err := rnnInitWeights(in, out, layerSeed, dt)
		if err != nil {
			return nil, err
		}
		m.Layers = append(m.Layers, RNNLayerManifest{
			Index: i, Path: rnnLayerPath(i), In: in, Out: out,
			LayerSeed: layerSeed, DType: dt, WeightFP: fp,
		})
	}
	m.NetworkFP = rnnManifestNetworkFP(m.Layers)
	fp, err := rnnManifestForwardFP(m)
	if err != nil {
		return nil, err
	}
	m.ForwardFP = fp
	return m, nil
}

// RebuildRNNManifest verifies seeds-only rebuild matches fingerprints.
func RebuildRNNManifest(m *RNNWeightManifest) (*RNNWeightManifest, error) {
	if m == nil {
		return nil, fmt.Errorf("rnn: nil manifest")
	}
	rebuilt, err := BuildRNNManifest(m.TopologySeed, m.Sizes, rnnLayerDTypes(m))
	if err != nil {
		return nil, err
	}
	if rebuilt.NetworkFP != m.NetworkFP {
		return nil, fmt.Errorf("rnn: network fp mismatch got 0x%x want 0x%x", rebuilt.NetworkFP, m.NetworkFP)
	}
	if m.ForwardFP != 0 && rebuilt.ForwardFP != m.ForwardFP {
		return nil, fmt.Errorf("rnn: forward fp mismatch got 0x%x want 0x%x", rebuilt.ForwardFP, m.ForwardFP)
	}
	return rebuilt, nil
}

// BuildRNNVolumetricFromManifest builds a poly VolumetricNetwork from RNN manifest seeds.
func BuildRNNVolumetricFromManifest(m *RNNWeightManifest) (*VolumetricNetwork, error) {
	if m == nil || len(m.Layers) == 0 {
		return nil, fmt.Errorf("rnn: empty manifest")
	}
	net := NewVolumetricNetwork(1, 1, 1, len(m.Layers))
	net.InitSeed = m.TopologySeed
	for i, layer := range m.Layers {
		l := net.GetLayer(0, 0, 0, i)
		l.Type = LayerRNN
		l.Activation = ActivationTanh
		l.InputHeight = layer.In
		l.OutputHeight = layer.Out
		l.SeqLength = 1
		l.DType = ParseDType(layer.DType)
		if l.DType == 0 {
			l.DType = DTypeFloat32
		}
		wCount := rnnWeightCount(layer.In, layer.Out)
		l.WeightStore = NewWeightStore(wCount)
		InitWeightStoreHeSeeded(l.WeightStore, layer.In, layer.LayerSeed)
		if l.DType != DTypeFloat32 {
			l.WeightStore.Morph(l.DType)
		}
	}
	return net, nil
}

// ManifestFromRNNNetwork extracts layer seeds from a seeded RNN volumetric stack.
func ManifestFromRNNNetwork(net *VolumetricNetwork, topologySeed uint64, sizes []int, dtypes []string) (*RNNWeightManifest, error) {
	if net == nil {
		return nil, fmt.Errorf("rnn: nil network")
	}
	if len(sizes) < 2 {
		return nil, fmt.Errorf("rnn: need sizes")
	}
	m := &RNNWeightManifest{
		Format: rnnManifestFormat, TopologySeed: topologySeed,
		Sizes: append([]int(nil), sizes...),
	}
	for i := 0; i < len(sizes)-1; i++ {
		dt := "float32"
		if i < len(dtypes) && dtypes[i] != "" {
			dt = dtypes[i]
		}
		l := net.GetLayer(0, 0, 0, i)
		if l.Type != LayerRNN {
			return nil, fmt.Errorf("rnn: layer %d type %v", i, l.Type)
		}
		seed := RNNLayerWeightSeed(topologySeed, i)
		ok, err := rnnLayerMatchesSeed(l, seed, dt)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, fmt.Errorf("rnn: layer %d weights do not match seed 0x%x", i, seed)
		}
		fp := weightStoreFingerprint(l.WeightStore)
		m.Layers = append(m.Layers, RNNLayerManifest{
			Index: i, Path: rnnLayerPath(i), In: sizes[i], Out: sizes[i+1],
			LayerSeed: seed, DType: dt, WeightFP: fp,
		})
	}
	m.NetworkFP = rnnManifestNetworkFP(m.Layers)
	fp, err := rnnManifestForwardFP(m)
	if err != nil {
		return nil, err
	}
	m.ForwardFP = fp
	return m, nil
}

// MarshalRNNManifest JSON-encodes a manifest.
func MarshalRNNManifest(m *RNNWeightManifest) ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

// ParseRNNManifest decodes JSON manifest.
func ParseRNNManifest(data []byte) (*RNNWeightManifest, error) {
	var m RNNWeightManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

func rnnWeightCount(in, out int) int {
	return in*out + out*out + out
}

func rnnLayerDTypes(m *RNNWeightManifest) []string {
	out := make([]string, len(m.Layers))
	for i, l := range m.Layers {
		out[i] = l.DType
	}
	return out
}

func rnnManifestNetworkFP(layers []RNNLayerManifest) uint64 {
	h := fnv.New64a()
	var buf [8]byte
	for _, l := range layers {
		binary.LittleEndian.PutUint64(buf[:], l.WeightFP)
		_, _ = h.Write(buf[:])
	}
	return h.Sum64()
}

func rnnInitWeights(in, out int, seed uint64, dtype string) (uint64, error) {
	wCount := rnnWeightCount(in, out)
	ws := NewWeightStore(wCount)
	InitWeightStoreHeSeeded(ws, in, seed)
	dt := ParseDType(dtype)
	if dt == 0 {
		dt = DTypeFloat32
	}
	if dt != DTypeFloat32 {
		ws.Morph(dt)
	}
	return weightStoreFingerprint(ws), nil
}

func rnnLayerMatchesSeed(l *VolumetricLayer, seed uint64, dtype string) (bool, error) {
	if l == nil || l.WeightStore == nil {
		return false, nil
	}
	fp, err := rnnInitWeights(l.InputHeight, l.OutputHeight, seed, dtype)
	if err != nil {
		return false, err
	}
	return weightStoreFingerprint(l.WeightStore) == fp, nil
}

func rnnManifestForwardFP(m *RNNWeightManifest) (uint64, error) {
	net, err := BuildRNNVolumetricFromManifest(m)
	if err != nil {
		return 0, err
	}
	if len(m.Sizes) == 0 {
		return 0, fmt.Errorf("rnn: empty sizes")
	}
	in := NewTensorFromSlice(seedDemoForwardInput(m.Sizes[0]), 1, m.Sizes[0])
	out, _, _ := ForwardPolymorphic(net, in)
	if out == nil {
		return 0, fmt.Errorf("rnn: forward nil")
	}
	return seedOutputHash(out.Data), nil
}
