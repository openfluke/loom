package poly

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"strconv"
)

const lstmManifestFormat = "loom-lstm-manifest-v1"

// LSTMLayerManifest is one LSTM layer seed + dtype (no weight bytes).
type LSTMLayerManifest struct {
	Index     int    `json:"index"`
	Path      string `json:"path"`
	In        int    `json:"in"`
	Out       int    `json:"out"`
	LayerSeed uint64 `json:"layer_seed"`
	DType     string `json:"dtype"`
	WeightFP  uint64 `json:"weight_fp"`
}

// LSTMWeightManifest is topology seed + per-layer weight seeds.
type LSTMWeightManifest struct {
	Format       string              `json:"format"`
	TopologySeed uint64              `json:"topology_seed"`
	Sizes        []int               `json:"sizes"`
	Layers       []LSTMLayerManifest `json:"layers"`
	NetworkFP    uint64              `json:"network_fp"`
	ForwardFP    uint64              `json:"forward_fp"`
}

// LSTMTopologySeed hashes layer widths into a topology-only seed.
func LSTMTopologySeed(name string, sizes []int) uint64 {
	parts := []any{"loom-lstm-v1", name}
	for _, s := range sizes {
		parts = append(parts, s)
	}
	return SeedFrom(parts...)
}

// LSTMLayerWeightSeed derives per-layer weight seed from topology seed.
func LSTMLayerWeightSeed(topologySeed uint64, layerIndex int) uint64 {
	return DeriveLayerSeed(topologySeed, layerIndex, lstmLayerPath(layerIndex))
}

func lstmLayerPath(index int) string {
	return "lstm." + strconv.Itoa(index)
}

// BuildLSTMManifest creates per-layer seeds and fingerprints.
func BuildLSTMManifest(topologySeed uint64, sizes []int, dtypes []string) (*LSTMWeightManifest, error) {
	if len(sizes) < 2 {
		return nil, fmt.Errorf("lstm: need at least input and output sizes")
	}
	m := &LSTMWeightManifest{
		Format:       lstmManifestFormat,
		TopologySeed: topologySeed,
		Sizes:        append([]int(nil), sizes...),
	}
	for i := 0; i < len(sizes)-1; i++ {
		dt := "float32"
		if i < len(dtypes) && dtypes[i] != "" {
			dt = dtypes[i]
		}
		in, out := sizes[i], sizes[i+1]
		layerSeed := LSTMLayerWeightSeed(topologySeed, i)
		fp, err := lstmInitWeights(in, out, layerSeed, dt)
		if err != nil {
			return nil, err
		}
		m.Layers = append(m.Layers, LSTMLayerManifest{
			Index: i, Path: lstmLayerPath(i), In: in, Out: out,
			LayerSeed: layerSeed, DType: dt, WeightFP: fp,
		})
	}
	m.NetworkFP = lstmManifestNetworkFP(m.Layers)
	fp, err := lstmManifestForwardFP(m)
	if err != nil {
		return nil, err
	}
	m.ForwardFP = fp
	return m, nil
}

// RebuildLSTMManifest verifies seeds-only rebuild matches fingerprints.
func RebuildLSTMManifest(m *LSTMWeightManifest) (*LSTMWeightManifest, error) {
	if m == nil {
		return nil, fmt.Errorf("lstm: nil manifest")
	}
	rebuilt, err := BuildLSTMManifest(m.TopologySeed, m.Sizes, lstmLayerDTypes(m))
	if err != nil {
		return nil, err
	}
	if rebuilt.NetworkFP != m.NetworkFP {
		return nil, fmt.Errorf("lstm: network fp mismatch got 0x%x want 0x%x", rebuilt.NetworkFP, m.NetworkFP)
	}
	if m.ForwardFP != 0 && rebuilt.ForwardFP != m.ForwardFP {
		return nil, fmt.Errorf("lstm: forward fp mismatch got 0x%x want 0x%x", rebuilt.ForwardFP, m.ForwardFP)
	}
	return rebuilt, nil
}

// BuildLSTMVolumetricFromManifest builds a poly VolumetricNetwork from LSTM manifest seeds.
func BuildLSTMVolumetricFromManifest(m *LSTMWeightManifest) (*VolumetricNetwork, error) {
	if m == nil || len(m.Layers) == 0 {
		return nil, fmt.Errorf("lstm: empty manifest")
	}
	net := NewVolumetricNetwork(1, 1, 1, len(m.Layers))
	net.InitSeed = m.TopologySeed
	for i, layer := range m.Layers {
		l := net.GetLayer(0, 0, 0, i)
		l.Type = LayerLSTM
		l.Activation = ActivationTanh
		l.InputHeight = layer.In
		l.OutputHeight = layer.Out
		l.SeqLength = 1
		l.DType = ParseDType(layer.DType)
		if l.DType == 0 {
			l.DType = DTypeFloat32
		}
		wCount := lstmWeightCount(layer.In, layer.Out)
		l.WeightStore = NewWeightStore(wCount)
		InitWeightStoreHeSeeded(l.WeightStore, layer.In, layer.LayerSeed)
		if l.DType != DTypeFloat32 {
			l.WeightStore.Morph(l.DType)
		}
	}
	return net, nil
}

// ManifestFromLSTMNetwork extracts layer seeds from a seeded LSTM volumetric stack.
func ManifestFromLSTMNetwork(net *VolumetricNetwork, topologySeed uint64, sizes []int, dtypes []string) (*LSTMWeightManifest, error) {
	if net == nil {
		return nil, fmt.Errorf("lstm: nil network")
	}
	if len(sizes) < 2 {
		return nil, fmt.Errorf("lstm: need sizes")
	}
	m := &LSTMWeightManifest{
		Format: lstmManifestFormat, TopologySeed: topologySeed,
		Sizes: append([]int(nil), sizes...),
	}
	for i := 0; i < len(sizes)-1; i++ {
		dt := "float32"
		if i < len(dtypes) && dtypes[i] != "" {
			dt = dtypes[i]
		}
		l := net.GetLayer(0, 0, 0, i)
		if l.Type != LayerLSTM {
			return nil, fmt.Errorf("lstm: layer %d type %v", i, l.Type)
		}
		seed := LSTMLayerWeightSeed(topologySeed, i)
		ok, err := lstmLayerMatchesSeed(l, seed, dt)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, fmt.Errorf("lstm: layer %d weights do not match seed 0x%x", i, seed)
		}
		fp := weightStoreFingerprint(l.WeightStore)
		m.Layers = append(m.Layers, LSTMLayerManifest{
			Index: i, Path: lstmLayerPath(i), In: sizes[i], Out: sizes[i+1],
			LayerSeed: seed, DType: dt, WeightFP: fp,
		})
	}
	m.NetworkFP = lstmManifestNetworkFP(m.Layers)
	fp, err := lstmManifestForwardFP(m)
	if err != nil {
		return nil, err
	}
	m.ForwardFP = fp
	return m, nil
}

// MarshalLSTMManifest JSON-encodes a manifest.
func MarshalLSTMManifest(m *LSTMWeightManifest) ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

// ParseLSTMManifest decodes JSON manifest.
func ParseLSTMManifest(data []byte) (*LSTMWeightManifest, error) {
	var m LSTMWeightManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

func lstmWeightCount(in, out int) int {
	gate := in*out + out*out + out
	return 4 * gate
}

func lstmLayerDTypes(m *LSTMWeightManifest) []string {
	out := make([]string, len(m.Layers))
	for i, l := range m.Layers {
		out[i] = l.DType
	}
	return out
}

func lstmManifestNetworkFP(layers []LSTMLayerManifest) uint64 {
	h := fnv.New64a()
	var buf [8]byte
	for _, l := range layers {
		binary.LittleEndian.PutUint64(buf[:], l.WeightFP)
		_, _ = h.Write(buf[:])
	}
	return h.Sum64()
}

func lstmInitWeights(in, out int, seed uint64, dtype string) (uint64, error) {
	wCount := lstmWeightCount(in, out)
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

func lstmLayerMatchesSeed(l *VolumetricLayer, seed uint64, dtype string) (bool, error) {
	if l == nil || l.WeightStore == nil {
		return false, nil
	}
	fp, err := lstmInitWeights(l.InputHeight, l.OutputHeight, seed, dtype)
	if err != nil {
		return false, err
	}
	return weightStoreFingerprint(l.WeightStore) == fp, nil
}

func lstmManifestForwardFP(m *LSTMWeightManifest) (uint64, error) {
	net, err := BuildLSTMVolumetricFromManifest(m)
	if err != nil {
		return 0, err
	}
	if len(m.Sizes) == 0 {
		return 0, fmt.Errorf("lstm: empty sizes")
	}
	in := NewTensorFromSlice(seedDemoForwardInput(m.Sizes[0]), 1, m.Sizes[0])
	out, _, _ := ForwardPolymorphic(net, in)
	if out == nil {
		return 0, fmt.Errorf("lstm: forward nil")
	}
	return seedOutputHash(out.Data), nil
}
