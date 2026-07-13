package poly

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"strconv"
)

const swigluManifestFormat = "loom-swiglu-manifest-v1"

// SwiGLUSpec is one SwiGLU block: hidden input dim and intermediate (FFN) width.
type SwiGLUSpec struct {
	Hidden       int `json:"hidden"`
	Intermediate int `json:"intermediate"`
}

// SwiGLULayerManifest is one SwiGLU layer seed + dtype (no weight bytes).
type SwiGLULayerManifest struct {
	Index        int    `json:"index"`
	Path         string `json:"path"`
	Hidden       int    `json:"hidden"`
	Intermediate int    `json:"intermediate"`
	LayerSeed    uint64 `json:"layer_seed"`
	DType        string `json:"dtype"`
	WeightFP     uint64 `json:"weight_fp"`
}

// SwiGLUWeightManifest is topology seed + per-layer weight seeds.
type SwiGLUWeightManifest struct {
	Format       string                `json:"format"`
	TopologySeed uint64                `json:"topology_seed"`
	Specs        []SwiGLUSpec          `json:"specs"`
	Layers       []SwiGLULayerManifest `json:"layers"`
	NetworkFP    uint64                `json:"network_fp"`
	ForwardFP    uint64                `json:"forward_fp"`
}

// SwiGLUTopologySeed hashes SwiGLU block shapes into a topology-only seed.
func SwiGLUTopologySeed(name string, specs []SwiGLUSpec) uint64 {
	parts := []any{"loom-swiglu-v1", name}
	for _, s := range specs {
		parts = append(parts, s.Hidden, s.Intermediate)
	}
	return SeedFrom(parts...)
}

// SwiGLULayerWeightSeed derives per-layer weight seed from topology seed.
func SwiGLULayerWeightSeed(topologySeed uint64, layerIndex int) uint64 {
	return DeriveLayerSeed(topologySeed, layerIndex, swigluLayerPath(layerIndex))
}

func swigluLayerPath(index int) string {
	return "swiglu." + strconv.Itoa(index)
}

// BuildSwiGLUManifest creates per-layer seeds and fingerprints.
func BuildSwiGLUManifest(topologySeed uint64, specs []SwiGLUSpec, dtypes []string) (*SwiGLUWeightManifest, error) {
	if len(specs) == 0 {
		return nil, fmt.Errorf("swiglu: need at least one spec")
	}
	hidden := specs[0].Hidden
	if hidden <= 0 {
		return nil, fmt.Errorf("swiglu: hidden must be positive")
	}
	m := &SwiGLUWeightManifest{
		Format:       swigluManifestFormat,
		TopologySeed: topologySeed,
		Specs:        append([]SwiGLUSpec(nil), specs...),
	}
	for i, spec := range specs {
		if spec.Hidden != hidden {
			return nil, fmt.Errorf("swiglu: chained layers must share hidden=%d, layer %d has %d", hidden, i, spec.Hidden)
		}
		if spec.Intermediate <= 0 {
			return nil, fmt.Errorf("swiglu: layer %d intermediate must be positive", i)
		}
		dt := "float32"
		if i < len(dtypes) && dtypes[i] != "" {
			dt = dtypes[i]
		}
		layerSeed := SwiGLULayerWeightSeed(topologySeed, i)
		fp, err := swigluInitWeights(spec.Hidden, spec.Intermediate, layerSeed, dt)
		if err != nil {
			return nil, err
		}
		m.Layers = append(m.Layers, SwiGLULayerManifest{
			Index: i, Path: swigluLayerPath(i), Hidden: spec.Hidden, Intermediate: spec.Intermediate,
			LayerSeed: layerSeed, DType: dt, WeightFP: fp,
		})
	}
	m.NetworkFP = swigluManifestNetworkFP(m.Layers)
	fp, err := swigluManifestForwardFP(m)
	if err != nil {
		return nil, err
	}
	m.ForwardFP = fp
	return m, nil
}

// RebuildSwiGLUManifest verifies seeds-only rebuild matches fingerprints.
func RebuildSwiGLUManifest(m *SwiGLUWeightManifest) (*SwiGLUWeightManifest, error) {
	if m == nil {
		return nil, fmt.Errorf("swiglu: nil manifest")
	}
	rebuilt, err := BuildSwiGLUManifest(m.TopologySeed, m.Specs, swigluLayerDTypes(m))
	if err != nil {
		return nil, err
	}
	if rebuilt.NetworkFP != m.NetworkFP {
		return nil, fmt.Errorf("swiglu: network fp mismatch got 0x%x want 0x%x", rebuilt.NetworkFP, m.NetworkFP)
	}
	if m.ForwardFP != 0 && rebuilt.ForwardFP != m.ForwardFP {
		return nil, fmt.Errorf("swiglu: forward fp mismatch got 0x%x want 0x%x", rebuilt.ForwardFP, m.ForwardFP)
	}
	return rebuilt, nil
}

// BuildSwiGLUVolumetricFromManifest builds a poly VolumetricNetwork from SwiGLU manifest seeds.
func BuildSwiGLUVolumetricFromManifest(m *SwiGLUWeightManifest) (*VolumetricNetwork, error) {
	if m == nil || len(m.Layers) == 0 {
		return nil, fmt.Errorf("swiglu: empty manifest")
	}
	net := NewVolumetricNetwork(1, 1, 1, len(m.Layers))
	net.InitSeed = m.TopologySeed
	for i, layer := range m.Layers {
		l := net.GetLayer(0, 0, 0, i)
		l.Type = LayerSwiGLU
		l.Activation = ActivationSilu
		l.InputHeight = layer.Hidden
		l.OutputHeight = layer.Intermediate
		l.DType = ParseDType(layer.DType)
		if l.DType == 0 {
			l.DType = DTypeFloat32
		}
		wCount := 3*layer.Hidden*layer.Intermediate + 2*layer.Intermediate + layer.Hidden
		l.WeightStore = NewWeightStore(wCount)
		InitWeightStoreHeSeeded(l.WeightStore, layer.Hidden, layer.LayerSeed)
		if l.DType != DTypeFloat32 {
			l.WeightStore.Morph(l.DType)
		}
	}
	return net, nil
}

// ManifestFromSwiGLUNetwork extracts layer seeds from a seeded SwiGLU volumetric stack.
func ManifestFromSwiGLUNetwork(net *VolumetricNetwork, topologySeed uint64, specs []SwiGLUSpec, dtypes []string) (*SwiGLUWeightManifest, error) {
	if net == nil {
		return nil, fmt.Errorf("swiglu: nil network")
	}
	if len(specs) == 0 {
		return nil, fmt.Errorf("swiglu: need specs")
	}
	m := &SwiGLUWeightManifest{
		Format: swigluManifestFormat, TopologySeed: topologySeed,
		Specs: append([]SwiGLUSpec(nil), specs...),
	}
	for i, spec := range specs {
		dt := "float32"
		if i < len(dtypes) && dtypes[i] != "" {
			dt = dtypes[i]
		}
		l := net.GetLayer(0, 0, 0, i)
		if l.Type != LayerSwiGLU {
			return nil, fmt.Errorf("swiglu: layer %d type %v", i, l.Type)
		}
		seed := SwiGLULayerWeightSeed(topologySeed, i)
		ok, err := swigluLayerMatchesSeed(l, seed, dt)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, fmt.Errorf("swiglu: layer %d weights do not match seed 0x%x", i, seed)
		}
		fp := weightStoreFingerprint(l.WeightStore)
		m.Layers = append(m.Layers, SwiGLULayerManifest{
			Index: i, Path: swigluLayerPath(i), Hidden: spec.Hidden, Intermediate: spec.Intermediate,
			LayerSeed: seed, DType: dt, WeightFP: fp,
		})
	}
	m.NetworkFP = swigluManifestNetworkFP(m.Layers)
	fp, err := swigluManifestForwardFP(m)
	if err != nil {
		return nil, err
	}
	m.ForwardFP = fp
	return m, nil
}

// MarshalSwiGLUManifest JSON-encodes a manifest.
func MarshalSwiGLUManifest(m *SwiGLUWeightManifest) ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

// ParseSwiGLUManifest decodes JSON manifest.
func ParseSwiGLUManifest(data []byte) (*SwiGLUWeightManifest, error) {
	var m SwiGLUWeightManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

func swigluLayerDTypes(m *SwiGLUWeightManifest) []string {
	out := make([]string, len(m.Layers))
	for i, l := range m.Layers {
		out[i] = l.DType
	}
	return out
}

func swigluManifestNetworkFP(layers []SwiGLULayerManifest) uint64 {
	h := fnv.New64a()
	var buf [8]byte
	for _, l := range layers {
		binary.LittleEndian.PutUint64(buf[:], l.WeightFP)
		_, _ = h.Write(buf[:])
	}
	return h.Sum64()
}

func swigluInitWeights(hidden, intermediate int, seed uint64, dtype string) (uint64, error) {
	wCount := 3*hidden*intermediate + 2*intermediate + hidden
	ws := NewWeightStore(wCount)
	InitWeightStoreHeSeeded(ws, hidden, seed)
	dt := ParseDType(dtype)
	if dt == 0 {
		dt = DTypeFloat32
	}
	if dt != DTypeFloat32 {
		ws.Morph(dt)
	}
	return weightStoreFingerprint(ws), nil
}

func swigluLayerMatchesSeed(l *VolumetricLayer, seed uint64, dtype string) (bool, error) {
	if l == nil || l.WeightStore == nil {
		return false, nil
	}
	fp, err := swigluInitWeights(l.InputHeight, l.OutputHeight, seed, dtype)
	if err != nil {
		return false, err
	}
	return weightStoreFingerprint(l.WeightStore) == fp, nil
}

func swigluManifestForwardFP(m *SwiGLUWeightManifest) (uint64, error) {
	net, err := BuildSwiGLUVolumetricFromManifest(m)
	if err != nil {
		return 0, err
	}
	if len(m.Specs) == 0 {
		return 0, fmt.Errorf("swiglu: empty specs")
	}
	in := NewTensorFromSlice(seedDemoForwardInput(m.Specs[0].Hidden), 1, m.Specs[0].Hidden)
	out, _, _ := ForwardPolymorphic(net, in)
	if out == nil {
		return 0, fmt.Errorf("swiglu: forward nil")
	}
	return seedOutputHash(out.Data), nil
}
