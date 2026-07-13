package poly

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"strconv"
)

const cnnManifestFormat = "loom-cnn-manifest-v1"

// CNNSpec is one conv layer: Dim 1/2/3 selects CNN1/CNN2/CNN3.
type CNNSpec struct {
	Dim           int `json:"dim"`
	InputChannels int `json:"input_channels"`
	Filters       int `json:"filters"`
	Spatial       int `json:"spatial"`
	KernelSize    int `json:"kernel_size"`
	Stride        int `json:"stride,omitempty"`
	Padding       int `json:"padding,omitempty"`
}

// CNNLayerManifest is one CNN layer seed + dtype (no weight bytes).
type CNNLayerManifest struct {
	Index         int    `json:"index"`
	Path          string `json:"path"`
	Dim           int    `json:"dim"`
	InputChannels int    `json:"input_channels"`
	Filters       int    `json:"filters"`
	Spatial       int    `json:"spatial"`
	KernelSize    int    `json:"kernel_size"`
	Stride        int    `json:"stride"`
	Padding       int    `json:"padding"`
	LayerSeed     uint64 `json:"layer_seed"`
	DType         string `json:"dtype"`
	WeightFP      uint64 `json:"weight_fp"`
}

// CNNWeightManifest is topology seed + per-layer weight seeds.
type CNNWeightManifest struct {
	Format       string             `json:"format"`
	TopologySeed uint64             `json:"topology_seed"`
	Specs        []CNNSpec          `json:"specs"`
	Layers       []CNNLayerManifest `json:"layers"`
	NetworkFP    uint64             `json:"network_fp"`
	ForwardFP    uint64             `json:"forward_fp"`
}

// CNNTopologySeed hashes CNN block shapes into a topology-only seed.
func CNNTopologySeed(name string, specs []CNNSpec) uint64 {
	parts := []any{"loom-cnn-v1", name}
	for _, s := range specs {
		n, err := normalizeCNNSpec(s)
		if err != nil {
			continue
		}
		parts = append(parts, n.Dim, n.InputChannels, n.Filters, n.Spatial, n.KernelSize, n.Stride, n.Padding)
	}
	return SeedFrom(parts...)
}

// CNNLayerWeightSeed derives per-layer weight seed from topology seed.
func CNNLayerWeightSeed(topologySeed uint64, spec CNNSpec, layerIndex int) uint64 {
	n, _ := normalizeCNNSpec(spec)
	return DeriveLayerSeed(topologySeed, layerIndex, cnnLayerPath(n.Dim, layerIndex))
}

func cnnLayerPath(dim, index int) string {
	return "cnn" + strconv.Itoa(dim) + "." + strconv.Itoa(index)
}

// BuildCNNManifest creates per-layer seeds and fingerprints.
func BuildCNNManifest(topologySeed uint64, specs []CNNSpec, dtypes []string) (*CNNWeightManifest, error) {
	if len(specs) == 0 {
		return nil, fmt.Errorf("cnn: need at least one spec")
	}
	m := &CNNWeightManifest{
		Format:       cnnManifestFormat,
		TopologySeed: topologySeed,
		Specs:        append([]CNNSpec(nil), specs...),
	}
	for i, raw := range specs {
		spec, err := normalizeCNNSpec(raw)
		if err != nil {
			return nil, fmt.Errorf("cnn: layer %d: %w", i, err)
		}
		if i > 0 {
			prev, _ := normalizeCNNSpec(specs[i-1])
			if spec.Dim != prev.Dim {
				return nil, fmt.Errorf("cnn: chained layers must share dim, layer %d has %d want %d", i, spec.Dim, prev.Dim)
			}
			if spec.Spatial != prev.Spatial {
				return nil, fmt.Errorf("cnn: chained layers must share spatial=%d, layer %d has %d", prev.Spatial, i, spec.Spatial)
			}
			if spec.InputChannels != prev.Filters {
				return nil, fmt.Errorf("cnn: layer %d input_channels=%d must match prev filters=%d", i, spec.InputChannels, prev.Filters)
			}
		}
		m.Specs[i] = spec
		dt := "float32"
		if i < len(dtypes) && dtypes[i] != "" {
			dt = dtypes[i]
		}
		layerSeed := CNNLayerWeightSeed(topologySeed, spec, i)
		fp, err := cnnInitWeights(spec, layerSeed, dt)
		if err != nil {
			return nil, err
		}
		m.Layers = append(m.Layers, cnnLayerManifestFromSpec(i, spec, layerSeed, dt, fp))
	}
	m.NetworkFP = cnnManifestNetworkFP(m.Layers)
	fp, err := cnnManifestForwardFP(m)
	if err != nil {
		return nil, err
	}
	m.ForwardFP = fp
	return m, nil
}

// RebuildCNNManifest verifies seeds-only rebuild matches fingerprints.
func RebuildCNNManifest(m *CNNWeightManifest) (*CNNWeightManifest, error) {
	if m == nil {
		return nil, fmt.Errorf("cnn: nil manifest")
	}
	rebuilt, err := BuildCNNManifest(m.TopologySeed, m.Specs, cnnLayerDTypes(m))
	if err != nil {
		return nil, err
	}
	if rebuilt.NetworkFP != m.NetworkFP {
		return nil, fmt.Errorf("cnn: network fp mismatch got 0x%x want 0x%x", rebuilt.NetworkFP, m.NetworkFP)
	}
	if m.ForwardFP != 0 && rebuilt.ForwardFP != m.ForwardFP {
		return nil, fmt.Errorf("cnn: forward fp mismatch got 0x%x want 0x%x", rebuilt.ForwardFP, m.ForwardFP)
	}
	return rebuilt, nil
}

// BuildCNNVolumetricFromManifest builds a poly VolumetricNetwork from CNN manifest seeds.
func BuildCNNVolumetricFromManifest(m *CNNWeightManifest) (*VolumetricNetwork, error) {
	if m == nil || len(m.Layers) == 0 {
		return nil, fmt.Errorf("cnn: empty manifest")
	}
	net := NewVolumetricNetwork(1, 1, 1, len(m.Layers))
	net.InitSeed = m.TopologySeed
	for i, layer := range m.Layers {
		l := net.GetLayer(0, 0, 0, i)
		spec := cnnSpecFromManifest(layer)
		applyCNNSpec(l, spec)
		l.DType = ParseDType(layer.DType)
		if l.DType == 0 {
			l.DType = DTypeFloat32
		}
		wCount := cnnWeightCount(spec)
		l.WeightStore = NewWeightStore(wCount)
		InitWeightStoreHeSeeded(l.WeightStore, spec.InputChannels, layer.LayerSeed)
		if l.DType != DTypeFloat32 {
			l.WeightStore.Morph(l.DType)
		}
	}
	return net, nil
}

// ManifestFromCNNNetwork extracts layer seeds from a seeded CNN volumetric stack.
func ManifestFromCNNNetwork(net *VolumetricNetwork, topologySeed uint64, specs []CNNSpec, dtypes []string) (*CNNWeightManifest, error) {
	if net == nil {
		return nil, fmt.Errorf("cnn: nil network")
	}
	if len(specs) == 0 {
		return nil, fmt.Errorf("cnn: need specs")
	}
	m := &CNNWeightManifest{
		Format: cnnManifestFormat, TopologySeed: topologySeed,
		Specs: append([]CNNSpec(nil), specs...),
	}
	for i, raw := range specs {
		spec, err := normalizeCNNSpec(raw)
		if err != nil {
			return nil, fmt.Errorf("cnn: layer %d: %w", i, err)
		}
		m.Specs[i] = spec
		dt := "float32"
		if i < len(dtypes) && dtypes[i] != "" {
			dt = dtypes[i]
		}
		l := net.GetLayer(0, 0, 0, i)
		wantType := cnnLayerType(spec.Dim)
		if l.Type != wantType {
			return nil, fmt.Errorf("cnn: layer %d type %v want %v", i, l.Type, wantType)
		}
		seed := CNNLayerWeightSeed(topologySeed, spec, i)
		ok, err := cnnLayerMatchesSeed(l, seed, dt)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, fmt.Errorf("cnn: layer %d weights do not match seed 0x%x", i, seed)
		}
		fp := weightStoreFingerprint(l.WeightStore)
		m.Layers = append(m.Layers, cnnLayerManifestFromSpec(i, spec, seed, dt, fp))
	}
	m.NetworkFP = cnnManifestNetworkFP(m.Layers)
	fp, err := cnnManifestForwardFP(m)
	if err != nil {
		return nil, err
	}
	m.ForwardFP = fp
	return m, nil
}

// CNNDemoInput builds a deterministic forward tensor for the first layer spec.
func CNNDemoInput(spec CNNSpec) *Tensor[float32] {
	n, err := normalizeCNNSpec(spec)
	if err != nil {
		return nil
	}
	elems := n.InputChannels
	for d := 0; d < n.Dim; d++ {
		elems *= n.Spatial
	}
	data := seedDemoForwardInput(elems)
	switch n.Dim {
	case 1:
		return NewTensorFromSlice(data, 1, n.InputChannels, n.Spatial)
	case 2:
		return NewTensorFromSlice(data, 1, n.InputChannels, n.Spatial, n.Spatial)
	case 3:
		return NewTensorFromSlice(data, 1, n.InputChannels, n.Spatial, n.Spatial, n.Spatial)
	default:
		return nil
	}
}

// MarshalCNNManifest JSON-encodes a manifest.
func MarshalCNNManifest(m *CNNWeightManifest) ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

// ParseCNNManifest decodes JSON manifest.
func ParseCNNManifest(data []byte) (*CNNWeightManifest, error) {
	var m CNNWeightManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

func normalizeCNNSpec(spec CNNSpec) (CNNSpec, error) {
	if spec.Dim < 1 || spec.Dim > 3 {
		return spec, fmt.Errorf("dim must be 1, 2, or 3")
	}
	if spec.InputChannels <= 0 {
		return spec, fmt.Errorf("input_channels must be positive")
	}
	if spec.Filters <= 0 {
		return spec, fmt.Errorf("filters must be positive")
	}
	if spec.Spatial <= 0 {
		return spec, fmt.Errorf("spatial must be positive")
	}
	if spec.KernelSize <= 0 {
		return spec, fmt.Errorf("kernel_size must be positive")
	}
	if spec.Stride <= 0 {
		spec.Stride = 1
	}
	if spec.Padding < 0 {
		spec.Padding = 0
	}
	if spec.Padding == 0 && spec.Stride == 1 {
		spec.Padding = 1
	}
	return spec, nil
}

func cnnLayerType(dim int) LayerType {
	switch dim {
	case 1:
		return LayerCNN1
	case 2:
		return LayerCNN2
	case 3:
		return LayerCNN3
	default:
		return LayerCNN1
	}
}

func cnnWeightCount(spec CNNSpec) int {
	k := spec.KernelSize
	n := spec.Filters * spec.InputChannels * k
	switch spec.Dim {
	case 2:
		n *= k
	case 3:
		n *= k * k
	}
	return n
}

func applyCNNSpec(l *VolumetricLayer, spec CNNSpec) {
	l.Type = cnnLayerType(spec.Dim)
	l.InputChannels = spec.InputChannels
	l.Filters = spec.Filters
	l.KernelSize = spec.KernelSize
	l.Stride = spec.Stride
	l.Padding = spec.Padding
	l.Activation = ActivationReLU
	switch spec.Dim {
	case 1:
		l.InputHeight = spec.Spatial
		l.OutputHeight = spec.Spatial
	case 2:
		l.InputHeight = spec.Spatial
		l.InputWidth = spec.Spatial
		l.OutputHeight = spec.Spatial
		l.OutputWidth = spec.Spatial
	case 3:
		l.InputDepth = spec.Spatial
		l.InputHeight = spec.Spatial
		l.InputWidth = spec.Spatial
		l.OutputDepth = spec.Spatial
		l.OutputHeight = spec.Spatial
		l.OutputWidth = spec.Spatial
	}
}

func cnnSpecFromManifest(layer CNNLayerManifest) CNNSpec {
	return CNNSpec{
		Dim: layer.Dim, InputChannels: layer.InputChannels, Filters: layer.Filters,
		Spatial: layer.Spatial, KernelSize: layer.KernelSize, Stride: layer.Stride, Padding: layer.Padding,
	}
}

func cnnLayerManifestFromSpec(index int, spec CNNSpec, seed uint64, dtype string, fp uint64) CNNLayerManifest {
	return CNNLayerManifest{
		Index: index, Path: cnnLayerPath(spec.Dim, index),
		Dim: spec.Dim, InputChannels: spec.InputChannels, Filters: spec.Filters,
		Spatial: spec.Spatial, KernelSize: spec.KernelSize, Stride: spec.Stride, Padding: spec.Padding,
		LayerSeed: seed, DType: dtype, WeightFP: fp,
	}
}

func cnnLayerDTypes(m *CNNWeightManifest) []string {
	out := make([]string, len(m.Layers))
	for i, l := range m.Layers {
		out[i] = l.DType
	}
	return out
}

func cnnManifestNetworkFP(layers []CNNLayerManifest) uint64 {
	h := fnv.New64a()
	var buf [8]byte
	for _, l := range layers {
		binary.LittleEndian.PutUint64(buf[:], l.WeightFP)
		_, _ = h.Write(buf[:])
	}
	return h.Sum64()
}

func cnnInitWeights(spec CNNSpec, seed uint64, dtype string) (uint64, error) {
	wCount := cnnWeightCount(spec)
	ws := NewWeightStore(wCount)
	InitWeightStoreHeSeeded(ws, spec.InputChannels, seed)
	dt := ParseDType(dtype)
	if dt == 0 {
		dt = DTypeFloat32
	}
	if dt != DTypeFloat32 {
		ws.Morph(dt)
	}
	return weightStoreFingerprint(ws), nil
}

func cnnLayerMatchesSeed(l *VolumetricLayer, seed uint64, dtype string) (bool, error) {
	if l == nil || l.WeightStore == nil {
		return false, nil
	}
	spec := CNNSpec{
		Dim: cnnDimFromLayerType(l.Type), InputChannels: l.InputChannels, Filters: l.Filters,
		Spatial: cnnSpatialFromLayer(l), KernelSize: l.KernelSize, Stride: l.Stride, Padding: l.Padding,
	}
	spec, err := normalizeCNNSpec(spec)
	if err != nil {
		return false, err
	}
	fp, err := cnnInitWeights(spec, seed, dtype)
	if err != nil {
		return false, err
	}
	return weightStoreFingerprint(l.WeightStore) == fp, nil
}

func cnnDimFromLayerType(t LayerType) int {
	switch t {
	case LayerCNN1:
		return 1
	case LayerCNN2:
		return 2
	case LayerCNN3:
		return 3
	default:
		return 1
	}
}

func cnnSpatialFromLayer(l *VolumetricLayer) int {
	if l == nil {
		return 0
	}
	switch l.Type {
	case LayerCNN1:
		return l.InputHeight
	case LayerCNN2:
		return l.InputHeight
	case LayerCNN3:
		return l.InputDepth
	default:
		return l.InputHeight
	}
}

func cnnManifestForwardFP(m *CNNWeightManifest) (uint64, error) {
	net, err := BuildCNNVolumetricFromManifest(m)
	if err != nil {
		return 0, err
	}
	if len(m.Specs) == 0 {
		return 0, fmt.Errorf("cnn: empty specs")
	}
	in := CNNDemoInput(m.Specs[0])
	if in == nil {
		return 0, fmt.Errorf("cnn: demo input nil")
	}
	out, _, _ := ForwardPolymorphic(net, in)
	if out == nil {
		return 0, fmt.Errorf("cnn: forward nil")
	}
	return seedOutputHash(out.Data), nil
}
