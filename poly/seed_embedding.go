package poly

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"strconv"
)

const embeddingManifestFormat = "loom-embedding-manifest-v1"

// EmbeddingSpec is one embedding table (token lookup).
type EmbeddingSpec struct {
	VocabSize    int `json:"vocab_size"`
	EmbeddingDim int `json:"embedding_dim"`
	SeqLen       int `json:"seq_len,omitempty"`
}

// EmbeddingLayerManifest is one embedding table seed + dtype (no weight bytes).
type EmbeddingLayerManifest struct {
	Index        int    `json:"index"`
	Path         string `json:"path"`
	VocabSize    int    `json:"vocab_size"`
	EmbeddingDim int    `json:"embedding_dim"`
	SeqLen       int    `json:"seq_len"`
	LayerSeed    uint64 `json:"layer_seed"`
	DType        string `json:"dtype"`
	WeightFP     uint64 `json:"weight_fp"`
}

// EmbeddingWeightManifest is topology seed + per-table seeds.
type EmbeddingWeightManifest struct {
	Format       string                   `json:"format"`
	TopologySeed uint64                   `json:"topology_seed"`
	Specs        []EmbeddingSpec          `json:"specs"`
	Layers       []EmbeddingLayerManifest `json:"layers"`
	NetworkFP    uint64                   `json:"network_fp"`
	ForwardFP    uint64                   `json:"forward_fp"`
}

// EmbeddingTopologySeed hashes embedding shapes into a topology-only seed.
func EmbeddingTopologySeed(name string, specs []EmbeddingSpec) uint64 {
	parts := []any{"loom-embedding-v1", name}
	for _, s := range specs {
		n, err := normalizeEmbeddingSpec(s)
		if err != nil {
			continue
		}
		parts = append(parts, n.VocabSize, n.EmbeddingDim, n.SeqLen)
	}
	return SeedFrom(parts...)
}

// EmbeddingLayerWeightSeed derives per-table seed from topology seed.
func EmbeddingLayerWeightSeed(topologySeed uint64, layerIndex int) uint64 {
	return DeriveLayerSeed(topologySeed, layerIndex, embeddingLayerPath(layerIndex))
}

func embeddingLayerPath(index int) string {
	return "embedding." + strconv.Itoa(index)
}

// BuildEmbeddingManifest creates per-table seeds and fingerprints.
func BuildEmbeddingManifest(topologySeed uint64, specs []EmbeddingSpec, dtypes []string) (*EmbeddingWeightManifest, error) {
	if len(specs) == 0 {
		return nil, fmt.Errorf("embedding: need at least one spec")
	}
	vocab := specs[0].VocabSize
	seqLen := 0
	m := &EmbeddingWeightManifest{
		Format:       embeddingManifestFormat,
		TopologySeed: topologySeed,
		Specs:        append([]EmbeddingSpec(nil), specs...),
	}
	for i, raw := range specs {
		spec, err := normalizeEmbeddingSpec(raw)
		if err != nil {
			return nil, fmt.Errorf("embedding: layer %d: %w", i, err)
		}
		if spec.VocabSize != vocab {
			return nil, fmt.Errorf("embedding: tables must share vocab_size=%d, layer %d has %d", vocab, i, spec.VocabSize)
		}
		if seqLen == 0 {
			seqLen = spec.SeqLen
		} else if spec.SeqLen != seqLen {
			return nil, fmt.Errorf("embedding: tables must share seq_len=%d, layer %d has %d", seqLen, i, spec.SeqLen)
		}
		m.Specs[i] = spec
		dt := "float32"
		if i < len(dtypes) && dtypes[i] != "" {
			dt = dtypes[i]
		}
		layerSeed := EmbeddingLayerWeightSeed(topologySeed, i)
		fp, err := embeddingInitWeights(spec, layerSeed, dt)
		if err != nil {
			return nil, err
		}
		m.Layers = append(m.Layers, embeddingLayerManifestFromSpec(i, spec, layerSeed, dt, fp))
	}
	m.NetworkFP = embeddingManifestNetworkFP(m.Layers)
	fp, err := embeddingManifestForwardFP(m)
	if err != nil {
		return nil, err
	}
	m.ForwardFP = fp
	return m, nil
}

// RebuildEmbeddingManifest verifies seeds-only rebuild matches fingerprints.
func RebuildEmbeddingManifest(m *EmbeddingWeightManifest) (*EmbeddingWeightManifest, error) {
	if m == nil {
		return nil, fmt.Errorf("embedding: nil manifest")
	}
	rebuilt, err := BuildEmbeddingManifest(m.TopologySeed, m.Specs, embeddingLayerDTypes(m))
	if err != nil {
		return nil, err
	}
	if rebuilt.NetworkFP != m.NetworkFP {
		return nil, fmt.Errorf("embedding: network fp mismatch got 0x%x want 0x%x", rebuilt.NetworkFP, m.NetworkFP)
	}
	if m.ForwardFP != 0 && rebuilt.ForwardFP != m.ForwardFP {
		return nil, fmt.Errorf("embedding: forward fp mismatch got 0x%x want 0x%x", rebuilt.ForwardFP, m.ForwardFP)
	}
	return rebuilt, nil
}

// BuildEmbeddingVolumetricFromManifest builds one embedding layer per manifest entry.
func BuildEmbeddingVolumetricFromManifest(m *EmbeddingWeightManifest) (*VolumetricNetwork, error) {
	if m == nil || len(m.Layers) == 0 {
		return nil, fmt.Errorf("embedding: empty manifest")
	}
	net := NewVolumetricNetwork(1, 1, 1, len(m.Layers))
	net.InitSeed = m.TopologySeed
	for i, layer := range m.Layers {
		l := net.GetLayer(0, 0, 0, i)
		spec := embeddingSpecFromManifest(layer)
		applyEmbeddingSpec(l, spec)
		l.DType = ParseDType(layer.DType)
		if l.DType == 0 {
			l.DType = DTypeFloat32
		}
		wCount := spec.VocabSize * spec.EmbeddingDim
		l.WeightStore = NewWeightStore(wCount)
		InitWeightStoreHeSeeded(l.WeightStore, spec.EmbeddingDim, layer.LayerSeed)
		if l.DType != DTypeFloat32 {
			l.WeightStore.Morph(l.DType)
		}
	}
	return net, nil
}

// ManifestFromEmbeddingNetwork extracts table seeds from a seeded embedding stack.
func ManifestFromEmbeddingNetwork(net *VolumetricNetwork, topologySeed uint64, specs []EmbeddingSpec, dtypes []string) (*EmbeddingWeightManifest, error) {
	if net == nil {
		return nil, fmt.Errorf("embedding: nil network")
	}
	if len(specs) == 0 {
		return nil, fmt.Errorf("embedding: need specs")
	}
	m := &EmbeddingWeightManifest{
		Format: embeddingManifestFormat, TopologySeed: topologySeed,
		Specs: append([]EmbeddingSpec(nil), specs...),
	}
	for i, raw := range specs {
		spec, err := normalizeEmbeddingSpec(raw)
		if err != nil {
			return nil, fmt.Errorf("embedding: layer %d: %w", i, err)
		}
		m.Specs[i] = spec
		dt := "float32"
		if i < len(dtypes) && dtypes[i] != "" {
			dt = dtypes[i]
		}
		l := net.GetLayer(0, 0, 0, i)
		if l.Type != LayerEmbedding {
			return nil, fmt.Errorf("embedding: layer %d type %v", i, l.Type)
		}
		seed := EmbeddingLayerWeightSeed(topologySeed, i)
		ok, err := embeddingLayerMatchesSeed(l, seed, dt)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, fmt.Errorf("embedding: layer %d weights do not match seed 0x%x", i, seed)
		}
		fp := weightStoreFingerprint(l.WeightStore)
		m.Layers = append(m.Layers, embeddingLayerManifestFromSpec(i, spec, seed, dt, fp))
	}
	m.NetworkFP = embeddingManifestNetworkFP(m.Layers)
	fp, err := embeddingManifestForwardFP(m)
	if err != nil {
		return nil, err
	}
	m.ForwardFP = fp
	return m, nil
}

// EmbeddingDemoTokens builds deterministic token-id input for embedding forward.
func EmbeddingDemoTokens(vocab, seqLen int) *Tensor[float32] {
	if vocab <= 0 {
		vocab = 1
	}
	if seqLen <= 0 {
		seqLen = 8
	}
	data := make([]float32, seqLen)
	for i := range data {
		data[i] = float32(i % vocab)
	}
	return NewTensorFromSlice(data, seqLen, 1)
}

// ForwardEmbeddingManifest looks up the same tokens on every table and hashes concatenated outputs.
func ForwardEmbeddingManifest(m *EmbeddingWeightManifest) ([]float32, error) {
	if m == nil || len(m.Layers) == 0 {
		return nil, fmt.Errorf("embedding: empty manifest")
	}
	net, err := BuildEmbeddingVolumetricFromManifest(m)
	if err != nil {
		return nil, err
	}
	seqLen := m.Layers[0].SeqLen
	vocab := m.Layers[0].VocabSize
	tokens := EmbeddingDemoTokens(vocab, seqLen)
	var out []float32
	for i := range m.Layers {
		l := net.GetLayer(0, 0, 0, i)
		_, post := EmbeddingForwardPolymorphic(l, tokens)
		if post == nil {
			return nil, fmt.Errorf("embedding: forward nil layer %d", i)
		}
		out = append(out, post.Data...)
	}
	return out, nil
}

// MarshalEmbeddingManifest JSON-encodes a manifest.
func MarshalEmbeddingManifest(m *EmbeddingWeightManifest) ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

// ParseEmbeddingManifest decodes JSON manifest.
func ParseEmbeddingManifest(data []byte) (*EmbeddingWeightManifest, error) {
	var m EmbeddingWeightManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

func normalizeEmbeddingSpec(spec EmbeddingSpec) (EmbeddingSpec, error) {
	if spec.VocabSize <= 0 {
		return spec, fmt.Errorf("vocab_size must be positive")
	}
	if spec.EmbeddingDim <= 0 {
		return spec, fmt.Errorf("embedding_dim must be positive")
	}
	if spec.SeqLen <= 0 {
		spec.SeqLen = 8
	}
	return spec, nil
}

func applyEmbeddingSpec(l *VolumetricLayer, spec EmbeddingSpec) {
	l.Type = LayerEmbedding
	l.VocabSize = spec.VocabSize
	l.EmbeddingDim = spec.EmbeddingDim
	l.Activation = ActivationLinear
}

func embeddingSpecFromManifest(layer EmbeddingLayerManifest) EmbeddingSpec {
	return EmbeddingSpec{
		VocabSize: layer.VocabSize, EmbeddingDim: layer.EmbeddingDim, SeqLen: layer.SeqLen,
	}
}

func embeddingLayerManifestFromSpec(index int, spec EmbeddingSpec, seed uint64, dtype string, fp uint64) EmbeddingLayerManifest {
	return EmbeddingLayerManifest{
		Index: index, Path: embeddingLayerPath(index),
		VocabSize: spec.VocabSize, EmbeddingDim: spec.EmbeddingDim, SeqLen: spec.SeqLen,
		LayerSeed: seed, DType: dtype, WeightFP: fp,
	}
}

func embeddingLayerDTypes(m *EmbeddingWeightManifest) []string {
	out := make([]string, len(m.Layers))
	for i, l := range m.Layers {
		out[i] = l.DType
	}
	return out
}

func embeddingManifestNetworkFP(layers []EmbeddingLayerManifest) uint64 {
	h := fnv.New64a()
	var buf [8]byte
	for _, l := range layers {
		binary.LittleEndian.PutUint64(buf[:], l.WeightFP)
		_, _ = h.Write(buf[:])
	}
	return h.Sum64()
}

func embeddingInitWeights(spec EmbeddingSpec, seed uint64, dtype string) (uint64, error) {
	wCount := spec.VocabSize * spec.EmbeddingDim
	ws := NewWeightStore(wCount)
	InitWeightStoreHeSeeded(ws, spec.EmbeddingDim, seed)
	dt := ParseDType(dtype)
	if dt == 0 {
		dt = DTypeFloat32
	}
	if dt != DTypeFloat32 {
		ws.Morph(dt)
	}
	return weightStoreFingerprint(ws), nil
}

func embeddingLayerMatchesSeed(l *VolumetricLayer, seed uint64, dtype string) (bool, error) {
	if l == nil || l.WeightStore == nil {
		return false, nil
	}
	spec := EmbeddingSpec{VocabSize: l.VocabSize, EmbeddingDim: l.EmbeddingDim, SeqLen: 8}
	spec, err := normalizeEmbeddingSpec(spec)
	if err != nil {
		return false, err
	}
	fp, err := embeddingInitWeights(spec, seed, dtype)
	if err != nil {
		return false, err
	}
	return weightStoreFingerprint(l.WeightStore) == fp, nil
}

func embeddingManifestForwardFP(m *EmbeddingWeightManifest) (uint64, error) {
	out, err := ForwardEmbeddingManifest(m)
	if err != nil {
		return 0, err
	}
	return seedOutputHash(out), nil
}
