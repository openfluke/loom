package poly

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"strconv"
)

const mhaManifestFormat = "loom-mha-manifest-v1"

// MHASpec is one multi-head attention block (Llama-style GQA dims).
type MHASpec struct {
	DModel       int     `json:"d_model"`
	NumHeads     int     `json:"num_heads"`
	NumKVHeads   int     `json:"num_kv_heads,omitempty"`
	HeadDim      int     `json:"head_dim,omitempty"`
	QueryDim     int     `json:"query_dim,omitempty"`
	RoPEFreqBase float64 `json:"rope_freq_base,omitempty"`
}

// MHALayerManifest is one MHA layer seed + dtype (no weight bytes).
type MHALayerManifest struct {
	Index        int     `json:"index"`
	Path         string  `json:"path"`
	DModel       int     `json:"d_model"`
	NumHeads     int     `json:"num_heads"`
	NumKVHeads   int     `json:"num_kv_heads"`
	HeadDim      int     `json:"head_dim"`
	QueryDim     int     `json:"query_dim"`
	RoPEFreqBase float64 `json:"rope_freq_base"`
	LayerSeed    uint64  `json:"layer_seed"`
	DType        string  `json:"dtype"`
	WeightFP     uint64  `json:"weight_fp"`
}

// MHAWeightManifest is topology seed + per-layer weight seeds.
type MHAWeightManifest struct {
	Format       string             `json:"format"`
	TopologySeed uint64             `json:"topology_seed"`
	Specs        []MHASpec          `json:"specs"`
	Layers       []MHALayerManifest `json:"layers"`
	NetworkFP    uint64             `json:"network_fp"`
	ForwardFP    uint64             `json:"forward_fp"`
}

// MHATopologySeed hashes MHA block shapes into a topology-only seed.
func MHATopologySeed(name string, specs []MHASpec) uint64 {
	parts := []any{"loom-mha-v1", name}
	for _, s := range specs {
		n, err := normalizeMHASpec(s)
		if err != nil {
			continue
		}
		parts = append(parts, n.DModel, n.NumHeads, n.NumKVHeads, n.HeadDim, n.QueryDim)
	}
	return SeedFrom(parts...)
}

// MHALayerWeightSeed derives per-layer weight seed from topology seed.
func MHALayerWeightSeed(topologySeed uint64, layerIndex int) uint64 {
	return DeriveLayerSeed(topologySeed, layerIndex, mhaLayerPath(layerIndex))
}

func mhaLayerPath(index int) string {
	return "mha." + strconv.Itoa(index)
}

// BuildMHAManifest creates per-layer seeds and fingerprints.
func BuildMHAManifest(topologySeed uint64, specs []MHASpec, dtypes []string) (*MHAWeightManifest, error) {
	if len(specs) == 0 {
		return nil, fmt.Errorf("mha: need at least one spec")
	}
	dModel := specs[0].DModel
	if dModel <= 0 {
		return nil, fmt.Errorf("mha: d_model must be positive")
	}
	m := &MHAWeightManifest{
		Format:       mhaManifestFormat,
		TopologySeed: topologySeed,
		Specs:        append([]MHASpec(nil), specs...),
	}
	for i, raw := range specs {
		spec, err := normalizeMHASpec(raw)
		if err != nil {
			return nil, fmt.Errorf("mha: layer %d: %w", i, err)
		}
		if spec.DModel != dModel {
			return nil, fmt.Errorf("mha: chained layers must share d_model=%d, layer %d has %d", dModel, i, spec.DModel)
		}
		m.Specs[i] = spec
		dt := "float32"
		if i < len(dtypes) && dtypes[i] != "" {
			dt = dtypes[i]
		}
		layerSeed := MHALayerWeightSeed(topologySeed, i)
		fp, err := mhaInitWeights(spec, layerSeed, dt)
		if err != nil {
			return nil, err
		}
		m.Layers = append(m.Layers, mhaLayerManifestFromSpec(i, spec, layerSeed, dt, fp))
	}
	m.NetworkFP = mhaManifestNetworkFP(m.Layers)
	fp, err := mhaManifestForwardFP(m)
	if err != nil {
		return nil, err
	}
	m.ForwardFP = fp
	return m, nil
}

// RebuildMHAManifest verifies seeds-only rebuild matches fingerprints.
func RebuildMHAManifest(m *MHAWeightManifest) (*MHAWeightManifest, error) {
	if m == nil {
		return nil, fmt.Errorf("mha: nil manifest")
	}
	rebuilt, err := BuildMHAManifest(m.TopologySeed, m.Specs, mhaLayerDTypes(m))
	if err != nil {
		return nil, err
	}
	if rebuilt.NetworkFP != m.NetworkFP {
		return nil, fmt.Errorf("mha: network fp mismatch got 0x%x want 0x%x", rebuilt.NetworkFP, m.NetworkFP)
	}
	if m.ForwardFP != 0 && rebuilt.ForwardFP != m.ForwardFP {
		return nil, fmt.Errorf("mha: forward fp mismatch got 0x%x want 0x%x", rebuilt.ForwardFP, m.ForwardFP)
	}
	return rebuilt, nil
}

// BuildMHAVolumetricFromManifest builds a poly VolumetricNetwork from MHA manifest seeds.
func BuildMHAVolumetricFromManifest(m *MHAWeightManifest) (*VolumetricNetwork, error) {
	if m == nil || len(m.Layers) == 0 {
		return nil, fmt.Errorf("mha: empty manifest")
	}
	net := NewVolumetricNetwork(1, 1, 1, len(m.Layers))
	net.InitSeed = m.TopologySeed
	for i, layer := range m.Layers {
		l := net.GetLayer(0, 0, 0, i)
		spec := mhaSpecFromManifest(layer)
		applyMHASpec(l, spec)
		l.DType = ParseDType(layer.DType)
		if l.DType == 0 {
			l.DType = DTypeFloat32
		}
		wCount := mhaWeightCount(spec)
		l.WeightStore = NewWeightStore(wCount)
		InitWeightStoreHeSeeded(l.WeightStore, spec.DModel, layer.LayerSeed)
		if l.DType != DTypeFloat32 {
			l.WeightStore.Morph(l.DType)
		}
	}
	return net, nil
}

// ManifestFromMHANetwork extracts layer seeds from a seeded MHA volumetric stack.
func ManifestFromMHANetwork(net *VolumetricNetwork, topologySeed uint64, specs []MHASpec, dtypes []string) (*MHAWeightManifest, error) {
	if net == nil {
		return nil, fmt.Errorf("mha: nil network")
	}
	if len(specs) == 0 {
		return nil, fmt.Errorf("mha: need specs")
	}
	m := &MHAWeightManifest{
		Format: mhaManifestFormat, TopologySeed: topologySeed,
		Specs: append([]MHASpec(nil), specs...),
	}
	for i, raw := range specs {
		spec, err := normalizeMHASpec(raw)
		if err != nil {
			return nil, fmt.Errorf("mha: layer %d: %w", i, err)
		}
		m.Specs[i] = spec
		dt := "float32"
		if i < len(dtypes) && dtypes[i] != "" {
			dt = dtypes[i]
		}
		l := net.GetLayer(0, 0, 0, i)
		if l.Type != LayerMultiHeadAttention {
			return nil, fmt.Errorf("mha: layer %d type %v", i, l.Type)
		}
		seed := MHALayerWeightSeed(topologySeed, i)
		ok, err := mhaLayerMatchesSeed(l, seed, dt)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, fmt.Errorf("mha: layer %d weights do not match seed 0x%x", i, seed)
		}
		fp := weightStoreFingerprint(l.WeightStore)
		m.Layers = append(m.Layers, mhaLayerManifestFromSpec(i, spec, seed, dt, fp))
	}
	m.NetworkFP = mhaManifestNetworkFP(m.Layers)
	fp, err := mhaManifestForwardFP(m)
	if err != nil {
		return nil, err
	}
	m.ForwardFP = fp
	return m, nil
}

// MarshalMHAManifest JSON-encodes a manifest.
func MarshalMHAManifest(m *MHAWeightManifest) ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

// ParseMHAManifest decodes JSON manifest.
func ParseMHAManifest(data []byte) (*MHAWeightManifest, error) {
	var m MHAWeightManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

func normalizeMHASpec(spec MHASpec) (MHASpec, error) {
	if spec.DModel <= 0 {
		return spec, fmt.Errorf("d_model must be positive")
	}
	if spec.NumHeads <= 0 {
		return spec, fmt.Errorf("num_heads must be positive")
	}
	if spec.NumKVHeads <= 0 {
		spec.NumKVHeads = spec.NumHeads
	}
	if spec.HeadDim <= 0 {
		if spec.DModel%spec.NumHeads != 0 {
			return spec, fmt.Errorf("cannot derive head_dim from d_model=%d num_heads=%d", spec.DModel, spec.NumHeads)
		}
		spec.HeadDim = spec.DModel / spec.NumHeads
	}
	if spec.QueryDim <= 0 {
		spec.QueryDim = spec.NumHeads * spec.HeadDim
	}
	if spec.RoPEFreqBase <= 0 {
		spec.RoPEFreqBase = 10000
	}
	return spec, nil
}

func mhaWeightCount(spec MHASpec) int {
	q := spec.QueryDim
	kv := spec.NumKVHeads * spec.HeadDim
	d := spec.DModel
	return q*d + kv*d + kv*d + d*q + q + kv + kv + d
}

func applyMHASpec(l *VolumetricLayer, spec MHASpec) {
	l.Type = LayerMultiHeadAttention
	l.DModel = spec.DModel
	l.NumHeads = spec.NumHeads
	l.NumKVHeads = spec.NumKVHeads
	l.HeadDim = spec.HeadDim
	l.QueryDim = spec.QueryDim
	l.RoPEFreqBase = spec.RoPEFreqBase
	l.SeqLength = 1
}

func mhaSpecFromManifest(layer MHALayerManifest) MHASpec {
	return MHASpec{
		DModel: layer.DModel, NumHeads: layer.NumHeads, NumKVHeads: layer.NumKVHeads,
		HeadDim: layer.HeadDim, QueryDim: layer.QueryDim, RoPEFreqBase: layer.RoPEFreqBase,
	}
}

func mhaLayerManifestFromSpec(index int, spec MHASpec, seed uint64, dtype string, fp uint64) MHALayerManifest {
	return MHALayerManifest{
		Index: index, Path: mhaLayerPath(index),
		DModel: spec.DModel, NumHeads: spec.NumHeads, NumKVHeads: spec.NumKVHeads,
		HeadDim: spec.HeadDim, QueryDim: spec.QueryDim, RoPEFreqBase: spec.RoPEFreqBase,
		LayerSeed: seed, DType: dtype, WeightFP: fp,
	}
}

func mhaLayerDTypes(m *MHAWeightManifest) []string {
	out := make([]string, len(m.Layers))
	for i, l := range m.Layers {
		out[i] = l.DType
	}
	return out
}

func mhaManifestNetworkFP(layers []MHALayerManifest) uint64 {
	h := fnv.New64a()
	var buf [8]byte
	for _, l := range layers {
		binary.LittleEndian.PutUint64(buf[:], l.WeightFP)
		_, _ = h.Write(buf[:])
	}
	return h.Sum64()
}

func mhaInitWeights(spec MHASpec, seed uint64, dtype string) (uint64, error) {
	wCount := mhaWeightCount(spec)
	ws := NewWeightStore(wCount)
	InitWeightStoreHeSeeded(ws, spec.DModel, seed)
	dt := ParseDType(dtype)
	if dt == 0 {
		dt = DTypeFloat32
	}
	if dt != DTypeFloat32 {
		ws.Morph(dt)
	}
	return weightStoreFingerprint(ws), nil
}

func mhaLayerMatchesSeed(l *VolumetricLayer, seed uint64, dtype string) (bool, error) {
	if l == nil || l.WeightStore == nil {
		return false, nil
	}
	spec := MHASpec{
		DModel: l.DModel, NumHeads: l.NumHeads, NumKVHeads: l.NumKVHeads,
		HeadDim: l.HeadDim, QueryDim: l.QueryDim,
	}
	spec, err := normalizeMHASpec(spec)
	if err != nil {
		return false, err
	}
	fp, err := mhaInitWeights(spec, seed, dtype)
	if err != nil {
		return false, err
	}
	return weightStoreFingerprint(l.WeightStore) == fp, nil
}

func mhaManifestForwardFP(m *MHAWeightManifest) (uint64, error) {
	net, err := BuildMHAVolumetricFromManifest(m)
	if err != nil {
		return 0, err
	}
	if len(m.Specs) == 0 {
		return 0, fmt.Errorf("mha: empty specs")
	}
	in := NewTensorFromSlice(seedDemoForwardInput(m.Specs[0].DModel), 1, m.Specs[0].DModel)
	out, _, _ := ForwardPolymorphic(net, in)
	if out == nil {
		return 0, fmt.Errorf("mha: forward nil")
	}
	return seedOutputHash(out.Data), nil
}
