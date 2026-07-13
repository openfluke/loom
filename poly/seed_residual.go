package poly

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/fnv"
)

const residualManifestFormat = "loom-residual-manifest-v1"

// ResidualSpec is a dense transform + residual add block (skip has no weights).
type ResidualSpec struct {
	In  int `json:"in"`
	Out int `json:"out"`
}

// ResidualWeightManifest is topology seed + dense branch seed (residual add is weightless).
type ResidualWeightManifest struct {
	Format        string       `json:"format"`
	TopologySeed  uint64       `json:"topology_seed"`
	Spec          ResidualSpec `json:"spec"`
	DenseSeed     uint64       `json:"dense_seed"`
	DType         string       `json:"dtype"`
	DenseWeightFP uint64       `json:"dense_weight_fp"`
	ForwardFP     uint64       `json:"forward_fp"`
}

// ResidualTopologySeed hashes dense branch shape into a topology-only seed.
func ResidualTopologySeed(name string, spec ResidualSpec) uint64 {
	return SeedFrom("loom-residual-v1", name, spec.In, spec.Out)
}

// ResidualDenseSeed derives the dense branch weight seed.
func ResidualDenseSeed(topologySeed uint64) uint64 {
	return DeriveLayerSeed(topologySeed, 0, "residual.dense")
}

// BuildResidualManifest creates dense seed + fingerprints for a residual block.
func BuildResidualManifest(topologySeed uint64, spec ResidualSpec, dtype string) (*ResidualWeightManifest, error) {
	if spec.In <= 0 || spec.Out <= 0 {
		return nil, fmt.Errorf("residual: in and out must be positive")
	}
	if spec.In != spec.Out {
		return nil, fmt.Errorf("residual: in=%d must equal out=%d for skip add", spec.In, spec.Out)
	}
	dt := dtype
	if dt == "" {
		dt = "float32"
	}
	denseSeed := ResidualDenseSeed(topologySeed)
	_, fp, err := denseInitWeights(spec.In, spec.Out, denseSeed, dt)
	if err != nil {
		return nil, err
	}
	m := &ResidualWeightManifest{
		Format: residualManifestFormat, TopologySeed: topologySeed,
		Spec: spec, DenseSeed: denseSeed, DType: dt, DenseWeightFP: fp,
	}
	fwd, err := residualManifestForwardFP(m)
	if err != nil {
		return nil, err
	}
	m.ForwardFP = fwd
	return m, nil
}

// RebuildResidualManifest verifies seeds-only rebuild matches fingerprints.
func RebuildResidualManifest(m *ResidualWeightManifest) (*ResidualWeightManifest, error) {
	if m == nil {
		return nil, fmt.Errorf("residual: nil manifest")
	}
	rebuilt, err := BuildResidualManifest(m.TopologySeed, m.Spec, m.DType)
	if err != nil {
		return nil, err
	}
	if rebuilt.DenseWeightFP != m.DenseWeightFP {
		return nil, fmt.Errorf("residual: weight fp mismatch got 0x%x want 0x%x", rebuilt.DenseWeightFP, m.DenseWeightFP)
	}
	if m.ForwardFP != 0 && rebuilt.ForwardFP != m.ForwardFP {
		return nil, fmt.Errorf("residual: forward fp mismatch got 0x%x want 0x%x", rebuilt.ForwardFP, m.ForwardFP)
	}
	return rebuilt, nil
}

// BuildResidualVolumetricFromManifest builds Dense + Residual layers from manifest seeds.
func BuildResidualVolumetricFromManifest(m *ResidualWeightManifest) (*VolumetricNetwork, error) {
	if m == nil {
		return nil, fmt.Errorf("residual: nil manifest")
	}
	net := NewVolumetricNetwork(1, 1, 1, 2)
	net.InitSeed = m.TopologySeed

	dense := net.GetLayer(0, 0, 0, 0)
	dense.Type = LayerDense
	dense.Activation = ActivationReLU
	dense.InputHeight = m.Spec.In
	dense.OutputHeight = m.Spec.Out
	dense.DType = ParseDType(m.DType)
	if dense.DType == 0 {
		dense.DType = DTypeFloat32
	}
	dense.WeightStore = NewWeightStore(m.Spec.In * m.Spec.Out)
	InitWeightStoreHeSeeded(dense.WeightStore, m.Spec.In, m.DenseSeed)
	if dense.DType != DTypeFloat32 {
		dense.WeightStore.Morph(dense.DType)
	}

	res := net.GetLayer(0, 0, 0, 1)
	res.Type = LayerResidual
	res.InputHeight = m.Spec.Out
	res.OutputHeight = m.Spec.Out
	res.DType = dense.DType

	return net, nil
}

// ManifestFromResidualNetwork extracts dense seed from a seeded residual block.
func ManifestFromResidualNetwork(net *VolumetricNetwork, topologySeed uint64, spec ResidualSpec, dtype string) (*ResidualWeightManifest, error) {
	if net == nil {
		return nil, fmt.Errorf("residual: nil network")
	}
	if len(net.Layers) < 2 {
		return nil, fmt.Errorf("residual: need dense+residual layers")
	}
	dense := net.GetLayer(0, 0, 0, 0)
	res := net.GetLayer(0, 0, 0, 1)
	if dense.Type != LayerDense {
		return nil, fmt.Errorf("residual: layer 0 type %v", dense.Type)
	}
	if res.Type != LayerResidual {
		return nil, fmt.Errorf("residual: layer 1 type %v", res.Type)
	}
	dt := dtype
	if dt == "" {
		dt = "float32"
	}
	seed := ResidualDenseSeed(topologySeed)
	ok, err := denseLayerMatchesSeed(dense, seed, dt)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("residual: dense weights do not match seed 0x%x", seed)
	}
	m := &ResidualWeightManifest{
		Format: residualManifestFormat, TopologySeed: topologySeed,
		Spec: spec, DenseSeed: seed, DType: dt,
		DenseWeightFP: weightStoreFingerprint(dense.WeightStore),
	}
	fp, err := residualManifestForwardFP(m)
	if err != nil {
		return nil, err
	}
	m.ForwardFP = fp
	return m, nil
}

// ForwardResidualManifest runs dense then residual add with skip from input.
func ForwardResidualManifest(m *ResidualWeightManifest) ([]float32, error) {
	net, err := BuildResidualVolumetricFromManifest(m)
	if err != nil {
		return nil, err
	}
	in := NewTensorFromSlice(seedDemoForwardInput(m.Spec.In), 1, m.Spec.In)
	skip := in.Clone()
	dense := net.GetLayer(0, 0, 0, 0)
	res := net.GetLayer(0, 0, 0, 1)
	_, transformed := DenseForwardPolymorphic(dense, in)
	if transformed == nil {
		return nil, fmt.Errorf("residual: dense forward nil")
	}
	_, out := ResidualForwardPolymorphic(res, transformed, skip)
	if out == nil {
		return nil, fmt.Errorf("residual: add forward nil")
	}
	return out.Data, nil
}

// MarshalResidualManifest JSON-encodes a manifest.
func MarshalResidualManifest(m *ResidualWeightManifest) ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

// ParseResidualManifest decodes JSON manifest.
func ParseResidualManifest(data []byte) (*ResidualWeightManifest, error) {
	var m ResidualWeightManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

func residualManifestForwardFP(m *ResidualWeightManifest) (uint64, error) {
	out, err := ForwardResidualManifest(m)
	if err != nil {
		return 0, err
	}
	return seedOutputHash(out), nil
}

// ResidualManifestNetworkFP is the network fingerprint for a residual block.
func ResidualManifestNetworkFP(m *ResidualWeightManifest) uint64 {
	if m == nil {
		return 0
	}
	h := fnv.New64a()
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], m.DenseWeightFP)
	_, _ = h.Write(buf[:])
	return h.Sum64()
}
