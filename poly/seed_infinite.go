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

const (
	infiniteLayerFormat    = "loom-infinite-layer-v2"
	infiniteResidualFormat = "loom-infinite-residual-v1"

	// v1 format strings kept for backward-compat wrappers.
	infiniteDenseLayerFormat     = "loom-infinite-dense-layer-v1"
	infiniteSwiGLULayerFormat    = "loom-infinite-swiglu-layer-v1"
	infiniteMHALayerFormat       = "loom-infinite-mha-layer-v1"
	infiniteCNNLayerFormat       = "loom-infinite-cnn-layer-v1"
	infiniteRNNLayerFormat       = "loom-infinite-rnn-layer-v1"
	infiniteLSTMLayerFormat      = "loom-infinite-lstm-layer-v1"
	infiniteEmbeddingLayerFormat = "loom-infinite-embedding-layer-v1"
)

// DefaultDenseWeightChunk is the tile size for matrix weight overrides [out, in].
var DefaultDenseWeightChunk = []int{8, 8}

// DefaultFlatWeightChunk is the 1D tile size for flat weight overrides.
const DefaultFlatWeightChunk = 64

// LayerForwardProfile captures runtime flags that affect forward numerics beyond weights.
type LayerForwardProfile struct {
	TileSize              int     `json:"tile_size,omitempty"`
	UseTiling             bool    `json:"use_tiling,omitempty"`
	EnableMultiCoreTiling bool    `json:"enable_multi_core_tiling,omitempty"`
	WeightScale           float32 `json:"weight_scale,omitempty"`
	MaxSeqLen             int     `json:"max_seq_len,omitempty"`
	SeqLength             int     `json:"seq_length,omitempty"`
	RMSNormEps            float64 `json:"rms_norm_eps,omitempty"`
}

// WeightChunkOverride is one weight tile that differs from He-init(layer_seed).
type WeightChunkOverride struct {
	At      []int  `json:"at"`
	Shape   []int  `json:"shape"`
	Payload []byte `json:"payload"`
}

// InfiniteLayerManifest is one procedural layer as root seed + optional sparse weight diffs.
type InfiniteLayerManifest struct {
	Format         string                 `json:"format"`
	Kind           string                 `json:"kind"`
	DType          string                 `json:"dtype"`
	LayerSeed      uint64                 `json:"layer_seed"`
	WeightFP       uint64                 `json:"weight_fp"`
	ChunkSize      []int                  `json:"chunk_size,omitempty"`
	Overrides      []WeightChunkOverride  `json:"overrides,omitempty"`
	ForwardProfile *LayerForwardProfile   `json:"forward_profile,omitempty"`
	In             int                    `json:"in,omitempty"`
	Out            int                    `json:"out,omitempty"`
	Hidden         int                    `json:"hidden,omitempty"`
	Intermediate   int                    `json:"intermediate,omitempty"`
	MHA            *MHASpec               `json:"mha,omitempty"`
	CNN            *CNNSpec               `json:"cnn,omitempty"`
	Embedding      *EmbeddingSpec         `json:"embedding,omitempty"`
}

// InfiniteResidualManifest is dense branch seed+overrides plus weightless residual add.
type InfiniteResidualManifest struct {
	Format       string                `json:"format"`
	Spec         ResidualSpec          `json:"spec"`
	TopologySeed uint64                `json:"topology_seed"`
	DType        string                `json:"dtype"`
	Dense        InfiniteLayerManifest `json:"dense"`
	ForwardFP    uint64                `json:"forward_fp,omitempty"`
}

// OverrideCount returns sparse chunk count (0 = pure procedural layer seed).
func (m *InfiniteLayerManifest) OverrideCount() int {
	if m == nil {
		return 0
	}
	return len(m.Overrides)
}

// OverrideCount returns sparse chunk count on the dense branch.
func (m *InfiniteResidualManifest) OverrideCount() int {
	if m == nil {
		return 0
	}
	return m.Dense.OverrideCount()
}

// WeightStoreFingerprint returns FNV-1a hash of master float32 weights.
func WeightStoreFingerprint(ws *WeightStore) uint64 {
	return weightStoreFingerprint(ws)
}

// SyncWeightStoreForForward clears cached native views so forward uses Master weights.
func SyncWeightStoreForForward(l *VolumetricLayer) {
	syncWeightStoreForForward(l)
}

// BuildLayerFromSeed materializes one procedural layer from kind, seed, and optional shape config.
func BuildLayerFromSeed(kind string, layerSeed uint64, dtype DType, cfg ...*InfiniteLayerManifest) (*VolumetricLayer, error) {
	m := defaultInfiniteLayerManifest(kind, layerSeed, dtype)
	if len(cfg) > 0 && cfg[0] != nil {
		mergeInfiniteLayerConfig(m, cfg[0])
	}
	m.Kind = normalizeInfiniteKind(kind)
	m.LayerSeed = layerSeed
	if dtype != 0 {
		m.DType = dtype.String()
	} else if m.DType == "" {
		m.DType = DTypeFloat32.String()
	}
	return buildProceduralFromManifest(m)
}

// BuildLayerFromManifest is seed → layer (with sparse overrides applied).
func BuildLayerFromManifest(m *InfiniteLayerManifest) (*VolumetricLayer, error) {
	if m == nil {
		return nil, fmt.Errorf("infinite layer: nil manifest")
	}
	l, err := buildProceduralFromManifest(m)
	if err != nil {
		return nil, err
	}
	if err := applyInfiniteOverrides(m, l.WeightStore); err != nil {
		return nil, err
	}
	dtype := ParseDType(m.DType)
	if dtype == 0 {
		dtype = DTypeFloat32
	}
	if m.WeightFP != 0 && weightStoreFingerprint(l.WeightStore) != m.WeightFP {
		return nil, fmt.Errorf("infinite layer %s: weight fp mismatch", m.Kind)
	}
	if dtype != DTypeFloat32 {
		l.WeightStore.Morph(dtype)
	}
	applyLayerForwardProfile(l, m.ForwardProfile)
	syncWeightStoreForForward(l)
	return l, nil
}

// ManifestFromLayer extracts seeds from a layer built via He-init(layer_seed).
func ManifestFromLayer(l *VolumetricLayer, layerSeed uint64) (*InfiniteLayerManifest, error) {
	if l == nil || l.WeightStore == nil {
		return nil, fmt.Errorf("infinite layer: nil layer")
	}
	kind, err := infiniteKindFromLayer(l)
	if err != nil {
		return nil, err
	}
	dtype := l.DType
	if dtype == 0 {
		dtype = DTypeFloat32
	}
	cfg := infiniteLayerConfigFromVolumetric(l, kind)
	m, err := encodeInfiniteLayer(l.WeightStore, kind, layerSeed, dtype, cfg)
	if err != nil {
		return nil, err
	}
	if kind == "dense" {
		ok, err := denseLayerMatchesSeed(l, layerSeed, dtype.String())
		if err != nil {
			return nil, err
		}
		if !ok && len(m.Overrides) == 0 {
			return nil, fmt.Errorf("dense layer: weights do not match seed 0x%x", layerSeed)
		}
	}
	m.ForwardProfile = captureLayerForwardProfile(l)
	return m, nil
}

// MarshalInfiniteLayer JSON-encodes a layer manifest.
func MarshalInfiniteLayer(m *InfiniteLayerManifest) ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

// ParseInfiniteLayer decodes JSON.
func ParseInfiniteLayer(data []byte) (*InfiniteLayerManifest, error) {
	var m InfiniteLayerManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// ManifestFromResidualBlock extracts dense infinite manifest from a residual network.
func ManifestFromResidualBlock(net *VolumetricNetwork, topologySeed uint64, spec ResidualSpec) (*InfiniteResidualManifest, error) {
	if net == nil || len(net.Layers) < 2 {
		return nil, fmt.Errorf("residual: need dense+residual layers")
	}
	dense := net.GetLayer(0, 0, 0, 0)
	if dense.Type != LayerDense {
		return nil, fmt.Errorf("residual: layer 0 not dense")
	}
	dtype := dense.DType
	if dtype == 0 {
		dtype = DTypeFloat32
	}
	denseSeed := ResidualDenseSeed(topologySeed)
	dm, err := ManifestFromLayer(dense, denseSeed)
	if err != nil {
		return nil, err
	}
	return &InfiniteResidualManifest{
		Format: infiniteResidualFormat, Spec: spec, TopologySeed: topologySeed,
		DType: dtype.String(), Dense: *dm,
	}, nil
}

// BuildResidualFromManifest builds Dense + Residual layers from manifest.
func BuildResidualFromManifest(m *InfiniteResidualManifest) (*VolumetricNetwork, error) {
	if m == nil {
		return nil, fmt.Errorf("residual: nil manifest")
	}
	if m.Spec.In != m.Spec.Out {
		return nil, fmt.Errorf("residual: in=%d out=%d", m.Spec.In, m.Spec.Out)
	}
	dense, err := BuildLayerFromManifest(&m.Dense)
	if err != nil {
		return nil, err
	}
	net := NewVolumetricNetwork(1, 1, 1, 2)
	net.InitSeed = m.TopologySeed
	*net.GetLayer(0, 0, 0, 0) = *dense
	res := net.GetLayer(0, 0, 0, 1)
	res.Type = LayerResidual
	res.InputHeight = m.Spec.Out
	res.OutputHeight = m.Spec.Out
	res.DType = dense.DType
	return net, nil
}

// --- backward-compat: dense v1 ---

// InfiniteDenseLayerManifest is the v1 dense-only manifest (converts to InfiniteLayerManifest).
type InfiniteDenseLayerManifest struct {
	Format         string                 `json:"format"`
	In             int                    `json:"in"`
	Out            int                    `json:"out"`
	DType          string                 `json:"dtype"`
	LayerSeed      uint64                 `json:"layer_seed"`
	WeightFP       uint64                 `json:"weight_fp"`
	ChunkSize      []int                  `json:"chunk_size,omitempty"`
	Overrides      []WeightChunkOverride  `json:"overrides,omitempty"`
	ForwardProfile *LayerForwardProfile   `json:"forward_profile,omitempty"`
}

func (m *InfiniteDenseLayerManifest) OverrideCount() int { return overrideCount(m.Overrides) }

func (m *InfiniteDenseLayerManifest) toUnified() *InfiniteLayerManifest {
	if m == nil {
		return nil
	}
	u := &InfiniteLayerManifest{
		Format: infiniteLayerFormat, Kind: "dense",
		In: m.In, Out: m.Out, DType: m.DType, LayerSeed: m.LayerSeed, WeightFP: m.WeightFP,
		ChunkSize: append([]int(nil), m.ChunkSize...), Overrides: append([]WeightChunkOverride(nil), m.Overrides...),
		ForwardProfile: m.ForwardProfile,
	}
	if len(u.ChunkSize) == 0 {
		u.ChunkSize = append([]int(nil), DefaultDenseWeightChunk...)
	}
	return u
}

func infiniteDenseLayerManifestFromUnified(u *InfiniteLayerManifest) *InfiniteDenseLayerManifest {
	if u == nil {
		return nil
	}
	return &InfiniteDenseLayerManifest{
		Format: infiniteDenseLayerFormat, In: u.In, Out: u.Out,
		DType: u.DType, LayerSeed: u.LayerSeed, WeightFP: u.WeightFP,
		ChunkSize: append([]int(nil), u.ChunkSize...), Overrides: append([]WeightChunkOverride(nil), u.Overrides...),
		ForwardProfile: u.ForwardProfile,
	}
}

func BuildDenseLayerFromSeed(layerSeed uint64, in, out int, dtype DType) (*VolumetricLayer, error) {
	cfg := &InfiniteLayerManifest{In: in, Out: out}
	return BuildLayerFromSeed("dense", layerSeed, dtype, cfg)
}

func EncodeInfiniteDenseLayer(ws *WeightStore, in, out int, dtype DType, layerSeed uint64) (*InfiniteDenseLayerManifest, error) {
	cfg := &InfiniteLayerManifest{In: in, Out: out}
	m, err := encodeInfiniteLayer(ws, "dense", layerSeed, dtype, cfg)
	if err != nil {
		return nil, err
	}
	return infiniteDenseLayerManifestFromUnified(m), nil
}

func BuildDenseLayerFromInfiniteManifest(m *InfiniteDenseLayerManifest) (*VolumetricLayer, error) {
	return BuildLayerFromManifest(m.toUnified())
}

func ManifestFromDenseLayer(l *VolumetricLayer, layerSeed uint64) (*InfiniteDenseLayerManifest, error) {
	u, err := ManifestFromLayer(l, layerSeed)
	if err != nil {
		return nil, err
	}
	return infiniteDenseLayerManifestFromUnified(u), nil
}

func MarshalInfiniteDenseLayer(m *InfiniteDenseLayerManifest) ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

func ParseInfiniteDenseLayer(data []byte) (*InfiniteDenseLayerManifest, error) {
	var m InfiniteDenseLayerManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// --- backward-compat: SwiGLU v1 ---

type InfiniteSwiGLULayerManifest struct {
	Format         string                `json:"format"`
	Hidden         int                   `json:"hidden"`
	Intermediate   int                   `json:"intermediate"`
	DType          string                `json:"dtype"`
	LayerSeed      uint64                `json:"layer_seed"`
	WeightFP       uint64                `json:"weight_fp"`
	ChunkSize      int                   `json:"chunk_size,omitempty"`
	Overrides      []WeightChunkOverride `json:"overrides,omitempty"`
	ForwardProfile *LayerForwardProfile  `json:"forward_profile,omitempty"`
}

func (m *InfiniteSwiGLULayerManifest) OverrideCount() int { return overrideCount(m.Overrides) }

func (m *InfiniteSwiGLULayerManifest) toUnified() *InfiniteLayerManifest {
	u := &InfiniteLayerManifest{
		Format: infiniteLayerFormat, Kind: "swiglu",
		Hidden: m.Hidden, Intermediate: m.Intermediate,
		DType: m.DType, LayerSeed: m.LayerSeed, WeightFP: m.WeightFP,
		Overrides: append([]WeightChunkOverride(nil), m.Overrides...), ForwardProfile: m.ForwardProfile,
	}
	cs := m.ChunkSize
	if cs <= 0 {
		cs = DefaultFlatWeightChunk
	}
	u.ChunkSize = []int{cs}
	return u
}

func infiniteSwiGLUManifestFromUnified(u *InfiniteLayerManifest) *InfiniteSwiGLULayerManifest {
	cs := DefaultFlatWeightChunk
	if len(u.ChunkSize) > 0 {
		cs = u.ChunkSize[0]
	}
	return &InfiniteSwiGLULayerManifest{
		Format: infiniteSwiGLULayerFormat, Hidden: u.Hidden, Intermediate: u.Intermediate,
		DType: u.DType, LayerSeed: u.LayerSeed, WeightFP: u.WeightFP,
		ChunkSize: cs, Overrides: append([]WeightChunkOverride(nil), u.Overrides...),
		ForwardProfile: u.ForwardProfile,
	}
}

func BuildSwiGLULayerFromSeed(layerSeed uint64, hidden, intermediate int, dtype DType) (*VolumetricLayer, error) {
	return BuildLayerFromSeed("swiglu", layerSeed, dtype, &InfiniteLayerManifest{Hidden: hidden, Intermediate: intermediate})
}

func BuildSwiGLULayerFromInfiniteManifest(m *InfiniteSwiGLULayerManifest) (*VolumetricLayer, error) {
	return BuildLayerFromManifest(m.toUnified())
}

func ManifestFromSwiGLULayer(l *VolumetricLayer, layerSeed uint64) (*InfiniteSwiGLULayerManifest, error) {
	u, err := ManifestFromLayer(l, layerSeed)
	if err != nil {
		return nil, err
	}
	return infiniteSwiGLUManifestFromUnified(u), nil
}

func MarshalInfiniteSwiGLULayer(m *InfiniteSwiGLULayerManifest) ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

func ParseInfiniteSwiGLULayer(data []byte) (*InfiniteSwiGLULayerManifest, error) {
	var m InfiniteSwiGLULayerManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// --- backward-compat: MHA v1 ---

type InfiniteMHALayerManifest struct {
	Format         string                `json:"format"`
	Spec           MHASpec               `json:"spec"`
	DType          string                `json:"dtype"`
	LayerSeed      uint64                `json:"layer_seed"`
	WeightFP       uint64                `json:"weight_fp"`
	ChunkSize      int                   `json:"chunk_size,omitempty"`
	Overrides      []WeightChunkOverride `json:"overrides,omitempty"`
	ForwardProfile *LayerForwardProfile  `json:"forward_profile,omitempty"`
}

func (m *InfiniteMHALayerManifest) OverrideCount() int { return overrideCount(m.Overrides) }

func (m *InfiniteMHALayerManifest) toUnified() *InfiniteLayerManifest {
	spec := m.Spec
	u := &InfiniteLayerManifest{
		Format: infiniteLayerFormat, Kind: "mha", MHA: &spec,
		DType: m.DType, LayerSeed: m.LayerSeed, WeightFP: m.WeightFP,
		Overrides: append([]WeightChunkOverride(nil), m.Overrides...), ForwardProfile: m.ForwardProfile,
	}
	cs := m.ChunkSize
	if cs <= 0 {
		cs = DefaultFlatWeightChunk
	}
	u.ChunkSize = []int{cs}
	return u
}

func infiniteMHAManifestFromUnified(u *InfiniteLayerManifest) *InfiniteMHALayerManifest {
	spec := MHASpec{}
	if u.MHA != nil {
		spec = *u.MHA
	}
	cs := DefaultFlatWeightChunk
	if len(u.ChunkSize) > 0 {
		cs = u.ChunkSize[0]
	}
	return &InfiniteMHALayerManifest{
		Format: infiniteMHALayerFormat, Spec: spec,
		DType: u.DType, LayerSeed: u.LayerSeed, WeightFP: u.WeightFP,
		ChunkSize: cs, Overrides: append([]WeightChunkOverride(nil), u.Overrides...),
		ForwardProfile: u.ForwardProfile,
	}
}

func BuildMHALayerFromSeed(layerSeed uint64, spec MHASpec, dtype DType) (*VolumetricLayer, error) {
	s := spec
	return BuildLayerFromSeed("mha", layerSeed, dtype, &InfiniteLayerManifest{MHA: &s})
}

func BuildMHALayerFromInfiniteManifest(m *InfiniteMHALayerManifest) (*VolumetricLayer, error) {
	return BuildLayerFromManifest(m.toUnified())
}

func ManifestFromMHALayer(l *VolumetricLayer, layerSeed uint64) (*InfiniteMHALayerManifest, error) {
	u, err := ManifestFromLayer(l, layerSeed)
	if err != nil {
		return nil, err
	}
	return infiniteMHAManifestFromUnified(u), nil
}

func MarshalInfiniteMHALayer(m *InfiniteMHALayerManifest) ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

func ParseInfiniteMHALayer(data []byte) (*InfiniteMHALayerManifest, error) {
	var m InfiniteMHALayerManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// --- backward-compat: CNN v1 ---

type InfiniteCNNLayerManifest struct {
	Format         string                `json:"format"`
	Spec           CNNSpec               `json:"spec"`
	DType          string                `json:"dtype"`
	LayerSeed      uint64                `json:"layer_seed"`
	WeightFP       uint64                `json:"weight_fp"`
	ChunkSize      int                   `json:"chunk_size,omitempty"`
	Overrides      []WeightChunkOverride `json:"overrides,omitempty"`
	ForwardProfile *LayerForwardProfile  `json:"forward_profile,omitempty"`
}

func (m *InfiniteCNNLayerManifest) OverrideCount() int { return overrideCount(m.Overrides) }

func (m *InfiniteCNNLayerManifest) toUnified() *InfiniteLayerManifest {
	spec := m.Spec
	kind := fmt.Sprintf("cnn%d", spec.Dim)
	if spec.Dim <= 0 {
		kind = "cnn1"
	}
	u := &InfiniteLayerManifest{
		Format: infiniteLayerFormat, Kind: kind, CNN: &spec,
		DType: m.DType, LayerSeed: m.LayerSeed, WeightFP: m.WeightFP,
		Overrides: append([]WeightChunkOverride(nil), m.Overrides...), ForwardProfile: m.ForwardProfile,
	}
	cs := m.ChunkSize
	if cs <= 0 {
		cs = DefaultFlatWeightChunk
	}
	u.ChunkSize = []int{cs}
	return u
}

func infiniteCNNManifestFromUnified(u *InfiniteLayerManifest) *InfiniteCNNLayerManifest {
	spec := CNNSpec{}
	if u.CNN != nil {
		spec = *u.CNN
	}
	cs := DefaultFlatWeightChunk
	if len(u.ChunkSize) > 0 {
		cs = u.ChunkSize[0]
	}
	return &InfiniteCNNLayerManifest{
		Format: infiniteCNNLayerFormat, Spec: spec,
		DType: u.DType, LayerSeed: u.LayerSeed, WeightFP: u.WeightFP,
		ChunkSize: cs, Overrides: append([]WeightChunkOverride(nil), u.Overrides...),
		ForwardProfile: u.ForwardProfile,
	}
}

func BuildCNNLayerFromSeed(layerSeed uint64, spec CNNSpec, dtype DType) (*VolumetricLayer, error) {
	s := spec
	kind := fmt.Sprintf("cnn%d", s.Dim)
	if s.Dim <= 0 {
		kind = "cnn1"
	}
	return BuildLayerFromSeed(kind, layerSeed, dtype, &InfiniteLayerManifest{CNN: &s})
}

func BuildCNNLayerFromInfiniteManifest(m *InfiniteCNNLayerManifest) (*VolumetricLayer, error) {
	return BuildLayerFromManifest(m.toUnified())
}

func ManifestFromCNNLayer(l *VolumetricLayer, layerSeed uint64) (*InfiniteCNNLayerManifest, error) {
	u, err := ManifestFromLayer(l, layerSeed)
	if err != nil {
		return nil, err
	}
	return infiniteCNNManifestFromUnified(u), nil
}

func MarshalInfiniteCNNLayer(m *InfiniteCNNLayerManifest) ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

func ParseInfiniteCNNLayer(data []byte) (*InfiniteCNNLayerManifest, error) {
	var m InfiniteCNNLayerManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// --- backward-compat: RNN v1 ---

type InfiniteRNNLayerManifest struct {
	Format         string                `json:"format"`
	In             int                   `json:"in"`
	Out            int                   `json:"out"`
	DType          string                `json:"dtype"`
	LayerSeed      uint64                `json:"layer_seed"`
	WeightFP       uint64                `json:"weight_fp"`
	ChunkSize      int                   `json:"chunk_size,omitempty"`
	Overrides      []WeightChunkOverride `json:"overrides,omitempty"`
	ForwardProfile *LayerForwardProfile  `json:"forward_profile,omitempty"`
}

func (m *InfiniteRNNLayerManifest) OverrideCount() int { return overrideCount(m.Overrides) }

func (m *InfiniteRNNLayerManifest) toUnified() *InfiniteLayerManifest {
	u := &InfiniteLayerManifest{
		Format: infiniteLayerFormat, Kind: "rnn",
		In: m.In, Out: m.Out, DType: m.DType, LayerSeed: m.LayerSeed, WeightFP: m.WeightFP,
		Overrides: append([]WeightChunkOverride(nil), m.Overrides...), ForwardProfile: m.ForwardProfile,
	}
	cs := m.ChunkSize
	if cs <= 0 {
		cs = DefaultFlatWeightChunk
	}
	u.ChunkSize = []int{cs}
	return u
}

func infiniteRNNManifestFromUnified(u *InfiniteLayerManifest) *InfiniteRNNLayerManifest {
	cs := DefaultFlatWeightChunk
	if len(u.ChunkSize) > 0 {
		cs = u.ChunkSize[0]
	}
	return &InfiniteRNNLayerManifest{
		Format: infiniteRNNLayerFormat, In: u.In, Out: u.Out,
		DType: u.DType, LayerSeed: u.LayerSeed, WeightFP: u.WeightFP,
		ChunkSize: cs, Overrides: append([]WeightChunkOverride(nil), u.Overrides...),
		ForwardProfile: u.ForwardProfile,
	}
}

func BuildRNNLayerFromSeed(layerSeed uint64, in, out int, dtype DType) (*VolumetricLayer, error) {
	return BuildLayerFromSeed("rnn", layerSeed, dtype, &InfiniteLayerManifest{In: in, Out: out})
}

func BuildRNNLayerFromInfiniteManifest(m *InfiniteRNNLayerManifest) (*VolumetricLayer, error) {
	return BuildLayerFromManifest(m.toUnified())
}

func ManifestFromRNNLayer(l *VolumetricLayer, layerSeed uint64) (*InfiniteRNNLayerManifest, error) {
	u, err := ManifestFromLayer(l, layerSeed)
	if err != nil {
		return nil, err
	}
	return infiniteRNNManifestFromUnified(u), nil
}

func MarshalInfiniteRNNLayer(m *InfiniteRNNLayerManifest) ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

func ParseInfiniteRNNLayer(data []byte) (*InfiniteRNNLayerManifest, error) {
	var m InfiniteRNNLayerManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// --- backward-compat: LSTM v1 ---

type InfiniteLSTMLayerManifest struct {
	Format         string                `json:"format"`
	In             int                   `json:"in"`
	Out            int                   `json:"out"`
	DType          string                `json:"dtype"`
	LayerSeed      uint64                `json:"layer_seed"`
	WeightFP       uint64                `json:"weight_fp"`
	ChunkSize      int                   `json:"chunk_size,omitempty"`
	Overrides      []WeightChunkOverride `json:"overrides,omitempty"`
	ForwardProfile *LayerForwardProfile  `json:"forward_profile,omitempty"`
}

func (m *InfiniteLSTMLayerManifest) OverrideCount() int { return overrideCount(m.Overrides) }

func (m *InfiniteLSTMLayerManifest) toUnified() *InfiniteLayerManifest {
	u := &InfiniteLayerManifest{
		Format: infiniteLayerFormat, Kind: "lstm",
		In: m.In, Out: m.Out, DType: m.DType, LayerSeed: m.LayerSeed, WeightFP: m.WeightFP,
		Overrides: append([]WeightChunkOverride(nil), m.Overrides...), ForwardProfile: m.ForwardProfile,
	}
	cs := m.ChunkSize
	if cs <= 0 {
		cs = DefaultFlatWeightChunk
	}
	u.ChunkSize = []int{cs}
	return u
}

func infiniteLSTMManifestFromUnified(u *InfiniteLayerManifest) *InfiniteLSTMLayerManifest {
	cs := DefaultFlatWeightChunk
	if len(u.ChunkSize) > 0 {
		cs = u.ChunkSize[0]
	}
	return &InfiniteLSTMLayerManifest{
		Format: infiniteLSTMLayerFormat, In: u.In, Out: u.Out,
		DType: u.DType, LayerSeed: u.LayerSeed, WeightFP: u.WeightFP,
		ChunkSize: cs, Overrides: append([]WeightChunkOverride(nil), u.Overrides...),
		ForwardProfile: u.ForwardProfile,
	}
}

func BuildLSTMLayerFromSeed(layerSeed uint64, in, out int, dtype DType) (*VolumetricLayer, error) {
	return BuildLayerFromSeed("lstm", layerSeed, dtype, &InfiniteLayerManifest{In: in, Out: out})
}

func BuildLSTMLayerFromInfiniteManifest(m *InfiniteLSTMLayerManifest) (*VolumetricLayer, error) {
	return BuildLayerFromManifest(m.toUnified())
}

func ManifestFromLSTMLayer(l *VolumetricLayer, layerSeed uint64) (*InfiniteLSTMLayerManifest, error) {
	u, err := ManifestFromLayer(l, layerSeed)
	if err != nil {
		return nil, err
	}
	return infiniteLSTMManifestFromUnified(u), nil
}

func MarshalInfiniteLSTMLayer(m *InfiniteLSTMLayerManifest) ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

func ParseInfiniteLSTMLayer(data []byte) (*InfiniteLSTMLayerManifest, error) {
	var m InfiniteLSTMLayerManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// --- backward-compat: Embedding v1 ---

type InfiniteEmbeddingLayerManifest struct {
	Format         string                `json:"format"`
	VocabSize      int                   `json:"vocab_size"`
	EmbeddingDim   int                   `json:"embedding_dim"`
	SeqLen         int                   `json:"seq_len"`
	DType          string                `json:"dtype"`
	LayerSeed      uint64                `json:"layer_seed"`
	WeightFP       uint64                `json:"weight_fp"`
	ChunkSize      []int                 `json:"chunk_size,omitempty"`
	Overrides      []WeightChunkOverride `json:"overrides,omitempty"`
	ForwardProfile *LayerForwardProfile  `json:"forward_profile,omitempty"`
}

func (m *InfiniteEmbeddingLayerManifest) OverrideCount() int { return overrideCount(m.Overrides) }

func (m *InfiniteEmbeddingLayerManifest) toUnified() *InfiniteLayerManifest {
	spec := EmbeddingSpec{VocabSize: m.VocabSize, EmbeddingDim: m.EmbeddingDim, SeqLen: m.SeqLen}
	u := &InfiniteLayerManifest{
		Format: infiniteLayerFormat, Kind: "embedding", Embedding: &spec,
		DType: m.DType, LayerSeed: m.LayerSeed, WeightFP: m.WeightFP,
		ChunkSize: append([]int(nil), m.ChunkSize...), Overrides: append([]WeightChunkOverride(nil), m.Overrides...),
		ForwardProfile: m.ForwardProfile,
	}
	if len(u.ChunkSize) == 0 {
		u.ChunkSize = append([]int(nil), DefaultDenseWeightChunk...)
	}
	return u
}

func infiniteEmbeddingManifestFromUnified(u *InfiniteLayerManifest) *InfiniteEmbeddingLayerManifest {
	spec := EmbeddingSpec{SeqLen: 8}
	if u.Embedding != nil {
		spec = *u.Embedding
	}
	return &InfiniteEmbeddingLayerManifest{
		Format: infiniteEmbeddingLayerFormat,
		VocabSize: spec.VocabSize, EmbeddingDim: spec.EmbeddingDim, SeqLen: spec.SeqLen,
		DType: u.DType, LayerSeed: u.LayerSeed, WeightFP: u.WeightFP,
		ChunkSize: append([]int(nil), u.ChunkSize...), Overrides: append([]WeightChunkOverride(nil), u.Overrides...),
		ForwardProfile: u.ForwardProfile,
	}
}

func BuildEmbeddingLayerFromSeed(layerSeed uint64, spec EmbeddingSpec, dtype DType) (*VolumetricLayer, error) {
	s := spec
	return BuildLayerFromSeed("embedding", layerSeed, dtype, &InfiniteLayerManifest{Embedding: &s})
}

func BuildEmbeddingLayerFromInfiniteManifest(m *InfiniteEmbeddingLayerManifest) (*VolumetricLayer, error) {
	return BuildLayerFromManifest(m.toUnified())
}

func ManifestFromEmbeddingLayer(l *VolumetricLayer, layerSeed uint64) (*InfiniteEmbeddingLayerManifest, error) {
	u, err := ManifestFromLayer(l, layerSeed)
	if err != nil {
		return nil, err
	}
	return infiniteEmbeddingManifestFromUnified(u), nil
}

func MarshalInfiniteEmbeddingLayer(m *InfiniteEmbeddingLayerManifest) ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

func ParseInfiniteEmbeddingLayer(data []byte) (*InfiniteEmbeddingLayerManifest, error) {
	var m InfiniteEmbeddingLayerManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// --- backward-compat: residual ---

func BuildResidualFromInfiniteManifest(m *InfiniteResidualManifest) (*VolumetricNetwork, error) {
	return BuildResidualFromManifest(m)
}

func EncodeInfiniteResidual(net *VolumetricNetwork, topologySeed uint64, spec ResidualSpec) (*InfiniteResidualManifest, error) {
	return ManifestFromResidualBlock(net, topologySeed, spec)
}

func MarshalInfiniteResidual(m *InfiniteResidualManifest) ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

func ParseInfiniteResidual(data []byte) (*InfiniteResidualManifest, error) {
	var m InfiniteResidualManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// --- internals ---

func normalizeInfiniteKind(kind string) string {
	switch strings.ToLower(kind) {
	case "dense", "swiglu", "mha", "cnn1", "cnn2", "cnn3", "rnn", "lstm", "embedding":
		return strings.ToLower(kind)
	default:
		return kind
	}
}

func infiniteKindFromLayer(l *VolumetricLayer) (string, error) {
	switch l.Type {
	case LayerDense:
		return "dense", nil
	case LayerSwiGLU:
		return "swiglu", nil
	case LayerMultiHeadAttention:
		return "mha", nil
	case LayerCNN1:
		return "cnn1", nil
	case LayerCNN2:
		return "cnn2", nil
	case LayerCNN3:
		return "cnn3", nil
	case LayerRNN:
		return "rnn", nil
	case LayerLSTM:
		return "lstm", nil
	case LayerEmbedding:
		return "embedding", nil
	default:
		return "", fmt.Errorf("infinite layer: unsupported type %v", l.Type)
	}
}

func defaultChunkSizeForKind(kind string) []int {
	switch normalizeInfiniteKind(kind) {
	case "dense", "embedding":
		return append([]int(nil), DefaultDenseWeightChunk...)
	default:
		return []int{DefaultFlatWeightChunk}
	}
}

func isMatrixChunkKind(kind string) bool {
	switch normalizeInfiniteKind(kind) {
	case "dense", "embedding":
		return true
	default:
		return false
	}
}

func defaultInfiniteLayerManifest(kind string, layerSeed uint64, dtype DType) *InfiniteLayerManifest {
	kind = normalizeInfiniteKind(kind)
	dt := dtype
	if dt == 0 {
		dt = DTypeFloat32
	}
	m := &InfiniteLayerManifest{
		Kind: kind, LayerSeed: layerSeed, DType: dt.String(),
		ChunkSize: defaultChunkSizeForKind(kind),
	}
	switch kind {
	case "dense":
		m.In, m.Out = 4, 8
	case "swiglu":
		m.Hidden, m.Intermediate = 8, 16
	case "mha":
		spec := MHASpec{DModel: 8, NumHeads: 2, NumKVHeads: 2, HeadDim: 4, QueryDim: 8}
		m.MHA = &spec
	case "cnn1":
		spec := CNNSpec{Dim: 1, InputChannels: 2, Filters: 4, Spatial: 8, KernelSize: 3}
		m.CNN = &spec
	case "cnn2":
		spec := CNNSpec{Dim: 2, InputChannels: 2, Filters: 4, Spatial: 8, KernelSize: 3}
		m.CNN = &spec
	case "cnn3":
		spec := CNNSpec{Dim: 3, InputChannels: 2, Filters: 4, Spatial: 6, KernelSize: 3}
		m.CNN = &spec
	case "rnn", "lstm":
		m.In, m.Out = 4, 8
	case "embedding":
		spec := EmbeddingSpec{VocabSize: 32, EmbeddingDim: 8, SeqLen: 8}
		m.Embedding = &spec
	}
	return m
}

func mergeInfiniteLayerConfig(dst, src *InfiniteLayerManifest) {
	if src.In > 0 {
		dst.In = src.In
	}
	if src.Out > 0 {
		dst.Out = src.Out
	}
	if src.Hidden > 0 {
		dst.Hidden = src.Hidden
	}
	if src.Intermediate > 0 {
		dst.Intermediate = src.Intermediate
	}
	if src.MHA != nil {
		spec := *src.MHA
		dst.MHA = &spec
	}
	if src.CNN != nil {
		spec := *src.CNN
		dst.CNN = &spec
	}
	if src.Embedding != nil {
		spec := *src.Embedding
		dst.Embedding = &spec
	}
	if len(src.ChunkSize) > 0 {
		dst.ChunkSize = append([]int(nil), src.ChunkSize...)
	}
	if src.DType != "" {
		dst.DType = src.DType
	}
}

func infiniteLayerConfigFromVolumetric(l *VolumetricLayer, kind string) *InfiniteLayerManifest {
	m := &InfiniteLayerManifest{Kind: kind}
	switch kind {
	case "dense":
		m.In, m.Out = l.InputHeight, l.OutputHeight
	case "swiglu":
		m.Hidden, m.Intermediate = l.InputHeight, l.OutputHeight
	case "mha":
		spec := MHASpec{
			DModel: l.DModel, NumHeads: l.NumHeads, NumKVHeads: l.NumKVHeads,
			HeadDim: l.HeadDim, QueryDim: l.QueryDim, RoPEFreqBase: l.RoPEFreqBase,
		}
		m.MHA = &spec
	case "cnn1", "cnn2", "cnn3":
		spec := CNNSpec{
			Dim: cnnDimFromLayerType(l.Type), InputChannels: l.InputChannels, Filters: l.Filters,
			Spatial: cnnSpatialFromLayer(l), KernelSize: l.KernelSize, Stride: l.Stride, Padding: l.Padding,
		}
		m.CNN = &spec
	case "rnn", "lstm":
		m.In, m.Out = l.InputHeight, l.OutputHeight
	case "embedding":
		seq := l.SeqLength
		if seq <= 0 {
			seq = 8
		}
		spec := EmbeddingSpec{VocabSize: l.VocabSize, EmbeddingDim: l.EmbeddingDim, SeqLen: seq}
		m.Embedding = &spec
	}
	return m
}

func buildProceduralFromManifest(m *InfiniteLayerManifest) (*VolumetricLayer, error) {
	if m == nil {
		return nil, fmt.Errorf("infinite layer: nil manifest")
	}
	kind := normalizeInfiniteKind(m.Kind)
	dtype := ParseDType(m.DType)
	if dtype == 0 {
		dtype = DTypeFloat32
	}
	switch kind {
	case "dense":
		if m.In <= 0 || m.Out <= 0 {
			return nil, fmt.Errorf("dense layer: invalid shape %dx%d", m.In, m.Out)
		}
		l := &VolumetricLayer{
			Type: LayerDense, Activation: ActivationReLU,
			InputHeight: m.In, OutputHeight: m.Out, DType: dtype,
			WeightStore: NewWeightStore(m.In * m.Out),
		}
		InitWeightStoreHeSeeded(l.WeightStore, m.In, m.LayerSeed)
		if dtype != DTypeFloat32 {
			l.WeightStore.Morph(dtype)
		}
		return l, nil
	case "swiglu":
		if m.Hidden <= 0 || m.Intermediate <= 0 {
			return nil, fmt.Errorf("swiglu: invalid shape %dx%d", m.Hidden, m.Intermediate)
		}
		l := &VolumetricLayer{
			Type: LayerSwiGLU, Activation: ActivationSilu,
			InputHeight: m.Hidden, OutputHeight: m.Intermediate, DType: dtype,
			WeightStore: NewWeightStore(3*m.Hidden*m.Intermediate + 2*m.Intermediate + m.Hidden),
		}
		InitWeightStoreHeSeeded(l.WeightStore, m.Hidden, m.LayerSeed)
		if dtype != DTypeFloat32 {
			l.WeightStore.Morph(dtype)
		}
		return l, nil
	case "mha":
		if m.MHA == nil {
			return nil, fmt.Errorf("mha: nil spec")
		}
		n, err := normalizeMHASpec(*m.MHA)
		if err != nil {
			return nil, err
		}
		l := &VolumetricLayer{DType: dtype, WeightStore: NewWeightStore(mhaWeightCount(n))}
		applyMHASpec(l, n)
		InitWeightStoreHeSeeded(l.WeightStore, n.DModel, m.LayerSeed)
		if dtype != DTypeFloat32 {
			l.WeightStore.Morph(dtype)
		}
		return l, nil
	case "cnn1", "cnn2", "cnn3":
		if m.CNN == nil {
			return nil, fmt.Errorf("cnn: nil spec")
		}
		n, err := normalizeCNNSpec(*m.CNN)
		if err != nil {
			return nil, err
		}
		l := &VolumetricLayer{DType: dtype, WeightStore: NewWeightStore(cnnWeightCount(n))}
		applyCNNSpec(l, n)
		InitWeightStoreHeSeeded(l.WeightStore, n.InputChannels, m.LayerSeed)
		if dtype != DTypeFloat32 {
			l.WeightStore.Morph(dtype)
		}
		return l, nil
	case "rnn":
		if m.In <= 0 || m.Out <= 0 {
			return nil, fmt.Errorf("rnn: invalid shape %dx%d", m.In, m.Out)
		}
		l := &VolumetricLayer{
			Type: LayerRNN, Activation: ActivationTanh,
			InputHeight: m.In, OutputHeight: m.Out, SeqLength: 1, DType: dtype,
			WeightStore: NewWeightStore(rnnWeightCount(m.In, m.Out)),
		}
		InitWeightStoreHeSeeded(l.WeightStore, m.In, m.LayerSeed)
		if dtype != DTypeFloat32 {
			l.WeightStore.Morph(dtype)
		}
		return l, nil
	case "lstm":
		if m.In <= 0 || m.Out <= 0 {
			return nil, fmt.Errorf("lstm: invalid shape %dx%d", m.In, m.Out)
		}
		l := &VolumetricLayer{
			Type: LayerLSTM, Activation: ActivationTanh,
			InputHeight: m.In, OutputHeight: m.Out, SeqLength: 1, DType: dtype,
			WeightStore: NewWeightStore(lstmWeightCount(m.In, m.Out)),
		}
		InitWeightStoreHeSeeded(l.WeightStore, m.In, m.LayerSeed)
		if dtype != DTypeFloat32 {
			l.WeightStore.Morph(dtype)
		}
		return l, nil
	case "embedding":
		if m.Embedding == nil {
			return nil, fmt.Errorf("embedding: nil spec")
		}
		n, err := normalizeEmbeddingSpec(*m.Embedding)
		if err != nil {
			return nil, err
		}
		l := &VolumetricLayer{DType: dtype, WeightStore: NewWeightStore(n.VocabSize * n.EmbeddingDim)}
		applyEmbeddingSpec(l, n)
		InitWeightStoreHeSeeded(l.WeightStore, n.EmbeddingDim, m.LayerSeed)
		if dtype != DTypeFloat32 {
			l.WeightStore.Morph(dtype)
		}
		return l, nil
	default:
		return nil, fmt.Errorf("infinite layer: unknown kind %q", kind)
	}
}

func encodeInfiniteLayer(ws *WeightStore, kind string, layerSeed uint64, dtype DType, cfg *InfiniteLayerManifest) (*InfiniteLayerManifest, error) {
	if ws == nil {
		return nil, fmt.Errorf("infinite layer %s: nil weight store", kind)
	}
	if dtype == 0 {
		dtype = DTypeFloat32
	}
	kind = normalizeInfiniteKind(kind)
	base := defaultInfiniteLayerManifest(kind, layerSeed, dtype)
	if cfg != nil {
		mergeInfiniteLayerConfig(base, cfg)
	}
	base.LayerSeed = layerSeed
	want, err := buildProceduralFromManifest(base)
	if err != nil {
		return nil, err
	}
	m := &InfiniteLayerManifest{
		Format: infiniteLayerFormat, Kind: kind, DType: dtype.String(),
		LayerSeed: layerSeed, WeightFP: weightStoreFingerprint(ws),
		ChunkSize: append([]int(nil), base.ChunkSize...),
	}
	mergeInfiniteLayerConfig(m, base)
	if weightStoreFingerprint(ws) == weightStoreFingerprint(want.WeightStore) {
		return m, nil
	}
	if isMatrixChunkKind(kind) {
		overrides, err := encodeInfiniteMatrixWeights(ws, want.WeightStore, kind, base)
		if err != nil {
			return nil, err
		}
		m.Overrides = overrides
		return m, nil
	}
	cs := DefaultFlatWeightChunk
	if len(base.ChunkSize) > 0 {
		cs = base.ChunkSize[0]
	}
	overrides, err := encodeInfiniteFlatWeights(ws, want.WeightStore, cs)
	if err != nil {
		return nil, err
	}
	m.Overrides = overrides
	return m, nil
}

func encodeInfiniteMatrixWeights(ws, baseline *WeightStore, kind string, cfg *InfiniteLayerManifest) ([]WeightChunkOverride, error) {
	cs := append([]int(nil), cfg.ChunkSize...)
	if len(cs) == 0 {
		cs = append([]int(nil), DefaultDenseWeightChunk...)
	}
	var shape []int
	var in, out int
	switch kind {
	case "dense":
		in, out = cfg.In, cfg.Out
		shape = []int{out, in}
	case "embedding":
		if cfg.Embedding == nil {
			return nil, fmt.Errorf("embedding: nil spec")
		}
		in, out = cfg.Embedding.EmbeddingDim, cfg.Embedding.VocabSize
		shape = []int{out, in}
	default:
		return nil, fmt.Errorf("matrix encode: kind %s", kind)
	}
	overrides := make(map[string]WeightChunkOverride)
	err := foreachWeightChunk(shape, cs, func(chunkCoord, localOrigin, localShape []int) error {
		tile, err := extractWeightTile(ws.Master, in, out, localOrigin, localShape)
		if err != nil {
			return err
		}
		wantTile, err := extractWeightTile(baseline.Master, in, out, localOrigin, localShape)
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
		overrides[weightChunkKey(chunkCoord)] = WeightChunkOverride{
			At: chunkCoord, Shape: append([]int(nil), localShape...), Payload: payload,
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	outList := make([]WeightChunkOverride, 0, len(overrides))
	for _, o := range overrides {
		outList = append(outList, o)
	}
	return outList, nil
}

func applyInfiniteOverrides(m *InfiniteLayerManifest, ws *WeightStore) error {
	if m == nil || len(m.Overrides) == 0 {
		return nil
	}
	if isMatrixChunkKind(m.Kind) {
		cs := m.ChunkSize
		if len(cs) == 0 {
			cs = DefaultDenseWeightChunk
		}
		var in, out int
		switch normalizeInfiniteKind(m.Kind) {
		case "dense":
			in, out = m.In, m.Out
		case "embedding":
			if m.Embedding == nil {
				return fmt.Errorf("embedding: nil spec")
			}
			in, out = m.Embedding.EmbeddingDim, m.Embedding.VocabSize
		default:
			return fmt.Errorf("matrix decode: kind %s", m.Kind)
		}
		for _, o := range m.Overrides {
			raw, err := decompressSeedBytes(o.Payload)
			if err != nil {
				return err
			}
			tile, err := unpackFloat32s(raw, numelInt(o.Shape))
			if err != nil {
				return err
			}
			localOrigin := make([]int, 2)
			for i := range o.At {
				localOrigin[i] = o.At[i] * cs[i]
			}
			if err := pasteWeightTile(ws.Master, in, out, localOrigin, o.Shape, tile); err != nil {
				return err
			}
		}
		return nil
	}
	cs := DefaultFlatWeightChunk
	if len(m.ChunkSize) > 0 {
		cs = m.ChunkSize[0]
	}
	return decodeInfiniteFlatOverrides(ws.Master, m.Overrides, cs)
}

func encodeInfiniteFlatWeights(ws, baseline *WeightStore, chunkSize int) ([]WeightChunkOverride, error) {
	if ws == nil || baseline == nil {
		return nil, fmt.Errorf("flat layer: nil weight store")
	}
	if len(ws.Master) != len(baseline.Master) {
		return nil, fmt.Errorf("flat layer: weight len %d vs %d", len(ws.Master), len(baseline.Master))
	}
	if weightStoreFingerprint(ws) == weightStoreFingerprint(baseline) {
		return nil, nil
	}
	if chunkSize <= 0 {
		chunkSize = DefaultFlatWeightChunk
	}
	n := len(ws.Master)
	overrides := make(map[string]WeightChunkOverride)
	for start := 0; start < n; start += chunkSize {
		end := start + chunkSize
		if end > n {
			end = n
		}
		tile := ws.Master[start:end]
		wantTile := baseline.Master[start:end]
		if weightTileEqual(tile, wantTile) {
			delete(overrides, flatChunkKey(start/chunkSize))
			continue
		}
		raw := packFloat32s(tile)
		payload, err := compressSeedBytes(raw)
		if err != nil {
			return nil, err
		}
		idx := start / chunkSize
		overrides[flatChunkKey(idx)] = WeightChunkOverride{
			At: []int{idx}, Shape: []int{end - start}, Payload: payload,
		}
	}
	out := make([]WeightChunkOverride, 0, len(overrides))
	for _, o := range overrides {
		out = append(out, o)
	}
	return out, nil
}

func decodeInfiniteFlatOverrides(master []float32, overrides []WeightChunkOverride, chunkSize int) error {
	if chunkSize <= 0 {
		chunkSize = DefaultFlatWeightChunk
	}
	for _, o := range overrides {
		raw, err := decompressSeedBytes(o.Payload)
		if err != nil {
			return err
		}
		tile, err := unpackFloat32s(raw, numelInt(o.Shape))
		if err != nil {
			return err
		}
		if len(o.At) != 1 {
			return fmt.Errorf("flat layer: override at rank %d", len(o.At))
		}
		start := o.At[0] * chunkSize
		if start+len(tile) > len(master) {
			return fmt.Errorf("flat layer: override past end")
		}
		copy(master[start:start+len(tile)], tile)
	}
	return nil
}

func captureLayerForwardProfile(l *VolumetricLayer) *LayerForwardProfile {
	if l == nil {
		return nil
	}
	p := &LayerForwardProfile{
		TileSize: l.TileSize, UseTiling: l.UseTiling, EnableMultiCoreTiling: l.EnableMultiCoreTiling,
		MaxSeqLen: l.MaxSeqLen, SeqLength: l.SeqLength, RMSNormEps: l.RMSNormEps,
	}
	if l.WeightStore != nil {
		p.WeightScale = l.WeightStore.Scale
	}
	return p
}

func applyLayerForwardProfile(l *VolumetricLayer, p *LayerForwardProfile) {
	if l == nil || p == nil {
		return
	}
	l.TileSize = p.TileSize
	l.UseTiling = p.UseTiling
	l.EnableMultiCoreTiling = p.EnableMultiCoreTiling
	l.MaxSeqLen = p.MaxSeqLen
	if p.SeqLength > 0 {
		l.SeqLength = p.SeqLength
	}
	if p.RMSNormEps > 0 {
		l.RMSNormEps = p.RMSNormEps
	}
	if l.WeightStore != nil && p.WeightScale != 0 {
		l.WeightStore.Scale = p.WeightScale
	}
}

func syncWeightStoreForForward(l *VolumetricLayer) {
	if l == nil || l.WeightStore == nil || len(l.WeightStore.Master) == 0 {
		return
	}
	ws := l.WeightStore
	ws.Versions = nil
	ws.GPUWeights = nil
	ws.CPUPacked = nil
}

func overrideCount(overrides []WeightChunkOverride) int {
	return len(overrides)
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
		return nil, fmt.Errorf("infinite layer: payload %d bytes want %d", len(raw), want)
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

func flatChunkKey(idx int) string {
	return weightChunkKey([]int{idx})
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
		return nil, fmt.Errorf("matrix layer: weight tile rank %d", len(localShape))
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
		return fmt.Errorf("matrix layer: weight tile rank %d", len(localShape))
	}
	if len(tile) != localShape[0]*localShape[1] {
		return fmt.Errorf("matrix layer: tile size mismatch")
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
