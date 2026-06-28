package poly

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// ENTITY — Every Numerical Type In Native TopologY
//
// Native Loom checkpoint format (.entity). Spec: docs/entity.md
//
// Why this exists:
//
// HuggingFace .safetensors is the right *import* lane for PyTorch/HF checkpoints, but
// it cannot express what Loom actually trains and saves:
//
//   - All 21 DTypes with native bit-packing (Int4 nibbles, Binary 8:1, Ternary, …)
//   - Per-layer WeightStore.Scale (quant mapping the checkpoint was trained with)
//   - Volumetric placement (Z,Y,X,L) — not just flat tensor names
//   - Layer topology: parallel branches, sequential stacks, remote links
//   - One bit-perfect reload of a full VolumetricNetwork (see persistence.go idempotency)
//
// Today we have:
//
//   - safetensors.go  → read HF weights (decode mostly to FP32); SaveSafetensors is F32-only
//   - persistence.go  → full fidelity via JSON + Base64 (works, but huge and slow)
//
// ENTITY is the binary middle path: safe length-prefixed header + contiguous blobs, same
// native packing rules as persistence.go (encodeNativeWeights / decodeNativeWeights), without
// pretending to be .safetensors.
//
// Import:  model.safetensors  (HF)
// Native:  fluffy.entity       (ENTITY)

const (
	entityMagic         = "ENTITY\x00\x00"
	entityFormatVersion = 1
	entityHeaderMaxSize = 256 << 20
)

// EntityWeightBlob indexes one native-packed weight payload in the blob section.
type EntityWeightBlob struct {
	Path   string  `json:"path"`
	Offset uint64  `json:"offset"`
	Length uint64  `json:"length"`
	DType  string  `json:"dtype"`
	Scale  float32 `json:"scale,omitempty"`
	Native bool    `json:"native"`
}

// EntityTransformerSpec is the optional universal-transformer add-on in the ENTITY header.
// When present, blob paths transformer.embeddings / transformer.lm_head / transformer.final_norm
// hold FP32 globals outside net.Layers (embed tokens, LM head, pre-head RMSNorm).
type EntityTransformerSpec struct {
	Architecture string                    `json:"architecture"`
	HiddenSize   int                       `json:"hidden_size"`
	VocabSize    int                       `json:"vocab_size"`
	LMHeadTied   bool                      `json:"lm_head_tied,omitempty"`
	HasFinalNorm bool                      `json:"has_final_norm,omitempty"`
	WeightDType  string                    `json:"weight_dtype,omitempty"` // decoder quant baked in at save (INT4, INT8, FLOAT32, …)
	Dims         *EntityTransformerDimsSpec `json:"dims,omitempty"`
}

// EntityTransformerDimsSpec records decoder hyperparameters for validation on reload.
type EntityTransformerDimsSpec struct {
	NumLayers        int     `json:"num_layers"`
	NumHeads         int     `json:"num_heads"`
	NumKVHeads       int     `json:"num_kv_heads"`
	HeadDim          int     `json:"head_dim"`
	QueryDim         int     `json:"query_dim,omitempty"`
	KVDim            int     `json:"kv_dim,omitempty"`
	IntermediateSize int     `json:"intermediate_size"`
	RMSNormEps       float64 `json:"rms_norm_eps,omitempty"`
	RoPEFreqBase     float64 `json:"rope_freq_base,omitempty"`
	Activation       string  `json:"activation,omitempty"`
}

// EntityTransformer is a full causal-LM bundle: volumetric decoder stack + global weights.
type EntityTransformer struct {
	Network      *VolumetricNetwork
	Architecture HFArchitectureKind
	HiddenSize   int
	VocabSize    int
	LMHeadTied   bool
	HasFinalNorm bool
	Dims         HFDecoderDims
	WeightDType  DType
	Embeddings   []float32
	LMHead       []float32
	FinalNorm    []float32
}

type entityHeaderDoc struct {
	FormatVersion uint16                 `json:"format_version"`
	Network       PersistenceNetworkSpec `json:"network"`
	Transformer   *EntityTransformerSpec `json:"transformer,omitempty"`
	Blobs         []EntityWeightBlob     `json:"blobs"`
}

// EntityHeader is the parsed metadata section of an .entity file (no weight bytes).
type EntityHeader struct {
	FormatVersion uint16
	Flags         uint16
	Network       PersistenceNetworkSpec
	Transformer   *EntityTransformerSpec
	Blobs         []EntityWeightBlob
	DataOffset    int
}

// HasTransformer reports whether the checkpoint includes universal-transformer globals.
func (h *EntityHeader) HasTransformer() bool {
	return h != nil && h.Transformer != nil
}

// EntityLoadOptions controls selective weight loading from an .entity file.
// When LayerIndices is non-empty, only blobs under layers.<index> are decoded.
// SkipLayerWeights loads topology only (no layers.* blobs) for staged GPU mount.
type EntityLoadOptions struct {
	LayerIndices     []int
	SkipLayerWeights bool
}

// SerializeEntity writes a VolumetricNetwork to the native .entity wire format.
func SerializeEntity(net *VolumetricNetwork) ([]byte, error) {
	return serializeEntityWire(net, nil, nil, nil, nil, 0)
}

// NewEntityTransformer builds a universal-transformer ENTITY bundle from loaded weights.
func NewEntityTransformer(
	net *VolumetricNetwork,
	arch HFArchitectureKind,
	dims HFDecoderDims,
	embeddings, lmHead, finalNorm []float32,
	hasFinalNorm bool,
) *EntityTransformer {
	hiddenSize := dims.HiddenSize
	if hiddenSize <= 0 && len(finalNorm) > 0 {
		hiddenSize = len(finalNorm)
	}
	vocabSize := 0
	if hiddenSize > 0 && len(embeddings) > 0 {
		vocabSize = len(embeddings) / hiddenSize
	}
	return &EntityTransformer{
		Network:      net,
		Architecture: arch,
		HiddenSize:   hiddenSize,
		VocabSize:    vocabSize,
		LMHeadTied:   entityLMHeadTied(embeddings, lmHead),
		HasFinalNorm: hasFinalNorm,
		Dims:         dims,
		Embeddings:   embeddings,
		LMHead:       lmHead,
		FinalNorm:    finalNorm,
	}
}

// SerializeEntityTransformer writes decoder layers plus universal-transformer globals.
func SerializeEntityTransformer(et *EntityTransformer) ([]byte, error) {
	if et == nil || et.Network == nil {
		return nil, fmt.Errorf("entity transformer: nil network")
	}
	if len(et.Embeddings) == 0 {
		return nil, fmt.Errorf("entity transformer: missing embeddings")
	}
	spec := buildEntityTransformerSpec(et)
	wdt := et.WeightDType
	if wdt == 0 {
		wdt = DTypeFloat32
	}
	return serializeEntityWire(et.Network, spec, et.Embeddings, et.LMHead, et.FinalNorm, wdt)
}

// DeserializeEntityTransformer loads a universal-transformer .entity checkpoint.
func DeserializeEntityTransformer(data []byte) (*EntityTransformer, error) {
	hdr, err := ParseEntityHeader(data)
	if err != nil {
		return nil, err
	}
	if !hdr.HasTransformer() {
		return nil, fmt.Errorf("entity file has no transformer section (network-only checkpoint)")
	}
	net, err := deserializeEntityNetwork(hdr, entityBlobReaderFromBytes(data, hdr), nil)
	if err != nil {
		return nil, err
	}
	embeddings, lmHead, finalNorm, err := loadEntityTransformerGlobals(hdr, entityBlobReaderFromBytes(data, hdr), hdr.Transformer)
	if err != nil {
		return nil, err
	}
	dims := entityTransformerDimsFromSpec(hdr.Transformer.Dims)
	dims.HiddenSize = hdr.Transformer.HiddenSize
	storedDT := entityTransformerWeightDType(hdr)
	return &EntityTransformer{
		Network:      net,
		Architecture: parseEntityArchitecture(hdr.Transformer.Architecture),
		HiddenSize:   hdr.Transformer.HiddenSize,
		VocabSize:    hdr.Transformer.VocabSize,
		LMHeadTied:   hdr.Transformer.LMHeadTied,
		HasFinalNorm: hdr.Transformer.HasFinalNorm,
		Dims:         dims,
		WeightDType:  storedDT,
		Embeddings:   embeddings,
		LMHead:       lmHead,
		FinalNorm:    finalNorm,
	}, nil
}

// EntityTransformerWeightDType reads the decoder dtype stored in a .entity header.
// Legacy checkpoints without weight_dtype infer from native layer blobs (default FLOAT32).
func EntityTransformerWeightDType(path string) (DType, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	hdr, err := ParseEntityHeader(data)
	if err != nil {
		return 0, err
	}
	return entityTransformerWeightDType(hdr), nil
}

func entityTransformerWeightDType(hdr *EntityHeader) DType {
	if hdr == nil || !hdr.HasTransformer() {
		return DTypeFloat32
	}
	if hdr.Transformer.WeightDType != "" {
		return ParseDType(hdr.Transformer.WeightDType)
	}
	for _, blob := range hdr.Blobs {
		if entityBlobIsQ4_0(blob) {
			return DTypeInt4
		}
		if entityBlobIsBitNetTernary(blob) {
			return DTypeTernary
		}
	}
	for _, blob := range hdr.Blobs {
		if !strings.HasPrefix(blob.Path, "layers.") || !blob.Native {
			continue
		}
		parts := strings.Split(blob.Path, ".")
		if len(parts) < 2 {
			continue
		}
		idx, err := strconv.Atoi(parts[1])
		if err != nil || idx%4 == 0 {
			continue // skip RMSNorm slots in HF blocks
		}
		dt := ParseDType(blob.DType)
		if dt != DTypeFloat32 && dt != 0 {
			return dt
		}
	}
	return DTypeFloat32
}

// BuildTransformerFromEntity constructs a runnable Transformer from a loaded bundle.
func BuildTransformerFromEntity[T Numeric](et *EntityTransformer, template Template) *Transformer[T] {
	finalNorm := et.FinalNorm
	if !et.HasFinalNorm {
		finalNorm = nil
	}
	tr := NewTransformer[T](et.Network, et.Embeddings, et.LMHead, finalNorm, template)
	tr.lmHeadTied = et.LMHeadTied
	if !tr.lmHeadTied {
		tr.lmHeadTied = entityLMHeadTied(et.Embeddings, et.LMHead)
	}
	return tr
}

// EntityGPUWeightDType picks GPU upload quant for a loaded .entity checkpoint.
// FP32 on disk → Q4_0 at GPU (Poly Talk [1] path). Baked Q4_0 INT4 uses cached blocks (no re-quant).
func EntityGPUWeightDType(stored DType, useGPU bool) DType {
	if !useGPU {
		return stored
	}
	if stored == 0 || stored == DTypeFloat32 {
		return DTypeInt4
	}
	return stored
}

func restoreEntityTransformerLayerFields(et *EntityTransformer) {
	if et == nil || et.Network == nil {
		return
	}
	queryDim := et.Dims.QueryDim
	if queryDim <= 0 {
		queryDim = et.Dims.NumHeads * et.Dims.HeadDim
	}
	for i := range et.Network.Layers {
		l := &et.Network.Layers[i]
		switch l.Type {
		case LayerSwiGLU:
			if et.Dims.Activation != 0 {
				l.Activation = et.Dims.Activation
			}
		case LayerMultiHeadAttention:
			if queryDim > 0 {
				l.QueryDim = queryDim
			}
			if l.RoPEFreqBase <= 0 && et.Dims.RoPEFreqBase > 0 {
				l.RoPEFreqBase = et.Dims.RoPEFreqBase
			}
		case LayerRMSNorm:
			if l.RMSNormEps <= 0 && et.Dims.RMSNormEps > 0 {
				l.RMSNormEps = et.Dims.RMSNormEps
			}
		}
	}
}

// PrepareEntityTransformerInference restores layer fields and unpacks legacy native INT4 blobs.
// Q4_0 baked checkpoints skip Unpack on load — GPU uses uploadQ4_0Cached; CPU materializes
// FP32 Master in VolumetricLayer.SyncToCPU via MaterializeQ4_0ForCPU.
func PrepareEntityTransformerInference(et *EntityTransformer) {
	if et == nil || et.Network == nil {
		return
	}
	if et.WeightDType == DTypeTernary {
		et.Network.UseExactDType = true
	}
	restoreEntityTransformerLayerFields(et)
	dt := et.WeightDType
	if dt == 0 {
		dt = DTypeFloat32
	}
	for i := range et.Network.Layers {
		l := &et.Network.Layers[i]
		if l.WeightStore == nil {
			continue
		}
		if l.Type == LayerRMSNorm {
			l.DType = DTypeFloat32
			if _, ok := l.WeightStore.Versions[DTypeFloat32]; ok {
				l.WeightStore.Unpack(DTypeFloat32)
			}
			continue
		}
		if l.WeightStore.HasAnyQ4_0() {
			l.DType = DTypeInt4
			continue
		}
		if l.WeightStore.HasAnyBitNetTernary() {
			l.DType = DTypeTernary
			continue
		}
		if dt == DTypeFloat32 {
			l.DType = DTypeFloat32
			continue
		}
		l.DType = dt
		if _, ok := l.WeightStore.Versions[dt]; ok {
			l.WeightStore.Unpack(dt)
		}
	}
}

// PrepareEntityTransformerLayerIndices restores inference fields for specific top-level layer indices.
func PrepareEntityTransformerLayerIndices(et *EntityTransformer, indices []int) {
	if et == nil || et.Network == nil || len(indices) == 0 {
		return
	}
	dt := et.WeightDType
	if dt == 0 {
		dt = DTypeFloat32
	}
	for _, idx := range indices {
		if idx < 0 || idx >= len(et.Network.Layers) {
			continue
		}
		l := &et.Network.Layers[idx]
		if l.WeightStore == nil {
			continue
		}
		if l.Type == LayerRMSNorm {
			l.DType = DTypeFloat32
			if _, ok := l.WeightStore.Versions[DTypeFloat32]; ok {
				l.WeightStore.Unpack(DTypeFloat32)
			}
			continue
		}
		if l.WeightStore.HasAnyQ4_0() {
			l.DType = DTypeInt4
			continue
		}
		if l.WeightStore.HasAnyBitNetTernary() {
			l.DType = DTypeTernary
			continue
		}
		if dt == DTypeFloat32 {
			l.DType = DTypeFloat32
			continue
		}
		l.DType = dt
		if _, ok := l.WeightStore.Versions[dt]; ok {
			l.WeightStore.Unpack(dt)
		}
	}
}

// LoadEntityTransformer reads a universal-transformer .entity checkpoint from disk.
// Uses random-access blob reads (EntityFile) so the full file is not slurped into RAM.
func LoadEntityTransformer(path string) (*EntityTransformer, error) {
	et, err := LoadEntityTransformerFromFile(path)
	if err != nil {
		return nil, err
	}
	ReleaseInferenceTransientMemory()
	return et, nil
}

// LoadEntityTransformerAs reads a checkpoint and returns a runnable Transformer.
func LoadEntityTransformerAs[T Numeric](path string, template Template) (*Transformer[T], error) {
	et, err := LoadEntityTransformer(path)
	if err != nil {
		return nil, err
	}
	PrepareEntityTransformerInference(et)
	return BuildTransformerFromEntity[T](et, template), nil
}

// SaveEntityTransformer writes a universal-transformer .entity checkpoint.
func SaveEntityTransformer(path string, et *EntityTransformer) error {
	data, err := SerializeEntityTransformer(et)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func serializeEntityWire(
	net *VolumetricNetwork,
	transformer *EntityTransformerSpec,
	embeddings, lmHead, finalNorm []float32,
	entityQuant DType,
) ([]byte, error) {
	spec := BuildPersistenceNetworkSpec(net)
	canonicalEntityTopology(&spec)
	stripPersistenceWeights(&spec)

	var blobs []EntityWeightBlob
	var payload bytes.Buffer
	for i := range net.Layers {
		collectEntityWeightBlobs(&net.Layers[i], fmt.Sprintf("layers.%d", i), &payload, &blobs, entityQuant)
	}
	if transformer != nil {
		collectEntityGlobalBlob("embeddings", embeddings, &payload, &blobs)
		if !transformer.LMHeadTied {
			collectEntityGlobalBlob("lm_head", lmHead, &payload, &blobs)
		}
		if transformer.HasFinalNorm {
			collectEntityGlobalBlob("final_norm", finalNorm, &payload, &blobs)
		}
	}

	header := entityHeaderDoc{
		FormatVersion: entityFormatVersion,
		Network:       spec,
		Transformer:   transformer,
		Blobs:         blobs,
	}
	headerJSON, err := json.Marshal(header)
	if err != nil {
		return nil, err
	}
	if len(headerJSON) > entityHeaderMaxSize {
		return nil, fmt.Errorf("entity header too large: %d bytes", len(headerJSON))
	}

	out := make([]byte, 0, entityFixedHeaderSize()+len(headerJSON)+payload.Len())
	out = append(out, entityMagic...)
	var ver [2]byte
	binary.LittleEndian.PutUint16(ver[:], entityFormatVersion)
	out = append(out, ver[:]...)
	out = append(out, 0, 0) // flags
	var hlen [8]byte
	binary.LittleEndian.PutUint64(hlen[:], uint64(len(headerJSON)))
	out = append(out, hlen[:]...)
	out = append(out, headerJSON...)
	out = append(out, payload.Bytes()...)
	return out, nil
}

// ParseEntityHeader reads magic, version, and JSON header from an .entity file.
func ParseEntityHeader(data []byte) (*EntityHeader, error) {
	if len(data) < entityFixedHeaderSize() {
		return nil, fmt.Errorf("entity file too short: %d bytes", len(data))
	}
	if string(data[:8]) != entityMagic {
		return nil, fmt.Errorf("invalid entity magic: %q", data[:8])
	}
	version := binary.LittleEndian.Uint16(data[8:10])
	if version != entityFormatVersion {
		return nil, fmt.Errorf("unsupported entity version: %d", version)
	}
	flags := binary.LittleEndian.Uint16(data[10:12])
	headerLen := binary.LittleEndian.Uint64(data[12:20])
	if headerLen > entityHeaderMaxSize {
		return nil, fmt.Errorf("entity header size unreasonable: %d", headerLen)
	}
	dataOffset := entityFixedHeaderSize() + int(headerLen)
	if dataOffset > len(data) {
		return nil, fmt.Errorf("entity header truncated: need %d bytes, have %d", dataOffset, len(data))
	}

	var doc entityHeaderDoc
	if err := json.Unmarshal(data[entityFixedHeaderSize():dataOffset], &doc); err != nil {
		return nil, fmt.Errorf("entity header JSON: %w", err)
	}
	return &EntityHeader{
		FormatVersion: version,
		Flags:         flags,
		Network:       doc.Network,
		Transformer:   doc.Transformer,
		Blobs:         doc.Blobs,
		DataOffset:    dataOffset,
	}, nil
}

type entityBlobReader func(blob EntityWeightBlob) ([]byte, error)

func entityBlobReaderFromBytes(data []byte, hdr *EntityHeader) entityBlobReader {
	return func(blob EntityWeightBlob) ([]byte, error) {
		return readEntityBlobBytes(hdr, data, blob)
	}
}

func readEntityBlobBytes(hdr *EntityHeader, data []byte, blob EntityWeightBlob) ([]byte, error) {
	end := int(blob.Offset) + int(blob.Length)
	if end > len(data)-hdr.DataOffset {
		return nil, fmt.Errorf("entity blob %q out of range", blob.Path)
	}
	raw := data[hdr.DataOffset+int(blob.Offset) : hdr.DataOffset+end]
	return raw, nil
}

func applyEntityBlobToNetwork(net *VolumetricNetwork, blob EntityWeightBlob, raw []byte) error {
	layerPath := entityBlobLayerPath(blob.Path)
	l, err := layerAtEntityPath(net, layerPath)
	if err != nil {
		return err
	}
	ensureLayerWeightStore(l)
	switch {
	case entityBlobIsQ4_0(blob):
		if err := applyEntityQ4_0Blob(l, blob.Path, raw, blob); err != nil {
			return err
		}
	case entityBlobIsBitNetTernary(blob):
		if err := applyEntityBitNetTernaryBlob(l, blob.Path, raw); err != nil {
			return err
		}
	case strings.HasSuffix(blob.Path, ".biases"):
		if err := applyEntityBiasBlob(l, raw); err != nil {
			return err
		}
	case strings.HasSuffix(blob.Path, ".q_norm"), strings.HasSuffix(blob.Path, ".k_norm"):
		if err := applyEntityMHANormBlob(l, blob.Path, raw); err != nil {
			return err
		}
	case strings.HasSuffix(blob.Path, ".inner_norm"):
		if err := applyEntityBitNetAuxBlob(l, blob.Path, raw); err != nil {
			return err
		}
	default:
		if err := applyEntityWeightBlob(l, raw, blob); err != nil {
			return err
		}
	}
	return nil
}

func deserializeEntityNetwork(hdr *EntityHeader, readBlob entityBlobReader, opts *EntityLoadOptions) (*VolumetricNetwork, error) {
	net := NewVolumetricNetwork(hdr.Network.Depth, hdr.Network.Rows, hdr.Network.Cols, hdr.Network.LayersPerCell)
	for _, ls := range hdr.Network.Layers {
		l := net.GetLayer(ls.Z, ls.Y, ls.X, ls.L)
		if err := applyPersistenceLayerSpec(l, ls); err != nil {
			return nil, err
		}
	}
	if err := applyEntityNetworkLayerBlobs(net, hdr, readBlob, opts); err != nil {
		return nil, err
	}
	return net, nil
}

// applyEntityNetworkLayerBlobs decodes layers.* weight blobs into net.
// Drops each raw blob after apply and GCs after each transformer sub-block (layer index % 4 == 3).
func applyEntityNetworkLayerBlobs(net *VolumetricNetwork, hdr *EntityHeader, readBlob entityBlobReader, opts *EntityLoadOptions) error {
	for _, blob := range hdr.Blobs {
		if !strings.HasPrefix(blob.Path, "layers.") {
			continue
		}
		if opts != nil && opts.SkipLayerWeights {
			continue
		}
		if opts != nil && len(opts.LayerIndices) > 0 && !entityBlobLayerAllowed(blob.Path, opts.LayerIndices) {
			continue
		}
		raw, err := readBlob(blob)
		if err != nil {
			return err
		}
		if err := applyEntityBlobToNetwork(net, blob, raw); err != nil {
			return err
		}
		raw = nil
		if idx := entityBlobTopLayerIndex(blob.Path); idx >= 0 && idx%4 == 3 {
			ReleaseInferenceTransientMemory()
		}
	}
	return nil
}

// DeserializeEntity reconstructs a VolumetricNetwork from an .entity byte slice.
func DeserializeEntity(data []byte) (*VolumetricNetwork, error) {
	return DeserializeEntityWithOptions(data, nil)
}

// DeserializeEntityWithOptions loads topology from the header and optionally only
// decodes weight blobs for the requested top-level layer indices.
func DeserializeEntityWithOptions(data []byte, opts *EntityLoadOptions) (*VolumetricNetwork, error) {
	hdr, err := ParseEntityHeader(data)
	if err != nil {
		return nil, err
	}
	return deserializeEntityNetwork(hdr, entityBlobReaderFromBytes(data, hdr), opts)
}

// DeserializeEntityLayer loads topology plus weights for one top-level layer index.
func DeserializeEntityLayer(data []byte, layerIndex int) (*VolumetricNetwork, error) {
	return DeserializeEntityWithOptions(data, &EntityLoadOptions{LayerIndices: []int{layerIndex}})
}

// SaveEntity writes net to path as a native .entity checkpoint.
func SaveEntity(path string, net *VolumetricNetwork) error {
	data, err := SerializeEntity(net)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// LoadEntity reads a native .entity checkpoint.
func LoadEntity(path string) (*VolumetricNetwork, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return DeserializeEntity(data)
}

// LoadEntityWithOptions reads a checkpoint with selective layer weight loading.
func LoadEntityWithOptions(path string, opts *EntityLoadOptions) (*VolumetricNetwork, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return DeserializeEntityWithOptions(data, opts)
}

// LayerPersistenceFromEntity reads on-disk weight fields for a top-level layer index.
func LayerPersistenceFromEntity(data []byte, layerIndex int) (raw []byte, scale float32, native bool, err error) {
	hdr, err := ParseEntityHeader(data)
	if err != nil {
		return nil, 0, false, err
	}
	want := fmt.Sprintf("layers.%d", layerIndex)
	for _, blob := range hdr.Blobs {
		if blob.Path != want {
			continue
		}
		end := int(blob.Offset) + int(blob.Length)
		if end > len(data)-hdr.DataOffset {
			return nil, 0, false, fmt.Errorf("entity blob %q truncated", blob.Path)
		}
		raw = append([]byte(nil), data[hdr.DataOffset+int(blob.Offset):hdr.DataOffset+end]...)
		return raw, blob.Scale, blob.Native, nil
	}
	return nil, 0, false, fmt.Errorf("entity layer %d has no weight blob", layerIndex)
}

// EntityBlobBytes returns the raw bytes for a blob index without decoding dtype.
func EntityBlobBytes(data []byte, blobIndex int) ([]byte, error) {
	hdr, err := ParseEntityHeader(data)
	if err != nil {
		return nil, err
	}
	if blobIndex < 0 || blobIndex >= len(hdr.Blobs) {
		return nil, fmt.Errorf("entity blob index %d out of range (blobs=%d)", blobIndex, len(hdr.Blobs))
	}
	blob := hdr.Blobs[blobIndex]
	end := int(blob.Offset) + int(blob.Length)
	if end > len(data)-hdr.DataOffset {
		return nil, fmt.Errorf("entity blob %q out of range", blob.Path)
	}
	return append([]byte(nil), data[hdr.DataOffset+int(blob.Offset):hdr.DataOffset+end]...), nil
}

func entityFixedHeaderSize() int { return 20 }

func stripPersistenceWeights(spec *PersistenceNetworkSpec) {
	for i := range spec.Layers {
		stripLayerWeights(&spec.Layers[i])
	}
}

func stripLayerWeights(ls *PersistenceLayerSpec) {
	ls.Weights = ""
	ls.Native = false
	ls.Scale = 0
	for i := range ls.ParallelBranches {
		stripLayerWeights(&ls.ParallelBranches[i])
	}
	for i := range ls.SequentialLayers {
		stripLayerWeights(&ls.SequentialLayers[i])
	}
	if ls.MetaObservedLayer != nil {
		stripLayerWeights(ls.MetaObservedLayer)
	}
}

func canonicalEntityTopology(spec *PersistenceNetworkSpec) {
	for i := range spec.Layers {
		canonicalEntityLayerSpec(&spec.Layers[i])
	}
}

func canonicalEntityLayerSpec(ls *PersistenceLayerSpec) {
	switch ParseLayerType(ls.Type) {
	case LayerMultiHeadAttention, LayerRNN, LayerLSTM:
	default:
		if ls.SeqLength == 1 {
			ls.SeqLength = 0
		}
	}
	for i := range ls.ParallelBranches {
		canonicalEntityLayerSpec(&ls.ParallelBranches[i])
	}
	for i := range ls.SequentialLayers {
		canonicalEntityLayerSpec(&ls.SequentialLayers[i])
	}
	if ls.MetaObservedLayer != nil {
		canonicalEntityLayerSpec(ls.MetaObservedLayer)
	}
}

func collectEntityWeightBlobs(l *VolumetricLayer, path string, payload *bytes.Buffer, blobs *[]EntityWeightBlob, entityQuant DType) {
	if entityQuant == DTypeTernary && (l.Type == LayerMultiHeadAttention || l.Type == LayerSwiGLU) {
		if l.WeightStore != nil && l.WeightStore.hasBitNetCPUPacked() {
			collectEntityBitNetLayer(l, path, payload, blobs)
		}
	} else if entityQuant == DTypeInt4 && (l.Type == LayerMultiHeadAttention || l.Type == LayerSwiGLU) {
		if l.WeightStore != nil && len(l.WeightStore.Master) > 0 {
			collectEntityQ4_0Layer(l, path, payload, blobs)
		}
	} else if l.WeightStore != nil {
		dt := l.DType
		if l.Type == LayerRMSNorm || entityQuant == DTypeInt4 || entityQuant == DTypeTernary {
			dt = DTypeFloat32
		}
		scale := l.WeightStore.Scale
		active := l.WeightStore.Versions[dt]
		if active == nil && len(l.WeightStore.Master) > 0 {
			delete(l.WeightStore.Versions, dt)
			l.WeightStore.Morph(dt)
			active = l.WeightStore.Versions[dt]
		}
		if active == nil {
			active = l.WeightStore.GetNative(dt)
		}
		if active != nil {
			raw := EncodeNativeWeightsRaw(active, dt)
			if len(raw) > 0 {
				offset := payload.Len()
				payload.Write(raw)
				*blobs = append(*blobs, EntityWeightBlob{
					Path:   path,
					Offset: uint64(offset),
					Length: uint64(len(raw)),
					DType:  dt.String(),
					Scale:  scale,
					Native: true,
				})
			}
		} else if len(l.WeightStore.Master) > 0 {
			raw := EncodeWeightsRaw(l.WeightStore.Master)
			offset := payload.Len()
			payload.Write(raw)
			*blobs = append(*blobs, EntityWeightBlob{
				Path:   path,
				Offset: uint64(offset),
				Length: uint64(len(raw)),
				DType:  DTypeFloat32.String(),
				Scale:  scale,
				Native: false,
			})
		}
	}
	for i := range l.ParallelBranches {
		collectEntityWeightBlobs(&l.ParallelBranches[i], fmt.Sprintf("%s.parallel_branches.%d", path, i), payload, blobs, entityQuant)
	}
	for i := range l.SequentialLayers {
		collectEntityWeightBlobs(&l.SequentialLayers[i], fmt.Sprintf("%s.sequential_layers.%d", path, i), payload, blobs, entityQuant)
	}
	if l.MetaObservedLayer != nil {
		collectEntityWeightBlobs(l.MetaObservedLayer, path+".meta_observed_layer", payload, blobs, entityQuant)
	}
}

func applyEntityWeightBlob(l *VolumetricLayer, raw []byte, blob EntityWeightBlob) error {
	ensureLayerWeightStore(l)
	l.WeightStore.Scale = blob.Scale
	dt := ParseDType(blob.DType)
	l.DType = dt
	if blob.Native {
		decoded, err := DecodeNativeWeightsRaw(raw, dt)
		if err != nil {
			return err
		}
		l.WeightStore.SetLoadedWeights(dt, decoded)
		if len(l.WeightStore.Master) == 0 && dt == DTypeFloat32 {
			if w, ok := decoded.([]float32); ok {
				l.WeightStore.Master = w
			}
		}
		return nil
	}
	m, err := DecodeWeightsRaw(raw)
	if err != nil {
		return err
	}
	l.WeightStore.Master = m
	l.WeightStore.Morph(dt)
	return nil
}

func layerAtEntityPath(net *VolumetricNetwork, path string) (*VolumetricLayer, error) {
	parts := strings.Split(path, ".")
	if len(parts) < 2 || parts[0] != "layers" {
		return nil, fmt.Errorf("invalid entity layer path: %q", path)
	}
	idx, err := strconv.Atoi(parts[1])
	if err != nil || idx < 0 || idx >= len(net.Layers) {
		return nil, fmt.Errorf("entity layer index out of range: %q", path)
	}
	l := &net.Layers[idx]
	for i := 2; i < len(parts); i++ {
		switch parts[i] {
		case "parallel_branches":
			i++
			if i >= len(parts) {
				return nil, fmt.Errorf("invalid entity path: %q", path)
			}
			bi, err := strconv.Atoi(parts[i])
			if err != nil || bi < 0 || bi >= len(l.ParallelBranches) {
				return nil, fmt.Errorf("entity parallel branch out of range: %q", path)
			}
			l = &l.ParallelBranches[bi]
		case "sequential_layers":
			i++
			if i >= len(parts) {
				return nil, fmt.Errorf("invalid entity path: %q", path)
			}
			si, err := strconv.Atoi(parts[i])
			if err != nil || si < 0 || si >= len(l.SequentialLayers) {
				return nil, fmt.Errorf("entity sequential layer out of range: %q", path)
			}
			l = &l.SequentialLayers[si]
		case "meta_observed_layer":
			if l.MetaObservedLayer == nil {
				return nil, fmt.Errorf("entity path missing meta_observed_layer: %q", path)
			}
			l = l.MetaObservedLayer
		default:
			return nil, fmt.Errorf("unknown entity path segment %q in %q", parts[i], path)
		}
	}
	return l, nil
}

func entityBlobTopLayerIndex(path string) int {
	parts := strings.Split(path, ".")
	if len(parts) < 2 || parts[0] != "layers" {
		return -1
	}
	idx, err := strconv.Atoi(parts[1])
	if err != nil {
		return -1
	}
	return idx
}

func entityBlobLayerAllowed(path string, indices []int) bool {
	idx := entityBlobTopLayerIndex(path)
	if idx < 0 {
		return false
	}
	for _, want := range indices {
		if idx == want {
			return true
		}
	}
	return false
}

func buildEntityTransformerSpec(et *EntityTransformer) *EntityTransformerSpec {
	wdt := et.WeightDType
	if wdt == 0 {
		wdt = DTypeFloat32
	}
	spec := &EntityTransformerSpec{
		Architecture: et.Architecture.String(),
		HiddenSize:   et.HiddenSize,
		VocabSize:    et.VocabSize,
		LMHeadTied:   et.LMHeadTied,
		HasFinalNorm: et.HasFinalNorm,
		WeightDType:  wdt.String(),
		Dims: &EntityTransformerDimsSpec{
			NumLayers:        et.Dims.NumLayers,
			NumHeads:         et.Dims.NumHeads,
			NumKVHeads:       et.Dims.NumKVHeads,
			HeadDim:          et.Dims.HeadDim,
			QueryDim:         et.Dims.QueryDim,
			KVDim:            et.Dims.KVDim,
			IntermediateSize: et.Dims.IntermediateSize,
			RMSNormEps:       et.Dims.RMSNormEps,
			RoPEFreqBase:     et.Dims.RoPEFreqBase,
			Activation:       et.Dims.Activation.String(),
		},
	}
	return spec
}

func entityTransformerDimsFromSpec(spec *EntityTransformerDimsSpec) HFDecoderDims {
	if spec == nil {
		return HFDecoderDims{}
	}
	queryDim := spec.QueryDim
	if queryDim <= 0 {
		queryDim = spec.NumHeads * spec.HeadDim
	}
	kvDim := spec.KVDim
	if kvDim <= 0 {
		kvDim = spec.NumKVHeads * spec.HeadDim
	}
	return HFDecoderDims{
		NumLayers:        spec.NumLayers,
		HiddenSize:       0,
		NumHeads:         spec.NumHeads,
		NumKVHeads:       spec.NumKVHeads,
		HeadDim:          spec.HeadDim,
		QueryDim:         queryDim,
		KVDim:            kvDim,
		IntermediateSize: spec.IntermediateSize,
		RMSNormEps:       spec.RMSNormEps,
		RoPEFreqBase:     spec.RoPEFreqBase,
		Activation:       ParseActivationType(spec.Activation),
	}
}

func parseEntityArchitecture(s string) HFArchitectureKind {
	switch s {
	case HFArchLlamaStyleDecoder.String():
		return HFArchLlamaStyleDecoder
	default:
		return HFArchUnknown
	}
}

func entityLMHeadTied(embeddings, lmHead []float32) bool {
	return len(embeddings) > 0 && len(lmHead) > 0 && len(embeddings) == len(lmHead) && &embeddings[0] == &lmHead[0]
}

func collectEntityGlobalBlob(name string, weights []float32, payload *bytes.Buffer, blobs *[]EntityWeightBlob) {
	if len(weights) == 0 {
		return
	}
	raw := EncodeWeightsRaw(weights)
	offset := payload.Len()
	payload.Write(raw)
	*blobs = append(*blobs, EntityWeightBlob{
		Path:   "transformer." + name,
		Offset: uint64(offset),
		Length: uint64(len(raw)),
		DType:  DTypeFloat32.String(),
		Native: false,
	})
}

func loadEntityTransformerGlobals(hdr *EntityHeader, readBlob entityBlobReader, spec *EntityTransformerSpec) (embeddings, lmHead, finalNorm []float32, err error) {
	for _, blob := range hdr.Blobs {
		if !strings.HasPrefix(blob.Path, "transformer.") {
			continue
		}
		raw, readErr := readBlob(blob)
		if readErr != nil {
			return nil, nil, nil, readErr
		}
		decoded, decErr := DecodeWeightsRawOwned(raw)
		raw = nil
		if decErr != nil {
			return nil, nil, nil, fmt.Errorf("entity blob %q: %w", blob.Path, decErr)
		}
		switch blob.Path {
		case "transformer.embeddings":
			embeddings = decoded
			ReleaseInferenceTransientMemory()
		case "transformer.lm_head":
			lmHead = decoded
		case "transformer.final_norm":
			finalNorm = decoded
		}
	}
	ReleaseInferenceTransientMemory()
	if len(embeddings) == 0 {
		return nil, nil, nil, fmt.Errorf("entity transformer: missing embeddings blob")
	}
	if spec.LMHeadTied {
		lmHead = embeddings
	} else if len(lmHead) == 0 {
		return nil, nil, nil, fmt.Errorf("entity transformer: missing lm_head blob")
	}
	if spec.HasFinalNorm && len(finalNorm) == 0 {
		return nil, nil, nil, fmt.Errorf("entity transformer: missing final_norm blob")
	}
	return embeddings, lmHead, finalNorm, nil
}

func ensureLayerWeightStore(l *VolumetricLayer) {
	if l != nil && l.WeightStore == nil {
		l.WeightStore = NewWeightStore(0)
	}
}
