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

type entityHeaderDoc struct {
	FormatVersion uint16                 `json:"format_version"`
	Network       PersistenceNetworkSpec `json:"network"`
	Blobs         []EntityWeightBlob     `json:"blobs"`
}

// EntityHeader is the parsed metadata section of an .entity file (no weight bytes).
type EntityHeader struct {
	FormatVersion uint16
	Flags         uint16
	Network       PersistenceNetworkSpec
	Blobs         []EntityWeightBlob
	DataOffset    int
}

// EntityLoadOptions controls selective weight loading from an .entity file.
// When LayerIndices is non-nil, only blobs under layers.<index> are decoded.
type EntityLoadOptions struct {
	LayerIndices []int
}

// SerializeEntity writes a VolumetricNetwork to the native .entity wire format.
func SerializeEntity(net *VolumetricNetwork) ([]byte, error) {
	spec := BuildPersistenceNetworkSpec(net)
	canonicalEntityTopology(&spec)
	stripPersistenceWeights(&spec)

	var blobs []EntityWeightBlob
	var payload bytes.Buffer
	for i := range net.Layers {
		collectEntityWeightBlobs(&net.Layers[i], fmt.Sprintf("layers.%d", i), &payload, &blobs)
	}

	header := entityHeaderDoc{
		FormatVersion: entityFormatVersion,
		Network:       spec,
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
		Blobs:         doc.Blobs,
		DataOffset:    dataOffset,
	}, nil
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
	net := NewVolumetricNetwork(hdr.Network.Depth, hdr.Network.Rows, hdr.Network.Cols, hdr.Network.LayersPerCell)
	for _, ls := range hdr.Network.Layers {
		l := net.GetLayer(ls.Z, ls.Y, ls.X, ls.L)
		if err := applyPersistenceLayerSpec(l, ls); err != nil {
			return nil, err
		}
	}
	for _, blob := range hdr.Blobs {
		if opts != nil && len(opts.LayerIndices) > 0 && !entityBlobLayerAllowed(blob.Path, opts.LayerIndices) {
			continue
		}
		end := int(blob.Offset) + int(blob.Length)
		if end > len(data)-hdr.DataOffset {
			return nil, fmt.Errorf("entity blob %q out of range", blob.Path)
		}
		raw := data[hdr.DataOffset+int(blob.Offset) : hdr.DataOffset+end]
		l, err := layerAtEntityPath(net, blob.Path)
		if err != nil {
			return nil, err
		}
		if err := applyEntityWeightBlob(l, raw, blob); err != nil {
			return nil, err
		}
	}
	return net, nil
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

func collectEntityWeightBlobs(l *VolumetricLayer, path string, payload *bytes.Buffer, blobs *[]EntityWeightBlob) {
	if l.WeightStore != nil {
		dt := l.DType
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
		collectEntityWeightBlobs(&l.ParallelBranches[i], fmt.Sprintf("%s.parallel_branches.%d", path, i), payload, blobs)
	}
	for i := range l.SequentialLayers {
		collectEntityWeightBlobs(&l.SequentialLayers[i], fmt.Sprintf("%s.sequential_layers.%d", path, i), payload, blobs)
	}
	if l.MetaObservedLayer != nil {
		collectEntityWeightBlobs(l.MetaObservedLayer, path+".meta_observed_layer", payload, blobs)
	}
}

func applyEntityWeightBlob(l *VolumetricLayer, raw []byte, blob EntityWeightBlob) error {
	if l.WeightStore == nil {
		return fmt.Errorf("no WeightStore for entity blob %q", blob.Path)
	}
	l.WeightStore.Scale = blob.Scale
	dt := ParseDType(blob.DType)
	if blob.Native {
		decoded, err := DecodeNativeWeightsRaw(raw, dt)
		if err != nil {
			return err
		}
		l.WeightStore.SetLoadedWeights(dt, decoded)
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

func entityBlobLayerAllowed(path string, indices []int) bool {
	parts := strings.Split(path, ".")
	if len(parts) < 2 || parts[0] != "layers" {
		return false
	}
	idx, err := strconv.Atoi(parts[1])
	if err != nil {
		return false
	}
	for _, want := range indices {
		if idx == want {
			return true
		}
	}
	return false
}
