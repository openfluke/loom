package poly

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
)

// EntityFile reads .entity weight blobs via ReadAt without loading the full file into RAM.
// baseOffset is the byte offset of the loom ENTITY section (0 for plain .entity; CHGLUE standalone uses 0).
// maxLoomEnd, when > 0, is the absolute file offset past the last byte of the loom ENTITY section (CHGLUE wrapper).
type EntityFile struct {
	f          *os.File
	baseOffset int64
	maxLoomEnd int64
	hdr        *EntityHeader
}

// OpenEntityFile opens a plain .entity checkpoint for random-access blob reads.
func OpenEntityFile(path string) (*EntityFile, error) {
	return OpenEntityFileAt(path, 0, 0)
}

// OpenEntityFileAt opens path and parses the ENTITY header at baseOffset.
// maxLoomEnd is the absolute file offset one past the loom bytes (0 = use full file size).
func OpenEntityFileAt(path string, baseOffset int64, maxLoomEnd int64) (*EntityFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	ef := &EntityFile{f: f, baseOffset: baseOffset, maxLoomEnd: maxLoomEnd}
	if err := ef.readHeader(); err != nil {
		_ = f.Close()
		return nil, err
	}
	return ef, nil
}

// Close releases the underlying file handle.
func (ef *EntityFile) Close() error {
	if ef == nil || ef.f == nil {
		return nil
	}
	err := ef.f.Close()
	ef.f = nil
	return err
}

// Header returns parsed ENTITY metadata.
func (ef *EntityFile) Header() *EntityHeader {
	if ef == nil {
		return nil
	}
	return ef.hdr
}

func (ef *EntityFile) readHeader() error {
	if ef.f == nil {
		return fmt.Errorf("entity file: closed")
	}
	fixed := make([]byte, entityFixedHeaderSize())
	if _, err := ef.f.ReadAt(fixed, ef.baseOffset); err != nil {
		return fmt.Errorf("entity fixed header: %w", err)
	}
	if string(fixed[:8]) != entityMagic {
		return fmt.Errorf("invalid entity magic: %q", fixed[:8])
	}
	version := binary.LittleEndian.Uint16(fixed[8:10])
	if version != entityFormatVersion {
		return fmt.Errorf("unsupported entity version: %d", version)
	}
	flags := binary.LittleEndian.Uint16(fixed[10:12])
	headerLen := binary.LittleEndian.Uint64(fixed[12:20])
	if headerLen > entityHeaderMaxSize {
		return fmt.Errorf("entity header size unreasonable: %d", headerLen)
	}
	headerJSON := make([]byte, headerLen)
	if _, err := ef.f.ReadAt(headerJSON, ef.baseOffset+int64(entityFixedHeaderSize())); err != nil {
		return fmt.Errorf("entity header JSON: %w", err)
	}
	var doc entityHeaderDoc
	if err := json.Unmarshal(headerJSON, &doc); err != nil {
		return fmt.Errorf("entity header JSON: %w", err)
	}
	ef.hdr = &EntityHeader{
		FormatVersion: version,
		Flags:         flags,
		Network:       doc.Network,
		Transformer:   doc.Transformer,
		Blobs:         doc.Blobs,
		DataOffset:    entityFixedHeaderSize() + int(headerLen),
	}
	return nil
}

func (ef *EntityFile) readBlob(blob EntityWeightBlob) ([]byte, error) {
	if ef == nil || ef.f == nil || ef.hdr == nil {
		return nil, fmt.Errorf("entity file: not open")
	}
	end := int(blob.Offset) + int(blob.Length)
	absEnd := ef.baseOffset + int64(ef.hdr.DataOffset) + int64(end)
	if ef.maxLoomEnd > 0 && absEnd > ef.maxLoomEnd {
		return nil, fmt.Errorf("entity blob %q extends past loom section", blob.Path)
	}
	if end > ef.payloadSize() {
		return nil, fmt.Errorf("entity blob %q out of range", blob.Path)
	}
	raw := make([]byte, int(blob.Length))
	off := ef.baseOffset + int64(ef.hdr.DataOffset) + int64(blob.Offset)
	if _, err := ef.f.ReadAt(raw, off); err != nil {
		return nil, fmt.Errorf("entity blob %q: %w", blob.Path, err)
	}
	return raw, nil
}

func (ef *EntityFile) payloadSize() int {
	if ef == nil || ef.f == nil {
		return 0
	}
	st, err := ef.f.Stat()
	if err != nil {
		return 0
	}
	avail := int(st.Size() - ef.baseOffset - int64(ef.hdr.DataOffset))
	if ef.maxLoomEnd > ef.baseOffset {
		loomPayload := int(ef.maxLoomEnd - ef.baseOffset - int64(ef.hdr.DataOffset))
		if loomPayload >= 0 && (avail <= 0 || loomPayload < avail) {
			avail = loomPayload
		}
	}
	if avail < 0 {
		return 0
	}
	return avail
}

func (ef *EntityFile) blobReader() entityBlobReader {
	return ef.readBlob
}

// LoadEntityTransformer reads a full universal-transformer checkpoint from disk without slurping the file.
func (ef *EntityFile) LoadEntityTransformer() (*EntityTransformer, error) {
	return ef.loadEntityTransformer(nil)
}

// LoadEntityTransformerTopology loads decoder topology and globals only (no layers.* weights).
func (ef *EntityFile) LoadEntityTransformerTopology() (*EntityTransformer, error) {
	return ef.loadEntityTransformer(&EntityLoadOptions{SkipLayerWeights: true})
}

func (ef *EntityFile) loadEntityTransformer(opts *EntityLoadOptions) (*EntityTransformer, error) {
	if ef.hdr == nil || !ef.hdr.HasTransformer() {
		return nil, fmt.Errorf("entity file has no transformer section (network-only checkpoint)")
	}
	net, err := deserializeEntityNetwork(ef.hdr, ef.blobReader(), opts)
	if err != nil {
		return nil, err
	}
	embeddings, lmHead, finalNorm, lmHeadQ4Scales, lmHeadQ4Packed, err := loadEntityTransformerGlobals(ef.hdr, ef.blobReader(), ef.hdr.Transformer)
	if err != nil {
		return nil, err
	}
	dims := entityTransformerDimsFromSpec(ef.hdr.Transformer.Dims)
	dims.HiddenSize = ef.hdr.Transformer.HiddenSize
	storedDT := entityTransformerWeightDType(ef.hdr)
	return &EntityTransformer{
		Network:        net,
		Architecture:   parseEntityArchitecture(ef.hdr.Transformer.Architecture),
		HiddenSize:     ef.hdr.Transformer.HiddenSize,
		VocabSize:      ef.hdr.Transformer.VocabSize,
		LMHeadTied:     ef.hdr.Transformer.LMHeadTied,
		HasFinalNorm:   ef.hdr.Transformer.HasFinalNorm,
		Dims:           dims,
		WeightDType:    storedDT,
		Embeddings:     embeddings,
		LMHead:         lmHead,
		FinalNorm:      finalNorm,
		LMHeadQ4Scales: lmHeadQ4Scales,
		LMHeadQ4Packed: lmHeadQ4Packed,
	}, nil
}

// LoadNetworkLayerWeights decodes weight blobs for the given top-level layer indices into net.
func (ef *EntityFile) LoadNetworkLayerWeights(net *VolumetricNetwork, layerIndices []int) error {
	if net == nil {
		return fmt.Errorf("nil network")
	}
	if len(layerIndices) == 0 {
		return nil
	}
	return applyEntityNetworkLayerBlobs(net, ef.hdr, ef.blobReader(), &EntityLoadOptions{LayerIndices: layerIndices})
}

// LoadEntityTransformerFromFile reads a universal-transformer .entity from disk without loading the full file into RAM.
func LoadEntityTransformerFromFile(path string) (*EntityTransformer, error) {
	return LoadEntityTransformerFromFileAt(path, 0, 0)
}

// LoadEntityTransformerFromFileAt is like LoadEntityTransformerFromFile with a byte offset and optional loom end (CHGLUE).
func LoadEntityTransformerFromFileAt(path string, baseOffset int64, maxLoomEnd int64) (*EntityTransformer, error) {
	ef, err := OpenEntityFileAt(path, baseOffset, maxLoomEnd)
	if err != nil {
		return nil, err
	}
	defer ef.Close()
	et, err := ef.LoadEntityTransformer()
	if err != nil {
		return nil, err
	}
	ReleaseInferenceTransientMemory()
	return et, nil
}
