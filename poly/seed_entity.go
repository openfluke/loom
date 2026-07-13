package poly

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// EntitySeedInfo is topology + resolved init seed from an .entity header.
type EntitySeedInfo struct {
	Path           string
	ModelID        string
	HasTransformer bool
	Architecture   string
	NumLayers      int
	HiddenSize     int
	VocabSize      int
	WeightDType    DType
	StoredSeed     uint64
	ResolvedSeed   uint64
	SeedDerived    bool
}

// EntityModelIDFromPath maps lucy_entities/HF--model.entity → HF/model.
func EntityModelIDFromPath(path string) string {
	base := strings.TrimSuffix(filepath.Base(path), ".entity")
	return strings.ReplaceAll(base, "--", "/")
}

// EntityTransformerRecipeSeed derives init seed from model topology (no weight bytes).
func EntityTransformerRecipeSeed(modelID string, arch HFArchitectureKind, dims HFDecoderDims, wdt DType) uint64 {
	parts := []any{
		"loom-entity-v1",
		modelID,
		arch.String(),
		dims.HiddenSize,
		dims.NumLayers,
		dims.NumHeads,
		dims.NumKVHeads,
		dims.HeadDim,
		dims.QueryDim,
		dims.KVDim,
		dims.IntermediateSize,
		dims.RMSNormEps,
		dims.RoPEFreqBase,
		dims.Activation.String(),
		wdt.String(),
	}
	return SeedFrom(parts...)
}

// BuildSeededEntityTransformer builds topology + He-init weights from init seed.
func BuildSeededEntityTransformer(
	initSeed uint64,
	dims HFDecoderDims,
	vocab int,
	wdt DType,
	tiedLMHead, hasFinalNorm bool,
) *EntityTransformer {
	if dims.NumLayers <= 0 || dims.HiddenSize <= 0 || vocab <= 0 {
		return nil
	}
	net := NewVolumetricNetwork(1, 1, 1, dims.NumLayers*4)
	InitHFDecoderBlocks(net, dims)
	embeddings := make([]float32, vocab*dims.HiddenSize)
	var lmHead []float32
	if tiedLMHead {
		lmHead = embeddings
	} else {
		lmHead = make([]float32, vocab*dims.HiddenSize)
	}
	var finalNorm []float32
	if hasFinalNorm {
		finalNorm = make([]float32, dims.HiddenSize)
	}
	et := NewEntityTransformer(net, HFArchLlamaStyleDecoder, dims, embeddings, lmHead, finalNorm, hasFinalNorm)
	et.WeightDType = wdt
	et.VocabSize = vocab
	et.HiddenSize = dims.HiddenSize
	_ = InitSeededEntity(et, initSeed)
	return et
}

// InspectEntitySeed reads .entity header and resolves init seed.
func InspectEntitySeed(path string) (*EntitySeedInfo, error) {
	hdr, err := readEntityHeaderFile(path)
	if err != nil {
		return nil, err
	}
	info := entitySeedInfoFromHeader(path, hdr)
	info.ResolvedSeed = ResolveEntityInitSeed(hdr, info.ModelID)
	info.SeedDerived = info.StoredSeed == 0 || info.StoredSeed != info.ResolvedSeed
	return info, nil
}

// ResolveEntityInitSeed returns stored init_seed or recipe seed from topology.
func ResolveEntityInitSeed(hdr *EntityHeader, modelID string) uint64 {
	if hdr == nil {
		return 0
	}
	if hdr.Network.InitSeed != 0 {
		return hdr.Network.InitSeed
	}
	if !hdr.HasTransformer() {
		return 0
	}
	dims := entityTransformerDimsFromSpec(hdr.Transformer.Dims)
	dims.HiddenSize = hdr.Transformer.HiddenSize
	wdt := ParseDType(hdr.Transformer.WeightDType)
	if wdt == 0 {
		wdt = entityTransformerWeightDType(hdr)
	}
	return EntityTransformerRecipeSeed(modelID, parseSeedArchitecture(hdr.Transformer.Architecture), dims, wdt)
}

// ExtractEntitySeed reads init_seed from serialized .entity bytes.
func ExtractEntitySeed(data []byte) (uint64, error) {
	hdr, err := ParseEntityHeader(data)
	if err != nil {
		return 0, err
	}
	if hdr.Network.InitSeed != 0 {
		return hdr.Network.InitSeed, nil
	}
	return 0, fmt.Errorf("entity: no init_seed in header (legacy checkpoint)")
}

// RecreateEntityTransformerFromSeed loads topology and He-inits from resolved seed.
func RecreateEntityTransformerFromSeed(path, modelID string) (*EntityTransformer, uint64, error) {
	et, seed, err := LoadEntityTopologySeeded(path, modelID, 0)
	return et, seed, err
}

// LoadEntityTopologySeeded loads header topology only; initSeed 0 uses resolved recipe seed.
func LoadEntityTopologySeeded(path, modelID string, initSeed uint64) (*EntityTransformer, uint64, error) {
	prefix, hdr, err := readEntityHeaderBytesFile(path)
	if err != nil {
		return nil, 0, err
	}
	if modelID == "" {
		modelID = EntityModelIDFromPath(path)
	}
	if initSeed == 0 {
		initSeed = ResolveEntityInitSeed(hdr, modelID)
	}
	if !hdr.HasTransformer() {
		return nil, initSeed, fmt.Errorf("%s: no transformer section", filepath.Base(path))
	}
	net, err := DeserializeEntityWithOptions(prefix, &EntityLoadOptions{SkipLayerWeights: true})
	if err != nil {
		return nil, initSeed, err
	}
	hidden := hdr.Transformer.HiddenSize
	vocab := hdr.Transformer.VocabSize
	embeddings := make([]float32, vocab*hidden)
	var lmHead []float32
	if hdr.Transformer.LMHeadTied {
		lmHead = embeddings
	} else {
		lmHead = make([]float32, vocab*hidden)
	}
	var finalNorm []float32
	if hdr.Transformer.HasFinalNorm {
		finalNorm = make([]float32, hidden)
	}
	dims := entityTransformerDimsFromSpec(hdr.Transformer.Dims)
	dims.HiddenSize = hidden
	wdt := ParseDType(hdr.Transformer.WeightDType)
	if wdt == 0 {
		wdt = entityTransformerWeightDType(hdr)
	}
	et := NewEntityTransformer(
		net,
		parseSeedArchitecture(hdr.Transformer.Architecture),
		dims,
		embeddings, lmHead, finalNorm,
		hdr.Transformer.HasFinalNorm,
	)
	et.WeightDType = wdt
	et.VocabSize = vocab
	et.HiddenSize = hidden
	if err := InitSeededEntity(et, initSeed); err != nil {
		return nil, initSeed, err
	}
	return et, initSeed, nil
}

// DeserializeEntityTransformerWithOptions loads transformer with selective weights.
func DeserializeEntityTransformerWithOptions(data []byte, opts *EntityLoadOptions) (*EntityTransformer, error) {
	hdr, err := ParseEntityHeader(data)
	if err != nil {
		return nil, err
	}
	if !hdr.HasTransformer() {
		return nil, fmt.Errorf("entity file has no transformer section")
	}
	net, err := deserializeEntityNetwork(hdr, entityBlobReaderFromBytes(data, hdr), opts)
	if err != nil {
		return nil, err
	}
	var embeddings, lmHead, finalNorm []float32
	if opts == nil || !opts.SkipLayerWeights {
		embeddings, lmHead, finalNorm, err = loadEntityTransformerGlobals(hdr, entityBlobReaderFromBytes(data, hdr), hdr.Transformer)
		if err != nil {
			return nil, err
		}
	} else {
		hidden := hdr.Transformer.HiddenSize
		vocab := hdr.Transformer.VocabSize
		embeddings = make([]float32, vocab*hidden)
		if hdr.Transformer.LMHeadTied {
			lmHead = embeddings
		} else {
			lmHead = make([]float32, vocab*hidden)
		}
		if hdr.Transformer.HasFinalNorm {
			finalNorm = make([]float32, hidden)
		}
	}
	dims := entityTransformerDimsFromSpec(hdr.Transformer.Dims)
	dims.HiddenSize = hdr.Transformer.HiddenSize
	storedDT := entityTransformerWeightDType(hdr)
	return &EntityTransformer{
		Network:      net,
		Architecture: parseSeedArchitecture(hdr.Transformer.Architecture),
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

func entitySeedInfoFromHeader(path string, hdr *EntityHeader) *EntitySeedInfo {
	info := &EntitySeedInfo{
		Path:       path,
		ModelID:    EntityModelIDFromPath(path),
		StoredSeed: hdr.Network.InitSeed,
	}
	if hdr.HasTransformer() {
		info.HasTransformer = true
		info.Architecture = hdr.Transformer.Architecture
		info.HiddenSize = hdr.Transformer.HiddenSize
		info.VocabSize = hdr.Transformer.VocabSize
		info.NumLayers = 0
		if hdr.Transformer.Dims != nil {
			info.NumLayers = hdr.Transformer.Dims.NumLayers
		}
		info.WeightDType = ParseDType(hdr.Transformer.WeightDType)
		if info.WeightDType == 0 {
			info.WeightDType = entityTransformerWeightDType(hdr)
		}
	}
	return info
}

func parseSeedArchitecture(s string) HFArchitectureKind {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "llama_style_decoder", "llama", "decoder":
		return HFArchLlamaStyleDecoder
	default:
		if strings.Contains(strings.ToLower(s), "llama") {
			return HFArchLlamaStyleDecoder
		}
		return HFArchUnknown
	}
}

func readEntityHeaderFile(path string) (*EntityHeader, error) {
	_, hdr, err := readEntityHeaderBytesFile(path)
	return hdr, err
}

func readEntityHeaderBytesFile(path string) ([]byte, *EntityHeader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	fixed := make([]byte, 20)
	if _, err := f.Read(fixed); err != nil {
		return nil, nil, fmt.Errorf("entity header: %w", err)
	}
	if string(fixed[:8]) != entityMagic {
		return nil, nil, fmt.Errorf("invalid entity magic")
	}
	headerLen := binary.LittleEndian.Uint64(fixed[12:20])
	if headerLen > entityHeaderMaxSize {
		return nil, nil, fmt.Errorf("entity header too large")
	}
	prefix := make([]byte, 20+headerLen)
	copy(prefix, fixed)
	if _, err := f.Read(prefix[20:]); err != nil {
		return nil, nil, fmt.Errorf("entity header json: %w", err)
	}
	hdr, err := ParseEntityHeader(prefix)
	if err != nil {
		return nil, nil, err
	}
	return prefix, hdr, nil
}

// ForwardEntityNetwork runs one forward on decoder stack; returns output + hash.
func ForwardEntityNetwork(et *EntityTransformer, batch, dim int) ([]float32, uint64, error) {
	if et == nil || et.Network == nil {
		return nil, 0, fmt.Errorf("nil entity network")
	}
	in := seedDemoForwardInput(batch * dim)
	tensor := NewTensorFromSlice(in, batch, dim)
	out, _, _ := ForwardPolymorphic(et.Network, tensor)
	if out == nil {
		return nil, 0, fmt.Errorf("forward returned nil")
	}
	return out.Data, seedOutputHash(out.Data), nil
}

func seedDemoForwardInput(n int) []float32 {
	in := make([]float32, n)
	for i := range in {
		in[i] = 0.01 * float32(i%11)
	}
	return in
}

func seedOutputHash(out []float32) uint64 {
	h := fnv.New64a()
	var buf [4]byte
	for _, v := range out {
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
		_, _ = h.Write(buf[:])
	}
	return h.Sum64()
}

// MarshalEntitySeedManifest writes a tiny JSON topology+layer-seed manifest.
func MarshalEntitySeedManifest(ws *WeightSeedFile) ([]byte, error) {
	return json.MarshalIndent(ws, "", "  ")
}
