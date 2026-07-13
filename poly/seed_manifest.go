package poly

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// LayerSeedEntry is one seedrng layer slot: DeriveLayerSeed(init, index, path).
type LayerSeedEntry struct {
	Path      string `json:"path"`
	Index     int    `json:"index"`
	LayerSeed uint64 `json:"layer_seed"`
}

// WeightSeedFile is a tiny seed manifest (init_seed + per-layer seeds). No weight bytes.
type WeightSeedFile struct {
	ModelID      string                     `json:"model_id"`
	InitSeed     uint64                     `json:"init_seed"`
	Architecture string                     `json:"architecture,omitempty"`
	HiddenSize   int                        `json:"hidden_size,omitempty"`
	VocabSize    int                        `json:"vocab_size,omitempty"`
	LMHeadTied   bool                       `json:"lm_head_tied,omitempty"`
	HasFinalNorm bool                       `json:"has_final_norm,omitempty"`
	WeightDType  string                     `json:"weight_dtype,omitempty"`
	Dims         *EntityTransformerDimsSpec `json:"dims,omitempty"`
	Layers       []LayerSeedEntry           `json:"layers,omitempty"`
	Globals      []LayerSeedEntry           `json:"globals,omitempty"`
	SeedFP       uint64                     `json:"seed_fingerprint"`
	Format       string                     `json:"format"`
}

const weightSeedFormat = "loom-seed-manifest-v3"

// BuildWeightSeedFile builds a tiny manifest from topology + init seed.
func BuildWeightSeedFile(modelID string, et *EntityTransformer) (*WeightSeedFile, error) {
	if et == nil || et.Network == nil {
		return nil, fmt.Errorf("seed manifest: nil entity")
	}
	initSeed := et.Network.InitSeed
	if initSeed == 0 && modelID != "" {
		initSeed = EntityTransformerRecipeSeed(modelID, et.Architecture, et.Dims, et.WeightDType)
	}
	ws := &WeightSeedFile{
		ModelID:      modelID,
		InitSeed:     initSeed,
		Architecture: et.Architecture.String(),
		HiddenSize:   et.HiddenSize,
		VocabSize:    et.VocabSize,
		LMHeadTied:   et.LMHeadTied,
		HasFinalNorm: et.HasFinalNorm,
		WeightDType:  et.WeightDType.String(),
		Dims:         entityDimsToSpec(et.Dims),
		Format:       weightSeedFormat,
	}
	ws.Layers, ws.Globals = deriveWeightSeedEntries(et, initSeed)
	seedET, err := RebuildEntityFromWeightSeedFile(ws)
	if err != nil {
		return nil, err
	}
	ws.SeedFP = EntityTransformerFingerprint(seedET)
	return ws, nil
}

func entityDimsToSpec(d HFDecoderDims) *EntityTransformerDimsSpec {
	return &EntityTransformerDimsSpec{
		NumLayers:        d.NumLayers,
		NumHeads:         d.NumHeads,
		NumKVHeads:       d.NumKVHeads,
		HeadDim:          d.HeadDim,
		QueryDim:         d.QueryDim,
		KVDim:            d.KVDim,
		IntermediateSize: d.IntermediateSize,
		RMSNormEps:       d.RMSNormEps,
		RoPEFreqBase:     d.RoPEFreqBase,
		Activation:       d.Activation.String(),
	}
}

func deriveWeightSeedEntries(et *EntityTransformer, initSeed uint64) (layers, globals []LayerSeedEntry) {
	walkSeedLayers(et.Network, func(l *VolumetricLayer, idx int, path string) {
		if seedLayerWeightCount(l) <= 0 {
			return
		}
		layers = append(layers, LayerSeedEntry{
			Path:      path,
			Index:     idx,
			LayerSeed: DeriveLayerSeed(initSeed, idx, path),
		})
	})
	for _, g := range []struct {
		path string
		ok   bool
	}{
		{"transformer.embeddings", len(et.Embeddings) > 0},
		{"transformer.lm_head", !et.LMHeadTied && len(et.LMHead) > 0},
		{"transformer.final_norm", et.HasFinalNorm && len(et.FinalNorm) > 0},
	} {
		if !g.ok {
			continue
		}
		globals = append(globals, LayerSeedEntry{
			Path:      g.path,
			Index:     -1,
			LayerSeed: DeriveLayerSeed(initSeed, 0, g.path),
		})
	}
	return layers, globals
}

// RebuildEntityFromWeightSeedFile reconstructs via He-init from manifest seeds.
func RebuildEntityFromWeightSeedFile(ws *WeightSeedFile) (*EntityTransformer, error) {
	if ws == nil {
		return nil, fmt.Errorf("rebuild: nil manifest")
	}
	dims := entityTransformerDimsFromSpec(ws.Dims)
	if ws.HiddenSize > 0 {
		dims.HiddenSize = ws.HiddenSize
	}
	wdt := DTypeFloat32
	if ws.WeightDType != "" {
		wdt = ParseDType(ws.WeightDType)
	}
	et := BuildSeededEntityTransformer(ws.InitSeed, dims, ws.VocabSize, wdt, ws.LMHeadTied, ws.HasFinalNorm)
	if et == nil {
		return nil, fmt.Errorf("build from init_seed 0x%x", ws.InitSeed)
	}
	if ws.SeedFP != 0 {
		got := EntityTransformerFingerprint(et)
		if got != ws.SeedFP {
			return nil, fmt.Errorf("seed fingerprint mismatch: got 0x%x want 0x%x", got, ws.SeedFP)
		}
	}
	return et, nil
}

// SaveWeightSeedFile writes a tiny .wseed JSON.
func SaveWeightSeedFile(path string, ws *WeightSeedFile) error {
	if ws == nil {
		return fmt.Errorf("save seed manifest: nil")
	}
	if ws.Format == "" {
		ws.Format = weightSeedFormat
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(ws, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// LoadWeightSeedFile reads a .wseed manifest; rejects bloated legacy weight dumps.
func LoadWeightSeedFile(path string) (*WeightSeedFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) > 512*1024 {
		return nil, fmt.Errorf("bloated .wseed (%d bytes) — delete and re-extract", len(data))
	}
	var probe struct {
		WeightSeeds    []uint64 `json:"weight_seeds"`
		WeightSeedsB64 string   `json:"weight_seeds_b64"`
	}
	_ = json.Unmarshal(data, &probe)
	if len(probe.WeightSeeds) > 0 || probe.WeightSeedsB64 != "" {
		return nil, fmt.Errorf("legacy weight-blob .wseed — delete %s", filepath.Base(path))
	}
	var ws WeightSeedFile
	if err := json.Unmarshal(data, &ws); err != nil {
		return nil, err
	}
	return &ws, nil
}

// ExtractWeightSeedFileFromEntityPath writes manifest from .entity topology only.
func ExtractWeightSeedFileFromEntityPath(entityPath, wseedPath, modelID string) (*WeightSeedFile, error) {
	if modelID == "" {
		modelID = EntityModelIDFromPath(entityPath)
	}
	et, initSeed, err := LoadEntityTopologySeeded(entityPath, modelID, 0)
	if err != nil {
		return nil, err
	}
	et.Network.InitSeed = initSeed
	ws, err := BuildWeightSeedFile(modelID, et)
	if err != nil {
		return nil, err
	}
	if wseedPath != "" {
		if err := SaveWeightSeedFile(wseedPath, ws); err != nil {
			return nil, err
		}
	}
	return ws, nil
}

// BuildWeightSeedFileFromPreset builds manifest without an .entity file.
func BuildWeightSeedFileFromPreset(
	modelID string,
	initSeed uint64,
	dims HFDecoderDims,
	vocab int,
	wdt DType,
	tied, hasNorm bool,
) (*WeightSeedFile, error) {
	et := BuildSeededEntityTransformer(initSeed, dims, vocab, wdt, tied, hasNorm)
	if et == nil {
		return nil, fmt.Errorf("build preset skeleton")
	}
	return BuildWeightSeedFile(modelID, et)
}

// TotalLayerSeeds returns layers + globals count.
func TotalLayerSeeds(ws *WeightSeedFile) int {
	return len(ws.Layers) + len(ws.Globals)
}

// LayerSeedCount is an alias for TotalLayerSeeds.
func LayerSeedCount(ws *WeightSeedFile) int {
	return TotalLayerSeeds(ws)
}
