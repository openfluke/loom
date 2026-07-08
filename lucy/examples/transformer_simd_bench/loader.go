package tfsimdbench

import (
	"fmt"
	"strings"

	"github.com/openfluke/loom/poly"
)

// loadedModel is a CPU-resident transformer plus its tokenizer, loaded once and
// re-used across every BenchProfile (only flags are flipped between profiles).
type loadedModel struct {
	spec       ModelSpec
	tr         *poly.Transformer[float32]
	tk         *poly.Tokenizer
	eosTokens  []int
	entityFile *poly.EntityFile
	numLayers  int
	hiddenSize int
	dtype      poly.DType
	ternary    bool // packed-ternary CPU path (SIMD forward does not apply)
}

// loadModelCPU loads a .entity checkpoint for CPU inference. All decoder blocks
// are materialized on the host and never released, so the model can be re-run
// under every execution profile without reloading.
func loadModelCPU(spec ModelSpec) (*loadedModel, error) {
	path := entityPath(spec.RepoID)
	snap, err := resolveSnapshotDir(spec.RepoID)
	if err != nil {
		return nil, err
	}

	tk, err := poly.LoadTokenizer(snap + "/tokenizer.json")
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %w", err)
	}

	ef, err := poly.OpenEntityFile(path)
	if err != nil {
		return nil, fmt.Errorf("OpenEntityFile: %w", err)
	}

	et, err := ef.LoadEntityTransformerTopology()
	if err != nil {
		ef.Close()
		return nil, fmt.Errorf("LoadEntityTransformerTopology: %w", err)
	}

	storedDType := et.WeightDType
	if storedDType == 0 {
		storedDType = poly.DTypeFloat32
	}
	poly.PrepareEntityTransformerInference(et)

	template := poly.TemplateForHFModelID(spec.RepoID)
	numLayers := et.Dims.NumLayers
	if numLayers <= 0 && et.Network != nil {
		numLayers = len(et.Network.Layers) / 4
	}
	hiddenSize := et.HiddenSize

	tr := poly.BuildTransformerFromEntity[float32](et, template)
	tr.SetRMSNormEps(et.Dims.RMSNormEps)

	eosTokens := poly.LoadEOSTokenIDsFromConfigPath(snap + "/config.json")

	ternary := isBitNetModel(spec.RepoID) || storedDType == poly.DTypeTernary

	net := tr.Network
	net.UseGPU = false
	for i := range net.Layers {
		net.Layers[i].MaxSeqLen = maxSeqLen
	}
	if ternary {
		net.UseExactDType = true
	}

	// Materialize every decoder block on the host.
	for li := 0; li < numLayers; li++ {
		base := li * 4
		indices := []int{base, base + 1, base + 2, base + 3}
		if err := ef.LoadNetworkLayerWeights(net, indices); err != nil {
			ef.Close()
			return nil, fmt.Errorf("load block %d: %w", li, err)
		}
		poly.PrepareEntityTransformerLayerIndices(et, indices)
		poly.ReleaseInferenceTransientMemory()
	}
	tr.SyncInferenceCPU()
	poly.AggressiveReleaseMemoryToOS()

	return &loadedModel{
		spec:       spec,
		tr:         tr,
		tk:         tk,
		eosTokens:  eosTokens,
		entityFile: ef,
		numLayers:  numLayers,
		hiddenSize: hiddenSize,
		dtype:      storedDType,
		ternary:    ternary,
	}, nil
}

func (m *loadedModel) close() {
	if m.tr != nil && m.tr.Network != nil {
		m.tr.Network.DestroyWGPU()
	}
	if m.entityFile != nil {
		m.entityFile.Close()
	}
	m.tr = nil
	m.entityFile = nil
	poly.ReleaseInferenceTransientMemory()
	poly.AggressiveReleaseMemoryToOS()
}

func (m *loadedModel) encode(text string) []uint32 {
	addSpecial := strings.Contains(strings.ToLower(m.spec.RepoID), "microsoft/bitnet-b1.58-2b-4t")
	return m.tk.Encode(text, addSpecial)
}

func (m *loadedModel) decode(tokens []uint32) string {
	return m.tk.Decode(tokens, false)
}
