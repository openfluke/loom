package contextsuite

import (
	"fmt"
	"strings"

	"github.com/openfluke/loom/poly"
)

type loadedModel struct {
	spec         ModelSpec
	tr           *poly.Transformer[float32]
	tk           *poly.Tokenizer
	eosTokens    []int
	entityFile   *poly.EntityFile
	entityBundle *poly.EntityTransformer
	numLayers    int
	hiddenSize   int
	storedDType  poly.DType
	useGPU       bool
}

func loadModel(spec ModelSpec, prof ExecProfile) (*loadedModel, error) {
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
	rmsNormEps := et.Dims.RMSNormEps

	tr := poly.BuildTransformerFromEntity[float32](et, template)
	tr.SetRMSNormEps(rmsNormEps)

	eosTokens := poly.LoadEOSTokenIDsFromConfigPath(snap + "/config.json")

	gpuWeightDType := poly.EntityGPUWeightDType(storedDType, prof.UseGPU)
	isQwen := strings.Contains(strings.ToLower(spec.RepoID), "qwen")
	useBitNet := isBitNetModel(spec.RepoID) || storedDType == poly.DTypeTernary

	cfg := inferenceConfig{
		useGPU:            prof.UseGPU,
		useTiling:         prof.UseGPU || prof.MultiCore,
		multiCore:         prof.MultiCore,
		weightDType:       gpuWeightDType,
		numLayers:         numLayers,
		hiddenSize:        hiddenSize,
		isQwen:            isQwen,
		useBitNetCPU:      useBitNet && !prof.UseGPU,
		fromEntity:        true,
		entityFile:        ef,
		entityBundle:      et,
		sequentialGPULoad: prof.UseGPU,
	}

	useGPU, err := setupInference(tr, cfg)
	if err != nil {
		ef.Close()
		return nil, err
	}
	if prof.UseGPU && !useGPU {
		ef.Close()
		return nil, fmt.Errorf("GPU requested but unavailable")
	}

	return &loadedModel{
		spec:         spec,
		tr:           tr,
		tk:           tk,
		eosTokens:    eosTokens,
		entityFile:   ef,
		entityBundle: et,
		numLayers:    numLayers,
		hiddenSize:   hiddenSize,
		storedDType:  storedDType,
		useGPU:       useGPU,
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
	m.entityBundle = nil
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

type inferenceConfig struct {
	useGPU            bool
	useTiling         bool
	multiCore         bool
	weightDType       poly.DType
	numLayers         int
	hiddenSize        int
	isQwen            bool
	useBitNetCPU      bool
	fromEntity        bool
	entityFile        *poly.EntityFile
	entityBundle      *poly.EntityTransformer
	sequentialGPULoad bool
}

func setupInference(tr *poly.Transformer[float32], cfg inferenceConfig) (bool, error) {
	if tr == nil || tr.Network == nil {
		return false, fmt.Errorf("nil transformer")
	}

	normalizeConfig(&cfg)

	for i := range tr.Network.Layers {
		tr.Network.Layers[i].MaxSeqLen = maxSeqLen
	}
	if cfg.useTiling {
		tr.EnableTiling(-1)
	}

	useGPU := cfg.useGPU
	if useGPU {
		if err := tr.Network.InitWGPU(); err != nil {
			return false, fmt.Errorf("InitWGPU: %w", err)
		}
		applyTiling(tr.Network, true, cfg.useTiling, cfg.multiCore)
		if cfg.fromEntity && cfg.weightDType == poly.DTypeTernary {
			tr.Network.UseExactDType = true
		}
		if cfg.sequentialGPULoad {
			for li := 0; li < cfg.numLayers; li++ {
				if err := loadEntityBlock(tr, cfg, li); err != nil {
					return false, err
				}
				base := li * 4
				for j := 0; j < 4; j++ {
					idx := base + j
					layer := &tr.Network.Layers[idx]
					if layer.Type == poly.LayerRMSNorm {
						layer.DType = poly.DTypeFloat32
					} else {
						layer.DType = cfg.weightDType
					}
					if err := layer.SyncToGPU(); err != nil {
						return false, fmt.Errorf("GPU sync block %d layer %d: %w", li, j, err)
					}
				}
				for j := 0; j < 4; j++ {
					(&tr.Network.Layers[base+j]).ReleaseInferenceHostWeights()
				}
				poly.ReleaseInferenceTransientMemory()
			}
		} else {
			if err := loadAllEntityBlocks(tr, cfg); err != nil {
				return false, err
			}
			for i := range tr.Network.Layers {
				if tr.Network.Layers[i].Type == poly.LayerRMSNorm {
					tr.Network.Layers[i].DType = poly.DTypeFloat32
				} else {
					tr.Network.Layers[i].DType = cfg.weightDType
				}
				if err := (&tr.Network.Layers[i]).SyncToGPU(); err != nil {
					return false, fmt.Errorf("GPU sync layer %d: %w", i, err)
				}
			}
			for i := range tr.Network.Layers {
				tr.Network.Layers[i].ReleaseInferenceHostWeights()
			}
			poly.ReleaseInferenceTransientMemory()
		}
		if err := tr.SyncGlobalWeightsToGPUSequential(); err != nil {
			return false, fmt.Errorf("global GPU sync: %w", err)
		}
		_, _ = tr.ForwardTokenIDsWGPU([]uint32{0}, nil, true, true)
		tr.Reset()
		tr.ReleaseInferenceHostWeights()
		poly.AggressiveReleaseMemoryToOS()
	}

	if !useGPU {
		applyTiling(tr.Network, false, cfg.useTiling, cfg.multiCore)
		if cfg.useBitNetCPU || (cfg.fromEntity && cfg.weightDType == poly.DTypeTernary) {
			tr.Network.UseExactDType = true
		}
		if err := loadAllEntityBlocks(tr, cfg); err != nil {
			return false, err
		}
		tr.SyncInferenceCPU()
		poly.AggressiveReleaseMemoryToOS()
	}
	return useGPU, nil
}

func normalizeConfig(cfg *inferenceConfig) {
	if cfg.useGPU && cfg.useTiling && cfg.hiddenSize >= 1536 {
		cfg.useTiling = false
		cfg.multiCore = false
	}
}

func applyTiling(net *poly.VolumetricNetwork, useGPU, useTiling, multiCore bool) {
	if useGPU {
		net.EnableMultiCoreTiling = useTiling && multiCore
	} else {
		net.EnableMultiCoreTiling = useTiling
	}
}

func loadEntityBlock(tr *poly.Transformer[float32], cfg inferenceConfig, blockIndex int) error {
	if cfg.entityFile == nil {
		return nil
	}
	base := blockIndex * 4
	indices := []int{base, base + 1, base + 2, base + 3}
	if err := cfg.entityFile.LoadNetworkLayerWeights(tr.Network, indices); err != nil {
		return fmt.Errorf("load block %d: %w", blockIndex, err)
	}
	if cfg.entityBundle != nil {
		poly.PrepareEntityTransformerLayerIndices(cfg.entityBundle, indices)
	}
	poly.ReleaseInferenceTransientMemory()
	return nil
}

func loadAllEntityBlocks(tr *poly.Transformer[float32], cfg inferenceConfig) error {
	for li := 0; li < cfg.numLayers; li++ {
		if err := loadEntityBlock(tr, cfg, li); err != nil {
			return err
		}
	}
	return nil
}
