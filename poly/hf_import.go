package poly

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// HF import compatibility layer — config.json + *.safetensors → VolumetricNetwork (+ optional .entity).
//
// SafeTensors is the HuggingFace *import* lane; ENTITY is the native *export* lane.
// Today import logic also lives in lucy.go (BitNet block load, GPU paths). This file
// centralizes the reusable “read HF folder → wire Loom decoder” path so tools and
// ConvertHFToEntity share one API. Spec: docs/safetensors_import.md

// HFArchitectureKind classifies a checkpoint for topology + tensor routing.
type HFArchitectureKind int

const (
	HFArchUnknown HFArchitectureKind = iota
	HFArchLlamaStyleDecoder // Llama, Mistral, Qwen, SmolLM, … — InitHFDecoderBlocks
)

func (k HFArchitectureKind) String() string {
	switch k {
	case HFArchLlamaStyleDecoder:
		return "llama_style_decoder"
	default:
		return "unknown"
	}
}

// MorphHFDecoderWeights applies baked checkpoint quant to MHA/SwiGLU only.
// RMSNorm must stay FP32 — INT4 norm weights corrupt the forward pass and GPU upload.
func MorphHFDecoderWeights(net *VolumetricNetwork, dt DType) {
	if net == nil || dt == 0 || dt == DTypeFloat32 {
		return
	}
	for i := range net.Layers {
		l := &net.Layers[i]
		if l.Type == LayerRMSNorm {
			l.DType = DTypeFloat32
			continue
		}
		l.DType = dt
		if l.WeightStore != nil {
			l.WeightStore.Morph(dt)
		}
	}
}

// HFImportOptions configures ImportHFCheckpointDir.
type HFImportOptions struct {
	// WeightDType morphs each layer after load (default: leave FP32 master).
	WeightDType DType
	// RequireNumHiddenLayers fails when config lacks num_hidden_layers and tensor scan finds no blocks.
	RequireNumHiddenLayers bool
}

// HFImportResult is a loaded HF causal-LM decoder checkpoint.
type HFImportResult struct {
	Architecture    HFArchitectureKind
	Config          map[string]interface{}
	Dims            HFDecoderDims
	Network         *VolumetricNetwork
	Embeddings      []float32
	LMHead          []float32
	FinalNorm       []float32
	HasFinalNorm    bool
	TensorsLoaded   int
	TensorsSkipped  int
	SafetensorFiles []string
}

// DetectHFArchitecture reads model_type / architectures[] from an unmarshaled config.json.
func DetectHFArchitecture(config map[string]interface{}) HFArchitectureKind {
	modelType := strings.ToLower(HFConfigStringDefault(config, "model_type", ""))
	for _, arch := range hfConfigArchitectureStrings(config) {
		a := strings.ToLower(arch)
		switch {
		case strings.Contains(a, "llama"),
			strings.Contains(a, "mistral"),
			strings.Contains(a, "qwen"),
			strings.Contains(a, "smollm"),
			strings.Contains(a, "gemma"),
			strings.Contains(a, "phi"):
			return HFArchLlamaStyleDecoder
		case strings.Contains(modelType, "llama"),
			strings.Contains(modelType, "mistral"),
			strings.Contains(modelType, "qwen"),
			strings.Contains(modelType, "smol"),
			strings.Contains(modelType, "gemma"),
			strings.Contains(modelType, "phi"):
			return HFArchLlamaStyleDecoder
		}
	}
	switch modelType {
	case "llama", "mistral", "qwen2", "qwen3", "smollm", "gemma", "gemma2", "phi", "phi3":
		return HFArchLlamaStyleDecoder
	}
	if _, ok := HFConfigInt(config, "num_hidden_layers"); ok {
		if _, ok2 := HFConfigInt(config, "hidden_size"); ok2 {
			if _, ok3 := HFConfigInt(config, "num_attention_heads"); ok3 {
				return HFArchLlamaStyleDecoder
			}
		}
	}
	return HFArchUnknown
}

func hfConfigArchitectureStrings(config map[string]interface{}) []string {
	raw, ok := config["architectures"]
	if !ok {
		return nil
	}
	list, ok := raw.([]interface{})
	if !ok {
		return nil
	}
	out := make([]string, 0, len(list))
	for _, item := range list {
		if s, ok := item.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

// ParseHFDecoderDims extracts decoder dimensions from config.json (+ optional safetensors for layer count).
func ParseHFDecoderDims(config map[string]interface{}, safetensorFiles []string) (HFDecoderDims, error) {
	numHeads, ok := HFConfigInt(config, "num_attention_heads")
	if !ok || numHeads <= 0 {
		return HFDecoderDims{}, fmt.Errorf("config missing num_attention_heads")
	}
	numKVHeads := numHeads
	if v, ok := HFConfigInt(config, "num_key_value_heads"); ok && v > 0 {
		numKVHeads = v
	}
	hiddenSize, ok := HFConfigInt(config, "hidden_size")
	if !ok || hiddenSize <= 0 {
		return HFDecoderDims{}, fmt.Errorf("config missing hidden_size")
	}
	intermediateSize, ok := HFConfigInt(config, "intermediate_size")
	if !ok || intermediateSize <= 0 {
		return HFDecoderDims{}, fmt.Errorf("config missing intermediate_size")
	}
	numLayers, ok := HFConfigInt(config, "num_hidden_layers")
	if !ok || numLayers <= 0 {
		maxLi := MaxHFWeightLayerIndexInSafetensorsFiles(safetensorFiles)
		if maxLi < 0 {
			return HFDecoderDims{}, fmt.Errorf("could not determine num_hidden_layers")
		}
		numLayers = maxLi + 1
	}
	headDim := hiddenSize / numHeads
	if v, ok := HFConfigInt(config, "head_dim"); ok && v > 0 {
		headDim = v
	}
	activation := ActivationSilu
	if strings.EqualFold(HFConfigStringDefault(config, "hidden_act", ""), "relu2") {
		activation = ActivationReLU2
	}
	return HFDecoderDims{
		NumLayers:        numLayers,
		HiddenSize:       hiddenSize,
		NumHeads:         numHeads,
		NumKVHeads:       numKVHeads,
		HeadDim:          headDim,
		QueryDim:         numHeads * headDim,
		KVDim:            numKVHeads * headDim,
		IntermediateSize: intermediateSize,
		RMSNormEps:       HFConfigFloat64Default(config, "rms_norm_eps", 1e-6),
		RoPEFreqBase:     HFConfigFloat64Default(config, "rope_theta", 10000.0),
		Activation:       activation,
	}, nil
}

// ImportHFCheckpointDir loads config.json + *.safetensors from a HuggingFace snapshot directory.
//
// Supported today: Llama-style causal decoder stacks (see DetectHFArchitecture).
// Tensor name routing uses prefix_safetensor.go — not every HF model family yet.
func ImportHFCheckpointDir(modelDir string, opts HFImportOptions) (*HFImportResult, error) {
	configPath := filepath.Join(modelDir, "config.json")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("read config.json: %w", err)
	}
	var config map[string]interface{}
	if err := json.Unmarshal(configData, &config); err != nil {
		return nil, fmt.Errorf("parse config.json: %w", err)
	}

	safetensorFiles, err := filepath.Glob(filepath.Join(modelDir, "*.safetensors"))
	if err != nil {
		return nil, err
	}
	if len(safetensorFiles) == 0 {
		return nil, fmt.Errorf("no .safetensors in %s", modelDir)
	}

	kind := DetectHFArchitecture(config)
	if kind == HFArchUnknown {
		return nil, fmt.Errorf("unsupported HF architecture (model_type=%q); see docs/safetensors_import.md",
			HFConfigStringDefault(config, "model_type", ""))
	}

	dims, err := ParseHFDecoderDims(config, safetensorFiles)
	if err != nil {
		if opts.RequireNumHiddenLayers {
			return nil, err
		}
		return nil, fmt.Errorf("parse decoder dims: %w", err)
	}

	allTensors := make(map[string][]float32)
	skipped := 0
	for _, f := range safetensorFiles {
		part, err := LoadSafetensors(f)
		if err != nil {
			return nil, fmt.Errorf("load %s: %w", filepath.Base(f), err)
		}
		for k, v := range part {
			allTensors[k] = v
		}
	}
	headerCount := 0
	for _, f := range safetensorFiles {
		names, err := SafetensorsTensorNames(f)
		if err == nil {
			headerCount += len(names)
		}
	}
	if headerCount > len(allTensors) {
		skipped = headerCount - len(allTensors)
	}

	mapper := NewPrefixWeightMapper()
	embeddings, lmHead, finalNorm, hasFinalNorm := mapper.MapWeights(allTensors)

	net := NewVolumetricNetwork(1, 1, 1, dims.NumLayers*4)
	InitHFDecoderBlocks(net, dims)
	if err := LoadWithPrefixes(net, allTensors); err != nil {
		return nil, err
	}

	MorphHFDecoderWeights(net, opts.WeightDType)

	return &HFImportResult{
		Architecture:    kind,
		Config:          config,
		Dims:            dims,
		Network:         net,
		Embeddings:      embeddings,
		LMHead:          lmHead,
		FinalNorm:       finalNorm,
		HasFinalNorm:    hasFinalNorm,
		TensorsLoaded:   len(allTensors),
		TensorsSkipped:  skipped,
		SafetensorFiles: safetensorFiles,
	}, nil
}

// ImportHFToEntity loads a HuggingFace snapshot and writes a universal-transformer .entity checkpoint.
func ImportHFToEntity(modelDir, entityPath string, opts HFImportOptions) error {
	res, err := ImportHFCheckpointDir(modelDir, opts)
	if err != nil {
		return err
	}
	et := NewEntityTransformer(
		res.Network,
		res.Architecture,
		res.Dims,
		res.Embeddings,
		res.LMHead,
		res.FinalNorm,
		res.HasFinalNorm,
	)
	return SaveEntityTransformer(entityPath, et)
}
