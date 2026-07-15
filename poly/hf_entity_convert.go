package poly

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
)

// HFEntityConvertProgress reports block-wise HF → .entity convert progress.
type HFEntityConvertProgress func(blockIndex, blockTotal int, detail string)

// ImportHFSaveEntityTransformerBlockwise converts a llama-style HF snapshot to .entity
// without holding the full FP32 decoder in RAM. Peak is roughly one transformer block
// plus globals and a streaming payload file (see docs/memory_history.md).
func ImportHFSaveEntityTransformerBlockwise(modelDir, entityPath string, weightDType DType) error {
	return ImportHFSaveEntityTransformerBlockwiseProgress(modelDir, entityPath, weightDType, nil)
}

// ImportHFSaveEntityTransformerBlockwiseProgress is like ImportHFSaveEntityTransformerBlockwise
// with optional progress callbacks (blockIndex is 1-based).
func ImportHFSaveEntityTransformerBlockwiseProgress(
	modelDir, entityPath string,
	weightDType DType,
	progress HFEntityConvertProgress,
) error {
	if weightDType == 0 {
		weightDType = DTypeFloat32
	}
	configPath := filepath.Join(modelDir, "config.json")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("read config.json: %w", err)
	}
	var config map[string]interface{}
	if err := json.Unmarshal(configData, &config); err != nil {
		return fmt.Errorf("parse config.json: %w", err)
	}

	safetensorFiles, err := filepath.Glob(filepath.Join(modelDir, "*.safetensors"))
	if err != nil {
		return err
	}
	if len(safetensorFiles) == 0 {
		return fmt.Errorf("no .safetensors in %s", modelDir)
	}
	kind := DetectHFArchitecture(config)
	if kind == HFArchUnknown {
		return fmt.Errorf("unsupported HF architecture (model_type=%q)", HFConfigStringDefault(config, "model_type", ""))
	}
	dims, err := ParseHFDecoderDims(config, safetensorFiles)
	if err != nil {
		return fmt.Errorf("parse decoder dims: %w", err)
	}

	if progress != nil {
		progress(0, dims.NumLayers, "loading globals…")
	}

	globalTensors := make(map[string][]float32)
	for _, f := range safetensorFiles {
		part, err := LoadSafetensorsSelective(f, HFWeightIsGlobal)
		if err != nil {
			return fmt.Errorf("load globals from %s: %w", filepath.Base(f), err)
		}
		for k, v := range part {
			globalTensors[k] = v
		}
	}
	mapper := NewPrefixWeightMapper()
	embeddings, lmHead, finalNorm, hasFinalNorm := mapper.MapWeights(globalTensors)
	ReleaseTransientSafetensorMap(globalTensors, embeddings, lmHead, finalNorm)

	net := NewVolumetricNetwork(1, 1, 1, dims.NumLayers*4)
	InitHFDecoderBlocks(net, dims)
	layerFiles := BuildLayerShardIndex(safetensorFiles, dims.NumLayers)

	payloadTmp, err := os.CreateTemp(filepath.Dir(entityPath), ".entity-payload-*")
	if err != nil {
		return err
	}
	payloadPath := payloadTmp.Name()
	defer os.Remove(payloadPath)
	acc := newEntityPayloadAcc(payloadTmp)
	var blobs []EntityWeightBlob

	// Match Lucy SaveEntityTransformer: entityLMHeadTied (not config tie_word_embeddings alone).
	lmHeadTied := entityLMHeadTied(embeddings, lmHead)
	trSpec := buildEntityTransformerSpecFromImport(kind, dims, embeddings, lmHead, finalNorm, hasFinalNorm, lmHeadTied, weightDType)
	collectEntityGlobalBlobAcc("embeddings", embeddings, acc, &blobs)
	if !lmHeadTied {
		collectEntityGlobalBlobAcc("lm_head", lmHead, acc, &blobs)
		if weightDType == DTypeInt4 {
			collectEntityLMHeadQ4Acc(lmHead, acc, &blobs)
		}
	} else if weightDType == DTypeInt4 {
		// Tied: still bake a Q4 logits matrix from embeddings for CPU ApplyLMHead.
		collectEntityLMHeadQ4Acc(embeddings, acc, &blobs)
	}
	if hasFinalNorm {
		collectEntityGlobalBlobAcc("final_norm", finalNorm, acc, &blobs)
	}
	embeddings, lmHead, finalNorm = nil, nil, nil
	runtime.GC()
	debug.FreeOSMemory()

	for li := 0; li < dims.NumLayers; li++ {
		layerMap := make(map[string][]float32)
		for _, sf := range layerFiles[li] {
			part, err := LoadSafetensorsSelective(sf, func(k string) bool {
				return HFWeightMatchesLayer(k, li)
			})
			if err != nil {
				_ = payloadTmp.Close()
				return fmt.Errorf("load block %d from %s: %w", li, filepath.Base(sf), err)
			}
			for k, v := range part {
				layerMap[k] = v
			}
		}
		if err := LoadWithPrefixes(net, layerMap); err != nil {
			_ = payloadTmp.Close()
			return err
		}
		ReleaseTransientSafetensorMap(layerMap)

		base := li * 4
		for j := 0; j < 4; j++ {
			idx := base + j
			if idx >= len(net.Layers) {
				break
			}
			path := fmt.Sprintf("layers.%d", idx)
			collectEntityWeightBlobsAcc(&net.Layers[idx], path, acc, &blobs, weightDType)
			releaseEntityConvertLayerWeights(&net.Layers[idx])
		}
		runtime.GC()
		if progress != nil {
			progress(li+1, dims.NumLayers, fmt.Sprintf("packed block %d/%d", li+1, dims.NumLayers))
		}
	}

	if err := payloadTmp.Close(); err != nil {
		return err
	}
	runtime.GC()
	debug.FreeOSMemory()

	if progress != nil {
		progress(dims.NumLayers, dims.NumLayers, "writing .entity…")
	}
	if err := os.MkdirAll(filepath.Dir(entityPath), 0o755); err != nil {
		return err
	}
	_ = os.Remove(entityPath)
	if err := writeEntityWireStreaming(entityPath, net, trSpec, blobs, payloadPath); err != nil {
		return err
	}
	runtime.GC()
	debug.FreeOSMemory()
	return nil
}
