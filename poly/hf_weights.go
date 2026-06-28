package poly

import (
	"fmt"
	"strings"
)

// HFWeightLayerIndex returns the transformer block index for HuggingFace-style keys such as
// model.layers.N.* or transformer.h.N.* (GPT-2-style). Keys without a block index (embeddings, final norm, lm_head) return ok=false.
func HFWeightLayerIndex(key string) (idx int, ok bool) {
	parts := strings.Split(key, ".")
	for i := 0; i < len(parts)-1; i++ {
		if parts[i] == "layers" || parts[i] == "h" {
			var n int
			if _, err := fmt.Sscanf(parts[i+1], "%d", &n); err == nil {
				return n, true
			}
		}
	}
	return 0, false
}

// HFWeightIsGlobal reports whether key is not tied to a numbered transformer block (embedding, head, final norm, etc.).
func HFWeightIsGlobal(key string) bool {
	_, ok := HFWeightLayerIndex(key)
	return !ok
}

// HFWeightMatchesLayer reports whether key belongs to block layerIdx (model.layers.{layerIdx}.…).
func HFWeightMatchesLayer(key string, layerIdx int) bool {
	n, ok := HFWeightLayerIndex(key)
	return ok && n == layerIdx
}

// MaxHFWeightLayerIndex scans tensor names and returns the largest block index found, or -1 if none.
func MaxHFWeightLayerIndex(names []string) int {
	max := -1
	for _, n := range names {
		if li, ok := HFWeightLayerIndex(n); ok && li > max {
			max = li
		}
	}
	return max
}

// MaxHFWeightLayerIndexInSafetensorsFiles reads only safetensors headers and returns the largest
// transformer block index across all given files. Returns -1 if none found.
func MaxHFWeightLayerIndexInSafetensorsFiles(paths []string) int {
	max := -1
	for _, p := range paths {
		names, err := SafetensorsTensorNames(p)
		if err != nil {
			continue
		}
		if m := MaxHFWeightLayerIndex(names); m > max {
			max = m
		}
	}
	return max
}

// BuildLayerShardIndex maps each transformer block index to the safetensors files that contain its tensors.
func BuildLayerShardIndex(safetensorFiles []string, numLayers int) [][]string {
	layerFiles := make([][]string, numLayers)
	if numLayers <= 0 {
		return layerFiles
	}
	for _, sf := range safetensorFiles {
		names, err := SafetensorsTensorNames(sf)
		if err != nil {
			for li := 0; li < numLayers; li++ {
				layerFiles[li] = append(layerFiles[li], sf)
			}
			continue
		}
		seen := make(map[int]struct{})
		for _, n := range names {
			if li, ok := HFWeightLayerIndex(n); ok && li >= 0 && li < numLayers {
				seen[li] = struct{}{}
			}
		}
		for li := range seen {
			layerFiles[li] = append(layerFiles[li], sf)
		}
	}
	for li := 0; li < numLayers; li++ {
		if len(layerFiles[li]) == 0 {
			layerFiles[li] = append(layerFiles[li], safetensorFiles...)
		}
	}
	return layerFiles
}
