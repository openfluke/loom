package poly

import (
	"runtime"
	"runtime/debug"
)

// SharesFloat32Backing reports whether candidate shares the same underlying array
// as any of the keep slices (pointer equality on first element).
func SharesFloat32Backing(candidate []float32, keep ...[]float32) bool {
	if len(candidate) == 0 {
		return false
	}
	for _, k := range keep {
		if len(k) == 0 {
			continue
		}
		if &candidate[0] == &k[0] {
			return true
		}
	}
	return false
}

// ReleaseTransientSafetensorMap deletes map entries whose slices do not share
// backing with keep slices (e.g. embeddings after PrefixWeightMapper.MapWeights).
func ReleaseTransientSafetensorMap(tensors map[string][]float32, keep ...[]float32) {
	if len(tensors) == 0 {
		return
	}
	for name, data := range tensors {
		if SharesFloat32Backing(data, keep...) {
			continue
		}
		tensors[name] = nil
		delete(tensors, name)
	}
	runtime.GC()
	debug.FreeOSMemory()
}

// ReleaseTransientHFStoredMap drops raw safetensors tensor payloads (BitNet / staged loads).
func ReleaseTransientHFStoredMap(tensors map[string]HFStoredTensor) {
	if len(tensors) == 0 {
		return
	}
	for k := range tensors {
		delete(tensors, k)
	}
	runtime.GC()
	debug.FreeOSMemory()
}

// CloneMappedGlobalWeights copies slices returned from PrefixWeightMapper.MapWeights
// so a temporary global tensor map can be freed before loading layers sequentially.
func CloneMappedGlobalWeights(embeddings, lmHead, finalNorm []float32) (embOut, lmOut, fnOut []float32) {
	tied := len(embeddings) > 0 && len(lmHead) == len(embeddings) &&
		&embeddings[0] == &lmHead[0]
	embOut = append([]float32(nil), embeddings...)
	if tied {
		lmOut = embOut
	} else if lmHead != nil {
		lmOut = append([]float32(nil), lmHead...)
	}
	if finalNorm != nil {
		fnOut = append([]float32(nil), finalNorm...)
	}
	return embOut, lmOut, fnOut
}
