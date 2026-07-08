package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"unsafe"

	"github.com/openfluke/loom/poly"
)

//export LoomRunNativeLayerMatrix
func LoomRunNativeLayerMatrix(configJSON *C.char) *C.char {
	cfg := poly.DefaultNativeMatrixConfig()
	if configJSON != nil {
		s := C.GoString(configJSON)
		if strings.TrimSpace(s) != "" {
			if err := json.Unmarshal([]byte(s), &cfg); err != nil {
				return errJSON(err.Error())
			}
		} else {
			fast, err := poly.SelectNativeMatrixCases("float32", nil)
			if err != nil {
				return errJSON(err.Error())
			}
			cfg.Cases = fast
			cfg.Epochs = 1
			cfg.BenchIters = 1
			cfg.SkipGPU = true
		}
	} else {
		fast, err := poly.SelectNativeMatrixCases("float32", nil)
		if err != nil {
			return errJSON(err.Error())
		}
		cfg.Cases = fast
		cfg.Epochs = 1
		cfg.BenchIters = 1
		cfg.SkipGPU = true
	}
	if err := poly.RunNativeLayerMatrixBuiltin(cfg); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status":"ok"}`)
}

//export LoomCalculateOptimalCNN1TileSizeForLayer
func LoomCalculateOptimalCNN1TileSizeForLayer(networkHandle C.longlong, layerIdx C.int, dtype C.int) C.int {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return 0
	}
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return 0
	}
	return C.int(poly.CalculateOptimalCNN1TileSizeForLayer(&n.Layers[int(layerIdx)], poly.DType(dtype)))
}

func layerTileSizeForNetwork(networkHandle C.longlong, layerIdx C.int, dtype C.int, fn func(*poly.VolumetricLayer, poly.DType) int) C.int {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return 0
	}
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return 0
	}
	return C.int(fn(&n.Layers[int(layerIdx)], poly.DType(dtype)))
}

//export LoomCalculateOptimalCNN1SimdTileSizeForLayer
func LoomCalculateOptimalCNN1SimdTileSizeForLayer(networkHandle C.longlong, layerIdx C.int, dtype C.int) C.int {
	return layerTileSizeForNetwork(networkHandle, layerIdx, dtype, poly.CalculateOptimalCNN1SimdTileSizeForLayer)
}

//export LoomCalculateOptimalCNN2TileSizeForLayer
func LoomCalculateOptimalCNN2TileSizeForLayer(networkHandle C.longlong, layerIdx C.int, dtype C.int) C.int {
	return layerTileSizeForNetwork(networkHandle, layerIdx, dtype, poly.CalculateOptimalCNN2TileSizeForLayer)
}

//export LoomCalculateOptimalCNN2SimdTileSizeForLayer
func LoomCalculateOptimalCNN2SimdTileSizeForLayer(networkHandle C.longlong, layerIdx C.int, dtype C.int) C.int {
	return layerTileSizeForNetwork(networkHandle, layerIdx, dtype, poly.CalculateOptimalCNN2SimdTileSizeForLayer)
}

//export LoomCalculateOptimalCNN3TileSizeForLayer
func LoomCalculateOptimalCNN3TileSizeForLayer(networkHandle C.longlong, layerIdx C.int, dtype C.int) C.int {
	return layerTileSizeForNetwork(networkHandle, layerIdx, dtype, poly.CalculateOptimalCNN3TileSizeForLayer)
}

//export LoomCalculateOptimalCNN3SimdTileSizeForLayer
func LoomCalculateOptimalCNN3SimdTileSizeForLayer(networkHandle C.longlong, layerIdx C.int, dtype C.int) C.int {
	return layerTileSizeForNetwork(networkHandle, layerIdx, dtype, poly.CalculateOptimalCNN3SimdTileSizeForLayer)
}

//export LoomCalculateOptimalDenseSimdTileSizeForLayer
func LoomCalculateOptimalDenseSimdTileSizeForLayer(networkHandle C.longlong, layerIdx C.int, dtype C.int) C.int {
	return layerTileSizeForNetwork(networkHandle, layerIdx, dtype, poly.CalculateOptimalDenseSimdTileSizeForLayer)
}

//export LoomCalculateOptimalMHASimdTileSizeForLayer
func LoomCalculateOptimalMHASimdTileSizeForLayer(networkHandle C.longlong, layerIdx C.int, dtype C.int) C.int {
	return layerTileSizeForNetwork(networkHandle, layerIdx, dtype, poly.CalculateOptimalMHASimdTileSizeForLayer)
}

//export LoomCalculateOptimalSwiGLUSimdTileSizeForLayer
func LoomCalculateOptimalSwiGLUSimdTileSizeForLayer(networkHandle C.longlong, layerIdx C.int, dtype C.int) C.int {
	return layerTileSizeForNetwork(networkHandle, layerIdx, dtype, poly.CalculateOptimalSwiGLUSimdTileSizeForLayer)
}

//export LoomCalculateOptimalRNNSimdTileSizeForLayer
func LoomCalculateOptimalRNNSimdTileSizeForLayer(networkHandle C.longlong, layerIdx C.int, dtype C.int) C.int {
	return layerTileSizeForNetwork(networkHandle, layerIdx, dtype, poly.CalculateOptimalRNNSimdTileSizeForLayer)
}

//export LoomCalculateOptimalLSTMSimdTileSizeForLayer
func LoomCalculateOptimalLSTMSimdTileSizeForLayer(networkHandle C.longlong, layerIdx C.int, dtype C.int) C.int {
	return layerTileSizeForNetwork(networkHandle, layerIdx, dtype, poly.CalculateOptimalLSTMSimdTileSizeForLayer)
}

//export LoomCNN1ForwardPackedCPUFloat32
func LoomCNN1ForwardPackedCPUFloat32(networkHandle C.longlong, layerIdx C.int, inputJSON *C.char) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	var in poly.Tensor[float32]
	if err := json.Unmarshal([]byte(C.GoString(inputJSON)), &in); err != nil {
		return errJSON("invalid input tensor JSON")
	}
	pre, post := poly.CNN1ForwardPackedCPU(&n.Layers[int(layerIdx)], &in)
	out := map[string]interface{}{
		"preAct":  pre,
		"postAct": post,
	}
	b, err := json.Marshal(out)
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomCNN1BackwardPackedCPUFloat32
func LoomCNN1BackwardPackedCPUFloat32(networkHandle C.longlong, layerIdx C.int, gradOutputJSON *C.char, inputJSON *C.char, preActJSON *C.char) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	var gradOut, in, pre poly.Tensor[float32]
	if err := json.Unmarshal([]byte(C.GoString(gradOutputJSON)), &gradOut); err != nil {
		return errJSON("invalid gradOutput tensor JSON")
	}
	if err := json.Unmarshal([]byte(C.GoString(inputJSON)), &in); err != nil {
		return errJSON("invalid input tensor JSON")
	}
	if err := json.Unmarshal([]byte(C.GoString(preActJSON)), &pre); err != nil {
		return errJSON("invalid preAct tensor JSON")
	}
	gIn, gW := poly.CNN1BackwardPackedCPU(&n.Layers[int(layerIdx)], &gradOut, &in, &pre)
	out := map[string]interface{}{
		"gradInput":    gIn,
		"gradWeights": gW,
	}
	b, err := json.Marshal(out)
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomMHABackwardTiledFloat32
func LoomMHABackwardTiledFloat32(networkHandle C.longlong, layerIdx C.int, gradOutputJSON *C.char, inputJSON *C.char, preActJSON *C.char) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	var gradOut, in, pre poly.Tensor[float32]
	if err := json.Unmarshal([]byte(C.GoString(gradOutputJSON)), &gradOut); err != nil {
		return errJSON("invalid gradOutput tensor JSON")
	}
	if err := json.Unmarshal([]byte(C.GoString(inputJSON)), &in); err != nil {
		return errJSON("invalid input tensor JSON")
	}
	if err := json.Unmarshal([]byte(C.GoString(preActJSON)), &pre); err != nil {
		return errJSON("invalid preAct tensor JSON")
	}
	gIn, gW := poly.MHABackwardTiled(&n.Layers[int(layerIdx)], &gradOut, &in, &pre)
	out := map[string]interface{}{
		"gradInput":    gIn,
		"gradWeights": gW,
	}
	b, err := json.Marshal(out)
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomLoadSafetensorsSelectiveFromBytes
func LoomLoadSafetensorsSelectiveFromBytes(buffer *C.char, length C.int, tensorNamesJSON *C.char) *C.char {
	ptr := unsafe.Pointer(buffer)
	slice := C.GoBytes(ptr, length)
	var names []string
	s := C.GoString(tensorNamesJSON)
	if strings.TrimSpace(s) != "" {
		if err := json.Unmarshal([]byte(s), &names); err != nil {
			return errJSON("tensorNamesJSON must be a JSON array of strings")
		}
	}
	var keep func(string) bool
	if len(names) > 0 {
		set := make(map[string]struct{}, len(names))
		for _, n := range names {
			set[n] = struct{}{}
		}
		keep = func(name string) bool {
			_, ok := set[name]
			return ok
		}
	}
	weights, err := poly.LoadSafetensorsSelectiveFromBytes(slice, keep)
	if err != nil {
		return errJSON(err.Error())
	}
	b, err := json.Marshal(weights)
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomSaveSafetensorsToBytes
func LoomSaveSafetensorsToBytes(networkHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	data, err := poly.SaveSafetensorsToBytes(n)
	if err != nil {
		return errJSON(err.Error())
	}
	out := map[string]string{
		"b64": base64.StdEncoding.EncodeToString(data),
	}
	b, err := json.Marshal(out)
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomApplyGradientsNative
func LoomApplyGradientsNative(networkHandle C.longlong, layerIdx C.int, dtype C.int, gradWeightsJSON *C.char, learningRate C.float, clipVal C.float) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]
	if l.WeightStore == nil {
		return errJSON("layer has no WeightStore")
	}
	var grad poly.Tensor[float32]
	if err := json.Unmarshal([]byte(C.GoString(gradWeightsJSON)), &grad); err != nil {
		return errJSON("invalid grad tensor JSON")
	}
	okNative := l.WeightStore.ApplyGradientsNative(poly.DType(dtype), &grad, float32(learningRate), float32(clipVal))
	b, _ := json.Marshal(map[string]bool{"applied": okNative})
	return C.CString(string(b))
}

//export LoomLoadEOSTokenIDsFromConfigPath
func LoomLoadEOSTokenIDsFromConfigPath(path *C.char) *C.char {
	ids := poly.LoadEOSTokenIDsFromConfigPath(C.GoString(path))
	b, err := json.Marshal(ids)
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomNewTokenizerFromJSON
func LoomNewTokenizerFromJSON(buffer *C.char, length C.int) C.longlong {
	ptr := unsafe.Pointer(buffer)
	slice := C.GoBytes(ptr, length)
	t, err := poly.NewTokenizerFromJSON(slice)
	if err != nil {
		return -1
	}
	networkMu.Lock()
	id := tokenizerNextID
	tokenizerNextID++
	tokenizers[id] = t
	networkMu.Unlock()
	return C.longlong(id)
}

//export LoomSystemAuditToJSON
func LoomSystemAuditToJSON(networkHandle C.longlong) *C.char {
	var n *poly.VolumetricNetwork
	if int64(networkHandle) > 0 {
		if nn, ok := getNetwork(int64(networkHandle)); ok {
			n = nn
		}
	}
	return C.CString(poly.AuditSystem(n).ToJSON())
}
