package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"encoding/json"
	"unsafe"

	"github.com/openfluke/loom/poly"
)

//export LoomAsmEnabled
func LoomAsmEnabled() C.int {
	return 0
}

//export LoomSimdEnabled
func LoomSimdEnabled() C.int {
	if poly.Plan9SimdEnabled() {
		return 1
	}
	return 0
}

//export LoomSetNetworkUseSimdForward
func LoomSetNetworkUseSimdForward(networkHandle C.longlong, enabled C.int) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	n.SetSimdForwardRecursive(enabled != 0)
	return C.CString(`{"status":"ok"}`)
}

//export LoomGetNetworkUseSimdForward
func LoomGetNetworkUseSimdForward(networkHandle C.longlong) C.int {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return 0
	}
	if n.UseSimdForward {
		return 1
	}
	return 0
}

//export LoomPlan9SimdForwardForLayer
func LoomPlan9SimdForwardForLayer(layerType C.int) C.int {
	if poly.Plan9SimdForwardForLayer(poly.LayerType(layerType)) {
		return 1
	}
	return 0
}

//export LoomLayerSupportsSimdForward
func LoomLayerSupportsSimdForward(layerType C.int) C.int {
	if poly.LayerSupportsSimdForward(poly.LayerType(layerType)) {
		return 1
	}
	return 0
}

//export LoomSetBitNetTernarySimdForward
func LoomSetBitNetTernarySimdForward(enabled C.int) {
	poly.SetBitNetTernarySimdForward(enabled != 0)
}

//export LoomSetBitNetTL1Forward
func LoomSetBitNetTL1Forward(enabled C.int) {
	poly.SetBitNetTL1Forward(enabled != 0)
}

//export LoomStackLayerCount
func LoomStackLayerCount(networkHandle C.longlong) C.int {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return 0
	}
	return C.int(n.StackLayerCount())
}

//export LoomSetNetworkUseAsmForward
func LoomSetNetworkUseAsmForward(networkHandle C.longlong, enabled C.int) *C.char {
	_, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	_ = enabled
	return C.CString(`{"status":"ok"}`)
}

//export LoomGetNetworkUseAsmForward
func LoomGetNetworkUseAsmForward(networkHandle C.longlong) C.int {
	_, ok := getNetwork(int64(networkHandle))
	if !ok {
		return 0
	}
	return 0
}

//export LoomSetLayerUseAsmForward
func LoomSetLayerUseAsmForward(networkHandle C.longlong, layerIndex C.int, enabled C.int) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if int(layerIndex) < 0 || int(layerIndex) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	_ = enabled
	return C.CString(`{"status":"ok"}`)
}

//export LoomGPUPackedWeightsKey
func LoomGPUPackedWeightsKey(dtype C.int) C.int {
	return C.int(poly.GPUPackedWeightsKey(poly.DType(dtype)))
}

//export LoomGPUPackedNativeByteSize
func LoomGPUPackedNativeByteSize(dtype C.int, weightCount C.int) C.int {
	return C.int(poly.GPUPackedNativeByteSize(poly.DType(dtype), int(weightCount)))
}

//export LoomStoredTensorNumElements
func LoomStoredTensorNumElements(tensorJSON *C.char) C.int {
	var t poly.HFStoredTensor
	if err := json.Unmarshal([]byte(C.GoString(tensorJSON)), &t); err != nil {
		return -1
	}
	return C.int(poly.StoredTensorNumElements(t))
}

//export LoomDecodeTensorBytesInto
func LoomDecodeTensorBytesInto(dstJSON *C.char, bytesJSON *C.char, dtype *C.char) *C.char {
	var dst []float32
	if err := json.Unmarshal([]byte(C.GoString(dstJSON)), &dst); err != nil {
		return errJSON("dstJSON must be a JSON array of float32")
	}
	var tensorBytes []byte
	if err := json.Unmarshal([]byte(C.GoString(bytesJSON)), &tensorBytes); err != nil {
		return errJSON("bytesJSON must be a JSON byte array")
	}
	if err := poly.DecodeTensorBytesInto(dst, tensorBytes, C.GoString(dtype)); err != nil {
		return errJSON(err.Error())
	}
	out, _ := json.Marshal(dst)
	return C.CString(string(out))
}

//export LoomLoadSafetensorsSelectiveRawFromBytes
func LoomLoadSafetensorsSelectiveRawFromBytes(buffer *C.char, length C.int, tensorNamesJSON *C.char) *C.char {
	slice := C.GoBytes(unsafe.Pointer(buffer), length)
	var names []string
	s := C.GoString(tensorNamesJSON)
	if s != "" {
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
	tensors, err := poly.LoadSafetensorsSelectiveRawFromBytes(slice, keep)
	if err != nil {
		return errJSON(err.Error())
	}
	b, err := json.Marshal(tensors)
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomSetLayerLoadedWeights
func LoomSetLayerLoadedWeights(networkHandle C.longlong, layerIndex C.int, dtype C.int, weightsJSON *C.char) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if int(layerIndex) < 0 || int(layerIndex) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIndex)]
	if l.WeightStore == nil {
		return errJSON("layer has no WeightStore")
	}
	dt := poly.DType(dtype)
	var data any
	switch dt {
	case poly.DTypeFloat64:
		var w []float64
		if err := json.Unmarshal([]byte(C.GoString(weightsJSON)), &w); err != nil {
			return errJSON("invalid weights JSON for float64")
		}
		data = w
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		var w []float32
		if err := json.Unmarshal([]byte(C.GoString(weightsJSON)), &w); err != nil {
			return errJSON("invalid weights JSON for float32 family")
		}
		data = w
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		var w []int8
		if err := json.Unmarshal([]byte(C.GoString(weightsJSON)), &w); err != nil {
			return errJSON("invalid weights JSON for int8 family")
		}
		data = w
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		var w []uint8
		if err := json.Unmarshal([]byte(C.GoString(weightsJSON)), &w); err != nil {
			return errJSON("invalid weights JSON for uint8 family")
		}
		data = w
	default:
		var w []float32
		if err := json.Unmarshal([]byte(C.GoString(weightsJSON)), &w); err != nil {
			return errJSON("invalid weights JSON")
		}
		data = w
	}
	l.WeightStore.SetLoadedWeights(dt, data)
	return C.CString(`{"status":"ok"}`)
}

//export LoomMorphLayerBitNetTernary
func LoomMorphLayerBitNetTernary(networkHandle C.longlong, layerIndex C.int) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if int(layerIndex) < 0 || int(layerIndex) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	if err := poly.MorphLayerBitNetTernary(&n.Layers[int(layerIndex)]); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status":"ok"}`)
}

//export LoomMorphLayerBitNetNativeTernary
func LoomMorphLayerBitNetNativeTernary(networkHandle C.longlong, layerIndex C.int) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if int(layerIndex) < 0 || int(layerIndex) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	if err := poly.MorphLayerBitNetNativeTernary(&n.Layers[int(layerIndex)]); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status":"ok"}`)
}

//export LoomDenseForwardPackedTernaryCPU
func LoomDenseForwardPackedTernaryCPU(networkHandle C.longlong, layerIndex C.int, inputJSON *C.char) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if int(layerIndex) < 0 || int(layerIndex) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	var in poly.Tensor[float32]
	if err := json.Unmarshal([]byte(C.GoString(inputJSON)), &in); err != nil {
		return errJSON("invalid input tensor JSON")
	}
	pre, post := poly.DenseForwardPackedTernaryCPU(&n.Layers[int(layerIndex)], &in)
	out := map[string]interface{}{"preAct": pre, "postAct": post}
	b, err := json.Marshal(out)
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomMHAForwardPackedTernaryCPU
func LoomMHAForwardPackedTernaryCPU(networkHandle C.longlong, layerIndex C.int, inputJSON *C.char) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if int(layerIndex) < 0 || int(layerIndex) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	var in poly.Tensor[float32]
	if err := json.Unmarshal([]byte(C.GoString(inputJSON)), &in); err != nil {
		return errJSON("invalid input tensor JSON")
	}
	pre, post := poly.MHAForwardPackedTernaryCPU(&n.Layers[int(layerIndex)], &in)
	out := map[string]interface{}{"preAct": pre, "postAct": post}
	b, err := json.Marshal(out)
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomSwiGLUForwardPackedTernaryCPU
func LoomSwiGLUForwardPackedTernaryCPU(networkHandle C.longlong, layerIndex C.int, inputJSON *C.char) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if int(layerIndex) < 0 || int(layerIndex) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	var in poly.Tensor[float32]
	if err := json.Unmarshal([]byte(C.GoString(inputJSON)), &in); err != nil {
		return errJSON("invalid input tensor JSON")
	}
	pre, post := poly.SwiGLUForwardPackedTernaryCPU(&n.Layers[int(layerIndex)], &in)
	out := map[string]interface{}{"preAct": pre, "postAct": post}
	b, err := json.Marshal(out)
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomMorphToFloat32ForGPUMasterSlice
func LoomMorphToFloat32ForGPUMasterSlice(networkHandle C.longlong, layerIndex C.int, dtype C.int, masterJSON *C.char) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if int(layerIndex) < 0 || int(layerIndex) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIndex)]
	if l.WeightStore == nil {
		return errJSON("layer has no WeightStore")
	}
	var master []float32
	if err := json.Unmarshal([]byte(C.GoString(masterJSON)), &master); err != nil {
		return errJSON("masterJSON must be a JSON array of float32")
	}
	out := l.WeightStore.MorphToFloat32ForGPUMasterSlice(master, poly.DType(dtype))
	b, err := json.Marshal(out)
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}
