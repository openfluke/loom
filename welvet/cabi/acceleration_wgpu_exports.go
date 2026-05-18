package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"encoding/json"

	"github.com/openfluke/loom/poly"
)

// ── Layout / introspection (JSON of zero values; names embed poly type names for C header discovery) ──

//export LoomNativeGPUParityRowJSON
func LoomNativeGPUParityRowJSON() *C.char {
	b, err := json.Marshal(poly.NativeGPUParityRow{})
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomWGPUDenseBitNetTernaryParamsJSON
func LoomWGPUDenseBitNetTernaryParamsJSON() *C.char {
	b, _ := json.Marshal(poly.WGPUDenseBitNetTernaryParams{})
	return C.CString(string(b))
}

//export LoomWGPUBitNetQuantizeActivationParamsJSON
func LoomWGPUBitNetQuantizeActivationParamsJSON() *C.char {
	b, _ := json.Marshal(poly.WGPUBitNetQuantizeActivationParams{})
	return C.CString(string(b))
}

//export LoomWGPUDenseBitNetTernaryQuantizedParamsJSON
func LoomWGPUDenseBitNetTernaryQuantizedParamsJSON() *C.char {
	b, _ := json.Marshal(poly.WGPUDenseBitNetTernaryQuantizedParams{})
	return C.CString(string(b))
}

//export LoomWGPUBitNetGateProductParamsJSON
func LoomWGPUBitNetGateProductParamsJSON() *C.char {
	b, _ := json.Marshal(poly.WGPUBitNetGateProductParams{})
	return C.CString(string(b))
}

//export LoomBitNetShaderNamesJSON
func LoomBitNetShaderNamesJSON() *C.char {
	names := []string{
		poly.ShaderTiledDenseBitNetTernary(32),
		poly.ShaderTiledDenseBitNetTernaryQuantized(32),
		poly.ShaderTiledDenseBitNetTernaryQuantizedReduce(32),
	}
	b, _ := json.Marshal(map[string]interface{}{"sample_tile_32": names})
	return C.CString(string(b))
}

//export LoomGPUInfoJSON
func LoomGPUInfoJSON() *C.char {
	b, err := json.Marshal(poly.GPUInfo{})
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomWGPUSoftmaxParamsJSON
func LoomWGPUSoftmaxParamsJSON() *C.char {
	b, err := json.Marshal(poly.WGPUSoftmaxParams{})
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomWGPUDenseParamsJSON
func LoomWGPUDenseParamsJSON() *C.char {
	b, err := json.Marshal(poly.WGPUDenseParams{})
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomWGPUMHAParamsJSON
func LoomWGPUMHAParamsJSON() *C.char {
	b, err := json.Marshal(poly.WGPUMHAParams{})
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomWGPURNNParamsJSON
func LoomWGPURNNParamsJSON() *C.char {
	b, err := json.Marshal(poly.WGPURNNParams{})
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomWGPULSTMParamsJSON
func LoomWGPULSTMParamsJSON() *C.char {
	b, err := json.Marshal(poly.WGPULSTMParams{})
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomWGPUCNN1ParamsJSON
func LoomWGPUCNN1ParamsJSON() *C.char {
	b, err := json.Marshal(poly.WGPUCNN1Params{})
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomWGPUCNN2ParamsJSON
func LoomWGPUCNN2ParamsJSON() *C.char {
	b, err := json.Marshal(poly.WGPUCNN2Params{})
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomWGPUCNN3ParamsJSON
func LoomWGPUCNN3ParamsJSON() *C.char {
	b, err := json.Marshal(poly.WGPUCNN3Params{})
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomWGPURMSNormParamsJSON
func LoomWGPURMSNormParamsJSON() *C.char {
	b, err := json.Marshal(poly.WGPURMSNormParams{})
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomWGPUKVParamsJSON
func LoomWGPUKVParamsJSON() *C.char {
	b, err := json.Marshal(poly.WGPUKVParams{})
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomWGPURoPEParamsJSON
func LoomWGPURoPEParamsJSON() *C.char {
	b, err := json.Marshal(poly.WGPURoPEParams{})
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomWGPUEmbeddingParamsJSON
func LoomWGPUEmbeddingParamsJSON() *C.char {
	b, err := json.Marshal(poly.WGPUEmbeddingParams{})
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomWGPUDenseI8ParamsJSON
func LoomWGPUDenseI8ParamsJSON() *C.char {
	b, err := json.Marshal(poly.WGPUDenseI8Params{})
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomWGPUSwiGLUI8ParamsJSON
func LoomWGPUSwiGLUI8ParamsJSON() *C.char {
	b, err := json.Marshal(poly.WGPUSwiGLUI8Params{})
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomWGPUEmbeddingShardParamsJSON
func LoomWGPUEmbeddingShardParamsJSON() *C.char {
	b, err := json.Marshal(poly.WGPUEmbeddingShardParams{})
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomWGPUMultiHeadSoftmaxCEParamsJSON
func LoomWGPUMultiHeadSoftmaxCEParamsJSON() *C.char {
	b, err := json.Marshal(poly.WGPUMultiHeadSoftmaxCEParams{})
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomWGPUQuantizeParamsJSON
func LoomWGPUQuantizeParamsJSON() *C.char {
	b, err := json.Marshal(poly.WGPUQuantizeParams{})
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomWGPUCNN1PackedUpdateParamsJSON
func LoomWGPUCNN1PackedUpdateParamsJSON() *C.char {
	b, err := json.Marshal(poly.WGPUCNN1PackedUpdateParams{})
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomWGPUCNN1BackwardScaleParamsJSON
func LoomWGPUCNN1BackwardScaleParamsJSON() *C.char {
	b, err := json.Marshal(poly.WGPUCNN1BackwardScaleParams{})
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

// ── Shader sources ──

//export LoomShaderTiledDenseQ4
func LoomShaderTiledDenseQ4(tileSize C.int) *C.char {
	return C.CString(poly.ShaderTiledDenseQ4(int(tileSize)))
}

//export LoomShaderTiledDenseI8
func LoomShaderTiledDenseI8(tileSize C.int) *C.char {
	return C.CString(poly.ShaderTiledDenseI8(int(tileSize)))
}

//export LoomShaderTiledDenseN
func LoomShaderTiledDenseN(tileSize C.int) *C.char {
	return C.CString(poly.ShaderTiledDenseN(int(tileSize)))
}

//export LoomShaderTiledSwiGLUQ4
func LoomShaderTiledSwiGLUQ4(tileSize C.int) *C.char {
	return C.CString(poly.ShaderTiledSwiGLUQ4(int(tileSize)))
}

//export LoomShaderTiledSwiGLUI8
func LoomShaderTiledSwiGLUI8(tileSize C.int) *C.char {
	return C.CString(poly.ShaderTiledSwiGLUI8(int(tileSize)))
}

//export LoomShaderTiledSwiGLUN
func LoomShaderTiledSwiGLUN(tileSize C.int) *C.char {
	return C.CString(poly.ShaderTiledSwiGLUN(int(tileSize)))
}

//export LoomShaderTiledMHAN
func LoomShaderTiledMHAN(tileSize C.int, headDim C.int) *C.char {
	return C.CString(poly.ShaderTiledMHAN(int(tileSize), int(headDim)))
}

// ── Tile sizing ──

//export LoomCNN1GPUTileSizesForLayer
func LoomCNN1GPUTileSizesForLayer(networkHandle C.longlong, layerIdx C.int, dtype C.int, scTile *C.int, mcTile *C.int) {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return
	}
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return
	}
	sc, mc := poly.CNN1GPUTileSizesForLayer(n.GPUContext, &n.Layers[int(layerIdx)], poly.DType(dtype))
	*scTile = C.int(sc)
	*mcTile = C.int(mc)
}

// ── CPU-side weight view for GPU upload path ──

//export LoomMorphToFloat32ForGPU
func LoomMorphToFloat32ForGPU(networkHandle C.longlong, layerIdx C.int, dtype C.int) *C.char {
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
	out := l.WeightStore.MorphToFloat32ForGPU(poly.DType(dtype))
	b, err := json.Marshal(out)
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

// ── Tanhi / softmax hints ──

//export LoomTanhiGPULayerShapeHint
func LoomTanhiGPULayerShapeHint(networkHandle C.longlong, layerIdx C.int, numTokens C.int) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	hint := poly.TanhiGPULayerShapeHint(&n.Layers[int(layerIdx)], int(numTokens))
	b, err := json.Marshal(hint)
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomDispatchSoftmaxForward
func LoomDispatchSoftmaxForward(networkHandle C.longlong, layerIdx C.int, batchSize C.int, inputHandle C.longlong, outputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return errJSON("invalid network or WGPU not initialized")
	}
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	in, ok := getGPUBufferFromHandle(int64(inputHandle))
	if !ok {
		return errJSON("invalid input buffer handle")
	}
	out, ok := getGPUBufferFromHandle(int64(outputHandle))
	if !ok {
		return errJSON("invalid output buffer handle")
	}
	if err := n.GPUContext.DispatchSoftmaxForward(&n.Layers[int(layerIdx)], int(batchSize), in, out); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomDispatchSoftmaxBackward
func LoomDispatchSoftmaxBackward(networkHandle C.longlong, batchSize C.int, size C.int, temp C.float, gradOutputHandle C.longlong, softmaxOutputHandle C.longlong, gradInputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return errJSON("invalid network or WGPU not initialized")
	}
	gOut, ok := getGPUBufferFromHandle(int64(gradOutputHandle))
	if !ok {
		return errJSON("invalid gradOutput buffer handle")
	}
	sOut, ok := getGPUBufferFromHandle(int64(softmaxOutputHandle))
	if !ok {
		return errJSON("invalid softmaxOutput buffer handle")
	}
	gIn, ok := getGPUBufferFromHandle(int64(gradInputHandle))
	if !ok {
		return errJSON("invalid gradInput buffer handle")
	}
	if err := n.GPUContext.DispatchSoftmaxBackward(int(batchSize), int(size), float32(temp), gOut, sOut, gIn); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

// ── INT8 dense / SwiGLU ──

//export LoomDispatchDenseI8
func LoomDispatchDenseI8(networkHandle C.longlong, batchSize C.int, inputSize C.int, outputSize C.int, inputHandle C.longlong, weightHandle C.longlong, outputHandle C.longlong, scale C.float, tileSize C.int) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return errJSON("invalid network or WGPU not initialized")
	}
	in, ok := getGPUBufferFromHandle(int64(inputHandle))
	if !ok {
		return errJSON("invalid input buffer handle")
	}
	wt, ok := getGPUBufferFromHandle(int64(weightHandle))
	if !ok {
		return errJSON("invalid weight buffer handle")
	}
	out, ok := getGPUBufferFromHandle(int64(outputHandle))
	if !ok {
		return errJSON("invalid output buffer handle")
	}
	if err := n.GPUContext.DispatchDenseI8(int(batchSize), int(inputSize), int(outputSize), in, wt, out, float32(scale), int(tileSize)); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomDispatchSwiGLUI8
func LoomDispatchSwiGLUI8(networkHandle C.longlong, batchSize C.int, inputSize C.int, outputSize C.int, inputHandle C.longlong, gateWHandle C.longlong, upWHandle C.longlong, gateBiasHandle C.longlong, upBiasHandle C.longlong, outputHandle C.longlong, gScale C.float, uScale C.float, tileSize C.int) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return errJSON("invalid network or WGPU not initialized")
	}
	in, ok := getGPUBufferFromHandle(int64(inputHandle))
	if !ok {
		return errJSON("invalid input buffer handle")
	}
	gw, ok := getGPUBufferFromHandle(int64(gateWHandle))
	if !ok {
		return errJSON("invalid gate weight buffer handle")
	}
	uw, ok := getGPUBufferFromHandle(int64(upWHandle))
	if !ok {
		return errJSON("invalid up weight buffer handle")
	}
	gb, ok := getGPUBufferFromHandle(int64(gateBiasHandle))
	if !ok {
		return errJSON("invalid gate bias buffer handle")
	}
	ub, ok := getGPUBufferFromHandle(int64(upBiasHandle))
	if !ok {
		return errJSON("invalid up bias buffer handle")
	}
	out, ok := getGPUBufferFromHandle(int64(outputHandle))
	if !ok {
		return errJSON("invalid output buffer handle")
	}
	if err := n.GPUContext.DispatchSwiGLUI8(int(batchSize), int(inputSize), int(outputSize), in, gw, uw, gb, ub, out, float32(gScale), float32(uScale), int(tileSize)); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

// ── Packed CNN1 (native quant GPU path) ──

//export LoomDispatchCNN1Packed
func LoomDispatchCNN1Packed(networkHandle C.longlong, dtype C.int, batchSize C.int, inC C.int, inL C.int, outC C.int, outL C.int, kSize C.int, stride C.int, padding C.int, scale C.float, inputHandle C.longlong, weightHandle C.longlong, outputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return errJSON("invalid network or WGPU not initialized")
	}
	in, ok := getGPUBufferFromHandle(int64(inputHandle))
	if !ok {
		return errJSON("invalid input buffer handle")
	}
	wt, ok := getGPUBufferFromHandle(int64(weightHandle))
	if !ok {
		return errJSON("invalid weight buffer handle")
	}
	out, ok := getGPUBufferFromHandle(int64(outputHandle))
	if !ok {
		return errJSON("invalid output buffer handle")
	}
	if err := n.GPUContext.DispatchCNN1Packed(poly.DType(dtype), int(batchSize), int(inC), int(inL), int(outC), int(outL), int(kSize), int(stride), int(padding), float32(scale), in, wt, out); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomDispatchCNN1PackedTiled
func LoomDispatchCNN1PackedTiled(networkHandle C.longlong, dtype C.int, tileSize C.int, kernelVol C.int, batchSize C.int, inC C.int, inL C.int, outC C.int, outL C.int, kSize C.int, stride C.int, padding C.int, scale C.float, inputHandle C.longlong, weightHandle C.longlong, outputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return errJSON("invalid network or WGPU not initialized")
	}
	in, ok := getGPUBufferFromHandle(int64(inputHandle))
	if !ok {
		return errJSON("invalid input buffer handle")
	}
	wt, ok := getGPUBufferFromHandle(int64(weightHandle))
	if !ok {
		return errJSON("invalid weight buffer handle")
	}
	out, ok := getGPUBufferFromHandle(int64(outputHandle))
	if !ok {
		return errJSON("invalid output buffer handle")
	}
	if err := n.GPUContext.DispatchCNN1PackedTiled(poly.DType(dtype), int(tileSize), int(kernelVol), int(batchSize), int(inC), int(inL), int(outC), int(outL), int(kSize), int(stride), int(padding), float32(scale), in, wt, out); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomDispatchCNN1PackedBackwardDX
func LoomDispatchCNN1PackedBackwardDX(networkHandle C.longlong, dtype C.int, batchSize C.int, inC C.int, inL C.int, filters C.int, outL C.int, kSize C.int, stride C.int, padding C.int, activation C.int, scale C.float, gradOutHandle C.longlong, weightHandle C.longlong, preActHandle C.longlong, gradInHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return errJSON("invalid network or WGPU not initialized")
	}
	gradOut, ok := getGPUBufferFromHandle(int64(gradOutHandle))
	if !ok {
		return errJSON("invalid gradOut buffer handle")
	}
	wt, ok := getGPUBufferFromHandle(int64(weightHandle))
	if !ok {
		return errJSON("invalid weight buffer handle")
	}
	pa, ok := getGPUBufferFromHandle(int64(preActHandle))
	if !ok {
		return errJSON("invalid preAct buffer handle")
	}
	gi, ok := getGPUBufferFromHandle(int64(gradInHandle))
	if !ok {
		return errJSON("invalid gradIn buffer handle")
	}
	if err := n.GPUContext.DispatchCNN1PackedBackwardDX(poly.DType(dtype), int(batchSize), int(inC), int(inL), int(filters), int(outL), int(kSize), int(stride), int(padding), poly.ActivationType(activation), float32(scale), gradOut, wt, pa, gi); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomDispatchCNN1PackedBackwardDXTiled
func LoomDispatchCNN1PackedBackwardDXTiled(networkHandle C.longlong, dtype C.int, tileSize C.int, kernelVol C.int, batchSize C.int, inC C.int, inL C.int, filters C.int, outL C.int, kSize C.int, stride C.int, padding C.int, activation C.int, scale C.float, gradOutHandle C.longlong, weightHandle C.longlong, preActHandle C.longlong, gradInHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return errJSON("invalid network or WGPU not initialized")
	}
	gradOut, ok := getGPUBufferFromHandle(int64(gradOutHandle))
	if !ok {
		return errJSON("invalid gradOut buffer handle")
	}
	wt, ok := getGPUBufferFromHandle(int64(weightHandle))
	if !ok {
		return errJSON("invalid weight buffer handle")
	}
	pa, ok := getGPUBufferFromHandle(int64(preActHandle))
	if !ok {
		return errJSON("invalid preAct buffer handle")
	}
	gi, ok := getGPUBufferFromHandle(int64(gradInHandle))
	if !ok {
		return errJSON("invalid gradIn buffer handle")
	}
	if err := n.GPUContext.DispatchCNN1PackedBackwardDXTiled(poly.DType(dtype), int(tileSize), int(kernelVol), int(batchSize), int(inC), int(inL), int(filters), int(outL), int(kSize), int(stride), int(padding), poly.ActivationType(activation), float32(scale), gradOut, wt, pa, gi); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomDispatchCNN1PackedApplyGradients
func LoomDispatchCNN1PackedApplyGradients(networkHandle C.longlong, dtype C.int, size C.int, lr C.float, clipVal C.float, scale C.float, packedHandle C.longlong, gradHandle C.longlong, masterHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return errJSON("invalid network or WGPU not initialized")
	}
	pk, ok := getGPUBufferFromHandle(int64(packedHandle))
	if !ok {
		return errJSON("invalid packed buffer handle")
	}
	gd, ok := getGPUBufferFromHandle(int64(gradHandle))
	if !ok {
		return errJSON("invalid grad buffer handle")
	}
	ms, ok := getGPUBufferFromHandle(int64(masterHandle))
	if !ok {
		return errJSON("invalid master buffer handle")
	}
	if err := n.GPUContext.DispatchCNN1PackedApplyGradients(poly.DType(dtype), int(size), float32(lr), float32(clipVal), float32(scale), pk, gd, ms); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

// ── Misc GPU kernels ──

//export LoomDispatchFillZero
func LoomDispatchFillZero(networkHandle C.longlong, size C.int, bufHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return errJSON("invalid network or WGPU not initialized")
	}
	buf, ok := getGPUBufferFromHandle(int64(bufHandle))
	if !ok {
		return errJSON("invalid buffer handle")
	}
	if err := n.GPUContext.DispatchFillZero(int(size), buf); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomDispatchQuantizeI8
func LoomDispatchQuantizeI8(networkHandle C.longlong, size C.int, scale C.float, masterHandle C.longlong, nativeHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return errJSON("invalid network or WGPU not initialized")
	}
	m, ok := getGPUBufferFromHandle(int64(masterHandle))
	if !ok {
		return errJSON("invalid master buffer handle")
	}
	nv, ok := getGPUBufferFromHandle(int64(nativeHandle))
	if !ok {
		return errJSON("invalid native buffer handle")
	}
	if err := n.GPUContext.DispatchQuantizeI8(int(size), float32(scale), m, nv); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomDispatchQuantizeI4
func LoomDispatchQuantizeI4(networkHandle C.longlong, size C.int, scale C.float, masterHandle C.longlong, nativeHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return errJSON("invalid network or WGPU not initialized")
	}
	m, ok := getGPUBufferFromHandle(int64(masterHandle))
	if !ok {
		return errJSON("invalid master buffer handle")
	}
	nv, ok := getGPUBufferFromHandle(int64(nativeHandle))
	if !ok {
		return errJSON("invalid native buffer handle")
	}
	if err := n.GPUContext.DispatchQuantizeI4(int(size), float32(scale), m, nv); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomDispatchQuantizeFP4
func LoomDispatchQuantizeFP4(networkHandle C.longlong, size C.int, scale C.float, masterHandle C.longlong, nativeHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return errJSON("invalid network or WGPU not initialized")
	}
	m, ok := getGPUBufferFromHandle(int64(masterHandle))
	if !ok {
		return errJSON("invalid master buffer handle")
	}
	nv, ok := getGPUBufferFromHandle(int64(nativeHandle))
	if !ok {
		return errJSON("invalid native buffer handle")
	}
	if err := n.GPUContext.DispatchQuantizeFP4(int(size), float32(scale), m, nv); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomDispatchQuantizeTernary
func LoomDispatchQuantizeTernary(networkHandle C.longlong, size C.int, scale C.float, masterHandle C.longlong, nativeHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return errJSON("invalid network or WGPU not initialized")
	}
	m, ok := getGPUBufferFromHandle(int64(masterHandle))
	if !ok {
		return errJSON("invalid master buffer handle")
	}
	nv, ok := getGPUBufferFromHandle(int64(nativeHandle))
	if !ok {
		return errJSON("invalid native buffer handle")
	}
	if err := n.GPUContext.DispatchQuantizeTernary(int(size), float32(scale), m, nv); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomDispatchQuantizeBinary
func LoomDispatchQuantizeBinary(networkHandle C.longlong, size C.int, scale C.float, masterHandle C.longlong, nativeHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return errJSON("invalid network or WGPU not initialized")
	}
	m, ok := getGPUBufferFromHandle(int64(masterHandle))
	if !ok {
		return errJSON("invalid master buffer handle")
	}
	nv, ok := getGPUBufferFromHandle(int64(nativeHandle))
	if !ok {
		return errJSON("invalid native buffer handle")
	}
	if err := n.GPUContext.DispatchQuantizeBinary(int(size), float32(scale), m, nv); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomDispatchCEGradPartialLoss
func LoomDispatchCEGradPartialLoss(networkHandle C.longlong, size C.int, outputHandle C.longlong, targetHandle C.longlong, gradHandle C.longlong, partialsHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return errJSON("invalid network or WGPU not initialized")
	}
	o, ok := getGPUBufferFromHandle(int64(outputHandle))
	if !ok {
		return errJSON("invalid output buffer handle")
	}
	t, ok := getGPUBufferFromHandle(int64(targetHandle))
	if !ok {
		return errJSON("invalid target buffer handle")
	}
	g, ok := getGPUBufferFromHandle(int64(gradHandle))
	if !ok {
		return errJSON("invalid grad buffer handle")
	}
	p, ok := getGPUBufferFromHandle(int64(partialsHandle))
	if !ok {
		return errJSON("invalid partials buffer handle")
	}
	if err := n.GPUContext.DispatchCEGradPartialLoss(int(size), o, t, g, p); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomDispatchMultiHeadSoftmaxCEGradPartialLoss
func LoomDispatchMultiHeadSoftmaxCEGradPartialLoss(networkHandle C.longlong, batchSize C.int, rowWidth C.int, h0 C.int, h1 C.int, h2 C.int, outputHandle C.longlong, targetHandle C.longlong, gradHandle C.longlong, partialsHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return errJSON("invalid network or WGPU not initialized")
	}
	o, ok := getGPUBufferFromHandle(int64(outputHandle))
	if !ok {
		return errJSON("invalid output buffer handle")
	}
	t, ok := getGPUBufferFromHandle(int64(targetHandle))
	if !ok {
		return errJSON("invalid target buffer handle")
	}
	g, ok := getGPUBufferFromHandle(int64(gradHandle))
	if !ok {
		return errJSON("invalid grad buffer handle")
	}
	p, ok := getGPUBufferFromHandle(int64(partialsHandle))
	if !ok {
		return errJSON("invalid partials buffer handle")
	}
	if err := n.GPUContext.DispatchMultiHeadSoftmaxCEGradPartialLoss(int(batchSize), int(rowWidth), int(h0), int(h1), int(h2), o, t, g, p); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomDispatchMultiHeadSoftmaxCEGradPartialLossMasked
func LoomDispatchMultiHeadSoftmaxCEGradPartialLossMasked(networkHandle C.longlong, batchSize C.int, rowWidth C.int, h0 C.int, h1 C.int, h2 C.int, outputHandle C.longlong, targetHandle C.longlong, gradHandle C.longlong, partialsHandle C.longlong, headMaskHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return errJSON("invalid network or WGPU not initialized")
	}
	o, ok := getGPUBufferFromHandle(int64(outputHandle))
	if !ok {
		return errJSON("invalid output buffer handle")
	}
	t, ok := getGPUBufferFromHandle(int64(targetHandle))
	if !ok {
		return errJSON("invalid target buffer handle")
	}
	g, ok := getGPUBufferFromHandle(int64(gradHandle))
	if !ok {
		return errJSON("invalid grad buffer handle")
	}
	p, ok := getGPUBufferFromHandle(int64(partialsHandle))
	if !ok {
		return errJSON("invalid partials buffer handle")
	}
	hm, ok := getGPUBufferFromHandle(int64(headMaskHandle))
	if !ok {
		return errJSON("invalid headMask buffer handle")
	}
	if err := n.GPUContext.DispatchMultiHeadSoftmaxCEGradPartialLossMasked(int(batchSize), int(rowWidth), int(h0), int(h1), int(h2), o, t, g, p, hm); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomGetGPUWeightBuffer
func LoomGetGPUWeightBuffer(networkHandle C.longlong, layerIdx C.int) C.longlong {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return -1
	}
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return -1
	}
	buf := poly.GetGPUWeightBuffer(&n.Layers[int(layerIdx)])
	if buf == nil {
		return -1
	}
	networkMu.Lock()
	id := stepNextID
	stepNextID++
	stepStates[id] = &stepContainer{State: buf, DType: poly.DTypeFloat32, Borrowed: true}
	networkMu.Unlock()
	return C.longlong(id)
}

// ── BitNet GPU kernels ──

//export LoomDispatchDenseBitNetTernary
func LoomDispatchDenseBitNetTernary(networkHandle C.longlong, batchSize C.int, inputSize C.int, outputSize C.int, inputHandle C.longlong, weightHandle C.longlong, biasHandle C.longlong, outputHandle C.longlong, weightScale C.float, activation C.int, tileSize C.int) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return errJSON("invalid network or WGPU not initialized")
	}
	in, ok := getGPUBufferFromHandle(int64(inputHandle))
	if !ok {
		return errJSON("invalid input buffer handle")
	}
	wt, ok := getGPUBufferFromHandle(int64(weightHandle))
	if !ok {
		return errJSON("invalid weight buffer handle")
	}
	out, ok := getGPUBufferFromHandle(int64(outputHandle))
	if !ok {
		return errJSON("invalid output buffer handle")
	}
	var err error
	if biasHandle == 0 {
		err = n.GPUContext.DispatchDenseBitNetTernary(int(batchSize), int(inputSize), int(outputSize), in, wt, nil, out, float32(weightScale), poly.ActivationType(activation), int(tileSize))
	} else {
		bias, ok := getGPUBufferFromHandle(int64(biasHandle))
		if !ok {
			return errJSON("invalid bias buffer handle")
		}
		err = n.GPUContext.DispatchDenseBitNetTernary(int(batchSize), int(inputSize), int(outputSize), in, wt, bias, out, float32(weightScale), poly.ActivationType(activation), int(tileSize))
	}
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomDispatchBitNetQuantizeActivation
func LoomDispatchBitNetQuantizeActivation(networkHandle C.longlong, batchSize C.int, inputSize C.int, inputHandle C.longlong, qPackedHandle C.longlong, scaleHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return errJSON("invalid network or WGPU not initialized")
	}
	in, ok := getGPUBufferFromHandle(int64(inputHandle))
	if !ok {
		return errJSON("invalid input buffer handle")
	}
	qp, ok := getGPUBufferFromHandle(int64(qPackedHandle))
	if !ok {
		return errJSON("invalid qPacked buffer handle")
	}
	sc, ok := getGPUBufferFromHandle(int64(scaleHandle))
	if !ok {
		return errJSON("invalid scale buffer handle")
	}
	if err := n.GPUContext.DispatchBitNetQuantizeActivation(int(batchSize), int(inputSize), in, qp, sc); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomDispatchDenseBitNetTernaryQuantized
func LoomDispatchDenseBitNetTernaryQuantized(networkHandle C.longlong, batchSize C.int, inputSize C.int, outputSize C.int, qPackedHandle C.longlong, scaleHandle C.longlong, weightHandle C.longlong, biasHandle C.longlong, outputHandle C.longlong, weightScale C.float, activation C.int, tileSize C.int) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return errJSON("invalid network or WGPU not initialized")
	}
	qp, ok := getGPUBufferFromHandle(int64(qPackedHandle))
	if !ok {
		return errJSON("invalid qPacked buffer handle")
	}
	sc, ok := getGPUBufferFromHandle(int64(scaleHandle))
	if !ok {
		return errJSON("invalid scale buffer handle")
	}
	wt, ok := getGPUBufferFromHandle(int64(weightHandle))
	if !ok {
		return errJSON("invalid weight buffer handle")
	}
	out, ok := getGPUBufferFromHandle(int64(outputHandle))
	if !ok {
		return errJSON("invalid output buffer handle")
	}
	var err error
	if biasHandle == 0 {
		err = n.GPUContext.DispatchDenseBitNetTernaryQuantized(int(batchSize), int(inputSize), int(outputSize), qp, sc, wt, nil, out, float32(weightScale), poly.ActivationType(activation), int(tileSize))
	} else {
		bias, ok := getGPUBufferFromHandle(int64(biasHandle))
		if !ok {
			return errJSON("invalid bias buffer handle")
		}
		err = n.GPUContext.DispatchDenseBitNetTernaryQuantized(int(batchSize), int(inputSize), int(outputSize), qp, sc, wt, bias, out, float32(weightScale), poly.ActivationType(activation), int(tileSize))
	}
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomDispatchBitNetGateProduct
func LoomDispatchBitNetGateProduct(networkHandle C.longlong, batchSize C.int, hiddenSize C.int, activation C.int, gatePreHandle C.longlong, upPreHandle C.longlong, outputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return errJSON("invalid network or WGPU not initialized")
	}
	gate, ok := getGPUBufferFromHandle(int64(gatePreHandle))
	if !ok {
		return errJSON("invalid gatePre buffer handle")
	}
	up, ok := getGPUBufferFromHandle(int64(upPreHandle))
	if !ok {
		return errJSON("invalid upPre buffer handle")
	}
	out, ok := getGPUBufferFromHandle(int64(outputHandle))
	if !ok {
		return errJSON("invalid output buffer handle")
	}
	if err := n.GPUContext.DispatchBitNetGateProduct(int(batchSize), int(hiddenSize), poly.ActivationType(activation), gate, up, out); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}
