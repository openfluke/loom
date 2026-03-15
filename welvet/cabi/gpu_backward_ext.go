package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"github.com/openfluke/loom/poly"
	"github.com/openfluke/webgpu/wgpu"
)

// ──────────────────────────────────────────────────────────────────────────────
// GPU Buffer Management
// ──────────────────────────────────────────────────────────────────────────────

//export LoomCreateGPUBuffer
func LoomCreateGPUBuffer(networkHandle C.longlong, sizeBytes C.longlong) C.longlong {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || n.GPUContext == nil {
		return -1
	}

	buf, err := n.GPUContext.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "loom_gpu_buf",
		Size:  uint64(sizeBytes),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return -1
	}

	networkMu.Lock()
	id := systolicNextID
	systolicNextID++
	systolicStates[id] = &systolicContainer{State: buf, DType: poly.DTypeFloat32}
	networkMu.Unlock()
	return C.longlong(id)
}

//export LoomFreeGPUBuffer
func LoomFreeGPUBuffer(bufHandle C.longlong) {
	networkMu.Lock()
	c, ok := systolicStates[int64(bufHandle)]
	if ok {
		delete(systolicStates, int64(bufHandle))
	}
	networkMu.Unlock()
	if ok {
		if buf, ok2 := c.State.(*wgpu.Buffer); ok2 {
			buf.Destroy()
		}
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// Shader Source Getters
// ──────────────────────────────────────────────────────────────────────────────

//export LoomShaderDenseBackwardDX
func LoomShaderDenseBackwardDX(tileSize C.int) *C.char {
	return C.CString(poly.ShaderDenseBackwardDX(int(tileSize)))
}

//export LoomShaderDenseBackwardDW
func LoomShaderDenseBackwardDW(tileSize C.int) *C.char {
	return C.CString(poly.ShaderDenseBackwardDW(int(tileSize)))
}

// ──────────────────────────────────────────────────────────────────────────────
// Dense Backward
// ──────────────────────────────────────────────────────────────────────────────

//export LoomDispatchDenseBackwardDX
func LoomDispatchDenseBackwardDX(networkHandle C.longlong, batchSize C.int, inputSize C.int, outputSize C.int, gradOutputHandle C.longlong, weightHandle C.longlong, gradInputHandle C.longlong, tileSize C.int) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if n.GPUContext == nil {
		return errJSON("WGPU not initialized")
	}
	gradOut, ok := getGPUBufferFromHandle(int64(gradOutputHandle))
	if !ok {
		return errJSON("invalid gradOutput buffer handle")
	}
	wt, ok := getGPUBufferFromHandle(int64(weightHandle))
	if !ok {
		return errJSON("invalid weight buffer handle")
	}
	gradIn, ok := getGPUBufferFromHandle(int64(gradInputHandle))
	if !ok {
		return errJSON("invalid gradInput buffer handle")
	}
	if err := n.GPUContext.DispatchDenseBackwardDX(int(batchSize), int(inputSize), int(outputSize), gradOut, wt, gradIn, int(tileSize)); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomDispatchDenseBackwardDW
func LoomDispatchDenseBackwardDW(networkHandle C.longlong, batchSize C.int, inputSize C.int, outputSize C.int, gradOutputHandle C.longlong, inputHandle C.longlong, gradWeightHandle C.longlong, tileSize C.int) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if n.GPUContext == nil {
		return errJSON("WGPU not initialized")
	}
	gradOut, ok := getGPUBufferFromHandle(int64(gradOutputHandle))
	if !ok {
		return errJSON("invalid gradOutput buffer handle")
	}
	in, ok := getGPUBufferFromHandle(int64(inputHandle))
	if !ok {
		return errJSON("invalid input buffer handle")
	}
	gradWt, ok := getGPUBufferFromHandle(int64(gradWeightHandle))
	if !ok {
		return errJSON("invalid gradWeight buffer handle")
	}
	if err := n.GPUContext.DispatchDenseBackwardDW(int(batchSize), int(inputSize), int(outputSize), gradOut, in, gradWt, int(tileSize)); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

// ──────────────────────────────────────────────────────────────────────────────
// SwiGLU Backward
// ──────────────────────────────────────────────────────────────────────────────

//export LoomDispatchSwiGLUBackward
func LoomDispatchSwiGLUBackward(networkHandle C.longlong, batchSize C.int, inputSize C.int, outputSize C.int, gradOutputHandle C.longlong, gateInHandle C.longlong, upInHandle C.longlong, gradGateHandle C.longlong, gradUpHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if n.GPUContext == nil {
		return errJSON("WGPU not initialized")
	}
	gradOut, ok := getGPUBufferFromHandle(int64(gradOutputHandle))
	if !ok {
		return errJSON("invalid gradOutput buffer handle")
	}
	gateIn, ok := getGPUBufferFromHandle(int64(gateInHandle))
	if !ok {
		return errJSON("invalid gateIn buffer handle")
	}
	upIn, ok := getGPUBufferFromHandle(int64(upInHandle))
	if !ok {
		return errJSON("invalid upIn buffer handle")
	}
	gradGate, ok := getGPUBufferFromHandle(int64(gradGateHandle))
	if !ok {
		return errJSON("invalid gradGate buffer handle")
	}
	gradUp, ok := getGPUBufferFromHandle(int64(gradUpHandle))
	if !ok {
		return errJSON("invalid gradUp buffer handle")
	}
	if err := n.GPUContext.DispatchSwiGLUBackward(int(batchSize), int(inputSize), int(outputSize), gradOut, gateIn, upIn, gradGate, gradUp); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

// ──────────────────────────────────────────────────────────────────────────────
// RMSNorm Backward
// ──────────────────────────────────────────────────────────────────────────────

//export LoomDispatchRMSNormBackward
func LoomDispatchRMSNormBackward(networkHandle C.longlong, batchSize C.int, size C.int, epsilon C.float, gradOutputHandle C.longlong, inputHandle C.longlong, rmsHandle C.longlong, weightHandle C.longlong, gradInputHandle C.longlong, gradWeightHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if n.GPUContext == nil {
		return errJSON("WGPU not initialized")
	}
	gradOut, ok := getGPUBufferFromHandle(int64(gradOutputHandle))
	if !ok {
		return errJSON("invalid gradOutput buffer handle")
	}
	in, ok := getGPUBufferFromHandle(int64(inputHandle))
	if !ok {
		return errJSON("invalid input buffer handle")
	}
	rms, ok := getGPUBufferFromHandle(int64(rmsHandle))
	if !ok {
		return errJSON("invalid rms buffer handle")
	}
	wt, ok := getGPUBufferFromHandle(int64(weightHandle))
	if !ok {
		return errJSON("invalid weight buffer handle")
	}
	gradIn, ok := getGPUBufferFromHandle(int64(gradInputHandle))
	if !ok {
		return errJSON("invalid gradInput buffer handle")
	}
	gradWt, ok := getGPUBufferFromHandle(int64(gradWeightHandle))
	if !ok {
		return errJSON("invalid gradWeight buffer handle")
	}
	if err := n.GPUContext.DispatchRMSNormBackward(int(batchSize), int(size), float32(epsilon), gradOut, in, rms, wt, gradIn, gradWt); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

// ──────────────────────────────────────────────────────────────────────────────
// Embedding Backward
// ──────────────────────────────────────────────────────────────────────────────

//export LoomDispatchEmbeddingBackward
func LoomDispatchEmbeddingBackward(networkHandle C.longlong, vocabSize C.int, hiddenSize C.int, numTokens C.int, indicesHandle C.longlong, gradOutputHandle C.longlong, gradWeightHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if n.GPUContext == nil {
		return errJSON("WGPU not initialized")
	}
	idx, ok := getGPUBufferFromHandle(int64(indicesHandle))
	if !ok {
		return errJSON("invalid indices buffer handle")
	}
	gradOut, ok := getGPUBufferFromHandle(int64(gradOutputHandle))
	if !ok {
		return errJSON("invalid gradOutput buffer handle")
	}
	gradWt, ok := getGPUBufferFromHandle(int64(gradWeightHandle))
	if !ok {
		return errJSON("invalid gradWeight buffer handle")
	}
	if err := n.GPUContext.DispatchEmbeddingBackward(int(vocabSize), int(hiddenSize), int(numTokens), idx, gradOut, gradWt); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

// ──────────────────────────────────────────────────────────────────────────────
// Residual Backward
// ──────────────────────────────────────────────────────────────────────────────

//export LoomDispatchResidualBackward
func LoomDispatchResidualBackward(networkHandle C.longlong, size C.int, gradOutputHandle C.longlong, gradInputHandle C.longlong, gradResidualHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if n.GPUContext == nil {
		return errJSON("WGPU not initialized")
	}
	gradOut, ok := getGPUBufferFromHandle(int64(gradOutputHandle))
	if !ok {
		return errJSON("invalid gradOutput buffer handle")
	}
	gradIn, ok := getGPUBufferFromHandle(int64(gradInputHandle))
	if !ok {
		return errJSON("invalid gradInput buffer handle")
	}
	gradRes, ok := getGPUBufferFromHandle(int64(gradResidualHandle))
	if !ok {
		return errJSON("invalid gradResidual buffer handle")
	}
	if err := n.GPUContext.DispatchResidualBackward(int(size), gradOut, gradIn, gradRes); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

// ──────────────────────────────────────────────────────────────────────────────
// CNN1 Backward
// ──────────────────────────────────────────────────────────────────────────────

//export LoomDispatchCNN1BackwardDX
func LoomDispatchCNN1BackwardDX(networkHandle C.longlong, batchSize C.int, inC C.int, inL C.int, filters C.int, outL C.int, kSize C.int, stride C.int, padding C.int, activation C.int, gradOutputHandle C.longlong, weightHandle C.longlong, preActHandle C.longlong, gradInputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if n.GPUContext == nil {
		return errJSON("WGPU not initialized")
	}
	gradOut, ok := getGPUBufferFromHandle(int64(gradOutputHandle))
	if !ok {
		return errJSON("invalid gradOutput buffer handle")
	}
	wt, ok := getGPUBufferFromHandle(int64(weightHandle))
	if !ok {
		return errJSON("invalid weight buffer handle")
	}
	preAct, ok := getGPUBufferFromHandle(int64(preActHandle))
	if !ok {
		return errJSON("invalid preAct buffer handle")
	}
	gradIn, ok := getGPUBufferFromHandle(int64(gradInputHandle))
	if !ok {
		return errJSON("invalid gradInput buffer handle")
	}
	if err := n.GPUContext.DispatchCNN1BackwardDX(int(batchSize), int(inC), int(inL), int(filters), int(outL), int(kSize), int(stride), int(padding), poly.ActivationType(activation), gradOut, wt, preAct, gradIn); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomDispatchCNN1BackwardDW
func LoomDispatchCNN1BackwardDW(networkHandle C.longlong, batchSize C.int, inC C.int, inL C.int, filters C.int, outL C.int, kSize C.int, stride C.int, padding C.int, activation C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong, gradWeightHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if n.GPUContext == nil {
		return errJSON("WGPU not initialized")
	}
	gradOut, ok := getGPUBufferFromHandle(int64(gradOutputHandle))
	if !ok {
		return errJSON("invalid gradOutput buffer handle")
	}
	in, ok := getGPUBufferFromHandle(int64(inputHandle))
	if !ok {
		return errJSON("invalid input buffer handle")
	}
	preAct, ok := getGPUBufferFromHandle(int64(preActHandle))
	if !ok {
		return errJSON("invalid preAct buffer handle")
	}
	gradWt, ok := getGPUBufferFromHandle(int64(gradWeightHandle))
	if !ok {
		return errJSON("invalid gradWeight buffer handle")
	}
	if err := n.GPUContext.DispatchCNN1BackwardDW(int(batchSize), int(inC), int(inL), int(filters), int(outL), int(kSize), int(stride), int(padding), poly.ActivationType(activation), gradOut, in, preAct, gradWt); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

// ──────────────────────────────────────────────────────────────────────────────
// CNN2 Backward
// ──────────────────────────────────────────────────────────────────────────────

//export LoomDispatchCNN2BackwardDX
func LoomDispatchCNN2BackwardDX(networkHandle C.longlong, batchSize C.int, inC C.int, inH C.int, inW C.int, filters C.int, outH C.int, outW C.int, kSize C.int, stride C.int, padding C.int, activation C.int, gradOutputHandle C.longlong, weightHandle C.longlong, preActHandle C.longlong, gradInputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if n.GPUContext == nil {
		return errJSON("WGPU not initialized")
	}
	gradOut, ok := getGPUBufferFromHandle(int64(gradOutputHandle))
	if !ok {
		return errJSON("invalid gradOutput buffer handle")
	}
	wt, ok := getGPUBufferFromHandle(int64(weightHandle))
	if !ok {
		return errJSON("invalid weight buffer handle")
	}
	preAct, ok := getGPUBufferFromHandle(int64(preActHandle))
	if !ok {
		return errJSON("invalid preAct buffer handle")
	}
	gradIn, ok := getGPUBufferFromHandle(int64(gradInputHandle))
	if !ok {
		return errJSON("invalid gradInput buffer handle")
	}
	if err := n.GPUContext.DispatchCNN2BackwardDX(int(batchSize), int(inC), int(inH), int(inW), int(filters), int(outH), int(outW), int(kSize), int(stride), int(padding), poly.ActivationType(activation), gradOut, wt, preAct, gradIn); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomDispatchCNN2BackwardDW
func LoomDispatchCNN2BackwardDW(networkHandle C.longlong, batchSize C.int, inC C.int, inH C.int, inW C.int, filters C.int, outH C.int, outW C.int, kSize C.int, stride C.int, padding C.int, activation C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong, gradWeightHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if n.GPUContext == nil {
		return errJSON("WGPU not initialized")
	}
	gradOut, ok := getGPUBufferFromHandle(int64(gradOutputHandle))
	if !ok {
		return errJSON("invalid gradOutput buffer handle")
	}
	in, ok := getGPUBufferFromHandle(int64(inputHandle))
	if !ok {
		return errJSON("invalid input buffer handle")
	}
	preAct, ok := getGPUBufferFromHandle(int64(preActHandle))
	if !ok {
		return errJSON("invalid preAct buffer handle")
	}
	gradWt, ok := getGPUBufferFromHandle(int64(gradWeightHandle))
	if !ok {
		return errJSON("invalid gradWeight buffer handle")
	}
	if err := n.GPUContext.DispatchCNN2BackwardDW(int(batchSize), int(inC), int(inH), int(inW), int(filters), int(outH), int(outW), int(kSize), int(stride), int(padding), poly.ActivationType(activation), gradOut, in, preAct, gradWt); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

// ──────────────────────────────────────────────────────────────────────────────
// CNN3 Backward
// ──────────────────────────────────────────────────────────────────────────────

//export LoomDispatchCNN3BackwardDX
func LoomDispatchCNN3BackwardDX(networkHandle C.longlong, batchSize C.int, inC C.int, inD C.int, inH C.int, inW C.int, filters C.int, outD C.int, outH C.int, outW C.int, kSize C.int, stride C.int, padding C.int, activation C.int, gradOutputHandle C.longlong, weightHandle C.longlong, preActHandle C.longlong, gradInputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if n.GPUContext == nil {
		return errJSON("WGPU not initialized")
	}
	gradOut, ok := getGPUBufferFromHandle(int64(gradOutputHandle))
	if !ok {
		return errJSON("invalid gradOutput buffer handle")
	}
	wt, ok := getGPUBufferFromHandle(int64(weightHandle))
	if !ok {
		return errJSON("invalid weight buffer handle")
	}
	preAct, ok := getGPUBufferFromHandle(int64(preActHandle))
	if !ok {
		return errJSON("invalid preAct buffer handle")
	}
	gradIn, ok := getGPUBufferFromHandle(int64(gradInputHandle))
	if !ok {
		return errJSON("invalid gradInput buffer handle")
	}
	if err := n.GPUContext.DispatchCNN3BackwardDX(int(batchSize), int(inC), int(inD), int(inH), int(inW), int(filters), int(outD), int(outH), int(outW), int(kSize), int(stride), int(padding), poly.ActivationType(activation), gradOut, wt, preAct, gradIn); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomDispatchCNN3BackwardDW
func LoomDispatchCNN3BackwardDW(networkHandle C.longlong, batchSize C.int, inC C.int, inD C.int, inH C.int, inW C.int, filters C.int, outD C.int, outH C.int, outW C.int, kSize C.int, stride C.int, padding C.int, activation C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong, gradWeightHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if n.GPUContext == nil {
		return errJSON("WGPU not initialized")
	}
	gradOut, ok := getGPUBufferFromHandle(int64(gradOutputHandle))
	if !ok {
		return errJSON("invalid gradOutput buffer handle")
	}
	in, ok := getGPUBufferFromHandle(int64(inputHandle))
	if !ok {
		return errJSON("invalid input buffer handle")
	}
	preAct, ok := getGPUBufferFromHandle(int64(preActHandle))
	if !ok {
		return errJSON("invalid preAct buffer handle")
	}
	gradWt, ok := getGPUBufferFromHandle(int64(gradWeightHandle))
	if !ok {
		return errJSON("invalid gradWeight buffer handle")
	}
	if err := n.GPUContext.DispatchCNN3BackwardDW(int(batchSize), int(inC), int(inD), int(inH), int(inW), int(filters), int(outD), int(outH), int(outW), int(kSize), int(stride), int(padding), poly.ActivationType(activation), gradOut, in, preAct, gradWt); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

// ──────────────────────────────────────────────────────────────────────────────
// MHA Backward
// ──────────────────────────────────────────────────────────────────────────────

//export LoomDispatchMHABackward
func LoomDispatchMHABackward(networkHandle C.longlong, batchSize C.int, numHeads C.int, numKVHeads C.int, headDim C.int, seqLen C.int, scale C.float, gradOutputHandle C.longlong, qHandle C.longlong, kHandle C.longlong, vHandle C.longlong, dQHandle C.longlong, dKHandle C.longlong, dVHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if n.GPUContext == nil {
		return errJSON("WGPU not initialized")
	}
	gradOut, ok := getGPUBufferFromHandle(int64(gradOutputHandle))
	if !ok {
		return errJSON("invalid gradOutput buffer handle")
	}
	q, ok := getGPUBufferFromHandle(int64(qHandle))
	if !ok {
		return errJSON("invalid Q buffer handle")
	}
	k, ok := getGPUBufferFromHandle(int64(kHandle))
	if !ok {
		return errJSON("invalid K buffer handle")
	}
	v, ok := getGPUBufferFromHandle(int64(vHandle))
	if !ok {
		return errJSON("invalid V buffer handle")
	}
	dq, ok := getGPUBufferFromHandle(int64(dQHandle))
	if !ok {
		return errJSON("invalid dQ buffer handle")
	}
	dk, ok := getGPUBufferFromHandle(int64(dKHandle))
	if !ok {
		return errJSON("invalid dK buffer handle")
	}
	dv, ok := getGPUBufferFromHandle(int64(dVHandle))
	if !ok {
		return errJSON("invalid dV buffer handle")
	}
	if err := n.GPUContext.DispatchMHABackward(int(batchSize), int(numHeads), int(numKVHeads), int(headDim), int(seqLen), float32(scale), gradOut, q, k, v, dq, dk, dv); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

// ──────────────────────────────────────────────────────────────────────────────
// Apply Gradients
// ──────────────────────────────────────────────────────────────────────────────

//export LoomDispatchApplyGradients
func LoomDispatchApplyGradients(networkHandle C.longlong, size C.int, lr C.float, weightHandle C.longlong, gradHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if n.GPUContext == nil {
		return errJSON("WGPU not initialized")
	}
	wt, ok := getGPUBufferFromHandle(int64(weightHandle))
	if !ok {
		return errJSON("invalid weight buffer handle")
	}
	grad, ok := getGPUBufferFromHandle(int64(gradHandle))
	if !ok {
		return errJSON("invalid grad buffer handle")
	}
	if err := n.GPUContext.DispatchApplyGradients(int(size), float32(lr), wt, grad); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

// ──────────────────────────────────────────────────────────────────────────────
// MSE Grad + Partial Loss
// ──────────────────────────────────────────────────────────────────────────────

//export LoomDispatchMSEGradPartialLoss
func LoomDispatchMSEGradPartialLoss(networkHandle C.longlong, size C.int, outputHandle C.longlong, targetHandle C.longlong, gradHandle C.longlong, partialsHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if n.GPUContext == nil {
		return errJSON("WGPU not initialized")
	}
	out, ok := getGPUBufferFromHandle(int64(outputHandle))
	if !ok {
		return errJSON("invalid output buffer handle")
	}
	tgt, ok := getGPUBufferFromHandle(int64(targetHandle))
	if !ok {
		return errJSON("invalid target buffer handle")
	}
	grad, ok := getGPUBufferFromHandle(int64(gradHandle))
	if !ok {
		return errJSON("invalid grad buffer handle")
	}
	partials, ok := getGPUBufferFromHandle(int64(partialsHandle))
	if !ok {
		return errJSON("invalid partials buffer handle")
	}
	if err := n.GPUContext.DispatchMSEGradPartialLoss(int(size), out, tgt, grad, partials); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

// ──────────────────────────────────────────────────────────────────────────────
// Layer-level Dispatch (forward + backward)
// ──────────────────────────────────────────────────────────────────────────────

//export LoomDispatchForwardLayer
func LoomDispatchForwardLayer(networkHandle C.longlong, layerIdx C.int, batchSize C.int, inputHandle C.longlong, outputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if n.GPUContext == nil {
		return errJSON("WGPU not initialized")
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
	if err := n.GPUContext.DispatchForwardLayer(&n.Layers[int(layerIdx)], int(batchSize), in, out); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomDispatchBackwardLayer
func LoomDispatchBackwardLayer(networkHandle C.longlong, layerIdx C.int, batchSize C.int, gradOutHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong, dxHandle C.longlong, dwHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if n.GPUContext == nil {
		return errJSON("WGPU not initialized")
	}
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	gradOut, ok := getGPUBufferFromHandle(int64(gradOutHandle))
	if !ok {
		return errJSON("invalid gradOut buffer handle")
	}
	in, ok := getGPUBufferFromHandle(int64(inputHandle))
	if !ok {
		return errJSON("invalid input buffer handle")
	}
	preAct, ok := getGPUBufferFromHandle(int64(preActHandle))
	if !ok {
		return errJSON("invalid preAct buffer handle")
	}
	dx, ok := getGPUBufferFromHandle(int64(dxHandle))
	if !ok {
		return errJSON("invalid dx buffer handle")
	}
	dw, ok := getGPUBufferFromHandle(int64(dwHandle))
	if !ok {
		return errJSON("invalid dw buffer handle")
	}
	if err := n.GPUContext.DispatchBackwardLayer(&n.Layers[int(layerIdx)], int(batchSize), gradOut, in, preAct, dx, dw); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

// ──────────────────────────────────────────────────────────────────────────────
// Activation Dispatch
// ──────────────────────────────────────────────────────────────────────────────

//export LoomDispatchActivation
func LoomDispatchActivation(networkHandle C.longlong, size C.int, activation C.int, inputHandle C.longlong, outputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if n.GPUContext == nil {
		return errJSON("WGPU not initialized")
	}
	in, ok := getGPUBufferFromHandle(int64(inputHandle))
	if !ok {
		return errJSON("invalid input buffer handle")
	}
	out, ok := getGPUBufferFromHandle(int64(outputHandle))
	if !ok {
		return errJSON("invalid output buffer handle")
	}
	if err := n.GPUContext.DispatchActivation(int(size), poly.ActivationType(activation), in, out); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomDispatchActivationBackward
func LoomDispatchActivationBackward(networkHandle C.longlong, size C.int, activation C.int, gradOutHandle C.longlong, preActHandle C.longlong, gradInHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if n.GPUContext == nil {
		return errJSON("WGPU not initialized")
	}
	gradOut, ok := getGPUBufferFromHandle(int64(gradOutHandle))
	if !ok {
		return errJSON("invalid gradOut buffer handle")
	}
	preAct, ok := getGPUBufferFromHandle(int64(preActHandle))
	if !ok {
		return errJSON("invalid preAct buffer handle")
	}
	gradIn, ok := getGPUBufferFromHandle(int64(gradInHandle))
	if !ok {
		return errJSON("invalid gradIn buffer handle")
	}
	if err := n.GPUContext.DispatchActivationBackward(int(size), poly.ActivationType(activation), gradOut, preAct, gradIn); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}
