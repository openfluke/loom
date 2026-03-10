package poly

import (
	"fmt"
	"github.com/openfluke/webgpu/wgpu"
)

// ForwardWGPU handles both prefill (multi-token) and decode (single-token) GPU forward passes.
func (t *Transformer[T]) ForwardWGPU(input *Tensor[T]) (*Tensor[T], error) {
	if t.Network.GPUContext == nil {
		return nil, fmt.Errorf("GPU context not initialized")
	}
	ctx := t.Network.GPUContext
	
	numTokens := 1
	if len(input.Shape) >= 2 {
		numTokens = input.Shape[len(input.Shape)-2]
	}

	// 1. Initial State Upload
	inData := ConvertTensor[T, float32](input).Data
	currentBuf := ctx.GetActivationBuffer("hidden_A", uint64(numTokens * t.HiddenSize * 4), wgpu.BufferUsageStorage)
	ctx.Queue.WriteBuffer(currentBuf, 0, wgpu.ToBytes(inData))

	numBlocks := len(t.Network.Layers) / 4

	for b := 0; b < numBlocks; b++ {
		base := b * 4
		
		// --- 1. Norm 1 ---
		lNorm1 := &t.Network.Layers[base+0]
		norm1Out := ctx.GetActivationBuffer("norm_out", uint64(numTokens * lNorm1.InputHeight * 4), wgpu.BufferUsageStorage)
		weightBuf1 := lNorm1.WeightStore.GPUWeights[DTypeFloat32].(*wgpu.Buffer)
		ctx.DispatchRMSNorm(numTokens, lNorm1.InputHeight, 1e-5, currentBuf, weightBuf1, norm1Out)

		residual := currentBuf
		
		// --- 2. MHA ---
		lMHA := &t.Network.Layers[base+1]
		kvDim := lMHA.NumKVHeads * lMHA.HeadDim
		
		qBuf := ctx.GetActivationBuffer("q_proj", uint64(numTokens * lMHA.DModel * 4), wgpu.BufferUsageStorage)
		kBuf := ctx.GetActivationBuffer("k_proj", uint64(numTokens * kvDim * 4), wgpu.BufferUsageStorage)
		vBuf := ctx.GetActivationBuffer("v_proj", uint64(numTokens * kvDim * 4), wgpu.BufferUsageStorage)

		wqBuf := lMHA.WeightStore.GPUWeights[DType(200)].(*wgpu.Buffer)
		wkBuf := lMHA.WeightStore.GPUWeights[DType(201)].(*wgpu.Buffer)
		wvBuf := lMHA.WeightStore.GPUWeights[DType(202)].(*wgpu.Buffer)
		
		// Use the adapter-derived tile size for optimal GPU shared memory usage.
		tileSize := ctx.GPUTileSize
		if tileSize <= 0 { tileSize = 32 } // defensive fallback

		ctx.DispatchDense(numTokens, lMHA.DModel, lMHA.DModel, norm1Out, wqBuf, qBuf, tileSize)
		ctx.DispatchDense(numTokens, lMHA.DModel, kvDim, norm1Out, wkBuf, kBuf, tileSize)
		ctx.DispatchDense(numTokens, lMHA.DModel, kvDim, norm1Out, wvBuf, vBuf, tileSize)

		// Apply RoPE on Q and K if needed
		roPETheta := float32(10000.0) // Basic default, ideally mapped from layer config
		if t.Network.Layers[base+1].RoPEFreqBase > 0 {
			roPETheta = float32(t.Network.Layers[base+1].RoPEFreqBase)
		}
		if roPETheta > 0 {
			ctx.DispatchRoPE(numTokens, lMHA.HeadDim, lMHA.NumHeads, lMHA.KVOffset, roPETheta, qBuf)
			ctx.DispatchRoPE(numTokens, lMHA.HeadDim, lMHA.NumKVHeads, lMHA.KVOffset, roPETheta, kBuf)
		}

		kCacheBuf := lMHA.GPUKVCacheK.(*wgpu.Buffer)
		vCacheBuf := lMHA.GPUKVCacheV.(*wgpu.Buffer)
		ctx.DispatchKVUpdate(lMHA.KVOffset, lMHA.HeadDim, lMHA.MaxSeqLen, lMHA.NumKVHeads, numTokens, kCacheBuf, vCacheBuf, kBuf, vBuf)
		
		attnOut := ctx.GetActivationBuffer("attn_out", uint64(numTokens * lMHA.DModel * 4), wgpu.BufferUsageStorage)
		ctx.DispatchMHA(lMHA.NumHeads, lMHA.NumKVHeads, lMHA.HeadDim, numTokens, lMHA.KVOffset, lMHA.MaxSeqLen, qBuf, kCacheBuf, vCacheBuf, attnOut, tileSize)
		
		woBuf := lMHA.WeightStore.GPUWeights[DType(203)].(*wgpu.Buffer)
		mhaOut := ctx.GetActivationBuffer("hidden_B", uint64(numTokens * lMHA.DModel * 4), wgpu.BufferUsageStorage)
		ctx.DispatchDense(numTokens, lMHA.DModel, lMHA.DModel, attnOut, woBuf, mhaOut, tileSize)

		lMHA.KVOffset += numTokens
		
		// Add Residual 1 (mhaOut += residual)
		ctx.DispatchResidual(numTokens * lMHA.DModel, mhaOut, residual)
		currentBuf = mhaOut
		
		// --- 3. Norm 2 ---
		lNorm2 := &t.Network.Layers[base+2]
		norm2Out := ctx.GetActivationBuffer("norm_out", uint64(numTokens * lNorm2.InputHeight * 4), wgpu.BufferUsageStorage)
		weightBuf2 := lNorm2.WeightStore.GPUWeights[DTypeFloat32].(*wgpu.Buffer)
		ctx.DispatchRMSNorm(numTokens, lNorm2.InputHeight, 1e-5, currentBuf, weightBuf2, norm2Out)

		residual = currentBuf

		// --- 4. SwiGLU / MLP ---
		lMLP := &t.Network.Layers[base+3]
		interOut := ctx.GetActivationBuffer("mlp_inter", uint64(numTokens * lMLP.OutputHeight * 4), wgpu.BufferUsageStorage)
		
		gateW := lMLP.WeightStore.GPUWeights[DType(100)].(*wgpu.Buffer)
		upW := lMLP.WeightStore.GPUWeights[DType(101)].(*wgpu.Buffer)
		downW := lMLP.WeightStore.GPUWeights[DType(102)].(*wgpu.Buffer)
		
		mlpTile := ctx.GPUTileSize
		if mlpTile <= 0 { mlpTile = 32 } // defensive fallback

		// silu(gate) * up
		ctx.DispatchSwiGLU(numTokens, lMLP.InputHeight, lMLP.OutputHeight, norm2Out, gateW, upW, interOut, mlpTile)
		
		// Down projection
		mlpOut := ctx.GetActivationBuffer("hidden_A", uint64(numTokens * lMLP.InputHeight * 4), wgpu.BufferUsageStorage)
		ctx.DispatchDense(numTokens, lMLP.OutputHeight, lMLP.InputHeight, interOut, downW, mlpOut, mlpTile)

		// --- 5. Residual 2 ---
		ctx.DispatchResidual(numTokens * lMLP.InputHeight, mlpOut, residual)
		currentBuf = mlpOut
	}
	
	// Final Download
	resSize := uint64(numTokens * t.HiddenSize * 4)
	stagingBuf := ctx.GetActivationBuffer("staging", resSize, wgpu.BufferUsageMapRead)

	encoder, _ := ctxEncoder(ctx)
	encoder.CopyBufferToBuffer(currentBuf, 0, stagingBuf, 0, resSize)
	cmd, _ := encoder.Finish(nil)
	ctx.Queue.Submit(cmd)

	done := make(chan struct{})
	stagingBuf.MapAsync(wgpu.MapModeRead, 0, resSize, func(status wgpu.BufferMapAsyncStatus) {
		close(done)
	})

	for {
		ctx.Device.Poll(true, nil)
		select {
		case <-done:
			goto Finished
		default:
		}
	}

Finished:
	outBytes := stagingBuf.GetMappedRange(0, uint(resSize))
	outData := wgpu.FromBytes[float32](outBytes)
	stagingBuf.Unmap()
	
	return NewTensorFromSlice[T](CastWeights[T](outData), input.Shape...), nil
}
