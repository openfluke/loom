package poly

import (
	"fmt"
	"github.com/openfluke/webgpu/wgpu"
)

// ForwardWGPU handles both prefill (multi-token) and decode (single-token) GPU forward passes.
// All layer dispatches are recorded into a single CommandEncoder (BeginFrame/FlushFrame),
// reducing GPU submission overhead from ~150+ submits/token to just 1 submit + 1 download.
// ForwardTokenIDsWGPU is the "true" GPU residency path. If tokens are provided, 
// embedding lookup happens on GPU. If final norm/LM head are synced, they run on GPU too.
func (t *Transformer[T]) ForwardTokenIDsWGPU(tokens []uint32, input *Tensor[T], computeLogits bool, onlyLast bool) (*Tensor[T], error) {
	if t.Network.GPUContext == nil {
		return nil, fmt.Errorf("GPU context not initialized")
	}
	ctx := t.Network.GPUContext

	numTokens := 1
	if tokens != nil {
		numTokens = len(tokens)
	} else if input != nil && len(input.Shape) >= 2 {
		numTokens = input.Shape[len(input.Shape)-2]
	} else if input != nil && len(input.Data) > 0 {
		numTokens = 1
	}

	var currentBuf *wgpu.Buffer

	// 1. Embedding / Initial Hidden State
	if tokens != nil && t.Network.GPUEmbeddings != nil {
		// Optimization: Use WriteBuffer instead of CreateBufferInit to avoid sync allocation
		tBuf := ctx.GetActivationBuffer("token_ids", uint64(numTokens*4), wgpu.BufferUsageStorage)
		ctx.Queue.WriteBuffer(tBuf, 0, wgpu.ToBytes(tokens))

		currentBuf = ctx.GetActivationBuffer("hidden_A", uint64(numTokens*t.HiddenSize*4), wgpu.BufferUsageStorage)
		eWeights := t.Network.GPUEmbeddings.(*wgpu.Buffer)
		ctx.DispatchEmbedding(t.VocabSize, t.HiddenSize, numTokens, tBuf, eWeights, currentBuf)
	} else if input != nil {
		// Fallback: upload from CPU
		inData := ConvertTensor[T, float32](input).Data
		currentBuf = ctx.GetActivationBuffer("hidden_A", uint64(numTokens*t.HiddenSize*4), wgpu.BufferUsageStorage)
		ctx.Queue.WriteBuffer(currentBuf, 0, wgpu.ToBytes(inData))
	} else {
		return nil, fmt.Errorf("no input tokens or data provided")
	}

	numBlocks := len(t.Network.Layers) / 4

	// 2. Transformer Blocks
	if err := ctx.BeginFrame(); err != nil {
		return nil, fmt.Errorf("BeginFrame failed: %w", err)
	}

	for b := 0; b < numBlocks; b++ {
		base := b * 4

		// --- Norm 1 ---
		lNorm1 := &t.Network.Layers[base+0]
		norm1Out := ctx.GetActivationBuffer("norm_out", uint64(numTokens*lNorm1.InputHeight*4), wgpu.BufferUsageStorage)
		weightBuf1 := lNorm1.WeightStore.GPUWeights[DTypeFloat32].(*wgpu.Buffer)
		ctx.DispatchRMSNorm(numTokens, lNorm1.InputHeight, 1e-5, currentBuf, weightBuf1, norm1Out)

		residual := currentBuf

		// --- MHA ---
		lMHA := &t.Network.Layers[base+1]
		kvDim := lMHA.NumKVHeads * lMHA.HeadDim
		qBuf := ctx.GetActivationBuffer("q_proj", uint64(numTokens*lMHA.DModel*4), wgpu.BufferUsageStorage)
		kBuf := ctx.GetActivationBuffer("k_proj", uint64(numTokens*kvDim*4), wgpu.BufferUsageStorage)
		vBuf := ctx.GetActivationBuffer("v_proj", uint64(numTokens*kvDim*4), wgpu.BufferUsageStorage)

		wqBuf := lMHA.WeightStore.GPUWeights[DType(200)].(*wgpu.Buffer)
		wkBuf := lMHA.WeightStore.GPUWeights[DType(201)].(*wgpu.Buffer)
		wvBuf := lMHA.WeightStore.GPUWeights[DType(202)].(*wgpu.Buffer)

		tileSize := ctx.GPUTileSize
		if tileSize <= 0 { tileSize = 32 }

		ctx.DispatchDense(numTokens, lMHA.DModel, lMHA.DModel, norm1Out, wqBuf, qBuf, tileSize)
		ctx.DispatchDense(numTokens, lMHA.DModel, kvDim, norm1Out, wkBuf, kBuf, tileSize)
		ctx.DispatchDense(numTokens, lMHA.DModel, kvDim, norm1Out, wvBuf, vBuf, tileSize)

		theta := float32(10000.0)
		if lMHA.RoPEFreqBase > 0 { theta = float32(lMHA.RoPEFreqBase) }
		ctx.DispatchRoPE(numTokens, lMHA.HeadDim, lMHA.NumHeads, lMHA.KVOffset, theta, qBuf)
		ctx.DispatchRoPE(numTokens, lMHA.HeadDim, lMHA.NumKVHeads, lMHA.KVOffset, theta, kBuf)

		kCacheBuf := lMHA.GPUKVCacheK.(*wgpu.Buffer)
		vCacheBuf := lMHA.GPUKVCacheV.(*wgpu.Buffer)
		ctx.DispatchKVUpdate(lMHA.KVOffset, lMHA.HeadDim, lMHA.MaxSeqLen, lMHA.NumKVHeads, numTokens, kCacheBuf, vCacheBuf, kBuf, vBuf)

		attnOut := ctx.GetActivationBuffer("attn_out", uint64(numTokens*lMHA.DModel*4), wgpu.BufferUsageStorage)
		ctx.DispatchMHA(lMHA.NumHeads, lMHA.NumKVHeads, lMHA.HeadDim, numTokens, lMHA.KVOffset, lMHA.MaxSeqLen, qBuf, kCacheBuf, vCacheBuf, attnOut, tileSize)

		woBuf := lMHA.WeightStore.GPUWeights[DType(203)].(*wgpu.Buffer)
		mhaOut := ctx.GetActivationBuffer("hidden_B", uint64(numTokens*lMHA.DModel*4), wgpu.BufferUsageStorage)
		ctx.DispatchDense(numTokens, lMHA.DModel, lMHA.DModel, attnOut, woBuf, mhaOut, tileSize)

		lMHA.KVOffset += numTokens
		ctx.DispatchResidual(numTokens*lMHA.DModel, mhaOut, residual)
		currentBuf = mhaOut

		// --- Norm 2 ---
		lNorm2 := &t.Network.Layers[base+2]
		norm2Out := ctx.GetActivationBuffer("norm_out", uint64(numTokens*lNorm2.InputHeight*4), wgpu.BufferUsageStorage)
		weightBuf2 := lNorm2.WeightStore.GPUWeights[DTypeFloat32].(*wgpu.Buffer)
		ctx.DispatchRMSNorm(numTokens, lNorm2.InputHeight, 1e-5, currentBuf, weightBuf2, norm2Out)

		residual = currentBuf

		// --- SwiGLU ---
		lMLP := &t.Network.Layers[base+3]
		interOut := ctx.GetActivationBuffer("mlp_inter", uint64(numTokens*lMLP.OutputHeight*4), wgpu.BufferUsageStorage)
		gW := lMLP.WeightStore.GPUWeights[DType(100)].(*wgpu.Buffer)
		uW := lMLP.WeightStore.GPUWeights[DType(101)].(*wgpu.Buffer)
		dW := lMLP.WeightStore.GPUWeights[DType(102)].(*wgpu.Buffer)

		ctx.DispatchSwiGLU(numTokens, lMLP.InputHeight, lMLP.OutputHeight, norm2Out, gW, uW, interOut, tileSize)

		mlpOut := ctx.GetActivationBuffer("hidden_A", uint64(numTokens*lMLP.InputHeight*4), wgpu.BufferUsageStorage)
		ctx.DispatchDense(numTokens, lMLP.OutputHeight, lMLP.InputHeight, interOut, dW, mlpOut, tileSize)

		ctx.DispatchResidual(numTokens*lMLP.InputHeight, mlpOut, residual)
		currentBuf = mlpOut
	}

	// 3. Final Norm on GPU
	if t.finalNormLayer != nil && t.finalNormLayer.IsGPUResident {
		fNormOut := ctx.GetActivationBuffer("norm_out", uint64(numTokens*t.HiddenSize*4), wgpu.BufferUsageStorage)
		fW := t.finalNormLayer.WeightStore.GPUWeights[DTypeFloat32].(*wgpu.Buffer)
		ctx.DispatchRMSNorm(numTokens, t.HiddenSize, 1e-5, currentBuf, fW, fNormOut)
		currentBuf = fNormOut
	}

	// 4. LM Head on GPU if requested and available
	isReturningLogits := false
	downloadTokens := numTokens
	readOffset := uint64(0)

	if computeLogits && t.Network.GPULMHead != nil {
		effectiveInput := currentBuf
		dispatchTokens := numTokens

		if onlyLast && numTokens > 1 {
			// Optimization: Copy ONLY the last token's hidden state to a scratch buffer
			// This avoids running the heavy LM Head matrix mult on the entire prompt.
			scratchSize := uint64(t.HiddenSize * 4)
			scratchBuf := ctx.GetActivationBuffer("hidden_scratch", scratchSize, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst|wgpu.BufferUsageCopySrc)
			offset := uint64((numTokens - 1) * t.HiddenSize * 4)
			
			// Use the existing command encoder to perform a sync-less copy
			ctx.ActiveEncoder.CopyBufferToBuffer(currentBuf, offset, scratchBuf, 0, scratchSize)
			
			effectiveInput = scratchBuf
			dispatchTokens = 1
		}

		lmOut := ctx.GetActivationBuffer("logits", uint64(dispatchTokens*t.VocabSize*4), wgpu.BufferUsageStorage)
		lmW := t.Network.GPULMHead.(*wgpu.Buffer)
		ctx.DispatchDense(dispatchTokens, t.HiddenSize, t.VocabSize, effectiveInput, lmW, lmOut, ctx.GPUTileSize)
		
		currentBuf = lmOut
		isReturningLogits = true
		downloadTokens = dispatchTokens
		readOffset = 0 // We already Isolated the token or processed all
	} else if onlyLast && numTokens > 1 {
		downloadTokens = 1
		readOffset = uint64((numTokens - 1) * t.HiddenSize * 4)
	}

	ctx.FlushFrame()

	// 5. Download
	dim := t.HiddenSize
	if isReturningLogits {
		dim = t.VocabSize
	}
	resSize := uint64(downloadTokens * dim * 4)
	stagingBuf := ctx.GetActivationBuffer("staging", resSize, wgpu.BufferUsageMapRead)

	encoder, _ := ctx.Device.CreateCommandEncoder(nil)
	encoder.CopyBufferToBuffer(currentBuf, readOffset, stagingBuf, 0, resSize)
	cmd, _ := encoder.Finish(nil)
	ctx.Queue.Submit(cmd)

	done := make(chan struct{})
	stagingBuf.MapAsync(wgpu.MapModeRead, 0, resSize, func(status wgpu.BufferMapAsyncStatus) { close(done) })

	for {
		ctx.Device.Poll(true, nil)
		select {
		case <-done: goto Finished
		default:
		}
	}

Finished:
	outBytes := stagingBuf.GetMappedRange(0, uint(resSize))
	outData := wgpu.FromBytes[float32](outBytes)
	stagingBuf.Unmap()

	shape := []int{downloadTokens, dim}
	return NewTensorFromSlice[T](CastWeights[T](outData), shape...), nil
}

func (t *Transformer[T]) ForwardWGPU(input *Tensor[T]) (*Tensor[T], error) {
	return t.ForwardTokenIDsWGPU(nil, input, false, false)
}
