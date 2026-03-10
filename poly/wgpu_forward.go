package poly

import (
	"fmt"
	"runtime"

	"github.com/openfluke/webgpu/wgpu"
)

func getGPUBuffer(ws *WeightStore, dtype DType) (*wgpu.Buffer, error) {
	if ws == nil {
		return nil, fmt.Errorf("WeightStore is nil")
	}
	val, ok := ws.GPUWeights[dtype]
	if !ok || val == nil {
		return nil, fmt.Errorf("GPU buffer for DType %v not found", dtype)
	}
	buf, ok := val.(*wgpu.Buffer)
	if !ok {
		return nil, fmt.Errorf("GPU buffer for DType %v is not a *wgpu.Buffer", dtype)
	}
	return buf, nil
}

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
		eWeights, ok := t.Network.GPUEmbeddings.(*wgpu.Buffer)
		if !ok || eWeights == nil {
			return nil, fmt.Errorf("GPU embeddings not loaded")
		}
		if err := ctx.DispatchEmbedding(t.VocabSize, t.HiddenSize, numTokens, tBuf, eWeights, currentBuf); err != nil {
			return nil, err
		}
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
		weightBuf1, err := getGPUBuffer(lNorm1.WeightStore, DTypeFloat32)
		if err != nil {
			return nil, fmt.Errorf("layer %d Norm1 sync error: %w", b, err)
		}
		if err := ctx.DispatchRMSNorm(numTokens, lNorm1.InputHeight, 1e-5, currentBuf, weightBuf1, norm1Out); err != nil {
			return nil, err
		}

		residual := currentBuf

		// --- MHA ---
		lMHA := &t.Network.Layers[base+1]
		kvDim := lMHA.NumKVHeads * lMHA.HeadDim
		qBuf := ctx.GetActivationBuffer("q_proj", uint64(numTokens*lMHA.DModel*4), wgpu.BufferUsageStorage)
		kBuf := ctx.GetActivationBuffer("k_proj", uint64(numTokens*kvDim*4), wgpu.BufferUsageStorage)
		vBuf := ctx.GetActivationBuffer("v_proj", uint64(numTokens*kvDim*4), wgpu.BufferUsageStorage)

		wqBuf, errQ := getGPUBuffer(lMHA.WeightStore, DType(200))
		wkBuf, errK := getGPUBuffer(lMHA.WeightStore, DType(201))
		wvBuf, errV := getGPUBuffer(lMHA.WeightStore, DType(202))
		if errQ != nil || errK != nil || errV != nil {
			return nil, fmt.Errorf("layer %d MHA sync error: %v, %v, %v", b, errQ, errK, errV)
		}

		tileSize := ctx.GPUTileSize
		if tileSize <= 0 {
			tileSize = 32
		}

		sq := lMHA.WeightStore.GPUScales[DType(200)]
		sk := lMHA.WeightStore.GPUScales[DType(201)]
		sv := lMHA.WeightStore.GPUScales[DType(202)]

		if lMHA.DType == DTypeInt4 && sq != nil && sk != nil && sv != nil {
			if err := ctx.DispatchDenseQ4(numTokens, lMHA.DModel, lMHA.DModel, norm1Out, sq, wqBuf, qBuf, tileSize); err != nil { return nil, err }
			if err := ctx.DispatchDenseQ4(numTokens, lMHA.DModel, kvDim, norm1Out, sk, wkBuf, kBuf, tileSize); err != nil { return nil, err }
			if err := ctx.DispatchDenseQ4(numTokens, lMHA.DModel, kvDim, norm1Out, sv, wvBuf, vBuf, tileSize); err != nil { return nil, err }
		} else {
			if err := ctx.DispatchDense(numTokens, lMHA.DModel, lMHA.DModel, norm1Out, wqBuf, qBuf, tileSize); err != nil { return nil, err }
			if err := ctx.DispatchDense(numTokens, lMHA.DModel, kvDim, norm1Out, wkBuf, kBuf, tileSize); err != nil { return nil, err }
			if err := ctx.DispatchDense(numTokens, lMHA.DModel, kvDim, norm1Out, wvBuf, vBuf, tileSize); err != nil { return nil, err }
		}

		theta := float32(10000.0)
		if lMHA.RoPEFreqBase > 0 { theta = float32(lMHA.RoPEFreqBase) }
		if err := ctx.DispatchRoPE(numTokens, lMHA.HeadDim, lMHA.NumHeads, lMHA.KVOffset, theta, qBuf); err != nil { return nil, err }
		if err := ctx.DispatchRoPE(numTokens, lMHA.HeadDim, lMHA.NumKVHeads, lMHA.KVOffset, theta, kBuf); err != nil { return nil, err }

		kCacheBuf, okK := lMHA.GPUKVCacheK.(*wgpu.Buffer)
		vCacheBuf, okV := lMHA.GPUKVCacheV.(*wgpu.Buffer)
		if !okK || kCacheBuf == nil || !okV || vCacheBuf == nil {
			return nil, fmt.Errorf("layer %d GPU KV Cache not initialized", b)
		}
		if err := ctx.DispatchKVUpdate(lMHA.KVOffset, lMHA.HeadDim, lMHA.MaxSeqLen, lMHA.NumKVHeads, numTokens, kCacheBuf, vCacheBuf, kBuf, vBuf); err != nil {
			return nil, err
		}

		attnOut := ctx.GetActivationBuffer("attn_out", uint64(numTokens*lMHA.DModel*4), wgpu.BufferUsageStorage)
		if err := ctx.DispatchMHA(lMHA.NumHeads, lMHA.NumKVHeads, lMHA.HeadDim, numTokens, lMHA.KVOffset, lMHA.MaxSeqLen, qBuf, kCacheBuf, vCacheBuf, attnOut, tileSize); err != nil {
			return nil, err
		}

		woBuf, errO := getGPUBuffer(lMHA.WeightStore, DType(203))
		if errO != nil {
			return nil, fmt.Errorf("layer %d MHA O-proj sync error: %w", b, errO)
		}
		mhaOut := ctx.GetActivationBuffer("hidden_B", uint64(numTokens*lMHA.DModel*4), wgpu.BufferUsageStorage)
		so := lMHA.WeightStore.GPUScales[DType(203)]
		if lMHA.DType == DTypeInt4 && so != nil {
			if err := ctx.DispatchDenseQ4(numTokens, lMHA.DModel, lMHA.DModel, attnOut, so, woBuf, mhaOut, tileSize); err != nil { return nil, err }
		} else {
			if err := ctx.DispatchDense(numTokens, lMHA.DModel, lMHA.DModel, attnOut, woBuf, mhaOut, tileSize); err != nil { return nil, err }
		}

		lMHA.KVOffset += numTokens
		if err := ctx.DispatchResidual(numTokens*lMHA.DModel, mhaOut, residual); err != nil {
			return nil, err
		}
		currentBuf = mhaOut

		// --- Norm 2 ---
		lNorm2 := &t.Network.Layers[base+2]
		norm2Out := ctx.GetActivationBuffer("norm_out", uint64(numTokens*lNorm2.InputHeight*4), wgpu.BufferUsageStorage)
		weightBuf2, errN2 := getGPUBuffer(lNorm2.WeightStore, DTypeFloat32)
		if errN2 != nil {
			return nil, fmt.Errorf("layer %d Norm2 sync error: %w", b, errN2)
		}
		if err := ctx.DispatchRMSNorm(numTokens, lNorm2.InputHeight, 1e-5, currentBuf, weightBuf2, norm2Out); err != nil {
			return nil, err
		}

		residual = currentBuf

		// --- SwiGLU ---
		lMLP := &t.Network.Layers[base+3]
		interOut := ctx.GetActivationBuffer("mlp_inter", uint64(numTokens*lMLP.OutputHeight*4), wgpu.BufferUsageStorage)
		gW, errG := getGPUBuffer(lMLP.WeightStore, DType(100))
		uW, errU := getGPUBuffer(lMLP.WeightStore, DType(101))
		dW, errD := getGPUBuffer(lMLP.WeightStore, DType(102))
		if errG != nil || errU != nil || errD != nil {
			return nil, fmt.Errorf("layer %d MLP sync error: %v, %v, %v", b, errG, errU, errD)
		}

		tileSize = ctx.GPUTileSize
		if tileSize <= 0 { tileSize = 32 }

		if lMLP.DType == DTypeInt4 {
			sg := lMLP.WeightStore.GPUScales[DType(100)]
			su := lMLP.WeightStore.GPUScales[DType(101)]
			if sg != nil && su != nil {
				if err := ctx.DispatchSwiGLUQ4(numTokens, lMLP.InputHeight, lMLP.OutputHeight, norm2Out, sg, gW, su, uW, interOut, tileSize); err != nil { return nil, err }
			} else {
				if err := ctx.DispatchSwiGLU(numTokens, lMLP.InputHeight, lMLP.OutputHeight, norm2Out, gW, uW, interOut, tileSize); err != nil { return nil, err }
			}
		} else {
			if err := ctx.DispatchSwiGLU(numTokens, lMLP.InputHeight, lMLP.OutputHeight, norm2Out, gW, uW, interOut, tileSize); err != nil { return nil, err }
		}

		mlpOut := ctx.GetActivationBuffer("hidden_A", uint64(numTokens*lMLP.InputHeight*4), wgpu.BufferUsageStorage)
		sd := lMLP.WeightStore.GPUScales[DType(102)]
		if lMLP.DType == DTypeInt4 && sd != nil {
			if err := ctx.DispatchDenseQ4(numTokens, lMLP.OutputHeight, lMLP.InputHeight, interOut, sd, dW, mlpOut, tileSize); err != nil { return nil, err }
		} else {
			if err := ctx.DispatchDense(numTokens, lMLP.OutputHeight, lMLP.InputHeight, interOut, dW, mlpOut, tileSize); err != nil { return nil, err }
		}

		if err := ctx.DispatchResidual(numTokens*lMLP.InputHeight, mlpOut, residual); err != nil {
			return nil, err
		}
		currentBuf = mlpOut
	}

	// 3. Final Norm on GPU
	if t.finalNormLayer != nil && t.finalNormLayer.IsGPUResident {
		fNormOut := ctx.GetActivationBuffer("norm_out", uint64(numTokens*t.HiddenSize*4), wgpu.BufferUsageStorage)
		fW := t.finalNormLayer.WeightStore.GPUWeights[DTypeFloat32].(*wgpu.Buffer)
		if err := ctx.DispatchRMSNorm(numTokens, t.HiddenSize, 1e-5, currentBuf, fW, fNormOut); err != nil {
			return nil, err
		}
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
		lmW, ok := t.Network.GPULMHead.(*wgpu.Buffer)
		if !ok || lmW == nil {
			return nil, fmt.Errorf("GPU LM Head not loaded")
		}
		if t.Network.Layers[0].DType == DTypeInt4 { // Use first layer's dtype hint
			// We need a way to store LMHead scales. 
			// Let's assume LMHead is not quantized for now or use a global scale map.
			if err := ctx.DispatchDense(dispatchTokens, t.HiddenSize, t.VocabSize, effectiveInput, lmW, lmOut, ctx.GPUTileSize); err != nil { return nil, err }
		} else {
			if err := ctx.DispatchDense(dispatchTokens, t.HiddenSize, t.VocabSize, effectiveInput, lmW, lmOut, ctx.GPUTileSize); err != nil { return nil, err }
		}
		
		currentBuf = lmOut
		isReturningLogits = true
		downloadTokens = dispatchTokens
		readOffset = 0 // We already Isolated the token or processed all
	} else if onlyLast && numTokens > 1 {
		downloadTokens = 1
		readOffset = uint64((numTokens - 1) * t.HiddenSize * 4)
	}

	dim := t.HiddenSize
	if isReturningLogits {
		dim = t.VocabSize
	}
	resSize := uint64(downloadTokens * dim * 4)
	stagingBuf := ctx.GetActivationBuffer("staging", resSize, wgpu.BufferUsageMapRead)

	// Perform the copy within the ACTIVE encoder before flushing
	ctx.ActiveEncoder.CopyBufferToBuffer(currentBuf, readOffset, stagingBuf, 0, resSize)
	
	ctx.FlushFrame()

	// 5. Wait for download
	done := make(chan struct{})
	stagingBuf.MapAsync(wgpu.MapModeRead, 0, resSize, func(status wgpu.BufferMapAsyncStatus) { close(done) })

	for {
		ctx.Device.Poll(false, nil) // use non-blocking poll
		select {
		case <-done:
			goto Finished
		default:
			runtime.Gosched() // yield to other goroutines
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
