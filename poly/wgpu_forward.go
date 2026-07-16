package poly

import (
	"fmt"
	"runtime"
	"time"

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
//
// When Transformer.gpuReturnGreedyToken is set (see ForwardSampleGreedyTokenWGPU), the LM-head
// logits stay on GPU, an ArgMax shader picks the next token, and only a 4-byte u32 is mapped
// back — not the full vocab logits vector.
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

	anchorZ, anchorY, anchorX, anchorL := 0, 0, 0, 0
	if len(t.Network.Layers) > 0 {
		anchorZ = t.Network.Layers[0].Z
		anchorY = t.Network.Layers[0].Y
		anchorX = t.Network.Layers[0].X
		anchorL = t.Network.Layers[0].L
	}

	// Begin the shared encoder BEFORE embedding so decode is one Queue.Submit, not
	// embedding-submit + block-submit + MapAsync. Chunked greedy decode reuses an open frame.
	if !t.gpuChunkRecording {
		if err := ctx.BeginFrame(); err != nil {
			return nil, fmt.Errorf("BeginFrame failed: %w", err)
		}
	} else if ctx.ActiveEncoder == nil {
		return nil, fmt.Errorf("gpu chunk recording requires an open BeginFrame")
	}

	var currentBuf *wgpu.Buffer

	// 1. Embedding / Initial Hidden State
	if t.gpuUseDecodeTokenBuf && t.Network.GPUEmbeddings != nil {
		tBuf := ctx.GetActivationBuffer("decode_token", 64, wgpu.BufferUsageStorage)
		numTokens = 1
		currentBuf = ctx.GetActivationBuffer("hidden_A", uint64(numTokens*t.HiddenSize*4), wgpu.BufferUsageStorage)
		eWeights, ok := t.Network.GPUEmbeddings.(*wgpu.Buffer)
		if !ok || eWeights == nil {
			return nil, fmt.Errorf("GPU embeddings not loaded")
		}
		tEmb0 := time.Now()
		if err := ctx.DispatchEmbedding(t.VocabSize, t.HiddenSize, numTokens, tBuf, eWeights, currentBuf); err != nil {
			return nil, err
		}
		tEmb1 := time.Now()
		if t.Network.Tanhi != nil && t.Network.Tanhi.Enabled {
			embL := VolumetricLayer{
				Network:      t.Network,
				Type:         LayerEmbedding,
				DType:        DTypeFloat32,
				VocabSize:    t.VocabSize,
				EmbeddingDim: t.HiddenSize,
				Z:            anchorZ, Y: anchorY, X: anchorX, L: anchorL,
			}
			var sh []int
			if t.Network.Tanhi.SendShape {
				sh = TanhiGPULayerShapeHint(&embL, numTokens)
			}
			tanhiEmitWithConn(t.Network, "fwd", -1, &embL, tEmb0, tEmb1, sh, len(t.Embeddings))
		}
	} else if tokens != nil && t.Network.GPUEmbeddings != nil {
		// Optimization: Use WriteBuffer instead of CreateBufferInit to avoid sync allocation
		tBuf := ctx.GetActivationBuffer("token_ids", uint64(numTokens*4), wgpu.BufferUsageStorage)
		ctx.Queue.WriteBuffer(tBuf, 0, wgpu.ToBytes(tokens))

		currentBuf = ctx.GetActivationBuffer("hidden_A", uint64(numTokens*t.HiddenSize*4), wgpu.BufferUsageStorage)
		eWeights, ok := t.Network.GPUEmbeddings.(*wgpu.Buffer)
		if !ok || eWeights == nil {
			return nil, fmt.Errorf("GPU embeddings not loaded")
		}
		tEmb0 := time.Now()
		if err := ctx.DispatchEmbedding(t.VocabSize, t.HiddenSize, numTokens, tBuf, eWeights, currentBuf); err != nil {
			return nil, err
		}
		tEmb1 := time.Now()
		if t.Network.Tanhi != nil && t.Network.Tanhi.Enabled {
			embL := VolumetricLayer{
				Network:      t.Network,
				Type:         LayerEmbedding,
				DType:        DTypeFloat32,
				VocabSize:    t.VocabSize,
				EmbeddingDim: t.HiddenSize,
				Z:            anchorZ, Y: anchorY, X: anchorX, L: anchorL,
			}
			var sh []int
			if t.Network.Tanhi.SendShape {
				sh = TanhiGPULayerShapeHint(&embL, numTokens)
			}
			tanhiEmitWithConn(t.Network, "fwd", -1, &embL, tEmb0, tEmb1, sh, len(t.Embeddings))
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

	for b := 0; b < numBlocks; b++ {
		base := b * 4

		// --- Norm 1 ---
		lNorm1 := &t.Network.Layers[base+0]
		norm1Out := ctx.GetActivationBuffer(fmt.Sprintf("b%d_n1", b), uint64(numTokens*lNorm1.InputHeight*4), wgpu.BufferUsageStorage)
		weightBuf1, err := getGPUBuffer(lNorm1.WeightStore, DTypeFloat32)
		if err != nil {
			return nil, fmt.Errorf("layer %d Norm1 sync error: %w", b, err)
		}
		eps1 := float32(1e-6)
		if lNorm1.RMSNormEps > 0 {
			eps1 = float32(lNorm1.RMSNormEps)
		}
		tN1 := time.Now()
		if err := ctx.DispatchRMSNorm(numTokens, lNorm1.InputHeight, eps1, currentBuf, weightBuf1, norm1Out); err != nil {
			return nil, err
		}
		if t.Network.Tanhi != nil && t.Network.Tanhi.Enabled {
			var sh []int
			if t.Network.Tanhi.SendShape {
				sh = TanhiGPULayerShapeHint(lNorm1, numTokens)
			}
			tanhiEmit(t.Network, "fwd", base+0, lNorm1, tN1, time.Now(), sh)
		}

		residual := currentBuf

		// --- MHA ---
		tMHA0 := time.Now()
		lMHA := &t.Network.Layers[base+1]
		kvDim := lMHA.NumKVHeads * lMHA.HeadDim
		qDim := lMHA.QueryDim
		if qDim == 0 {
			qDim = lMHA.DModel
		}
		qBuf := ctx.GetActivationBuffer(fmt.Sprintf("b%d_q", b), uint64(numTokens*qDim*4), wgpu.BufferUsageStorage)
		kBuf := ctx.GetActivationBuffer(fmt.Sprintf("b%d_k", b), uint64(numTokens*kvDim*4), wgpu.BufferUsageStorage)
		vBuf := ctx.GetActivationBuffer(fmt.Sprintf("b%d_v", b), uint64(numTokens*kvDim*4), wgpu.BufferUsageStorage)

		wqBuf, errQ := getGPUBuffer(lMHA.WeightStore, DType(200))
		wkBuf, errK := getGPUBuffer(lMHA.WeightStore, DType(201))
		wvBuf, errV := getGPUBuffer(lMHA.WeightStore, DType(202))
		if errQ != nil || errK != nil || errV != nil {
			return nil, fmt.Errorf("layer %d MHA sync error: %v, %v, %v", b, errQ, errK, errV)
		}

		tileSize := lMHA.TileSize
		if tileSize <= 0 {
			tileSize = ctx.GPUTileSize
			if t.Network.EnableMultiCoreTiling {
				tileSize = lMHA.GetGPUMCTileSize(lMHA.DType)
			} else if lMHA.UseTiling {
				tileSize = lMHA.GetGPUSCTileSize(lMHA.DType)
			}
		}
		if tileSize <= 0 {
			tileSize = 32
		}

		sq := lMHA.WeightStore.GPUScales[DType(200)]
		sk := lMHA.WeightStore.GPUScales[DType(201)]
		sv := lMHA.WeightStore.GPUScales[DType(202)]

		if lMHA.DType == DTypeTernary {
			norm1Q, norm1Scale, err := bitNetQuantizeActivationGPU(ctx, fmt.Sprintf("b%d_n1_bitnet", b), numTokens, lMHA.DModel, norm1Out)
			if err != nil {
				return nil, err
			}
			qScale := bitNetGPUScaleValue(lMHA.WeightStore, WeightMHAQuery, 0)
			kScale := bitNetGPUScaleValue(lMHA.WeightStore, WeightMHAKey, qDim*lMHA.DModel)
			vScale := bitNetGPUScaleValue(lMHA.WeightStore, WeightMHAValue, qDim*lMHA.DModel+kvDim*lMHA.DModel)
			// Decode: one shared int8 act load → fused Q|K|V (Microsoft-style packed ternary).
			if numTokens == 1 && lMHA.DModel <= 8192 {
				if err := ctx.DispatchDecodeBitNetQKV(lMHA.DModel, qDim, kvDim, norm1Q, norm1Scale, wqBuf, wkBuf, wvBuf, qBuf, kBuf, vBuf, qScale, kScale, vScale); err != nil {
					return nil, err
				}
			} else {
				if err := ctx.DispatchDenseBitNetTernaryQuantized(numTokens, lMHA.DModel, qDim, norm1Q, norm1Scale, wqBuf, nil, qBuf, qScale, ActivationLinear, tileSize); err != nil {
					return nil, err
				}
				if err := ctx.DispatchDenseBitNetTernaryQuantized(numTokens, lMHA.DModel, kvDim, norm1Q, norm1Scale, wkBuf, nil, kBuf, kScale, ActivationLinear, tileSize); err != nil {
					return nil, err
				}
				if err := ctx.DispatchDenseBitNetTernaryQuantized(numTokens, lMHA.DModel, kvDim, norm1Q, norm1Scale, wvBuf, nil, vBuf, vScale, ActivationLinear, tileSize); err != nil {
					return nil, err
				}
			}
		} else if lMHA.DType == DTypeInt4 && sq != nil && sk != nil && sv != nil {
			if numTokens == 1 && lMHA.DModel <= 2048 {
				if err := ctx.DispatchDecodeQ4QKV(lMHA.DModel, qDim, kvDim, norm1Out, sq, wqBuf, sk, wkBuf, sv, wvBuf, qBuf, kBuf, vBuf); err != nil {
					return nil, err
				}
			} else {
				if err := ctx.DispatchDenseQ4(numTokens, lMHA.DModel, qDim, norm1Out, sq, wqBuf, qBuf, tileSize); err != nil {
					return nil, err
				}
				if err := ctx.DispatchDenseQ4(numTokens, lMHA.DModel, kvDim, norm1Out, sk, wkBuf, kBuf, tileSize); err != nil {
					return nil, err
				}
				if err := ctx.DispatchDenseQ4(numTokens, lMHA.DModel, kvDim, norm1Out, sv, wvBuf, vBuf, tileSize); err != nil {
					return nil, err
				}
			}
		} else if lMHA.DType == DTypeInt8 {
			s := lMHA.WeightStore.Scale
			if err := ctx.DispatchDenseI8(numTokens, lMHA.DModel, qDim, norm1Out, wqBuf, qBuf, s, tileSize); err != nil {
				return nil, err
			}
			if err := ctx.DispatchDenseI8(numTokens, lMHA.DModel, kvDim, norm1Out, wkBuf, kBuf, s, tileSize); err != nil {
				return nil, err
			}
			if err := ctx.DispatchDenseI8(numTokens, lMHA.DModel, kvDim, norm1Out, wvBuf, vBuf, s, tileSize); err != nil {
				return nil, err
			}
		} else {
			if err := ctx.DispatchDense(numTokens, lMHA.DModel, qDim, norm1Out, wqBuf, qBuf, tileSize); err != nil {
				return nil, err
			}
			if err := ctx.DispatchDense(numTokens, lMHA.DModel, kvDim, norm1Out, wkBuf, kBuf, tileSize); err != nil {
				return nil, err
			}
			if err := ctx.DispatchDense(numTokens, lMHA.DModel, kvDim, norm1Out, wvBuf, vBuf, tileSize); err != nil {
				return nil, err
			}
		}

		qkEps := float32(1e-6)
		if lMHA.RMSNormEps > 0 {
			qkEps = float32(lMHA.RMSNormEps)
		}
		if qNormBuf, ok := lMHA.WeightStore.GPUWeights[WeightMHAQNorm].(*wgpu.Buffer); ok && qNormBuf != nil {
			qNormOut := ctx.GetActivationBuffer(fmt.Sprintf("b%d_qn", b), uint64(numTokens*qDim*4), wgpu.BufferUsageStorage)
			if err := ctx.DispatchRMSNorm(numTokens*lMHA.NumHeads, lMHA.HeadDim, qkEps, qBuf, qNormBuf, qNormOut); err != nil {
				return nil, err
			}
			qBuf = qNormOut
		}
		if kNormBuf, ok := lMHA.WeightStore.GPUWeights[WeightMHAKNorm].(*wgpu.Buffer); ok && kNormBuf != nil {
			kNormOut := ctx.GetActivationBuffer(fmt.Sprintf("b%d_kn", b), uint64(numTokens*kvDim*4), wgpu.BufferUsageStorage)
			if err := ctx.DispatchRMSNorm(numTokens*lMHA.NumKVHeads, lMHA.HeadDim, qkEps, kBuf, kNormBuf, kNormOut); err != nil {
				return nil, err
			}
			kBuf = kNormOut
		}

		theta := float32(10000.0)
		if lMHA.RoPEFreqBase > 0 {
			theta = float32(lMHA.RoPEFreqBase)
		}
		kCacheBuf, okK := lMHA.GPUKVCacheK.(*wgpu.Buffer)
		vCacheBuf, okV := lMHA.GPUKVCacheV.(*wgpu.Buffer)
		if !okK || kCacheBuf == nil || !okV || vCacheBuf == nil {
			return nil, fmt.Errorf("layer %d GPU KV Cache not initialized", b)
		}
		if t.gpuChunkRecording {
			stepBuf := ctx.GetActivationBuffer("greedy_step", 64, wgpu.BufferUsageStorage)
			if err := ctx.DispatchRoPEStep(numTokens, lMHA.HeadDim, lMHA.NumHeads, theta, qBuf, stepBuf); err != nil {
				return nil, err
			}
			if err := ctx.DispatchRoPEStep(numTokens, lMHA.HeadDim, lMHA.NumKVHeads, theta, kBuf, stepBuf); err != nil {
				return nil, err
			}
			if err := ctx.DispatchKVUpdateStep(lMHA.HeadDim, lMHA.MaxSeqLen, lMHA.NumKVHeads, numTokens, kCacheBuf, vCacheBuf, kBuf, vBuf, stepBuf); err != nil {
				return nil, err
			}
		} else {
			if err := ctx.DispatchRoPE(numTokens, lMHA.HeadDim, lMHA.NumHeads, lMHA.KVOffset, theta, qBuf); err != nil {
				return nil, err
			}
			if err := ctx.DispatchRoPE(numTokens, lMHA.HeadDim, lMHA.NumKVHeads, lMHA.KVOffset, theta, kBuf); err != nil {
				return nil, err
			}
			if err := ctx.DispatchKVUpdate(lMHA.KVOffset, lMHA.HeadDim, lMHA.MaxSeqLen, lMHA.NumKVHeads, numTokens, kCacheBuf, vCacheBuf, kBuf, vBuf); err != nil {
				return nil, err
			}
		}

		attnOut := ctx.GetActivationBuffer(fmt.Sprintf("b%d_ao", b), uint64(numTokens*qDim*4), wgpu.BufferUsageStorage)
		if t.gpuChunkRecording {
			stepBuf := ctx.GetActivationBuffer("greedy_step", 64, wgpu.BufferUsageStorage)
			if err := ctx.DispatchMHAStep(lMHA.NumHeads, lMHA.NumKVHeads, lMHA.HeadDim, numTokens, lMHA.MaxSeqLen, qBuf, kCacheBuf, vCacheBuf, attnOut, tileSize, stepBuf); err != nil {
				return nil, err
			}
		} else if err := ctx.DispatchMHA(lMHA.NumHeads, lMHA.NumKVHeads, lMHA.HeadDim, numTokens, lMHA.KVOffset, lMHA.MaxSeqLen, qBuf, kCacheBuf, vCacheBuf, attnOut, tileSize); err != nil {
			return nil, err
		}
		if lMHA.DType == DTypeTernary {
			if innerNorm, ok := lMHA.WeightStore.GPUWeights[WeightMHAInnerNorm].(*wgpu.Buffer); ok && innerNorm != nil {
				attnNormOut := ctx.GetActivationBuffer(fmt.Sprintf("b%d_ao_norm", b), uint64(numTokens*qDim*4), wgpu.BufferUsageStorage)
				if err := ctx.DispatchRMSNorm(numTokens, qDim, qkEps, attnOut, innerNorm, attnNormOut); err != nil {
					return nil, err
				}
				attnOut = attnNormOut
			}
		}

		woBuf, errO := getGPUBuffer(lMHA.WeightStore, DType(203))
		if errO != nil {
			return nil, fmt.Errorf("layer %d MHA O-proj sync error: %w", b, errO)
		}
		mhaOut := ctx.GetActivationBuffer(fmt.Sprintf("b%d_hb", b), uint64(numTokens*lMHA.DModel*4), wgpu.BufferUsageStorage)
		so := lMHA.WeightStore.GPUScales[DType(203)]
		if lMHA.DType == DTypeTernary {
			owStart := qDim*lMHA.DModel + kvDim*lMHA.DModel + kvDim*lMHA.DModel
			attnQ, attnScale, err := bitNetQuantizeActivationGPU(ctx, fmt.Sprintf("b%d_attn_bitnet", b), numTokens, qDim, attnOut)
			if err != nil {
				return nil, err
			}
			if err := ctx.DispatchDenseBitNetTernaryQuantized(numTokens, qDim, lMHA.DModel, attnQ, attnScale, woBuf, nil, mhaOut, bitNetGPUScaleValue(lMHA.WeightStore, WeightMHAProjection, owStart), ActivationLinear, tileSize); err != nil {
				return nil, err
			}
		} else if lMHA.DType == DTypeInt4 && so != nil {
			if err := ctx.DispatchDenseQ4(numTokens, qDim, lMHA.DModel, attnOut, so, woBuf, mhaOut, tileSize); err != nil {
				return nil, err
			}
		} else if lMHA.DType == DTypeInt8 {
			if err := ctx.DispatchDenseI8(numTokens, qDim, lMHA.DModel, attnOut, woBuf, mhaOut, lMHA.WeightStore.Scale, tileSize); err != nil {
				return nil, err
			}
		} else {
			if err := ctx.DispatchDense(numTokens, qDim, lMHA.DModel, attnOut, woBuf, mhaOut, tileSize); err != nil {
				return nil, err
			}
		}

		if !t.gpuChunkRecording {
			lMHA.KVOffset += numTokens
		}
		if err := ctx.DispatchResidual(numTokens*lMHA.DModel, mhaOut, residual); err != nil {
			return nil, err
		}
		currentBuf = mhaOut
		if t.Network.Tanhi != nil && t.Network.Tanhi.Enabled {
			var sh []int
			if t.Network.Tanhi.SendShape {
				sh = TanhiGPULayerShapeHint(lMHA, numTokens)
			}
			tanhiEmit(t.Network, "fwd", base+1, lMHA, tMHA0, time.Now(), sh)
		}

		// --- Norm 2 ---
		lNorm2 := &t.Network.Layers[base+2]
		norm2Out := ctx.GetActivationBuffer(fmt.Sprintf("b%d_n2", b), uint64(numTokens*lNorm2.InputHeight*4), wgpu.BufferUsageStorage)
		weightBuf2, errN2 := getGPUBuffer(lNorm2.WeightStore, DTypeFloat32)
		if errN2 != nil {
			return nil, fmt.Errorf("layer %d Norm2 sync error: %w", b, errN2)
		}
		eps2 := float32(1e-6)
		if lNorm2.RMSNormEps > 0 {
			eps2 = float32(lNorm2.RMSNormEps)
		}
		tN2 := time.Now()
		if err := ctx.DispatchRMSNorm(numTokens, lNorm2.InputHeight, eps2, currentBuf, weightBuf2, norm2Out); err != nil {
			return nil, err
		}
		if t.Network.Tanhi != nil && t.Network.Tanhi.Enabled {
			var sh []int
			if t.Network.Tanhi.SendShape {
				sh = TanhiGPULayerShapeHint(lNorm2, numTokens)
			}
			tanhiEmit(t.Network, "fwd", base+2, lNorm2, tN2, time.Now(), sh)
		}

		residual = currentBuf

		// --- SwiGLU ---
		tMLP0 := time.Now()
		lMLP := &t.Network.Layers[base+3]
		interOut := ctx.GetActivationBuffer(fmt.Sprintf("b%d_mi", b), uint64(numTokens*lMLP.OutputHeight*4), wgpu.BufferUsageStorage)
		gW, errG := getGPUBuffer(lMLP.WeightStore, DType(100))
		uW, errU := getGPUBuffer(lMLP.WeightStore, DType(101))
		dW, errD := getGPUBuffer(lMLP.WeightStore, DType(102))
		if errG != nil || errU != nil || errD != nil {
			return nil, fmt.Errorf("layer %d MLP sync error: %v, %v, %v", b, errG, errU, errD)
		}

		tileSize = lMLP.TileSize
		if tileSize <= 0 {
			tileSize = ctx.GPUTileSize
			if t.Network.EnableMultiCoreTiling {
				tileSize = lMLP.GetGPUMCTileSize(lMLP.DType)
			} else if lMLP.UseTiling {
				tileSize = lMLP.GetGPUSCTileSize(lMLP.DType)
			}
		}
		if tileSize <= 0 {
			tileSize = 32
		}

		gB, _ := lMLP.WeightStore.GPUWeights[DType(110)].(*wgpu.Buffer)
		uB, _ := lMLP.WeightStore.GPUWeights[DType(111)].(*wgpu.Buffer)
		if gB == nil {
			gB = ctx.BlankBuffer
		}
		if uB == nil {
			uB = ctx.BlankBuffer
		}

		if lMLP.DType == DTypeTernary {
			gatePre := ctx.GetActivationBuffer(fmt.Sprintf("b%d_gate_bitnet", b), uint64(numTokens*lMLP.OutputHeight*4), wgpu.BufferUsageStorage)
			upPre := ctx.GetActivationBuffer(fmt.Sprintf("b%d_up_bitnet", b), uint64(numTokens*lMLP.OutputHeight*4), wgpu.BufferUsageStorage)
			norm2Q, norm2Scale, err := bitNetQuantizeActivationGPU(ctx, fmt.Sprintf("b%d_n2_bitnet", b), numTokens, lMLP.InputHeight, norm2Out)
			if err != nil {
				return nil, err
			}
			if err := ctx.DispatchDenseBitNetTernaryQuantized(numTokens, lMLP.InputHeight, lMLP.OutputHeight, norm2Q, norm2Scale, gW, nil, gatePre, bitNetGPUScaleValue(lMLP.WeightStore, DType(100), 0), ActivationLinear, tileSize); err != nil {
				return nil, err
			}
			if err := ctx.DispatchDenseBitNetTernaryQuantized(numTokens, lMLP.InputHeight, lMLP.OutputHeight, norm2Q, norm2Scale, uW, nil, upPre, bitNetGPUScaleValue(lMLP.WeightStore, DType(101), lMLP.InputHeight*lMLP.OutputHeight), ActivationLinear, tileSize); err != nil {
				return nil, err
			}
			if err := ctx.DispatchBitNetGateProduct(numTokens, lMLP.OutputHeight, lMLP.Activation, gatePre, upPre, interOut); err != nil {
				return nil, err
			}
			if innerNorm, ok := lMLP.WeightStore.GPUWeights[WeightSwiGLUInnerNorm].(*wgpu.Buffer); ok && innerNorm != nil {
				innerOut := ctx.GetActivationBuffer(fmt.Sprintf("b%d_mi_norm", b), uint64(numTokens*lMLP.OutputHeight*4), wgpu.BufferUsageStorage)
				if err := ctx.DispatchRMSNorm(numTokens, lMLP.OutputHeight, eps2, interOut, innerNorm, innerOut); err != nil {
					return nil, err
				}
				interOut = innerOut
			}
		} else if lMLP.DType == DTypeInt4 {
			sg := lMLP.WeightStore.GPUScales[DType(100)]
			su := lMLP.WeightStore.GPUScales[DType(101)]
			if sg != nil && su != nil {
				if err := ctx.DispatchSwiGLUQ4(numTokens, lMLP.InputHeight, lMLP.OutputHeight, norm2Out, sg, gW, su, uW, gB, uB, interOut, tileSize); err != nil {
					return nil, err
				}
			} else {
				if err := ctx.DispatchSwiGLU(numTokens, lMLP.InputHeight, lMLP.OutputHeight, norm2Out, gW, uW, gB, uB, interOut, tileSize); err != nil {
					return nil, err
				}
			}
		} else if lMLP.DType == DTypeInt8 {
			s := lMLP.WeightStore.Scale
			if err := ctx.DispatchSwiGLUI8(numTokens, lMLP.InputHeight, lMLP.OutputHeight, norm2Out, gW, uW, gB, uB, interOut, s, s, tileSize); err != nil {
				return nil, err
			}
		} else {
			if err := ctx.DispatchSwiGLU(numTokens, lMLP.InputHeight, lMLP.OutputHeight, norm2Out, gW, uW, gB, uB, interOut, tileSize); err != nil {
				return nil, err
			}
		}

		mlpOut := ctx.GetActivationBuffer(fmt.Sprintf("b%d_ha", b), uint64(numTokens*lMLP.InputHeight*4), wgpu.BufferUsageStorage)
		sd := lMLP.WeightStore.GPUScales[DType(102)]
		dB, _ := lMLP.WeightStore.GPUWeights[DType(112)].(*wgpu.Buffer)
		if dB == nil {
			dB = ctx.BlankBuffer
		}

		if lMLP.DType == DTypeTernary {
			downStart := 2 * lMLP.InputHeight * lMLP.OutputHeight
			interQ, interScale, err := bitNetQuantizeActivationGPU(ctx, fmt.Sprintf("b%d_mi_bitnet", b), numTokens, lMLP.OutputHeight, interOut)
			if err != nil {
				return nil, err
			}
			if err := ctx.DispatchDenseBitNetTernaryQuantized(numTokens, lMLP.OutputHeight, lMLP.InputHeight, interQ, interScale, dW, nil, mlpOut, bitNetGPUScaleValue(lMLP.WeightStore, DType(102), downStart), ActivationLinear, tileSize); err != nil {
				return nil, err
			}
		} else if lMLP.DType == DTypeInt4 && sd != nil {
			if err := ctx.DispatchDenseQ4(numTokens, lMLP.OutputHeight, lMLP.InputHeight, interOut, sd, dW, mlpOut, tileSize); err != nil {
				return nil, err
			}
		} else if lMLP.DType == DTypeInt8 {
			if err := ctx.DispatchDenseI8(numTokens, lMLP.OutputHeight, lMLP.InputHeight, interOut, dW, mlpOut, lMLP.WeightStore.Scale, tileSize); err != nil {
				return nil, err
			}
		} else {
			// Get actual activation or use 99 for Linear
			act := uint32(99) // default linear
			if err := ctx.DispatchDenseTiled(tileSize, numTokens, lMLP.OutputHeight, lMLP.InputHeight, act, 1.0, interOut, dW, dB, mlpOut); err != nil {
				return nil, err
			}
		}

		if err := ctx.DispatchResidual(numTokens*lMLP.InputHeight, mlpOut, residual); err != nil {
			return nil, err
		}
		currentBuf = mlpOut
		if t.Network.Tanhi != nil && t.Network.Tanhi.Enabled {
			var sh []int
			if t.Network.Tanhi.SendShape {
				sh = TanhiGPULayerShapeHint(lMLP, numTokens)
			}
			tanhiEmit(t.Network, "fwd", base+3, lMLP, tMLP0, time.Now(), sh)
		}
	}

	// 3. Final Norm on GPU
	fnIdx := len(t.Network.Layers)
	if t.finalNormLayer != nil && t.finalNormLayer.IsGPUResident {
		t.finalNormLayer.Z = anchorZ
		t.finalNormLayer.Y = anchorY
		t.finalNormLayer.X = anchorX
		t.finalNormLayer.L = -3
		fNormOut := ctx.GetActivationBuffer("final_norm", uint64(numTokens*t.HiddenSize*4), wgpu.BufferUsageStorage)
		fW := t.finalNormLayer.WeightStore.GPUWeights[DTypeFloat32].(*wgpu.Buffer)
		finalEps := float32(1e-6)
		if t.finalNormLayer != nil && t.finalNormLayer.RMSNormEps > 0 {
			finalEps = float32(t.finalNormLayer.RMSNormEps)
		}
		tFN := time.Now()
		if err := ctx.DispatchRMSNorm(numTokens, t.HiddenSize, finalEps, currentBuf, fW, fNormOut); err != nil {
			return nil, err
		}
		currentBuf = fNormOut
		if t.Network.Tanhi != nil && t.Network.Tanhi.Enabled {
			var sh []int
			if t.Network.Tanhi.SendShape {
				sh = TanhiGPULayerShapeHint(t.finalNormLayer, numTokens)
			}
			tanhiEmit(t.Network, "fwd", fnIdx, t.finalNormLayer, tFN, time.Now(), sh)
		}
	}

	// 4. LM Head on GPU if requested and available
	isReturningLogits := false
	downloadTokens := numTokens
	readOffset := uint64(0)

	hasQ4LM := t.Network.GPULMHeadQ4Packed != nil && t.Network.GPULMHeadQ4Scales != nil
	ternW, hasTernaryLM := t.Network.GPULMHeadTernaryPacked.(*wgpu.Buffer)
	hasTernaryLM = hasTernaryLM && ternW != nil
	if computeLogits && (t.Network.GPULMHead != nil || hasQ4LM || hasTernaryLM) {
		tLM := time.Now()
		effectiveInput := currentBuf
		dispatchTokens := numTokens

		if onlyLast && numTokens > 1 {
			// Optimization: Copy ONLY the last token's hidden state to a scratch buffer
			// This avoids running the heavy LM Head matrix mult on the entire prompt.
			scratchSize := uint64(t.HiddenSize * 4)
			scratchBuf := ctx.GetActivationBuffer("hidden_scratch", scratchSize, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst|wgpu.BufferUsageCopySrc)
			offset := uint64((numTokens - 1) * t.HiddenSize * 4)

			// Use the existing command encoder to perform a sync-less copy
			ctx.EndComputePass()
			ctx.ActiveEncoder.CopyBufferToBuffer(currentBuf, offset, scratchBuf, 0, scratchSize)

			effectiveInput = scratchBuf
			dispatchTokens = 1
		}

		lmOut := ctx.GetActivationBuffer("logits", uint64(dispatchTokens*t.VocabSize*4), wgpu.BufferUsageStorage)
		q4s, _ := t.Network.GPULMHeadQ4Scales.(*wgpu.Buffer)
		q4w, _ := t.Network.GPULMHeadQ4Packed.(*wgpu.Buffer)
		if hasTernaryLM {
			// BitNet: int8 act × packed ternary LM (same cost class as decoder Dense).
			lmQ, lmScale, err := bitNetQuantizeActivationGPU(ctx, "lm_bitnet", dispatchTokens, t.HiddenSize, effectiveInput)
			if err != nil {
				return nil, err
			}
			wScale := t.Network.GPULMHeadTernaryScale
			if wScale == 0 {
				wScale = 1
			}
			tile := ctx.GPUTileSize
			if tile <= 0 {
				tile = 32
			}
			if err := ctx.DispatchDenseBitNetTernaryQuantized(
				dispatchTokens, t.HiddenSize, t.VocabSize,
				lmQ, lmScale, ternW, nil, lmOut, wScale, ActivationLinear, tile,
			); err != nil {
				return nil, err
			}
		} else if q4s != nil && q4w != nil {
			if err := ctx.DispatchDenseQ4(dispatchTokens, t.HiddenSize, t.VocabSize, effectiveInput, q4s, q4w, lmOut, ctx.GPUTileSize); err != nil {
				return nil, err
			}
		} else {
			lmW, ok := t.Network.GPULMHead.(*wgpu.Buffer)
			if !ok || lmW == nil {
				return nil, fmt.Errorf("GPU LM Head not loaded")
			}
			if err := ctx.DispatchDense(dispatchTokens, t.HiddenSize, t.VocabSize, effectiveInput, lmW, lmOut, ctx.GPUTileSize); err != nil {
				return nil, err
			}
		}

		currentBuf = lmOut
		if t.Network.Tanhi != nil && t.Network.Tanhi.Enabled && len(t.Network.Layers) > 0 {
			lmConn := len(t.LMHead)
			if lmConn == 0 {
				lmConn = t.VocabSize * t.HiddenSize
			}
			lmLayer := VolumetricLayer{
				Network:      t.Network,
				Type:         LayerDense,
				DType:        t.Network.Layers[0].DType,
				InputHeight:  t.HiddenSize,
				OutputHeight: t.VocabSize,
				Z:            anchorZ, Y: anchorY, X: anchorX, L: -2,
			}
			var sh []int
			if t.Network.Tanhi.SendShape {
				sh = TanhiGPULayerShapeHint(&lmLayer, dispatchTokens)
			}
			tanhiEmitWithConn(t.Network, "fwd", fnIdx+1, &lmLayer, tLM, time.Now(), sh, lmConn)
		}
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

	// On-device ArgMax. Chunk mode keeps the token on GPU (advance + hist) with no MapAsync.
	if t.gpuReturnGreedyToken && isReturningLogits {
		tokenOut := ctx.GetActivationBuffer("greedy_token", 16, wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
		if err := ctx.DispatchArgMax(dim, currentBuf, tokenOut); err != nil {
			return nil, err
		}
		if t.gpuChunkRecording {
			stepBuf := ctx.GetActivationBuffer("greedy_step", 64, wgpu.BufferUsageStorage)
			// Size must already be allocated by ForwardSampleGreedyChunkWGPU.
			histBuf := ctx.GetActivationBuffer("greedy_hist", 64, wgpu.BufferUsageStorage)
			liveTok := ctx.GetActivationBuffer("decode_token", 64, wgpu.BufferUsageStorage)
			if err := ctx.DispatchAdvanceGreedy(stepBuf, tokenOut, histBuf, liveTok); err != nil {
				return nil, err
			}
			t.gpuChunkHistCount++
			return NewTensorFromSlice[T]([]T{0}, 1, 1), nil
		}
		stagingBuf := ctx.GetActivationBuffer("staging_u32", 64, wgpu.BufferUsageMapRead)
		ctx.EndComputePass()
		ctx.ActiveEncoder.CopyBufferToBuffer(tokenOut, 0, stagingBuf, 0, 4)
		ctx.FlushFrame()
		raw, err := ctx.pollMapRead(stagingBuf, 4)
		if err != nil {
			return nil, err
		}
		t.lastGPUSampledToken = uint32(raw[0]) | uint32(raw[1])<<8 | uint32(raw[2])<<16 | uint32(raw[3])<<24
		return NewTensorFromSlice[T]([]T{T(t.lastGPUSampledToken)}, 1, 1), nil
	}

	resSize := uint64(downloadTokens * dim * 4)
	stagingBuf := ctx.GetActivationBuffer("staging", resSize, wgpu.BufferUsageMapRead)

	// Perform the copy within the ACTIVE encoder before flushing
	ctx.EndComputePass()
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
			// Mobile Optimization: Yield more effectively to let driver threads work
			runtime.Gosched()
			if runtime.GOOS == "android" {
				// Don't starve the system; a 50us breath helps on Adreno
				runtime.KeepAlive(done)
			}
		}
	}

Finished:
	outBytes := stagingBuf.GetMappedRange(0, uint(resSize))
	outData := wgpu.FromBytes[float32](outBytes)
	stagingBuf.Unmap()

	shape := []int{downloadTokens, dim}
	return NewTensorFromSlice[T](CastWeights[T](outData), shape...), nil
}

// ForwardSampleGreedyTokenWGPU runs a GPU forward and returns the ArgMax token id.
// Only a 4-byte readback syncs to the host (not the full logits vector).
func (t *Transformer[T]) ForwardSampleGreedyTokenWGPU(tokens []uint32, onlyLast bool) (uint32, error) {
	t.gpuReturnGreedyToken = true
	defer func() { t.gpuReturnGreedyToken = false }()
	_, err := t.ForwardTokenIDsWGPU(tokens, nil, true, onlyLast)
	if err != nil {
		return 0, err
	}
	return t.lastGPUSampledToken, nil
}

func (t *Transformer[T]) ForwardWGPU(input *Tensor[T]) (*Tensor[T], error) {
	return t.ForwardTokenIDsWGPU(nil, input, false, false)
}
