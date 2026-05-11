package poly

import (
	"fmt"
	"strings"

	"github.com/openfluke/webgpu/wgpu"
)

// gpuParallelScratch records per-branch forward buffers for one GPU batch (concat parallel).
type gpuParallelScratch struct {
	Branches []gpuParallelBranchScratch
}

type gpuParallelBranchScratch struct {
	HistIn  []*wgpu.Buffer
	HistPre []*wgpu.Buffer
}

func gpuCopyBuffer(c *WGPUContext, src, dst *wgpu.Buffer, srcOff, dstOff, sizeBytes uint64) error {
	enc, owned, err := ctxEncoder(c)
	if err != nil {
		return err
	}
	enc.CopyBufferToBuffer(src, srcOff, dst, dstOff, sizeBytes)
	ctxSubmit(c, enc, owned)
	return nil
}

func gpuTrainLayerOutputSize(l *VolumetricLayer) int {
	outSize := l.OutputHeight
	switch l.Type {
	case LayerCNN2, LayerCNN3:
		d := l.OutputDepth
		if d == 0 {
			d = 1
		}
		h := l.OutputHeight
		if h == 0 {
			h = 1
		}
		w := l.OutputWidth
		if w == 0 {
			w = 1
		}
		outSize = d * h * w * l.Filters
	case LayerCNN1:
		outSize = l.OutputHeight * l.Filters
	case LayerMultiHeadAttention:
		sl := l.SeqLength
		if sl <= 0 {
			sl = 1
		}
		outSize = sl * l.DModel
	}
	if outSize == 0 {
		outSize = l.InputHeight
	}
	return outSize
}

func gpuTrainLayerInputSize(l *VolumetricLayer) int {
	inSize := l.InputHeight
	switch l.Type {
	case LayerCNN2, LayerCNN3:
		d := l.InputDepth
		if d == 0 {
			d = 1
		}
		h := l.InputHeight
		if h == 0 {
			h = 1
		}
		w := l.InputWidth
		if w == 0 {
			w = 1
		}
		inSize = d * h * w * l.InputChannels
	case LayerCNN1:
		inSize = l.InputHeight * l.InputChannels
	case LayerMultiHeadAttention:
		sl := l.SeqLength
		if sl <= 0 {
			sl = 1
		}
		inSize = sl * l.DModel
	}
	if inSize == 0 {
		inSize = l.OutputHeight
	}
	return inSize
}

func parallelConcatCombine(mode string) bool {
	m := strings.ToLower(strings.TrimSpace(mode))
	return m == "concat" || m == "" || m == "grid_scatter"
}

// gpuParallelConcatForward runs branch forwards and concatenates into concatOut (preBuf for this layer).
func gpuParallelConcatForward(ctx *WGPUContext, mode TrainingMode, batchSize, layerIdx int, par *VolumetricLayer, inBuf, concatOut *wgpu.Buffer) error {
	if par == nil {
		return fmt.Errorf("parallel: nil layer")
	}
	if !parallelConcatCombine(par.CombineMode) {
		return fmt.Errorf("parallel GPU training: combine_mode %q not supported (only concat)", par.CombineMode)
	}
	if len(par.ParallelBranches) == 0 {
		return fmt.Errorf("parallel: no branches")
	}

	byteOff := uint64(0)
	scratch := &gpuParallelScratch{Branches: make([]gpuParallelBranchScratch, len(par.ParallelBranches))}

	for bi := range par.ParallelBranches {
		br := &par.ParallelBranches[bi]
		if br.IsRemoteLink {
			return fmt.Errorf("parallel GPU: remote-link branch not supported")
		}
		if br.Type != LayerSequential || len(br.SequentialLayers) == 0 {
			return fmt.Errorf("parallel GPU: branch %d must be LayerSequential with ≥1 sublayer", bi)
		}

		st := &scratch.Branches[bi]
		subs := br.SequentialLayers
		st.HistIn = make([]*wgpu.Buffer, len(subs))
		st.HistPre = make([]*wgpu.Buffer, len(subs))

		curBuf := inBuf
		for si := range subs {
			sl := &br.SequentialLayers[si]
			if sl.IsDisabled {
				return fmt.Errorf("parallel GPU: disabled sublayer branch %d step %d not supported", bi, si)
			}
			outSize := gpuTrainLayerOutputSize(sl)
			preBuf := ctx.GetActivationBuffer(fmt.Sprintf("par_l%d_b%d_s%d_pre", layerIdx, bi, si), uint64(outSize*batchSize*4), wgpu.BufferUsageStorage)
			if preBuf == nil {
				return fmt.Errorf("parallel: failed preBuf b%d s%d", bi, si)
			}

			st.HistIn[si] = curBuf

			layerTileSize := 0
			if mode == TrainingModeGPUSC {
				layerTileSize = sl.GetGPUSCTileSize(sl.DType)
			} else if mode == TrainingModeGPUMC {
				layerTileSize = sl.GetGPUMCTileSize(sl.DType)
			}

			var fwdErr error
			if layerTileSize > 0 && sl.Type == LayerCNN1 {
				kernelVol := sl.InputChannels * sl.KernelSize
				wBuf := GetGPUWeightBuffer(sl)
				scale := float32(1.0)
				if sl.WeightStore != nil && sl.WeightStore.Scale != 0 {
					scale = sl.WeightStore.Scale
				}
				if isCNN1NativeGPUQuantDType(sl.DType) {
					scale = cnn1PackedGPUScale(sl)
					fwdErr = ctx.DispatchCNN1PackedTiled(sl.DType, layerTileSize, kernelVol, batchSize,
						sl.InputChannels, sl.InputHeight, sl.Filters, sl.OutputHeight,
						sl.KernelSize, sl.Stride, sl.Padding,
						scale, curBuf, wBuf, preBuf)
				} else {
					fwdErr = ctx.DispatchCNN1Tiled(layerTileSize, kernelVol, batchSize,
						sl.InputChannels, sl.InputHeight, sl.Filters, sl.OutputHeight,
						sl.KernelSize, sl.Stride, sl.Padding,
						scale, curBuf, wBuf, preBuf)
				}
			} else if layerTileSize > 0 && sl.Type == LayerCNN3 {
				kernelVol := sl.InputChannels * sl.KernelSize * sl.KernelSize * sl.KernelSize
				wBuf := GetGPUWeightBuffer(sl)
				scale := float32(1.0)
				if sl.WeightStore != nil && sl.WeightStore.Scale != 0 {
					scale = sl.WeightStore.Scale
				}
				fwdErr = ctx.DispatchCNN3Tiled(layerTileSize, kernelVol, batchSize,
					sl.InputChannels, sl.InputDepth, sl.InputHeight, sl.InputWidth,
					sl.Filters, sl.OutputDepth, sl.OutputHeight, sl.OutputWidth,
					sl.KernelSize, sl.KernelSize, sl.KernelSize,
					sl.Stride, sl.Stride, sl.Stride,
					sl.Padding, sl.Padding, sl.Padding,
					scale, curBuf, wBuf, preBuf)
			} else {
				fwdErr = ctx.DispatchForwardLayer(sl, batchSize, curBuf, preBuf)
			}
			if fwdErr != nil {
				return fmt.Errorf("parallel branch %d sub %d forward: %w", bi, si, fwdErr)
			}

			var nextBuf *wgpu.Buffer
			if sl.Activation != ActivationLinear {
				postBuf := ctx.GetActivationBuffer(fmt.Sprintf("par_l%d_b%d_s%d_post", layerIdx, bi, si), uint64(outSize*batchSize*4), wgpu.BufferUsageStorage)
				if postBuf == nil {
					return fmt.Errorf("parallel: failed postBuf b%d s%d", bi, si)
				}
				if err := ctx.DispatchActivation(outSize*batchSize, sl.Activation, preBuf, postBuf); err != nil {
					return err
				}
				nextBuf = postBuf
			} else {
				nextBuf = preBuf
			}
			st.HistPre[si] = preBuf
			curBuf = nextBuf
		}

		last := &br.SequentialLayers[len(subs)-1]
		branchElems := gpuTrainLayerOutputSize(last) * batchSize
		brBytes := uint64(branchElems * 4)
		if err := gpuCopyBuffer(ctx, curBuf, concatOut, 0, byteOff, brBytes); err != nil {
			return err
		}
		byteOff += brBytes
	}

	want := uint64(0)
	if par.OutputHeight > 0 {
		want = uint64(par.OutputHeight * batchSize * 4)
	}
	if want > 0 && byteOff != want {
		return fmt.Errorf("parallel forward: concat wrote %d bytes, expected %d (check branch heads vs OutputHeight)", byteOff, want)
	}

	par.gpuParScratch = scratch
	return nil
}

// gpuParallelConcatBackward splits concat gradients, runs branch backwards, sums input grads into dxOut.
func gpuParallelConcatBackward(ctx *WGPUContext, mode TrainingMode, batchSize, layerIdx int, par *VolumetricLayer, gradOut, dxOut *wgpu.Buffer, cfg *TrainingConfig) error {
	sc := par.gpuParScratch
	if sc == nil || len(sc.Branches) != len(par.ParallelBranches) {
		return fmt.Errorf("parallel backward: missing forward scratch (layer %d)", layerIdx)
	}

	inElems := gpuTrainLayerInputSize(par) * batchSize

	byteOff := uint64(0)
	for bi := range par.ParallelBranches {
		br := &par.ParallelBranches[bi]
		subs := br.SequentialLayers
		if len(subs) == 0 {
			continue
		}
		last := &subs[len(subs)-1]
		branchOut := gpuTrainLayerOutputSize(last) * batchSize
		brBytes := uint64(branchOut * 4)

		gradBranch := ctx.GetActivationBuffer(fmt.Sprintf("par_l%d_b%d_grad", layerIdx, bi), brBytes, wgpu.BufferUsageStorage)
		if gradBranch == nil {
			return fmt.Errorf("parallel: grad branch buf b%d", bi)
		}
		if err := gpuCopyBuffer(ctx, gradOut, gradBranch, byteOff, 0, brBytes); err != nil {
			return err
		}
		byteOff += brBytes

		st := &sc.Branches[bi]
		curGrad := gradBranch

		for si := len(subs) - 1; si >= 0; si-- {
			sl := &subs[si]
			if sl.IsDisabled {
				continue
			}

			inSize := gpuTrainLayerInputSize(sl)
			outSize := gpuTrainLayerOutputSize(sl)

			dxBuf := ctx.GetActivationBuffer(fmt.Sprintf("par_l%d_b%d_s%d_dx", layerIdx, bi, si), uint64(inSize*batchSize*4), wgpu.BufferUsageStorage)
			if dxBuf == nil {
				return fmt.Errorf("parallel: dx buf b%d s%d", bi, si)
			}

			wSize := 1
			if sl.WeightStore != nil {
				wSize = len(sl.WeightStore.Master)
				if wSize <= 0 {
					wSize = 1
				}
			}
			dwBuf := ctx.GetActivationBuffer(fmt.Sprintf("par_l%d_b%d_s%d_dw", layerIdx, bi, si), uint64(wSize*4), wgpu.BufferUsageStorage)
			if dwBuf == nil {
				return fmt.Errorf("parallel: dw buf b%d s%d", bi, si)
			}
			if err := ctx.DispatchFillZero(wSize, dwBuf); err != nil {
				return err
			}

			var gradPreBuf *wgpu.Buffer
			if sl.Activation != ActivationLinear {
				gradPreBuf = ctx.GetActivationBuffer(fmt.Sprintf("par_l%d_b%d_s%d_gpre", layerIdx, bi, si), uint64(outSize*batchSize*4), wgpu.BufferUsageStorage)
				if gradPreBuf == nil {
					return fmt.Errorf("parallel: gradPre b%d s%d", bi, si)
				}
				if err := ctx.DispatchActivationBackward(outSize*batchSize, sl.Activation, curGrad, st.HistPre[si], gradPreBuf); err != nil {
					return err
				}
			} else {
				gradPreBuf = curGrad
			}

			bwdTileSize := 0
			if mode == TrainingModeGPUSC {
				bwdTileSize = sl.GetGPUSCTileSize(sl.DType)
			} else if mode == TrainingModeGPUMC {
				bwdTileSize = sl.GetGPUMCTileSize(sl.DType)
			}

			var bwdWBuf *wgpu.Buffer
			if sl.WeightStore != nil {
				if sl.Type == LayerCNN1 && isCNN1NativeGPUQuantDType(sl.DType) {
					bwdWBuf = GetGPUWeightBuffer(sl)
				} else if buf, ok := sl.WeightStore.GPUWeights[DTypeFloat32].(*wgpu.Buffer); ok && buf != nil {
					bwdWBuf = buf
				} else {
					bwdWBuf = GetGPUWeightBuffer(sl)
				}
			}

			var bwdErr error
			if bwdTileSize > 0 && sl.Type == LayerCNN1 {
				kernelVol := sl.InputChannels * sl.KernelSize
				if isCNN1NativeGPUQuantDType(sl.DType) {
					if err := ctx.DispatchCNN1PackedBackwardDXTiled(sl.DType, bwdTileSize, kernelVol, batchSize,
						sl.InputChannels, sl.InputHeight, sl.Filters, sl.OutputHeight,
						sl.KernelSize, sl.Stride, sl.Padding,
						sl.Activation, cnn1PackedGPUScale(sl), gradPreBuf, bwdWBuf, st.HistPre[si], dxBuf); err != nil {
						return err
					}
				} else {
					if err := ctx.DispatchCNN1TiledBackwardDX(bwdTileSize, kernelVol, batchSize,
						sl.InputChannels, sl.InputHeight, sl.Filters, sl.OutputHeight,
						sl.KernelSize, sl.Stride, sl.Padding,
						sl.Activation, gradPreBuf, bwdWBuf, st.HistPre[si], dxBuf); err != nil {
						return err
					}
				}
				bwdErr = ctx.DispatchCNN1TiledBackwardDW(bwdTileSize, batchSize,
					sl.InputChannels, sl.InputHeight, sl.Filters, sl.OutputHeight,
					sl.KernelSize, sl.Stride, sl.Padding,
					sl.Activation, gradPreBuf, st.HistIn[si], st.HistPre[si], dwBuf)
			} else if bwdTileSize > 0 && sl.Type == LayerCNN3 {
				kernelVol := sl.InputChannels * sl.KernelSize * sl.KernelSize * sl.KernelSize
				if err := ctx.DispatchCNN3TiledBackwardDX(bwdTileSize, kernelVol, batchSize,
					sl.InputChannels, sl.InputDepth, sl.InputHeight, sl.InputWidth,
					sl.Filters, sl.OutputDepth, sl.OutputHeight, sl.OutputWidth,
					sl.KernelSize, sl.KernelSize, sl.KernelSize,
					sl.Stride, sl.Stride, sl.Stride,
					sl.Padding, sl.Padding, sl.Padding,
					sl.Activation, gradPreBuf, bwdWBuf, st.HistPre[si], dxBuf); err != nil {
					return err
				}
				bwdErr = ctx.DispatchCNN3TiledBackwardDW(bwdTileSize, batchSize,
					sl.InputChannels, sl.InputDepth, sl.InputHeight, sl.InputWidth,
					sl.Filters, sl.OutputDepth, sl.OutputHeight, sl.OutputWidth,
					sl.KernelSize, sl.KernelSize, sl.KernelSize,
					sl.Stride, sl.Stride, sl.Stride,
					sl.Padding, sl.Padding, sl.Padding,
					sl.Activation, gradPreBuf, st.HistIn[si], st.HistPre[si], dwBuf)
			} else {
				bwdErr = ctx.DispatchBackwardLayer(sl, batchSize, gradPreBuf, st.HistIn[si], st.HistPre[si], dxBuf, dwBuf)
			}
			if bwdErr != nil {
				return fmt.Errorf("parallel bwd branch %d sub %d: %w", bi, si, bwdErr)
			}

			if sl.WeightStore != nil {
				if sl.Type == LayerCNN1 && isCNN1NativeGPUQuantDType(sl.DType) {
					packedBuf, _ := sl.WeightStore.GPUWeights[sl.DType].(*wgpu.Buffer)
					masterBuf, _ := sl.WeightStore.GPUWeights[DTypeFloat32].(*wgpu.Buffer)
					if packedBuf != nil && masterBuf != nil {
						if err := ctx.DispatchCNN1PackedApplyGradients(
							sl.DType,
							wSize,
							cfg.LearningRate,
							cfg.GradientClip,
							cnn1PackedGPUScale(sl),
							packedBuf,
							dwBuf,
							masterBuf,
						); err != nil {
							return err
						}
					} else {
						return fmt.Errorf("parallel GPU: CNN1 quantized branch %d sub %d needs packed+FP32 master buffers", bi, si)
					}
				} else if wBuf, ok := sl.WeightStore.GPUWeights[DTypeFloat32].(*wgpu.Buffer); ok && wBuf != nil {
					if err := ctx.DispatchApplyGradients(wSize, cfg.LearningRate, cfg.GradientClip, wBuf, dwBuf); err != nil {
						return err
					}
					if ctx.ActiveEncoder != nil {
						ctx.propagateSplitWeights(sl, wBuf)
					}
					if sl.DType == DTypeInt8 {
						if native, ok := sl.WeightStore.GPUWeights[DTypeInt8].(*wgpu.Buffer); ok && native != nil {
							ctx.DispatchQuantizeI8(wSize, sl.WeightStore.Scale, wBuf, native)
						}
					} else if sl.DType == DTypeInt4 {
						if native, ok := sl.WeightStore.GPUWeights[DTypeInt4].(*wgpu.Buffer); ok && native != nil {
							ctx.DispatchQuantizeI4(wSize, sl.WeightStore.Scale, wBuf, native)
						}
					} else if sl.DType == DTypeFP4 {
						if native, ok := sl.WeightStore.GPUWeights[DTypeFP4].(*wgpu.Buffer); ok && native != nil {
							ctx.DispatchQuantizeFP4(wSize, sl.WeightStore.Scale, wBuf, native)
						}
					} else if sl.DType == DTypeTernary {
						if native, ok := sl.WeightStore.GPUWeights[DTypeTernary].(*wgpu.Buffer); ok && native != nil {
							ctx.DispatchQuantizeTernary(wSize, sl.WeightStore.Scale, wBuf, native)
						}
					} else if sl.DType == DTypeBinary {
						if native, ok := sl.WeightStore.GPUWeights[DTypeBinary].(*wgpu.Buffer); ok && native != nil {
							ctx.DispatchQuantizeBinary(wSize, sl.WeightStore.Scale, wBuf, native)
						}
					}
				}
			}

			curGrad = dxBuf
		}

		// Sum branch input gradients (shared packed input).
		inBytes := uint64(inElems * 4)
		if bi == 0 {
			if err := gpuCopyBuffer(ctx, curGrad, dxOut, 0, 0, inBytes); err != nil {
				return err
			}
		} else {
			if err := ctx.DispatchResidual(inElems, dxOut, curGrad); err != nil {
				return err
			}
		}
	}

	par.gpuParScratch = nil
	return nil
}
