package nn

// forward_fp4_gpu.go — WeightsFP4ToGPU
//
// Mounts the network onto the GPU using FP4-compressed weight buffers for
// Dense and SwiGLU layers.  All other layer types fall through to the normal
// WeightsToGPU path (MHA, RMSNorm, Embedding, etc.).
//
// Weight VRAM usage for quantised layers is roughly:
//   packed nibbles  ≈ WeightElements / 2  bytes
//   scales          ≈ (WeightElements / 16) * 4  bytes
//   total           ≈ WeightElements * 0.75  bytes   (vs. 4× for float32)
//
// Usage:
//   fp4 := net.BuildFP4Weights()
//   err  := net.WeightsFP4ToGPU(fp4)   // replaces WeightsToGPU
//   // then use net.Forward() as usual

import (
	"fmt"
	"math"
	"time"

	"github.com/openfluke/loom/gpu"
	"github.com/openfluke/webgpu/wgpu"
)

// WeightsFP4ToGPU is identical in contract to WeightsToGPU but replaces every
// Dense and SwiGLU layer that has a corresponding FP4LayerWeights entry with
// a gpu.FP4DenseLayer — storing nibble-packed weights in VRAM.
func (n *Network) WeightsFP4ToGPU(fp4w map[int]*FP4LayerWeights) error {
	if n.gpuMounted {
		return nil
	}

	ctx, err := gpu.GetContext()
	if err != nil {
		return fmt.Errorf("get GPU context: %w", err)
	}
	n.gpuCtx = ctx

	layers := make([]gpu.GPULayer, 0, len(n.Layers))
	outputSize := 0
	fp4Count := 0
	fp4VRAM := 0
	f32VRAM := 0

	for i, l := range n.Layers {
		if l.IsDisabled {
			continue
		}

		fw := fp4w[i]

		var gpuLayer gpu.GPULayer
		var layerOutputSize int
		var buildErr error

		switch l.Type {
		case LayerDense:
			if fw != nil && fw.Kernel != nil {
				// FP4 path — half the VRAM
				gpuLayer, layerOutputSize, buildErr = buildFP4DenseLayer(&l, fw, outputSize, i)
				if buildErr == nil {
					fp4Count++
					fp4VRAM += len(fw.Kernel.Data)/2*4 + len(fw.Kernel.Scales)*4
					f32VRAM += len(l.Kernel) * 4
				}
			}
			if buildErr != nil || gpuLayer == nil {
				// Fall back to float32 dense
				gpuLayer, layerOutputSize, buildErr = n.buildGPULayer(&l, outputSize, i)
			}
		case LayerSwiGLU:
			if fw != nil && fw.Gate != nil && fw.Up != nil && fw.Down != nil {
				gpuLayer, layerOutputSize, buildErr = buildFP4SwiGLULayer(&l, fw, outputSize, i)
				if buildErr == nil {
					fp4Count++
					fp4VRAM += nibbleVRAM(fw.Gate) + nibbleVRAM(fw.Up) + nibbleVRAM(fw.Down)
					f32VRAM += (len(l.GateWeights) + len(l.UpWeights) + len(l.DownWeights)) * 4
				}
			}
			if buildErr != nil || gpuLayer == nil {
				gpuLayer, layerOutputSize, buildErr = n.buildGPULayer(&l, outputSize, i)
			}
		case LayerMultiHeadAttention:
			if fw != nil && fw.Q != nil && fw.O != nil {
				gpuLayer, layerOutputSize, buildErr = n.buildFP4MHALayer(&l, fw, outputSize, i)
				if buildErr == nil {
					fp4Count++
					fp4VRAM += nibbleVRAM(fw.Q) + nibbleVRAM(fw.K) + nibbleVRAM(fw.V) + nibbleVRAM(fw.O)

					// F32 original VRAM calculation
					qkvSize := l.DModel * l.DModel
					if l.NumHeads != l.NumKVHeads {
						qkvSize += 2 * (l.DModel * l.NumKVHeads * l.HeadDim)
					} else {
						qkvSize *= 3
					}
					oSize := l.DModel * l.DModel
					f32VRAM += (qkvSize + oSize) * 4
				}
			}
			if buildErr != nil || gpuLayer == nil {
				gpuLayer, layerOutputSize, buildErr = n.buildGPULayer(&l, outputSize, i)
			}
		default:
			gpuLayer, layerOutputSize, buildErr = n.buildGPULayer(&l, outputSize, i)
		}

		if buildErr != nil {
			for _, built := range layers {
				built.Cleanup()
			}
			return fmt.Errorf("layer %d: %w", i, buildErr)
		}

		if gpuLayer != nil {
			// Apply batch/seq sizes (mirrors the logic in WeightsToGPU)
			if dense, ok := gpuLayer.(*gpu.DenseLayer); ok {
				seqMult := 1
				if dense.Spec.InputSize > 0 && outputSize > 0 {
					seqMult = outputSize / dense.Spec.InputSize
				}
				if seqMult < 1 {
					seqMult = 1
				}
				dense.BatchSize = n.BatchSize * seqMult
			} else if fp4d, ok := gpuLayer.(*gpu.FP4DenseLayer); ok {
				seqMult := 1
				if fp4d.Spec.InputSize > 0 && outputSize > 0 {
					seqMult = outputSize / fp4d.Spec.InputSize
				}
				if seqMult < 1 {
					seqMult = 1
				}
				fp4d.BatchSize = n.BatchSize * seqMult
			} else if swi, ok := gpuLayer.(*gpu.SwiGLULayer); ok {
				swi.BatchSize = n.BatchSize * swi.Spec.SeqLen
			} else if fp4s, ok := gpuLayer.(*gpu.FP4SwiGLULayer); ok {
				// Use l.SeqLength (= maxSeqLen, set in main.go before mount) so the
				// output buffer covers an entire prefill pass — same logic as SwiGLULayer.
				seqLen := l.SeqLength
				if seqLen < 1 {
					seqLen = 1
				}
				fp4s.BatchSize = n.BatchSize * seqLen
			} else if rms, ok := gpuLayer.(*gpu.RMSNormLayer); ok {
				rms.BatchSize = n.BatchSize * rms.Spec.BatchSize
			} else if ln, ok := gpuLayer.(*gpu.LayerNormLayer); ok {
				ln.BatchSize = n.BatchSize * ln.Spec.BatchSize
			} else if sm, ok := gpuLayer.(*gpu.SoftmaxLayer); ok {
				seqMult := 1
				if sm.Spec.Size > 0 && outputSize > 0 {
					seqMult = outputSize / sm.Spec.Size
				}
				if seqMult < 1 {
					seqMult = 1
				}
				sm.BatchSize = n.BatchSize * seqMult
			} else if mha, ok := gpuLayer.(*gpu.MHALayer); ok {
				mha.BatchSize = n.BatchSize
			} else if mhaFP4, ok := gpuLayer.(*gpu.FP4MHALayer); ok {
				mhaFP4.BatchSize = n.BatchSize
			} else if emb, ok := gpuLayer.(*gpu.EmbeddingLayer); ok {
				emb.BatchSize = n.BatchSize
			}

			layers = append(layers, gpuLayer)
			if layerOutputSize > 0 {
				outputSize = layerOutputSize
			}
		}
	}

	if len(layers) == 0 {
		return fmt.Errorf("no GPU-compatible layers found")
	}

	// Buffer chaining
	for i, l := range layers {
		label := fmt.Sprintf("L%d", i)
		if i > 0 {
			if prev := layers[i-1].GetOutputBuffer(); prev != nil {
				l.SetInputBuffer(prev)
			}
		}
		if err := l.AllocateBuffers(ctx, label); err != nil {
			n.cleanupGPULayers(layers)
			return fmt.Errorf("allocate buffers layer %d: %w", i, err)
		}
		if !n.GPUInferenceOnly {
			if err := l.AllocateBackwardBuffers(ctx, label); err != nil {
				n.cleanupGPULayers(layers)
				return fmt.Errorf("allocate backward layer %d: %w", i, err)
			}
		}
	}

	// Compile + bind groups
	for i, l := range layers {
		label := fmt.Sprintf("L%d", i)
		if err := l.Compile(ctx, label); err != nil {
			n.cleanupGPULayers(layers)
			return fmt.Errorf("compile layer %d: %w", i, err)
		}
		if err := l.CreateBindGroup(ctx, label); err != nil {
			n.cleanupGPULayers(layers)
			return fmt.Errorf("bind group layer %d: %w", i, err)
		}
		if !n.GPUInferenceOnly {
			if err := l.CompileBackward(ctx, label); err != nil {
				n.cleanupGPULayers(layers)
				return fmt.Errorf("compile backward layer %d: %w", i, err)
			}
		}
	}

	// Upload weights
	for _, l := range layers {
		l.UploadWeights(ctx)
	}

	// Residual buffers (transformer path)
	if n.EnableGPUResiduals {
		maxSize := 0
		for _, l := range layers {
			if b := l.GetInputBuffer(); b != nil {
				if s := int(b.GetSize() / 4); s > maxSize {
					maxSize = s
				}
			}
			if b := l.GetOutputBuffer(); b != nil {
				if s := int(b.GetSize() / 4); s > maxSize {
					maxSize = s
				}
			}
		}
		if maxSize > 0 {
			n.gpuResidualBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
				Label: "Residual",
				Size:  uint64(maxSize * 4),
				Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
			})
			if err != nil {
				return fmt.Errorf("create residual buffer: %w", err)
			}
			n.gpuResidualAdder, err = gpu.NewInPlaceResidual(ctx, maxSize)
			if err != nil {
				return fmt.Errorf("create residual adder: %w", err)
			}
		}
	}

	n.SyncGPU()
	n.gpuLayers = layers
	n.gpuMounted = true
	n.gpuOutputSize = outputSize

	gpu.RecordLayerCounts(len(layers), fp4Count)

	if fp4Count > 0 {
		savings := 100.0 * (1.0 - float64(fp4VRAM)/math.Max(1, float64(f32VRAM)))
		fmt.Printf("   FP4 GPU layers: %d  │  weight VRAM saved ≈ %.0f%%\n", fp4Count, savings)
	}

	return nil
}

// nibbleVRAM estimates VRAM bytes for a PackedWeights.
func nibbleVRAM(pw *PackedWeights) int {
	if pw == nil {
		return 0
	}
	return (len(pw.Data)+3)/4*4 + len(pw.Scales)*4
}

// ─────────────────────────────────────────────────────────────────────────────
// Builders
// ─────────────────────────────────────────────────────────────────────────────

func buildFP4DenseLayer(l *LayerConfig, fw *FP4LayerWeights, prevOutSize, _ int) (gpu.GPULayer, int, error) {
	pw := fw.Kernel
	if pw == nil {
		return nil, prevOutSize, nil
	}
	// FP4LayerWeights.Kernel stores [OutRows × InCols] with each row being one
	// output neuron.  PackedWeights.PackedWeights layout: index = row*nCols+col,
	// packed two per byte.
	numRowGroups := pw.NumRowGroups
	if numRowGroups == 0 {
		numRowGroups = (l.InputHeight + MicroScaleGroup - 1) / MicroScaleGroup
	}
	spec := gpu.FP4DenseSpec{
		InputSize:    l.InputHeight,
		OutputSize:   l.OutputHeight,
		NumRowGroups: numRowGroups,
		PackedData:   pw.Data,
		Scales:       pw.Scales,
		Biases:       l.Bias,
	}
	if spec.InputSize == 0 || spec.OutputSize == 0 {
		return nil, prevOutSize, nil
	}
	return &gpu.FP4DenseLayer{Spec: spec}, l.OutputHeight, nil
}

func buildFP4SwiGLULayer(l *LayerConfig, fw *FP4LayerWeights, prevOutSize, _ int) (gpu.GPULayer, int, error) {
	if fw.Gate == nil || fw.Up == nil || fw.Down == nil {
		return nil, prevOutSize, nil
	}
	spec := gpu.FP4SwiGLUSpec{
		InputSize:        l.InputHeight,
		IntermediateSize: l.OutputHeight,
		NumRowGroupsGUp:  fw.Gate.NumRowGroups,
		NumRowGroupsDown: fw.Down.NumRowGroups,
		GateData:         fw.Gate.Data,
		GateScales:       fw.Gate.Scales,
		UpData:           fw.Up.Data,
		UpScales:         fw.Up.Scales,
		DownData:         fw.Down.Data,
		DownScales:       fw.Down.Scales,
	}
	if spec.InputSize == 0 || spec.IntermediateSize == 0 {
		return nil, prevOutSize, nil
	}
	return &gpu.FP4SwiGLULayer{Spec: spec}, l.InputHeight, nil
}

func (n *Network) buildFP4MHALayer(l *LayerConfig, fw *FP4LayerWeights, prevOutSize, _ int) (gpu.GPULayer, int, error) {
	if fw.Q == nil || fw.O == nil {
		return nil, prevOutSize, nil
	}

	spec := gpu.FP4MHASpec{
		DModel:       l.DModel,
		NumHeads:     l.NumHeads,
		NumKVHeads:   l.NumKVHeads,
		SeqLen:       l.SeqLength,
		HeadDim:      l.HeadDim,
		QData:        fw.Q.Data,
		QScales:      fw.Q.Scales,
		KData:        nil,
		KScales:      nil,
		VData:        nil,
		VScales:      nil,
		OData:        fw.O.Data,
		OScales:      fw.O.Scales,
		QBias:        l.QBias,
		KBias:        l.KBias,
		VBias:        l.VBias,
		OBias:        l.OutputBias,
		MaxSeq:       l.SeqLength,
		RoPEFreqBase: l.RoPEFreqBase,
	}

	if fw.K != nil {
		spec.KData = fw.K.Data
		spec.KScales = fw.K.Scales
	}
	if fw.V != nil {
		spec.VData = fw.V.Data
		spec.VScales = fw.V.Scales
	}

	return &gpu.FP4MHALayer{Spec: spec}, l.DModel, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// ForwardFP4GPU — uses the FP4-GPU mounted network for inference
// (same signature as Forward, routes through forwardGPU)
// ─────────────────────────────────────────────────────────────────────────────

func (n *Network) ForwardFP4GPU(input []float32) ([]float32, error) {
	if !n.gpuMounted {
		return nil, fmt.Errorf("FP4 GPU not mounted; call WeightsFP4ToGPU first")
	}
	return n.forwardGPU(input)
}

// ─────────────────────────────────────────────────────────────────────────────
// Timing helper (used by the chatbot to show quantise time)
// ─────────────────────────────────────────────────────────────────────────────

var _ = time.Now // keep import used
