package nn

import (
	"fmt"
	"math"
	"time"
)

// =============================================================================
// FP4-Accelerated CPU Forward Pass
// =============================================================================

// FP4LayerWeights holds pre-quantised PackedWeights for one layer.
// For Dense layers, only Kernel is populated.
// For SwiGLU layers, Gate, Up, and Down are populated.
type FP4LayerWeights struct {
	Kernel *PackedWeights // Dense kernel, OR nil
	Gate   *PackedWeights // SwiGLU gate_proj weights
	Up     *PackedWeights // SwiGLU up_proj weights
	Down   *PackedWeights // SwiGLU down_proj weights

	// MHA weights
	Q *PackedWeights
	K *PackedWeights
	V *PackedWeights
	O *PackedWeights
}

// BuildFP4Weights walks all layers of a Network and pre-quantises every
// Dense and SwiGLU weight matrix into E2M1 PackedWeights.
//
// Returns a map keyed by the flattened layer index (same ordering as
// n.preActivations / n.activations).  Call this once after loading a
// model and pass the result to ForwardFP4CPU for every inference step.
func (n *Network) BuildFP4Weights() map[int]*FP4LayerWeights {
	result := make(map[int]*FP4LayerWeights)
	layerIdx := 0

	for row := 0; row < n.GridRows; row++ {
		for col := 0; col < n.GridCols; col++ {
			for layer := 0; layer < n.LayersPerCell; layer++ {
				cfg := n.GetLayer(row, col, layer)
				if cfg == nil || cfg.IsDisabled {
					layerIdx++
					continue
				}

				switch cfg.Type {
				case LayerDense:
					if len(cfg.Kernel) > 0 && cfg.InputHeight > 0 && cfg.OutputHeight > 0 {
						fw := &FP4LayerWeights{
							Kernel: NewPackedWeights(cfg.Kernel, cfg.InputHeight, cfg.OutputHeight),
						}
						result[layerIdx] = fw
					}

				case LayerSwiGLU:
					if len(cfg.GateWeights) > 0 && cfg.InputHeight > 0 && cfg.OutputHeight > 0 {
						inputSize := cfg.InputHeight
						intermediateSize := cfg.OutputHeight
						fw := &FP4LayerWeights{
							Gate: NewPackedWeights(cfg.GateWeights, inputSize, intermediateSize),
							Up:   NewPackedWeights(cfg.UpWeights, inputSize, intermediateSize),
							Down: NewPackedWeights(cfg.DownWeights, intermediateSize, inputSize),
						}
						result[layerIdx] = fw
					}

				case LayerMultiHeadAttention:
					// MHA projections: input size is DModel.
					// Output sizes:
					// Q: NumHeads * HeadDim
					// K, V: NumKVHeads * HeadDim
					// O: DModel * DModel

					qOut := cfg.NumHeads * cfg.HeadDim
					kvOut := cfg.NumKVHeads * cfg.HeadDim

					if len(cfg.QWeights) > 0 && cfg.DModel > 0 {
						fw := &FP4LayerWeights{}
						if len(cfg.QWeights) > 0 {
							fw.Q = NewPackedWeights(cfg.QWeights, cfg.DModel, qOut)
						}
						if len(cfg.KWeights) > 0 {
							fw.K = NewPackedWeights(cfg.KWeights, cfg.DModel, kvOut)
						}
						if len(cfg.VWeights) > 0 {
							fw.V = NewPackedWeights(cfg.VWeights, cfg.DModel, kvOut)
						}
						if len(cfg.OutputWeight) > 0 {
							fw.O = NewPackedWeights(cfg.OutputWeight, cfg.DModel, cfg.DModel)
						}
						result[layerIdx] = fw
					}
				}

				layerIdx++
			}
		}
	}
	return result
}

// ForwardFP4CPU executes the network forward pass on CPU, substituting
// E2M1 bitwise multiply-accumulate for every Dense and SwiGLU layer
// that has an entry in fp4Weights.  All other layer types fall back to
// the normal float32 implementation.
//
// The signature is intentionally identical to ForwardCPU so callers can
// swap one for the other.
func (n *Network) ForwardFP4CPU(input []float32, fp4Weights map[int]*FP4LayerWeights) ([]float32, time.Duration) {
	start := time.Now()

	n.activations[0] = make([]float32, len(input))
	copy(n.activations[0], input)

	data := make([]float32, len(input))
	copy(data, input)

	layerIdx := 0
	var residualInput []float32

	for row := 0; row < n.GridRows; row++ {
		for col := 0; col < n.GridCols; col++ {
			for layer := 0; layer < n.LayersPerCell; layer++ {
				config := n.GetLayer(row, col, layer)
				var preAct, postAct []float32

				if config.IsDisabled {
					n.preActivations[layerIdx] = make([]float32, len(data))
					copy(n.preActivations[layerIdx], data)

				} else if config.Type == LayerDense {
					actualBatch := n.BatchSize
					if config.InputHeight > 0 && len(data)%config.InputHeight == 0 {
						actualBatch = len(data) / config.InputHeight
					}

					if fw, ok := fp4Weights[layerIdx]; ok && fw.Kernel != nil {
						// ── FP4 path ──────────────────────────────────────────
						preAct, postAct = DenseForwardFP4(data, fw.Kernel, config.Bias, actualBatch, config.Activation)
					} else {
						// ── float32 fallback ──────────────────────────────────
						preAct, postAct = denseForwardCPU(data, config, actualBatch)
					}
					n.preActivations[layerIdx] = preAct
					data = postAct

				} else if config.Type == LayerSwiGLU {
					actualBatch := n.BatchSize
					if config.InputHeight > 0 && len(data)%config.InputHeight == 0 {
						actualBatch = len(data) / config.InputHeight
					}

					if fw, ok := fp4Weights[layerIdx]; ok && fw.Gate != nil {
						// ── FP4 SwiGLU path ───────────────────────────────────
						preAct, postAct = swiGLUForwardFP4(data, config, fw, actualBatch)
					} else {
						preAct, postAct = SwiGLUForwardCPU(data, config, actualBatch)
					}

					if residualInput != nil && len(residualInput) == len(postAct) {
						for i := range postAct {
							postAct[i] += residualInput[i]
						}
					}
					n.preActivations[layerIdx] = preAct
					residualInput = make([]float32, len(postAct))
					copy(residualInput, postAct)
					data = postAct

				} else if config.Type == LayerMultiHeadAttention {
					actualBatch := n.BatchSize
					if config.InputHeight > 0 && len(data)%config.InputHeight == 0 {
						actualBatch = len(data) / config.InputHeight
					}
					preAct, postAct = MultiHeadAttentionForwardCPU(data, config, actualBatch)
					if residualInput != nil && len(residualInput) == len(postAct) {
						for i := range postAct {
							postAct[i] += residualInput[i]
						}
					}
					n.preActivations[layerIdx] = preAct
					residualInput = make([]float32, len(postAct))
					copy(residualInput, postAct)
					data = postAct

				} else if config.Type == LayerRMSNorm {
					actualBatch := n.BatchSize
					if config.NormSize > 0 && len(data)%config.NormSize == 0 {
						actualBatch = len(data) / config.NormSize
					}
					normalized := rmsNormForwardCPU(data, nil, config, actualBatch)
					n.preActivations[layerIdx] = make([]float32, len(data))
					copy(n.preActivations[layerIdx], data)
					residualInput = make([]float32, len(data))
					copy(residualInput, data)
					data = normalized

				} else if config.Type == LayerNorm {
					actualBatch := n.BatchSize
					if config.NormSize > 0 && len(data)%config.NormSize == 0 {
						actualBatch = len(data) / config.NormSize
					}
					normalized := layerNormForwardCPU(data, residualInput, config, actualBatch)
					n.preActivations[layerIdx] = make([]float32, len(data))
					copy(n.preActivations[layerIdx], data)
					if residualInput != nil {
						residualInput = make([]float32, len(normalized))
						copy(residualInput, normalized)
					}
					data = normalized

				} else if config.Type == LayerEmbedding {
					output := embeddingForwardCPU(data, config)
					n.preActivations[layerIdx] = make([]float32, len(data))
					copy(n.preActivations[layerIdx], data)
					data = output

				} else if config.Type == LayerSoftmax {
					inputSize := config.InputHeight
					if inputSize == 0 {
						inputSize = len(data)
					}
					actualBatch := len(data) / inputSize
					if actualBatch == 0 {
						actualBatch = 1
					}
					probs := make([]float32, len(data))
					for i := 0; i < actualBatch; i++ {
						s := i * inputSize
						e := s + inputSize
						if e > len(data) {
							break
						}
						sp, err := ForwardSoftmaxCPU(data[s:e], config)
						if err != nil {
							sp = softmaxStandard(data[s:e], 1.0)
						}
						copy(probs[s:e], sp)
					}
					n.preActivations[layerIdx] = make([]float32, len(data))
					copy(n.preActivations[layerIdx], data)
					data = probs

				} else {
					// Default: element-wise activation only
					n.preActivations[layerIdx] = make([]float32, len(data))
					copy(n.preActivations[layerIdx], data)
					for i := range data {
						data[i] = activateCPU(data[i], config.Activation)
					}
				}

				n.activations[layerIdx+1] = make([]float32, len(data))
				copy(n.activations[layerIdx+1], data)

				if config.Type != LayerParallel && config.Observer != nil {
					notifyObserver(config, "normal", "forward", layerIdx, data, n.activations[layerIdx+1], 0)
				}

				layerIdx++
			}
		}
	}

	return data, time.Since(start)
}

// swiGLUForwardFP4 runs a SwiGLU layer using FP4 for all three projections.
// gate_proj and up_proj are run in FP4; their outputs feed the SiLU gate.
// down_proj is also FP4.  The intermediate float32 accumulation happens
// normally between the FP4 matmuls.
func swiGLUForwardFP4(input []float32, config *LayerConfig, fw *FP4LayerWeights, batchSize int) (preAct, postAct []float32) {
	inputSize := config.InputHeight
	intermediateSize := config.OutputHeight

	seqLen := len(input) / inputSize
	if seqLen == 0 {
		seqLen = 1
	}

	// Bias slices (may be nil/empty)
	gateBias := config.GateBias
	upBias := config.UpBias
	downBias := config.DownBias

	// For each token position in the sequence:
	// 1. gate_proj(x) via FP4
	// 2. up_proj(x) via FP4
	// 3. silu(gate) * up  — float32
	// 4. down_proj(intermediate) via FP4
	intermediate := make([]float32, seqLen*intermediateSize)

	for s := 0; s < seqLen; s++ {
		rowSlice := input[s*inputSize : (s+1)*inputSize]

		aNibbles, aScales := QuantiseInputRowFP4(rowSlice)

		// Gate projection
		gateRow := ForwardRowFP4(aNibbles, aScales, fw.Gate, gateBias)
		// Up projection
		upRow := ForwardRowFP4(aNibbles, aScales, fw.Up, upBias)

		// silu(gate) * up → intermediate
		for i := 0; i < intermediateSize; i++ {
			g := float64(gateRow[i])
			sig := 1.0 / (1.0 + expNeg(g))
			silu := float32(g * sig)
			intermediate[s*intermediateSize+i] = silu * upRow[i]
		}
	}

	// Down projection over all sequence positions at once
	downPreAct, downPostAct := DenseForwardFP4(intermediate, fw.Down, downBias, seqLen, ActivationType(-1))
	_ = downPostAct // postAct = preAct for linear activation

	// For SwiGLU, the final output has the same shape as the input
	outputSize := inputSize * seqLen
	if len(downPreAct) != outputSize {
		fmt.Printf("[FP4 SwiGLU] size mismatch: got %d want %d\n", len(downPreAct), outputSize)
	}

	return downPreAct, downPreAct
}

// expNeg approximates exp(-x) using the standard math package pathway.
// Inlining the float64 avoids an import of "math" in this file.
func expNeg(x float64) float64 {
	return math.Exp(-x)
}
