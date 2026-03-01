package tokenizer

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"

	"github.com/openfluke/loom/nn"
)

// LayerInference defines the interface for a layer's inference logic using KV cache
type LayerInference interface {
	ForwardKV(input []float32, cfg *nn.LayerConfig, pos int, state *KVCacheLayer) ([]float32, error)
}

// Global registry for layer inference handlers
var inferenceRegistry = make(map[nn.LayerType]LayerInference)

// RegisterLayerInference registers a handler for a specific layer type
func RegisterLayerInference(t nn.LayerType, handler LayerInference) {
	inferenceRegistry[t] = handler
}

// KVCacheLayer holds the K and V tensors for a single attention layer
type KVCacheLayer struct {
	K          []float32
	V          []float32
	MaxSeq     int
	NumKVHeads int
	HeadDim    int
}

// KVCacheState manages the KV cache across all layers of a network
type KVCacheState struct {
	Layers map[int]*KVCacheLayer
}

// InitKVCacheState initializes a new KV cache state for the given network
func InitKVCacheState(network *nn.Network, maxSeq int) *KVCacheState {
	state := &KVCacheState{Layers: make(map[int]*KVCacheLayer)}
	for i := range network.Layers {
		cfg := &network.Layers[i]
		if cfg.Type != nn.LayerMultiHeadAttention {
			continue
		}
		numKV := cfg.NumKVHeads
		if numKV == 0 {
			numKV = cfg.NumHeads
		}
		kvDim := numKV * cfg.HeadDim
		state.Layers[i] = &KVCacheLayer{
			K:          make([]float32, maxSeq*kvDim),
			V:          make([]float32, maxSeq*kvDim),
			MaxSeq:     maxSeq,
			NumKVHeads: numKV,
			HeadDim:    cfg.HeadDim,
		}
	}
	return state
}

// ForwardTokenKV performs a single-token forward pass using the KV cache and registered handlers
func ForwardTokenKV(network *nn.Network, tokenID int, pos int, state *KVCacheState, embeddings []float32, hiddenSize int) ([]float32, error) {
	// 1. Embedding lookup
	vocabSize := len(embeddings) / hiddenSize
	if tokenID < 0 || tokenID >= vocabSize {
		return nil, fmt.Errorf("token ID %d out of bounds for vocab", tokenID)
	}
	offset := tokenID * hiddenSize
	input := make([]float32, hiddenSize)
	copy(input, embeddings[offset:offset+hiddenSize])

	var residualInput []float32
	for i := range network.Layers {
		cfg := &network.Layers[i]

		// 2. Handle known structural layers (RMSNorm, SwiGLU) or delegated handlers
		handler, ok := inferenceRegistry[cfg.Type]
		if ok {
			cache := state.Layers[i] // May be nil for non-attention layers
			out, err := handler.ForwardKV(input, cfg, pos, cache)
			if err != nil {
				return nil, err
			}

			// Handle residuals if necessary (this logic stays here for orchestration)
			if cfg.Type == nn.LayerMultiHeadAttention || cfg.Type == nn.LayerSwiGLU {
				if residualInput != nil && len(residualInput) == len(out) {
					for j := range out {
						out[j] += residualInput[j]
					}
				}
				residualInput = make([]float32, len(out))
				copy(residualInput, out)
			}
			input = out
			continue
		}

		// Fallback for primitive structural layers if no handler registered
		switch cfg.Type {
		case nn.LayerEmbedding:
			continue
		case nn.LayerRMSNorm:
			residualInput = make([]float32, len(input))
			copy(residualInput, input)
			input = nn.RmsNormForwardCPU(input, nil, cfg, 1)
		default:
			return nil, fmt.Errorf("kv cache path does not support layer type %v (no handler registered)", cfg.Type)
		}
	}

	return input, nil
}

// Default implementation for Multi-Head Attention
type MHAInference struct{}

func (m MHAInference) ForwardKV(input []float32, cfg *nn.LayerConfig, pos int, cache *KVCacheLayer) ([]float32, error) {
	if cache == nil {
		return nil, fmt.Errorf("kv cache missing for MHA layer")
	}
	if pos >= cache.MaxSeq {
		return nil, fmt.Errorf("kv cache exceeded max seq len %d", cache.MaxSeq)
	}

	dModel := cfg.DModel
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	if numKVHeads == 0 {
		numKVHeads = numHeads
	}
	headDim := cfg.HeadDim
	kvDim := numKVHeads * headDim

	q := make([]float32, dModel)
	for outDim := 0; outDim < dModel; outDim++ {
		sum := cfg.QBias[outDim]
		for inDim := 0; inDim < dModel; inDim++ {
			sum += input[inDim] * cfg.QWeights[inDim*dModel+outDim]
		}
		q[outDim] = sum
	}

	k := make([]float32, kvDim)
	v := make([]float32, kvDim)
	for outDim := 0; outDim < kvDim; outDim++ {
		sumK := cfg.KBias[outDim]
		sumV := cfg.VBias[outDim]
		for inDim := 0; inDim < dModel; inDim++ {
			sumK += input[inDim] * cfg.KWeights[inDim*kvDim+outDim]
			sumV += input[inDim] * cfg.VWeights[inDim*kvDim+outDim]
		}
		k[outDim] = sumK
		v[outDim] = sumV
	}

	ropeTheta := float64(cfg.RoPEFreqBase)
	if ropeTheta == 0 {
		ropeTheta = 10000.0
	}

	for head := 0; head < numHeads; head++ {
		off := head * headDim
		ApplyRoPEHead(q[off:off+headDim], pos, headDim, ropeTheta)
	}
	for head := 0; head < numKVHeads; head++ {
		off := head * headDim
		ApplyRoPEHead(k[off:off+headDim], pos, headDim, ropeTheta)
	}

	cacheOffset := pos * kvDim
	copy(cache.K[cacheOffset:cacheOffset+kvDim], k)
	copy(cache.V[cacheOffset:cacheOffset+kvDim], v)

	headsPerKV := numHeads / numKVHeads
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	attnOut := make([]float32, dModel)

	for head := 0; head < numHeads; head++ {
		kvHead := head / headsPerKV
		qOff := head * headDim
		qHead := q[qOff : qOff+headDim]

		scores := make([]float32, pos+1)
		maxScore := float32(-1e9)
		for t := 0; t <= pos; t++ {
			kOff := t*kvDim + kvHead*headDim
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += qHead[d] * cache.K[kOff+d]
			}
			score := dot * scale
			scores[t] = score
			if score > maxScore {
				maxScore = score
			}
		}

		var expSum float32
		for t := 0; t <= pos; t++ {
			val := float32(math.Exp(float64(scores[t] - maxScore)))
			scores[t] = val
			expSum += val
		}
		if expSum == 0 {
			continue
		}
		for t := 0; t <= pos; t++ {
			scores[t] /= expSum
		}

		for d := 0; d < headDim; d++ {
			var sum float32
			for t := 0; t <= pos; t++ {
				vOff := t*kvDim + kvHead*headDim
				sum += scores[t] * cache.V[vOff+d]
			}
			attnOut[qOff+d] = sum
		}
	}

	output := make([]float32, dModel)
	for outDim := 0; outDim < dModel; outDim++ {
		sum := cfg.OutputBias[outDim]
		for inDim := 0; inDim < dModel; inDim++ {
			sum += attnOut[inDim] * cfg.OutputWeight[inDim*dModel+outDim]
		}
		output[outDim] = sum
	}

	return output, nil
}

// Default implementation for SwiGLU
type SwiGLUInference struct{}

func (s SwiGLUInference) ForwardKV(input []float32, cfg *nn.LayerConfig, pos int, cache *KVCacheLayer) ([]float32, error) {
	_, out := nn.SwiGLUForwardCPU(input, cfg, 1)
	return out, nil
}

func init() {
	RegisterLayerInference(nn.LayerMultiHeadAttention, MHAInference{})
	RegisterLayerInference(nn.LayerSwiGLU, SwiGLUInference{})
}

// ApplyRoPEHead applies RoPE to a single head's dimension vector
func ApplyRoPEHead(vec []float32, pos int, headDim int, theta float64) {
	orig := make([]float32, headDim)
	copy(orig, vec)
	half := headDim / 2
	for d := 0; d < headDim; d++ {
		freq := 1.0 / math.Pow(theta, float64(2*(d%half))/float64(headDim))
		angle := freq * float64(pos)
		c := float32(math.Cos(angle))
		s := float32(math.Sin(angle))
		var rotated float32
		if d < half {
			rotated = -orig[d+half]
		} else {
			rotated = orig[d-half]
		}
		vec[d] = orig[d]*c + rotated*s
	}
}

// SampleTopK performs top-K sampling with temperature and optional determinism
func SampleTopK(logits []float32, topK int, temperature float32, deterministic bool) int {
	if topK == 1 || temperature <= 0 {
		maxIdx := 0
		maxVal := logits[0]
		for i, v := range logits {
			if v > maxVal {
				maxVal = v
				maxIdx = i
			}
		}
		return maxIdx
	}

	type pair struct {
		idx int
		val float32
	}
	cands := make([]pair, 0, len(logits))
	for i, v := range logits {
		cands = append(cands, pair{i, v / temperature})
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].val > cands[j].val })
	if topK > 0 && topK < len(cands) {
		cands = cands[:topK]
	}

	maxV := cands[0].val
	var sum float64
	probs := make([]float64, len(cands))
	for i := range cands {
		p := math.Exp(float64(cands[i].val - maxV))
		probs[i] = p
		sum += p
	}

	r := rand.Float64() * sum
	acc := 0.0
	for i := range probs {
		acc += probs[i]
		if r <= acc {
			return cands[i].idx
		}
	}
	return cands[len(cands)-1].idx
}

// Streamer handles real-time output of generated tokens
type Streamer struct {
	tk           *Tokenizer
	lastLen      int
	promptLenRaw int
	sb           strings.Builder
	replacer     *strings.Replacer
}

func NewStreamer(tk *Tokenizer, promptTokens []uint32) *Streamer {
	promptTextRaw := tk.Decode(promptTokens, false)
	return &Streamer{
		tk:           tk,
		lastLen:      len(promptTextRaw),
		promptLenRaw: len(promptTextRaw),
		replacer: strings.NewReplacer(
			"<|im_end|>", "",
			"<|im_start|>assistant", "",
			"<|im_start|>user", "",
			"<|im_start|>system", "",
			"<|im_start|>assistant\n", "",
		),
	}
}

func (s *Streamer) Push(allTokens []uint32) {
	full := s.tk.Decode(allTokens, false)
	if len(full) > s.lastLen {
		diff := full[s.lastLen:]
		diff = s.replacer.Replace(diff)
		fmt.Print(diff)
		s.sb.WriteString(diff)
		s.lastLen = len(full)
	}
}

func (s *Streamer) String() string {
	return strings.TrimSpace(s.sb.String())
}

func (s *Streamer) HasNewUserTurn(allTokens []uint32) bool {
	fullRaw := s.tk.Decode(allTokens, false)
	if len(fullRaw) <= s.promptLenRaw {
		return false
	}
	return strings.Contains(fullRaw[s.promptLenRaw:], "<|im_start|>user")
}

func IntsToU32(xs []int) []uint32 {
	out := make([]uint32, len(xs))
	for i, v := range xs {
		out[i] = uint32(v)
	}
	return out
}
