package poly

import (
	"fmt"
	"runtime"
	"sync"
	"time"
)

// Transformer coordinates high-level generation logic using the underlying VolumetricNetwork
type Transformer[T Numeric] struct {
	Network    *VolumetricNetwork
	Embeddings []float32
	LMHead     []float32
	FinalNorm  []float32
	HiddenSize int
	VocabSize  int
	Template   Template

	// Internal RMSNorm config for final norm if needed
	finalNormLayer *VolumetricLayer
}

// NewTransformer creates a new polymorphic transformer
func NewTransformer[T Numeric](network *VolumetricNetwork, embeddings, lmHead, finalNorm []float32, template Template) *Transformer[T] {
	hiddenSize := 0
	if len(network.Layers) > 0 {
		// Try to find hidden size from first layer
		hiddenSize = network.Layers[0].DModel
		if hiddenSize == 0 {
			hiddenSize = network.Layers[0].InputHeight
		}
	}

	if hiddenSize == 0 && finalNorm != nil {
		hiddenSize = len(finalNorm)
	}
	if hiddenSize == 0 {
		// Fallback or error
		hiddenSize = 4096 // A common default, but we should really avoid 0
	}

	vocabSize := 0
	if hiddenSize > 0 {
		vocabSize = len(embeddings) / hiddenSize
	}
	tr := &Transformer[T]{
		Network:    network,
		Embeddings: embeddings,
		LMHead:     lmHead,
		FinalNorm:  finalNorm,
		HiddenSize: hiddenSize,
		VocabSize:  vocabSize,
		Template:   template,
	}

	if finalNorm != nil {
		tr.finalNormLayer = &VolumetricLayer{
			Type:        LayerRMSNorm,
			InputHeight: hiddenSize,
			OutputHeight: hiddenSize,
			DType:       DTypeFloat32,
			WeightStore: NewWeightStore(len(finalNorm)),
		}
		copy(tr.finalNormLayer.WeightStore.Master, finalNorm)
	}

	// Set MaxSeqLen for all layers
	for i := range network.Layers {
		network.Layers[i].MaxSeqLen = 2048 // Default
	}

	return tr
}

// Reset clears the KV cache for all layers
func (t *Transformer[T]) Reset() {
	for i := range t.Network.Layers {
		t.Network.Layers[i].KVOffset = 0
		// Optional: Clear tensors to save memory if needed
		// t.Network.Layers[i].KVCacheK = nil
		// t.Network.Layers[i].KVCacheV = nil
	}
}

// EnableTiling enables cache-tiling optimization for all layers in the transformer.
// If tileSize is <= 0, it dynamically auto-detects the best size for the hardware.
func (t *Transformer[T]) EnableTiling(tileSize int) {
	for i := range t.Network.Layers {
		l := &t.Network.Layers[i]
		l.UseTiling = true
		
		finalSize := tileSize
		if finalSize <= 0 {
			// Auto-detect based on headDim
			dim := l.HeadDim
			if dim == 0 { dim = 64 } // Sane default
			finalSize = CalculateOptimalTileSize(dim)
		}
		l.TileSize = finalSize
	}
	if t.finalNormLayer != nil {
		t.finalNormLayer.UseTiling = true
		t.finalNormLayer.TileSize = tileSize
		if tileSize <= 0 { t.finalNormLayer.TileSize = 32 }
	}
}

// Generate implements the stateless generation logic
func (t *Transformer[T]) Generate(
	encode func(text string) []uint32,
	decode func(tokens []uint32) string,
	turns []Turn,
	systemPrompt, userMsg string,
	opts GenOptions,
) string {
	prompt := t.Template.BuildPrompt(turns, systemPrompt, userMsg)
	inputIDs := encode(prompt)

	tokens := inputIDs
	stream := NewStreamer(decode, tokens)
	t.Reset()

	// 1. Prefill
	prefillStart := time.Now()
	var hidden *Tensor[T]
	if len(tokens) > 0 {
		allEmbeds := NewTensor[T](len(tokens), t.HiddenSize)
		for i, tok := range tokens {
			copy(allEmbeds.Data[i*t.HiddenSize:], t.getEmbedding(int(tok)))
		}
		hidden = t.forwardFull(allEmbeds)
	}
	prefillElapsed := time.Since(prefillStart)

	// 2. Generate
	decodeStart := time.Now()
	generatedCount := 0

	for i := 0; i < opts.MaxTokens; i++ {
		// Get logits from last hidden state (last row of hidden tensor)
		lastHalf := hidden.Data[len(hidden.Data)-t.HiddenSize:]
		logits := t.applyLMHead(lastHalf)
		t.applyRepetitionPenalty(logits, tokens, opts)

		nextToken := SampleTopK(logits, opts.TopK, opts.Temperature, opts.Deterministic)
		tokens = append(tokens, uint32(nextToken))
		generatedCount++

		stream.Push(tokens)

		if t.isEOS(nextToken, opts.EOSTokens) || stream.HasNewUserTurn(tokens) {
			break
		}

		// Forward next token (Incremental)
		nextEmbed := t.getEmbedding(nextToken)
		input := NewTensor[T](1, t.HiddenSize)
		copy(input.Data, nextEmbed)
		hidden = t.forwardOne(input)
	}

	decodeElapsed := time.Since(decodeStart)
	if generatedCount > 0 {
		decodeTPS := float64(generatedCount) / decodeElapsed.Seconds()
		totalTokens := len(inputIDs) + generatedCount
		totalElapsed := prefillElapsed + decodeElapsed
		totalTPS := float64(totalTokens) / totalElapsed.Seconds()
		prefillTPS := 0.0
		if len(inputIDs) > 0 && prefillElapsed > 0 {
			prefillTPS = float64(len(inputIDs)) / prefillElapsed.Seconds()
		}
		fmt.Printf(
			"\n\n(prefill: %.2f tok/s, %d prompt tokens | decode: %.2f tok/s, %d generated | total: %.2f tok/s)\n",
			prefillTPS,
			len(inputIDs),
			decodeTPS,
			generatedCount,
			totalTPS,
		)
	}

	return stream.String()
}

func (t *Transformer[T]) getEmbedding(tokenID int) []T {
	offset := tokenID * t.HiddenSize
	if offset+t.HiddenSize > len(t.Embeddings) {
		return make([]T, t.HiddenSize)
	}
	out := make([]T, t.HiddenSize)
	for i := 0; i < t.HiddenSize; i++ {
		out[i] = T(t.Embeddings[offset+i])
	}
	return out
}

func (t *Transformer[T]) tokensToTensor(tokens []uint32) *Tensor[T] {
	out := NewTensor[T](len(tokens), t.HiddenSize)
	for i, tok := range tokens {
		copy(out.Data[i*t.HiddenSize:], t.getEmbedding(int(tok)))
	}
	return out
}

func (t *Transformer[T]) forwardOne(input *Tensor[T]) *Tensor[T] {
	return t.forwardFull(input) // MHAForwardPolymorphic now handles incrementality internally
}

func (t *Transformer[T]) forwardFull(input *Tensor[T]) *Tensor[T] {
	current := input
	numBlocks := len(t.Network.Layers) / 4
	
	for b := 0; b < numBlocks; b++ {
		// Block index
		base := b * 4
		
		// 1. Attention path
		residual := current.Clone()
		
		// Norm 1
		lNorm1 := &t.Network.Layers[base+0]
		_, current = RMSNormForwardPolymorphic(lNorm1, current)
		
		// MHA
		lMHA := &t.Network.Layers[base+1]
		_, current = MHAForwardPolymorphic(lMHA, current)
		
		// Add
		current.Add(residual)
		
		// 2. MLP path
		residual = current.Clone()
		
		// Norm 2
		lNorm2 := &t.Network.Layers[base+2]
		_, current = RMSNormForwardPolymorphic(lNorm2, current)
		
		// SwiGLU
		lMLP := &t.Network.Layers[base+3]
		_, current = SwiGLUForwardPolymorphic(lMLP, current)
		
		// Add
		current.Add(residual)
	}
	
	// Final Norm
	if t.finalNormLayer != nil {
		_, current = RMSNormForwardPolymorphic(t.finalNormLayer, current)
	}
	
	return current
}

func (t *Transformer[T]) applyLMHead(hidden []T) []float32 {
	// Final Norm
	normalized := hidden
	if t.finalNormLayer != nil {
		// normalized = RMSNormForwardPolymorphic(t.finalNormLayer, hidden)
	}

	logits := make([]float32, t.VocabSize)
	workers := runtime.GOMAXPROCS(0)

	var wg sync.WaitGroup
	chunk := (t.VocabSize + workers - 1) / workers

	for w := 0; w < workers; w++ {
		start := w * chunk
		end := start + chunk
		if end > t.VocabSize {
			end = t.VocabSize
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for v := s; v < e; v++ {
				var sum float32
				offset := v * t.HiddenSize
				for d := 0; d < t.HiddenSize; d++ {
					sum += float32(normalized[d]) * t.LMHead[offset+d]
				}
				logits[v] = sum
			}
		}(start, end)
	}
	wg.Wait()
	return logits
}

func (t *Transformer[T]) applyRepetitionPenalty(logits []float32, tokens []uint32, opts GenOptions) {
	if opts.RepetitionPenalty <= 1 || len(tokens) == 0 {
		return
	}
	window := opts.RepetitionWindow
	if window <= 0 {
		window = 64
	}
	startIdx := len(tokens) - window
	if startIdx < 0 {
		startIdx = 0
	}
	recent := tokens[startIdx:]

	seen := make(map[uint32]struct{})
	for _, tok := range recent {
		if int(tok) >= len(logits) {
			continue
		}
		if _, ok := seen[tok]; ok {
			continue
		}
		seen[tok] = struct{}{}
		if logits[tok] > 0 {
			logits[tok] /= opts.RepetitionPenalty
		} else {
			logits[tok] *= opts.RepetitionPenalty
		}
	}
}

func (t *Transformer[T]) isEOS(token int, eosTokens []int) bool {
	for _, eos := range eosTokens {
		if token == eos {
			return true
		}
	}
	return false
}
