package poly

import (
	"fmt"
	"math"
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
			Network:      network,
			Type:         LayerRMSNorm,
			InputHeight:  hiddenSize,
			OutputHeight: hiddenSize,
			DType:        DTypeFloat32,
			WeightStore:  NewWeightStore(len(finalNorm)),
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
		if tileSize > 0 {
			l.TileSize = tileSize
		}
	}
	if t.finalNormLayer != nil {
		t.finalNormLayer.UseTiling = true
		if tileSize > 0 {
			t.finalNormLayer.TileSize = tileSize
		}

		if t.Network.UseGPU {
			t.finalNormLayer.SyncToGPU()
		}
	}
}

func (t *Transformer[T]) SyncToGPU() error {
	if !t.Network.UseGPU || t.Network.GPUContext == nil {
		return fmt.Errorf("GPU not enabled")
	}
	ctx := t.Network.GPUContext
	ctx.ResetCache()

	// Sync Embeddings
	if t.Embeddings != nil && t.Network.GPUEmbeddings == nil {
		buf, err := ctx.CreatePersistentBuffer(t.Embeddings, "Embeddings")
		if err != nil {
			return err
		}
		t.Network.GPUEmbeddings = buf
	}

	// Sync LMHead
	if t.LMHead != nil && t.Network.GPULMHead == nil {
		// Check for tied weights (same memory address)
		isTied := false
		if t.Embeddings != nil && len(t.LMHead) == len(t.Embeddings) && &t.LMHead[0] == &t.Embeddings[0] {
			isTied = true
		}

		if isTied && t.Network.GPUEmbeddings != nil {
			t.Network.GPULMHead = t.Network.GPUEmbeddings
		} else {
			buf, err := ctx.CreatePersistentBuffer(t.LMHead, "LMHead")
			if err != nil {
				return err
			}
			t.Network.GPULMHead = buf
		}
	}

	// Sync Final Norm
	if t.finalNormLayer != nil {
		t.finalNormLayer.SyncToGPU()
	}

	return nil
}

// GenMetrics holds the performance measurements of a generation pass
type GenMetrics struct {
	PrefillTime      time.Duration
	DecodeTime       time.Duration
	PrefillTokens    int
	GeneratedTokens  int
	PrefillTokPerSec float64
	DecodeTokPerSec  float64
	TotalTokPerSec   float64
	FirstLogit       float32
	RAMUsageMB       float64
	VRAMUsageMB      float64
}

// Generate implements the stateless generation logic
func (t *Transformer[T]) Generate(
	encode func(text string) []uint32,
	decode func(tokens []uint32) string,
	turns []Turn,
	systemPrompt, userMsg string,
	opts GenOptions,
) (string, GenMetrics) {
	prompt := t.Template.BuildPrompt(turns, systemPrompt, userMsg)
	inputIDs := encode(prompt)

	tokens := inputIDs
	stream := NewStreamer(decode, tokens)
	t.Reset()

	// 1. Prefill
	prefillStart := time.Now()
	var logits []float32
	if len(tokens) > 0 {
		if t.Network.UseGPU && t.Network.GPUEmbeddings != nil {
			// Optimization: compute and download ONLY the last token's logits
			logitTensor, err := t.ForwardTokenIDsWGPU(tokens, nil, true, true)
			if err == nil {
				rawLogits := logitTensor.Data
				logits = make([]float32, len(rawLogits))
				for i, v := range rawLogits {
					logits[i] = float32(v)
				}
			} else {
				fmt.Printf("⚠️  GPU Prefill Failed: %v (Falling back to CPU)\n", err)
				allEmbeds := t.TokensToTensor(tokens)
				hidden := t.ForwardFull(allEmbeds)
				lastHalf := hidden.Data[len(hidden.Data)-t.HiddenSize:]
				logits = t.ApplyLMHead(lastHalf)
			}
		} else {
			allEmbeds := t.TokensToTensor(tokens)
			hidden := t.ForwardFull(allEmbeds)
			lastHalf := hidden.Data[len(hidden.Data)-t.HiddenSize:]
			logits = t.ApplyLMHead(lastHalf)
		}
	}
	prefillElapsed := time.Since(prefillStart)

	metrics := GenMetrics{
		PrefillTime:   prefillElapsed,
		PrefillTokens: len(tokens),
	}
	if len(logits) > 0 {
		metrics.FirstLogit = logits[0]
	}

	// 2. Generate
	decodeStart := time.Now()
	generatedCount := 0

	for i := 0; i < opts.MaxTokens; i++ {
		t.applyRepetitionPenalty(logits, tokens, opts)
		t.applyBannedTokenMask(logits, opts)
		t.applyConsecutiveRepeatMask(logits, tokens, opts)
		t.applyNoRepeatNGramMask(logits, tokens, opts)

		nextToken := SampleTopK(logits, opts.TopK, opts.Temperature, opts.Deterministic)

		tokens = append(tokens, uint32(nextToken))
		stream.Push(tokens, opts.Silent, opts.StreamCallback)
		generatedCount++

		if (generatedCount >= opts.MinTokens && t.isEOS(nextToken, opts.EOSTokens)) || (opts.UseKVCache && stream.HasNewUserTurn(tokens)) {
			break
		}

		// Forward next token (Incremental)
		if t.Network.UseGPU && t.Network.GPUEmbeddings != nil {
			logitTensor, err := t.ForwardTokenIDsWGPU([]uint32{uint32(nextToken)}, nil, true, true)
			if err == nil {
				rawLogits := logitTensor.Data
				logits = make([]float32, len(rawLogits))
				for j, v := range rawLogits {
					logits[j] = float32(v)
				}
			} else {
				fmt.Printf("⚠️  GPU Incremental Failed: %v (Falling back to CPU)\n", err)
				nextEmbed := t.getEmbedding(nextToken)
				input := NewTensor[T](1, t.HiddenSize)
				copy(input.Data, nextEmbed)
				hidden := t.forwardOne(input)
				lastHalf := hidden.Data[len(hidden.Data)-t.HiddenSize:]
				logits = t.ApplyLMHead(lastHalf)
			}
		} else {
			nextEmbed := t.getEmbedding(nextToken)
			input := NewTensor[T](1, t.HiddenSize)
			copy(input.Data, nextEmbed)
			hidden := t.forwardOne(input)
			lastHalf := hidden.Data[len(hidden.Data)-t.HiddenSize:]
			logits = t.ApplyLMHead(lastHalf)
		}
	}

	decodeElapsed := time.Since(decodeStart)
	metrics.DecodeTime = decodeElapsed
	metrics.GeneratedTokens = generatedCount

	ramBytes := t.Network.CalculateTotalMemory()
	vramBytes := t.Network.GetVRAMUsage()
	metrics.RAMUsageMB = float64(ramBytes) / (1024 * 1024)
	metrics.VRAMUsageMB = float64(vramBytes) / (1024 * 1024)

	if generatedCount > 0 {
		metrics.DecodeTokPerSec = float64(generatedCount) / decodeElapsed.Seconds()
		totalTokens := len(inputIDs) + generatedCount
		totalElapsed := prefillElapsed + decodeElapsed
		metrics.TotalTokPerSec = float64(totalTokens) / totalElapsed.Seconds()
		if len(inputIDs) > 0 && prefillElapsed > 0 {
			metrics.PrefillTokPerSec = float64(len(inputIDs)) / prefillElapsed.Seconds()
		}
		if !opts.Silent {
			fmt.Printf(
				"\n\n(prefill: %.2f tok/s, %d prompt tokens | decode: %.2f tok/s, %d generated | total: %.2f tok/s)\n",
				metrics.PrefillTokPerSec,
				len(inputIDs),
				metrics.DecodeTokPerSec,
				generatedCount,
				metrics.TotalTokPerSec,
			)
			fmt.Printf("(ram: %.2f MB | vram: %.2f MB)\n", metrics.RAMUsageMB, metrics.VRAMUsageMB)
		}
	}

	return stream.String(), metrics
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

func (t *Transformer[T]) TokensToTensor(tokens []uint32) *Tensor[T] {
	out := NewTensor[T](len(tokens), t.HiddenSize)
	for i, tok := range tokens {
		copy(out.Data[i*t.HiddenSize:], t.getEmbedding(int(tok)))
	}
	return out
}

func (t *Transformer[T]) forwardOne(input *Tensor[T]) *Tensor[T] {
	if t.Network.UseGPU && t.Network.GPUContext != nil {
		res, err := t.ForwardWGPU(input)
		if err == nil {
			if t.finalNormLayer != nil {
				_, res = RMSNormForwardPolymorphic(t.finalNormLayer, res)
			}
			return res
		}
		fmt.Printf("⚠️  GPU Forward Failed: %v (Falling back to CPU)\n", err)
	}
	return t.ForwardFull(input)
}

func (t *Transformer[T]) ForwardFull(input *Tensor[T]) *Tensor[T] {
	if t.Network.UseGPU && t.Network.GPUContext != nil {
		res, err := t.ForwardWGPU(input)
		if err == nil {
			if t.finalNormLayer != nil {
				_, res = RMSNormForwardPolymorphic(t.finalNormLayer, res)
			}
			return res
		}
		fmt.Printf("⚠️  GPU Full Forward Failed: %v (Falling back to CPU)\n", err)
	}

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

func (t *Transformer[T]) ApplyLMHead(hidden []T) []float32 {
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

func (t *Transformer[T]) applyBannedTokenMask(logits []float32, opts GenOptions) {
	if len(logits) == 0 || len(opts.BannedTokens) == 0 {
		return
	}
	eosSet := make(map[int]struct{}, len(opts.EOSTokens))
	for _, eos := range opts.EOSTokens {
		eosSet[eos] = struct{}{}
	}
	for _, tok := range opts.BannedTokens {
		if tok < 0 || tok >= len(logits) {
			continue
		}
		// Never ban EOS, even if it appears in a broad special-token mask.
		if _, isEOS := eosSet[tok]; isEOS {
			continue
		}
		logits[tok] = -math.MaxFloat32
	}
}

func (t *Transformer[T]) applyConsecutiveRepeatMask(logits []float32, tokens []uint32, opts GenOptions) {
	if len(logits) == 0 || opts.MaxConsecutiveRepeats <= 0 || len(tokens) < opts.MaxConsecutiveRepeats {
		return
	}
	last := tokens[len(tokens)-1]
	repeats := 1
	for i := len(tokens) - 2; i >= 0; i-- {
		if tokens[i] != last {
			break
		}
		repeats++
	}
	if repeats < opts.MaxConsecutiveRepeats {
		return
	}
	if int(last) >= 0 && int(last) < len(logits) {
		logits[last] = -math.MaxFloat32
	}
}

func (t *Transformer[T]) applyNoRepeatNGramMask(logits []float32, tokens []uint32, opts GenOptions) {
	n := opts.NoRepeatNGram
	if n <= 1 || len(logits) == 0 || len(tokens) < n-1 {
		return
	}

	prefixStart := len(tokens) - (n - 1)
	prefix := tokens[prefixStart:]

	for i := 0; i+n-1 < len(tokens); i++ {
		match := true
		for j := 0; j < n-1; j++ {
			if tokens[i+j] != prefix[j] {
				match = false
				break
			}
		}
		if !match {
			continue
		}
		next := tokens[i+n-1]
		if int(next) >= 0 && int(next) < len(logits) {
			logits[next] = -math.MaxFloat32
		}
	}
}
