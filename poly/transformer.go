package poly

import (
	"fmt"
	"math"
	"runtime"
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

	// hostWeightsReleased is set after ReleaseInferenceHostWeights; CPU fallback paths are unsafe.
	hostWeightsReleased bool
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
			RMSNormEps:   1e-6,
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

// SyncInferenceCPU runs VolumetricNetwork.SyncToCPU and syncs the transformer's
// standalone final RMSNorm layer (not part of net.Layers) so CPU tile flags match.
func (t *Transformer[T]) SyncInferenceCPU() {
	t.Network.SyncToCPU()
	if t.finalNormLayer != nil {
		t.finalNormLayer.SyncToCPU()
	}
}

// SetRMSNormEps applies a consistent epsilon to all RMSNorm layers, including
// the transformer's final RMSNorm adapter.
func (t *Transformer[T]) SetRMSNormEps(eps float64) {
	if eps <= 0 {
		return
	}
	for i := range t.Network.Layers {
		switch t.Network.Layers[i].Type {
		case LayerRMSNorm, LayerMultiHeadAttention:
			t.Network.Layers[i].RMSNormEps = eps
		}
	}
	if t.finalNormLayer != nil {
		t.finalNormLayer.RMSNormEps = eps
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

// ReleaseInferenceHostWeights drops CPU-side weight tensors after VRAM buffers exist (GPU inference only).
// After this call, CPU fallback in Generate/ForwardFull is disabled for this transformer.
func (t *Transformer[T]) ReleaseInferenceHostWeights() {
	if t.Network == nil || !t.Network.UseGPU || t.Network.GPUContext == nil || t.Network.GPUEmbeddings == nil {
		return
	}
	for i := range t.Network.Layers {
		t.Network.Layers[i].ReleaseInferenceHostWeights()
	}
	t.Embeddings = nil
	t.LMHead = nil
	t.FinalNorm = nil
	if t.finalNormLayer != nil {
		t.finalNormLayer.ReleaseInferenceHostWeights()
	}
	t.hostWeightsReleased = true
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
	// RAMUsageMB is kept for backward compatibility and now represents
	// approximate current process RAM (not just static layer weights).
	RAMUsageMB float64
	// ModelRAMUsageMB is host-side model memory tracked by Loom (weights/tensors).
	ModelRAMUsageMB float64
	VRAMUsageMB     float64
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
				fmt.Printf("⚠️  GPU Prefill Failed: %v\n", err)
				if t.hostWeightsReleased || len(t.Embeddings) == 0 {
					fmt.Println("⚠️  CPU fallback unavailable (weights are GPU-resident only).")
					return stream.String(), GenMetrics{
						PrefillTime:   time.Since(prefillStart),
						PrefillTokens: len(tokens),
					}
				}
				allEmbeds := t.TokensToTensor(tokens)
				hidden := t.ForwardFull(allEmbeds)
				logits = t.ApplyLMHead(t.lastHiddenRow(hidden))
			}
		} else {
			allEmbeds := t.TokensToTensor(tokens)
			hidden := t.ForwardFull(allEmbeds)
			logits = t.ApplyLMHead(t.lastHiddenRow(hidden))
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
				fmt.Printf("⚠️  GPU Incremental Failed: %v\n", err)
				if t.hostWeightsReleased || len(t.Embeddings) == 0 {
					fmt.Println("⚠️  CPU fallback unavailable (weights are GPU-resident only).")
					return stream.String(), metrics
				}
				nextEmbed := t.getEmbedding(nextToken)
				input := NewTensor[T](1, t.HiddenSize)
				copy(input.Data, nextEmbed)
				hidden := t.forwardOne(input)
				logits = t.ApplyLMHead(t.lastHiddenRow(hidden))
			}
		} else {
			nextEmbed := t.getEmbedding(nextToken)
			input := NewTensor[T](1, t.HiddenSize)
			copy(input.Data, nextEmbed)
			hidden := t.forwardOne(input)
			logits = t.ApplyLMHead(t.lastHiddenRow(hidden))
		}
	}

	decodeElapsed := time.Since(decodeStart)
	metrics.DecodeTime = decodeElapsed
	metrics.GeneratedTokens = generatedCount

	modelRAMBytes := t.calculateHostModelBytes()
	processRAMBytes := currentProcessRAMBytes()
	vramBytes := t.Network.GetVRAMUsage()
	metrics.RAMUsageMB = float64(processRAMBytes) / (1024 * 1024)
	metrics.ModelRAMUsageMB = float64(modelRAMBytes) / (1024 * 1024)
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
			fp := NewMemoryFootprintFromTransformer(t)
			fmt.Printf("(%s)\n", fp.FormatOneLine())
		}
	}

	return stream.String(), metrics
}

func currentProcessRAMBytes() uint64 {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	// Sys is bytes obtained from the OS and generally tracks real process
	// memory far better than Alloc for large model loads.
	return ms.Sys
}

func (t *Transformer[T]) calculateHostModelBytes() uint64 {
	var total uint64
	total += uint64(t.Network.CalculateTotalMemory())
	total += uint64(len(t.Embeddings)) * 4
	if !slicesShareBackingStoreFloat32(t.LMHead, t.Embeddings) {
		total += uint64(len(t.LMHead)) * 4
	}
	total += uint64(len(t.FinalNorm)) * 4
	return total
}

func slicesShareBackingStoreFloat32(a, b []float32) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	return &a[0] == &b[0]
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
	h := t.HiddenSize
	for i, tok := range tokens {
		off := int(tok) * h
		row := out.Data[i*h : (i+1)*h]
		if off+h <= len(t.Embeddings) {
			emb := t.Embeddings[off : off+h]
			for j := 0; j < h; j++ {
				row[j] = T(emb[j])
			}
		}
	}
	return out
}

// forwardOnCPU runs transformer blocks on the host (polymorphic RMSNorm / MHA / SwiGLU only).
// Generate and ForwardFull use this whenever WGPU is not serving the forward.
func (t *Transformer[T]) forwardOnCPU(input *Tensor[T]) *Tensor[T] {
	if t.hostWeightsReleased {
		fmt.Println("⚠️  CPU forward skipped (host weights released after GPU upload).")
		return NewTensor[T](input.Shape...)
	}
	current := input
	numBlocks := len(t.Network.Layers) / 4

	for b := 0; b < numBlocks; b++ {
		base := b * 4

		residual := current.Clone()

		lNorm1 := &t.Network.Layers[base+0]
		_, current = RMSNormForwardPolymorphic(lNorm1, current)

		lMHA := &t.Network.Layers[base+1]
		_, current = MHAForwardPolymorphic(lMHA, current)

		current.Add(residual)

		residual = current.Clone()

		lNorm2 := &t.Network.Layers[base+2]
		_, current = RMSNormForwardPolymorphic(lNorm2, current)

		lMLP := &t.Network.Layers[base+3]
		_, current = SwiGLUForwardPolymorphic(lMLP, current)

		current.Add(residual)
	}

	if t.finalNormLayer != nil {
		_, current = RMSNormForwardPolymorphic(t.finalNormLayer, current)
	}

	return current
}

// lastHiddenRow returns activations for the last sequence position [HiddenSize].
// Uses flat layout rows = len(Data)/HiddenSize (must divide evenly for correct LM head input).
func (t *Transformer[T]) lastHiddenRow(hidden *Tensor[T]) []T {
	if hidden == nil || len(hidden.Data) == 0 || t.HiddenSize <= 0 {
		return nil
	}
	h := t.HiddenSize
	n := len(hidden.Data)
	if n < h {
		return nil
	}
	if n%h != 0 {
		return hidden.Data[n-h : n]
	}
	rows := n / h
	return hidden.Data[(rows-1)*h : rows*h]
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
	return t.forwardOnCPU(input)
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

	return t.forwardOnCPU(input)
}

func (t *Transformer[T]) ApplyLMHead(hidden []T) []float32 {
	if len(hidden) != t.HiddenSize {
		out := make([]float32, t.VocabSize)
		for i := range out {
			out[i] = -math.MaxFloat32
		}
		return out
	}
	if len(t.LMHead) == 0 {
		out := make([]float32, t.VocabSize)
		for i := range out {
			out[i] = -math.MaxFloat32
		}
		return out
	}

	normalized := hidden

	logits := make([]float32, t.VocabSize)
	for v := 0; v < t.VocabSize; v++ {
		var sum float64
		offset := v * t.HiddenSize
		for d := 0; d < t.HiddenSize; d++ {
			sum += float64(normalized[d]) * float64(t.LMHead[offset+d])
		}
		logits[v] = float32(sum)
	}
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
