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

	// ForwardMode controls CPU decoder execution (normal / stepped / queued).
	ForwardMode TransformerForwardMode

	// ForwardStepDebug prints or callbacks each sub-layer step in stepped/queued modes.
	ForwardStepDebug bool

	forwardStepCb func(step, total int, label string)

	// cpuFQ holds partial forward state when ForwardMode == TransformerForwardQueuedCPU.
	cpuFQ *cpuForwardQueueState[T]

	// QueueTickPause is called after each CPUForwardQueueTick when draining the queue
	// (e.g. Lucy waits for Enter between sub-layers). Nil = no pause.
	QueueTickPause func(step, total int, label string)

	// PipelineTickPause is called after each PipelineTick (macro wavefront clock). Nil = no pause.
	PipelineTickPause func(tick, total int, summary string)

	// pipe holds macro pipeline state when ForwardMode == TransformerForwardPipelineCPU.
	pipe *decoderPipelineState[T]

	// pipeStatsCur accumulates during forwardCPUHiddenPipeline; copied to lastPipelineStats at end.
	pipeStatsCur      PipelineForwardStats
	lastPipelineStats PipelineForwardStats

	// Internal RMSNorm config for final norm if needed
	finalNormLayer *VolumetricLayer

	// hostWeightsReleased is set after ReleaseInferenceHostWeights; CPU fallback paths are unsafe.
	hostWeightsReleased bool

	lmHeadPackedTernary *BitNetTernaryMatrix
	lmHeadPackedLen     int

	// lmHeadTied: LM head shares embedding table (entity LMHeadTied or slice alias).
	lmHeadTied bool

	// layerTrace is non-nil during Generate when GenOptions.LayerTrace is set.
	layerTrace *layerTraceState
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
	t.PipelineReset()
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

func (t *Transformer[T]) lmHeadTiedToEmbeddings() bool {
	return slicesShareBackingStoreFloat32(t.LMHead, t.Embeddings)
}

// SyncEmbeddingsToGPU uploads token embeddings to VRAM.
func (t *Transformer[T]) SyncEmbeddingsToGPU() error {
	if !t.Network.UseGPU || t.Network.GPUContext == nil {
		return fmt.Errorf("GPU not enabled")
	}
	if t.Embeddings == nil || t.Network.GPUEmbeddings != nil {
		return nil
	}
	buf, err := t.Network.GPUContext.CreatePersistentBuffer(t.Embeddings, "Embeddings")
	if err != nil {
		return err
	}
	t.Network.GPUEmbeddings = buf
	return nil
}

// ReleaseEmbeddingsHost drops CPU embeddings after GPUEmbeddings exists.
// When LM head weights are tied, both slices are cleared together.
func (t *Transformer[T]) ReleaseEmbeddingsHost() {
	if t.Network.GPUEmbeddings == nil {
		return
	}
	if t.lmHeadTiedToEmbeddings() {
		t.Embeddings = nil
		t.LMHead = nil
		return
	}
	t.Embeddings = nil
}

// SyncLMHeadToGPU uploads the LM head (or aliases tied embeddings).
func (t *Transformer[T]) SyncLMHeadToGPU() error {
	if !t.Network.UseGPU || t.Network.GPUContext == nil {
		return fmt.Errorf("GPU not enabled")
	}
	if t.Network.GPULMHead != nil {
		return nil
	}
	if t.lmHeadTiedToEmbeddings() || (t.lmHeadTied && t.Network.GPUEmbeddings != nil) {
		if t.Network.GPUEmbeddings == nil {
			return fmt.Errorf("LM head tied to embeddings but GPUEmbeddings is nil")
		}
		t.Network.GPULMHead = t.Network.GPUEmbeddings
		return nil
	}
	if t.LMHead == nil {
		return nil
	}
	buf, err := t.Network.GPUContext.CreatePersistentBuffer(t.LMHead, "LMHead")
	if err != nil {
		return err
	}
	t.Network.GPULMHead = buf
	return nil
}

// ReleaseLMHeadHost drops CPU LM head weights when a distinct GPULMHead buffer exists.
func (t *Transformer[T]) ReleaseLMHeadHost() {
	if t.Network.GPULMHead == nil || t.lmHeadTiedToEmbeddings() {
		return
	}
	t.LMHead = nil
}

// SyncFinalNormToGPU uploads the transformer's final RMSNorm weights.
func (t *Transformer[T]) SyncFinalNormToGPU() error {
	if t.finalNormLayer == nil {
		return nil
	}
	return t.finalNormLayer.SyncToGPU()
}

// ReleaseFinalNormHost drops CPU final-norm weights after GPU sync.
func (t *Transformer[T]) ReleaseFinalNormHost() {
	t.FinalNorm = nil
	if t.finalNormLayer != nil {
		t.finalNormLayer.ReleaseInferenceHostWeights()
	}
}

// SyncGlobalWeightsToGPUSequential uploads embeddings, LM head, and final norm one at a
// time, releasing each CPU copy immediately after its GPU buffer is created.
func (t *Transformer[T]) SyncGlobalWeightsToGPUSequential() error {
	if !t.Network.UseGPU || t.Network.GPUContext == nil {
		return fmt.Errorf("GPU not enabled")
	}
	t.Network.GPUContext.ResetCache()
	if err := t.SyncEmbeddingsToGPU(); err != nil {
		return fmt.Errorf("embeddings: %w", err)
	}
	if err := t.SyncLMHeadToGPU(); err != nil {
		return fmt.Errorf("lm head: %w", err)
	}
	t.ReleaseEmbeddingsHost()
	t.ReleaseLMHeadHost()
	ReleaseInferenceTransientMemory()
	if err := t.SyncFinalNormToGPU(); err != nil {
		return fmt.Errorf("final norm: %w", err)
	}
	t.ReleaseFinalNormHost()
	ReleaseInferenceTransientMemory()
	return nil
}

func (t *Transformer[T]) SyncToGPU() error {
	if !t.Network.UseGPU || t.Network.GPUContext == nil {
		return fmt.Errorf("GPU not enabled")
	}
	t.Network.GPUContext.ResetCache()

	if err := t.SyncEmbeddingsToGPU(); err != nil {
		return err
	}
	if err := t.SyncLMHeadToGPU(); err != nil {
		return err
	}
	if err := t.SyncFinalNormToGPU(); err != nil {
		return err
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
	// RAMUsageMB is host-side Poly model tensors (same as ModelRAMUsageMB).
	// It is not Go runtime MemStats.Sys and not OS RSS.
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
	traceOpts := opts
	if opts.LayerTrace && opts.LayerTraceMaxTokens > 0 {
		traceOpts.MaxTokens = opts.LayerTraceMaxTokens
		if traceOpts.MinTokens > traceOpts.MaxTokens {
			traceOpts.MinTokens = traceOpts.MaxTokens
		}
	}
	t.beginLayerTrace(traceOpts)
	defer t.endLayerTrace()
	tracePrefill := traceOpts.LayerTrace && traceOpts.LayerTracePrefill
	traceDecode := traceOpts.LayerTrace

	prompt := t.Template.BuildPrompt(turns, systemPrompt, userMsg)
	inputIDs := encode(prompt)

	tokens := inputIDs
	stream := NewStreamer(decode, tokens)
	t.Reset()

	// 1. Prefill
	prefillStart := time.Now()
	var logits []float32
	if len(tokens) > 0 {
		if traceOpts.LayerTrace && !traceOpts.LayerTracePrefill {
			t.printLayerTraceBanner("prefill-skip", len(tokens), traceOpts.MaxTokens)
		}
		if tracePrefill {
			t.setLayerTraceRecording(true)
			t.printLayerTraceBanner("prefill", len(tokens), traceOpts.MaxTokens)
		} else {
			t.setLayerTraceRecording(false)
		}
		useGPUPrefill := !tracePrefill && !t.forwardModeSkipsGPU() && t.Network.UseGPU && t.Network.GPUEmbeddings != nil
		if useGPUPrefill {
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
				if tracePrefill {
					t.layerTraceSetTokenOrdinal(-1)
				}
				hidden := t.ForwardFull(allEmbeds)
				logits = t.ApplyLMHead(t.lastHiddenRow(hidden))
			}
		} else {
			allEmbeds := t.TokensToTensor(tokens)
			if tracePrefill {
				t.layerTraceSetTokenOrdinal(-1)
			}
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

	if traceDecode {
		t.setLayerTraceRecording(true)
		t.printLayerTraceBanner("decode", len(inputIDs), traceOpts.MaxTokens)
	}

	for i := 0; i < traceOpts.MaxTokens; i++ {
		t.applyRepetitionPenalty(logits, tokens, traceOpts)
		t.applyBannedTokenMask(logits, traceOpts)
		t.applyConsecutiveRepeatMask(logits, tokens, traceOpts)
		t.applyNoRepeatNGramMask(logits, tokens, traceOpts)

		nextToken := SampleTopK(logits, traceOpts.TopK, traceOpts.Temperature, traceOpts.Deterministic)

		tokens = append(tokens, uint32(nextToken))
		stream.Push(tokens, traceOpts.Silent, traceOpts.StreamCallback)
		generatedCount++

		if traceDecode {
			fmt.Printf("→ sampled token id=%d\n", nextToken)
		}

		if (generatedCount >= traceOpts.MinTokens && t.isEOS(nextToken, traceOpts.EOSTokens)) || (traceOpts.UseKVCache && stream.HasNewUserTurn(tokens)) {
			break
		}

		// Forward next token (Incremental)
		useGPUDecode := !traceDecode && !t.forwardModeSkipsGPU() && t.Network.UseGPU && t.Network.GPUEmbeddings != nil
		if useGPUDecode {
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
				t.layerTraceSetTokenOrdinal(generatedCount - 1)
				hidden := t.forwardOne(input)
				logits = t.ApplyLMHead(t.lastHiddenRow(hidden))
			}
		} else {
			nextEmbed := t.getEmbedding(nextToken)
			input := NewTensor[T](1, t.HiddenSize)
			copy(input.Data, nextEmbed)
			t.layerTraceSetTokenOrdinal(generatedCount - 1)
			hidden := t.forwardOne(input)
			logits = t.ApplyLMHead(t.lastHiddenRow(hidden))
		}
	}

	if traceDecode {
		n := len(t.LayerTraceRecords())
		perTok := t.cpuLayerTraceStepTotal()
		fmt.Printf("📼 Layer trace complete: %d recorded sub-layer steps", n)
		if generatedCount > 0 && n > 0 {
			fmt.Printf(" (~%d per decode token)", perTok)
		}
		fmt.Println()
	}

	decodeElapsed := time.Since(decodeStart)
	metrics.DecodeTime = decodeElapsed
	metrics.GeneratedTokens = generatedCount

	modelRAMBytes := t.calculateHostModelBytes()
	vramBytes := t.Network.GetVRAMUsage()
	hostModelMB := float64(modelRAMBytes) / (1024 * 1024)
	metrics.ModelRAMUsageMB = hostModelMB
	metrics.RAMUsageMB = hostModelMB
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
	if t.layerTraceRecording() {
		return t.forwardOnCPUTraced(input)
	}
	return t.forwardCPUHidden(input)
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
	if !t.layerTraceRecording() && !t.forwardModeSkipsGPU() && t.Network.UseGPU && t.Network.GPUContext != nil {
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
	if !t.layerTraceRecording() && !t.forwardModeSkipsGPU() && t.Network.UseGPU && t.Network.GPUContext != nil {
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

	if t.usePackedTernaryLMHead() {
		if logits := t.applyPackedTernaryLMHead(hidden); logits != nil {
			return logits
		}
	}

	normalized := hidden

	logits := make([]float32, t.VocabSize)
	t.applyFP32LMHeadRows(normalized, logits)
	return logits
}

func (t *Transformer[T]) applyFP32LMHeadRows(hidden []T, logits []float32) {
	worker := func(start, end int) {
		for v := start; v < end; v++ {
			var sum float64
			offset := v * t.HiddenSize
			for d := 0; d < t.HiddenSize; d++ {
				sum += float64(hidden[d]) * float64(t.LMHead[offset+d])
			}
			logits[v] = float32(sum)
		}
	}

	work := t.VocabSize * t.HiddenSize
	if work < 262144 || t.VocabSize < 4 {
		worker(0, t.VocabSize)
		return
	}
	workers := runtime.GOMAXPROCS(0)
	if workers > t.VocabSize {
		workers = t.VocabSize
	}
	if workers <= 1 {
		worker(0, t.VocabSize)
		return
	}
	chunk := (t.VocabSize + workers - 1) / workers
	var wg sync.WaitGroup
	for start := 0; start < t.VocabSize; start += chunk {
		end := start + chunk
		if end > t.VocabSize {
			end = t.VocabSize
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			worker(start, end)
		}(start, end)
	}
	wg.Wait()
}

func (t *Transformer[T]) usePackedTernaryLMHead() bool {
	if t == nil || t.Network == nil || !t.Network.UseExactDType || t.Network.UseGPU {
		return false
	}
	if slicesShareBackingStoreFloat32(t.LMHead, t.Embeddings) {
		return false
	}
	for i := range t.Network.Layers {
		if t.Network.Layers[i].DType == DTypeTernary {
			return true
		}
	}
	return false
}

func (t *Transformer[T]) applyPackedTernaryLMHead(hidden []T) []float32 {
	if t.lmHeadPackedTernary == nil || t.lmHeadPackedLen != len(t.LMHead) ||
		t.lmHeadPackedTernary.Rows != t.VocabSize || t.lmHeadPackedTernary.Cols != t.HiddenSize {
		packed, ok := packFloat32AsBitNetTernaryMatrix(t.LMHead, t.VocabSize, t.HiddenSize)
		if !ok {
			return nil
		}
		t.lmHeadPackedTernary = packed
		t.lmHeadPackedLen = len(t.LMHead)
	}
	logits64 := make([]float64, t.VocabSize)
	if !bitNetTernaryMatVecNumeric(t.lmHeadPackedTernary, hidden, logits64) {
		return nil
	}
	logits := make([]float32, t.VocabSize)
	for i, v := range logits64 {
		logits[i] = float32(v)
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
