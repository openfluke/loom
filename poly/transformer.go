package poly

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"time"

	"github.com/openfluke/loom/poly/simd"
	"github.com/openfluke/webgpu/wgpu"
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

	// Q4 LM head for CPU logits (UsePackedQ4CPU). Untied FP32 LMHead may be nil after pack.
	lmHeadQ4Scales  []float32
	lmHeadQ4Packed  []uint32
	lmHeadQ4Rows    int
	lmHeadQ4Cols    int
	lmHeadLogitsF32 []float32
	lmHeadLogitsF64 []float64

	// embScratch avoids per-token alloc in getEmbedding during decode.
	embScratch []T

	// lmHeadTied: LM head shares embedding table (entity LMHeadTied or slice alias).
	lmHeadTied bool

	// layerTrace is non-nil during Generate when GenOptions.LayerTrace is set.
	layerTrace *layerTraceState

	// gpuReturnGreedyToken / lastGPUSampledToken: on-device ArgMax (see ForwardSampleGreedyTokenWGPU).
	// gpuChunkRecording: many decode steps in one BeginFrame → one MapAsync per chunk.
	gpuReturnGreedyToken   bool
	lastGPUSampledToken    uint32
	gpuChunkRecording      bool
	gpuChunkHistCount      int
	gpuUseDecodeTokenBuf   bool // embed from GPU-resident decode_token buffer
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
	}
	// Do not wipe GPU KV buffers here. Prefill/decode rewrite slots from offset 0 and
	// MHA only attends to [0, KVOffset+seq). Full-cache zero (~tens of MB) every turn
	// was a multi-turn latency tax with no correctness benefit when KVOffset is reset.
}

// zeroGPUKVCaches clears GPU KV (debug / explicit wipe). Not used by Reset.
func (t *Transformer[T]) zeroGPUKVCaches() {
	if t == nil || t.Network == nil || !t.Network.UseGPU || t.Network.GPUContext == nil {
		return
	}
	ctx := t.Network.GPUContext
	for i := range t.Network.Layers {
		l := &t.Network.Layers[i]
		if l.Type != LayerMultiHeadAttention {
			continue
		}
		if k, ok := l.GPUKVCacheK.(*wgpu.Buffer); ok && k != nil {
			ctx.zeroWriteBuffer(k, k.GetSize())
		}
		if v, ok := l.GPUKVCacheV.(*wgpu.Buffer); ok && v != nil {
			ctx.zeroWriteBuffer(v, v.GetSize())
		}
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
	t.EnsurePackedQ4LMHead()
	t.EnsurePackedTernaryLMHead()
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
// Prefer baked/packed Q4 for decode GEMV when available. BitNet/ternary uploads a
// packed ternary logits matrix (~82 MB for 128k×2560) — same encoding as CPU SIMD —
// and keeps FP32 embeddings for gather (no Q4/FP32 vocab×hidden logits GEMM).
func (t *Transformer[T]) SyncLMHeadToGPU() error {
	if !t.Network.UseGPU || t.Network.GPUContext == nil {
		return fmt.Errorf("GPU not enabled")
	}
	if t.hasTernaryDecoderLayers() {
		if err := t.syncLMHeadTernaryToGPU(); err != nil {
			return err
		}
		// Alias tied emb as GPULMHead for any non-logits paths that check presence.
		if t.Network.GPULMHead == nil && t.Network.GPUEmbeddings != nil {
			t.Network.GPULMHead = t.Network.GPUEmbeddings
		}
		return nil
	}
	if err := t.syncLMHeadQ4ToGPU(); err != nil {
		return err
	}
	// Q4 logits path is enough when packed; still alias tied emb as GPULMHead for fallbacks.
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

// syncLMHeadTernaryToGPU packs vocab×hidden as BitNet words and uploads for decode GEMV.
func (t *Transformer[T]) syncLMHeadTernaryToGPU() error {
	if t.Network.GPULMHeadTernaryPacked != nil {
		return nil
	}
	if !t.ensurePackedTernaryLMHeadMatrix() {
		return fmt.Errorf("BitNet GPU LM head: failed to pack ternary matrix (%d×%d)", t.VocabSize, t.HiddenSize)
	}
	m := t.lmHeadPackedTernary
	ctx := t.Network.GPUContext
	upStart := time.Now()
	wBuf, err := ctx.CreatePersistentBufferUint32(m.Words, "LMHeadTernaryPacked")
	if err != nil {
		return err
	}
	t.Network.GPULMHeadTernaryPacked = wBuf
	t.Network.GPULMHeadTernaryScale = m.Scale
	mb := float64(len(m.Words)*4) / (1024 * 1024)
	fmt.Printf("🧮 GPU LM head: BitNet ternary (%d×%d, %.1f MB packed) uploaded in %s\n",
		t.VocabSize, t.HiddenSize, mb, time.Since(upStart).Round(time.Millisecond))
	return nil
}

func (t *Transformer[T]) hasTernaryDecoderLayers() bool {
	if t == nil || t.Network == nil {
		return false
	}
	for i := range t.Network.Layers {
		if t.Network.Layers[i].DType == DTypeTernary {
			return true
		}
	}
	return false
}

// syncLMHeadQ4ToGPU uploads vocab×hidden Q4 scales/packed for GPU DispatchDenseQ4 logits.
func (t *Transformer[T]) syncLMHeadQ4ToGPU() error {
	if t.Network.GPULMHeadQ4Packed != nil && t.Network.GPULMHeadQ4Scales != nil {
		return nil
	}
	if t.VocabSize <= 0 || t.HiddenSize <= 0 {
		return nil
	}
	if len(t.lmHeadQ4Packed) == 0 || len(t.lmHeadQ4Scales) == 0 ||
		t.lmHeadQ4Rows != t.VocabSize || t.lmHeadQ4Cols != t.HiddenSize {
		// Pack from FP32 head / embeddings if entity bake missing.
		src := t.LMHead
		if len(src) < t.VocabSize*t.HiddenSize {
			src = t.Embeddings
		}
		if len(src) < t.VocabSize*t.HiddenSize {
			return nil
		}
		// BitNet / huge untied heads: packing ~128256×2560 on CPU then CreateBufferInit
		// appears hung (multi‑GB memcpy). Prefer FP32 GPU head / tied embeddings.
		const maxPackElems = 64 * 1024 * 1024 // 64M floats ≈ skip 128k×2560 and bigger
		if len(src) > maxPackElems {
			fmt.Printf("⚠️  Skipping on-the-fly Q4 LM head pack (%d×%d too large); using FP32 GPU head\n",
				t.VocabSize, t.HiddenSize)
			return nil
		}
		fmt.Printf("🧮 GPU LM head: packing Q4 (%d×%d)...\n", t.VocabSize, t.HiddenSize)
		packStart := time.Now()
		scales, packed := PackQ4_0GPUParallel(src[:t.VocabSize*t.HiddenSize])
		if len(scales) == 0 || len(packed) == 0 {
			return nil
		}
		t.lmHeadQ4Scales = scales
		t.lmHeadQ4Packed = packed
		t.lmHeadQ4Rows = t.VocabSize
		t.lmHeadQ4Cols = t.HiddenSize
		fmt.Printf("🧮 GPU LM head: packed Q4 (%d×%d) in %s — uploading...\n",
			t.VocabSize, t.HiddenSize, time.Since(packStart).Round(time.Millisecond))
	}
	ctx := t.Network.GPUContext
	upStart := time.Now()
	sBuf, err := ctx.CreatePersistentBuffer(t.lmHeadQ4Scales, "LMHeadQ4Scales")
	if err != nil {
		return err
	}
	wBuf, err := ctx.CreatePersistentBufferUint32(t.lmHeadQ4Packed, "LMHeadQ4Packed")
	if err != nil {
		return err
	}
	t.Network.GPULMHeadQ4Scales = sBuf
	t.Network.GPULMHeadQ4Packed = wBuf
	fmt.Printf("🧮 GPU LM head: Q4 upload done in %s\n", time.Since(upStart).Round(time.Millisecond))
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

	hasGPULM := t.Network.GPULMHead != nil ||
		(t.Network.GPULMHeadQ4Packed != nil && t.Network.GPULMHeadQ4Scales != nil) ||
		t.Network.GPULMHeadTernaryPacked != nil
	useGPUSample := traceOpts.GPUSampleGreedy &&
		!tracePrefill && !traceDecode &&
		!t.forwardModeSkipsGPU() &&
		t.Network.UseGPU && t.Network.GPUEmbeddings != nil && hasGPULM

	// 1. Prefill
	prefillStart := time.Now()
	var logits []float32
	var nextGPUToken uint32
	var haveGPUToken bool

	if useGPUSample {
		if len(tokens) == 0 {
			return stream.String(), GenMetrics{}
		}
		tok, err := t.ForwardSampleGreedyTokenWGPU(tokens, true)
		if err != nil {
			fmt.Printf("⚠️  GPU greedy prefill failed: %v\n", err)
			return stream.String(), GenMetrics{
				PrefillTime:   time.Since(prefillStart),
				PrefillTokens: len(tokens),
			}
		}
		nextGPUToken = tok
		haveGPUToken = true
	} else if len(tokens) > 0 {
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

	if useGPUSample {
		// Prefill produced the first greedy token. Remaining decode runs in GPU chunks
		// (one BeginFrame + one MapAsync per chunk). Host anti-loop still applied on map-back
		// (RepetitionPenalty needs logits — pure greedy cannot apply it).
		if !haveGPUToken {
			// nothing
		} else {
			nextToken := int(nextGPUToken)
			if t.greedyHostRejects(tokens, len(inputIDs), nextToken, traceOpts) {
				// first token alone is banned / looping — stop immediately
			} else {
				tokens = append(tokens, uint32(nextToken))
				stream.Push(tokens, traceOpts.Silent, traceOpts.StreamCallback)
				generatedCount++
			}
			remain := traceOpts.MaxTokens - generatedCount
			// Smaller chunks when host loop guards are active so we stop sooner.
			chunkSize := 32
			if traceOpts.NoRepeatNGram > 0 || traceOpts.MaxConsecutiveRepeats > 0 {
				chunkSize = 8
			}
			for remain > 0 && generatedCount > 0 {
				if generatedCount >= traceOpts.MinTokens && t.isEOS(nextToken, traceOpts.EOSTokens) {
					break
				}
				if traceOpts.UseKVCache && stream.HasNewUserTurn(tokens) {
					break
				}
				n := remain
				if n > chunkSize {
					n = chunkSize
				}
				chunkStartPos := 0
				if len(t.Network.Layers) > 1 {
					chunkStartPos = t.Network.Layers[1].KVOffset
				}
				chunk, err := t.ForwardSampleGreedyChunkWGPU(uint32(nextToken), n)
				if err != nil {
					fmt.Printf("⚠️  GPU greedy chunk decode failed: %v\n", err)
					break
				}
				stop := false
				used := 0
				for _, tok := range chunk {
					cand := int(tok)
					if t.greedyHostRejects(tokens, len(inputIDs), cand, traceOpts) {
						stop = true
						break
					}
					nextToken = cand
					tokens = append(tokens, tok)
					stream.Push(tokens, traceOpts.Silent, traceOpts.StreamCallback)
					generatedCount++
					remain--
					used++
					if ReplyLooksDegenerate(stream.String()) {
						if !traceOpts.Silent {
							fmt.Print("\n…(stopped: degenerate output)\n")
						}
						stop = true
						break
					}
					if (generatedCount >= traceOpts.MinTokens && t.isEOS(nextToken, traceOpts.EOSTokens)) ||
						(traceOpts.UseKVCache && stream.HasNewUserTurn(tokens)) {
						stop = true
						break
					}
				}
				// Chunk advanced GPU KV by n; host must match tokens we actually kept.
				if used < n {
					for b := 0; b < len(t.Network.Layers)/4; b++ {
						t.Network.Layers[b*4+1].KVOffset = chunkStartPos + used
					}
				}
				if stop || len(chunk) == 0 || used == 0 {
					break
				}
			}
		}
	} else {
		promptLen := len(inputIDs)
		for i := 0; i < traceOpts.MaxTokens; i++ {
			t.applyRepetitionPenalty(logits, tokens, traceOpts)
			t.applyBannedTokenMask(logits, traceOpts)
			t.applyConsecutiveRepeatMask(logits, tokens, promptLen, traceOpts)
			t.applyNoRepeatNGramMask(logits, tokens, promptLen, traceOpts)

			if logitsHaveNoFiniteCandidate(logits) {
				// Masks wiped every real option (common once chat history poisons n-grams).
				break
			}

			nextToken := SampleTopK(logits, traceOpts.TopK, traceOpts.Temperature, traceOpts.Deterministic)

			tokens = append(tokens, uint32(nextToken))
			stream.Push(tokens, traceOpts.Silent, traceOpts.StreamCallback)
			generatedCount++

			if traceDecode {
				fmt.Printf("→ sampled token id=%d\n", nextToken)
			}

			if ReplyLooksDegenerate(stream.String()) {
				if !traceOpts.Silent {
					fmt.Print("\n…(stopped: degenerate output)\n")
				}
				break
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
		// GPUSampleGreedy folds TTFT (first generated token) into PrefillTime; decode rate
		// should count only subsequent token steps so it is not artificially inflated.
		decodeTokCount := generatedCount
		if useGPUSample && generatedCount > 1 {
			decodeTokCount = generatedCount - 1
		}
		if decodeElapsed > 0 && decodeTokCount > 0 {
			metrics.DecodeTokPerSec = float64(decodeTokCount) / decodeElapsed.Seconds()
		}
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

	return SanitizeChatReply(stream.String()), metrics
}

func (t *Transformer[T]) calculateHostModelBytes() uint64 {
	var total uint64
	total += uint64(t.Network.CalculateTotalMemory())
	total += uint64(len(t.Embeddings)) * 4
	if t.usePackedQ4LMHead() {
		total += uint64(len(t.lmHeadQ4Scales))*4 + uint64(len(t.lmHeadQ4Packed))*4
	} else if !slicesShareBackingStoreFloat32(t.LMHead, t.Embeddings) {
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
	if len(t.embScratch) != t.HiddenSize {
		t.embScratch = make([]T, t.HiddenSize)
	}
	out := t.embScratch
	emb := t.Embeddings[offset : offset+t.HiddenSize]
	for i := 0; i < t.HiddenSize; i++ {
		out[i] = T(emb[i])
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

	if t.usePackedQ4LMHead() {
		if logits := t.applyPackedQ4LMHead(hidden); logits != nil {
			return logits
		}
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
	// FP32 LM head (used when the head is tied to FP32 embeddings, so the packed
	// ternary path does not apply — e.g. BitNet weight tying). At 128256×2560 this
	// dominates decode, so route it through the NEON/AVX2 DotTile kernel when the
	// network's SIMD forward is enabled. dotTileGo is the same sequential float64
	// loop as the scalar path below, so SIMD-off is unchanged.
	var simdHidden []float32
	if t.Network != nil && t.Network.UseSimdForward && simd.SimdEnabled() {
		if hf, ok := any(hidden).([]float32); ok {
			simdHidden = hf
		}
	}

	worker := func(start, end int) {
		for v := start; v < end; v++ {
			offset := v * t.HiddenSize
			if simdHidden != nil {
				logits[v] = float32(simd.DotTile(simdHidden, t.LMHead[offset:offset+t.HiddenSize], 0, t.HiddenSize, 0))
				continue
			}
			var sum float64
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

func (t *Transformer[T]) wantsPackedTernaryLMHead() bool {
	if t == nil || t.Network == nil || !t.Network.UseExactDType {
		return false
	}
	// Tied emb/LM head is OK — we pack a separate ternary logits matrix (keep FP32 emb for gather).
	return t.hasTernaryDecoderLayers()
}

// usePackedTernaryLMHead is the CPU ApplyLMHead path (GPU uses GPULMHeadTernaryPacked instead).
func (t *Transformer[T]) usePackedTernaryLMHead() bool {
	return t.wantsPackedTernaryLMHead() && !t.Network.UseGPU
}

// ensurePackedTernaryLMHeadMatrix packs vocab×hidden once (CPU or GPU upload prep).
func (t *Transformer[T]) ensurePackedTernaryLMHeadMatrix() bool {
	if !t.wantsPackedTernaryLMHead() {
		return false
	}
	if t.lmHeadPackedTernary != nil && t.lmHeadPackedTernary.Rows == t.VocabSize &&
		t.lmHeadPackedTernary.Cols == t.HiddenSize && len(t.lmHeadPackedTernary.Words) > 0 {
		return true
	}
	src := t.LMHead
	if len(src) < t.VocabSize*t.HiddenSize {
		src = t.Embeddings
	}
	if len(src) < t.VocabSize*t.HiddenSize {
		return false
	}
	start := time.Now()
	packed, ok := packFloat32AsBitNetTernaryMatrix(src[:t.VocabSize*t.HiddenSize], t.VocabSize, t.HiddenSize)
	if !ok {
		return false
	}
	t.lmHeadPackedTernary = packed
	t.lmHeadPackedLen = len(src)
	fmt.Printf("🧮 BitNet LM head → packed ternary for logits (%d×%d in %s)\n",
		t.VocabSize, t.HiddenSize, time.Since(start).Round(time.Millisecond))
	return true
}

// EnsurePackedTernaryLMHead builds the BitNet logits matvec once at CPU sync (including tied heads).
func (t *Transformer[T]) EnsurePackedTernaryLMHead() {
	if !t.usePackedTernaryLMHead() {
		return
	}
	t.ensurePackedTernaryLMHeadMatrix()
}

func (t *Transformer[T]) applyPackedTernaryLMHead(hidden []T) []float32 {
	if t.lmHeadPackedTernary == nil || t.lmHeadPackedTernary.Rows != t.VocabSize ||
		t.lmHeadPackedTernary.Cols != t.HiddenSize {
		src := t.LMHead
		if len(src) < t.VocabSize*t.HiddenSize {
			src = t.Embeddings
		}
		if len(src) < t.VocabSize*t.HiddenSize {
			return nil
		}
		packed, ok := packFloat32AsBitNetTernaryMatrix(src[:t.VocabSize*t.HiddenSize], t.VocabSize, t.HiddenSize)
		if !ok {
			return nil
		}
		t.lmHeadPackedTernary = packed
		t.lmHeadPackedLen = len(src)
	}
	logits64 := t.lmHeadLogitsF64
	if len(logits64) < t.VocabSize {
		logits64 = make([]float64, t.VocabSize)
		t.lmHeadLogitsF64 = logits64
	}
	if !bitNetTernaryMatVecNumeric(t.lmHeadPackedTernary, hidden, logits64[:t.VocabSize]) {
		return nil
	}
	out := t.lmHeadLogitsF32
	if len(out) < t.VocabSize {
		out = make([]float32, t.VocabSize)
		t.lmHeadLogitsF32 = out
	}
	for i := 0; i < t.VocabSize; i++ {
		out[i] = float32(logits64[i])
	}
	return out[:t.VocabSize]
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

// greedyHostRejects reports whether nextToken must not be emitted under host loop/ban rules.
// Used when GPUSampleGreedy skips logit masks (cannot re-sample without logits).
// promptLen scopes n-gram / consecutive checks to the current reply (not the chat prompt).
func (t *Transformer[T]) greedyHostRejects(tokens []uint32, promptLen, nextToken int, opts GenOptions) bool {
	if nextToken < 0 {
		return true
	}
	for _, b := range opts.BannedTokens {
		if nextToken == b {
			return true
		}
	}
	gen := tokens
	if promptLen > 0 && promptLen <= len(tokens) {
		gen = tokens[promptLen:]
	}
	if opts.MaxConsecutiveRepeats > 0 && len(gen) >= opts.MaxConsecutiveRepeats {
		last := uint32(nextToken)
		if gen[len(gen)-1] == last {
			repeats := 1 // the candidate itself
			for i := len(gen) - 1; i >= 0; i-- {
				if gen[i] != last {
					break
				}
				repeats++
			}
			if repeats > opts.MaxConsecutiveRepeats {
				return true
			}
		}
	}
	n := opts.NoRepeatNGram
	if n > 1 && len(gen) >= n-1 {
		// Would completing (prefix..., next) recreate an n-gram already in this reply?
		prefix := gen[len(gen)-(n-1):]
		for i := 0; i+n-1 < len(gen); i++ {
			match := true
			for j := 0; j < n-1; j++ {
				if gen[i+j] != prefix[j] {
					match = false
					break
				}
			}
			if match && int(gen[i+n-1]) == nextToken {
				return true
			}
		}
	}
	return false
}

func (t *Transformer[T]) applyConsecutiveRepeatMask(logits []float32, tokens []uint32, promptLen int, opts GenOptions) {
	if len(logits) == 0 || opts.MaxConsecutiveRepeats <= 0 {
		return
	}
	gen := tokens
	if promptLen > 0 && promptLen <= len(tokens) {
		gen = tokens[promptLen:]
	}
	if len(gen) < opts.MaxConsecutiveRepeats {
		return
	}
	last := gen[len(gen)-1]
	repeats := 1
	for i := len(gen) - 2; i >= 0; i-- {
		if gen[i] != last {
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

// applyNoRepeatNGramMask bans continuations that would recreate an n-gram already
// present in the *current reply* (tokens after promptLen). Scoping to the reply
// avoids locking common phrases from system/user history as the chat grows —
// that path previously forced TopK into obscure BPE junk on long conversations.
func (t *Transformer[T]) applyNoRepeatNGramMask(logits []float32, tokens []uint32, promptLen int, opts GenOptions) {
	n := opts.NoRepeatNGram
	if n <= 1 || len(logits) == 0 {
		return
	}
	gen := tokens
	if promptLen > 0 && promptLen <= len(tokens) {
		gen = tokens[promptLen:]
	}
	if len(gen) < n-1 {
		return
	}

	prefixStart := len(gen) - (n - 1)
	prefix := gen[prefixStart:]

	for i := 0; i+n-1 < len(gen); i++ {
		match := true
		for j := 0; j < n-1; j++ {
			if gen[i+j] != prefix[j] {
				match = false
				break
			}
		}
		if !match {
			continue
		}
		next := gen[i+n-1]
		if int(next) >= 0 && int(next) < len(logits) {
			logits[next] = -math.MaxFloat32
		}
	}
}

// logitsHaveNoFiniteCandidate is true when every logit is at/below the ban floor
// (masks erased the distribution — sampling would pick arbitrary -Inf ids).
func logitsHaveNoFiniteCandidate(logits []float32) bool {
	if len(logits) == 0 {
		return true
	}
	const dead = -1e30
	for _, v := range logits {
		if v > dead {
			return false
		}
	}
	return true
}
