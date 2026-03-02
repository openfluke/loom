package tokenizer

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// Transformer coordinates high-level generation logic using the underlying NN
type Transformer struct {
	Network         *nn.Network
	Embeddings      []float32
	LMHead          []float32
	FinalNorm       []float32
	HiddenSize      int
	VocabSize       int
	Template        Template
	finalNormConfig *nn.LayerConfig
}

func NewTransformer(network *nn.Network, embeddings, lmHead, finalNorm []float32, template Template) *Transformer {
	hiddenSize := network.InputSize
	vocabSize := len(embeddings) / hiddenSize
	tr := &Transformer{
		Network:    network,
		Embeddings: embeddings,
		LMHead:     lmHead,
		FinalNorm:  finalNorm,
		HiddenSize: hiddenSize,
		VocabSize:  vocabSize,
		Template:   template,
	}
	if finalNorm != nil {
		tr.finalNormConfig = &nn.LayerConfig{
			Type:     nn.LayerRMSNorm,
			NormSize: hiddenSize,
			Gamma:    finalNorm,
			Epsilon:  1e-6,
		}
	}
	return tr
}

// Generate starts a generation loop and returns the final string
func (t *Transformer) Generate(tk *Tokenizer, turns []Turn, systemPrompt, userMsg string, opts GenOptions) string {
	prompt := t.Template.BuildPrompt(turns, systemPrompt, userMsg)
	inputIDs := tk.Encode(prompt, false)

	tokens := inputIDs
	stream := NewStreamer(tk, tokens)

	// 1. Prefill (Batched for GPU efficiency)
	var hidden []float32
	if len(tokens) > 0 {
		// Get all embeddings at once
		allEmbeds := make([]float32, len(tokens)*t.HiddenSize)
		for i, tok := range tokens {
			copy(allEmbeds[i*t.HiddenSize:], t.getEmbedding(int(tok)))
		}
		// Forward entire sequence (pos=0 for prefill batch)
		// Our updated GPU backend will handle the sequence length correctly via UpdateParams
		out, _ := t.Network.ForwardTransformer(allEmbeds, 0)
		// We only need the last hidden state for the next generation step
		if len(out) >= t.HiddenSize {
			hidden = append([]float32(nil), out[len(out)-t.HiddenSize:]...)
		} else {
			hidden = out
		}
	}

	// 2. Generate
	start := time.Now()
	generatedCount := 0
	var lmHeadTime time.Duration
	var sampleTime time.Duration
	var decodeForwardTime time.Duration

	for i := 0; i < opts.MaxTokens; i++ {
		// Get logits from last hidden state
		lmStart := time.Now()
		logits := t.applyLMHead(hidden)
		t.applyRepetitionPenalty(logits, tokens, opts)
		lmHeadTime += time.Since(lmStart)

		sampleStart := time.Now()
		nextToken := SampleTopK(logits, opts.TopK, opts.Temperature, opts.Deterministic)
		sampleTime += time.Since(sampleStart)
		tokens = append(tokens, uint32(nextToken))
		generatedCount++

		stream.Push(tokens)

		if t.isEOS(nextToken, opts.EOSTokens) || stream.HasNewUserTurn(tokens) {
			break
		}

		// Forward next token
		fwdStart := time.Now()
		if opts.UseKVCache {
			// Fast path: single-token decode with cached KV states on GPU.
			input := t.getEmbedding(nextToken)
			hidden, _ = t.Network.ForwardTransformer(input, len(tokens)-1)
		} else {
			// Fallback path: recompute from full context each step.
			allEmbeds := make([]float32, len(tokens)*t.HiddenSize)
			for j, tok := range tokens {
				copy(allEmbeds[j*t.HiddenSize:], t.getEmbedding(int(tok)))
			}
			out, _ := t.Network.ForwardTransformer(allEmbeds, 0)
			if len(out) >= t.HiddenSize {
				hidden = append(hidden[:0], out[len(out)-t.HiddenSize:]...)
			} else {
				hidden = append(hidden[:0], out...)
			}
		}
		decodeForwardTime += time.Since(fwdStart)
	}

	elapsed := time.Since(start)
	if generatedCount > 0 {
		fmt.Printf("\n\n(%.2f tokens/s, %d tokens total)\n", float64(generatedCount)/elapsed.Seconds(), generatedCount)
		fmt.Printf("timing: forward=%.1fms/token lm_head=%.1fms/token sample=%.1fms/token\n",
			float64(decodeForwardTime)/float64(time.Millisecond)/float64(generatedCount),
			float64(lmHeadTime)/float64(time.Millisecond)/float64(generatedCount),
			float64(sampleTime)/float64(time.Millisecond)/float64(generatedCount),
		)
	}

	return stream.String()
}

func (t *Transformer) getEmbedding(tokenID int) []float32 {
	offset := tokenID * t.HiddenSize
	if offset+t.HiddenSize > len(t.Embeddings) {
		return make([]float32, t.HiddenSize)
	}
	// Return a view into embedding storage to avoid per-token allocation/copy.
	return t.Embeddings[offset : offset+t.HiddenSize]
}

func (t *Transformer) applyLMHead(hidden []float32) []float32 {
	normalized := hidden
	if t.finalNormConfig != nil {
		normalized = nn.RmsNormForwardCPU(hidden, nil, t.finalNormConfig, 1)
	}

	logits := make([]float32, t.VocabSize)
	workers := runtime.GOMAXPROCS(0)
	if workers > t.VocabSize {
		workers = t.VocabSize
	}
	if workers <= 1 || t.VocabSize < 4096 {
		for v := 0; v < t.VocabSize; v++ {
			var sum float32
			offset := v * t.HiddenSize
			for d := 0; d < t.HiddenSize; d++ {
				sum += normalized[d] * t.LMHead[offset+d]
			}
			logits[v] = sum
		}
		return logits
	}

	chunk := (t.VocabSize + workers - 1) / workers
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		start := w * chunk
		end := start + chunk
		if end > t.VocabSize {
			end = t.VocabSize
		}
		go func(s, e int) {
			defer wg.Done()
			for v := s; v < e; v++ {
				var sum float32
				offset := v * t.HiddenSize
				for d := 0; d < t.HiddenSize; d++ {
					sum += normalized[d] * t.LMHead[offset+d]
				}
				logits[v] = sum
			}
		}(start, end)
	}
	wg.Wait()
	return logits
}

func (t *Transformer) applyRepetitionPenalty(logits []float32, tokens []uint32, opts GenOptions) {
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

func (t *Transformer) isEOS(token int, eosTokens []int) bool {
	for _, eos := range eosTokens {
		if token == eos {
			return true
		}
	}
	return false
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
