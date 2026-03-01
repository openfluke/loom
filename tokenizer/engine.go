package tokenizer

import (
	"fmt"
	"time"

	"github.com/openfluke/loom/nn"
)

// GenOptions holds parameters for model generation
type GenOptions struct {
	MaxTokens         int
	Temperature       float32
	TopK              int
	Deterministic     bool
	UseKVCache        bool
	RepetitionPenalty float32
	RepetitionWindow  int
	EOSTokens         []int
}

// LLMEngine encapsulates a transformer network and its associated weights for inference
type LLMEngine struct {
	Network      *nn.Network
	Embeddings   []float32
	LMHead       []float32
	FinalNorm    []float32
	HasFinalNorm bool
	HiddenSize   int
	VocabSize    int
	Template     Template
}

// NewLLMEngine creates a new inference engine
func NewLLMEngine(network *nn.Network, embeddings, lmHead, finalNorm []float32, template Template) *LLMEngine {
	hiddenSize := network.InputSize
	vocabSize := len(embeddings) / hiddenSize

	return &LLMEngine{
		Network:      network,
		Embeddings:   embeddings,
		LMHead:       lmHead,
		FinalNorm:    finalNorm,
		HasFinalNorm: finalNorm != nil,
		HiddenSize:   hiddenSize,
		VocabSize:    vocabSize,
		Template:     template,
	}
}

// Generate starts a generation loop and returns the final string
func (e *LLMEngine) Generate(tk *Tokenizer, turns []Turn, systemPrompt, userMsg string, opts GenOptions) string {
	prompt := e.Template.BuildPrompt(turns, systemPrompt, userMsg)
	inputIDs := tk.Encode(prompt, false)

	if opts.UseKVCache {
		return e.generateWithKV(tk, inputIDs, opts)
	}

	tokens := inputIDs
	start := time.Now()
	generatedCount := 0
	stream := NewStreamer(tk, tokens)

	for i := 0; i < opts.MaxTokens; i++ {
		nextToken, err := e.generateNextToken(tokens, opts)
		if err != nil {
			return fmt.Sprintf("\n[Error: %v]", err)
		}

		tokens = append(tokens, uint32(nextToken))
		generatedCount++

		stream.Push(tokens)

		if e.isEOS(nextToken, opts.EOSTokens) || stream.HasNewUserTurn(tokens) {
			break
		}
	}

	elapsed := time.Since(start)
	if generatedCount > 0 {
		fmt.Printf("\n\n(%.2f tokens/s, %d tokens total)\n", float64(generatedCount)/elapsed.Seconds(), generatedCount)
	}
	return stream.String()
}

func (e *LLMEngine) generateWithKV(tk *Tokenizer, tokens []uint32, opts GenOptions) string {
	maxSeq := 0
	for _, l := range e.Network.Layers {
		if l.Type == nn.LayerMultiHeadAttention {
			if l.SeqLength > maxSeq {
				maxSeq = l.SeqLength
			}
		}
	}
	if maxSeq < len(tokens)+opts.MaxTokens {
		maxSeq = len(tokens) + opts.MaxTokens
	}

	state := InitKVCacheState(e.Network, maxSeq)
	var err error
	var hidden []float32

	// Prefill
	for i, tok := range tokens {
		hidden, err = ForwardTokenKV(e.Network, int(tok), i, state, e.Embeddings, e.HiddenSize)
		if err != nil {
			return fmt.Sprintf("\n[Error: %v]", err)
		}
	}

	start := time.Now()
	generatedCount := 0
	stream := NewStreamer(tk, tokens)

	for i := 0; i < opts.MaxTokens; i++ {
		nextToken, err := e.nextTokenFromHidden(hidden, tokens, opts)
		if err != nil {
			return fmt.Sprintf("\n[Error: %v]", err)
		}

		tokens = append(tokens, uint32(nextToken))
		generatedCount++

		stream.Push(tokens)

		if e.isEOS(nextToken, opts.EOSTokens) || stream.HasNewUserTurn(tokens) {
			break
		}

		hidden, err = ForwardTokenKV(e.Network, nextToken, len(tokens)-1, state, e.Embeddings, e.HiddenSize)
		if err != nil {
			return fmt.Sprintf("\n[Error: %v]", err)
		}
	}

	elapsed := time.Since(start)
	if generatedCount > 0 {
		fmt.Printf("\n\n(%.2f tokens/s, %d tokens total)\n", float64(generatedCount)/elapsed.Seconds(), generatedCount)
	}
	return stream.String()
}

func (e *LLMEngine) generateNextToken(tokens []uint32, opts GenOptions) (int, error) {
	hasEmbeddingLayer := false
	if len(e.Network.Layers) > 0 && e.Network.Layers[0].Type == nn.LayerEmbedding {
		hasEmbeddingLayer = true
	}

	var input []float32
	if hasEmbeddingLayer {
		input = make([]float32, len(tokens))
		for i, t := range tokens {
			input[i] = float32(t)
		}
	} else {
		input = make([]float32, len(tokens)*e.HiddenSize)
		for t, tokenID := range tokens {
			offset := int(tokenID) * e.HiddenSize
			if offset+e.HiddenSize > len(e.Embeddings) {
				return 0, fmt.Errorf("token ID %d out of bounds", tokenID)
			}
			for d := 0; d < e.HiddenSize; d++ {
				input[t*e.HiddenSize+d] = e.Embeddings[offset+d]
			}
		}
	}

	if !(e.Network.GPU && e.Network.IsGPUMounted()) {
		e.Network.BatchSize = len(tokens)
	}

	output, _ := e.Network.ForwardCPU(input)

	var normalized []float32
	if e.HasFinalNorm {
		finalNormConfig := &nn.LayerConfig{
			Type:     nn.LayerRMSNorm,
			NormSize: e.HiddenSize,
			Gamma:    e.FinalNorm,
			Epsilon:  1e-6,
		}
		normalized = nn.RmsNormForwardCPU(output, nil, finalNormConfig, len(tokens))
	} else {
		normalized = output
	}

	lastIdx := (len(tokens) - 1) * e.HiddenSize
	lastTokenNormalized := normalized[lastIdx : lastIdx+e.HiddenSize]

	logits := make([]float32, e.VocabSize)
	for v := 0; v < e.VocabSize; v++ {
		var sum float32
		offset := v * e.HiddenSize
		for d := 0; d < e.HiddenSize; d++ {
			sum += lastTokenNormalized[d] * e.LMHead[offset+d]
		}
		logits[v] = sum
	}

	e.applyRepetitionPenalty(logits, tokens, opts)

	return SampleTopK(logits, opts.TopK, opts.Temperature, opts.Deterministic), nil
}

func (e *LLMEngine) nextTokenFromHidden(hidden []float32, tokens []uint32, opts GenOptions) (int, error) {
	normalized := hidden
	if e.HasFinalNorm {
		finalNormConfig := &nn.LayerConfig{
			Type:     nn.LayerRMSNorm,
			NormSize: e.HiddenSize,
			Gamma:    e.FinalNorm,
			Epsilon:  1e-6,
		}
		normalized = nn.RmsNormForwardCPU(hidden, nil, finalNormConfig, 1)
	}

	logits := make([]float32, e.VocabSize)
	for v := 0; v < e.VocabSize; v++ {
		var sum float32
		offset := v * e.HiddenSize
		for d := 0; d < e.HiddenSize; d++ {
			sum += normalized[d] * e.LMHead[offset+d]
		}
		logits[v] = sum
	}

	e.applyRepetitionPenalty(logits, tokens, opts)

	return SampleTopK(logits, opts.TopK, opts.Temperature, opts.Deterministic), nil
}

func (e *LLMEngine) applyRepetitionPenalty(logits []float32, tokens []uint32, opts GenOptions) {
	if opts.RepetitionPenalty <= 1 || len(tokens) == 0 {
		return
	}

	window := opts.RepetitionWindow
	if window <= 0 {
		window = 64
	}
	start := len(tokens) - window
	if start < 0 {
		start = 0
	}
	recent := tokens[start:]

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

func (e *LLMEngine) isEOS(token int, eosTokens []int) bool {
	for _, eos := range eosTokens {
		if token == eos {
			return true
		}
	}
	return false
}
