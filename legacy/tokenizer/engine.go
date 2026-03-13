package tokenizer

import (
	"github.com/openfluke/loom/gpu"
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
	Transformer  *Transformer
	GPULMHead    *gpu.GPULMHead
	session      *KVSession // persistent KV cache across turns
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
		Transformer:  NewTransformer(network, embeddings, lmHead, finalNorm, template),
	}
}

func (e *LLMEngine) SetGPULMHead(lmh *gpu.GPULMHead) {
	e.GPULMHead = lmh
	if e.Transformer != nil {
		e.Transformer.GPULMHead = lmh
	}
}

// Generate runs one generation step, transparently reusing the KV cache
// from prior turns when UseKVCache is enabled.  The session is managed
// internally — callers never touch it.
func (e *LLMEngine) Generate(tk *Tokenizer, turns []Turn, systemPrompt, userMsg string, opts GenOptions) string {
	if !opts.UseKVCache {
		// Stateless path — no session, full rebuild each time.
		return e.Transformer.Generate(tk, turns, systemPrompt, userMsg, opts)
	}

	reply, updated := e.Transformer.GenerateWithSession(tk, e.session, turns, systemPrompt, userMsg, opts)
	e.session = updated
	return reply
}

// ResetSession clears the KV cache, forcing a full prefill on the next call.
// Call this on "!reset" or when the conversation context changes.
func (e *LLMEngine) ResetSession() {
	e.session = nil
}
