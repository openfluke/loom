package tokenizer

import (
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

// Generate starts a generation loop and returns the final string
func (e *LLMEngine) Generate(tk *Tokenizer, turns []Turn, systemPrompt, userMsg string, opts GenOptions) string {
	return e.Transformer.Generate(tk, turns, systemPrompt, userMsg, opts)
}
