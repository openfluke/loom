package tokenizer

import (
	"fmt"
	"strings"
)

// WeightMapper provides a dynamic way to find and map tensors to specific roles
type WeightMapper struct {
	Patterns map[string][]string // role -> list of search patterns (suffixes or contains)
}

// NewWeightMapper creates a default mapper with common patterns
func NewWeightMapper() *WeightMapper {
	return &WeightMapper{
		Patterns: map[string][]string{
			"embeddings": {
				"model.embed_tokens.weight",
				"transformer.wte.weight",
				"embeddings.weight",
				"embed_tokens.weight",
			},
			"final_norm": {
				"model.norm.weight",
				"transformer.ln_f.weight",
				"ln_f.weight",
				"norm.weight",
			},
			"lm_head": {
				"lm_head.weight",
				"output.weight",
			},
		},
	}
}

// MapWeights finds weights for specific roles in the provided tensor map
func (m *WeightMapper) MapWeights(tensors map[string][]float32) (embeddings, lmHead, finalNorm []float32, hasFinalNorm bool) {
	embeddings = m.Find(tensors, "embeddings")
	finalNorm = m.Find(tensors, "final_norm")
	hasFinalNorm = (finalNorm != nil)

	lmHead = m.Find(tensors, "lm_head")
	if lmHead == nil {
		fmt.Printf("ℹ️  No separate lm_head found, assuming tied weights (using embeddings).\n")
		lmHead = embeddings
	}

	return
}

// Find searches for a tensor based on the patterns registered for a role
func (m *WeightMapper) Find(tensors map[string][]float32, role string) []float32 {
	patterns, ok := m.Patterns[role]
	if !ok {
		return nil
	}

	for _, pattern := range patterns {
		// Exact match first
		if t, ok := tensors[pattern]; ok {
			fmt.Printf("  ✓ Loaded %s: %d values (role: %s)\n", pattern, len(t), role)
			return t
		}

		// Suffix match
		for k, v := range tensors {
			if strings.HasSuffix(k, pattern) {
				fmt.Printf("  ✓ Loaded %s (via suffix: %s): %d values (role: %s)\n", k, pattern, len(v), role)
				return v
			}
		}
	}

	return nil
}
