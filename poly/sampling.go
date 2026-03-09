package poly

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
)

// GenOptions defines the generation parameters
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

// SampleTopK performs top-K sampling with temperature and optional determinism
func SampleTopK(logits []float32, topK int, temperature float32, deterministic bool) int {
	if topK == 1 || temperature <= 0 || deterministic {
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
	Decode       func(tokens []uint32) string
	lastLen      int
	promptLenRaw int
	sb           strings.Builder
	replacer     *strings.Replacer
}

func NewStreamer(decode func(tokens []uint32) string, promptTokens []uint32) *Streamer {
	promptTextRaw := decode(promptTokens)
	return &Streamer{
		Decode:       decode,
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
	full := s.Decode(allTokens)
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
	fullRaw := s.Decode(allTokens)
	if len(fullRaw) <= s.promptLenRaw {
		return false
	}
	return strings.Contains(fullRaw[s.promptLenRaw:], "<|im_start|>user")
}
