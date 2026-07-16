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
	MaxTokens             int
	MinTokens             int
	Temperature           float32
	TopK                  int
	Deterministic         bool
	UseKVCache            bool
	RepetitionPenalty     float32
	RepetitionWindow      int
	MaxConsecutiveRepeats int
	NoRepeatNGram         int
	EOSTokens             []int
	BannedTokens          []int
	Silent                bool
	StreamCallback        func(string)

	// LayerTrace records every CPU sub-layer step and prints activation stats.
	// Forces host-side stepped forward (GPU forward is skipped).
	LayerTrace bool
	// LayerTraceMaxTokens caps generation when LayerTrace is set (>0 overrides MaxTokens).
	LayerTraceMaxTokens int
	// LayerTraceSilent suppresses per-step stderr/stdout lines (records are still kept).
	LayerTraceSilent bool
	// LayerTracePrefill traces the prompt forward (all layers × prompt length context).
	// Default off: "N tokens" usually means decode only (~181 steps per token for 30 layers).
	LayerTracePrefill bool
	// RepeatDecoderBlock is a 0-based decoder block index to run a second time after its
	// first pass; the repeat pass is labeled REPEAT in logs and records. <0 disables.
	RepeatDecoderBlock int

	// GPUSampleGreedy keeps ArgMax on the GPU and only maps back a 4-byte token id each
	// step (not the full logits vector). PoC for decode sync cost; implies greedy decode
	// and skips host logit masks (repetition / banned / n-gram). Requires UseGPU + synced LM head.
	GPUSampleGreedy bool
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

func (s *Streamer) Push(allTokens []uint32, silent bool, callback func(string)) {
	full := s.Decode(allTokens)
	if len(full) > s.lastLen {
		diff := full[s.lastLen:]
		diff = s.replacer.Replace(diff)
		if !silent {
			fmt.Print(diff)
		}
		if callback != nil {
			callback(diff)
		}
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

// ReplyLooksDegenerate detects token-salad / mojibake tails that small models
// often enter when a turn runs too long without <|im_end|>.
func ReplyLooksDegenerate(s string) bool {
	s = strings.TrimSpace(s)
	if len(s) < 48 {
		return false
	}
	runes := []rune(s)
	win := runes
	if len(win) > 96 {
		win = win[len(win)-96:]
	}
	bad := 0
	letters := 0
	spaces := 0
	weirdCase := 0 // AAaaa / CapsInside glue often precedes melt-down
	prevUpper := false
	for _, r := range win {
		switch {
		case r == '\uFFFD' || r == 'ï' || r == 'â' || r == 'Â' || r == 'Ã' ||
			r == '¼' || r == '½' || r == '¬' || r == 'ð' || r == 'Ð' || r == 'ÿ':
			bad++
		case r == ' ' || r == '\n' || r == '\t':
			spaces++
			prevUpper = false
		case (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z'):
			letters++
			up := r >= 'A' && r <= 'Z'
			if up && prevUpper {
				weirdCase++
			}
			prevUpper = up
		default:
			prevUpper = false
		}
	}
	if bad >= 2 {
		return true
	}
	// "Turpunda -Trustroom Bux Buggenes" style: many Titlecase clumps.
	if weirdCase >= 6 && letters >= 30 {
		return true
	}
	// Long glued run with almost no whitespace (classic BPE melt-down).
	if len(win) >= 48 && spaces < 2 && letters >= 24 {
		return true
	}
	if len(win) >= 72 && spaces < 4 && letters >= 36 {
		return true
	}
	return false
}

// SanitizeChatReply cuts a reply at the last clean sentence before degeneration.
// If nothing clean remains, returns empty (caller should drop the turn from history).
func SanitizeChatReply(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	if !ReplyLooksDegenerate(s) {
		return trimToLastSentence(s)
	}
	runes := []rune(s)
	best := ""
	for i := 1; i <= len(runes); i++ {
		pref := string(runes[:i])
		if ReplyLooksDegenerate(pref) {
			break
		}
		trimmed := strings.TrimSpace(pref)
		if endsSentence(trimmed) {
			best = trimmed
		} else if best == "" && len(trimmed) >= 24 {
			// keep growing fallback until we get a sentence end
			best = trimmed
		}
	}
	if best == "" {
		return ""
	}
	return trimToLastSentence(best)
}

func endsSentence(s string) bool {
	if s == "" {
		return false
	}
	switch s[len(s)-1] {
	case '.', '!', '?', '\n', '"', '\'':
		return true
	default:
		return false
	}
}

func trimToLastSentence(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	// Prefer an ending punctuation cut so MaxTokens mid-sentence tails get dropped.
	last := -1
	runes := []rune(s)
	for i, r := range runes {
		if r == '.' || r == '!' || r == '?' {
			last = i
		}
	}
	if last >= 0 && last+1 < len(runes) {
		// Keep through punctuation when the remainder is a short incomplete clause.
		rest := strings.TrimSpace(string(runes[last+1:]))
		if len(rest) > 0 && len(rest) < 40 && !strings.ContainsAny(rest, ".!?") {
			return strings.TrimSpace(string(runes[:last+1]))
		}
	}
	return s
}
