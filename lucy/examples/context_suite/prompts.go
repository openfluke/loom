package contextsuite

import (
	"fmt"
	"strings"
)

// Scenario is one multi-prompt or long-context test case.
type Scenario struct {
	ID          string
	Description string
	// Turns lists user messages in order. Each entry is one chat turn; prior assistant
	// replies are filled in by the runner as the conversation progresses.
	Turns []string
	// MaxTokens caps generation for this scenario (0 = model default).
	MaxTokens int
	// RecallNeedle, if set, is checked for in the final reply (case-insensitive).
	RecallNeedle string
	// ExpectOverflow marks scenarios designed to exceed maxSeqLen when templated.
	ExpectOverflow bool
}

// longPrefillBody repeats a paragraph to approach the 512-token KV cache limit.
func longPrefillBody() string {
	para := strings.TrimSpace(
		"Explain wavefront decoder scheduling, KV cache striping, and why multiple prompt tokens " +
			"can sit in different transformer blocks at the same clock tick. " +
			"The Loom poly stack uses MaxSeqLen=512 in Lucy which bounds how much conversation history " +
			"fits in the KV cache before older context is effectively unreachable.")
	return strings.Repeat(para+" ", 6) +
		"\n\nBased on the text above: what is the MaxSeqLen value mentioned? Reply with just the number."
}

func mediumFiller(i int) string {
	return strings.TrimSpace(
		fmt.Sprintf("Please summarize in one sentence why layered neural networks use residual connections "+
			"and normalization between sub-layers. Turn %d.", i))
}

// AllScenarios is the full prompt battery for long-context and multi-turn probing.
var AllScenarios = []Scenario{
	{
		ID:          "short_baseline",
		Description: "Single short prompt — sanity check generation works",
		Turns:       []string{"What is 2+2? Reply with just the number."},
		MaxTokens:   16,
	},
	{
		ID:          "long_prefill",
		Description: "Long single-turn prefill near MaxSeqLen=512 boundary",
		Turns:       []string{longPrefillBody()},
		MaxTokens:   24,
		RecallNeedle: "512",
	},
	{
		ID:          "multi_turn_4",
		Description: "Four short turns — KV cache accumulation across turns",
		Turns: []string{
			"My name is Alice. Remember this.",
			"What is the capital of France? One word.",
			"What is 3 times 7? Reply with just the number.",
			"What is my name?",
		},
		MaxTokens:    32,
		RecallNeedle: "alice",
	},
	{
		ID:          "multi_turn_recall",
		Description: "Needle-in-haystack recall after filler turns",
		Turns: []string{
			"The secret code is BANANA-42. Remember it exactly.",
			"What is the largest planet in our solar system? One word.",
			"How many continents are there? Reply with just the number.",
			"What was the secret code I gave you? Reply with the code only.",
		},
		MaxTokens:    32,
		RecallNeedle: "banana-42",
	},
	{
		ID:          "overflow_probe",
		Description: "Many medium turns — total context likely exceeds MaxSeqLen=512",
		Turns: []string{
			"Session start. The project codename is ORBIT-7. Remember it.",
			mediumFiller(1),
			mediumFiller(2),
			mediumFiller(3),
			mediumFiller(4),
			mediumFiller(5),
			mediumFiller(6),
			"What was the project codename from the first message?",
		},
		MaxTokens:      32,
		RecallNeedle:   "orbit-7",
		ExpectOverflow: true,
	},
}
