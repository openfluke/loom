package tokenizer

import (
	"strings"
)

// Turn represents a single turn in a chat conversation
type Turn struct {
	User      string
	Assistant string
}

// Template defines the formatting markers for different chat styles
type Template struct {
	Name         string
	RolePrefixes map[string]string
	RoleSuffixes map[string]string
	GlobalPrefix string
	GlobalSuffix string
}

// BuildPrompt constructs a full prompt string from conversation turns
func (t Template) BuildPrompt(turns []Turn, systemPrompt string, userMsg string) string {
	var sb strings.Builder
	sb.WriteString(t.GlobalPrefix)

	if systemPrompt != "" {
		if pre, ok := t.RolePrefixes["system"]; ok {
			sb.WriteString(pre)
			sb.WriteString(strings.TrimSpace(systemPrompt))
			sb.WriteString(t.RoleSuffixes["system"])
		}
	}

	for _, turn := range turns {
		if pre, ok := t.RolePrefixes["user"]; ok {
			sb.WriteString(pre)
			sb.WriteString(turn.User)
			sb.WriteString(t.RoleSuffixes["user"])
		}
		if pre, ok := t.RolePrefixes["assistant"]; ok {
			sb.WriteString(pre)
			sb.WriteString(turn.Assistant)
			sb.WriteString(t.RoleSuffixes["assistant"])
		}
	}

	// Add current user message
	if pre, ok := t.RolePrefixes["user"]; ok {
		sb.WriteString(pre)
		sb.WriteString(userMsg)
		sb.WriteString(t.RoleSuffixes["user"])
	}

	// Add final assistant prefix to trigger generation
	if pre, ok := t.RolePrefixes["assistant"]; ok {
		sb.WriteString(pre)
	}

	sb.WriteString(t.GlobalSuffix)
	return sb.String()
}

// BuildNextTurnSegment returns only the text that is NEW compared to what the
// KV cache already holds.  The previous assistant reply's suffix is the last
// thing in the cache, so we emit:
//
//	<user_prefix><msg><user_suffix><assistant_prefix>
//
// This is exactly the segment that must be tokenised and prefilled on turn N+1
// when the session already contains turns 0..N.
func (t Template) BuildNextTurnSegment(userMsg string) string {
	var sb strings.Builder
	if pre, ok := t.RolePrefixes["user"]; ok {
		sb.WriteString(pre)
		sb.WriteString(userMsg)
		sb.WriteString(t.RoleSuffixes["user"])
	}
	if pre, ok := t.RolePrefixes["assistant"]; ok {
		sb.WriteString(pre)
	}
	return sb.String()
}

// Preset templates
var (
	// ChatML is used by Qwen, SmolLM2, etc.
	ChatML = Template{
		Name: "chatml",
		RolePrefixes: map[string]string{
			"system":    "<|im_start|>system\n",
			"user":      "<|im_start|>user\n",
			"assistant": "<|im_start|>assistant\n",
		},
		RoleSuffixes: map[string]string{
			"system":    "<|im_end|>\n",
			"user":      "<|im_end|>\n",
			"assistant": "<|im_end|>\n",
		},
	}

	// Llama3 markers
	Llama3 = Template{
		Name: "llama3",
		RolePrefixes: map[string]string{
			"system":    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
			"user":      "<|start_header_id|>user<|end_header_id|>\n\n",
			"assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n",
		},
		RoleSuffixes: map[string]string{
			"system":    "<|eot_id|>",
			"user":      "<|eot_id|>",
			"assistant": "<|eot_id|>",
		},
	}
)
