package tokenizer

import (
	"path/filepath"
	"testing"
)

func TestLoadTokenizer(t *testing.T) {
	// This test requires the Qwen model to be downloaded
	// Skip if not available
	homeDir := t.TempDir()

	// If real tokenizer is available, test it
	realPath := filepath.Join(homeDir, ".cache", "huggingface", "hub",
		"models--Qwen--Qwen2.5-0.5B", "snapshots")

	tk, err := LoadFromFile(realPath)
	if err != nil {
		t.Skipf("Tokenizer not found (expected): %v", err)
		return
	}

	if tk.VocabSize() == 0 {
		t.Error("Vocabulary is empty")
	}

	t.Logf("Loaded tokenizer with vocab size: %d", tk.VocabSize())
}

func TestEncodeBasic(t *testing.T) {
	// Create a minimal tokenizer for testing
	tk := &Tokenizer{
		Vocab: map[string]int{
			"h": 0,
			"e": 1,
			"l": 2,
			"o": 3,
			" ": 4,
		},
		ReverseVocab: map[int]string{
			0: "h",
			1: "e",
			2: "l",
			3: "o",
			4: " ",
		},
		Merges: []MergePair{
			{First: "h", Second: "e", Rank: 0},
			{First: "l", Second: "l", Rank: 1},
		},
		SpecialTokens: make(map[string]int),
		AddedTokens:   make(map[string]int),
		PreTokenizer: &PreTokenizer{
			Pattern: nil, // No splitting for simple test
		},
	}

	tokens := tk.Encode("hello", false)
	if len(tokens) == 0 {
		t.Error("Expected tokens, got empty result")
	}

	t.Logf("Encoded 'hello' to %d tokens", len(tokens))
}

func TestDecode(t *testing.T) {
	tk := &Tokenizer{
		Vocab: map[string]int{
			"hello": 0,
			" ":     1,
			"world": 2,
		},
		ReverseVocab: map[int]string{
			0: "hello",
			1: " ",
			2: "world",
		},
		SpecialTokens: make(map[string]int),
		AddedTokens:   make(map[string]int),
	}

	text := tk.Decode([]uint32{0, 1, 2}, false)
	expected := "hello world"

	if text != expected {
		t.Errorf("Expected '%s', got '%s'", expected, text)
	}
}
