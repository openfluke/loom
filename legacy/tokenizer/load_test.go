package tokenizer_test

import (
	"os"
	"testing"

	"github.com/openfluke/loom/tokenizer"
)

// Example of loading tokenizer from bytes
func ExampleLoadFromBytes() {
	// In production, you might get this data from:
	// - Embedded files (go:embed)
	// - Network request
	// - Database
	// - Custom storage backend

	data := []byte(`{
		"model": {
			"type": "BPE",
			"vocab": {
				"hello": 0,
				"world": 1,
				" ": 2
			},
			"merges": []
		},
		"added_tokens": []
	}`)

	tk, err := tokenizer.LoadFromBytes(data)
	if err != nil {
		panic(err)
	}

	// Use the tokenizer
	tokens := tk.Encode("hello world", false)
	_ = tokens
}

// TestLoadFromBytesMatchesFile verifies both loading methods produce identical tokenizers
func TestLoadFromBytesMatchesFile(t *testing.T) {
	// Create a temp tokenizer file for testing
	tmpfile, err := os.CreateTemp("", "tokenizer*.json")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpfile.Name())

	testData := []byte(`{
		"model": {
			"type": "BPE",
			"vocab": {
				"a": 0,
				"b": 1,
				"ab": 2
			},
			"merges": ["a b"]
		},
		"added_tokens": [
			{"id": 3, "content": "<pad>", "special": true}
		]
	}`)

	if _, err := tmpfile.Write(testData); err != nil {
		t.Fatal(err)
	}
	tmpfile.Close()

	// Load from file
	tk1, err := tokenizer.LoadFromFile(tmpfile.Name())
	if err != nil {
		t.Fatalf("LoadFromFile failed: %v", err)
	}

	// Load from bytes
	tk2, err := tokenizer.LoadFromBytes(testData)
	if err != nil {
		t.Fatalf("LoadFromBytes failed: %v", err)
	}

	// Compare vocab sizes
	if tk1.VocabSize() != tk2.VocabSize() {
		t.Errorf("Vocab size mismatch: %d vs %d", tk1.VocabSize(), tk2.VocabSize())
	}

	// Compare encoding
	text := "ab"
	tokens1 := tk1.Encode(text, false)
	tokens2 := tk2.Encode(text, false)

	if len(tokens1) != len(tokens2) {
		t.Errorf("Token count mismatch: %d vs %d", len(tokens1), len(tokens2))
	}

	for i := range tokens1 {
		if tokens1[i] != tokens2[i] {
			t.Errorf("Token[%d] mismatch: %d vs %d", i, tokens1[i], tokens2[i])
		}
	}

	t.Logf("✓ Both loading methods produce identical tokenizers")
}
