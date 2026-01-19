package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
	"unicode/utf8"
)

// Tokenizer represents a BPE tokenizer
type Tokenizer struct {
	Vocab         map[string]int // token -> id
	ReverseVocab  map[int]string // id -> token
	Merges        []MergePair    // BPE merge rules
	SpecialTokens map[string]int // special tokens
	AddedTokens   map[string]int // added tokens
	PreTokenizer  *PreTokenizer  // pre-tokenization rules
	ByteFallback  bool           // use byte fallback for unknown chars
}

// MergePair represents a BPE merge rule
type MergePair struct {
	First  string
	Second string
	Rank   int
}

// PreTokenizer handles text splitting before BPE
type PreTokenizer struct {
	Pattern *regexp.Regexp
}

// TokenizerJSON represents the HuggingFace tokenizer.json format
type TokenizerJSON struct {
	Model struct {
		Type         string         `json:"type"`
		Vocab        map[string]int `json:"vocab"`
		Merges       []string       `json:"merges"`
		ByteFallback bool           `json:"byte_fallback,omitempty"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
	PreTokenizer struct {
		Type          string `json:"type"`
		Pretokenizers []struct {
			Type    string `json:"type"`
			Pattern struct {
				String string `json:"String"`
			} `json:"pattern,omitempty"`
		} `json:"pretokenizers,omitempty"`
	} `json:"pre_tokenizer"`
}

// LoadFromFile loads a tokenizer from a HuggingFace tokenizer.json file
func LoadFromFile(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read tokenizer file: %w", err)
	}

	return LoadFromBytes(data)
}

// LoadFromBytes loads a tokenizer from HuggingFace tokenizer.json data
func LoadFromBytes(data []byte) (*Tokenizer, error) {
	var tokJSON TokenizerJSON
	if err := json.Unmarshal(data, &tokJSON); err != nil {
		return nil, fmt.Errorf("failed to parse tokenizer JSON: %w", err)
	}

	t := &Tokenizer{
		Vocab:         tokJSON.Model.Vocab,
		ReverseVocab:  make(map[int]string),
		SpecialTokens: make(map[string]int),
		AddedTokens:   make(map[string]int),
		ByteFallback:  tokJSON.Model.ByteFallback,
	}

	// Build reverse vocab
	for token, id := range t.Vocab {
		t.ReverseVocab[id] = token
	}

	// Parse merges
	t.Merges = make([]MergePair, len(tokJSON.Model.Merges))
	for i, merge := range tokJSON.Model.Merges {
		parts := strings.Split(merge, " ")
		if len(parts) != 2 {
			continue
		}
		t.Merges[i] = MergePair{
			First:  parts[0],
			Second: parts[1],
			Rank:   i,
		}
	}

	// Handle added tokens
	for _, token := range tokJSON.AddedTokens {
		t.AddedTokens[token.Content] = token.ID
		if token.Special {
			t.SpecialTokens[token.Content] = token.ID
		}
	}

	// Set up pre-tokenizer (GPT-2 style pattern)
	// Simplified regex for Go (no negative lookahead support)
	// Splits on: contractions, letters, numbers, punctuation, whitespace
	pattern := `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`
	t.PreTokenizer = &PreTokenizer{
		Pattern: regexp.MustCompile(pattern),
	}

	return t, nil
}

// Encode converts text to token IDs
func (t *Tokenizer) Encode(text string, addSpecialTokens bool) []uint32 {
	if text == "" {
		return []uint32{}
	}

	// Combine special tokens and added tokens for preservation during splitting
	allSpecial := make(map[string]int)
	for k, v := range t.SpecialTokens {
		allSpecial[k] = v
	}
	for k, v := range t.AddedTokens {
		allSpecial[k] = v
	}

	// Pre-tokenization: split text into words, preserving special tokens
	words := t.PreTokenizer.SplitWithSpecialTokens(text, allSpecial)

	var tokens []uint32
	for _, word := range words {
		// Check if it's a special token
		if id, ok := t.SpecialTokens[word]; ok {
			tokens = append(tokens, uint32(id))
			continue
		}

		// Check if it's an added token
		if id, ok := t.AddedTokens[word]; ok {
			tokens = append(tokens, uint32(id))
			continue
		}

		// Apply BPE to the word
		wordTokens := t.bpeEncode(word)
		tokens = append(tokens, wordTokens...)
	}

	return tokens
}

// bpeEncode applies BPE algorithm to a word
func (t *Tokenizer) bpeEncode(word string) []uint32 {
	if word == "" {
		return []uint32{}
	}

	// Convert word to initial tokens (characters or bytes)
	// Text is first mapped to GPT-2 byte-encoding characters
	// e.g. " " (space) -> "Ġ" (U+0120)
	gpt2Word := encodeToGPT2Chars(word)

	chars := t.splitToChars(gpt2Word)
	if len(chars) == 0 {
		return []uint32{}
	}

	// Single character - look up directly
	if len(chars) == 1 {
		if id, ok := t.Vocab[chars[0]]; ok {
			return []uint32{uint32(id)}
		}
		// Fallback to byte encoding
		return t.encodeBytes(word)
	}

	// Apply BPE merges
	pairs := t.getPairs(chars)

	for len(pairs) > 0 {
		// Find the merge with lowest rank
		bestPair := t.findBestPair(pairs)
		if bestPair == nil {
			break
		}

		// Apply the merge
		chars = t.applyMerge(chars, bestPair.First, bestPair.Second)

		if len(chars) == 1 {
			break
		}

		pairs = t.getPairs(chars)
	}

	// Convert final tokens to IDs
	var ids []uint32
	for _, token := range chars {
		if id, ok := t.Vocab[token]; ok {
			ids = append(ids, uint32(id))
		} else {
			// Fallback: encode as bytes
			byteIDs := t.encodeBytes(token)
			ids = append(ids, byteIDs...)
		}
	}

	return ids
}

// splitToChars splits a word into initial character tokens
func (t *Tokenizer) splitToChars(word string) []string {
	var chars []string
	for len(word) > 0 {
		r, size := utf8.DecodeRuneInString(word)
		if r == utf8.RuneError {
			// Invalid UTF-8, use byte fallback
			chars = append(chars, word[:1])
			word = word[1:]
		} else {
			chars = append(chars, word[:size])
			word = word[size:]
		}
	}
	return chars
}

// getPairs generates all adjacent pairs from a list of tokens
func (t *Tokenizer) getPairs(tokens []string) []PairWithIndex {
	if len(tokens) < 2 {
		return nil
	}

	pairs := make([]PairWithIndex, 0, len(tokens)-1)
	for i := 0; i < len(tokens)-1; i++ {
		pairs = append(pairs, PairWithIndex{
			First:  tokens[i],
			Second: tokens[i+1],
			Index:  i,
		})
	}
	return pairs
}

// PairWithIndex tracks position of a pair
type PairWithIndex struct {
	First  string
	Second string
	Index  int
}

// findBestPair finds the pair with the lowest merge rank
func (t *Tokenizer) findBestPair(pairs []PairWithIndex) *MergePair {
	var bestMerge *MergePair
	bestRank := len(t.Merges) // Max possible rank

	for _, pair := range pairs {
		for i := range t.Merges {
			if t.Merges[i].First == pair.First && t.Merges[i].Second == pair.Second {
				if t.Merges[i].Rank < bestRank {
					bestRank = t.Merges[i].Rank
					bestMerge = &t.Merges[i]
				}
				break
			}
		}
	}

	return bestMerge
}

// applyMerge merges all occurrences of a pair in the token list
func (t *Tokenizer) applyMerge(tokens []string, first, second string) []string {
	if len(tokens) < 2 {
		return tokens
	}

	merged := make([]string, 0, len(tokens))
	i := 0
	for i < len(tokens) {
		if i < len(tokens)-1 && tokens[i] == first && tokens[i+1] == second {
			// Merge this pair
			merged = append(merged, first+second)
			i += 2
		} else {
			merged = append(merged, tokens[i])
			i++
		}
	}

	return merged
}

// encodeBytes encodes a string as raw bytes (fallback for unknown tokens)
func (t *Tokenizer) encodeBytes(s string) []uint32 {
	var ids []uint32
	for _, b := range []byte(s) {
		// Look for byte tokens in vocab (usually formatted as <0xHH>)
		byteToken := fmt.Sprintf("<0x%02X>", b)
		if id, ok := t.Vocab[byteToken]; ok {
			ids = append(ids, uint32(id))
		}
	}
	return ids
}

// Decode converts token IDs to text
func (t *Tokenizer) Decode(ids []uint32, skipSpecialTokens bool) string {
	var tokens []string

	for _, id := range ids {
		token, ok := t.ReverseVocab[int(id)]
		if !ok {
			continue
		}

		// Skip special tokens if requested
		if skipSpecialTokens {
			if _, isSpecial := t.SpecialTokens[token]; isSpecial {
				continue
			}
		}

		tokens = append(tokens, token)
	}

	// Join tokens and clean up
	text := strings.Join(tokens, "")

	// Handle GPT-2 style byte encoding (Ġ -> space, Ċ -> newline, etc.)
	text = decodeGPT2Bytes(text)

	// Handle byte tokens (e.g., <0x20> -> space)
	text = t.decodeByteFallback(text)

	return text
}

// encodeToGPT2Chars converts text to GPT-2/SmolLM2 style byte-encoded characters
// This is the inverse of decodeGPT2Bytes
func encodeToGPT2Chars(text string) string {
	var b strings.Builder
	// Convert string to bytes first, as the mapping operates on bytes
	bytes := []byte(text)
	for _, byteVal := range bytes {
		r := gpt2ByteEncode(byteVal)
		b.WriteRune(r)
	}
	return b.String()
}

// gpt2ByteEncode maps a single byte to its GPT-2 unicode representation
func gpt2ByteEncode(b byte) rune {
	// GPT-2 mapping logic (inverse of gpt2ByteDecode)

	// Printable ASCII characters (except space, which is 0x20)
	// In the decoder:
	// 0x21-0x7E map to themselves
	// 0xA1-0xAC map to themselves
	// 0xAE-0xFF map to themselves (Wait, strictly checking the decoder logic is safer)

	// Let's look at decoder:
	// if r >= 0x100 && r <= 0x1FF { ... decode ... }
	// else return r

	// So if we have a byte b:
	// If b is a printable char that GPT-2 preserves, we return rune(b).
	// If it's a control char or space or specific others, we map to 0x100 + offset.

	// Re-deriving from decoder logic:
	// Decoder:
	// offset <= 0x20  -> return offset (0x00-0x20)
	// offset == 0x21  -> return 0x7F (DEL)
	// offset >= 0x22 ... -> return 0x80 + (offset - 0x22)

	// So Encoder:
	// If b <= 0x20: offset = b -> rune = 0x100 + b
	// If b == 0x7F: offset = 0x21 -> rune = 0x100 + 0x21 = 0x121
	// If b >= 0x80: offset = b - 0x80 + 0x22 -> rune = 0x100 + b - 0x80 + 0x22 = b + 0x100 - 0x5E
	//    (Wait, let's check the math. 0x80 -> 0x122. Decoder: 0x122-0x100=0x22. 0x80 + (0x22 - 0x22) = 0x80. Correct.)

	val := int(b)
	if val <= 0x20 {
		return rune(0x100 + val)
	} else if val == 0x7F {
		return rune(0x121)
	} else if val >= 0x80 {
		return rune(0x100 + val - 0x80 + 0x22)
	}

	// Printable ASCII (0x21 - 0x7E) remains as is
	return rune(val)
}

// decodeGPT2Bytes converts GPT-2/SmolLM2 style byte-encoded characters back to normal text
// GPT-2 uses a byte-to-unicode mapping where printable ASCII is shifted to avoid special chars
func decodeGPT2Bytes(text string) string {
	// GPT-2 byte-to-unicode mapping (reverse direction)
	// The encoding uses Unicode chars starting at various points to represent bytes
	var result strings.Builder
	for _, r := range text {
		decoded := gpt2ByteDecode(r)
		result.WriteRune(decoded)
	}
	return result.String()
}

// gpt2ByteDecode reverses the GPT-2 byte-to-unicode encoding
func gpt2ByteDecode(r rune) rune {
	// GPT-2 uses a specific mapping to avoid control characters
	// Characters 0x21-0x7E and 0xA1-0xAC and 0xAE-0xFF are kept as-is (printable)
	// Others are shifted to Unicode range starting at 0x100

	// Common mappings for SmolLM2/GPT-2:
	// Ġ (U+0120 = 288) -> space (0x20 = 32)
	// Ċ (U+010A = 266) -> newline (0x0A = 10)
	// ĉ (U+0109 = 265) -> tab (0x09 = 9)
	// Ď (U+010E = 270) -> carriage return (0x0D = 13)

	// The pattern: for bytes 0-255 that aren't printable ASCII,
	// GPT-2 maps them to 0x100 + offset

	if r >= 0x100 && r <= 0x1FF {
		// This is an encoded byte - decode it
		// The offset depends on which "gap" in the printable range we're filling
		offset := int(r) - 0x100

		// Reconstruct the original byte
		// Bytes 0x00-0x20 (control chars + space) -> 0x100-0x120
		// Byte 0x7F (DEL) -> 0x121
		// Bytes 0x80-0xA0 -> 0x122-0x142
		// etc.

		if offset <= 0x20 {
			return rune(offset)
		} else if offset == 0x21 {
			return 0x7F // DEL
		} else if offset >= 0x22 && offset <= 0x42 {
			return rune(0x80 + (offset - 0x22))
		}
	}

	// Keep printable ASCII and already-decoded chars as-is
	return r
}

// decodeByteFallback decodes byte fallback tokens
func (t *Tokenizer) decodeByteFallback(text string) string {
	// Replace byte tokens like <0x20> with actual bytes
	re := regexp.MustCompile(`<0x([0-9A-F]{2})>`)
	result := re.ReplaceAllStringFunc(text, func(match string) string {
		var b byte
		fmt.Sscanf(match, "<0x%02X>", &b)
		return string([]byte{b})
	})
	return result
}

// Split splits text using the pre-tokenizer pattern
// It preserves special tokens by finding them first before regex splitting
func (pt *PreTokenizer) Split(text string) []string {
	return pt.SplitWithSpecialTokens(text, nil)
}

// SplitWithSpecialTokens splits text while preserving special tokens
func (pt *PreTokenizer) SplitWithSpecialTokens(text string, specialTokens map[string]int) []string {
	if text == "" {
		return []string{}
	}

	// If we have special tokens, find and preserve them
	if len(specialTokens) > 0 {
		var result []string
		remaining := text

		for len(remaining) > 0 {
			// Find the earliest special token in remaining text
			earliestIdx := -1
			earliestToken := ""

			for token := range specialTokens {
				idx := strings.Index(remaining, token)
				if idx != -1 && (earliestIdx == -1 || idx < earliestIdx) {
					earliestIdx = idx
					earliestToken = token
				}
			}

			if earliestIdx == -1 {
				// No more special tokens, process the rest normally
				if pt.Pattern != nil {
					matches := pt.Pattern.FindAllString(remaining, -1)
					if matches != nil {
						result = append(result, matches...)
					}
				} else {
					result = append(result, remaining)
				}
				break
			}

			// Process text before the special token
			if earliestIdx > 0 {
				before := remaining[:earliestIdx]
				if pt.Pattern != nil {
					matches := pt.Pattern.FindAllString(before, -1)
					if matches != nil {
						result = append(result, matches...)
					}
				} else {
					result = append(result, before)
				}
			}

			// Add the special token as-is
			result = append(result, earliestToken)

			// Continue with text after the special token
			remaining = remaining[earliestIdx+len(earliestToken):]
		}

		return result
	}

	// No special tokens provided, use normal regex splitting
	if pt.Pattern == nil {
		return []string{text}
	}

	matches := pt.Pattern.FindAllString(text, -1)
	if matches == nil {
		return []string{text}
	}

	return matches
}

// EncodeWithOffsets returns tokens with their character offsets
func (t *Tokenizer) EncodeWithOffsets(text string) ([]uint32, [][2]int) {
	tokens := t.Encode(text, false)
	// For now, return empty offsets - can be implemented if needed
	offsets := make([][2]int, len(tokens))
	return tokens, offsets
}

// VocabSize returns the size of the vocabulary
func (t *Tokenizer) VocabSize() int {
	return len(t.Vocab)
}

// TokenToID converts a token string to its ID
func (t *Tokenizer) TokenToID(token string) (int, bool) {
	id, ok := t.Vocab[token]
	return id, ok
}

// IDToToken converts a token ID to its string
func (t *Tokenizer) IDToToken(id int) (string, bool) {
	token, ok := t.ReverseVocab[id]
	return token, ok
}
