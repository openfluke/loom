package poly

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
		Type         string          `json:"type"`
		Vocab        map[string]int  `json:"vocab"`
		Merges       json.RawMessage `json:"merges"`
		ByteFallback bool            `json:"byte_fallback,omitempty"`
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

// LoadTokenizer loads a tokenizer from a HuggingFace tokenizer.json file
func LoadTokenizer(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read tokenizer file: %w", err)
	}

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

	// Parse merges flexibly (handles both []string and [][]string)
	var stringMerges []string
	if err := json.Unmarshal(tokJSON.Model.Merges, &stringMerges); err != nil {
		// Try [][]string fallback
		var stringSliceMerges [][]string
		if err2 := json.Unmarshal(tokJSON.Model.Merges, &stringSliceMerges); err2 == nil {
			for _, pair := range stringSliceMerges {
				if len(pair) == 2 {
					stringMerges = append(stringMerges, pair[0]+" "+pair[1])
				}
			}
		}
	}

	t.Merges = make([]MergePair, len(stringMerges))
	for i, merge := range stringMerges {
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

	allSpecial := make(map[string]int)
	for k, v := range t.SpecialTokens {
		allSpecial[k] = v
	}
	for k, v := range t.AddedTokens {
		allSpecial[k] = v
	}

	words := t.PreTokenizer.SplitWithSpecialTokens(text, allSpecial)

	var tokens []uint32
	for _, word := range words {
		if id, ok := t.SpecialTokens[word]; ok {
			tokens = append(tokens, uint32(id))
			continue
		}
		if id, ok := t.AddedTokens[word]; ok {
			tokens = append(tokens, uint32(id))
			continue
		}
		wordTokens := t.bpeEncode(word)
		tokens = append(tokens, wordTokens...)
	}
	return tokens
}

func (t *Tokenizer) bpeEncode(word string) []uint32 {
	if word == "" {
		return []uint32{}
	}

	gpt2Word := encodeToGPT2Chars(word)
	chars := t.splitToChars(gpt2Word)
	if len(chars) == 0 {
		return []uint32{}
	}

	if len(chars) == 1 {
		if id, ok := t.Vocab[chars[0]]; ok {
			return []uint32{uint32(id)}
		}
		return t.encodeBytes(word)
	}

	pairs := t.getPairs(chars)
	for len(pairs) > 0 {
		bestPair := t.findBestPair(pairs)
		if bestPair == nil {
			break
		}
		chars = t.applyMerge(chars, bestPair.First, bestPair.Second)
		if len(chars) == 1 {
			break
		}
		pairs = t.getPairs(chars)
	}

	var ids []uint32
	for _, token := range chars {
		if id, ok := t.Vocab[token]; ok {
			ids = append(ids, uint32(id))
		} else {
			byteIDs := t.encodeBytes(token)
			ids = append(ids, byteIDs...)
		}
	}
	return ids
}

func (t *Tokenizer) splitToChars(word string) []string {
	var chars []string
	for len(word) > 0 {
		r, size := utf8.DecodeRuneInString(word)
		if r == utf8.RuneError {
			chars = append(chars, word[:1])
			word = word[1:]
		} else {
			chars = append(chars, word[:size])
			word = word[size:]
		}
	}
	return chars
}

type PairWithIndex struct {
	First  string
	Second string
	Index  int
}

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

func (t *Tokenizer) findBestPair(pairs []PairWithIndex) *MergePair {
	var bestMerge *MergePair
	bestRank := len(t.Merges)
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

func (t *Tokenizer) applyMerge(tokens []string, first, second string) []string {
	if len(tokens) < 2 {
		return tokens
	}
	merged := make([]string, 0, len(tokens))
	i := 0
	for i < len(tokens) {
		if i < len(tokens)-1 && tokens[i] == first && tokens[i+1] == second {
			merged = append(merged, first+second)
			i += 2
		} else {
			merged = append(merged, tokens[i])
			i++
		}
	}
	return merged
}

func (t *Tokenizer) encodeBytes(s string) []uint32 {
	var ids []uint32
	for _, b := range []byte(s) {
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
		if skipSpecialTokens {
			if _, isSpecial := t.SpecialTokens[token]; isSpecial {
				continue
			}
		}
		tokens = append(tokens, token)
	}
	text := strings.Join(tokens, "")
	text = decodeGPT2Bytes(text)
	text = t.decodeByteFallback(text)
	return text
}

func encodeToGPT2Chars(text string) string {
	var b strings.Builder
	bytes := []byte(text)
	for _, byteVal := range bytes {
		r := gpt2ByteEncode(byteVal)
		b.WriteRune(r)
	}
	return b.String()
}

func gpt2ByteEncode(b byte) rune {
	val := int(b)
	if val <= 0x20 {
		return rune(0x100 + val)
	} else if val == 0x7F {
		return rune(0x121)
	} else if val >= 0x80 {
		return rune(0x100 + val - 0x80 + 0x22)
	}
	return rune(val)
}

func decodeGPT2Bytes(text string) string {
	var result strings.Builder
	for _, r := range text {
		decoded := gpt2ByteDecode(r)
		result.WriteRune(decoded)
	}
	return result.String()
}

func gpt2ByteDecode(r rune) rune {
	if r >= 0x100 && r <= 0x1FF {
		offset := int(r) - 0x100
		if offset <= 0x20 {
			return rune(offset)
		} else if offset == 0x21 {
			return 0x7F
		} else if offset >= 0x22 && offset <= 0x42 {
			return rune(0x80 + (offset - 0x22))
		}
	}
	return r
}

func (t *Tokenizer) decodeByteFallback(text string) string {
	re := regexp.MustCompile(`<0x([0-9A-F]{2})>`)
	result := re.ReplaceAllStringFunc(text, func(match string) string {
		var b byte
		fmt.Sscanf(match, "<0x%02X>", &b)
		return string([]byte{b})
	})
	return result
}

func (pt *PreTokenizer) SplitWithSpecialTokens(text string, specialTokens map[string]int) []string {
	if text == "" {
		return []string{}
	}
	if len(specialTokens) > 0 {
		var result []string
		remaining := text
		for len(remaining) > 0 {
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
			result = append(result, earliestToken)
			remaining = remaining[earliestIdx+len(earliestToken):]
		}
		return result
	}
	if pt.Pattern == nil {
		return []string{text}
	}
	matches := pt.Pattern.FindAllString(text, -1)
	if matches == nil {
		return []string{text}
	}
	return matches
}
