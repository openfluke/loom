package poly

import (
	"encoding/json"
	"os"
	"strings"
)

// HFConfigInt reads an integer from a Hugging Face config.json unmarshaled map.
func HFConfigInt(config map[string]interface{}, key string) (int, bool) {
	v, ok := config[key]
	if !ok {
		return 0, false
	}
	switch t := v.(type) {
	case float64:
		return int(t), true
	case int:
		return t, true
	case int64:
		return int(t), true
	default:
		return 0, false
	}
}

// HFConfigIntDefault returns d when key is missing or not coercible to int.
func HFConfigIntDefault(config map[string]interface{}, key string, d int) int {
	if v, ok := HFConfigInt(config, key); ok {
		return v
	}
	return d
}

// HFConfigFloat64 reads a float64 from config (HF JSON numbers decode as float64).
func HFConfigFloat64(config map[string]interface{}, key string) (float64, bool) {
	v, ok := config[key]
	if !ok {
		return 0, false
	}
	switch t := v.(type) {
	case float64:
		return t, true
	default:
		return 0, false
	}
}

// HFConfigFloat64Default returns d when key is missing or wrong type.
func HFConfigFloat64Default(config map[string]interface{}, key string, d float64) float64 {
	if v, ok := HFConfigFloat64(config, key); ok {
		return v
	}
	return d
}

func HFConfigStringDefault(config map[string]interface{}, key string, d string) string {
	if v, ok := config[key].(string); ok {
		return v
	}
	return d
}

// EOSTokenIDsFromHFConfig extracts eos_token_id from an unmarshaled config.json map.
// Returns default [2, 0] when absent or empty (matches lucy / welvet behavior).
func EOSTokenIDsFromHFConfig(config map[string]interface{}) []int {
	var tokens []int
	if eosID, ok := config["eos_token_id"]; ok {
		switch v := eosID.(type) {
		case float64:
			tokens = append(tokens, int(v))
		case []interface{}:
			for _, item := range v {
				if f, ok := item.(float64); ok {
					tokens = append(tokens, int(f))
				}
			}
		}
	}
	if len(tokens) == 0 {
		return []int{2, 0}
	}
	return tokens
}

// LoadEOSTokenIDsFromConfigPath reads config.json from disk and returns EOS IDs.
func LoadEOSTokenIDsFromConfigPath(configPath string) []int {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return []int{2, 0}
	}
	var config map[string]interface{}
	if json.Unmarshal(data, &config) != nil {
		return []int{2, 0}
	}
	return EOSTokenIDsFromHFConfig(config)
}

// TokenizerBannedSpecialExceptEOS returns tokenizer special token IDs that should be
// suppressed during generation, excluding EOS IDs from config.
func TokenizerBannedSpecialExceptEOS(tk *Tokenizer, eosTokens []int) []int {
	if tk == nil {
		return nil
	}
	eosSet := make(map[int]struct{}, len(eosTokens))
	for _, tok := range eosTokens {
		eosSet[tok] = struct{}{}
	}
	bannedSet := map[int]struct{}{}

	for _, id := range tk.SpecialTokens {
		if _, isEOS := eosSet[id]; isEOS {
			continue
		}
		bannedSet[id] = struct{}{}
	}

	for token, id := range tk.AddedTokens {
		if _, isEOS := eosSet[id]; isEOS {
			continue
		}
		if _, isSpecial := tk.SpecialTokens[token]; isSpecial {
			bannedSet[id] = struct{}{}
		}
	}

	out := make([]int, 0, len(bannedSet))
	for id := range bannedSet {
		out = append(out, id)
	}
	return out
}

// TemplateForHFModelID picks chat template from an HF-style model id string.
func TemplateForHFModelID(modelName string) Template {
	name := strings.ToLower(modelName)
	if strings.Contains(name, "microsoft/bitnet-b1.58-2b-4t") {
		return MicrosoftBitNetChat
	}
	if strings.Contains(name, "bitnet") || strings.Contains(name, "1bit") {
		return BitNetInstruction
	}
	if strings.Contains(name, "llama-3") || strings.Contains(name, "smollm3") {
		return Llama3
	}
	return ChatML
}
