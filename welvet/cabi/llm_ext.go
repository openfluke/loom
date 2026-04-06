package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/openfluke/loom/poly"
)

// ─── LLM handle registry ───────────────────────────────────────────────────────

// pollStatus values returned by LoomLLMPollToken
const (
	pollToken    = 0 // token chunk available
	pollDone     = 1 // generation complete, stats included
	pollEmpty    = 2 // queue empty, still generating
)

type streamResult struct {
	metrics poly.GenMetrics
	reply   string
	err     string
}

type llmState struct {
	tr            *poly.Transformer[float32]
	tk            *poly.Tokenizer
	eosTokens     []int
	bannedTokens  []int
	history       []poly.Turn
	deterministic bool

	// streaming
	streamMu      sync.Mutex
	tokenQueue    []string
	streamDone    bool
	streamResult  *streamResult
	pendingUser   string // stored for history append on completion
}

var (
	llmStates   = make(map[int64]*llmState)
	llmNextID   int64 = 1
	llmMu       sync.Mutex
)

// ─── LoomLLMListModels ────────────────────────────────────────────────────────

//export LoomLLMListModels
func LoomLLMListModels(hubDirC *C.char) *C.char {
	hubDir := C.GoString(hubDirC)
	entries, err := os.ReadDir(hubDir)
	if err != nil {
		return C.CString(`[]`)
	}
	var models []string
	for _, e := range entries {
		if e.IsDir() && strings.HasPrefix(e.Name(), "models--") {
			name := strings.TrimPrefix(e.Name(), "models--")
			name = strings.Replace(name, "--", "/", 1)
			models = append(models, name)
		}
	}
	data, _ := json.Marshal(models)
	return C.CString(string(data))
}

// ─── LoomCreateLLM ────────────────────────────────────────────────────────────
// snapshotDirC : full path to the snapshot directory (e.g. ~/.cache/huggingface/hub/models--…/snapshots/main)
// execMode     : 1=standard, 2=single-core tiled, 3=multi-core tiled
// precisionInt : 4=Q4, 8=INT8, 32=FP32
// useGPUInt    : 1=GPU, 0=CPU
// deterministicInt: 1=greedy, 0=sampling
// Returns LLM handle (>0) or -1 on error.

//export LoomCreateLLM
func LoomCreateLLM(snapshotDirC *C.char, execMode C.int, precisionInt C.int, useGPUInt C.int, deterministicInt C.int) C.longlong {
	snapshotDir := C.GoString(snapshotDirC)

	// ── Tokenizer ─────────────────────────────────────────────────────────────
	tk, err := poly.LoadTokenizer(filepath.Join(snapshotDir, "tokenizer.json"))
	if err != nil {
		return -1
	}

	// ── Config ────────────────────────────────────────────────────────────────
	configPath := filepath.Join(snapshotDir, "config.json")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return -1
	}
	var config map[string]interface{}
	if err := json.Unmarshal(configData, &config); err != nil {
		return -1
	}

	eosTokens := llmLoadEOS(config)

	// ── Load safetensors ──────────────────────────────────────────────────────
	safetensorFiles, _ := filepath.Glob(filepath.Join(snapshotDir, "*.safetensors"))
	if len(safetensorFiles) == 0 {
		return -1
	}
	allTensors := make(map[string][]float32)
	for _, f := range safetensorFiles {
		t, err := poly.LoadSafetensors(f)
		if err != nil {
			continue
		}
		for k, v := range t {
			allTensors[k] = v
		}
	}

	// ── Map weights ───────────────────────────────────────────────────────────
	mapper := poly.NewPrefixWeightMapper()
	embeddings, lmHead, finalNorm, _ := mapper.MapWeights(allTensors)

	// ── Count layers ──────────────────────────────────────────────────────────
	numLayers := 0
	for k := range allTensors {
		if strings.Contains(k, "layers.") {
			parts := strings.Split(k, ".")
			for i, p := range parts {
				if p == "layers" && i+1 < len(parts) {
					if idx, ok := parseInt(parts[i+1]); ok && idx+1 > numLayers {
						numLayers = idx + 1
					}
				}
			}
		}
	}
	if numLayers == 0 {
		if v, ok := config["num_hidden_layers"]; ok {
			numLayers = int(v.(float64))
		}
	}
	if numLayers == 0 {
		return -1
	}

	// ── Architecture params ───────────────────────────────────────────────────
	numHeads := intCfg(config, "num_attention_heads", 1)
	numKVHeads := intCfg(config, "num_key_value_heads", numHeads)
	hiddenSize := len(finalNorm)
	if hiddenSize == 0 {
		hiddenSize = intCfg(config, "hidden_size", 0)
	}
	if hiddenSize == 0 || numHeads == 0 {
		return -1
	}
	headDim := hiddenSize / numHeads
	intermediateSize := intCfg(config, "intermediate_size", hiddenSize*4)
	ropeFreqBase := floatCfg(config, "rope_theta", 10000.0)

	// ── dtype ─────────────────────────────────────────────────────────────────
	var dtype poly.DType
	switch int(precisionInt) {
	case 32:
		dtype = poly.DTypeFloat32
	case 8:
		dtype = poly.DTypeInt8
	default:
		dtype = poly.DTypeInt4
	}
	useGPU := int(useGPUInt) == 1

	// ── Tiling ────────────────────────────────────────────────────────────────
	useTiling := int(execMode) != 1
	tileSize := -1
	if !useTiling {
		tileSize = 0
	}
	multiCore := int(execMode) == 3

	// Large model guards (mirrors glitch.go)
	if useGPU && useTiling && hiddenSize >= 1536 {
		useTiling = false
		tileSize = 0
		multiCore = false
	}
	if useGPU && hiddenSize >= 1536 && dtype == poly.DTypeInt4 {
		dtype = poly.DTypeInt8
	}

	// ── Build network ─────────────────────────────────────────────────────────
	net := poly.NewVolumetricNetwork(1, 1, 1, numLayers*4)
	for b := 0; b < numLayers; b++ {
		base := b * 4
		mhaSize := (2*hiddenSize*hiddenSize) + (2*hiddenSize*(numKVHeads*headDim)) + (2*hiddenSize) + (2*(numKVHeads*headDim))
		mlpSize := (3*hiddenSize*intermediateSize) + (2*intermediateSize) + hiddenSize

		l0 := &net.Layers[base+0]
		l0.Type = poly.LayerRMSNorm
		l0.InputHeight = hiddenSize
		l0.OutputHeight = hiddenSize
		l0.WeightStore = poly.NewWeightStore(hiddenSize)

		l1 := &net.Layers[base+1]
		l1.Type = poly.LayerMultiHeadAttention
		l1.DModel = hiddenSize
		l1.NumHeads = numHeads
		l1.NumKVHeads = numKVHeads
		l1.HeadDim = headDim
		l1.RoPEFreqBase = ropeFreqBase
		l1.WeightStore = poly.NewWeightStore(mhaSize)

		l2 := &net.Layers[base+2]
		l2.Type = poly.LayerRMSNorm
		l2.InputHeight = hiddenSize
		l2.OutputHeight = hiddenSize
		l2.WeightStore = poly.NewWeightStore(hiddenSize)

		l3 := &net.Layers[base+3]
		l3.Type = poly.LayerSwiGLU
		l3.InputHeight = hiddenSize
		l3.OutputHeight = intermediateSize
		l3.WeightStore = poly.NewWeightStore(mlpSize)
	}
	poly.LoadWithPrefixes(net, allTensors)

	// ── Template ──────────────────────────────────────────────────────────────
	modelName := ""
	if v, ok := config["_name_or_path"]; ok {
		if s, ok := v.(string); ok {
			modelName = s
		}
	}
	template := llmTemplateFor(modelName)

	// ── Transformer ───────────────────────────────────────────────────────────
	tr := poly.NewTransformer[float32](net, embeddings, lmHead, finalNorm, template)
	if useTiling {
		tr.EnableTiling(tileSize)
	}
	net.EnableMultiCoreTiling = multiCore

	if useGPU {
		if initErr := net.InitWGPU(); initErr == nil {
			for i := range net.Layers {
				if net.Layers[i].Type == poly.LayerRMSNorm {
					net.Layers[i].DType = poly.DTypeFloat32
				} else {
					net.Layers[i].DType = dtype
				}
				net.Layers[i].SyncToGPU()
			}
			tr.SyncToGPU()
			// Warmup
			_, _ = tr.ForwardTokenIDsWGPU([]uint32{0}, nil, true, true)
			tr.Reset()
		}
	}

	// ── Special token mask ────────────────────────────────────────────────────
	banned := llmBuildBanned(tk, eosTokens)

	// ── Register ──────────────────────────────────────────────────────────────
	state := &llmState{
		tr:            tr,
		tk:            tk,
		eosTokens:     eosTokens,
		bannedTokens:  banned,
		deterministic: int(deterministicInt) == 1,
	}

	llmMu.Lock()
	id := llmNextID
	llmNextID++
	llmStates[id] = state
	llmMu.Unlock()

	return C.longlong(id)
}

// ─── LoomLLMGenerate ─────────────────────────────────────────────────────────
// systemPromptC : system prompt (empty string = use default)
// userMsgC      : the user's message
// temperature   : 0 = greedy, 0.7 = default
// topK          : top-K sampling, 40 default
// maxTokens     : max tokens to generate, 128 default
// Returns JSON: {"response":"...","prefill_tps":...,"decode_tps":...,"total_tps":...,"ram_mb":...,"vram_mb":...}

//export LoomLLMGenerate
func LoomLLMGenerate(handle C.longlong, systemPromptC *C.char, userMsgC *C.char, temperatureF C.float, topKI C.int, maxTokensI C.int) *C.char {
	llmMu.Lock()
	state, ok := llmStates[int64(handle)]
	llmMu.Unlock()
	if !ok {
		return C.CString(`{"error":"invalid LLM handle"}`)
	}

	systemPrompt := C.GoString(systemPromptC)
	userMsg := C.GoString(userMsgC)
	temp := float32(temperatureF)
	topK := int(topKI)
	maxTok := int(maxTokensI)
	if maxTok <= 0 {
		maxTok = 128
	}
	if topK <= 0 {
		topK = 40
	}

	if state.deterministic {
		temp = 0
	}

	opts := poly.GenOptions{
		MaxTokens:             maxTok,
		MinTokens:             8,
		Temperature:           temp,
		TopK:                  topK,
		Deterministic:         state.deterministic,
		EOSTokens:             state.eosTokens,
		BannedTokens:          state.bannedTokens,
		RepetitionPenalty:     1.1,
		RepetitionWindow:      64,
		MaxConsecutiveRepeats: 3,
		NoRepeatNGram:         3,
		Silent:                true,
	}

	encode := func(text string) []uint32 { return state.tk.Encode(text, false) }
	decode := func(tokens []uint32) string { return state.tk.Decode(tokens, false) }

	reply, metrics := state.tr.Generate(encode, decode, state.history, systemPrompt, userMsg, opts)
	state.history = append(state.history, poly.Turn{User: userMsg, Assistant: reply})

	result := map[string]interface{}{
		"response":    reply,
		"prefill_tps": metrics.PrefillTokPerSec,
		"decode_tps":  metrics.DecodeTokPerSec,
		"total_tps":   metrics.TotalTokPerSec,
		"ram_mb":      metrics.RAMUsageMB,
		"vram_mb":     metrics.VRAMUsageMB,
	}
	data, _ := json.Marshal(result)
	return C.CString(string(data))
}

// ─── LoomLLMStartGenerate ────────────────────────────────────────────────────
// Non-blocking: starts generation in a goroutine.
// Call LoomLLMPollToken repeatedly to receive tokens.

//export LoomLLMStartGenerate
func LoomLLMStartGenerate(handle C.longlong, systemPromptC *C.char, userMsgC *C.char, temperatureF C.float, topKI C.int, maxTokensI C.int) {
	llmMu.Lock()
	state, ok := llmStates[int64(handle)]
	llmMu.Unlock()
	if !ok {
		return
	}

	systemPrompt := C.GoString(systemPromptC)
	userMsg := C.GoString(userMsgC)
	temp := float32(temperatureF)
	topK := int(topKI)
	maxTok := int(maxTokensI)
	if maxTok <= 0 {
		maxTok = 128
	}
	if topK <= 0 {
		topK = 40
	}
	if state.deterministic {
		temp = 0
	}

	// Reset streaming state
	state.streamMu.Lock()
	state.tokenQueue = state.tokenQueue[:0]
	state.streamDone = false
	state.streamResult = nil
	state.pendingUser = userMsg
	state.streamMu.Unlock()

	opts := poly.GenOptions{
		MaxTokens:             maxTok,
		MinTokens:             8,
		Temperature:           temp,
		TopK:                  topK,
		Deterministic:         state.deterministic,
		EOSTokens:             state.eosTokens,
		BannedTokens:          state.bannedTokens,
		RepetitionPenalty:     1.1,
		RepetitionWindow:      64,
		MaxConsecutiveRepeats: 3,
		NoRepeatNGram:         3,
		Silent:                true,
		StreamCallback: func(token string) {
			state.streamMu.Lock()
			state.tokenQueue = append(state.tokenQueue, token)
			state.streamMu.Unlock()
		},
	}

	encode := func(text string) []uint32 { return state.tk.Encode(text, false) }
	decode := func(tokens []uint32) string { return state.tk.Decode(tokens, false) }
	history := append([]poly.Turn{}, state.history...) // snapshot

	go func() {
		reply, metrics := state.tr.Generate(encode, decode, history, systemPrompt, userMsg, opts)

		state.streamMu.Lock()
		state.streamDone = true
		state.streamResult = &streamResult{metrics: metrics, reply: reply}
		state.streamMu.Unlock()

		// Append to history
		llmMu.Lock()
		state.history = append(state.history, poly.Turn{User: userMsg, Assistant: reply})
		llmMu.Unlock()
	}()
}

// ─── LoomLLMPollToken ────────────────────────────────────────────────────────
// Returns JSON: {"s":0,"t":"token"} | {"s":1,...stats...} | {"s":2}
// s=0: token chunk, s=1: done+stats, s=2: empty (still generating)

//export LoomLLMPollToken
func LoomLLMPollToken(handle C.longlong) *C.char {
	llmMu.Lock()
	state, ok := llmStates[int64(handle)]
	llmMu.Unlock()
	if !ok {
		return C.CString(`{"s":1}`)
	}

	state.streamMu.Lock()
	defer state.streamMu.Unlock()

	// Drain one token from the queue
	if len(state.tokenQueue) > 0 {
		tok := state.tokenQueue[0]
		state.tokenQueue = state.tokenQueue[1:]
		data, _ := json.Marshal(map[string]interface{}{"s": pollToken, "t": tok})
		return C.CString(string(data))
	}

	// Queue empty — check if done
	if state.streamDone && state.streamResult != nil {
		m := state.streamResult.metrics
		data, _ := json.Marshal(map[string]interface{}{
			"s":           pollDone,
			"prefill_tps": m.PrefillTokPerSec,
			"decode_tps":  m.DecodeTokPerSec,
			"total_tps":   m.TotalTokPerSec,
			"ram_mb":      m.RAMUsageMB,
			"vram_mb":     m.VRAMUsageMB,
		})
		// Clear so we only report done once
		state.streamDone = false
		state.streamResult = nil
		return C.CString(string(data))
	}

	return C.CString(`{"s":2}`)
}

// ─── LoomLLMResetHistory ──────────────────────────────────────────────────────

//export LoomLLMResetHistory
func LoomLLMResetHistory(handle C.longlong) {
	llmMu.Lock()
	if state, ok := llmStates[int64(handle)]; ok {
		state.history = nil
		state.tr.Reset()
	}
	llmMu.Unlock()
}

// ─── LoomFreeLLM ──────────────────────────────────────────────────────────────

//export LoomFreeLLM
func LoomFreeLLM(handle C.longlong) {
	llmMu.Lock()
	delete(llmStates, int64(handle))
	llmMu.Unlock()
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

func parseInt(s string) (int, bool) {
	var n int
	_, err := parseIntHelper(s, &n)
	return n, err == nil
}

func parseIntHelper(s string, out *int) (int, error) {
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			return 0, &parseErr{}
		}
		n = n*10 + int(c-'0')
	}
	*out = n
	return n, nil
}

type parseErr struct{}

func (e *parseErr) Error() string { return "parse error" }

func intCfg(cfg map[string]interface{}, key string, def int) int {
	if v, ok := cfg[key]; ok {
		if f, ok := v.(float64); ok {
			return int(f)
		}
	}
	return def
}

func floatCfg(cfg map[string]interface{}, key string, def float64) float64 {
	if v, ok := cfg[key]; ok {
		if f, ok := v.(float64); ok {
			return f
		}
	}
	return def
}

func llmLoadEOS(config map[string]interface{}) []int {
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

func llmBuildBanned(tk *poly.Tokenizer, eosTokens []int) []int {
	if tk == nil {
		return nil
	}
	eosSet := make(map[int]struct{}, len(eosTokens))
	for _, t := range eosTokens {
		eosSet[t] = struct{}{}
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

func llmTemplateFor(modelName string) poly.Template {
	name := strings.ToLower(modelName)
	if strings.Contains(name, "llama-3") || strings.Contains(name, "smollm3") {
		return poly.Llama3
	}
	return poly.ChatML
}
