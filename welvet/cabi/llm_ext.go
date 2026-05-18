package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
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
	execMu        sync.Mutex // Ensures generation requests are executed sequentially
	eosTokens     []int
	bannedTokens  []int
	history       []poly.Turn
	deterministic bool
	// Microsoft BitNet b1.58 chat template parity with loom/lucy (tokenizer.encode add_special).
	encodeAddSpecial bool

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

	// gpuWorkerChan serialises all GPU operations onto a single OS thread,
	// mirroring glitch.go where everything runs on the main goroutine.
	gpuWorkerChan chan func()
	gpuWorkerOnce sync.Once
)

// runOnGPUThread dispatches fn to a single goroutine locked to one OS thread.
// This guarantees the same OS thread is used for every wgpu call, which
// prevents the "Parent device is lost" panic that occurs when the device is
// initialised on one thread but submitted from another.
func runOnGPUThread(fn func()) {
	gpuWorkerOnce.Do(func() {
		gpuWorkerChan = make(chan func(), 16)
		go func() {
			runtime.LockOSThread()
			for f := range gpuWorkerChan {
				f()
			}
		}()
	})
	done := make(chan struct{})
	gpuWorkerChan <- func() { fn(); close(done) }
	<-done
}

func llmDetectBitNetModel(config map[string]interface{}, snapshotDir string) bool {
	snap := strings.ToLower(snapshotDir)
	if strings.Contains(snap, "bitnet") || strings.Contains(snap, "1bit") {
		return true
	}
	if strings.EqualFold(poly.HFConfigStringDefault(config, "model_type", ""), "bitnet") {
		return true
	}
	if arch, ok := config["architectures"].([]interface{}); ok {
		for _, a := range arch {
			if s, ok := a.(string); ok && strings.Contains(strings.ToLower(s), "bitnet") {
				return true
			}
		}
	}
	return false
}

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

// ─── LoomLLMListInstalledModels ───────────────────────────────────────────────
// Returns JSON array of {"id":"org/model","snapshot_dir":"/abs/path"} for SoulGlitch.

//export LoomLLMListInstalledModels
func LoomLLMListInstalledModels(hubDirC *C.char) *C.char {
	hubDir := C.GoString(hubDirC)
	list, err := poly.HFListInstalledModels(hubDir)
	if err != nil {
		return C.CString(`[]`)
	}
	data, _ := json.Marshal(list)
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

	eosTokens := poly.EOSTokenIDsFromHFConfig(config)

	// ── Load safetensors / globals (Lucy parity: BitNet needs HFStored + CPU pack) ─
	safetensorFiles, _ := filepath.Glob(filepath.Join(snapshotDir, "*.safetensors"))
	if len(safetensorFiles) == 0 {
		return -1
	}

	isBitNetModel := llmDetectBitNetModel(config, snapshotDir)
	useGPU := int(useGPUInt) == 1
	if isBitNetModel && useGPU {
		fmt.Println("ℹ️  BitNet: SoulGlitch welvet uses CPU packed ternary inference (loom/lucy parity). Forcing CPU.")
		useGPU = false
	}
	// Block-wise GPU upload path only when we actually run on GPU.
	sequentialGPULoad := useGPU && int(useGPUInt) == 1
	useBitNetPacked := isBitNetModel && !useGPU

	mapper := poly.NewPrefixWeightMapper()
	var embeddings, lmHead, finalNorm []float32
	var allTensors map[string][]float32

	if (sequentialGPULoad && useGPU) || useBitNetPacked {
		globalStored := make(map[string]poly.HFStoredTensor)
		for _, f := range safetensorFiles {
			part, err := poly.LoadSafetensorsSelectiveRaw(f, poly.HFWeightIsGlobal)
			if err != nil {
				return -1
			}
			for k, v := range part {
				globalStored[k] = v
			}
		}
		embeddings, lmHead, finalNorm, _ = mapper.MapWeightsFromStored(globalStored)
		poly.ReleaseTransientHFStoredMap(globalStored)
		runtime.GC()
		debug.FreeOSMemory()
	} else {
		allTensors = make(map[string][]float32)
		for _, f := range safetensorFiles {
			t, err := poly.LoadSafetensors(f)
			if err != nil {
				continue
			}
			for k, v := range t {
				allTensors[k] = v
			}
		}
		embeddings, lmHead, finalNorm, _ = mapper.MapWeights(allTensors)
	}

	numLayers := 0
	if nl, ok := poly.HFConfigInt(config, "num_hidden_layers"); ok {
		numLayers = nl
	}
	if numLayers == 0 {
		maxLi := poly.MaxHFWeightLayerIndexInSafetensorsFiles(safetensorFiles)
		if maxLi >= 0 {
			numLayers = maxLi + 1
		}
	}
	if numLayers == 0 {
		return -1
	}

	numHeads := poly.HFConfigIntDefault(config, "num_attention_heads", 1)
	numKVHeads := numHeads
	if v, ok := poly.HFConfigInt(config, "num_key_value_heads"); ok {
		numKVHeads = v
	}
	hiddenSize := len(finalNorm)
	if hiddenSize == 0 {
		hiddenSize = poly.HFConfigIntDefault(config, "hidden_size", 0)
	}
	if hiddenSize == 0 || numHeads == 0 {
		return -1
	}
	headDim := hiddenSize / numHeads
	if v, ok := poly.HFConfigInt(config, "head_dim"); ok {
		headDim = v
	}
	queryDim := numHeads * headDim
	kvDim := numKVHeads * headDim

	intermediateSize := poly.HFConfigIntDefault(config, "intermediate_size", hiddenSize*4)
	rmsNormEps := poly.HFConfigFloat64Default(config, "rms_norm_eps", 1e-6)
	ropeFreqBase := poly.HFConfigFloat64Default(config, "rope_theta", 10000.0)

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

	activation := poly.ActivationSilu
	if strings.EqualFold(poly.HFConfigStringDefault(config, "hidden_act", ""), "relu2") {
		activation = poly.ActivationReLU2
	}

	net := poly.NewVolumetricNetwork(1, 1, 1, numLayers*4)
	poly.InitHFDecoderBlocks(net, poly.HFDecoderDims{
		NumLayers:          numLayers,
		HiddenSize:         hiddenSize,
		NumHeads:           numHeads,
		NumKVHeads:         numKVHeads,
		HeadDim:            headDim,
		QueryDim:           queryDim,
		KVDim:              kvDim,
		IntermediateSize:   intermediateSize,
		RMSNormEps:         rmsNormEps,
		RoPEFreqBase:       ropeFreqBase,
		Activation:         activation,
	})

	if useBitNetPacked {
		fmt.Printf("⏳ BitNet CPU block-wise load + pack (%d transformer blocks)...\n", numLayers)
		layerFiles := buildLayerShardIndex(safetensorFiles, numLayers)
		for li := 0; li < numLayers; li++ {
			layerMap := make(map[string]poly.HFStoredTensor)
			for _, sf := range layerFiles[li] {
				part, err := poly.LoadSafetensorsSelectiveRaw(sf, func(k string) bool {
					return poly.HFWeightMatchesLayer(k, li)
				})
				if err != nil {
					fmt.Printf("❌ BitNet safetensors %s block %d: %v\n", sf, li, err)
					return -1
				}
				for k, v := range part {
					layerMap[k] = v
				}
			}
			poly.LoadWithPrefixesFromHFStored(net, layerMap)
			if err := poly.PrepareDecoderBlockBitNetTernaryCPU(net, li); err != nil {
				fmt.Printf("❌ BitNet CPU preparation failed for block %d: %v\n", li, err)
				return -1
			}
			poly.ReleaseTransientHFStoredMap(layerMap)
			if li == 0 || (li+1)%4 == 0 || li+1 == numLayers {
				runtime.GC()
				debug.FreeOSMemory()
			}
		}
		net.UseExactDType = true
		runtime.GC()
		debug.FreeOSMemory()
	} else if !sequentialGPULoad {
		poly.LoadWithPrefixes(net, allTensors)
		poly.ReleaseTransientSafetensorMap(allTensors, embeddings, lmHead, finalNorm)
	}

	modelName := ""
	if v, ok := config["_name_or_path"]; ok {
		if s, ok := v.(string); ok {
			modelName = s
		}
	}
	template := poly.TemplateForHFModelID(modelName)

	// ── Transformer ───────────────────────────────────────────────────────────
	tr := poly.NewTransformer[float32](net, embeddings, lmHead, finalNorm, template)
	tr.SetRMSNormEps(rmsNormEps)
	// Keep GPU KV-cache reservation bounded for app usage; transformer defaults
	// layers to 2048 which can inflate VRAM on smaller models.
	for i := range tr.Network.Layers {
		tr.Network.Layers[i].MaxSeqLen = 512
	}
	if useTiling {
		tr.EnableTiling(tileSize)
	}
	// Match loom/glitch applyGlitchTilingFlags: GPU uses MC tiles only when execMode==3; CPU uses tiled flag whenever tiling is on.
	if useGPU {
		tr.Network.EnableMultiCoreTiling = useTiling && multiCore
	} else {
		tr.Network.EnableMultiCoreTiling = useTiling
	}

	if useGPU {
		// Run ALL GPU setup on the dedicated GPU OS thread so the device is
		// created and warmed-up on the same thread that will serve every
		// subsequent inference call (mirrors glitch.go's main goroutine).
		runOnGPUThread(func() {
			poly.Alog("SOULGLITCH: Requesting GPU initialization for VolumetricNetwork...")
			if sequentialGPULoad {
				fmt.Printf("⏳ GPU init + block-wise weight upload (%d transformer blocks)...\n", numLayers)
			} else {
				fmt.Print("⏳ GPU Synchronization... ")
			}
			if err := tr.Network.InitWGPU(); err != nil {
				fmt.Printf("❌ Failed: %v\n", err)
				poly.Alog(fmt.Sprintf("SOULGLITCH: ❌ InitWGPU Failed: %v. Falling back to CPU.", err))
			} else {
				if sequentialGPULoad {
					layerFiles := buildLayerShardIndex(safetensorFiles, numLayers)
					for li := 0; li < numLayers; li++ {
						layerMap := make(map[string][]float32)
						for _, sf := range layerFiles[li] {
							part, err := poly.LoadSafetensorsSelective(sf, func(k string) bool {
								return poly.HFWeightMatchesLayer(k, li)
							})
							if err != nil {
								continue
							}
							for k, v := range part {
								layerMap[k] = v
							}
						}
						poly.LoadWithPrefixes(net, layerMap)
						layerMap = nil
						if li == 0 || (li+1)%4 == 0 || li+1 == numLayers {
							runtime.GC()
							debug.FreeOSMemory()
						}
						base := li * 4
						for j := 0; j < 4; j++ {
							idx := base + j
							layer := &tr.Network.Layers[idx]
							if layer.Type == poly.LayerRMSNorm {
								layer.DType = poly.DTypeFloat32
							} else {
								layer.DType = dtype
							}
							if err := layer.SyncToGPU(); err != nil {
								fmt.Printf("❌ GPU sync block %d layer %d: %v\n", li, j, err)
							}
						}
						for j := 0; j < 4; j++ {
							(&tr.Network.Layers[base+j]).ReleaseInferenceHostWeights()
						}
					}
				} else {
					// Mirror glitch.go exactly: Iterate layers, set DType, SyncToGPU
					for i := range tr.Network.Layers {
						if tr.Network.Layers[i].Type == poly.LayerRMSNorm {
							tr.Network.Layers[i].DType = poly.DTypeFloat32
						} else {
							tr.Network.Layers[i].DType = dtype
						}
						(&tr.Network.Layers[i]).SyncToGPU()
					}
				}
				tr.SyncToGPU()

				// Warmup pass to compile WGPU Shaders before first chat!
				poly.Alog("SOULGLITCH: Performing GPU Warmup Pass...")
				_, _ = tr.ForwardTokenIDsWGPU([]uint32{0}, nil, true, true)
				tr.Reset()
				tr.ReleaseInferenceHostWeights()
				runtime.GC()
				debug.FreeOSMemory()

				fmt.Println("✅ Success!")
				poly.Alog("SOULGLITCH: GPU Pipeline Ready.")
			}
		})
	}
	if !useGPU {
		if useBitNetPacked {
			tr.Network.UseExactDType = true
		}
		tr.SyncInferenceCPU()
	}

	banned := poly.TokenizerBannedSpecialExceptEOS(tk, eosTokens)

	encodeAddSpecial := isBitNetModel &&
		strings.Contains(strings.ToLower(snapshotDir), "bitnet-b1.58")

	// ── Register ──────────────────────────────────────────────────────────────
	state := &llmState{
		tr:               tr,
		tk:               tk,
		eosTokens:        eosTokens,
		bannedTokens:     banned,
		deterministic:    int(deterministicInt) == 1,
		encodeAddSpecial: encodeAddSpecial,
	}

	llmMu.Lock()
	id := llmNextID
	llmNextID++
	llmStates[id] = state
	llmMu.Unlock()

	return C.longlong(id)
}

func buildLayerShardIndex(safetensorFiles []string, numLayers int) [][]string {
	layerFiles := make([][]string, numLayers)
	if numLayers <= 0 {
		return layerFiles
	}
	for _, sf := range safetensorFiles {
		names, err := poly.SafetensorsTensorNames(sf)
		if err != nil {
			for li := 0; li < numLayers; li++ {
				layerFiles[li] = append(layerFiles[li], sf)
			}
			continue
		}
		seen := make(map[int]struct{})
		for _, n := range names {
			if li, ok := poly.HFWeightLayerIndex(n); ok && li >= 0 && li < numLayers {
				seen[li] = struct{}{}
			}
		}
		for li := range seen {
			layerFiles[li] = append(layerFiles[li], sf)
		}
	}
	for li := 0; li < numLayers; li++ {
		if len(layerFiles[li]) == 0 {
			layerFiles[li] = append(layerFiles[li], safetensorFiles...)
		}
	}
	return layerFiles
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

	// Use MinTokens=1 for short calls (≤8 tokens) to allow early EOS — mirrors
	// glitch.go's voteOpts where MinTokens=1. For longer calls keep MinTokens=8.
	minTok := 8
	if maxTok <= 8 {
		minTok = 1
	}

	opts := poly.GenOptions{
		MaxTokens:             maxTok,
		MinTokens:             minTok,
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

	encode := func(text string) []uint32 { return state.tk.Encode(text, state.encodeAddSpecial) }
	decode := func(tokens []uint32) string { return state.tk.Decode(tokens, false) }

	var reply string
	var metrics poly.GenMetrics
	if state.tr.Network.UseGPU {
		runOnGPUThread(func() {
			reply, metrics = state.tr.Generate(encode, decode, state.history, systemPrompt, userMsg, opts)
		})
	} else {
		reply, metrics = state.tr.Generate(encode, decode, state.history, systemPrompt, userMsg, opts)
	}
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

	minTokS := 8
	if maxTok <= 8 {
		minTokS = 1
	}

	opts := poly.GenOptions{
		MaxTokens:             maxTok,
		MinTokens:             minTokS,
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

	encode := func(text string) []uint32 { return state.tk.Encode(text, state.encodeAddSpecial) }
	decode := func(tokens []uint32) string { return state.tk.Decode(tokens, false) }
	history := append([]poly.Turn{}, state.history...) // snapshot

	go func() {
		state.execMu.Lock()
		defer state.execMu.Unlock()

		var reply string
		var metrics poly.GenMetrics
		if state.tr.Network.UseGPU {
			runOnGPUThread(func() {
				reply, metrics = state.tr.Generate(encode, decode, history, systemPrompt, userMsg, opts)
			})
		} else {
			reply, metrics = state.tr.Generate(encode, decode, history, systemPrompt, userMsg, opts)
		}

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

