package main

// Lucy Bloom Rivers — HuggingFace cache Poly Talk interactive LLM mode (HF load, GPU/tiling, chat).
// Shared globals live in support.go.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strings"

	"github.com/openfluke/loom/poly"
)

func runHuggingFaceMode(reader *bufio.Reader) {
	hubDir, models, err := poly.HFInventoryMergedModels()
	if err != nil {
		log.Fatalf("Could not scan HuggingFace cache: %v", err)
	}

	if len(models) == 0 {
		log.Fatalf("No models found in HuggingFace cache at: %s", hubDir)
	}

	fmt.Println("\n⚛️  Poly Talk - Available models:")
	for i, model := range models {
		fmt.Printf("  [%d] %s\n", i+1, model)
	}

	detInput := readInput(reader, "🎯 Deterministic mode? (1=yes / 0=no) [0]: ", "0")
	deterministic = detInput == "1"

	useTiling := true
	tileSize := -1 // auto-detect

	var useGPU bool
	fmt.Print("🎮 Enable GPU Acceleration? (1=yes / 0=no) [0]: ")
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)
	useGPU = input == "1"

	fmt.Println("\n🚀 Select Execution Mode:")
	fmt.Println("  [1] Tiled — GPU: single-workgroup; CPU: multi-core tiled")
	fmt.Println("  [2] Tiled — GPU: multi-workgroup; CPU: multi-core tiled")
	execModeInput := readInput(reader, "Choice [2]: ", "2")

	var tilingMode string
	tilingMode, tileSize = parseLLMExecutionMode(execModeInput)

	if useGPU {
		fmt.Print("💎 Weight Precision? (4=Q4_0 / 8=INT8 / 32=FP32) [4]: ")
		input, _ = reader.ReadString('\n')
		input = strings.TrimSpace(input)
		if input == "32" {
			weightDType = poly.DTypeFloat32
		} else if input == "8" {
			weightDType = poly.DTypeInt8
		} else {
			weightDType = poly.DTypeInt4
		}
	}

	sequentialGPULoad := false
	if useGPU {
		sequentialGPULoad = readInput(reader, "📥 Load weights block-by-block into GPU (lower peak host RAM; skips holding full checkpoint map)? (1=yes / 0=no) [0]: ", "0") == "1"
	}

	modelInput := readInput(reader, "\nSelect model number: ", "1")
	var selectedIdx int
	fmt.Sscanf(modelInput, "%d", &selectedIdx)
	if selectedIdx < 1 || selectedIdx > len(models) {
		log.Fatalf("Invalid model selection: %d", selectedIdx)
	}
	modelName := models[selectedIdx-1]
	modelNameLower := strings.ToLower(modelName)
	isQwen := strings.Contains(modelNameLower, "qwen")
	isBitNetModel := strings.Contains(modelNameLower, "bitnet") || strings.Contains(modelNameLower, "1bit")
	useBitNetCPU := false
	useTernaryPTQCPU := false
	if !useGPU {
		if isBitNetModel {
			useBitNetCPU = true
			fmt.Println("🧮 BitNet model detected; enabling CPU packed ternary inference.")
		} else {
			quantInput := readInput(reader, "🧮 CPU weight precision? (32=FP32 / ternary=experimental PTQ) [32]: ", "32")
			switch strings.ToLower(strings.TrimSpace(quantInput)) {
			case "ternary", "t", "bitnet", "1bit", "b1.58", "158":
				useTernaryPTQCPU = true
				fmt.Println("⚠️  Ternary PTQ is experimental. It is not equivalent to BitNet training and may produce bad text.")
			}
		}
	}
	template := templateForModel(modelName)
	activeSystemPrompt := defaultSystemPromptForModel(modelName)
	if deterministic && isQwen {
		fmt.Println("⚠️  Qwen deterministic=1 can leak planning text. Keeping deterministic=1 because you explicitly selected it.")
	}
	if deterministic && strings.Contains(strings.ToLower(modelName), "instruct") && strings.Contains(strings.ToLower(modelName), "1.7b") {
		fmt.Println("⚠️  Deterministic=1 can collapse into punctuation-only outputs on 1.7B instruct models. Use Deterministic=0 for normal chat quality.")
	}

	snapshotDir, err := poly.HFResolveSnapshotDir(hubDir, modelName)
	if err != nil {
		log.Fatalf("❌ %v", err)
	}

	// Tokenizer
	tokenizerPath := filepath.Join(snapshotDir, "tokenizer.json")
	tk, err = poly.LoadTokenizer(tokenizerPath)
	if err != nil {
		log.Fatalf("⚠️  Tokenizer failure: %v", err)
	}

	configPath := filepath.Join(snapshotDir, "config.json")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		log.Fatalf("⚠️  config.json: %v", err)
	}
	var config map[string]interface{}
	if err := json.Unmarshal(configData, &config); err != nil {
		log.Fatalf("⚠️  config parse: %v", err)
	}
	eosTokens = poly.LoadEOSTokenIDsFromConfigPath(configPath)
	eosTokens = mergeIntSets(eosTokens, loadEOSTokensFromJSON(filepath.Join(snapshotDir, "generation_config.json")))

	safetensorFiles, _ := filepath.Glob(filepath.Join(snapshotDir, "*.safetensors"))
	if len(safetensorFiles) == 0 {
		log.Fatalf("No .safetensors files in %s", snapshotDir)
	}

	mapper := poly.NewPrefixWeightMapper()
	var embeddings, lmHead, finalNorm []float32
	var allTensors map[string][]float32

	if (sequentialGPULoad && useGPU) || useBitNetCPU {
		globalTensors := make(map[string][]float32)
		for _, f := range safetensorFiles {
			part, err := poly.LoadSafetensorsSelective(f, poly.HFWeightIsGlobal)
			if err != nil {
				log.Fatalf("⚠️  safetensors %s: %v", f, err)
			}
			for k, v := range part {
				globalTensors[k] = v
			}
		}
		embeddings, lmHead, finalNorm, _ = mapper.MapWeights(globalTensors)
		embeddings, lmHead, finalNorm = poly.CloneMappedGlobalWeights(embeddings, lmHead, finalNorm)
		globalTensors = nil
		runtime.GC()
		debug.FreeOSMemory()
	} else {
		allTensors = make(map[string][]float32)
		for _, f := range safetensorFiles {
			t, err := poly.LoadSafetensors(f)
			if err != nil {
				log.Fatalf("⚠️  safetensors %s: %v", f, err)
			}
			for k, v := range t {
				allTensors[k] = v
			}
		}
		embeddings, lmHead, finalNorm, _ = mapper.MapWeights(allTensors)
	}

	numHeads, ok := poly.HFConfigInt(config, "num_attention_heads")
	if !ok {
		log.Fatalf("config.json missing num_attention_heads")
	}
	numKVHeads := numHeads
	if v, ok := poly.HFConfigInt(config, "num_key_value_heads"); ok {
		numKVHeads = v
	}

	hiddenSize, hsOk := poly.HFConfigInt(config, "hidden_size")
	if !hsOk && finalNorm != nil {
		hiddenSize = len(finalNorm)
		hsOk = true
	}
	if !hsOk || hiddenSize <= 0 {
		log.Fatalf("Could not determine hidden size (need hidden_size in config or final norm weights)")
	}

	headDim := hiddenSize / numHeads
	if v, ok := poly.HFConfigInt(config, "head_dim"); ok {
		headDim = v
	}
	queryDim := numHeads * headDim
	kvDim := numKVHeads * headDim

	intermediateSize, ok := poly.HFConfigInt(config, "intermediate_size")
	if !ok {
		log.Fatalf("config.json missing intermediate_size")
	}

	numLayers, nlOk := poly.HFConfigInt(config, "num_hidden_layers")
	if !nlOk {
		maxLi := poly.MaxHFWeightLayerIndexInSafetensorsFiles(safetensorFiles)
		if maxLi < 0 {
			log.Fatalf("Could not determine layer count (need num_hidden_layers or recognizable layer tensors)")
		}
		numLayers = maxLi + 1
	}

	rmsNormEps := poly.HFConfigFloat64Default(config, "rms_norm_eps", 1e-6)
	ropeFreqBase := poly.HFConfigFloat64Default(config, "rope_theta", 10000.0)
	activation := poly.ActivationSilu
	if strings.EqualFold(poly.HFConfigStringDefault(config, "hidden_act", ""), "relu2") {
		activation = poly.ActivationReLU2
	}
	if useGPU && useTiling && hiddenSize >= 1536 {
		fmt.Printf("⚠️  Large model detected (hidden=%d). Tiled GPU path can destabilize logits here; forcing Standard Forward.\n", hiddenSize)
		useTiling = false
		tilingMode = "1"
		tileSize = 0
	}
	if useGPU && hiddenSize >= 1536 && weightDType == poly.DTypeInt4 {
		fmt.Printf("⚠️  Large model detected (hidden=%d). Q4 can degrade output quality; promoting weight precision to INT8.\n", hiddenSize)
		weightDType = poly.DTypeInt8
	}
	if useGPU && isQwen && weightDType == poly.DTypeInt4 {
		fmt.Println("⚠️  Qwen GPU + Q4 is experimental and may reduce output quality.")
	}
	if useGPU && isQwen && weightDType == poly.DTypeInt8 {
		fmt.Println("ℹ️  Qwen GPU + INT8 enabled.")
	}
	if useGPU && isQwen && useTiling {
		fmt.Println("⚠️  Qwen GPU tiled path is experimental and may reduce output quality.")
	}
	if useGPU && queryDim != hiddenSize {
		fmt.Printf("ℹ️  Model uses expanded attention query dim (q=%d, hidden=%d). Enabling Qwen-compatible GPU MHA path.\n", queryDim, hiddenSize)
	}

	net := poly.NewVolumetricNetwork(1, 1, 1, numLayers*4)
	poly.InitHFDecoderBlocks(net, poly.HFDecoderDims{
		NumLayers:        numLayers,
		HiddenSize:       hiddenSize,
		NumHeads:         numHeads,
		NumKVHeads:       numKVHeads,
		HeadDim:          headDim,
		QueryDim:         queryDim,
		KVDim:            kvDim,
		IntermediateSize: intermediateSize,
		RMSNormEps:       rmsNormEps,
		RoPEFreqBase:     ropeFreqBase,
		Activation:       activation,
	})

	if useBitNetCPU {
		fmt.Printf("⏳ BitNet CPU block-wise load + pack (%d transformer blocks)...\n", numLayers)
		layerFiles := buildLayerShardIndex(safetensorFiles, numLayers)
		for li := 0; li < numLayers; li++ {
			layerMap := make(map[string][]float32)
			for _, sf := range layerFiles[li] {
				part, err := poly.LoadSafetensorsSelective(sf, func(k string) bool {
					return poly.HFWeightMatchesLayer(k, li)
				})
				if err != nil {
					log.Fatalf("⚠️  safetensors %s: %v", sf, err)
				}
				for k, v := range part {
					layerMap[k] = v
				}
			}
			poly.LoadWithPrefixes(net, layerMap)
			if err := poly.PrepareDecoderBlockBitNetTernaryCPU(net, li); err != nil {
				log.Fatalf("❌ BitNet CPU preparation failed for block %d: %v", li, err)
			}
			poly.ReleaseTransientSafetensorMap(layerMap)
			fmt.Printf("   ✓ Block %d/%d packed\n", li+1, numLayers)
		}
		runtime.GC()
		debug.FreeOSMemory()
	} else if !(sequentialGPULoad && useGPU) {
		poly.LoadWithPrefixes(net, allTensors)
	}
	if useBitNetCPU {
		if isBitNetModel {
			fmt.Print("🧮 BitNet b1.58 packed CPU weights ready (FP32 projections released)... ")
		}
		net.UseExactDType = true
		fmt.Println("done.")
	}
	if useTernaryPTQCPU {
		fmt.Print("🧮 Quantizing FP32 transformer weights to experimental ternary PTQ... ")
		if err := poly.MorphNetworkBitNetTernary(net); err != nil {
			log.Fatalf("❌ Ternary PTQ failed: %v", err)
		}
		net.UseExactDType = true
		fmt.Println("done.")
	}

	tr = poly.NewTransformer[float32](net, embeddings, lmHead, finalNorm, template)
	if !(sequentialGPULoad && useGPU) && !useBitNetCPU {
		poly.ReleaseTransientSafetensorMap(allTensors, embeddings, lmHead, finalNorm)
	}
	tr.SetRMSNormEps(rmsNormEps)
	// Keep GPU KV-cache reservation bounded for desktop chat; default transformer
	// constructor sets 2048 which inflates VRAM significantly on smaller models.
	for i := range tr.Network.Layers {
		tr.Network.Layers[i].MaxSeqLen = maxSeqLen
	}
	if useTiling {
		tr.EnableTiling(tileSize)
	}
	if useGPU {
		if sequentialGPULoad {
			fmt.Printf("⏳ GPU init + block-wise weight upload (%d transformer blocks)...\n", numLayers)
		} else {
			fmt.Print("⏳ GPU Synchronization... ")
		}
		if err := tr.Network.InitWGPU(); err != nil {
			if sequentialGPULoad {
				log.Fatalf("❌ GPU init required for block-wise load: %v", err)
			}
			fmt.Printf("❌ Failed: %v\n", err)
			useGPU = false
		} else {
			applyGlitchTilingFlags(tr.Network, true, useTiling, tilingMode)
			if sequentialGPULoad {
				layerFiles := buildLayerShardIndex(safetensorFiles, numLayers)
				for li := 0; li < numLayers; li++ {
					layerMap := make(map[string][]float32)
					for _, sf := range layerFiles[li] {
						part, err := poly.LoadSafetensorsSelective(sf, func(k string) bool {
							return poly.HFWeightMatchesLayer(k, li)
						})
						if err != nil {
							log.Fatalf("⚠️  safetensors %s: %v", sf, err)
						}
						for k, v := range part {
							layerMap[k] = v
						}
					}
					poly.LoadWithPrefixes(net, layerMap)
					layerMap = nil
					runtime.GC()
					debug.FreeOSMemory()

					base := li * 4
					for j := 0; j < 4; j++ {
						idx := base + j
						layer := &tr.Network.Layers[idx]
						if layer.Type == poly.LayerRMSNorm {
							layer.DType = poly.DTypeFloat32
						} else {
							layer.DType = weightDType
						}
						if err := layer.SyncToGPU(); err != nil {
							log.Fatalf("❌ GPU sync block %d layer %d: %v", li, j, err)
						}
					}
					for j := 0; j < 4; j++ {
						(&tr.Network.Layers[base+j]).ReleaseInferenceHostWeights()
					}
					fmt.Printf("   ✓ Block %d/%d on GPU\n", li+1, numLayers)
				}
			} else {
				for i := range tr.Network.Layers {
					if tr.Network.Layers[i].Type == poly.LayerRMSNorm {
						tr.Network.Layers[i].DType = poly.DTypeFloat32
					} else {
						tr.Network.Layers[i].DType = weightDType
					}
					(&tr.Network.Layers[i]).SyncToGPU()
				}
			}
			if err := tr.SyncToGPU(); err != nil {
				log.Fatalf("❌ Embedding / LM head GPU sync: %v", err)
			}

			// Warmup pass to compile WGPU Shaders before first chat!
			_, _ = tr.ForwardTokenIDsWGPU([]uint32{0}, nil, true, true)
			tr.Reset()

			tr.ReleaseInferenceHostWeights()
			runtime.GC()
			debug.FreeOSMemory()

			fmt.Println("✅ Success!")
		}
	}
	if !useGPU {
		applyGlitchTilingFlags(tr.Network, false, useTiling, tilingMode)
		if useBitNetCPU || useTernaryPTQCPU {
			tr.Network.UseExactDType = true
		}
		tr.SyncInferenceCPU()
	}

	applyGlitchTanhiIfRequested(reader, tr.Network)

	fmt.Printf("\n✅ Model loaded! (%d layers)\n", numLayers)
	printPostLoadMemorySnapshot(tr)
	bannedTokens := poly.TokenizerBannedSpecialExceptEOS(tk, eosTokens)
	if len(bannedTokens) > 0 {
		fmt.Printf("🧯 Special-token mask active (%d banned IDs)\n", len(bannedTokens))
	}

	addSpecialTokens := false
	if strings.Contains(modelNameLower, "microsoft/bitnet-b1.58-2b-4t") {
		addSpecialTokens = true
	}
	encode := func(text string) []uint32 { return tk.Encode(text, addSpecialTokens) }
	decode := func(tokens []uint32) string { return tk.Decode(tokens, false) }

	maxTokens = 2048
	if isBitNetModel {
		maxTokens = 192
	}

	temp := float32(0.7)
	if deterministic {
		temp = 0
	}
	opts := poly.GenOptions{
		MaxTokens:             maxTokens,
		MinTokens:             8,
		Temperature:           temp,
		TopK:                  40,
		Deterministic:         deterministic,
		EOSTokens:             eosTokens,
		BannedTokens:          bannedTokens,
		RepetitionPenalty:     1.1,
		RepetitionWindow:      64,
		MaxConsecutiveRepeats: 3,
		NoRepeatNGram:         3,
	}

	for {
		fmt.Print("\nYou: ")
		userMsg, _ := reader.ReadString('\n')
		userMsg = strings.TrimSpace(userMsg)
		if userMsg == "exit" || userMsg == "quit" {
			break
		}

		fmt.Print("GlitchBot: ")
		reply, _ := tr.Generate(encode, decode, chatTurns, activeSystemPrompt, userMsg, opts)
		fmt.Println()

		chatTurns = append(chatTurns, poly.Turn{
			User:      userMsg,
			Assistant: reply,
		})
	}
}

func loadEOSTokensFromJSON(path string) []int {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var config map[string]interface{}
	if err := json.Unmarshal(data, &config); err != nil {
		return nil
	}
	return poly.EOSTokenIDsFromHFConfig(config)
}

func mergeIntSets(base []int, extra []int) []int {
	seen := make(map[int]struct{}, len(base)+len(extra))
	out := make([]int, 0, len(base)+len(extra))
	for _, v := range base {
		if _, ok := seen[v]; !ok {
			seen[v] = struct{}{}
			out = append(out, v)
		}
	}
	for _, v := range extra {
		if _, ok := seen[v]; !ok {
			seen[v] = struct{}{}
			out = append(out, v)
		}
	}
	return out
}

func buildLayerShardIndex(safetensorFiles []string, numLayers int) [][]string {
	layerFiles := make([][]string, numLayers)
	if numLayers <= 0 {
		return layerFiles
	}
	for _, sf := range safetensorFiles {
		names, err := poly.SafetensorsTensorNames(sf)
		if err != nil {
			// Fallback: if header scan fails, keep behavior identical by trying this
			// file for every layer.
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
	// Safety: if any layer had no indexed shard (unexpected naming), keep old behavior.
	for li := 0; li < numLayers; li++ {
		if len(layerFiles[li]) == 0 {
			layerFiles[li] = append(layerFiles[li], safetensorFiles...)
		}
	}
	return layerFiles
}
