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

	if forwardBenchOnly {
		idx, name, ok := findBitNetBenchModel(models)
		if !ok {
			log.Fatalf("Forward benchmark requires %q in the HuggingFace cache.", bitNetBenchModelNeedle)
		}
		fmt.Printf("\n📊 Forward benchmark: auto-selecting [%d] %s (CPU, deterministic)\n", idx, name)
	}

	launch := readPolyTalkLaunchOptions(reader)
	deterministic = launch.deterministic
	useGPU := launch.useGPU
	useTiling := launch.useTiling
	tilingMode := launch.tilingMode
	tileSize := launch.tileSize
	weightDType = launch.weightDType
	sequentialGPULoad := launch.sequentialGPULoad

	var modelName string
	var selectedIdx int
	if forwardBenchOnly {
		var ok bool
		selectedIdx, modelName, ok = findBitNetBenchModel(models)
		if !ok {
			log.Fatalf("Forward benchmark requires %q in the HuggingFace cache.", bitNetBenchModelNeedle)
		}
	} else {
		modelInput := readInput(reader, "\nSelect model number: ", "1")
		fmt.Sscanf(modelInput, "%d", &selectedIdx)
		if selectedIdx < 1 || selectedIdx > len(models) {
			log.Fatalf("Invalid model selection: %d", selectedIdx)
		}
		modelName = models[selectedIdx-1]
	}
	applyModelSpecificLaunchOptions(reader, modelName, &launch, 0)
	useBitNetCPU := launch.useBitNetCPU
	useBitNetGPU := launch.useBitNetGPU
	useTernaryPTQCPU := launch.useTernaryPTQCPU
	useBitNetPacked := launch.useBitNetPacked
	deterministic = launch.deterministic
	weightDType = launch.weightDType
	modelNameLower := strings.ToLower(modelName)
	isQwen := strings.Contains(modelNameLower, "qwen")
	isBitNetModel := strings.Contains(modelNameLower, "bitnet") || strings.Contains(modelNameLower, "1bit")
	template := templateForModel(modelName)

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

	if (sequentialGPULoad && useGPU) || useBitNetPacked {
		globalStored := make(map[string]poly.HFStoredTensor)
		for _, f := range safetensorFiles {
			part, err := poly.LoadSafetensorsSelectiveRaw(f, poly.HFWeightIsGlobal)
			if err != nil {
				log.Fatalf("⚠️  safetensors %s: %v", f, err)
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

	if useBitNetPacked {
		if useBitNetGPU {
			fmt.Printf("⏳ BitNet WebGPU block-wise load + pack (%d transformer blocks)...\n", numLayers)
		} else {
			fmt.Printf("⏳ BitNet CPU block-wise load + pack (%d transformer blocks)...\n", numLayers)
		}
		layerFiles := buildLayerShardIndex(safetensorFiles, numLayers)
		for li := 0; li < numLayers; li++ {
			layerMap := make(map[string]poly.HFStoredTensor)
			for _, sf := range layerFiles[li] {
				part, err := poly.LoadSafetensorsSelectiveRaw(sf, func(k string) bool {
					return poly.HFWeightMatchesLayer(k, li)
				})
				if err != nil {
					log.Fatalf("⚠️  safetensors %s: %v", sf, err)
				}
				for k, v := range part {
					layerMap[k] = v
				}
			}
			poly.LoadWithPrefixesFromHFStored(net, layerMap)
			if err := poly.PrepareDecoderBlockBitNetTernaryCPU(net, li); err != nil {
				log.Fatalf("❌ BitNet CPU preparation failed for block %d: %v", li, err)
			}
			poly.ReleaseTransientHFStoredMap(layerMap)
			fmt.Printf("   ✓ Block %d/%d packed\n", li+1, numLayers)
		}
		runtime.GC()
		debug.FreeOSMemory()
	} else if !(sequentialGPULoad && useGPU) {
		poly.LoadWithPrefixes(net, allTensors)
	}
	if useBitNetPacked {
		if isBitNetModel {
			if useBitNetGPU {
				fmt.Print("🧪 BitNet b1.58: decoder weights from raw safetensors (U8 offline → packed; dense → Master then pack); globals decoded once. ")
			} else {
				fmt.Print("🧮 BitNet b1.58: decoder weights from raw safetensors (U8 offline → packed; dense → Master then pack); globals decoded once. ")
			}
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
	if !(sequentialGPULoad && useGPU) && !useBitNetPacked {
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
		trackMemory := poly.MemoryHistoryEnabled()
		if trackMemory {
			fmt.Println("📈 Memory timeline recording — terminal chart prints when GPU load finishes.")
			beginMemoryHistorySession("hf_gpu_load")
			recordMemoryHistory("hf_cpu_weights_ready")
		}
		defer func() {
			if trackMemory {
				finishMemoryHistorySession()
			}
		}()
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
			recordMemoryHistory("wgpu_ready")
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
					recordMemoryHistory(fmt.Sprintf("block_%02d_before_sync", li+1))
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
					recordMemoryHistory(fmt.Sprintf("block_%02d_after_sync", li+1))
					for j := 0; j < 4; j++ {
						(&tr.Network.Layers[base+j]).ReleaseInferenceHostWeights()
					}
					recordMemoryHistory(fmt.Sprintf("block_%02d_after_release", li+1))
					fmt.Printf("   ✓ Block %d/%d on GPU\n", li+1, numLayers)
				}
			} else {
				recordMemoryHistory("bulk_sync_start")
				for i := range tr.Network.Layers {
					if tr.Network.Layers[i].Type == poly.LayerRMSNorm {
						tr.Network.Layers[i].DType = poly.DTypeFloat32
					} else {
						tr.Network.Layers[i].DType = weightDType
					}
					(&tr.Network.Layers[i]).SyncToGPU()
				}
				recordMemoryHistory("bulk_sync_done")
			}
			if err := syncTransformerGlobalWeightsSequential(tr); err != nil {
				log.Fatalf("❌ Global weight GPU sync: %v", err)
			}

			// Warmup pass to compile WGPU Shaders before first chat!
			_, _ = tr.ForwardTokenIDsWGPU([]uint32{0}, nil, true, true)
			tr.Reset()

			tr.ReleaseInferenceHostWeights()
			recordMemoryHistory("host_weights_released")
			runtime.GC()
			debug.FreeOSMemory()
			recordMemoryHistory("after_gc")

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

	if !forwardBenchOnly {
		applyGlitchTanhiIfRequested(reader, tr.Network)
	}

	fmt.Printf("\n✅ Model loaded! (%d layers)\n", numLayers)
	printPostLoadMemorySnapshot(tr)
	bannedTokens := poly.TokenizerBannedSpecialExceptEOS(tk, eosTokens)
	if len(bannedTokens) > 0 && !forwardBenchOnly {
		fmt.Printf("🧯 Special-token mask active (%d banned IDs)\n", len(bannedTokens))
	}

	runPolyTalkChatSession(reader, modelName, numLayers, launch, useGPU)
}

func configureCPUForwardMode(reader *bufio.Reader, tr *poly.Transformer[float32], useGPU bool) {
	fmt.Println("\n🧠 CPU forward schedule (decoder sub-layers):")
	fmt.Println("  [1] Normal — fused blocks per forward (default, fastest)")
	fmt.Println("  [2] Stepped — same math, one sub-layer at a time (auto-drained each forward)")
	fmt.Println("  [3] Queued — same as stepped; pause each sub-layer (micro-queue, one token)")
	fmt.Println("  [4] Pipeline — wavefront: tokens at different blocks; one tick = one clock")
	fwdInput := readInput(reader, "Choice [1]: ", "1")
	switch strings.TrimSpace(fwdInput) {
	case "2":
		tr.ForwardMode = poly.TransformerForwardSteppedCPU
	case "3":
		tr.ForwardMode = poly.TransformerForwardQueuedCPU
	case "4":
		tr.ForwardMode = poly.TransformerForwardPipelineCPU
	default:
		tr.ForwardMode = poly.TransformerForwardNormal
	}
	if tr.ForwardMode == poly.TransformerForwardNormal {
		return
	}
	if useGPU {
		fmt.Println("ℹ️  Non-normal CPU forward modes skip GPU during generation.")
	}
	steps := tr.CPUForwardQueueStepTotal()
	nb := len(tr.Network.Layers) / 4
	if tr.ForwardMode == poly.TransformerForwardPipelineCPU {
		fmt.Printf("ℹ️  Pipeline: tokens can overlap across blocks (%d blocks). One tick advances all ready sub-layers.\n", nb)
		fmt.Println("   Multiple prompt tokens can be in flight at different blocks during prefill.")
	} else {
		fmt.Printf("ℹ️  Each token forward = %d sub-layer steps (%d decoder blocks).\n", steps, nb)
	}
	if readInput(reader, "📝 Log each sub-layer step during generation? (1=yes / 0=no) [0]: ", "0") == "1" {
		tr.ForwardStepDebug = true
		tr.SetForwardStepObserver(func(step, total int, label string) {
			fmt.Printf("   [%s] step %d/%d: %s\n", tr.ForwardMode, step, total, label)
		})
	}
	if tr.ForwardMode == poly.TransformerForwardQueuedCPU &&
		readInput(reader, "⏯️  Interactive micro-queue (Enter per sub-layer)? (1=yes / 0=no) [0]: ", "0") == "1" {
		tr.QueueTickPause = func(step, total int, label string) {
			fmt.Printf("   ⏸ step %d/%d: %s — press Enter… ", step, total, label)
			_, _ = reader.ReadString('\n')
		}
		fmt.Println("   Interactive micro-queue enabled.")
	}
	if tr.ForwardMode == poly.TransformerForwardPipelineCPU &&
		readInput(reader, "⏯️  Interactive pipeline (Enter per wavefront tick)? (1=yes / 0=no) [0]: ", "0") == "1" {
		tr.PipelineTickPause = func(tick, total int, summary string) {
			fmt.Printf("   ⏸ %s — press Enter… ", summary)
			_, _ = reader.ReadString('\n')
		}
		fmt.Println("   Interactive pipeline enabled.")
	}
	fmt.Printf("✓ CPU forward mode: %s\n", tr.ForwardMode)
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
