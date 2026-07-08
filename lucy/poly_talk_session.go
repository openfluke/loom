package main

import (
	"bufio"
	"fmt"
	"log"
	"strings"

	"github.com/openfluke/loom/poly"
)

type polyTalkLaunch struct {
	deterministic     bool
	useGPU            bool
	useSIMD           bool // CPU only: Plan 9 AVX2/NEON SIMD forward (SetSimdForwardRecursive)
	useTiling         bool
	tilingMode        string
	tileSize          int
	weightDType       poly.DType
	sequentialGPULoad bool
	measureMemoryLoad bool
	useBitNetCPU      bool
	useBitNetGPU      bool
	useTernaryPTQCPU  bool
	useBitNetPacked   bool
}

func readPolyTalkLaunchOptions(reader *bufio.Reader) polyTalkLaunch {
	var cfg polyTalkLaunch
	poly.ResetMemoryHistoryRecording()

	if forwardBenchOnly {
		cfg.deterministic = true
		cfg.useGPU = false
		fmt.Println("🎮 GPU: off (benchmark uses CPU BitNet + pipeline wavefront stats)")
		cfg.tilingMode, cfg.tileSize = parseLLMExecutionMode("2")
		cfg.useTiling = true
		return cfg
	}

	detInput := readInput(reader, "🎯 Deterministic mode? (1=yes / 0=no) [0]: ", "0")
	cfg.deterministic = detInput == "1"

	cfg.useTiling = true
	cfg.tileSize = -1

	fmt.Print("🎮 Enable GPU Acceleration? (1=yes / 0=no) [0]: ")
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)
	cfg.useGPU = input == "1"

	fmt.Println("\n🚀 Select Execution Mode:")
	fmt.Println("  [1] Tiled — GPU: single-workgroup; CPU: multi-core tiled")
	fmt.Println("  [2] Tiled — GPU: multi-workgroup; CPU: multi-core tiled")
	execModeInput := readInput(reader, "Choice [2]: ", "2")
	cfg.tilingMode, cfg.tileSize = parseLLMExecutionMode(execModeInput)

	if cfg.useGPU {
		fmt.Print("💎 Weight Precision? (4=Q4_0 / 8=INT8 / 32=FP32) [4]: ")
		precInput, _ := reader.ReadString('\n')
		precInput = strings.TrimSpace(precInput)
		if precInput == "32" {
			cfg.weightDType = poly.DTypeFloat32
		} else if precInput == "8" {
			cfg.weightDType = poly.DTypeInt8
		} else {
			cfg.weightDType = poly.DTypeInt4
		}
		weightDType = cfg.weightDType
	}

	if cfg.useGPU {
		cfg.sequentialGPULoad = readInput(reader, "📥 Load weights block-by-block into GPU (lower peak host RAM; skips holding full checkpoint map)? (1=yes / 0=no) [0]: ", "0") == "1"
	}
	cfg.measureMemoryLoad = promptMeasureMemoryDuringLoad(reader)

	return cfg
}

func applyModelSpecificLaunchOptions(reader *bufio.Reader, modelName string, cfg *polyTalkLaunch, entityStoredDType poly.DType) {
	modelNameLower := strings.ToLower(modelName)
	isBitNetModel := strings.Contains(modelNameLower, "bitnet") || strings.Contains(modelNameLower, "1bit")
	isQwen := strings.Contains(modelNameLower, "qwen")

	if isBitNetModel {
		if cfg.useGPU {
			cfg.useBitNetGPU = true
			cfg.weightDType = poly.DTypeTernary
			weightDType = poly.DTypeTernary
			fmt.Println("🧪 BitNet model detected; enabling experimental WebGPU packed ternary inference.")
		} else {
			cfg.useBitNetCPU = true
			fmt.Println("🧮 BitNet model detected; enabling CPU packed ternary inference.")
		}
	} else if entityStoredDType == poly.DTypeTernary {
		if cfg.useGPU {
			cfg.useBitNetGPU = true
			cfg.weightDType = poly.DTypeTernary
			weightDType = poly.DTypeTernary
			fmt.Println("🧪 BitNet .entity — GPU packed ternary inference.")
		} else {
			cfg.useBitNetCPU = true
			fmt.Println("🧮 BitNet .entity — CPU packed ternary inference.")
		}
	} else if !cfg.useGPU {
		if entityStoredDType == poly.DTypeInt4 {
			fmt.Println("🧮 CPU: baked Q4_0 .entity — dequantizing to FP32 for tiled forward.")
		} else {
			quantInput := readInput(reader, "🧮 CPU weight precision? (32=FP32 / ternary=experimental PTQ) [32]: ", "32")
			switch strings.ToLower(strings.TrimSpace(quantInput)) {
			case "ternary", "t", "bitnet", "1bit", "b1.58", "158":
				cfg.useTernaryPTQCPU = true
				fmt.Println("⚠️  Ternary PTQ is experimental. It is not equivalent to BitNet training and may produce bad text.")
			}
		}
	}
	cfg.useBitNetPacked = cfg.useBitNetCPU || cfg.useBitNetGPU

	if cfg.deterministic && isQwen {
		fmt.Println("⚠️  Qwen deterministic=1 can leak planning text. Keeping deterministic=1 because you explicitly selected it.")
	}
	if cfg.deterministic && strings.Contains(modelNameLower, "instruct") && strings.Contains(modelNameLower, "1.7b") {
		fmt.Println("⚠️  Deterministic=1 can collapse into punctuation-only outputs on 1.7B instruct models. Use Deterministic=0 for normal chat quality.")
	}

	deterministic = cfg.deterministic
}

func runPolyTalkChatSession(reader *bufio.Reader, modelName string, numLayers int, cfg polyTalkLaunch, useGPU bool) {
	if forwardBenchOnly {
		modelNameLower := strings.ToLower(modelName)
		isBitNetModel := strings.Contains(modelNameLower, "bitnet") || strings.Contains(modelNameLower, "1bit")
		if !isBitNetModel || useGPU {
			log.Fatalf("Forward benchmark requires BitNet b1.58 loaded on CPU (no GPU).")
		}
		addSpecialTokens := strings.Contains(modelNameLower, "microsoft/bitnet-b1.58-2b-4t")
		encode := func(text string) []uint32 { return tk.Encode(text, addSpecialTokens) }
		activeSystemPrompt := defaultSystemPromptForModel(modelName)
		runForwardScheduleComparison(tr, encode, activeSystemPrompt, numLayers)
		forwardBenchOnly = false
		return
	}

	configureCPUForwardMode(reader, tr, useGPU)

	maxTokensLocal := 2048
	modelNameLower := strings.ToLower(modelName)
	if strings.Contains(modelNameLower, "bitnet") || strings.Contains(modelNameLower, "1bit") {
		maxTokensLocal = 192
	}

	layerTrace := readInput(reader, "\n📼 Layer action recording? (1=yes / 0=no) [0]: ", "0") == "1"
	layerTraceMaxTokens := 8
	layerTracePrefill := false
	repeatDecoderBlock := -1
	if layerTrace {
		fmt.Println("ℹ️  Each new token runs through all decoder blocks (30 layers → ~181 sub-steps per token).")
		fmt.Println("   Prefill (your whole prompt) is a separate full pass unless you opt in below.")
		if useGPU {
			fmt.Println("ℹ️  Traced decode uses CPU stepped forward; untraced prefill can still use GPU.")
		}
		tokInput := readInput(reader, "🔢 How many new tokens to trace per turn? [8]: ", "8")
		fmt.Sscanf(tokInput, "%d", &layerTraceMaxTokens)
		if layerTraceMaxTokens < 1 {
			layerTraceMaxTokens = 1
		}
		layerTracePrefill = readInput(reader, "📥 Also trace prefill (full prompt through all layers)? (1=yes / 0=no) [0]: ", "0") == "1"
		repeatInput := readInput(reader, fmt.Sprintf("🔁 Repeat a decoder block after its first pass? (1-%d, 0=off) [0]: ", numLayers), "0")
		var repeatOneBased int
		fmt.Sscanf(repeatInput, "%d", &repeatOneBased)
		if repeatOneBased >= 1 && repeatOneBased <= numLayers {
			repeatDecoderBlock = repeatOneBased - 1
			fmt.Printf("   Block %d will run twice; repeat pass is logged as REPEAT.\n", repeatOneBased)
		}
		maxTokensLocal = layerTraceMaxTokens
	}

	temp := float32(0.7)
	if cfg.deterministic {
		temp = 0
	}
	minTokens := 8
	if layerTrace {
		minTokens = 1
	}
	activeSystemPrompt := defaultSystemPromptForModel(modelName)
	opts := poly.GenOptions{
		MaxTokens:             maxTokensLocal,
		MinTokens:             minTokens,
		Temperature:           temp,
		TopK:                  40,
		Deterministic:         cfg.deterministic,
		EOSTokens:             eosTokens,
		BannedTokens:          poly.TokenizerBannedSpecialExceptEOS(tk, eosTokens),
		RepetitionPenalty:     1.1,
		RepetitionWindow:      64,
		MaxConsecutiveRepeats: 3,
		NoRepeatNGram:         3,
		LayerTrace:            layerTrace,
		LayerTraceMaxTokens:   layerTraceMaxTokens,
		LayerTracePrefill:     layerTracePrefill,
		RepeatDecoderBlock:    repeatDecoderBlock,
	}
	if layerTrace {
		opts.Silent = true
		stepsPerTok := numLayers*6 + 1
		if layerTracePrefill {
			fmt.Printf("📼 Trace mode: prefill + up to %d decode token(s) (~%d sub-layer lines each phase).\n", layerTraceMaxTokens, stepsPerTok)
		} else {
			fmt.Printf("📼 Trace mode: up to %d decode token(s) only (~%d sub-layer lines per token; prefill not traced).\n", layerTraceMaxTokens, stepsPerTok)
		}
	}

	addSpecialTokens := false
	if strings.Contains(modelNameLower, "microsoft/bitnet-b1.58-2b-4t") {
		addSpecialTokens = true
	}
	encode := func(text string) []uint32 { return tk.Encode(text, addSpecialTokens) }
	decode := func(tokens []uint32) string { return tk.Decode(tokens, false) }

	chatTurns = nil
	for {
		fmt.Print("\nYou: ")
		userMsg, _ := reader.ReadString('\n')
		userMsg = strings.TrimSpace(userMsg)
		if userMsg == "exit" || userMsg == "quit" {
			break
		}

		fmt.Print("GlitchBot: ")
		reply, _ := tr.Generate(encode, decode, chatTurns, activeSystemPrompt, userMsg, opts)
		if layerTrace {
			fmt.Printf("\n(decoded) %s\n", reply)
		} else {
			fmt.Println()
		}

		chatTurns = append(chatTurns, poly.Turn{
			User:      userMsg,
			Assistant: reply,
		})
	}
}
