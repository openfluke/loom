package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/openfluke/loom/poly"
)

var (
	tr            *poly.Transformer[float32]
	tk            *poly.Tokenizer
	eosTokens     []int
	chatTurns     []poly.Turn
	weightDType   poly.DType = poly.DTypeFloat32
	deterministic bool       = true
	maxTokens                = 2048
	maxSeqLen                = 512
	forwardBenchOnly         = false
)

func applyGlitchTilingFlags(net *poly.VolumetricNetwork, useGPU, useTiling bool, tilingMode string) {
	if useGPU {
		net.EnableMultiCoreTiling = useTiling && tilingMode == "3"
	} else {
		net.EnableMultiCoreTiling = useTiling
	}
}

func glitchTanhiHost() string {
	h := strings.TrimSpace(os.Getenv("TANHI_HOST"))
	if h == "" {
		return "127.0.0.1"
	}
	return h
}

func glitchTanhiPort() int {
	p := strings.TrimSpace(os.Getenv("TANHI_PORT"))
	if p == "" {
		return poly.DefaultTanhiUDPPort
	}
	v, err := strconv.Atoi(p)
	if err != nil || v <= 0 || v >= 65536 {
		return poly.DefaultTanhiUDPPort
	}
	return v
}

func glitchTanhiSendShape() bool {
	s := strings.TrimSpace(os.Getenv("TANHI_SHAPE"))
	return s == "1" || strings.EqualFold(s, "true") || strings.EqualFold(s, "yes")
}

func applyGlitchTanhiIfRequested(reader *bufio.Reader, net *poly.VolumetricNetwork) {
	if net == nil {
		return
	}
	env := strings.TrimSpace(os.Getenv("GLITCH_TANHI"))
	var enable bool
	switch {
	case env == "1" || strings.EqualFold(env, "true") || strings.EqualFold(env, "yes"):
		enable = true
	case env == "0" || strings.EqualFold(env, "false") || strings.EqualFold(env, "no"):
		return
	default:
		tanhiInput := readInput(reader, "\n📡 TANHI UDP telemetry → SoulGlitch? (1=yes / 0=no) [0]: ", "0")
		enable = tanhiInput == "1"
	}
	if !enable {
		return
	}
	net.Tanhi = &poly.TanhiUDPConfig{
		Enabled:   true,
		Host:      glitchTanhiHost(),
		Port:      glitchTanhiPort(),
		SendShape: glitchTanhiSendShape(),
	}
	fmt.Printf("   TANHI → udp://%s:%d  send_shape=%v\n", net.Tanhi.Host, net.Tanhi.Port, net.Tanhi.SendShape)
}

func parseLLMExecutionMode(execModeInput string) (tilingMode string, tileSize int) {
	switch strings.TrimSpace(execModeInput) {
	case "1":
		return "2", -1
	case "2", "3":
		return "3", -1
	default:
		return "3", -1
	}
}

func printPostLoadMemorySnapshot(tr *poly.Transformer[float32]) {
	if tr == nil || tr.Network == nil {
		return
	}
	m := poly.NewMemoryFootprintFromTransformer(tr)
	fmt.Printf("📊 Memory: host weights %.2f MB | GPU weights %.2f MB | GPU KV %.2f MB\n",
		m.HostWeightsMB, m.GPUWeightsMB, m.GPUKVMB)
}

func promptMeasureMemoryDuringLoad(reader *bufio.Reader) bool {
	enabled := readInput(reader,
		"📈 Measure memory during load? (terminal chart after mount — host weights, GPU upload, RSS) (1=yes / 0=no) [1]: ",
		"1") == "1"
	poly.SetMemoryHistoryRecording(enabled)
	if enabled {
		fmt.Println("   Recording at each step: entity decode → mount (CPU or GPU) → release (chart prints when load finishes).")
	}
	return enabled
}

func recordMemoryHistoryRuntime(label string) {
	poly.RecordRuntimeOnly(poly.GlobalMemoryHistory, label)
}

func beginMemoryHistorySession(name string) {
	if poly.MemoryHistoryEnabled() {
		poly.GlobalMemoryHistory.BeginSession(name)
	}
}

func recordMemoryHistory(label string) {
	poly.RecordFromTransformer(poly.GlobalMemoryHistory, tr, label)
}

func finishMemoryHistorySession() {
	_ = poly.GlobalMemoryHistory.FinishSession()
}

func syncTransformerGlobalWeightsSequential(tr *poly.Transformer[float32]) error {
	if tr == nil || tr.Network == nil || tr.Network.GPUContext == nil {
		return fmt.Errorf("GPU not ready for global weight sync")
	}
	tr.Network.GPUContext.ResetCache()

	recordMemoryHistory("embeddings_before_sync")
	if err := tr.SyncEmbeddingsToGPU(); err != nil {
		return err
	}
	recordMemoryHistory("embeddings_after_sync")

	recordMemoryHistory("lm_head_before_sync")
	if err := tr.SyncLMHeadToGPU(); err != nil {
		return err
	}
	recordMemoryHistory("lm_head_after_sync")

	tr.ReleaseEmbeddingsHost()
	recordMemoryHistory("embeddings_after_release")
	poly.ReleaseInferenceTransientMemory()
	tr.ReleaseLMHeadHost()
	recordMemoryHistory("lm_head_after_release")
	poly.ReleaseInferenceTransientMemory()

	recordMemoryHistory("final_norm_before_sync")
	if err := tr.SyncFinalNormToGPU(); err != nil {
		return err
	}
	recordMemoryHistory("final_norm_after_sync")
	tr.ReleaseFinalNormHost()
	recordMemoryHistory("final_norm_after_release")
	poly.ReleaseInferenceTransientMemory()
	return nil
}

func readInput(reader *bufio.Reader, prompt string, Default string) string {
	fmt.Print(prompt)
	txt, _ := reader.ReadString('\n')
	txt = strings.TrimSpace(txt)
	if txt == "" {
		return Default
	}
	return txt
}

func templateForModel(modelName string) poly.Template {
	return poly.TemplateForHFModelID(modelName)
}

func defaultSystemPromptForModel(modelName string) string {
	name := strings.ToLower(modelName)
	if strings.Contains(name, "bitnet") || strings.Contains(name, "1bit") {
		return ""
	}
	if strings.Contains(name, "qwen") {
		return "You are a helpful assistant. Respond directly with the final answer only. Do not expose internal reasoning or chain-of-thought."
	}
	if strings.Contains(name, "instruct") || strings.Contains(name, "qwen") || strings.Contains(name, "llama") || strings.Contains(name, "smollm") {
		return "You are a helpful assistant. Be concise, coherent, and avoid repetition."
	}
	return ""
}
