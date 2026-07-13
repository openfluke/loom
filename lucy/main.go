package main

import (
	"bufio"
	"fmt"
	"os"
	"runtime"
	"strings"

	"github.com/openfluke/loom/lucy/examples"
	lucytesting "github.com/openfluke/loom/lucy/testing"
)

func printPlatformInfo() {
	hostCPU := strings.TrimSpace(os.Getenv("PROCESSOR_ARCHITECTURE"))
	msg := fmt.Sprintf("Platform: %s/%s binary", runtime.GOOS, runtime.GOARCH)
	if hostCPU != "" {
		msg += fmt.Sprintf(" | host CPU=%s", hostCPU)
	}
	switch {
	case runtime.GOOS == "windows" && strings.EqualFold(hostCPU, "ARM64") && runtime.GOARCH == "amd64":
		msg += " → x64 build on ARM64 PC (WoA / Prism emulation)"
	case runtime.GOOS == "windows" && strings.EqualFold(hostCPU, "ARM64") && runtime.GOARCH == "arm64":
		msg += " → native ARM64"
	case runtime.GOOS == "windows" && runtime.GOARCH == "amd64":
		msg += " → native x64"
	}
	fmt.Println(msg)
}

func main() {
	fmt.Println("Initializing Lucy Bloom Rivers …")
	printPlatformInfo()
	reader := bufio.NewReader(os.Stdin)
	if os.Getenv("LOOM_NINE_LAYER") == "1" {
		_ = os.Unsetenv("LOOM_NINE_LAYER")
		examples.RunNineLayerMenu(reader)
		return
	}
	if os.Getenv("LOOM_CONTEXT_SUITE") == "1" {
		_ = os.Unsetenv("LOOM_CONTEXT_SUITE")
		examples.RunContextSuiteAuto()
		return
	}
	mode := readInput(reader, "\n[1] Poly Talk (HuggingFace cache)\n"+
		"[2] Tests — dense mid-stream adaptation benchmark\n"+
		"[3] Layer testing — CPU/GPU suites (optional save to "+lucytesting.DefaultOutputDir+")\n"+
		"[4] Download approved HF models (SoulGlitch-style HTTP → hub/manual-download)\n"+
		"[5] Forward benchmark — BitNet b1.58 CPU: normal vs stepped vs pipeline\n"+
		"[6] Five-layer examples — per-layer .go tutorials (→ "+lucytesting.DefaultOutputDir+"/five_layer.txt)\n"+
		"[7] Seven-layer CPU suite — JSON · SC/MC/ASM · train · save/reload (→ "+lucytesting.DefaultOutputDir+"/seven_layer.txt)\n"+
		"[8] ENTITY Talk — HF cache → .entity convert → chat (Qwen/SmolLM2/Llama-style)\n"+
		"[9] Intel NPU bridge — Loom ↔ libloom_accel_intel.so · all layers/dtypes (→ "+lucytesting.DefaultOutputDir+"/nine_layer.txt)\n"+
		"[10] Context suite — long context / multi-prompt .entity tests (→ lucy_testing_output/context_suite/)\n"+
		"[11] Transformer SIMD bench — .entity CPU SC/MC/SIMD decode tok/s (→ lucy_testing_output/transformer_simd_bench/)\n"+
		"[12] Snapdragon NPU bridge — Loom ↔ loom_accel_qualcomm.dll (QNN) · all layers/dtypes (→ "+lucytesting.DefaultOutputDir+"/snapdragon.txt)\n"+
		"[13] Apple GPU bridge — Loom ↔ libloom_accel_apple.dylib (Metal) · all layers/dtypes (→ "+lucytesting.DefaultOutputDir+"/apple.txt)\n"+
		"[14] Native layer suite — per-layer dtype forward/backward/train (→ "+lucytesting.DefaultOutputDir+"/native_layers.txt)\n"+
		"[15] Cross-path CPU suite — SC/MC/SIMD vs native vs native-SIMD (→ "+lucytesting.DefaultOutputDir+"/cross_path_layers.txt)\n"+
		"[16] Tween native suite — native SC vs native-SIMD target propagation (→ "+lucytesting.DefaultOutputDir+"/tween_native_layers.txt)\n"+
		"[17] Adaptation suite — mid-stream task flip · all layers/dtypes/QAT/Nat/SIMD (→ "+lucytesting.DefaultOutputDir+"/adaptation_suite.txt)\n"+
		"Choice [1]: ", "1")
	switch strings.TrimSpace(mode) {
	case "2":
		examples.RunTestsMenu(reader)
	case "3":
		lucytesting.RunTestingMode(reader)
	case "4":
		runApprovedHFModelsDownload(reader)
	case "5":
		forwardBenchOnly = true
		runHuggingFaceMode(reader)
	case "6":
		examples.RunFiveLayerMenu(reader)
	case "7":
		examples.RunSevenLayerMenu(reader)
	case "8":
		runEntityTalkMode(reader)
	case "9":
		examples.RunNineLayerMenu(reader)
	case "10":
		examples.RunContextSuiteMenu(reader)
	case "11":
		examples.RunTransformerSimdBenchMenu(reader)
	case "12":
		examples.RunSnapdragonMenu(reader)
	case "13":
		examples.RunAppleMenu(reader)
	case "14":
		examples.RunDenseNativeMenu(reader)
	case "15":
		examples.RunCrossPathMenu(reader)
	case "16":
		examples.RunTweenNativeMenu(reader)
	case "17":
		examples.RunAdaptationMenu(reader)
	default:
		runHuggingFaceMode(reader)
	}
}
