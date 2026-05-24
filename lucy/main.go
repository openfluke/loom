package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/openfluke/loom/lucy/examples"
	lucytesting "github.com/openfluke/loom/lucy/testing"
)

func main() {
	fmt.Println("Initializing Lucy Bloom Rivers …")
	reader := bufio.NewReader(os.Stdin)
	mode := readInput(reader, "\n[1] Poly Talk (HuggingFace cache)\n"+
		"[2] Tests — dense mid-stream adaptation benchmark\n"+
		"[3] Layer testing — CPU/GPU suites (optional save to "+lucytesting.DefaultOutputDir+")\n"+
		"[4] Download approved HF models (SoulGlitch-style HTTP → hub/manual-download)\n"+
		"[5] Forward benchmark — BitNet b1.58 CPU: normal vs stepped vs pipeline\n"+
		"[6] Five-layer examples — per-layer .go tutorials (→ "+lucytesting.DefaultOutputDir+"/five_layer.txt)\n"+
		"[7] Seven-layer CPU suite — JSON · SC/MC/ASM · train · save/reload (→ "+lucytesting.DefaultOutputDir+"/seven_layer.txt)\n"+
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
	default:
		runHuggingFaceMode(reader)
	}
}
