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
		"[2] Tests — dense forward vs step (100 samples)\n"+
		"[3] Layer testing — CPU/GPU suites (optional save to "+lucytesting.DefaultOutputDir+")\n"+
		"[4] Download approved HF models (SoulGlitch-style HTTP → hub/manual-download)\n"+
		"Choice [1]: ", "1")
	switch strings.TrimSpace(mode) {
	case "2":
		examples.RunTestsMenu(reader)
	case "3":
		lucytesting.RunTestingMode(reader)
	case "4":
		runApprovedHFModelsDownload(reader)
	default:
		runHuggingFaceMode(reader)
	}
}
