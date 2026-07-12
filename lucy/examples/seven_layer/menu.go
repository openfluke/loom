package sevenlayer

import (
	"bufio"
	"fmt"
	"strconv"
	"strings"
)

type exampleEntry struct {
	name string
	run  func() bool
}

var menuExamples = []exampleEntry{
	{"Dense", RunDense},
	{"SwiGLU", RunSwiGLU},
	{"MHA", RunMHA},
	{"CNN1", RunCNN1},
	{"CNN2", RunCNN2},
	{"CNN3", RunCNN3},
	{"RNN", RunRNN},
	{"LSTM", RunLSTM},
	{"Embedding", RunEmbedding},
	{"Residual", RunResidual},
}

// RunMenu is Lucy [7]. Resets lucy_testing_output/seven_layer.txt, then runs examples.
func RunMenu(reader *bufio.Reader) {
	defer BeginSession()()

	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  [7] Seven-layer CPU suite — JSON + .entity · SC/MC/SIMD · train · save/reload ║")
	fmt.Println("║  Log: lucy_testing_output/seven_layer.txt (reset each run)            ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Println("  Flow: BuildNetworkFromJSON → dtype morph → fwd/bwd SC·MC·SIMD timing")
	fmt.Println("        → JSON + .entity save/reload → CPU SC/MC/SIMD train → checkpoint verify")
	fmt.Println("  Grids: most layers 1³·2³·3³; CNN1/2 skip 3³; CNN3 is 1³ only (8³ cube); Embedding@(0,0,0)")
	fmt.Println()
	fmt.Println("  [0] Run all layer types")
	for i, e := range menuExamples {
		fmt.Printf("  [%d] %s — examples/seven_layer/layers.go\n", i+1, e.name)
	}
	fmt.Print("Choice [1]: ")

	line, _ := reader.ReadString('\n')
	line = strings.TrimSpace(line)
	if line == "" {
		line = "1"
	}

	if line == "0" {
		for _, e := range menuExamples {
			fmt.Printf("\n▶ Starting %s …\n", e.name)
			e.run()
		}
		return
	}

	idx, err := strconv.Atoi(line)
	if err != nil || idx < 1 || idx > len(menuExamples) {
		fmt.Println("Invalid selection.")
		return
	}
	menuExamples[idx-1].run()
}
