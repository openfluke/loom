package fivelayer

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

// RunMenu is Lucy [6]. Resets lucy_testing_output/five_layer.txt, then runs examples.
func RunMenu(reader *bufio.Reader) {
	defer BeginSession()()

	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  [6] Five-layer Loom examples — copy any *.go, rename Run* → main   ║")
	fmt.Println("║  Log: lucy_testing_output/five_layer.txt (reset each run)             ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Println("  Each file: JSON → BuildNetworkFromJSON → GPU → train → save → reload")
	fmt.Println("  Every file runs all 21 numerical types with deviation buckets.")
	fmt.Println()
	fmt.Println("  [0] Run all layer types")
	for i, e := range menuExamples {
		fmt.Printf("  [%d] %s — examples/five_layer/%s.go\n", i+1, e.name, strings.ToLower(e.name))
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
