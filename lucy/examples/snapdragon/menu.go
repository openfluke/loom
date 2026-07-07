package snapdragon

import (
	"bufio"
	"fmt"
	"strconv"
	"strings"
)

// RunMenu is Lucy [12]. Snapdragon (Hexagon) NPU CABI bridge — compile once, infer many.
func RunMenu(reader *bufio.Reader) {
	restore := quietQnnOutput()
	defer restore()

	ensureQnnEnv()

	npuNote := "not detected"
	if pluginPath := DefaultPluginPath(); NPUReady(pluginPath) {
		npuNote = "available"
	}

	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  [12] Snapdragon NPU bridge — Loom ↔ loom_accel_qualcomm.dll (QNN)  ║")
	fmt.Println("║  Log: lucy_testing_output/snapdragon.txt (reset each run)           ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Printf("  Plugin: %s\n", DefaultPluginPath())
	fmt.Printf("  Hexagon NPU: %s\n", npuNote)
	fmt.Println("  Matrix: 15 layers × FP32/FP16/INT16/INT8/INT4 × small/medium/large")
	fmt.Println()
	fmt.Println("  [4] DispatchLayer suite — medium (timing + determinism tables, seven-style) [default]")
	fmt.Println("  [5] DispatchLayer suite — all sizes (full matrix)")
	fmt.Println("  [0] Raw CABI matrix (direct plugin, per-cell compile)")
	fmt.Println("  [1] Medium tier CABI only")
	fmt.Println("  [2] Multi-hop demo (Loom → Qualcomm → Loom)")
	fmt.Println("  [3] Single layer picker (CABI)")
	fmt.Print("Choice [4]: ")

	line, _ := reader.ReadString('\n')
	line = strings.TrimSpace(line)
	if line == "" {
		line = "4"
	}

	cleanup := BeginSession()
	defer cleanup()

	switch line {
	case "0":
		RunBridgeSuite(nil)
	case "1":
		RunBridgeSuite([]string{"medium"})
	case "2":
		RunMultiHopDemo()
	case "3":
		runLayerPicker(reader)
	case "4":
		RunDispatchSuite([]string{"medium"})
	case "5":
		RunDispatchSuite(nil)
	default:
		fmt.Println("Invalid selection.")
	}
}

func runLayerPicker(reader *bufio.Reader) {
	m, err := LoadManifest()
	if err != nil {
		fmt.Println("manifest:", err)
		return
	}
	for i, layer := range m.Layers {
		fmt.Printf("  [%d] %s\n", i+1, layer.Name)
	}
	fmt.Print("Layer [1]: ")
	line, _ := reader.ReadString('\n')
	line = strings.TrimSpace(line)
	if line == "" {
		line = "1"
	}
	idx, err := strconv.Atoi(line)
	if err != nil || idx < 1 || idx > len(m.Layers) {
		fmt.Println("Invalid layer.")
		return
	}
	RunBridgeSuite([]string{"medium"}, m.Layers[idx-1].Name)
}
