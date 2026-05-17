package testing

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// DefaultOutputDir is wiped and recreated when you choose to save logs (relative to process cwd).
const DefaultOutputDir = "lucy_testing_output"

const defaultLogFile = "log.txt"

func readPrompt(reader *bufio.Reader, prompt, defaultVal string) string {
	fmt.Print(prompt)
	txt, _ := reader.ReadString('\n')
	txt = strings.TrimSpace(txt)
	if txt == "" {
		return defaultVal
	}
	return txt
}

// ClearOutputDir removes prior saved runs so a new save starts clean.
func ClearOutputDir() error {
	if err := os.RemoveAll(DefaultOutputDir); err != nil {
		return err
	}
	return os.MkdirAll(DefaultOutputDir, 0o755)
}

// LogFilePath returns the path used when saving (under DefaultOutputDir).
func LogFilePath() string {
	return filepath.Join(DefaultOutputDir, defaultLogFile)
}

// SetupLogFile tees stdout to logPath and the terminal. Call the returned cleanup when done.
func SetupLogFile(logPath string) func() {
	logFile, err := os.Create(logPath)
	if err != nil {
		fmt.Printf("Warning: could not create log file: %v\n", err)
		return func() {}
	}

	r, w, err := os.Pipe()
	if err != nil {
		logFile.Close()
		fmt.Printf("Warning: could not create pipe: %v\n", err)
		return func() {}
	}

	origStdout := os.Stdout
	os.Stdout = w

	done := make(chan struct{})
	go func() {
		mw := io.MultiWriter(origStdout, logFile)
		buf := make([]byte, 4096)
		for {
			n, readErr := r.Read(buf)
			if n > 0 {
				_, _ = mw.Write(buf[:n])
			}
			if readErr != nil {
				break
			}
		}
		close(done)
	}()

	return func() {
		_ = w.Close()
		<-done
		_ = r.Close()
		_ = logFile.Close()
		os.Stdout = origStdout
		fmt.Printf("\n📄 Output saved to: %s\n", logPath)
	}
}

// maybeLogCapture returns a deferred cleanup if saveLog=="1"; clears DefaultOutputDir first.
func maybeLogCapture(saveLog string) (cleanup func()) {
	cleanup = func() {}
	if saveLog != "1" {
		return cleanup
	}
	if err := ClearOutputDir(); err != nil {
		fmt.Printf("Warning: could not reset output dir: %v\n", err)
		return cleanup
	}
	return SetupLogFile(LogFilePath())
}

// RunTestingMode mirrors Glitch [3] Testing: pick layer, pick test, optional log under DefaultOutputDir.
func RunTestingMode(reader *bufio.Reader) {
	layers := []string{"CNN1", "CNN2", "CNN3", "MHA", "Dense", "SwiGLU", "RNN", "LSTM", "Embedding", "Residual"}
	fmt.Println("\n🧪 Layer testing (CPU/GPU suites)")
	fmt.Println("  [0] All layers (registered suites)")
	for i, name := range layers {
		fmt.Printf("  [%d] %s\n", i+1, name)
	}
	layerInput := readPrompt(reader, "Select layer [1]: ", "1")

	savePrompt := "Save output to " + DefaultOutputDir + "/" + defaultLogFile + "? (old folder is deleted first) (1=yes / 0=no) [0]: "

	if layerInput == "0" {
		confirm := readPrompt(reader, "Run all tests on all layers? (1=yes / 0=no) [0]: ", "0")
		if confirm != "1" {
			return
		}
		saveLog := readPrompt(reader, savePrompt, "0")
		defer maybeLogCapture(saveLog)()
		RunAllLayers()
		return
	}

	idx, err := strconv.Atoi(layerInput)
	if err != nil || idx < 1 || idx > len(layers) {
		fmt.Println("Invalid selection.")
		return
	}
	saveLog := readPrompt(reader, savePrompt, "0")
	defer maybeLogCapture(saveLog)()
	runLayerTests(reader, layers[idx-1], "")
}

type testEntry struct {
	name string
	fn   func()
}

func runLayerTests(reader *bufio.Reader, layerName, testInput string) {
	var tests []testEntry
	switch layerName {
	case "CNN1":
		tests = []testEntry{
			{"L1 Caching (CPU SC/MC + GPU SC/MC)", RunCNN1L1Caching},
			{"Training (4 tiled modes × 21 types)", RunCNN1Training},
			{"GPU Forward Parity", RunCNN1GPUForward},
			{"GPU Backward Parity", RunCNN1GPUBackward},
		}
	case "CNN2":
		tests = []testEntry{
			{"L1 Caching (CPU SC/MC + GPU SC/MC)", RunCNN2L1Caching},
			{"Training (4 tiled modes × 21 types)", RunCNN2Training},
			{"GPU Forward Parity", RunCNN2GPUForward},
			{"GPU Backward Parity", RunCNN2GPUBackward},
		}
	case "CNN3":
		tests = []testEntry{
			{"L1 Caching (CPU SC/MC + GPU SC/MC)", RunCNN3L1Caching},
			{"Training (4 tiled modes × 21 types)", RunCNN3Training},
			{"GPU Forward Parity", RunCNN3GPUForward},
			{"GPU Backward Parity", RunCNN3GPUBackward},
		}
	case "MHA":
		tests = []testEntry{
			{"L1 Caching (CPU SC/MC + GPU SC/MC)", RunMHAL1Caching},
			{"Training (4 tiled modes × 21 types)", RunMHATraining},
			{"GPU Forward Parity", RunMHAForward},
			{"GPU Backward Parity", RunMHABackward},
		}
	case "Dense":
		tests = []testEntry{
			{"L1 Caching (CPU/ASM SC/MC + GPU SC/MC)", RunDenseL1Caching},
			{"Training (4 tiled modes × 21 types)", RunDenseTraining},
			{"GPU Forward Parity (incl. ASM timers)", RunDenseGPUForward},
			{"GPU Backward Parity", RunDenseGPUBackward},
		}
	case "SwiGLU":
		tests = []testEntry{
			{"L1 Caching (CPU SC/MC + GPU SC/MC)", RunSwiGLUL1Caching},
			{"Training (4 tiled modes × 21 types)", RunSwiGLUTraining},
			{"GPU Forward Parity", RunSwiGLUGPUForward},
			{"GPU Backward Parity", RunSwiGLUGPUBackward},
		}
	case "RNN":
		tests = []testEntry{
			{"L1 Caching (CPU SC/MC + GPU SC/MC)", RunRNNL1Caching},
			{"Training (4 tiled modes × 21 types)", RunRNNTraining},
			{"GPU Forward Parity", RunRNNGPUForward},
			{"GPU Backward Parity", RunRNNGPUBackward},
		}
	case "LSTM":
		tests = []testEntry{
			{"L1 Caching (CPU SC/MC + GPU SC/MC)", RunLSTML1Caching},
			{"Training (4 tiled modes × 21 types)", RunLSTMTraining},
			{"GPU Forward Parity", RunLSTMGPUForward},
			{"GPU Backward Parity", RunLSTMGPUBackward},
		}
	case "Embedding":
		tests = []testEntry{
			{"L1 Caching (CPU SC/MC + GPU SC/MC)", RunEmbeddingL1Caching},
			{"Training (4 tiled modes × 21 types)", RunEmbeddingTraining},
			{"GPU Forward Parity", RunEmbeddingGPUForward},
			{"GPU Backward Parity", RunEmbeddingGPUBackward},
		}
	case "Residual":
		tests = []testEntry{
			{"L1 Caching (CPU SC/MC + GPU SC/MC)", RunResidualL1Caching},
			{"Training (4 tiled modes × 21 types)", RunResidualTraining},
			{"GPU Forward Parity", RunResidualGPUForward},
			{"GPU Backward Parity", RunResidualGPUBackward},
		}
	default:
		fmt.Printf("No tests registered for layer: %s\n", layerName)
		return
	}

	if testInput == "" {
		fmt.Printf("\n  Tests for %s:\n", layerName)
		fmt.Println("  [0] All")
		for i, t := range tests {
			fmt.Printf("  [%d] %s\n", i+1, t.name)
		}
		testInput = readPrompt(reader, "Select test [0]: ", "0")
	}

	if testInput == "0" {
		for _, t := range tests {
			fmt.Printf("\n--- %s ---\n", t.name)
			t.fn()
		}
		return
	}

	ti, err := strconv.Atoi(testInput)
	if err != nil || ti < 1 || ti > len(tests) {
		fmt.Println("Invalid selection.")
		return
	}
	t := tests[ti-1]
	fmt.Printf("\n--- %s ---\n", t.name)
	t.fn()
}
