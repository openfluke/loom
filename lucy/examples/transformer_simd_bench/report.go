package tfsimdbench

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

// OutputDir holds all logs + machine-readable results (gitignored under lucy/).
const OutputDir = "lucy_testing_output/transformer_simd_bench"

// LogFile is the tee log for a full run.
const LogFile = "transformer_simd_bench.txt"

// ResultsJSON is written at the end of each run.
const ResultsJSON = "results.json"

type runSummary struct {
	StartedAt  time.Time     `json:"started_at"`
	FinishedAt time.Time     `json:"finished_at"`
	Host       hostInfo      `json:"host"`
	Cells      []BenchResult `json:"cells"`
}

type hostInfo struct {
	Arch     string `json:"arch"`
	NumCPU   int    `json:"num_cpu"`
	SimdKind string `json:"simd_kind"`
}

var globalSummary runSummary

func resetSummary() {
	globalSummary = runSummary{StartedAt: time.Now(), Host: currentHostInfo()}
}

func recordCell(r BenchResult) {
	globalSummary.Cells = append(globalSummary.Cells, r)
}

// BeginSession resets OutputDir and tees stdout to the log + terminal.
func BeginSession() func() {
	resetSummary()
	_ = os.RemoveAll(OutputDir)
	_ = os.MkdirAll(OutputDir, 0o755)

	logPath := filepath.Join(OutputDir, LogFile)
	logFile, err := os.Create(logPath)
	if err != nil {
		fmt.Printf("Warning: could not create %s: %v\n", logPath, err)
		return func() {}
	}

	r, w, err := os.Pipe()
	if err != nil {
		_ = logFile.Close()
		return func() {}
	}

	orig := os.Stdout
	os.Stdout = w
	done := make(chan struct{})
	go func() {
		mw := io.MultiWriter(orig, logFile)
		buf := make([]byte, 4096)
		for {
			n, e := r.Read(buf)
			if n > 0 {
				_, _ = mw.Write(buf[:n])
			}
			if e != nil {
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
		os.Stdout = orig
		writeResultsJSON()
		fmt.Printf("\n📄 Bench log:    %s\n", logPath)
		fmt.Printf("📄 Results JSON: %s\n", filepath.Join(OutputDir, ResultsJSON))
	}
}

func writeResultsJSON() {
	globalSummary.FinishedAt = time.Now()
	path := filepath.Join(OutputDir, ResultsJSON)
	data, err := json.MarshalIndent(globalSummary, "", "  ")
	if err != nil {
		fmt.Printf("Warning: could not marshal results: %v\n", err)
		return
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		fmt.Printf("Warning: could not write %s: %v\n", path, err)
	}
}

// saveGeneration writes one cell's reply to outputs/ for manual inspection.
func saveGeneration(model, profile, body string) string {
	dir := filepath.Join(OutputDir, "outputs")
	_ = os.MkdirAll(dir, 0o755)
	name := fmt.Sprintf("%s__%s.txt", model, profile)
	path := filepath.Join(dir, name)
	header := fmt.Sprintf("# transformer_simd_bench\n# model=%s profile=%s saved=%s\n\n",
		model, profile, time.Now().Format(time.RFC3339))
	_ = os.WriteFile(path, []byte(header+body), 0o644)
	return path
}
