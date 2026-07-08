// Package contextsuite runs automated long-context and multi-prompt inference tests
// against .entity checkpoints (Lucy menu [10]).
package contextsuite

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

// OutputDir holds all context-suite logs and saved generation outputs (gitignored via lucy/.gitignore).
const OutputDir = "lucy_testing_output/context_suite"

// LogFile is the main tee log for a full suite run.
const LogFile = "context_suite.txt"

// ResultsJSON is the machine-readable summary written at the end of each run.
const ResultsJSON = "results.json"

// BeginSession resets OutputDir and tees stdout to context_suite.txt + terminal.
func BeginSession() func() {
	ResetSummary()
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
		PrintSummary()
		fmt.Printf("\n📄 Context suite log: %s\n", logPath)
		fmt.Printf("📄 Results JSON:      %s\n", filepath.Join(OutputDir, ResultsJSON))
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

// SaveGeneration writes one test cell's prompt + reply to a unique file under outputs/.
func SaveGeneration(model, execProfile, scenario string, body string) string {
	dir := filepath.Join(OutputDir, "outputs")
	_ = os.MkdirAll(dir, 0o755)
	name := fmt.Sprintf("%s__%s__%s.txt", model, execProfile, scenario)
	path := filepath.Join(dir, name)
	header := fmt.Sprintf("# context_suite generation\n# model=%s exec=%s scenario=%s\n# saved=%s\n\n",
		model, execProfile, scenario, time.Now().Format(time.RFC3339))
	_ = os.WriteFile(path, []byte(header+body), 0o644)
	return path
}
