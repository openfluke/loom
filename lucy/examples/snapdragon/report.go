package snapdragon

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
)

const OutputDir = "lucy_testing_output"
const LogFile = "snapdragon.txt"

// BeginSession wipes lucy_testing_output and tees stdout to snapdragon.txt + terminal.
func BeginSession() func() {
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
		fmt.Printf("\n📄 Bridge log: %s\n", logPath)
	}
}
