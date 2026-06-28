//go:build linux

package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/openfluke/loom/poly/accel"
)

// ensureOVEnv re-execs once with OpenVINO/NPU libs on LD_LIBRARY_PATH (Linux reads it at process start).
func ensureOVEnv() {
	if os.Getenv("LOOM_NPU_EXAMPLE_ENV") == "1" {
		return
	}
	ld := accel.RuntimeLDLibraryPath()
	if ld == "" {
		return
	}
	cur := os.Getenv("LD_LIBRARY_PATH")
	if strings.Contains(cur, "tbb/lib") || strings.Contains(cur, "libopenvino") {
		return
	}
	if cur != "" {
		ld = ld + ":" + cur
	}
	cmd := exec.Command(os.Args[0], os.Args[1:]...)
	cmd.Env = envWithLDLibraryPath(ld, "LOOM_NPU_EXAMPLE_ENV=1")
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			os.Exit(ee.ExitCode())
		}
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	os.Exit(0)
}

func envWithLDLibraryPath(ld string, extra ...string) []string {
	out := make([]string, 0, len(os.Environ())+len(extra)+1)
	for _, e := range os.Environ() {
		if strings.HasPrefix(e, "LD_LIBRARY_PATH=") {
			continue
		}
		out = append(out, e)
	}
	out = append(out, "LD_LIBRARY_PATH="+ld)
	out = append(out, extra...)
	return out
}
