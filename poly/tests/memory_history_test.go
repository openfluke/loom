package poly_test

import (
	"bytes"
	"io"
	"os"
	"strings"
	"testing"

	. "github.com/openfluke/loom/poly"
)

func TestMemoryHistoryRecordAndTerminalChart(t *testing.T) {
	t.Setenv("LOOM_MEMORY_HISTORY", "1")
	h := &MemoryHistory{}
	h.BeginSession("test")
	h.Record("start", MemoryFootprint{HostWeightsMB: 100, GPUWeightsMB: 0}, 0)
	h.Record("mid", MemoryFootprint{HostWeightsMB: 100, GPUWeightsMB: 50}, 50<<20)
	h.Record("end", MemoryFootprint{HostWeightsMB: 0, GPUWeightsMB: 100}, 100<<20)

	samples := h.Samples()
	if len(samples) != 3 {
		t.Fatalf("expected 3 samples, got %d", len(samples))
	}
	if samples[1].HostWeightsMB != 100 || samples[1].GPUWeightsMB != 50 {
		t.Fatalf("unexpected mid sample: %+v", samples[1])
	}

	var buf bytes.Buffer
	h.WriteTerminalChart(&buf)
	out := buf.String()
	if !strings.Contains(out, "Memory history") {
		t.Fatalf("expected chart header, got: %q", out)
	}
	if !strings.Contains(out, "host weights") || !strings.Contains(out, "GPU weights") {
		t.Fatalf("expected series labels, got: %q", out)
	}
}

func TestPeakHostGPUOverlap(t *testing.T) {
	t.Setenv("LOOM_MEMORY_HISTORY", "1")
	h := &MemoryHistory{}
	h.BeginSession("overlap")
	h.Record("start", MemoryFootprint{HostWeightsMB: 100, GPUWeightsMB: 0}, 0)
	h.Record("mid", MemoryFootprint{HostWeightsMB: 100, GPUWeightsMB: 50}, 0)
	h.Record("end", MemoryFootprint{HostWeightsMB: 0, GPUWeightsMB: 100}, 0)

	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w
	h.PrintTerminalSummary()
	w.Close()
	os.Stdout = old

	var buf bytes.Buffer
	_, _ = io.Copy(&buf, r)
	out := buf.String()
	if !strings.Contains(out, "peak host+gpu Poly weights overlap: 150.00 MB") {
		t.Fatalf("expected overlap warning, got: %q", out)
	}
}
