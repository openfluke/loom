//go:build linux && cgo

package accel_test

import (
	"os"
	"testing"

	"github.com/openfluke/loom/poly/accel"
)

func TestOpenIntelCPU(t *testing.T) {
	path := os.Getenv("LOOM_ACCEL_INTEL_SO")
	if path == "" {
		t.Skip("set LOOM_ACCEL_INTEL_SO")
	}
	p, err := accel.OpenIntel(path, "CPU")
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if p.VendorID() != "intel" {
		t.Fatalf("vendor=%q", p.VendorID())
	}
	compiled, err := p.CompileLayer(accel.LayerDesc{
		LayerName: "MatMul",
		DType:     "FP32",
		SizeLabel: "small",
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer compiled.Layer.Release()
	in := make([]byte, compiled.InBytes)
	out := make([]byte, compiled.OutBytes)
	if _, err := compiled.Layer.Infer(in, out); err != nil {
		t.Fatal(err)
	}
}
