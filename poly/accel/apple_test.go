//go:build darwin && cgo

package accel_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/openfluke/loom/poly/accel"
)

func f32bytes(v []float32) []byte {
	out := make([]byte, len(v)*4)
	for i, x := range v {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(x))
	}
	return out
}

func f32from(b []byte) []float32 {
	n := len(b) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

func runMatMul(t *testing.T, p accel.Plugin) []float32 {
	t.Helper()
	// small tier: dense_batch=4, dim=32. Identity weight → output == input.
	const B, D = 4, 32
	weights := make([]float32, D*D)
	for i := 0; i < D; i++ {
		weights[i*D+i] = 1.0 // identity
	}
	compiled, err := p.CompileLayer(accel.LayerDesc{
		LayerName: "MatMul",
		DType:     "FP32",
		SizeLabel: "small",
	}, f32bytes(weights))
	if err != nil {
		t.Fatal(err)
	}
	defer compiled.Layer.Release()

	in := make([]float32, B*D)
	for i := range in {
		in[i] = float32(i) * 0.01
	}
	inB := f32bytes(in)
	outB := make([]byte, compiled.OutBytes)
	if uintptr(len(inB)) != compiled.InBytes {
		t.Fatalf("in bytes = %d want %d", len(inB), compiled.InBytes)
	}
	if _, err := compiled.Layer.Infer(inB, outB); err != nil {
		t.Fatal(err)
	}
	out := f32from(outB)
	// identity matmul → out ≈ in
	for i := range in {
		if math.Abs(float64(out[i]-in[i])) > 1e-4 {
			t.Fatalf("identity matmul mismatch at %d: got %v want %v", i, out[i], in[i])
		}
	}
	return out
}

func bf16bytes(v []float32) []byte {
	out := make([]byte, len(v)*2)
	for i, x := range v {
		bits := math.Float32bits(x)
		var h uint16
		if bits&0x7fffffff > 0x7f800000 {
			h = uint16(bits>>16) | 0x0040
		} else {
			lsb := (bits >> 16) & 1
			bits += 0x7fff + lsb
			h = uint16(bits >> 16)
		}
		binary.LittleEndian.PutUint16(out[i*2:], h)
	}
	return out
}

func bf16from(b []byte) []float32 {
	n := len(b) / 2
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[i*2:])) << 16)
	}
	return out
}

// BF16 is an Apple-advertised dtype the shared bridge now supports. Weights and
// I/O cross the C ABI as 2-byte bfloat16; compute is fp32. Identity MatMul must
// round-trip within bf16 precision (~1/128 relative).
func TestAppleBF16MatMul(t *testing.T) {
	p, err := accel.OpenApple("", "CPU")
	if err != nil {
		t.Skipf("apple plugin unavailable: %v", err)
	}
	defer p.Close()

	const B, D = 4, 32
	weights := make([]float32, D*D)
	for i := 0; i < D; i++ {
		weights[i*D+i] = 1.0
	}
	compiled, err := p.CompileLayer(accel.LayerDesc{
		LayerName: "MatMul",
		DType:     "BF16",
		SizeLabel: "small",
	}, bf16bytes(weights))
	if err != nil {
		t.Fatal(err)
	}
	defer compiled.Layer.Release()

	in := make([]float32, B*D)
	for i := range in {
		in[i] = float32(i) * 0.01
	}
	inB := bf16bytes(in)
	if uintptr(len(inB)) != compiled.InBytes {
		t.Fatalf("bf16 in bytes = %d want %d", len(inB), compiled.InBytes)
	}
	outB := make([]byte, compiled.OutBytes)
	if _, err := compiled.Layer.Infer(inB, outB); err != nil {
		t.Fatal(err)
	}
	out := bf16from(outB)
	wantIn := bf16from(inB) // input as the plugin actually sees it
	for i := range out {
		tol := 0.01 + 0.01*math.Abs(float64(wantIn[i]))
		if math.Abs(float64(out[i]-wantIn[i])) > tol {
			t.Fatalf("bf16 identity matmul mismatch at %d: got %v want %v", i, out[i], wantIn[i])
		}
	}
}

func TestAppleCPUMatMul(t *testing.T) {
	p, err := accel.OpenApple("", "CPU")
	if err != nil {
		t.Skipf("apple plugin unavailable: %v", err)
	}
	defer p.Close()
	if p.VendorID() != "apple" {
		t.Fatalf("vendor=%q", p.VendorID())
	}
	runMatMul(t, p)
}

func TestAppleGPUMatMul(t *testing.T) {
	if !accel.AppleGPUAvailable("") {
		t.Skip("no Metal GPU")
	}
	p, err := accel.OpenApple("", "GPU")
	if err != nil {
		t.Skipf("apple GPU unavailable: %v", err)
	}
	defer p.Close()
	runMatMul(t, p)
}

// inferLayer compiles + runs one layer and returns the fp32 output.
func inferLayer(t *testing.T, p accel.Plugin, name string, in []float32) []float32 {
	t.Helper()
	compiled, err := p.CompileLayer(accel.LayerDesc{
		LayerName: name,
		DType:     "FP32",
		SizeLabel: "small",
	}, nil)
	if err != nil {
		t.Fatalf("%s compile: %v", name, err)
	}
	defer compiled.Layer.Release()
	inB := f32bytes(in)
	if uintptr(len(inB)) != compiled.InBytes {
		t.Fatalf("%s in bytes = %d want %d", name, len(inB), compiled.InBytes)
	}
	outB := make([]byte, compiled.OutBytes)
	if _, err := compiled.Layer.Infer(inB, outB); err != nil {
		t.Fatalf("%s infer: %v", name, err)
	}
	return f32from(outB)
}

// On the GPU device: ReLU is MPSGraph-accelerated, Conv2D/LayerNorm silently fall
// back to the CPU reference. All must return correct values.
func TestAppleGPUReLUAndFallback(t *testing.T) {
	if !accel.AppleGPUAvailable("") {
		t.Skip("no Metal GPU")
	}
	p, err := accel.OpenApple("", "GPU")
	if err != nil {
		t.Skipf("apple GPU unavailable: %v", err)
	}
	defer p.Close()

	// ReLU (GPU) on [4,32].
	relIn := make([]float32, 4*32)
	for i := range relIn {
		relIn[i] = float32(i-64) * 0.1 // mix of negative + positive
	}
	relOut := inferLayer(t, p, "ReLU", relIn)
	for i := range relIn {
		want := relIn[i]
		if want < 0 {
			want = 0
		}
		if math.Abs(float64(relOut[i]-want)) > 1e-5 {
			t.Fatalf("relu mismatch at %d: got %v want %v", i, relOut[i], want)
		}
	}

	// Conv2D (CPU fallback on the GPU device) — small tier input [4,2,4,4].
	convIn := make([]float32, 4*2*4*4)
	for i := range convIn {
		convIn[i] = 1.0
	}
	convOut := inferLayer(t, p, "Conv2D", convIn)
	if len(convOut) != 4*2*4*4 { // filters=2,h=4,w=4
		t.Fatalf("conv2d out len = %d want %d", len(convOut), 4*2*4*4)
	}

	// LayerNorm (CPU fallback) — each row should be ~zero mean.
	lnIn := make([]float32, 4*32)
	for i := range lnIn {
		lnIn[i] = float32(i%32) * 0.5
	}
	lnOut := inferLayer(t, p, "LayerNorm", lnIn)
	for b := 0; b < 4; b++ {
		var sum float64
		for j := 0; j < 32; j++ {
			sum += float64(lnOut[b*32+j])
		}
		if math.Abs(sum/32.0) > 1e-3 {
			t.Fatalf("layernorm row %d mean = %v, want ~0", b, sum/32.0)
		}
	}
}
