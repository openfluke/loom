// Monolithic MLP — Loom CPU vs Intel OpenVINO CPU vs Intel NPU (same weights, full-network forward).
//
// Requires Linux, CGO_ENABLED=1, libloom_accel_intel.so, OpenVINO + NPU driver.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/openfluke/loom/poly"
	"github.com/openfluke/loom/poly/accel"
)

const (
	sizeLabel = "medium" // matches loom/accel/intel layer_models shapes
	batch     = 16
	dim       = 256
	warmup    = 3
	iters     = 20
)

type benchResult struct {
	label   string
	median  float64
	p95     float64
	compile float64
	drift   float64
	samples []float32
	ok      bool
	note    string
}

func main() {
	flag.Parse()
	ensureOVEnv()

	pluginPath := accel.DefaultIntelPath()
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║  Monolithic MLP — Loom CPU vs Intel CPU vs Intel NPU         ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Printf("  Plugin: %s\n", pluginPath)

	reg, regErr := poly.DiscoverAccel(accel.AccelConfig{IntelSO: pluginPath})
	hasPlugin := regErr == nil
	if !hasPlugin {
		fmt.Printf("  Accel: unavailable (%v)\n", regErr)
		fmt.Println("  Running Loom CPU only. On Linux+NPU: build loom/accel/intel first.")
	} else {
		defer reg.Close()
		fmt.Printf("  NPU: %s\n", npuStatus(reg.IntelNPU != nil))
	}
	fmt.Printf("  Network: 5-layer MLP (ReLU×2 → MatMul → LayerNorm → Softmax), FP32, %dx%d\n\n", batch, dim)

	spec := buildMLPSpec()
	input := makeInput()

	modes := []struct {
		label  string
		target accel.ExecTarget
		skip   bool
	}{
		{"Loom CPU", accel.ExecLoomCPU, false},
		{"Intel CPU", accel.ExecIntelCPU, !hasPlugin},
		{"Intel NPU", accel.ExecIntelNPU, !hasPlugin || reg.IntelNPU == nil},
	}

	var baseline []float32
	var rows []benchResult

	for _, mode := range modes {
		if mode.skip {
			note := "NPU unavailable"
			if !hasPlugin {
				note = "plugin unavailable"
			}
			rows = append(rows, benchResult{label: mode.label, note: note})
			continue
		}
		r := runMode(spec, input, reg, mode.target, mode.label)
		if r.ok && len(baseline) == 0 {
			baseline = r.samples
		} else if r.ok && len(baseline) > 0 {
			r.drift = maxAbsDiff(baseline, r.samples)
		}
		rows = append(rows, r)
	}

	printTable(rows, baseline)
}

func npuStatus(ok bool) string {
	if ok {
		return "available"
	}
	return "not detected (Intel CPU path still runs)"
}

func buildMLPSpec() []byte {
	layers := []map[string]any{
		{"z": 0, "y": 0, "x": 0, "l": 0, "type": "DENSE", "activation": "RELU", "dtype": "FLOAT32",
			"input_height": dim, "output_height": dim},
		{"z": 0, "y": 0, "x": 0, "l": 1, "type": "DENSE", "activation": "RELU", "dtype": "FLOAT32",
			"input_height": dim, "output_height": dim},
		{"z": 0, "y": 0, "x": 0, "l": 2, "type": "DENSE", "activation": "LINEAR", "dtype": "FLOAT32",
			"input_height": dim, "output_height": dim},
		{"z": 0, "y": 0, "x": 0, "l": 3, "type": "LAYERNORM", "activation": "RELU", "dtype": "FLOAT32",
			"input_height": dim, "output_height": dim},
		{"z": 0, "y": 0, "x": 0, "l": 4, "type": "SOFTMAX", "activation": "RELU", "dtype": "FLOAT32",
			"input_height": dim, "output_height": dim,
			"softmax_type": 1, "softmax_rows": batch, "softmax_cols": dim},
	}
	spec := map[string]any{
		"id": "npu-example-mlp", "depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 5,
		"layers": layers,
	}
	b, _ := json.Marshal(spec)
	return b
}

func makeInput() *poly.Tensor[float32] {
	data := make([]float32, batch*dim)
	for i := range data {
		data[i] = 0.01 * float32(i%97)
	}
	return poly.NewTensorFromSlice(data, batch, dim)
}

func runMode(spec []byte, input *poly.Tensor[float32], reg *accel.Registry, target accel.ExecTarget, label string) benchResult {
	net, err := poly.BuildNetworkFromJSON(spec)
	if err != nil {
		return benchResult{label: label, note: err.Error()}
	}
	if err := poly.ConfigureNetworkForMode(net, poly.TrainingModeCPUMC); err != nil {
		return benchResult{label: label, note: err.Error()}
	}
	net.Accel = reg
	for i := range net.Layers {
		net.Layers[i].ExecTarget = target
		net.Layers[i].AccelBinding = nil
	}

	var compileMs float64
	if target.UseAccel() {
		if err := net.SyncToAccel(sizeLabel); err != nil {
			return benchResult{label: label, note: "SyncToAccel: " + err.Error()}
		}
		for i := range net.Layers {
			if net.Layers[i].AccelBinding != nil {
				compileMs += net.Layers[i].AccelBinding.CompileMs
			}
		}
	}

	for w := 0; w < warmup; w++ {
		resetNet(net)
		out, _, _ := poly.ForwardPolymorphic(net, input)
		if out == nil {
			return benchResult{label: label, note: "warmup forward failed"}
		}
	}

	samples := make([]float64, 0, iters)
	var last *poly.Tensor[float32]
	for i := 0; i < iters; i++ {
		resetNet(net)
		t0 := time.Now()
		out, _, _ := poly.ForwardPolymorphic(net, input)
		if out == nil {
			return benchResult{label: label, note: "forward failed"}
		}
		samples = append(samples, float64(time.Since(t0).Microseconds())/1000.0)
		last = out
	}

	outData := append([]float32(nil), last.Data...)
	return benchResult{
		label:   label,
		median:  median(samples),
		p95:     p95(samples),
		compile: compileMs,
		samples: outData,
		ok:      true,
		note:    "OK",
	}
}

func resetNet(net *poly.VolumetricNetwork) {
	for i := range net.Layers {
		net.Layers[i].ResetState()
	}
}

func printTable(rows []benchResult, baseline []float32) {
	fmt.Println("  ┌─────────────┬──────────┬──────────┬───────────┬──────────────┬────────┐")
	fmt.Println("  │ Backend     │ median ms│ p95 ms   │ compile ms│ vs Loom drift│ status │")
	fmt.Println("  ├─────────────┼──────────┼──────────┼───────────┼──────────────┼────────┤")
	for _, r := range rows {
		if !r.ok {
			fmt.Printf("  │ %-11s │    —     │    —     │     —     │      —       │ %-6s │\n", r.label, trunc(r.note, 6))
			continue
		}
		speedup := ""
		for _, base := range rows {
			if base.label == "Loom CPU" && base.ok && base.median > 0 {
				ratio := base.median / r.median
				if r.label != "Loom CPU" {
					speedup = fmt.Sprintf(" (%.2fx)", ratio)
				}
				break
			}
		}
		driftStr := "—"
		if r.label != "Loom CPU" && len(baseline) > 0 {
			driftStr = fmt.Sprintf("%.2e", r.drift)
		}
		fmt.Printf("  │ %-11s │ %8.3f │ %8.3f │ %9.2f │ %-12s │ OK     │\n",
			r.label+speedup, r.median, r.p95, r.compile, driftStr)
	}
	fmt.Println("  └─────────────┴──────────┴──────────┴───────────┴──────────────┴────────┘")

	if len(baseline) > 0 {
		fmt.Printf("\n  Loom CPU output (first 8): %v\n", baseline[:min(8, len(baseline))])
		for _, r := range rows {
			if r.ok && r.label != "Loom CPU" && len(r.samples) > 0 {
				fmt.Printf("  %s output (first 8): %v\n", r.label, r.samples[:min(8, len(r.samples))])
			}
		}
	}
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

func median(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	cp := append([]float64(nil), v...)
	sort.Float64s(cp)
	m := len(cp) / 2
	if len(cp)%2 == 0 {
		return (cp[m-1] + cp[m]) / 2
	}
	return cp[m]
}

func p95(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	cp := append([]float64(nil), v...)
	sort.Float64s(cp)
	idx := int(float64(len(cp)-1) * 0.95)
	return cp[idx]
}

func maxAbsDiff(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.MaxFloat64
	}
	var m float64
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		if d > m {
			m = d
		}
	}
	return m
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
