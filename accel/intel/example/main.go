// MatMul stack — Loom CPU vs Intel CPU vs Intel NPU with dtype-aware weight upload.
//
// Only layers that upload weights via SyncToAccel (MatMul / Conv / MHA-MatMul). No baked-graph
// ops (ReLU, LayerNorm, Softmax) — those ignore Loom weights on Intel.
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
	sizeLabel  = "medium" // matches accel/intel layer_models shapes
	batch      = 16
	dim        = 256
	numMatMuls = 3
	weightSeed = int64(42)
	warmup     = 3
	iters      = 20
)

type benchResult struct {
	label   string
	median  float64
	p95     float64
	compile float64
	drift   float64
	ok      bool
	note    string
}

func main() {
	flag.Parse()
	ensureOVEnv()

	pluginPath := accel.DefaultIntelPath()
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║  MatMul stack — same Loom weights → Intel CPU / NPU          ║")
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
	fmt.Printf("  Network: %d× MatMul (Dense LINEAR), FP32, %d×%d — weights via SyncToAccel\n\n",
		numMatMuls, batch, dim)

	net, err := buildNet()
	if err != nil {
		fmt.Println("build:", err)
		return
	}
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
		r := benchMode(net, input, reg, mode.target, mode.label)
		if r.ok && len(baseline) == 0 {
			baseline = lastForward(net, input)
		} else if r.ok && len(baseline) > 0 {
			out := lastForward(net, input)
			if len(out) > 0 {
				r.drift = maxAbsDiff(baseline, out)
			}
		}
		rows = append(rows, r)
	}

	printTable(rows)
}

func npuStatus(ok bool) string {
	if ok {
		return "available"
	}
	return "not detected (Intel CPU path still runs)"
}

func buildNet() (*poly.VolumetricNetwork, error) {
	spec := buildMatMulSpec()
	net, err := poly.BuildNetworkFromJSON(spec)
	if err != nil {
		return nil, err
	}
	if err := poly.ConfigureNetworkForMode(net, poly.TrainingModeCPUMC); err != nil {
		return nil, err
	}
	pinWeights(net, weightSeed)
	return net, nil
}

// pinWeights replaces time-based init with a fixed seed so every backend sees identical Master slabs.
func pinWeights(net *poly.VolumetricNetwork, seed int64) {
	for i := range net.Layers {
		if net.Layers[i].WeightStore != nil {
			net.Layers[i].WeightStore.Randomize(seed+int64(i), 0.1)
		}
	}
}

func buildMatMulSpec() []byte {
	layers := make([]map[string]any, numMatMuls)
	for i := 0; i < numMatMuls; i++ {
		layers[i] = map[string]any{
			"z": 0, "y": 0, "x": 0, "l": i,
			"type": "DENSE", "activation": "LINEAR", "dtype": "FLOAT32",
			"input_height": dim, "output_height": dim,
		}
	}
	spec := map[string]any{
		"id": "npu-example-matmul", "depth": 1, "rows": 1, "cols": 1,
		"layers_per_cell": numMatMuls, "layers": layers,
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

func prepareTarget(net *poly.VolumetricNetwork, reg *accel.Registry, target accel.ExecTarget) (compileMs float64, err error) {
	net.Accel = reg
	for i := range net.Layers {
		net.Layers[i].ExecTarget = target
		if net.Layers[i].AccelBinding != nil {
			net.Layers[i].AccelBinding.Release()
			net.Layers[i].AccelBinding = nil
		}
	}
	if !target.UseAccel() {
		return 0, nil
	}
	if err := net.SyncToAccel(sizeLabel); err != nil {
		return 0, err
	}
	for i := range net.Layers {
		if net.Layers[i].AccelBinding != nil {
			compileMs += net.Layers[i].AccelBinding.CompileMs
		}
	}
	return compileMs, nil
}

func benchMode(net *poly.VolumetricNetwork, input *poly.Tensor[float32], reg *accel.Registry, target accel.ExecTarget, label string) benchResult {
	compileMs, err := prepareTarget(net, reg, target)
	if err != nil {
		return benchResult{label: label, note: err.Error()}
	}

	for w := 0; w < warmup; w++ {
		resetNet(net)
		if out, _, _ := poly.ForwardPolymorphic(net, input); out == nil {
			return benchResult{label: label, note: "warmup forward failed"}
		}
	}

	samples := make([]float64, 0, iters)
	for i := 0; i < iters; i++ {
		resetNet(net)
		t0 := time.Now()
		out, _, _ := poly.ForwardPolymorphic(net, input)
		if out == nil {
			return benchResult{label: label, note: "forward failed"}
		}
		samples = append(samples, float64(time.Since(t0).Microseconds())/1000.0)
	}

	return benchResult{
		label:   label,
		median:  median(samples),
		p95:     p95(samples),
		compile: compileMs,
		ok:      true,
		note:    "OK",
	}
}

func lastForward(net *poly.VolumetricNetwork, input *poly.Tensor[float32]) []float32 {
	resetNet(net)
	out, _, _ := poly.ForwardPolymorphic(net, input)
	if out == nil {
		return nil
	}
	return append([]float32(nil), out.Data...)
}

func resetNet(net *poly.VolumetricNetwork) {
	for i := range net.Layers {
		net.Layers[i].ResetState()
	}
}

func printTable(rows []benchResult) {
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
			if base.label == "Loom CPU" && base.ok && base.median > 0 && r.label != "Loom CPU" {
				speedup = fmt.Sprintf(" (%.2fx)", base.median/r.median)
				break
			}
		}
		driftStr := "—"
		if r.label != "Loom CPU" {
			driftStr = fmt.Sprintf("%.2e", r.drift)
		}
		fmt.Printf("  │ %-11s │ %8.3f │ %8.3f │ %9.2f │ %-12s │ OK     │\n",
			r.label+speedup, r.median, r.p95, r.compile, driftStr)
	}
	fmt.Println("  └─────────────┴──────────┴──────────┴───────────┴──────────────┴────────┘")
	fmt.Println("\n  Weights: one network, fixed seed — SyncToAccel uploads per-layer dtype bytes.")
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
