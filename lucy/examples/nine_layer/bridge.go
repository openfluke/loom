package ninelayer

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/openfluke/loom/poly"
	"github.com/openfluke/loom/poly/accel"
)

type bridgeRow struct {
	Size         string
	Layer        string
	DType        string
	Device       string
	OK           bool
	LoomMs       float64
	HandoverMs   float64
	CompileMs    float64
	FirstInferMs float64
	MedianMs     float64
	P95Ms        float64
	Note         string
}

// RunBridgeSuite exercises Loom → Intel CABI handover for all manifest layers/dtypes.
// sizes nil = all tiers from manifest. onlyLayer filters to one layer name.
func RunBridgeSuite(sizes []string, onlyLayer ...string) {
	m, err := LoadManifest()
	if err != nil {
		fmt.Println("manifest:", err)
		return
	}

	path := DefaultPluginPath()
	if err := accel.PrepareRuntime(); err != nil {
		fmt.Println("OpenVINO runtime:", err)
		fmt.Println("Try: source accel/intel/setup_env.sh  (from Loom repo root)")
		return
	}
	cpuPlug, err := openPlugin("CPU")
	if err != nil {
		fmt.Println("Intel CABI CPU:", err)
		return
	}
	defer cpuPlug.Close()

	var npuPlug accel.Plugin
	if NPUReady(path) {
		npuPlug, err = openPlugin("NPU")
		if err != nil {
			fmt.Printf("NPU plugin: %v (CPU-only run)\n", err)
		} else {
			defer npuPlug.Close()
		}
	}

	sizeOrder := m.SizeOrder
	if len(sizes) > 0 {
		sizeOrder = sizes
	}

	layerFilter := ""
	if len(onlyLayer) > 0 {
		layerFilter = onlyLayer[0]
	}

	var rows []bridgeRow
	for _, sizeName := range sizeOrder {
		profile, ok := m.Sizes[sizeName]
		if !ok {
			continue
		}
		warmup, iters := benchLimits(m, sizeName)

		for _, layer := range m.Layers {
			if layerFilter != "" && layer.Name != layerFilter {
				continue
			}
			for _, dtypeLabel := range m.DTypes {
				base := bridgeRow{Size: sizeName, Layer: layer.Name, DType: dtypeLabel}

				var weights []byte
				var inBuf []byte
				if layer.Loom != nil {
					var note string
					base.LoomMs, base.HandoverMs, weights, inBuf, note = loomForwardAndWeights(layer, profile, dtypeLabel, warmup, iters)
					if note != "" {
						base.Note = note
					}
				}

				desc := accel.LayerDesc{LayerName: layer.Name, DType: dtypeLabel, SizeLabel: sizeName}

				cpuRow := base
				cpuRow.Device = "CABI-CPU"
				fillCABI(&cpuRow, cpuPlug, desc, weights, inBuf, dtypeLabel, warmup, iters)
				rows = append(rows, cpuRow)

				if npuPlug != nil {
					npuRow := base
					npuRow.Device = "CABI-NPU"
					fillCABI(&npuRow, npuPlug, desc, weights, inBuf, dtypeLabel, warmup, iters)
					rows = append(rows, npuRow)
				}
			}
		}
	}

	printBridgeTable(rows, m)
}

func fillCABI(row *bridgeRow, plug accel.Plugin, desc accel.LayerDesc, weights []byte, inBuf []byte, dtypeLabel string, warmup, iters int) {
	compiled, err := plug.CompileLayer(desc, weights)
	if err != nil {
		row.Note = err.Error()
		return
	}
	defer compiled.Layer.Release()

	row.CompileMs = compiled.CompileMs
	row.FirstInferMs = compiled.FirstInferMs

	outBuf := make([]byte, compiled.OutBytes)
	if len(inBuf) == 0 {
		inBuf = syntheticInput(int(compiled.InBytes), dtypeLabel)
		row.HandoverMs = 0
	} else if len(inBuf) != int(compiled.InBytes) {
		row.Note = fmt.Sprintf("input size %d != cabi %d", len(inBuf), compiled.InBytes)
		return
	}

	for i := 0; i < warmup; i++ {
		if _, err := compiled.Layer.Infer(inBuf, outBuf); err != nil {
			row.Note = err.Error()
			return
		}
	}

	samples := make([]float64, 0, iters)
	for i := 0; i < iters; i++ {
		res, err := compiled.Layer.Infer(inBuf, outBuf)
		if err != nil {
			row.Note = err.Error()
			return
		}
		samples = append(samples, res.InferMs)
	}

	row.OK = true
	row.MedianMs = median(samples)
	row.P95Ms = percentile(samples, 0.95)
}

func loomForwardAndWeights(
	layer ManifestLayer,
	profile SizeProfile,
	dtypeLabel string,
	warmup, iters int,
) (loomMs, handoverMs float64, weights []byte, inBuf []byte, note string) {
	if layer.Loom == nil {
		return 0, 0, nil, nil, "no Loom layer type"
	}
	dt, ok := parseDType(dtypeLabel)
	if !ok {
		return 0, 0, nil, nil, "unsupported dtype"
	}

	spec, err := buildLoomSpec(layer, profile, dt)
	if err != nil {
		return 0, 0, nil, nil, err.Error()
	}
	net, err := poly.BuildNetworkFromJSON(spec)
	if err != nil {
		return 0, 0, nil, nil, err.Error()
	}
	if err := poly.ConfigureNetworkForMode(net, poly.TrainingModeCPUMC); err != nil {
		return 0, 0, nil, nil, err.Error()
	}

	inKind := inputKind(layer.Name)
	for i := 0; i < warmup; i++ {
		resetNet(net)
		if !forwardOK(net, inKind, profile, dt) {
			return 0, 0, nil, nil, "loom forward failed"
		}
	}

	samples := make([]float64, 0, iters)
	for i := 0; i < iters; i++ {
		resetNet(net)
		t0 := time.Now()
		if !forwardOK(net, inKind, profile, dt) {
			return 0, 0, nil, nil, "loom forward failed"
		}
		samples = append(samples, float64(time.Since(t0).Microseconds())/1000.0)
	}
	loomMs = median(samples)

	if len(net.Layers) > 0 {
		weights = poly.LayerWeightBytesForAccel(&net.Layers[0])
	}

	inAny := makeLayerInput(inKind, profile, dt)
	inBuf, err = tensorToBytes(inAny, dtypeLabel)
	if err != nil {
		return loomMs, 0, weights, nil, err.Error()
	}

	t0 := time.Now()
	_ = append([]byte(nil), inBuf...)
	handoverMs = float64(time.Since(t0).Microseconds()) / 1000.0

	return loomMs, handoverMs, weights, inBuf, ""
}

func benchLimits(m BenchManifest, sizeName string) (warmup, iters int) {
	warmup, iters = 3, 20
	if v, ok := m.WarmupBySize[sizeName]; ok {
		warmup = v
	}
	if v, ok := m.ItersBySize[sizeName]; ok {
		iters = v
	}
	if warmup < 1 {
		warmup = 1
	}
	if iters < 1 {
		iters = 1
	}
	return warmup, iters
}

func printBridgeTable(rows []bridgeRow, m BenchManifest) {
	current := ""
	for _, r := range rows {
		if r.Size != current {
			current = r.Size
			note := ""
			if p, ok := m.Sizes[current]; ok {
				note = p.Note
			}
			fmt.Printf("\n=== size: %s — %s ===\n", current, note)
			fmt.Printf("%-14s %-6s %-10s %-8s %-8s %-8s %-8s %-8s %-8s %s\n",
				"Layer", "DType", "Device", "Loom", "Handov", "Compile", "1stInf", "Infer", "P95", "Status")
			fmt.Println(strings.Repeat("-", 110))
		}
		if r.Device == "" {
			fmt.Printf("%-14s %-6s %-10s %-8.3f %-8.3f %-8s %-8s %-8s %-8s %s\n",
				r.Layer, r.DType, "Loom-CPU", r.LoomMs, r.HandoverMs, "-", "-", "-", "-", statusNote(r))
			continue
		}
		if r.OK {
			fmt.Printf("%-14s %-6s %-10s %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f OK\n",
				r.Layer, r.DType, r.Device, r.LoomMs, r.HandoverMs, r.CompileMs, r.FirstInferMs, r.MedianMs, r.P95Ms)
		} else {
			fmt.Printf("%-14s %-6s %-10s %-8.3f %-8.3f %-8s %-8s %-8s %-8s FAIL: %s\n",
				r.Layer, r.DType, r.Device, r.LoomMs, r.HandoverMs, "-", "-", "-", "-", r.Note)
		}
	}
}

func statusNote(r bridgeRow) string {
	if r.Note != "" {
		return r.Note
	}
	return "OK"
}

func median(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	return s[len(s)/2]
}

func percentile(v []float64, p float64) float64 {
	if len(v) == 0 {
		return 0
	}
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	idx := int(p * float64(len(s)-1))
	return s[idx]
}
