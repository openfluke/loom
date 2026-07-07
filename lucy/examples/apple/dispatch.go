package apple

import (
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/openfluke/loom/poly"
	"github.com/openfluke/loom/poly/accel"
)

type dispatchRow struct {
	Size          string
	Layer         string
	DType         string
	LoomMs        float64
	AppleCPUMs    float64
	MetalGPUMs    float64
	CPUDrift      float64
	GPUDrift      float64
	InferCPUDrift float64
	InferGPUDrift float64
	ParityCPUSpec driftSpectrum
	ParityGPUSpec driftSpectrum
	InferCPUSpec  driftSpectrum
	InferGPUSpec  driftSpectrum
	CompileCPU    float64
	CompileGPU    float64
	Note          string
}

// RunDispatchSuite exercises DispatchLayer → Apple CABI (init once, infer many).
func RunDispatchSuite(sizes []string) {
	resetDispatchSession()
	m, err := LoadManifest()
	if err != nil {
		fmt.Println("manifest:", err)
		return
	}
	reg, err := poly.DiscoverAppleAccel(accel.AccelConfig{AppleSO: DefaultPluginPath()})
	if err != nil {
		fmt.Println("DiscoverAppleAccel:", err)
		return
	}
	defer reg.Close()

	hasGPU := reg.AppleGPU != nil
	sizeOrder := m.SizeOrder
	if len(sizes) > 0 {
		sizeOrder = sizes
	}

	var rows []dispatchRow
	for _, sizeName := range sizeOrder {
		profile, ok := m.Sizes[sizeName]
		if !ok {
			continue
		}
		for _, layer := range m.Layers {
			if layer.Loom == nil {
				continue
			}
			for _, dtypeLabel := range m.DTypes {
				row := runDispatchCase(sizeName, layer, profile, dtypeLabel, reg, hasGPU)
				rows = append(rows, row)
			}
		}
	}
	printDispatchTable(rows, hasGPU)
}

func runDispatchCase(
	sizeName string,
	layer ManifestLayer,
	profile SizeProfile,
	dtypeLabel string,
	reg *accel.Registry,
	hasGPU bool,
) dispatchRow {
	row := dispatchRow{Size: sizeName, Layer: layer.Name, DType: dtypeLabel}
	dt, ok := parseDType(dtypeLabel)
	if !ok {
		row.Note = "bad dtype"
		return row
	}
	spec, err := buildLoomSpec(layer, profile, dt)
	if err != nil {
		row.Note = err.Error()
		return row
	}
	net, err := poly.BuildNetworkFromJSON(spec)
	if err != nil {
		row.Note = err.Error()
		return row
	}
	defer releaseDispatchNet(net)
	if err := poly.ConfigureNetworkForMode(net, poly.TrainingModeCPUMC); err != nil {
		row.Note = err.Error()
		return row
	}
	net.Accel = reg
	l := &net.Layers[0]
	inKind := inputKind(layer.Name)
	in := makeLayerInput(inKind, profile, dt)

	tol := parityTolerance(dtypeLabel)

	// Loom CPU baseline
	l.ExecTarget = accel.ExecLoomCPU
	l.AccelBinding = nil
	resetNet(net)
	loomOut, loomMs := captureDispatchForward(net, in, 10)
	row.LoomMs = loomMs

	// Apple CPU reference via DispatchLayer
	l.ExecTarget = accel.ExecAppleCPU
	if err := net.SyncToAccel(sizeName); err != nil {
		row.Note = "sync CPU: " + err.Error()
		return row
	}
	if l.AccelBinding != nil {
		row.CompileCPU = l.AccelBinding.CompileMs
	}
	resetNet(net)
	cpuOut1, cpuMs := captureDispatchForward(net, in, 10)
	row.AppleCPUMs = cpuMs
	resetNet(net)
	cpuOut2, _ := captureDispatchForward(net, in, 1)
	parityCPU := maxAbsDiffTensor(loomOut, cpuOut1)
	inferDetCPU := maxAbsDiffTensor(cpuOut1, cpuOut2)
	row.CPUDrift = parityCPU
	row.InferCPUDrift = inferDetCPU
	row.ParityCPUSpec = classifyDrift(parityCPU, tol)
	row.InferCPUSpec = classifyDrift(inferDetCPU, inferDriftTolerance(dtypeLabel))

	if hasGPU {
		l.ExecTarget = accel.ExecAppleGPU
		l.AccelBinding = nil
		if err := net.SyncToAccel(sizeName); err != nil {
			row.Note = "sync GPU: " + err.Error()
			return row
		}
		if l.AccelBinding != nil {
			row.CompileGPU = l.AccelBinding.CompileMs
		}
		resetNet(net)
		gpuOut1, gpuMs := captureDispatchForward(net, in, 10)
		row.MetalGPUMs = gpuMs
		resetNet(net)
		gpuOut2, _ := captureDispatchForward(net, in, 1)
		parityGPU := maxAbsDiffTensor(loomOut, gpuOut1)
		inferDetGPU := maxAbsDiffTensor(gpuOut1, gpuOut2)
		row.GPUDrift = parityGPU
		row.InferGPUDrift = inferDetGPU
		gpuTol := tol * 10
		if gpuTol < 0.05 {
			gpuTol = 0.05
		}
		row.ParityGPUSpec = classifyDrift(parityGPU, gpuTol)
		row.InferGPUSpec = classifyDrift(inferDetGPU, inferDriftTolerance(dtypeLabel)*10)
	}

	if row.Note == "" {
		row.Note = "OK"
	}
	return row
}

func releaseDispatchNet(net *poly.VolumetricNetwork) {
	if net == nil {
		return
	}
	for i := range net.Layers {
		if net.Layers[i].AccelBinding != nil {
			net.Layers[i].AccelBinding.Release()
			net.Layers[i].AccelBinding = nil
		}
	}
}

func captureDispatchForward(net *poly.VolumetricNetwork, in any, iters int) (any, float64) {
	var samples []float64
	var last any
	for i := 0; i < iters; i++ {
		resetNet(net)
		t0 := time.Now()
		last = dispatchForwardOnce(net, in)
		samples = append(samples, float64(time.Since(t0).Microseconds())/1000.0)
	}
	return last, medianFloat(samples)
}

func dispatchForwardOnce(net *poly.VolumetricNetwork, in any) any {
	switch in := in.(type) {
	case *poly.Tensor[int8]:
		out, _, _ := poly.ForwardPolymorphic(net, in)
		return out
	default:
		out, _, _ := poly.ForwardPolymorphic(net, in.(*poly.Tensor[float32]))
		return out
	}
}

func maxAbsDiffTensor(a, b any) float64 {
	if a == nil || b == nil {
		return math.MaxFloat64
	}
	switch ta := a.(type) {
	case *poly.Tensor[int8]:
		tb := b.(*poly.Tensor[int8])
		if len(ta.Data) != len(tb.Data) {
			return math.MaxFloat64
		}
		var m float64
		for i := range ta.Data {
			d := math.Abs(float64(ta.Data[i]) - float64(tb.Data[i]))
			if d > m {
				m = d
			}
		}
		return m
	default:
		taf := a.(*poly.Tensor[float32])
		tbf := b.(*poly.Tensor[float32])
		if len(taf.Data) != len(tbf.Data) {
			return math.MaxFloat64
		}
		var m float64
		for i := range taf.Data {
			d := math.Abs(float64(taf.Data[i] - tbf.Data[i]))
			if d > m {
				m = d
			}
		}
		return m
	}
}

func parityTolerance(dtypeLabel string) float64 {
	switch dtypeLabel {
	case "FP32":
		return 1e-2
	case "FP16":
		return 5e-2
	case "INT16":
		return 0.5
	case "INT8":
		return 2.0
	case "INT4":
		return 4.0
	default:
		return 1e-2
	}
}

func medianFloat(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	return s[len(s)/2]
}

func printDispatchTable(rows []dispatchRow, hasGPU bool) {
	registerDispatchRows(rows)

	fmt.Println()
	fmt.Println("=== DispatchLayer — Loom CPU vs Apple CPU/GPU (seven-style tables) ===")

	bySize := map[string][]dispatchRow{}
	sizeOrder := make([]string, 0)
	for _, r := range rows {
		if _, ok := bySize[r.Size]; !ok {
			sizeOrder = append(sizeOrder, r.Size)
		}
		bySize[r.Size] = append(bySize[r.Size], r)
	}

	for _, sizeName := range sizeOrder {
		group := bySize[sizeName]
		printAccelTimingTable(sizeName, group, hasGPU)
		printAccelDeterminismTable(sizeName, group, hasGPU)
	}
	printDispatchManifest(hasGPU)
}
