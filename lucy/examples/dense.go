package examples

import (
	"bufio"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/openfluke/loom/poly"
)

const (
	benchLayers  = 30
	benchWidth   = 64
	benchSamples = 100
	benchSeed    = 0xC0FFEE
)

// RunDenseForwardComparison runs the dense benchmark (compat alias).
func RunDenseForwardComparison(reader *bufio.Reader) {
	RunDenseBench(reader)
}

// RunDenseBench builds a dense-only stack from JSON, loads one FP32 master, then for each poly numerical type
// deserializes a fresh net and quantizes weights. Results print as one combined four-column table (rows run downward).
func RunDenseBench(_ *bufio.Reader) {
	runDenseBench()
}

func runDenseBench() {
	path, err := denseCachePath()
	if err != nil {
		fmt.Printf("❌ cache path: %v\n", err)
		return
	}
	spec := denseNetSpecJSON()
	netMaster, err := loadNetFromSpecJSON(spec, path)
	if err != nil {
		fmt.Printf("❌ network (master): %v\n", err)
		return
	}
	wire, err := poly.SerializeNetwork(netMaster)
	if err != nil {
		fmt.Printf("❌ serialize master: %v\n", err)
		return
	}

	inputs := denseSampleInputs(benchSamples, benchSeed)

	rows := make([]denseDTypeBench, 0, len(poly.NativeMatrixAllCases))
	for _, tc := range poly.NativeMatrixAllCases {
		rows = append(rows, benchDenseOneDType(wire, tc, inputs))
	}

	fmt.Println()
	fmt.Println("── Dense benchmark │ forward = 1 stamp per full pass │ step = 1 stamp per layer output ──")
	fmt.Println("    Single table: Numerical type │ Metric │ Forward │ Step (one row group per dtype, read downward)")
	printDenseCompareMegaTable(rows)
}

// denseDTypeBench holds one quantized dtype’s timings for the combined benchmark table.
type denseDTypeBench struct {
	Name       string
	Bits       int
	WeightMem  int64
	FwdN       int
	FwdDelay   time.Duration
	FwdAvgGap  time.Duration
	FwdHz      float64
	StepN      int
	StepDelay  time.Duration
	StepAvgGap time.Duration
	StepHz     float64
	Err        string
}

func quantizeDenseBenchNet(net *poly.VolumetricNetwork, dt poly.DType) {
	net.UseTiling = false
	net.EnableMultiCoreTiling = false
	for i := range net.Layers {
		layer := &net.Layers[i]
		layer.DType = dt
		layer.UseTiling = false
		layer.EnableMultiCoreTiling = false
		if layer.WeightStore == nil {
			continue
		}
		layer.WeightStore.Scale = 1.0
		layer.WeightStore.InvalidateVersions()
		if dt != poly.DTypeFloat32 {
			layer.WeightStore.Morph(dt)
		}
	}
	net.SyncToCPU()
}

func denseApproxWeightBytes(net *poly.VolumetricNetwork) int64 {
	var n int64
	for i := range net.Layers {
		l := &net.Layers[i]
		if l.WeightStore == nil {
			continue
		}
		n += int64(l.WeightStore.SizeInBytes(l.DType))
	}
	return n
}

func benchDenseOneDType(wire []byte, tc poly.NativeMatrixCase, inputs []*poly.Tensor[float32]) denseDTypeBench {
	row := denseDTypeBench{Name: tc.Name, Bits: poly.DTypeBits(tc.DType)}
	netFwd, err := poly.DeserializeNetwork(wire)
	if err != nil {
		row.Err = err.Error()
		return row
	}
	quantizeDenseBenchNet(netFwd, tc.DType)
	row.WeightMem = denseApproxWeightBytes(netFwd)

	netStep, err := poly.DeserializeNetwork(wire)
	if err != nil {
		row.Err = err.Error()
		return row
	}
	quantizeDenseBenchNet(netStep, tc.DType)

	var panicVal any
	func() {
		defer func() { panicVal = recover() }()
		fwdTimes := benchDenseForwardLoop(netFwd, inputs)
		s := poly.NewStepState[float32](netStep)
		stepTimes := benchDenseStepLoop(s, netStep, inputs)
		row.FwdN = len(fwdTimes)
		row.StepN = len(stepTimes)
		row.FwdDelay, row.FwdAvgGap = outputDelayAndAvgGap(fwdTimes)
		row.StepDelay, row.StepAvgGap = outputDelayAndAvgGap(stepTimes)
		row.FwdHz = outputsPerSec(fwdTimes)
		row.StepHz = outputsPerSec(stepTimes)
	}()
	if panicVal != nil {
		row.Err = fmt.Sprintf("panic: %v", panicVal)
	}
	return row
}

func humanBytesIEC(n int64) string {
	if n < 1024 {
		return fmt.Sprintf("%d B", n)
	}
	if n < 1024*1024 {
		return fmt.Sprintf("%.1f KiB", float64(n)/1024)
	}
	return fmt.Sprintf("%.2f MiB", float64(n)/(1024*1024))
}

const (
	colType   = 22
	colMetric = 23
	colFwd    = 30
	colStep   = 30
)

func denseCompareTableBorders() (top, mid, bot string) {
	top = "┌" + strings.Repeat("─", colType+2) + "┬" + strings.Repeat("─", colMetric+2) + "┬" +
		strings.Repeat("─", colFwd+2) + "┬" + strings.Repeat("─", colStep+2) + "┐"
	mid = "├" + strings.Repeat("─", colType+2) + "┼" + strings.Repeat("─", colMetric+2) + "┼" +
		strings.Repeat("─", colFwd+2) + "┼" + strings.Repeat("─", colStep+2) + "┤"
	bot = "└" + strings.Repeat("─", colType+2) + "┴" + strings.Repeat("─", colMetric+2) + "┴" +
		strings.Repeat("─", colFwd+2) + "┴" + strings.Repeat("─", colStep+2) + "┘"
	return top, mid, bot
}

func printDenseCompareMegaTable(rows []denseDTypeBench) {
	top, mid, bot := denseCompareTableBorders()
	fmt.Println()
	fmt.Println(top)
	fmt.Printf("│ %-*s │ %-*s │ %-*s │ %-*s │\n", colType, "Numerical type", colMetric, "Metric", colFwd, "Forward", colStep, "Step")
	fmt.Println(mid)

	for i, r := range rows {
		if i > 0 {
			fmt.Println(mid)
		}
		if r.Err != "" {
			fmt.Printf("│ %-*s │ %-*s │ %-*s │ %-*s │\n", colType, r.Name, colMetric, "ERROR",
				colFwd, truncateRunes(r.Err, colFwd), colStep, "")
			continue
		}

		fd := r.FwdDelay.Round(time.Microsecond)
		fa := r.FwdAvgGap.Round(time.Microsecond)
		sd := r.StepDelay.Round(time.Microsecond)
		sa := r.StepAvgGap.Round(time.Microsecond)

		typeLabel := fmt.Sprintf("%s (%d-bit)", r.Name, r.Bits)
		wSize := truncateRunes(humanBytesIEC(r.WeightMem), colType)

		fmt.Printf("│ %-*s │ %-*s │ %-*s │ %-*s │\n", colType, typeLabel, colMetric, "Output events",
			colFwd, fmt.Sprintf("%d (full passes)", r.FwdN), colStep, fmt.Sprintf("%d (layer outputs)", r.StepN))
		fmt.Printf("│ %-*s │ %-*s │ %-*s │ %-*s │\n", colType, wSize, colMetric, "Delay (first output)", colFwd, fd.String(), colStep, sd.String())
		fmt.Printf("│ %-*s │ %-*s │ %-*s │ %-*s │\n", colType, "", colMetric, "Avg gap (between outputs)", colFwd, fa.String(), colStep, sa.String())
		fmt.Printf("│ %-*s │ %-*s │ %-*s │ %-*s │\n", colType, "", colMetric, "~Outputs/sec",
			colFwd, fmt.Sprintf("%.1f", r.FwdHz), colStep, fmt.Sprintf("%.1f", r.StepHz))
	}
	fmt.Println(bot)
}

func truncateRunes(s string, max int) string {
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	if max <= 1 {
		return "…"
	}
	return string(r[:max-1]) + "…"
}


// benchDenseForwardLoop records wall time since loop start after each ForwardPolymorphic completes (one output event).
func benchDenseForwardLoop(net *poly.VolumetricNetwork, inputs []*poly.Tensor[float32]) []time.Duration {
	times := make([]time.Duration, 0, len(inputs))
	loopStart := time.Now()
	for i := range inputs {
		_, _, _ = poly.ForwardPolymorphic(net, inputs[i])
		times = append(times, time.Since(loopStart))
	}
	return times
}

// benchDenseStepLoop records wall time since loop start after every layer output inside each mesh clock
// (same ordering as StepForward’s sequential dispatch — one timestamp per layer write).
func benchDenseStepLoop(s *poly.StepState[float32], net *poly.VolumetricNetwork, inputs []*poly.Tensor[float32]) []time.Duration {
	nLayers := len(net.Layers)
	times := make([]time.Duration, 0, len(inputs)*nLayers)
	loopStart := time.Now()
	for i := range inputs {
		s.SetInput(inputs[i])
		denseMeshClockAppendTimestamps(net, s, loopStart, &times)
	}
	return times
}

// denseMeshClockAppendTimestamps performs one classic mesh clock (same layer order as poly.StepForward
// when UseTiling is false), appending time.Since(origin) after each layer writes an output to NextBuffer.
// Benchmark-only: single-threaded; does not lock StepState (same as calling StepForward from one goroutine).
func denseMeshClockAppendTimestamps(net *poly.VolumetricNetwork, s *poly.StepState[float32], origin time.Time, times *[]time.Duration) {
	if net.UseTiling {
		return
	}
	for idx := range net.Layers {
		l := &net.Layers[idx]
		if l.IsDisabled {
			if idx > 0 {
				s.NextBuffer[idx] = s.LayerData[idx-1]
			} else {
				s.NextBuffer[idx] = s.LayerData[idx]
			}
			*times = append(*times, time.Since(origin))
			continue
		}
		var in *poly.Tensor[float32]
		if l.IsRemoteLink {
			tIdx := net.GetIndex(l.TargetZ, l.TargetY, l.TargetX, l.TargetL)
			in = s.LayerData[tIdx]
		} else if idx > 0 {
			in = s.LayerData[idx-1]
		} else {
			in = s.LayerData[0]
		}
		if in != nil {
			_, post := poly.DispatchLayer(l, in, nil)
			s.NextBuffer[idx] = post
			*times = append(*times, time.Since(origin))
		}
	}
	for i := range s.LayerData {
		s.LayerData[i] = s.NextBuffer[i]
	}
	s.StepCount++
}

// outputDelayAndAvgGap: delay = elapsed to first output; avg = mean interval between consecutive outputs.
func outputDelayAndAvgGap(times []time.Duration) (delay, avgGap time.Duration) {
	if len(times) == 0 {
		return 0, 0
	}
	delay = times[0]
	if len(times) == 1 {
		return delay, delay
	}
	var sumGap time.Duration
	for i := 1; i < len(times); i++ {
		sumGap += times[i] - times[i-1]
	}
	avgGap = sumGap / time.Duration(len(times)-1)
	return delay, avgGap
}

// outputsPerSec uses span from loop start to last output.
func outputsPerSec(times []time.Duration) float64 {
	if len(times) == 0 {
		return 0
	}
	span := times[len(times)-1]
	if span <= 0 {
		return 0
	}
	return float64(len(times)) / span.Seconds()
}

func denseSampleInputs(n int, seed uint64) []*poly.Tensor[float32] {
	rng := rand.New(rand.NewPCG(seed, seed^0xFFFF))
	out := make([]*poly.Tensor[float32], n)
	for i := 0; i < n; i++ {
		t := poly.NewTensor[float32](1, benchWidth)
		for j := range t.Data {
			t.Data[j] = rng.Float32()*2 - 1
		}
		out[i] = t
	}
	return out
}

func denseNetSpecJSON() []byte {
	var b strings.Builder
	b.WriteString(`{"id":"lucy-bench-dense","depth":1,"rows":1,"cols":1,"layers_per_cell":`)
	b.WriteString(fmt.Sprintf("%d", benchLayers))
	b.WriteString(`,"layers":[`)
	for i := 0; i < benchLayers; i++ {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(denseLayerJSON(i))
	}
	b.WriteString(`]}`)
	return []byte(b.String())
}

func denseLayerJSON(i int) string {
	return fmt.Sprintf(
		`{"z":0,"y":0,"x":0,"l":%d,"type":"DENSE","activation":"RELU","dtype":"FLOAT32","input_height":%d,"output_height":%d}`,
		i, benchWidth, benchWidth,
	)
}

func loadNetFromSpecJSON(specJSON []byte, persistencePath string) (*poly.VolumetricNetwork, error) {
	if data, err := os.ReadFile(persistencePath); err == nil {
		return poly.DeserializeNetwork(data)
	} else if !os.IsNotExist(err) {
		return nil, err
	}

	net, err := poly.BuildNetworkFromJSON(specJSON)
	if err != nil {
		return nil, err
	}
	net.UseTiling = false

	if err := os.MkdirAll(filepath.Dir(persistencePath), 0o755); err != nil {
		return nil, err
	}
	out, err := poly.SerializeNetwork(net)
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(persistencePath, out, 0o644); err != nil {
		return nil, err
	}
	return net, nil
}

func denseCachePath() (string, error) {
	name := fmt.Sprintf("bench_dense_%dx%d_weights.json", benchLayers, benchWidth)
	tryDirs := []func() (string, error){
		func() (string, error) {
			d, err := os.UserCacheDir()
			if err != nil {
				return "", err
			}
			return filepath.Join(d, "openfluke-lucy"), nil
		},
		func() (string, error) {
			wd, err := os.Getwd()
			if err != nil {
				return "", err
			}
			return filepath.Join(wd, ".lucy-cache"), nil
		},
	}
	var lastErr error
	for _, dirFn := range tryDirs {
		dir, err := dirFn()
		if err != nil {
			lastErr = err
			continue
		}
		if err := os.MkdirAll(dir, 0o755); err != nil {
			lastErr = err
			continue
		}
		return filepath.Join(dir, name), nil
	}
	if lastErr != nil {
		return "", lastErr
	}
	return "", fmt.Errorf("no writable cache directory for bench weights")
}
