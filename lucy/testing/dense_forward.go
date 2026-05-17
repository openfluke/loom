package testing

import (
	"encoding/json"
	"fmt"
	"sort"
	"time"

	"github.com/openfluke/loom/poly"
	"github.com/openfluke/loom/poly/asm"
	"github.com/openfluke/webgpu/wgpu"
)

// RunDenseL1Caching runs forward parity timing with ASM SC / ASM MC timers vs Go CPU and GPU.
func RunDenseL1Caching() {
	runDenseForwardBenchmark("L1 Caching (CPU/ASM SC/MC + GPU SC/MC)")
}

// RunDenseGPUForward uses the same forward table as L1 (includes ASM timers).
func RunDenseGPUForward() {
	runDenseForwardBenchmark("GPU Forward Parity")
}

func runDenseForwardBenchmark(sectionLabel string) {
	fmt.Printf("\n--- [%s] Generic Layer Suite ---\n", denseSpec.Name)
	if sectionLabel != "" {
		fmt.Printf("    Section: %s\n", sectionLabel)
	}
	stats.StartLayer()
	stats.ResetSub()
	runDenseForwardSuite(denseSpec)
	stats.ReportSub("Forward Parity")
	stats.ReportLayer(denseSpec.Name)
}

type denseAsmRow struct {
	dtype                            string
	tGoSC, tGoMC, tAsmSC, tAsmMC     time.Duration
	tGpuSC, tGpuMC                   time.Duration
}

func runDenseForwardSuite(spec TestSpec) bool {
	fmt.Printf("\n=== %s Forward — CPU Go / CPU ASM / GPU (all numerical types) ===\n", spec.Name)
	if asm.Enabled() {
		fmt.Println("    Timers: CPU SC/MC = Go tiled forward; ASM SC/MC = Plan 9 .s kernels (UseAsmForward).")
		fmt.Println("    Go/Asm↑ = CPU÷ASM (>1 means assembly is faster). GPU/Asm↑ = GPU÷ASM.")
	} else {
		fmt.Println("    ASM timers unavailable on this GOARCH — rebuild on amd64/arm64 for assembly columns.")
	}
	fmt.Println()

	input := genInput(spec.InputShape)

	fullSpec := poly.PersistenceNetworkSpec{
		ID: "test_net", Depth: 1, Rows: 1, Cols: 1, LayersPerCell: 1,
		Layers: []poly.PersistenceLayerSpec{spec.Layer},
	}
	fullSpec.Layers[0].Z, fullSpec.Layers[0].Y, fullSpec.Layers[0].X, fullSpec.Layers[0].L = 0, 0, 0, 0
	js, _ := jsonMarshalNet(fullSpec)
	net, err := poly.DeserializeNetwork(js)
	if err != nil {
		fmt.Printf("Deserialization failed: %v\n", err)
		return false
	}
	l := net.GetLayer(0, 0, 0, 0)

	ctx := l.Network.GPUContext
	if ctx == nil {
		if err := l.Network.InitWGPU(); err != nil {
			fmt.Printf("GPU init failed: %v\n", err)
			return false
		}
		ctx = l.Network.GPUContext
	}

	if asm.Enabled() {
		fmt.Printf("| %-10s | %-4s | %-11s | %-11s | %-11s | %-11s | %-11s | %-11s | %-7s | %-7s | %-7s | %-7s | %-8s | %-8s | %-8s | %-8s | %-7s | %-7s |\n",
			"DType", "Tile",
			"Go SC", "Go MC", "ASM SC", "ASM MC", "GPU SC", "GPU MC",
			"Go/Asm↑SC", "Go/Asm↑MC", "GPU/Asm↑SC", "GPU/Asm↑MC",
			"Dcpu", "Dgpu", "D(G,SC)", "D(G,MC)", "SCspd", "MCspd")
		fmt.Println("|------------|------|-------------|-------------|-------------|-------------|-------------|-------------|---------|---------|---------|---------|----------|----------|----------|----------|---------|---------|")
	} else {
		fmt.Printf("| %-10s | %-4s | %-12s | %-12s | %-12s | %-12s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s |\n",
			"DType", "Tile", "CPU SC", "CPU MC", "GPU SC", "GPU MC", "Dcpu", "Dgpu", "D(G,SC)", "D(G,MC)", "SCspd", "MCspd")
		fmt.Println("|------------|------|--------------|--------------|--------------|--------------|----------|----------|----------|----------|----------|----------|")
	}

	var asmRows []denseAsmRow
	allPass := true

	for _, cfg := range allTypes {
		l.DType = cfg.dtype
		if l.WeightStore != nil {
			l.WeightStore.InvalidateVersions()
			l.WeightStore.Scale = cfg.scale
			l.WeightStore.Morph(cfg.dtype)
			l.SyncToCPU()
		}

		l.ResetState()
		l.UseTiling = true
		l.UseAsmForward = false
		l.EnableMultiCoreTiling = false
		t0 := time.Now()
		_, postSC := poly.DispatchLayer(l, input, nil)
		tCPUSC := time.Since(t0)

		l.ResetState()
		l.UseTiling = true
		l.UseAsmForward = false
		l.EnableMultiCoreTiling = true
		t0 = time.Now()
		_, postMC := poly.DispatchLayer(l, input, nil)
		tCPUMC := time.Since(t0)

		var tASMSC, tASMMC time.Duration
		var postASMSC, postASMMC []float32
		if asm.Enabled() {
			l.ResetState()
			l.UseTiling = true
			l.UseAsmForward = true
			l.EnableMultiCoreTiling = false
			t0 = time.Now()
			_, p := poly.DispatchLayer(l, input, nil)
			tASMSC = time.Since(t0)
			postASMSC = p.Data

			l.ResetState()
			l.UseTiling = true
			l.UseAsmForward = true
			l.EnableMultiCoreTiling = true
			t0 = time.Now()
			_, p = poly.DispatchLayer(l, input, nil)
			tASMMC = time.Since(t0)
			postASMMC = p.Data
			l.UseAsmForward = false
		}

		l.Network.SyncToGPU()
		inBuf, _ := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label:    "FwdIn",
			Contents: wgpu.ToBytes(input.Data),
			Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
		})
		outSize := len(postSC.Data)
		outBufSC, _ := zeroF32Buf(ctx, outSize, "FwdOutSC")
		outBufMC, _ := zeroF32Buf(ctx, outSize, "FwdOutMC")
		defer inBuf.Destroy()
		defer outBufSC.Destroy()
		defer outBufMC.Destroy()

		l.ResetState()
		l.UseAsmForward = false
		ctx.GPUTileSize = l.GetGPUSCTileSize(cfg.dtype)
		t0 = time.Now()
		ctx.DispatchForwardLayer(l, spec.InputShape[0], inBuf, outBufSC)
		ctx.Device.Poll(true, nil)
		tGPUSC := time.Since(t0)
		gpuSCData, _ := ctx.ReadBuffer(outBufSC)

		l.ResetState()
		ctx.GPUTileSize = l.GetGPUMCTileSize(cfg.dtype)
		t0 = time.Now()
		ctx.DispatchForwardLayer(l, spec.InputShape[0], inBuf, outBufMC)
		ctx.Device.Poll(true, nil)
		tGPUMC := time.Since(t0)
		gpuMCData, _ := ctx.ReadBuffer(outBufMC)

		diffCpuSCMC := maxAbsDiff(postSC.Data, postMC.Data)
		diffGpuSCMC := maxAbsDiff(gpuSCData, gpuMCData)
		diffGSC := maxAbsDiff(postSC.Data, gpuSCData)
		diffGMC := maxAbsDiff(postMC.Data, gpuMCData)

		scSpd := ratio(tCPUSC, tGPUSC)
		mcSpd := ratio(tCPUMC, tGPUMC)

		goAsmSC := ratio(tCPUSC, tASMSC)
		goAsmMC := ratio(tCPUMC, tASMMC)
		gpuAsmSC := ratio(tGPUSC, tASMSC)
		gpuAsmMC := ratio(tGPUMC, tASMMC)

		if asm.Enabled() {
			fmt.Printf("| %-10s | %-4d | %-11v | %-11v | %-11v | %-11v | %-11v | %-11v | %-6.2fx | %-6.2fx | %-6.2fx | %-6.2fx | %-8.2e | %-8.2e | %-8.2e | %-8.2e | %-6.2fx | %-6.2fx |\n",
				cfg.name, l.GetCPUTileSize(cfg.dtype),
				tCPUSC, tCPUMC, tASMSC, tASMMC, tGPUSC, tGPUMC,
				goAsmSC, goAsmMC, gpuAsmSC, gpuAsmMC,
				diffCpuSCMC, diffGpuSCMC, diffGSC, diffGMC, scSpd, mcSpd)
			asmRows = append(asmRows, denseAsmRow{
				dtype: cfg.name, tGoSC: tCPUSC, tGoMC: tCPUMC,
				tAsmSC: tASMSC, tAsmMC: tASMMC, tGpuSC: tGPUSC, tGpuMC: tGPUMC,
			})
			if len(postASMSC) > 0 {
				stats.AddSpectrum(spectrumMark(maxAbsDiff(postSC.Data, postASMSC), cfg.tolerance, postASMSC, postSC.Data))
			}
			if len(postASMMC) > 0 {
				stats.AddSpectrum(spectrumMark(maxAbsDiff(postMC.Data, postASMMC), cfg.tolerance, postASMMC, postMC.Data))
			}
			stats.AddPerf(spec.Name, cfg.name, "Forward Go-SC", tCPUSC, tASMSC)
			stats.AddPerf(spec.Name, cfg.name, "Forward Go-MC", tCPUMC, tASMMC)
			if tGPUSC > 0 && tASMSC > 0 {
				stats.AddPerf(spec.Name, cfg.name, "Forward GPU-SC vs ASM", tGPUSC, tASMSC)
			}
		} else {
			fmt.Printf("| %-10s | %-4d | %-12v | %-12v | %-12v | %-12v | %-8.2e | %-8.2e | %-8.2e | %-8.2e | %-8.2fx | %-8.2fx |\n",
				cfg.name, l.GetCPUTileSize(cfg.dtype), tCPUSC, tCPUMC, tGPUSC, tGPUMC,
				diffCpuSCMC, diffGpuSCMC, diffGSC, diffGMC, scSpd, mcSpd)
		}

		if diffCpuSCMC > 1e-10 || diffGpuSCMC > 1e-10 || diffGSC > 1e-10 || diffGMC > 1e-10 ||
			diffGSC >= cfg.tolerance || diffGMC >= cfg.tolerance {
			allPass = false
		}
		stats.AddSpectrum(spectrumMark(diffCpuSCMC, 1e-10, postMC.Data, postSC.Data))
		stats.AddSpectrum(spectrumMark(diffGpuSCMC, 1e-10, gpuMCData, gpuSCData))
		stats.AddSpectrum(spectrumMark(diffGSC, cfg.tolerance, gpuSCData, postSC.Data))
		stats.AddSpectrum(spectrumMark(diffGMC, cfg.tolerance, gpuMCData, postMC.Data))
		stats.AddPerf(spec.Name, cfg.name, "Forward", tCPUMC, tGPUMC)
	}

	if asm.Enabled() && len(asmRows) > 0 {
		printDenseAsmSummary(asmRows)
	}
	return allPass
}

func ratio(slow, fast time.Duration) float64 {
	if fast <= 0 {
		return 0
	}
	return float64(slow) / float64(fast)
}

func printDenseAsmSummary(rows []denseAsmRow) {
	type ranked struct {
		name  string
		sc, mc float64
	}
	var bySC, byMC []ranked
	for _, r := range rows {
		bySC = append(bySC, ranked{r.dtype, ratio(r.tGoSC, r.tAsmSC), ratio(r.tGoMC, r.tAsmMC)})
		byMC = append(byMC, ranked{r.dtype, ratio(r.tGoSC, r.tAsmSC), ratio(r.tGoMC, r.tAsmMC)})
	}
	sort.Slice(bySC, func(i, j int) bool { return bySC[i].sc > bySC[j].sc })
	sort.Slice(byMC, func(i, j int) bool { return byMC[i].mc > byMC[j].mc })

	fmt.Printf("\n>> [ASM speedup summary] Go÷ASM and GPU÷ASM (values >1.0 = assembly wins on wall time)\n")
	fmt.Printf("| %-10s | %-11s | %-11s | %-11s | %-11s | %-11s | %-11s |\n",
		"DType", "Go SC", "ASM SC", "Go/Asm↑SC", "Go MC", "ASM MC", "Go/Asm↑MC")
	fmt.Println("|------------|-------------|-------------|-------------|-------------|-------------|-------------|")
	for _, r := range rows {
		fmt.Printf("| %-10s | %-11v | %-11v | %-10.2fx | %-11v | %-11v | %-10.2fx |\n",
			r.dtype, r.tGoSC.Round(time.Microsecond), r.tAsmSC.Round(time.Microsecond), ratio(r.tGoSC, r.tAsmSC),
			r.tGoMC.Round(time.Microsecond), r.tAsmMC.Round(time.Microsecond), ratio(r.tGoMC, r.tAsmMC))
	}

	if len(bySC) > 0 {
		best := bySC[0]
		fmt.Printf("\n    Best Go/Asm↑ single-core: %s at %.2fx  |  Best multi-core: %s at %.2fx\n",
			best.name, best.sc, byMC[0].name, byMC[0].mc)
	}
}

func jsonMarshalNet(spec poly.PersistenceNetworkSpec) ([]byte, error) {
	return json.Marshal(spec)
}
