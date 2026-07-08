package testing

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/openfluke/loom/poly"
	"github.com/openfluke/webgpu/wgpu"
)

// RunDenseL1Caching runs forward parity timing with CPU SC/MC vs GPU SC/MC.
func RunDenseL1Caching() {
	runDenseForwardBenchmark("L1 Caching (CPU SC/MC + GPU SC/MC)")
}

// RunDenseGPUForward uses the same forward table as L1.
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

func runDenseForwardSuite(spec TestSpec) bool {
	fmt.Printf("\n=== %s Forward — CPU Go / GPU (all numerical types) ===\n", spec.Name)
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

	fmt.Printf("| %-10s | %-4s | %-12s | %-12s | %-12s | %-12s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s |\n",
		"DType", "Tile", "CPU SC", "CPU MC", "GPU SC", "GPU MC", "Dcpu", "Dgpu", "D(G,SC)", "D(G,MC)", "SCspd", "MCspd")
	fmt.Println("|------------|------|--------------|--------------|--------------|--------------|----------|----------|----------|----------|----------|----------|")

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
		l.EnableMultiCoreTiling = false
		t0 := time.Now()
		_, postSC := poly.DispatchLayer(l, input, nil)
		tCPUSC := time.Since(t0)

		l.ResetState()
		l.UseTiling = true
		l.EnableMultiCoreTiling = true
		t0 = time.Now()
		_, postMC := poly.DispatchLayer(l, input, nil)
		tCPUMC := time.Since(t0)

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

		fmt.Printf("| %-10s | %-4d | %-12v | %-12v | %-12v | %-12v | %-8.2e | %-8.2e | %-8.2e | %-8.2e | %-8.2fx | %-8.2fx |\n",
			cfg.name, l.GetCPUTileSize(cfg.dtype), tCPUSC, tCPUMC, tGPUSC, tGPUMC,
			diffCpuSCMC, diffGpuSCMC, diffGSC, diffGMC, scSpd, mcSpd)

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

	return allPass
}

func ratio(slow, fast time.Duration) float64 {
	if fast <= 0 {
		return 0
	}
	return float64(slow) / float64(fast)
}

func jsonMarshalNet(spec poly.PersistenceNetworkSpec) ([]byte, error) {
	return json.Marshal(spec)
}
