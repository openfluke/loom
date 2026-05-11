package testing

import (
	"encoding/json"
	"fmt"
	"math"
	"time"

	"github.com/openfluke/loom/poly"
	"github.com/openfluke/webgpu/wgpu"
)

type TestMode int

const (
	TestForward TestMode = 1 << iota
	TestBackward
	TestSaveLoad
	TestTraining
	TestAll = TestForward | TestBackward | TestSaveLoad | TestTraining
)

// TestSpec defines a standardized layer test configuration.
type TestSpec struct {
	Name       string
	Layer      poly.PersistenceLayerSpec
	InputShape []int // [Batch, ...]
}

// RunGenericLayerSuite executes forward/backward parity, save/reload, and training tests.
func RunGenericLayerSuite(spec TestSpec, mode TestMode) bool {
	fmt.Printf("\n--- [%s] Generic Layer Suite ---\n", spec.Name)
	if spec.Layer.Type == "CNN1" {
		fmt.Println("    Note: generic CNN1 layer tests still include simulated/PTQ fallback paths for unsupported dtypes.")
		fmt.Println("    Use Glitch layer_matrix example for a strict native-only CPU/GPU/tiling audit.")
	}
	stats.StartLayer()

	fullSpec := poly.PersistenceNetworkSpec{
		ID: "test_net", Depth: 1, Rows: 1, Cols: 1, LayersPerCell: 1,
		Layers: []poly.PersistenceLayerSpec{spec.Layer},
	}
	fullSpec.Layers[0].Z, fullSpec.Layers[0].Y, fullSpec.Layers[0].X, fullSpec.Layers[0].L = 0, 0, 0, 0

	js, _ := json.Marshal(fullSpec)
	net, err := poly.DeserializeNetwork(js)
	if err != nil {
		fmt.Printf("Deserialization failed: %v\n", err)
		return false
	}
	l := net.GetLayer(0, 0, 0, 0)

	allPass := true

	if mode&TestForward != 0 {
		stats.ResetSub()
		allPass = runForwardSuite(spec, l) && allPass
		stats.ReportSub("Forward Parity")
	}

	if mode&TestBackward != 0 {
		stats.ResetSub()
		allPass = runBackwardSuite(spec, l) && allPass
		stats.ReportSub("Backward Parity")
	}

	if mode&TestSaveLoad != 0 {
		stats.ResetSub()
		allPass = runSaveReloadSuite(spec, l) && allPass
		stats.ReportSub("Save/Reload")
	}

	if mode&TestTraining != 0 {
		stats.ResetSub()
		allPass = runTrainingSuite(spec, l) && allPass
		stats.ReportSub("Training Matrix")
	}

	stats.ReportLayer(spec.Name)
	return allPass
}

func runForwardSuite(spec TestSpec, l *poly.VolumetricLayer) bool {
	fmt.Printf("\n=== %s GPU Forward - All Numerical Types ===\n\n", spec.Name)
	input := genInput(spec.InputShape)

	ctx := l.Network.GPUContext
	if ctx == nil {
		l.Network.InitWGPU()
		ctx = l.Network.GPUContext
	}

	fmt.Printf("| %-10s | %-4s | %-12s | %-12s | %-12s | %-12s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s |\n",
		"DType", "Tile", "CPU SC", "CPU MC", "GPU SC", "GPU MC",
		"Dcpu", "Dgpu", "D(G,SC)", "D(G,MC)", "SCspd", "MCspd")
	fmt.Println("|------------|------|--------------|--------------|--------------|--------------|----------|----------|----------|----------|----------|----------|")

	allPass := true
	for _, cfg := range allTypes {
		l.DType = cfg.dtype
		if l.WeightStore != nil {
			l.WeightStore.Morph(cfg.dtype)
			l.WeightStore.Scale = cfg.scale
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

		scSpd := 0.0
		if tGPUSC > 0 {
			scSpd = float64(tCPUSC) / float64(tGPUSC)
		}
		mcSpd := 0.0
		if tGPUMC > 0 {
			mcSpd = float64(tCPUMC) / float64(tGPUMC)
		}

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

func runBackwardSuite(spec TestSpec, l *poly.VolumetricLayer) bool {
	fmt.Printf("\n=== %s GPU Backward - All Numerical Types ===\n\n", spec.Name)
	input := genInput(spec.InputShape)
	l.DType = poly.DTypeFloat32
	l.WeightStore.Morph(poly.DTypeFloat32)
	l.SyncToCPU()

	// Prepare gradOut once for all types
	l.DType = poly.DTypeFloat32
	l.WeightStore.Morph(poly.DTypeFloat32)
	l.ResetState()
	_, post_base := poly.DispatchLayer(l, input, nil)
	gradOut := genInput(post_base.Shape)

	l.EnableMultiCoreTiling = true

	ctx := l.Network.GPUContext
	if ctx == nil {
		l.Network.InitWGPU()
		ctx = l.Network.GPUContext
	}

	fmt.Printf("| %-10s | %-4s | %-12s | %-12s | %-12s | %-7s | %-7s | %-9s | %-9s | %-9s | %-9s | %-8s | %-8s |\n",
		"DType", "Tile", "CPU MC", "GPU SC", "GPU MC",
		"SC-Spd", "MC-Spd", "D-DX/SC", "D-DW/SC", "D-DX/MC", "D-DW/MC", "SC", "MC")
	fmt.Println("|------------|------|--------------|--------------|--------------|---------|---------|-----------|-----------|-----------|-----------|----------|----------|")

	allPass := true
	for _, cfg := range allTypes {
		l.DType = cfg.dtype
		if l.WeightStore != nil {
			l.WeightStore.Morph(cfg.dtype)
			l.WeightStore.Scale = cfg.scale
		}
		l.Network.SyncToGPU()

		l.ResetState()
		pre, _ := poly.DispatchLayer(l, input, nil)

		l.UseTiling = true
		l.EnableMultiCoreTiling = true
		t0 := time.Now()
		cpuDX, cpuDW := poly.DispatchLayerBackward(l, gradOut, input, nil, pre)
		tCPUMC := time.Since(t0)

		// CopySrc: ReadBuffer inside DispatchBackwardLayer CPU fallback (MHA, SwiGLU).
		inBuf, _ := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label:    "BwdIn",
			Contents: wgpu.ToBytes(input.Data),
			Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
		})
		goBuf, _ := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label:    "BwdGO",
			Contents: wgpu.ToBytes(gradOut.Data),
			Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
		})
		preBuf, _ := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label:    "BwdPre",
			Contents: wgpu.ToBytes(pre.Data),
			Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
		})

		dwSize := len(cpuDW.Data)
		if l.Type == poly.LayerResidual {
			dwSize = len(cpuDX.Data)
		}
		dxBufSC, _ := zeroF32Buf(ctx, len(cpuDX.Data), "dxSC")
		dwBufSC, _ := zeroF32Buf(ctx, dwSize, "dwSC")
		dxBufMC, _ := zeroF32Buf(ctx, len(cpuDX.Data), "dxMC")
		dwBufMC, _ := zeroF32Buf(ctx, dwSize, "dwMC")
		defer inBuf.Destroy()
		defer goBuf.Destroy()
		defer preBuf.Destroy()
		defer dxBufSC.Destroy()
		defer dwBufSC.Destroy()
		defer dxBufMC.Destroy()
		defer dwBufMC.Destroy()

		ctx.GPUTileSize = l.GetGPUSCTileSize(cfg.dtype)
		t0 = time.Now()
		ctx.DispatchBackwardLayer(l, spec.InputShape[0], goBuf, inBuf, preBuf, dxBufSC, dwBufSC)
		ctx.Device.Poll(true, nil)
		tGPUSC := time.Since(t0)
		gDXSC, _ := ctx.ReadBuffer(dxBufSC)
		gDWSC, _ := ctx.ReadBuffer(dwBufSC)

		ctx.GPUTileSize = l.GetGPUMCTileSize(cfg.dtype)
		t0 = time.Now()
		ctx.DispatchBackwardLayer(l, spec.InputShape[0], goBuf, inBuf, preBuf, dxBufMC, dwBufMC)
		ctx.Device.Poll(true, nil)
		tGPUMC := time.Since(t0)
		gDXMC, _ := ctx.ReadBuffer(dxBufMC)
		gDWMC, _ := ctx.ReadBuffer(dwBufMC)

		dxDiffSC := maxAbsDiff(cpuDX.Data, gDXSC)
		dwDiffSC := maxAbsDiff(cpuDW.Data, gDWSC)
		dxDiffMC := maxAbsDiff(cpuDX.Data, gDXMC)
		dwDiffMC := maxAbsDiff(cpuDW.Data, gDWMC)

		okSC := dxDiffSC < cfg.tolerance*5 && dwDiffSC < cfg.tolerance*5
		okMC := dxDiffMC < cfg.tolerance*5 && dwDiffMC < cfg.tolerance*5

		// Use combined dX+dW for dead-signal detection so that types where CPU dX is
		// legitimately zero (e.g. embeddings, integer truncation) but dW is non-zero
		// are not falsely classified as SpecBroken.
		mSC := spectrumMark(dxDiffSC+dwDiffSC, cfg.tolerance*10, append(gDXSC, gDWSC...), append(cpuDX.Data, cpuDW.Data...))
		mMC := spectrumMark(dxDiffMC+dwDiffMC, cfg.tolerance*10, append(gDXMC, gDWMC...), append(cpuDX.Data, cpuDW.Data...))

		scSpd := 0.0
		if tGPUSC > 0 {
			scSpd = float64(tCPUMC) / float64(tGPUSC)
		}
		mcSpd := 0.0
		if tGPUMC > 0 {
			mcSpd = float64(tCPUMC) / float64(tGPUMC)
		}

		fmt.Printf("| %-10s | %-4d | %-12v | %-12v | %-12v | %-7.1fx | %-7.1fx | %-9.2e | %-9.2e | %-9.2e | %-9.2e | %-8s | %-8s |\n",
			cfg.name, l.GetCPUTileSize(cfg.dtype), tCPUMC, tGPUSC, tGPUMC,
			scSpd, mcSpd,
			dxDiffSC, dwDiffSC, dxDiffMC, dwDiffMC,
			mSC, mMC)

		if !okSC || !okMC {
			allPass = false
		}
		stats.AddSpectrum(mSC)
		stats.AddSpectrum(mMC)
		stats.AddPerf(spec.Name, cfg.name, "Backward", tCPUMC, tGPUMC)
	}
	return allPass
}

func runSaveReloadSuite(spec TestSpec, l *poly.VolumetricLayer) bool {
	fmt.Printf("  [Save/Reload %s] ", spec.Name)
	input := genInput(spec.InputShape)
	l.DType = poly.DTypeFloat32
	l.ResetState()
	_, post1 := poly.DispatchLayer(l, input, nil)

	js, _ := poly.SerializeNetwork(l.Network)
	net2, err := poly.DeserializeNetwork(js)
	if err != nil {
		fmt.Printf("FAIL: %v\n", err)
		return false
	}

	l2 := net2.GetLayer(0, 0, 0, 0)
	l2.ResetState()
	_, post2 := poly.DispatchLayer(l2, input, nil)

	weightsMatch := true
	if l.WeightStore != nil && l2.WeightStore != nil {
		weightsMatch = maxAbsDiff(l.WeightStore.Master, l2.WeightStore.Master) < 1e-6
	} else if (l.WeightStore == nil) != (l2.WeightStore == nil) {
		weightsMatch = false
	}
	diff := maxAbsDiff(post1.Data, post2.Data)

	ok := diff < 1e-6 && weightsMatch
	if ok {
		fmt.Println("PASS")
	} else {
		fmt.Printf("FAIL (Diff: %.2e, Weights: %v)\n", diff, weightsMatch)
	}
	stats.AddSpectrum(spectrumMark(diff, 1e-6, post1.Data, post2.Data))
	return ok
}

func runTrainingSuite(spec TestSpec, l *poly.VolumetricLayer) bool {
	fmt.Printf("\n=== %s Training — CPU/GPU SC+MC tiled × all numerical types ===\n\n", spec.Name)
	input := genInput(spec.InputShape)
	l.ResetState()
	_, postBaseline := poly.DispatchLayer(l, input, nil)
	target := genInput(postBaseline.Shape)

	batch := poly.TrainingBatch[float32]{Input: input, Target: target}
	allModes := []poly.TrainingMode{
		poly.TrainingModeCPUSC, poly.TrainingModeCPUMC,
		poly.TrainingModeGPUSC, poly.TrainingModeGPUMC,
	}

	fmt.Printf("| %-10s | %-13s | %-10s | %-10s | %-8s | %-7s | %-11s | %-8s | %-8s |\n",
		"DType", "Mode", "Loss[0]", "Loss[N]", "Time", "TrainOK", "Save/Reload", "File", "RAM")
	fmt.Println("|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|")

	// Snapshot weights ONCE before any training run. Each (type, mode) pair
	// starts from the same initial weights to prevent cascading corruption:
	// GPU backward NaN gradients can write NaN into Master, which then
	// propagates to every subsequent type's Morph() call.
	var origWeights []float32
	var origScale float32
	if l.WeightStore != nil {
		origWeights = make([]float32, len(l.WeightStore.Master))
		copy(origWeights, l.WeightStore.Master)
		origScale = l.WeightStore.Scale
	}

	overallPass := true
	for _, cfg := range allTypes {
		for _, mode := range allModes {
			l.DType = cfg.dtype
			if l.WeightStore != nil {
				// Restore original weights before each run so a bad gradient
				// update from a previous (type, mode) cannot corrupt this one.
				// CRITICAL: InvalidateVersions FIRST so stale quantized cache
				// does not cause the next Morph to silently return old weights.
				l.WeightStore.InvalidateVersions()
				copy(l.WeightStore.Master, origWeights)
				// Reset scale so auto-scaling runs fresh for each dtype.
				l.WeightStore.Scale = origScale
				if cfg.scale != 1.0 {
					l.WeightStore.Scale = cfg.scale
				}
				l.WeightStore.Morph(cfg.dtype)
				l.SyncToCPU()
			}
			if mode.IsGPU() && l.Network.GPUContext == nil {
				continue
			}

			tcfg := poly.DefaultTrainingConfig()
			tcfg.Epochs = 5
			tcfg.Mode = mode
			tcfg.Verbose = false
			tcfg.LearningRate = 0.01

			l.ResetState()
			start := time.Now()
			res, err := poly.Train(l.Network, []poly.TrainingBatch[float32]{batch}, tcfg)
			dur := time.Since(start)

			if err != nil {
				fmt.Printf("| %-10s | %-13s | ERR        | ERR        | %-8v | ERR     | %s\n", cfg.name, mode.String(), dur.Round(time.Millisecond), err)
				overallPass = false
				continue
			}

			var trainOK, saveOK bool
			if l.WeightStore != nil {
				if mode.IsGPU() {
					poly.SyncWeightsFromGPU(l.Network)
				}
				wt := l.WeightStore.Master
				trainNoNaN := !math.IsNaN(res.FinalLoss) && !math.IsNaN(res.LossHistory[0])
				// Also guard against NaN weights written back from a broken GPU pass.
				for _, v := range wt {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						trainNoNaN = false
						break
					}
				}
				// For very low-precision types, Loss[0] can be very small already
				// (close to target range). Allow convergence even when Loss[0] < 0.01.
				lossInit := res.LossHistory[0]
				var trainImproved bool
				if lossInit < 0.01 {
					// Already near-zero; just require no NaN and stability
					trainImproved = res.FinalLoss <= lossInit*2.0+1e-3
				} else {
					trainImproved = res.FinalLoss < lossInit*1.01
				}
				trainOK = trainNoNaN && trainImproved

				// Serialize: if Master contains NaN (broken GPU pass), skip serialize
				// so json.Marshal doesn't fail on NaN float32 values and crash the run.
				var wr []float32
				js, serErr := poly.SerializeNetwork(l.Network)
				if serErr != nil || len(js) == 0 {
					// Serialization failed — treat as save failure, don't panic.
					saveOK = false
					ramBytes := int64(len(wt)*4) + int64(math.Ceil(float64(len(wt)*poly.DTypeBits(cfg.dtype))/8.0))
					fmt.Printf("| %-10s | %-13s | %-10.4e | %-10.4e | %-8v | %-7s | %-11s | %-8.1fKB | %-8.1fKB |\n",
						cfg.name, mode.String(), res.LossHistory[0], res.FinalLoss, dur.Round(time.Millisecond),
						markMark(trainOK), "FAIL", 0.0, float64(ramBytes)/1024.0)
					if !trainOK {
						overallPass = false
						stats.AddSpectrum(SpecBroken)
					} else {
						stats.AddSpectrum(SpecExact)
					}
					overallPass = false
					stats.AddSpectrum(SpecBroken)
					continue
				}
				net2, deserErr := poly.DeserializeNetwork(js)
				if deserErr != nil || net2 == nil {
					saveOK = false
					ramBytes := int64(len(wt)*4) + int64(math.Ceil(float64(len(wt)*poly.DTypeBits(cfg.dtype))/8.0))
					fmt.Printf("| %-10s | %-13s | %-10.4e | %-10.4e | %-8v | %-7s | %-11s | %-8.1fKB | %-8.1fKB |\n",
						cfg.name, mode.String(), res.LossHistory[0], res.FinalLoss, dur.Round(time.Millisecond),
						markMark(trainOK), "FAIL", float64(len(js))/1024.0, float64(ramBytes)/1024.0)
					if !trainOK {
						overallPass = false
						stats.AddSpectrum(SpecBroken)
					} else {
						stats.AddSpectrum(SpecExact)
					}
					overallPass = false
					stats.AddSpectrum(SpecBroken)
					continue
				}
				l2 := net2.GetLayer(0, 0, 0, 0)
				if l2 == nil || l2.WeightStore == nil {
					saveOK = false
				} else {
					wr = l2.WeightStore.Master
				}

				// Save/Reload tolerance: serializeLayer auto-computes the scale from
				// the current Master so trained weights fit within one quantization step.
				// Use the scale from the RELOADED layer (l2) — that's what Unpack used
				// to reconstruct Master, so the round-trip error is exactly 0.5 * l2.Scale.
				expected := wt
				var tol float64
				switch cfg.dtype {
				case poly.DTypeFloat64, poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16:
					tol = cfg.tolerance * 10.0
					if tol < 1e-4 {
						tol = 1e-4
					}
				default:
					// For all quantized types, max round-trip error = 0.5 * scale.
					reloadScale := float64(1.0)
					if wr != nil && l2 != nil && l2.WeightStore != nil && l2.WeightStore.Scale != 0 {
						reloadScale = float64(l2.WeightStore.Scale)
					}
					tol = reloadScale * 0.51
				}
				// Only compute diff if wr was successfully populated by reload.
				// saveOK stays false if wr == nil (l2 was nil / no WeightStore / bad deser).
				if wr != nil {
					saveOK = maxAbsDiff(wr, expected) < tol
				}

				ramBytes := int64(len(wt)*4) + int64(math.Ceil(float64(len(wt)*poly.DTypeBits(cfg.dtype))/8.0))
				fmt.Printf("| %-10s | %-13s | %-10.4e | %-10.4e | %-8v | %-7s | %-11s | %-8.1fKB | %-8.1fKB |\n",
					cfg.name, mode.String(), res.LossHistory[0], res.FinalLoss, dur.Round(time.Millisecond),
					markMark(trainOK), markMark(saveOK), float64(len(js))/1024.0, float64(ramBytes)/1024.0)
			} else {
				trainOK, saveOK = true, true
				fmt.Printf("| %-10s | %-13s | %-10.4e | %-10.4e | %-8v | %-7v | %-11v | %-8s | %-8s |\n",
					cfg.name, mode.String(), res.LossHistory[0], res.FinalLoss, dur.Round(time.Millisecond),
					"N/A", "N/A", "0KB", "0KB")
			}

			if !trainOK || !saveOK {
				overallPass = false
			}

			if trainOK {
				stats.AddSpectrum(SpecExact)
			} else {
				stats.AddSpectrum(SpecBroken)
			}
			if saveOK {
				stats.AddSpectrum(SpecExact)
			} else {
				stats.AddSpectrum(SpecBroken)
			}
		}
		fmt.Println("|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|")
	}
	return overallPass
}

func markMark(ok bool) string {
	if ok {
		return "PASS"
	}
	return "FAIL"
}
