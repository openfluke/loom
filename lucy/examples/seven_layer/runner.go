package sevenlayer

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/openfluke/loom/poly"
)

// LayerSuite configures one seven-layer example (JSON build + tensors).
type LayerSuite struct {
	Name          string
	Grid          GridSpec
	PrimaryType   poly.LayerType
	BuildJSON     func(jsonDType string) []byte
	MakeInput     func() *poly.Tensor[float32]
	MakeTarget    func(net *poly.VolumetricNetwork, input *poly.Tensor[float32]) *poly.Tensor[float32]
	Banner        string
	CheckpointTag string
}

type savePhase string

const (
	phaseBefore savePhase = "before"
	phaseAfter  savePhase = "after"
)

type saveResult struct {
	forwardDiff  float64
	weightDiff   float64
	lossDelta    float64
	trainedLoss  float64
	reloadedLoss float64
	bucket       spectrum
	nativeOK     bool
	pass         bool
	err          string
}

func forwardLoss(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32]) float64 {
	out, _, _ := poly.ForwardPolymorphic(net, input)
	return poly.CalculateLoss(out, target, "mse")
}

func checkSaveReload(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32], tc dtypeCase, refLoss float64, phase savePhase, primary poly.LayerType) saveResult {
	return checkSaveReloadFormat(net, input, target, tc, refLoss, phase, primary, formatJSON)
}

func checkEntitySaveReload(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32], tc dtypeCase, refLoss float64, phase savePhase, primary poly.LayerType) saveResult {
	return checkSaveReloadFormat(net, input, target, tc, refLoss, phase, primary, formatEntity)
}

type checkpointFormat int

const (
	formatJSON checkpointFormat = iota
	formatEntity
)

func checkSaveReloadFormat(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32], tc dtypeCase, refLoss float64, phase savePhase, primary poly.LayerType, fmt checkpointFormat) saveResult {
	r := saveResult{}
	finalizeTrainingNet(net, tc)
	setCPUMode(net, true)

	out0, _, _ := poly.ForwardPolymorphic(net, input)
	baseline := append([]float32(nil), out0.Data...)

	var wire []byte
	var err error
	switch fmt {
	case formatJSON:
		wire, err = poly.SerializeNetwork(net)
	case formatEntity:
		wire, err = poly.SerializeEntity(net)
	}
	if err != nil {
		r.err = err.Error()
		r.bucket = specFatal
		return r
	}
	var reloaded *poly.VolumetricNetwork
	switch fmt {
	case formatJSON:
		reloaded, err = poly.DeserializeNetwork(wire)
	case formatEntity:
		reloaded, err = poly.DeserializeEntity(wire)
	}
	if err != nil {
		r.err = err.Error()
		r.bucket = specFatal
		return r
	}
	wireLayerTree(reloaded)
	setCPUMode(reloaded, true)

	out1, _, _ := poly.ForwardPolymorphic(reloaded, input)
	r.forwardDiff = maxAbsDiff(baseline, out1.Data)
	r.bucket = spectrumMark(r.forwardDiff, tc.tolerance, out1.Data, baseline)
	r.weightDiff = nativeWeightDiff(net, reloaded, tc.dtype)
	r.trainedLoss = poly.CalculateLoss(out0, target, "mse")
	outR, _, _ := poly.ForwardPolymorphic(reloaded, input)
	r.reloadedLoss = poly.CalculateLoss(outR, target, "mse")
	if phase == phaseAfter {
		r.lossDelta = math.Abs(r.reloadedLoss - r.trainedLoss)
	} else {
		r.lossDelta = math.Abs(r.reloadedLoss - refLoss)
	}

	switch fmt {
	case formatJSON:
		r.nativeOK = nativePersistenceOK(net, reloaded, wire, tc)
	case formatEntity:
		r.nativeOK = nativeEntityPersistenceOK(net, reloaded, wire, tc)
	}
	maxBucket := saveReloadMaxBucket(phase, tc.dtype, primary)
	fwdTol := saveReloadFwdTol(phase, tc, primary)
	wTol := tc.tolerance
	if poly.IsDenseNativeTrainDType(tc.dtype) {
		wTol = tc.tolerance * 100
	}
	r.pass = r.forwardDiff <= fwdTol && r.weightDiff <= wTol &&
		r.bucket <= maxBucket && r.nativeOK && r.err == "" &&
		!math.IsNaN(r.forwardDiff) && lossFinite(r.trainedLoss) && lossFinite(r.reloadedLoss)
	return r
}

func nativeWeightDiff(a, b *poly.VolumetricNetwork, dt poly.DType) float64 {
	var maxD float64
	for i := range a.Layers {
		d := nativeWeightDiffLayer(&a.Layers[i], &b.Layers[i], dt)
		if d > maxD {
			maxD = d
		}
	}
	return maxD
}

func nativeWeightDiffLayer(a, b *poly.VolumetricLayer, dt poly.DType) float64 {
	if a == nil || b == nil || a.WeightStore == nil || b.WeightStore == nil {
		return 0
	}
	if poly.IsDenseNativeTrainDType(dt) {
		b64a, scaleA, oka := poly.LayerNativePersistenceSnapshot(a.WeightStore, dt)
		b64b, scaleB, okb := poly.LayerNativePersistenceSnapshot(b.WeightStore, dt)
		if !oka || !okb || b64a != b64b || scaleA != scaleB {
			return 1
		}
		return 0
	}
	return maxLayerWeightDiff(a, b)
}

func nativePersistenceOK(net, reloaded *poly.VolumetricNetwork, wire []byte, tc dtypeCase) bool {
	for i := range net.Layers {
		if !nativePersistenceOKLayer(&net.Layers[i], &reloaded.Layers[i], wire, i, tc.dtype) {
			return false
		}
	}
	return true
}

func nativePersistenceOKLayer(src, dst *poly.VolumetricLayer, wire []byte, layerIndex int, dt poly.DType) bool {
	if src == nil || src.WeightStore == nil {
		return true
	}
	b64, scale, native, fileErr := poly.LayerPersistenceFromJSON(wire, layerIndex)
	if dst == nil || dst.WeightStore == nil {
		return false
	}
	if fileErr != nil || !native || b64 == "" {
		return false
	}
	decoded, decErr := poly.DecodeNativeWeights(b64, dst.DType)
	loaded := dst.WeightStore.Versions[dt]
	if loaded == nil {
		loaded = dst.WeightStore.GetNative(dt)
	}
	return decErr == nil && loaded != nil && dst.WeightStore.Scale == scale &&
		poly.NativeWeightsEncoded(decoded, loaded, dt)
}

func nativeEntityPersistenceOK(net, reloaded *poly.VolumetricNetwork, wire []byte, tc dtypeCase) bool {
	for i := range net.Layers {
		if !nativeEntityPersistenceOKLayer(&net.Layers[i], &reloaded.Layers[i], wire, i, tc.dtype) {
			return false
		}
	}
	return true
}

func nativeEntityPersistenceOKLayer(src, dst *poly.VolumetricLayer, wire []byte, layerIndex int, dt poly.DType) bool {
	if src == nil || src.WeightStore == nil {
		return true
	}
	raw, scale, native, fileErr := poly.LayerPersistenceFromEntity(wire, layerIndex)
	if dst == nil || dst.WeightStore == nil {
		return false
	}
	if fileErr != nil || !native || len(raw) == 0 {
		return false
	}
	decoded, decErr := poly.DecodeNativeWeightsRaw(raw, dst.DType)
	loaded := dst.WeightStore.Versions[dt]
	if loaded == nil {
		loaded = dst.WeightStore.GetNative(dt)
	}
	return decErr == nil && loaded != nil && dst.WeightStore.Scale == scale &&
		poly.NativeWeightsEncoded(decoded, loaded, dt)
}

func trainCPU(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32], mode poly.TrainingMode, tc dtypeCase, primary poly.LayerType, epochs int) (*poly.TrainingResult, time.Duration, error) {
	configureTrainingNet(net, tc, primary)
	prepareTrainingNet(net, tc.dtype)
	cfg := poly.DefaultTrainingConfig()
	cfg.Epochs = epochs
	cfg.LearningRate = trainingLearningRate(tc.dtype)
	cfg.GradientClip = 1.0
	cfg.Mode = mode
	cfg.Verbose = false
	cfg.LossType = "mse"
	t0 := time.Now()
	res, err := poly.Train(net, []poly.TrainingBatch[float32]{{Input: input, Target: target}}, cfg)
	return res, time.Since(t0), err
}

func saveCheckpoint(net *poly.VolumetricNetwork, tag, dtypeName string) string {
	wire, err := poly.SerializeNetwork(net)
	if err != nil {
		return ""
	}
	_ = os.MkdirAll(OutputDir, 0o755)
	path := filepath.Join(OutputDir, tag+"_"+dtypeName+".json")
	_ = os.WriteFile(path, wire, 0o644)
	return path
}

func saveEntityCheckpoint(net *poly.VolumetricNetwork, tag, dtypeName string) string {
	wire, err := poly.SerializeEntity(net)
	if err != nil {
		return ""
	}
	_ = os.MkdirAll(OutputDir, 0o755)
	path := filepath.Join(OutputDir, tag+"_"+dtypeName+".entity")
	_ = os.WriteFile(path, wire, 0o644)
	return path
}

// RunLayerSuite executes the full [7] matrix for one layer type on one grid.
func RunLayerSuite(s LayerSuite) bool {
	epochs := trainEpochsForGrid(s.Grid)
	activeBenchIters = benchItersForGrid(s.Grid)
	suiteLabel := fmt.Sprintf("%s %s", s.Name, s.Grid)

	fmt.Println()
	fmt.Println("══════════════════════════════════════════════════════════════════════")
	fmt.Printf("  Loom seven-layer %s — JSON + .entity · CPU SC/MC/SIMD · train · save/reload\n", suiteLabel)
	fmt.Println(s.Banner)
	fmt.Println("══════════════════════════════════════════════════════════════════════")
	fmt.Printf("  %d dtypes × %d epochs · grid %s (%d-layer stack)\n",
		len(allDTypes), epochs, s.Grid, s.Grid.StackLayers())
	fmt.Println()

	var rows []DTypeRow
	passed, failed := 0, 0

	for _, tc := range allDTypes {
		fmt.Printf("  · %-10s ", tc.name)
		row := DTypeRow{DType: tc.name}

		net, err := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
		if err != nil {
			row.Err = "BUILD"
			rows = append(rows, row)
			failed++
			fmt.Println("BUILD ERR")
			continue
		}
		applyDType(net, tc)
		input := s.MakeInput()
		target := s.MakeTarget(net, input)

		mem0 := readMemSnapshot()
		row.MemHeap = formatBytes(mem0.HeapAlloc)
		row.MemSys = formatBytes(mem0.Sys)

		// Forward determinism: CPU Go SC vs MC.
		fwdSC := captureForward(net, input, false)
		fwdMC := captureForward(net, input, true)
		row.FwdSCDur = formatDur(fwdSC.dur)
		row.FwdMCDur = formatDur(fwdMC.dur)
		row.FwdSCMC = maxAbsDiff(fwdSC.out, fwdMC.out)

		if poly.Plan9SimdForwardForLayer(s.PrimaryType) {
			fwdSimd := captureForwardSimd(net, input, true)
			row.FwdSimdDur = formatDur(fwdSimd.dur)
			// Compare vs SC tiled: SIMD uses wide serial tiles; MC tiled uses smaller
			// tiles + goroutines and can differ in float reduction order.
			row.FwdTiledSimd = maxAbsDiff(fwdSC.out, fwdSimd.out)
			row.FwdSimdPct = formatSimdSpeedup(fwdMC.dur, fwdSimd.dur)
		}

		// Backward: SC vs MC and SIMD vs tiled SC (when layer supports Plan 9 SIMD).
		bwdSC := captureBackward(net, input, target, false)
		bwdMC := captureBackward(net, input, target, true)
		row.BwdSCDur = formatDur(bwdSC.dur)
		row.BwdMCDur = formatDur(bwdMC.dur)
		row.BwdSCMC = maxAbsDiff(append(bwdSC.dx, bwdSC.dw...), append(bwdMC.dx, bwdMC.dw...))

		if poly.Plan9SimdForwardForLayer(s.PrimaryType) {
			bwdSimd := captureBackwardSimd(net, input, target, true)
			row.BwdSimdDur = formatDur(bwdSimd.dur)
			row.BwdTiledSimd = maxAbsDiff(append(bwdSC.dx, bwdSC.dw...), append(bwdSimd.dx, bwdSimd.dw...))
			row.BwdSimdPct = formatSimdSpeedup(bwdMC.dur, bwdSimd.dur)
		}

		// Infer-native weight accounting only (after parity benches; not on forward hot path).
		configureInferenceNet(net)
		row.WeightBytes = formatBytes(networkWeightBytes(net)) + " (infer)"

		detTol := tc.tolerance
		if detTol < 1e-10 {
			detTol = 1e-10
		}
		if s.PrimaryType == poly.LayerMultiHeadAttention && detTol < 1e-4 {
			detTol = 1e-4
		}
		row.DetOK = row.FwdSCMC <= detTol && row.BwdSCMC <= detTol*10
		row.FwdSCMCOK = row.FwdSCMC <= detTol
		row.BwdSCMCOK = row.BwdSCMC <= detTol*10
		if poly.Plan9SimdForwardForLayer(s.PrimaryType) {
			row.SimdOK = row.FwdTiledSimd <= simdParityTol(s.PrimaryType, tc)
			bwdSimdTol := simdParityTol(s.PrimaryType, tc) * 10
			row.BwdSimdOK = row.BwdTiledSimd <= bwdSimdTol
			row.DetOK = row.DetOK && row.SimdOK && row.BwdSimdOK
		}

		lossBefore := forwardLoss(net, input, target)
		requiresLearn := layerRequiresLearn(s.PrimaryType)
		if !lossFiniteOK(lossBefore, lossBefore, requiresLearn) {
			row.Err = "LOSS"
			rows = append(rows, row)
			failed++
			fmt.Printf("FAIL  forward loss %.4e (non-finite or degenerate)\n", lossBefore)
			continue
		}
		before := checkSaveReload(net, input, target, tc, lossBefore, phaseBefore, s.PrimaryType)
		row.BeforeBucket = before.bucket.String()
		row.BeforeOK = before.pass
		row.NativeOK = before.nativeOK
		entityBefore := checkEntitySaveReload(net, input, target, tc, lossBefore, phaseBefore, s.PrimaryType)
		row.EntityBeforeOK = entityBefore.pass
		if entityBefore.nativeOK {
			row.EntityNativeOK = true
		}

		// Train CPU SC then MC (fresh weight copy via rebuild for fairness)
		netSC, _ := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
		applyDType(netSC, tc)
		configureTrainingNet(netSC, tc, s.PrimaryType)
		netSC.ReleaseFP32MasterWhenIdle = true
		resSC, durSC, err := trainCPU(netSC, input, target, poly.TrainingModeCPUSC, tc, s.PrimaryType, epochs)
		if err != nil {
			row.Err = "TRAIN-SC"
			rows = append(rows, row)
			failed++
			fmt.Println("TRAIN SC ERR")
			continue
		}
		lossSC := resSC.FinalLoss
		if len(resSC.LossHistory) > 0 {
			lossSC = resSC.LossHistory[len(resSC.LossHistory)-1]
		}
		row.TrainSCDur = formatDur(durSC)
		row.TrainSCSps = samplesPerSec(durSC, epochs)

		netMC, _ := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
		applyDType(netMC, tc)
		configureTrainingNet(netMC, tc, s.PrimaryType)
		netMC.ReleaseFP32MasterWhenIdle = true
		resMC, durMC, err := trainCPU(netMC, input, target, poly.TrainingModeCPUMC, tc, s.PrimaryType, epochs)
		if err != nil {
			row.Err = "TRAIN-MC"
			rows = append(rows, row)
			failed++
			fmt.Println("TRAIN MC ERR")
			continue
		}
		row.TrainMCDur = formatDur(durMC)
		row.TrainMCSps = samplesPerSec(durMC, epochs)

		if poly.Plan9SimdForwardForLayer(s.PrimaryType) {
			netSimd, _ := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
			applyDType(netSimd, tc)
			configureTrainingNet(netSimd, tc, s.PrimaryType)
			netSimd.ReleaseFP32MasterWhenIdle = true
			resSimd, durSimd, err := trainCPU(netSimd, input, target, poly.TrainingModeCPUSimd, tc, s.PrimaryType, epochs)
			if err != nil {
				row.Err = "TRAIN-SIMD"
				rows = append(rows, row)
				failed++
				fmt.Println("TRAIN SIMD ERR")
				continue
			}
			row.TrainSimdDur = formatDur(durSimd)
			row.TrainSimdSps = samplesPerSec(durSimd, epochs)
			row.LossSimd = resSimd.FinalLoss
			if len(resSimd.LossHistory) > 0 {
				row.LossSimd = resSimd.LossHistory[len(resSimd.LossHistory)-1]
			}
			lossSimdInit := resSimd.LossHistory[0]
			row.TrainSimdOK = lossFiniteOK(lossSimdInit, row.LossSimd, requiresLearn) &&
				(!requiresLearn || trainingOK(lossSimdInit, row.LossSimd, tc.dtype))
		}

		lossInit := resMC.LossHistory[0]
		lossFinal := resMC.FinalLoss
		if len(resMC.LossHistory) > 0 {
			lossFinal = resMC.LossHistory[len(resMC.LossHistory)-1]
		}
		row.LossInit = lossInit
		row.LossFinal = lossFinal
		row.LossSC = lossSC
		row.Learned = lossFiniteOK(lossInit, lossFinal, requiresLearn) &&
			(!requiresLearn || trainingOK(lossInit, lossFinal, tc.dtype))

		finalizeTrainingNet(netMC, tc)
		memTrain := readMemSnapshot()
		row.MemHeapTrain = formatBytes(memTrain.HeapAlloc)
		row.WeightBytes = formatBytes(networkWeightBytes(netMC)) + " (trained-native)"

		ckptPath := saveCheckpoint(netMC, s.CheckpointTag, tc.name)
		if ckptPath != "" {
			if st, err := os.Stat(ckptPath); err == nil {
				row.Checkpoint = formatBytes(uint64(st.Size()))
			}
		}
		entityPath := saveEntityCheckpoint(netMC, s.CheckpointTag, tc.name)
		if entityPath != "" {
			if st, err := os.Stat(entityPath); err == nil {
				row.EntityCheckpoint = formatBytes(uint64(st.Size()))
			}
		}

		after := checkSaveReload(netMC, input, target, tc, lossFinal, phaseAfter, s.PrimaryType)
		row.AfterBucket = after.bucket.String()
		row.AfterOK = after.pass
		row.ReloadFwdDiff = after.forwardDiff
		row.ReloadLossDelta = after.lossDelta
		row.TrainedLoss = after.trainedLoss
		row.ReloadedLoss = after.reloadedLoss
		if !after.nativeOK {
			row.NativeOK = false
		}
		entityAfter := checkEntitySaveReload(netMC, input, target, tc, lossFinal, phaseAfter, s.PrimaryType)
		row.EntityAfterOK = entityAfter.pass
		if !entityAfter.nativeOK {
			row.EntityNativeOK = false
		}

		// CPU SC/MC parity + train + JSON/.entity save/reload.
		row.OverallOK = row.BeforeOK && row.AfterOK && row.EntityBeforeOK && row.EntityAfterOK &&
			row.Learned && row.DetOK && lossFiniteOK(lossInit, lossFinal, requiresLearn)
		if poly.Plan9SimdForwardForLayer(s.PrimaryType) {
			row.OverallOK = row.OverallOK && row.TrainSimdOK
		}
		rows = append(rows, row)

		if row.OverallOK {
			passed++
			if poly.Plan9SimdForwardForLayer(s.PrimaryType) {
				fmt.Printf("PASS  loss %.4e→%.4e det=%s fwd SC/MC/SIMD=%s/%s/%s bwd SC/MC/SIMD=%s/%s/%s\n",
					lossInit, lossFinal, markOK(row.DetOK),
					row.FwdSCDur, row.FwdMCDur, row.FwdSimdDur,
					row.BwdSCDur, row.BwdMCDur, row.BwdSimdDur)
			} else {
				fmt.Printf("PASS  loss %.4e→%.4e det=%s json=%s entity=%s fwd/bwd SC=%s/%s MC=%s/%s\n",
					lossInit, lossFinal, markOK(row.DetOK), markOK(row.AfterOK), markOK(row.EntityAfterOK),
					row.FwdSCDur, row.BwdSCDur, row.FwdMCDur, row.BwdMCDur)
			}
		} else {
			failed++
			if poly.Plan9SimdForwardForLayer(s.PrimaryType) {
				fmt.Printf("FAIL  loss %.4e→%.4e learn=%s det=%s fwdΔ(SC↔MC)=%.2e fwdΔ(SC↔SIMD)=%.2e bwdΔ(SC↔MC)=%.2e bwdΔ(SC↔SIMD)=%.2e\n",
					lossInit, lossFinal, markOK(row.Learned), markOK(row.DetOK),
					row.FwdSCMC, row.FwdTiledSimd, row.BwdSCMC, row.BwdTiledSimd)
			} else {
				fmt.Printf("FAIL  loss %.4e→%.4e learn=%s det=%s reload_Δloss=%.2e\n",
					lossInit, lossFinal, markOK(row.Learned), markOK(row.DetOK), row.ReloadLossDelta)
			}
		}
		_ = lossSC
	}

	simdLayer := poly.Plan9SimdForwardForLayer(s.PrimaryType)
	PrintDeterminismTable(suiteLabel, rows, simdLayer)
	PrintForwardBackwardTimingTable(suiteLabel, rows, simdLayer)
	PrintMemoryTable(suiteLabel, rows)
	PrintTimingTable(suiteLabel, rows, simdLayer)
	PrintTrainedReloadTable(suiteLabel, rows, simdLayer)
	PrintDTypeResultsTable(suiteLabel, rows, simdLayer)
	RegisterLayerSummary(suiteLabel, passed, failed, rows)
	return failed == 0
}
