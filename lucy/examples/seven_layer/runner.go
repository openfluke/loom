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

func checkSaveReload(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32], tc dtypeCase, refLoss float64, phase savePhase) saveResult {
	r := saveResult{}
	finalizeTrainingNet(net, tc)
	setCPUMode(net, true, false)

	out0, _, _ := poly.ForwardPolymorphic(net, input)
	baseline := append([]float32(nil), out0.Data...)

	wire, err := poly.SerializeNetwork(net)
	if err != nil {
		r.err = err.Error()
		r.bucket = specFatal
		return r
	}
	reloaded, err := poly.DeserializeNetwork(wire)
	if err != nil {
		r.err = err.Error()
		r.bucket = specFatal
		return r
	}
	wireLayerTree(reloaded)
	setCPUMode(reloaded, true, false)

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

	r.nativeOK = nativePersistenceOK(net, reloaded, wire, tc)
	maxBucket := saveReloadMaxBucket(phase, tc.dtype)
	fwdTol := tc.tolerance
	wTol := tc.tolerance
	if poly.IsDenseNativeTrainDType(tc.dtype) {
		wTol = tc.tolerance * 100
	}
	if phase == phaseAfter {
		fwdTol = tc.tolerance * 100
		if poly.IsDenseNativeTrainDType(tc.dtype) {
			fwdTol = tc.tolerance * 1000
		}
	}
	r.pass = r.forwardDiff <= fwdTol && r.weightDiff <= wTol &&
		r.bucket <= maxBucket && r.nativeOK && r.err == ""
	return r
}

func nativeWeightDiff(a, b *poly.VolumetricNetwork, dt poly.DType) float64 {
	if a.Layers[0].WeightStore == nil || b.Layers[0].WeightStore == nil {
		return 0
	}
	if poly.IsDenseNativeTrainDType(dt) {
		b64a, scaleA, oka := poly.LayerNativePersistenceSnapshot(a.Layers[0].WeightStore, dt)
		b64b, scaleB, okb := poly.LayerNativePersistenceSnapshot(b.Layers[0].WeightStore, dt)
		if !oka || !okb || b64a != b64b || scaleA != scaleB {
			return 1
		}
		return 0
	}
	return maxWeightDiff(a, b)
}

func nativePersistenceOK(net, reloaded *poly.VolumetricNetwork, wire []byte, tc dtypeCase) bool {
	if net.Layers[0].WeightStore == nil {
		return true
	}
	b64, scale, native, fileErr := poly.LayerPersistenceFromJSON(wire, 0)
	l2 := reloaded.GetLayer(0, 0, 0, 0)
	if fileErr != nil || !native || b64 == "" || l2 == nil || l2.WeightStore == nil {
		return false
	}
	decoded, decErr := poly.DecodeNativeWeights(b64, l2.DType)
	loaded := l2.WeightStore.Versions[tc.dtype]
	if loaded == nil {
		loaded = l2.WeightStore.GetNative(tc.dtype)
	}
	return decErr == nil && loaded != nil && l2.WeightStore.Scale == scale &&
		poly.NativeWeightsEncoded(decoded, loaded, tc.dtype)
}

func trainCPU(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32], mode poly.TrainingMode, tc dtypeCase) (*poly.TrainingResult, time.Duration, error) {
	configureTrainingNet(net, tc)
	prepareTrainingNet(net, tc.dtype)
	cfg := poly.DefaultTrainingConfig()
	cfg.Epochs = trainEpochs
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

// RunLayerSuite executes the full [7] matrix for one layer type.
func RunLayerSuite(s LayerSuite) bool {
	asmSt := layerAsmStatus(s.PrimaryType)

	fmt.Println()
	fmt.Println("══════════════════════════════════════════════════════════════════════")
	fmt.Printf("  Loom seven-layer %s — JSON · CPU SC/MC · train · save/reload\n", s.Name)
	fmt.Println(s.Banner)
	fmt.Println("══════════════════════════════════════════════════════════════════════")
	fmt.Printf("  %d dtypes × %d epochs · ASM: %s\n", len(allDTypes), trainEpochs, asmSt.Note)
	if asmSt.ForwardCapable {
		fmt.Println("  ASM enabled via net.UseAsmForward after JSON build (Dense layers only)")
	}
	fmt.Println()

	var rows []DTypeRow
	passed, failed := 0, 0

	for _, tc := range allDTypes {
		fmt.Printf("  · %-10s ", tc.name)
		row := DTypeRow{DType: tc.name, AsmCapable: asmSt.ForwardCapable}

		net, err := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
		if err != nil {
			row.Err = "BUILD"
			rows = append(rows, row)
			failed++
			fmt.Println("BUILD ERR")
			continue
		}
		configureTrainingNet(net, tc)
		applyDType(net, tc)
		configureInferenceNet(net)
		input := s.MakeInput()
		target := s.MakeTarget(net, input)

		mem0 := readMemSnapshot()
		row.MemHeap = formatBytes(mem0.HeapAlloc)
		row.MemSys = formatBytes(mem0.Sys)
		row.WeightBytes = formatBytes(networkWeightBytes(net)) + " (infer)"

		// Forward determinism: CPU Go SC vs MC (Go tiled path; ASM checked separately for floats).
		fwdSC := captureForward(net, input, false, false)
		fwdMC := captureForward(net, input, true, false)
		row.FwdSCDur = formatDur(fwdSC.dur)
		row.FwdMCDur = formatDur(fwdMC.dur)
		row.FwdSCMC = maxAbsDiff(fwdSC.out, fwdMC.out)

		// Backward determinism: SC vs MC
		bwdSC := captureBackward(net, input, target, false)
		bwdMC := captureBackward(net, input, target, true)
		row.BwdSCDur = formatDur(bwdSC.dur)
		row.BwdMCDur = formatDur(bwdMC.dur)
		row.BwdSCMC = maxAbsDiff(append(bwdSC.dx, bwdSC.dw...), append(bwdMC.dx, bwdMC.dw...))

		// ASM forward (Dense float paths only — native quant uses integer matmul in ASM).
		if asmSt.ForwardCapable && requiresAsmDeterminism(tc.dtype) {
			SetNetworkAsm(net, true)
			asmSt.RuntimeEnabled = net.UseAsmForward
			fwdGo := captureForward(net, input, true, false)
			fwdAsm := captureForward(net, input, true, true)
			row.FwdGoAsm = maxAbsDiff(fwdGo.out, fwdAsm.out)
			row.AsmUsed = true
			row.AsmOK = row.FwdGoAsm <= tc.tolerance
			SetNetworkAsm(net, false)
		} else if asmSt.ForwardCapable {
			row.AsmUsed = false
			row.AsmOK = true
			row.FwdGoAsm = 0
		} else {
			// Non-Dense: verify toggling ASM flag does not break CPU paths
			SetNetworkAsm(net, true)
			fwdWithFlag := captureForward(net, input, true, true)
			SetNetworkAsm(net, false)
			fwdWithout := captureForward(net, input, true, false)
			row.FwdGoAsm = maxAbsDiff(fwdWithFlag.out, fwdWithout.out)
			row.AsmUsed = false
			row.AsmOK = row.FwdGoAsm <= tc.tolerance // should match (ASM ignored)
		}

		detTol := tc.tolerance
		if detTol < 1e-10 {
			detTol = 1e-10
		}
		row.DetOK = row.FwdSCMC <= detTol && row.BwdSCMC <= detTol*10 &&
			(!requiresAsmDeterminism(tc.dtype) || row.FwdGoAsm <= tc.tolerance)

		lossBefore := forwardLoss(net, input, target)
		before := checkSaveReload(net, input, target, tc, lossBefore, phaseBefore)
		row.BeforeBucket = before.bucket.String()
		row.BeforeOK = before.pass
		row.NativeOK = before.nativeOK

		// Train CPU SC then MC (fresh weight copy via rebuild for fairness)
		netSC, _ := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
		applyDType(netSC, tc)
		configureTrainingNet(netSC, tc)
		netSC.ReleaseFP32MasterWhenIdle = true
		resSC, durSC, err := trainCPU(netSC, input, target, poly.TrainingModeCPUSC, tc)
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
		row.TrainSCSps = samplesPerSec(durSC, trainEpochs)

		netMC, _ := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
		applyDType(netMC, tc)
		configureTrainingNet(netMC, tc)
		netMC.ReleaseFP32MasterWhenIdle = true
		resMC, durMC, err := trainCPU(netMC, input, target, poly.TrainingModeCPUMC, tc)
		if err != nil {
			row.Err = "TRAIN-MC"
			rows = append(rows, row)
			failed++
			fmt.Println("TRAIN MC ERR")
			continue
		}
		row.TrainMCDur = formatDur(durMC)
		row.TrainMCSps = samplesPerSec(durMC, trainEpochs)

		lossInit := resMC.LossHistory[0]
		lossFinal := resMC.FinalLoss
		if len(resMC.LossHistory) > 0 {
			lossFinal = resMC.LossHistory[len(resMC.LossHistory)-1]
		}
		row.LossInit = lossInit
		row.LossFinal = lossFinal
		row.Learned = trainingOK(lossInit, lossFinal, tc.dtype)

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

		after := checkSaveReload(netMC, input, target, tc, lossFinal, phaseAfter)
		row.AfterBucket = after.bucket.String()
		row.AfterOK = after.pass
		row.ReloadFwdDiff = after.forwardDiff
		row.ReloadLossDelta = after.lossDelta
		row.TrainedLoss = after.trainedLoss
		row.ReloadedLoss = after.reloadedLoss
		if !after.nativeOK {
			row.NativeOK = false
		}

		// CPU SC/MC parity + train + save/reload (ASM reported but not required for pass).
		row.OverallOK = row.BeforeOK && row.AfterOK && row.Learned && row.DetOK
		rows = append(rows, row)

		asmTag := "N/A"
		if asmSt.ForwardCapable {
			asmTag = markOK(row.AsmOK)
		}
		if row.OverallOK {
			passed++
			fmt.Printf("PASS  loss %.4e→%.4e det=%s reload=%s fwd/bwd SC=%s/%s MC=%s/%s mem=%s ckpt=%s\n",
				lossInit, lossFinal, markOK(row.DetOK), markOK(row.AfterOK),
				row.FwdSCDur, row.BwdSCDur, row.FwdMCDur, row.BwdMCDur,
				row.MemHeapTrain, row.Checkpoint)
		} else {
			failed++
			fmt.Printf("FAIL  loss %.4e→%.4e learn=%s save=%s det=%s asm=%s reload_Δloss=%.2e\n",
				lossInit, lossFinal, markOK(row.Learned), markOK(row.BeforeOK && row.AfterOK),
				markOK(row.DetOK), asmTag, row.ReloadLossDelta)
		}
		_ = lossSC
	}

	PrintDeterminismTable(s.Name, rows)
	PrintForwardBackwardTimingTable(s.Name, rows)
	PrintMemoryTable(s.Name, rows)
	PrintTimingTable(s.Name, rows)
	PrintTrainedReloadTable(s.Name, rows)
	PrintDTypeResultsTable(s.Name, rows)
	RegisterLayerSummary(s.Name, passed, failed, rows)
	return failed == 0
}
