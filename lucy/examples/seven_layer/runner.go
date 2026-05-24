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
	forwardDiff float64
	weightDiff  float64
	lossDelta   float64
	bucket      spectrum
	nativeOK    bool
	pass        bool
	err         string
}

func forwardLoss(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32]) float64 {
	out, _, _ := poly.ForwardPolymorphic(net, input)
	return poly.CalculateLoss(out, target, "mse")
}

func checkSaveReload(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32], tc dtypeCase, refLoss float64) saveResult {
	r := saveResult{}
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
	r.weightDiff = maxWeightDiff(net, reloaded)
	outR, _, _ := poly.ForwardPolymorphic(reloaded, input)
	r.lossDelta = math.Abs(poly.CalculateLoss(outR, target, "mse") - refLoss)

	r.nativeOK = true
	if net.Layers[0].WeightStore != nil {
		b64, scale, native, fileErr := poly.LayerPersistenceFromJSON(wire, 0)
		l2 := reloaded.GetLayer(0, 0, 0, 0)
		if fileErr != nil || !native || b64 == "" || l2 == nil || l2.WeightStore == nil {
			r.nativeOK = false
		} else {
			decoded, decErr := poly.DecodeNativeWeights(b64, l2.DType)
			loaded := l2.WeightStore.Versions[tc.dtype]
			r.nativeOK = decErr == nil && loaded != nil && l2.WeightStore.Scale == scale &&
				poly.NativeWeightsEncoded(decoded, loaded, tc.dtype)
		}
	}
	r.pass = r.forwardDiff <= tc.tolerance && r.weightDiff <= tc.tolerance*10 &&
		r.bucket <= specLowBit && r.nativeOK && r.err == ""
	return r
}

func trainCPU(net *poly.VolumetricNetwork, input, target *poly.Tensor[float32], mode poly.TrainingMode) (*poly.TrainingResult, time.Duration, error) {
	if err := poly.ConfigureNetworkForMode(net, mode); err != nil {
		return nil, 0, err
	}
	// SC = single-core tiling; MC = multi-core (matches layer-testing [3]).
	for i := range net.Layers {
		l := &net.Layers[i]
		l.UseTiling = true
		switch mode {
		case poly.TrainingModeCPUSC:
			l.EnableMultiCoreTiling = false
		case poly.TrainingModeCPUMC:
			l.EnableMultiCoreTiling = true
		}
	}
	t0 := time.Now()
	res, err := poly.Train(net, []poly.TrainingBatch[float32]{{Input: input, Target: target}}, &poly.TrainingConfig{
		Epochs:       trainEpochs,
		LearningRate: learningRate,
		Mode:         mode,
		Verbose:      false,
		LossType:     "mse",
	})
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
		applyDType(net, tc)
		input := s.MakeInput()
		target := s.MakeTarget(net, input)

		// Forward determinism: CPU Go SC vs MC
		fwdSC := captureForward(net, input, false, false)
		fwdMC := captureForward(net, input, true, false)
		row.FwdSCMC = maxAbsDiff(fwdSC.out, fwdMC.out)

		// Backward determinism: SC vs MC
		bwdSC := captureBackward(net, input, target, false)
		bwdMC := captureBackward(net, input, target, true)
		row.BwdSCMC = maxAbsDiff(append(bwdSC.dx, bwdSC.dw...), append(bwdMC.dx, bwdMC.dw...))

		// ASM forward (Dense only)
		if asmSt.ForwardCapable {
			SetNetworkAsm(net, true)
			asmSt.RuntimeEnabled = net.UseAsmForward
			fwdGo := captureForward(net, input, true, false)
			fwdAsm := captureForward(net, input, true, true)
			row.FwdGoAsm = maxAbsDiff(fwdGo.out, fwdAsm.out)
			row.AsmUsed = true
			row.AsmOK = row.FwdGoAsm <= tc.tolerance
			SetNetworkAsm(net, false)
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
			(!asmSt.ForwardCapable || row.FwdGoAsm <= tc.tolerance)

		lossBefore := forwardLoss(net, input, target)
		before := checkSaveReload(net, input, target, tc, lossBefore)
		row.BeforeBucket = before.bucket.String()
		row.BeforeOK = before.pass
		row.NativeOK = before.nativeOK

		// Train CPU SC then MC (fresh weight copy via rebuild for fairness)
		netSC, _ := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
		applyDType(netSC, tc)
		resSC, durSC, err := trainCPU(netSC, input, target, poly.TrainingModeCPUSC)
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
		resMC, durMC, err := trainCPU(netMC, input, target, poly.TrainingModeCPUMC)
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

		_ = saveCheckpoint(netMC, s.CheckpointTag, tc.name)
		after := checkSaveReload(netMC, input, target, tc, lossFinal)
		row.AfterBucket = after.bucket.String()
		row.AfterOK = after.pass
		if !after.nativeOK {
			row.NativeOK = false
		}

		row.OverallOK = row.BeforeOK && row.AfterOK && row.Learned && row.DetOK &&
			(!asmSt.ForwardCapable || row.AsmOK)
		rows = append(rows, row)

		asmTag := "N/A"
		if asmSt.ForwardCapable {
			asmTag = markOK(row.AsmOK)
		}
		if row.OverallOK {
			passed++
			fmt.Printf("PASS  loss %.4e→%.4e det=%s asm=%s  timing: SC=%s (%.0f/s) MC=%s (%.0f/s)\n",
				lossInit, lossFinal, markOK(row.DetOK), asmTag,
				row.TrainSCDur, row.TrainSCSps, row.TrainMCDur, row.TrainMCSps)
		} else {
			failed++
			fmt.Printf("FAIL  loss %.4e→%.4e learn=%s save=%s det=%s asm=%s\n",
				lossInit, lossFinal, markOK(row.Learned), markOK(row.BeforeOK && row.AfterOK),
				markOK(row.DetOK), asmTag)
		}
		_ = lossSC
	}

	PrintDeterminismTable(s.Name, rows)
	PrintTimingTable(s.Name, rows)
	PrintDTypeResultsTable(s.Name, rows)
	RegisterLayerSummary(s.Name, passed, failed, rows)
	return failed == 0
}
