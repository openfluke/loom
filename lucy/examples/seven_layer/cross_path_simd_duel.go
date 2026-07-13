package sevenlayer

import (
	"fmt"
	"math"

	"github.com/openfluke/loom/poly"
)

// runCrossPathSimdDuelSuite benchmarks only QAT-SIMD vs Nat-SIMD at 3³ (grid menu [5]).
func runCrossPathSimdDuelSuite(s LayerSuite, primary poly.LayerType) {
	activeBenchIters = benchItersForGrid(s.Grid)
	simdLayer := poly.Plan9SimdForwardForLayer(primary)
	if !simdLayer {
		fmt.Printf("\n  %s: skipped (layer has no Plan 9 SIMD forward path)\n", s.Name)
		return
	}

	epochs := trainEpochsForGrid(s.Grid)
	requiresLearn := layerRequiresLearn(primary)

	fmt.Printf("\n  ┌─ %s SIMD duel · %s · %d-layer stack ─────────────────────────────\n",
		s.Name, s.Grid, s.Grid.StackLayers())
	fmt.Printf("  │ QAT-SIMD (tiled FP32) vs Nat-SIMD (UseExactDType) · %d train epochs\n", epochs)

	var rows []crossPathRow
	layerTally := newTestTally()
	passed, failed := 0, 0

	for _, tc := range allDTypes {
		fmt.Printf("  · %-10s ", tc.name)
		row := crossPathRow{DType: tc.name, NativeApplicable: poly.IsLayerNativeExactDType(tc.dtype)}
		tally := newTestTally()

		if !row.NativeApplicable {
			row.Err = "NO-NATIVE"
			rows = append(rows, row)
			failed++
			fmt.Println("SKIP (no native path)")
			continue
		}

		net, err := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
		if err != nil {
			row.Err = "BUILD"
			rows = append(rows, row)
			failed++
			fmt.Println("BUILD ERR")
			continue
		}
		applyDType(net, tc)
		prepareTrainingNet(net, tc.dtype)
		finalizeTrainingNet(net, tc)
		input := s.MakeInput()
		target := s.MakeTarget(net, input)

		// ── QAT SIMD forward / backward ──────────────────────────────────
		configureTiledNet(net)
		resetNetwork(net)
		fwdSimd := captureForwardSimd(net, input, true)
		row.fwdSimd = fwdSimd.dur
		row.FwdSimdDur = formatDur(fwdSimd.dur)
		row.FwdSimdOK = len(fwdSimd.out) > 0 && tensorFinite(fwdSimd.out)
		tally.record("tiled.fwd.simd", row.FwdSimdOK)

		resetNetwork(net)
		bwdSimd := captureBackwardSimd(net, input, target, true)
		row.bwdSimd = bwdSimd.dur
		row.BwdSimdDur = formatDur(bwdSimd.dur)
		row.BwdSimdOK = len(bwdSimd.dx) > 0 && tensorFinite(bwdSimd.dx)
		if primary != poly.LayerResidual {
			row.BwdSimdOK = row.BwdSimdOK && len(bwdSimd.dw) > 0 && tensorFinite(bwdSimd.dw)
		}
		tally.record("tiled.bwd.simd", row.BwdSimdOK)

		row.LossInit = forwardLoss(net, input, target)

		// ── Native SIMD forward / backward ───────────────────────────────
		configureNativeNet(net, tc)
		row.NativePathOK = verifyNativePath(net, primary, tc)
		tally.record("native.path", row.NativePathOK)

		resetNetwork(net)
		fwdNatSimd := captureForwardSimd(net, input, true)
		row.natFwdSimd = fwdNatSimd.dur
		row.NatFwdSimdDur = formatDur(fwdNatSimd.dur)
		row.NatFwdSimdOK = len(fwdNatSimd.out) > 0 && tensorFinite(fwdNatSimd.out)
		tally.record("native.fwd.simd", row.NatFwdSimdOK)

		resetNetwork(net)
		bwdNatSimd := captureBackwardSimd(net, input, target, true)
		row.natBwdSimd = bwdNatSimd.dur
		row.NatBwdSimdDur = formatDur(bwdNatSimd.dur)
		row.NatBwdSimdFinite = len(bwdNatSimd.dx) > 0 && tensorFinite(bwdNatSimd.dx)
		if primary != poly.LayerResidual {
			row.NatBwdSimdFinite = row.NatBwdSimdFinite && len(bwdNatSimd.dw) > 0 && tensorFinite(bwdNatSimd.dw)
		}
		tally.record("native.bwd.simd", row.NatBwdSimdFinite)

		// ── Training: QAT-SIMD vs Nat-SIMD ───────────────────────────────
		netSimd, _ := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
		applyDType(netSimd, tc)
		configureTrainingNet(netSimd, tc, primary)
		netSimd.ReleaseFP32MasterWhenIdle = true
		resSimd, durSimd, err := trainCPU(netSimd, input, target, poly.TrainingModeCPUSimd, tc, primary, epochs)
		if err != nil {
			row.Err = "TRAIN-QAT-SIMD"
			rows = append(rows, row)
			failed++
			fmt.Println("TRAIN QAT-SIMD ERR")
			continue
		}
		row.TrainSimdDur = formatDur(durSimd)
		row.trainSimd = durSimd
		row.LossFinalSimd = resSimd.FinalLoss
		if len(resSimd.LossHistory) > 0 {
			row.LossFinalSimd = resSimd.LossHistory[len(resSimd.LossHistory)-1]
		}
		lossInitSimd := resSimd.LossHistory[0]
		if row.LossInit == 0 || math.IsNaN(row.LossInit) {
			row.LossInit = lossInitSimd
		}
		row.TrainSimdOK = lossFiniteOK(lossInitSimd, row.LossFinalSimd, requiresLearn) &&
			(!requiresLearn || trainingOK(lossInitSimd, row.LossFinalSimd, tc.dtype))
		tally.record("train.simd", row.TrainSimdOK)

		netNatSimd, _ := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
		applyDType(netNatSimd, tc)
		configureNativeNet(netNatSimd, tc)
		resNatSimd, durNatSimd, err := trainNativeExact(netNatSimd, input, target, tc, poly.TrainingModeCPUSimd, epochs)
		row.TrainNativeSimdDur = formatDur(durNatSimd)
		row.trainNativeSimd = durNatSimd
		if err != nil {
			row.Err = "TRAIN-NAT-SIMD"
			rows = append(rows, row)
			failed++
			fmt.Println("TRAIN NAT-SIMD ERR")
			continue
		}
		row.LossFinalNativeSimd = resNatSimd.FinalLoss
		if len(resNatSimd.LossHistory) > 0 {
			row.LossFinalNativeSimd = resNatSimd.LossHistory[len(resNatSimd.LossHistory)-1]
		}
		lossInitNatSimd := resNatSimd.LossHistory[0]
		row.TrainNativeSimdOK = lossFiniteOK(lossInitNatSimd, row.LossFinalNativeSimd, requiresLearn) &&
			(!requiresLearn || trainingOK(lossInitNatSimd, row.LossFinalNativeSimd, tc.dtype))
		tally.record("train.native.simd", row.TrainNativeSimdOK)

		computeSimdDuelComparisons(&row)

		row.TestsTotal = tally.total
		row.TestsPassed = tally.passed
		row.OverallOK = row.TestsPassed == row.TestsTotal && row.Err == ""
		rows = append(rows, row)
		layerTally.merge(tally)

		if row.OverallOK {
			passed++
			fmt.Printf("PASS  %d/%d  fwd %s %s  bwd %s %s  train %s %s\n",
				row.TestsPassed, row.TestsTotal,
				row.fwdWinner, row.fwdWinRatio,
				row.bwdWinner, row.bwdWinRatio,
				row.trainWinner, row.trainWinRatio)
		} else {
			failed++
			fmt.Printf("FAIL  %d/%d tests  err=%s\n", row.TestsPassed, row.TestsTotal, row.Err)
		}
	}

	printSimdDuelTimingTable(s.Name, rows)
	printSimdDuelComparisonTable(s.Name, rows)
	printDtypeSpreadTable(s.Name, "Best SIMD path per dtype (QAT-SIMD vs Nat-SIMD)", rows, simdDuelDtypeSpreadPhases)
	printSimdDuelTrainTable(s.Name, rows, epochs)
	printSimdDuelTestTally(s.Name, layerTally)
	fmt.Printf("\n  %s SIMD duel · %s: %d passed · %d failed (of %d dtypes) · tests %d/%d\n",
		s.Name, s.Grid, passed, failed, len(rows), layerTally.passed, layerTally.total)

	crossPathSessionLayers = append(crossPathSessionLayers, crossPathLayerSummary{
		Name: s.Name, Grid: s.Grid, Passed: passed, Failed: failed, Rows: rows,
		TestsTotal: layerTally.total, TestsPassed: layerTally.passed,
	})
	crossPathSessionTally.merge(layerTally)
}

func computeSimdDuelComparisons(row *crossPathRow) {
	row.duelFwd = makePair("QAT SIMD-f", row.fwdSimd, row.natFwdSimd, "NatS-f")
	row.duelBwd = makePair("QAT SIMD-b", row.bwdSimd, row.natBwdSimd, "NatS-b")
	row.duelTrain = makePair("QAT SIMD", row.trainSimd, row.trainNativeSimd, "NatS")

	qatFwd := namedDur{"SIMD-f", row.fwdSimd}
	natFwd := namedDur{"NatS-f", row.natFwdSimd}
	row.fwdWinner, row.fwdWinRatio, row.fwdWinFaster = paradigmWinner(qatFwd, natFwd)

	qatBwd := namedDur{"SIMD-b", row.bwdSimd}
	natBwd := namedDur{"NatS-b", row.natBwdSimd}
	row.bwdWinner, row.bwdWinRatio, row.bwdWinFaster = paradigmWinner(qatBwd, natBwd)

	qatTrain := namedDur{"SIMD", row.trainSimd}
	natTrain := namedDur{"NatS", row.trainNativeSimd}
	row.trainWinner, row.trainWinRatio, row.trainWinFaster = paradigmWinner(qatTrain, natTrain)
}

func printSimdDuelTimingTable(layerName string, rows []crossPathRow) {
	fmt.Printf("\n  ┌─ %s SIMD duel — raw timing (fwd / bwd) ───────────────────────────\n", layerName)
	fmt.Printf("  │ %-10s │ %-10s %-10s │ %-10s %-10s\n",
		"DType", "QAT SIMD-f", "NatS-f", "QAT SIMD-b", "NatS-b")
	fmt.Println("  ├──────────┼──────────┬──────────┼──────────┬──────────")
	for _, r := range rows {
		if r.Err != "" {
			fmt.Printf("  │ %-10s │ ERR %s\n", r.DType, r.Err)
			continue
		}
		fmt.Printf("  │ %-10s │ %-10s %-10s │ %-10s %-10s\n",
			r.DType, r.FwdSimdDur, r.NatFwdSimdDur, r.BwdSimdDur, r.NatBwdSimdDur)
	}
	fmt.Println("  └──────────┴──────────┴──────────┴──────────┴──────────")
}

func printSimdDuelComparisonTable(layerName string, rows []crossPathRow) {
	fmt.Printf("\n  ┌─ %s SIMD duel — QAT-SIMD vs Nat-SIMD ─────────────────────────────\n", layerName)
	fmt.Println("  │ Apples to apples: fastest SIMD path per paradigm @ 3³")
	fmt.Printf("  │ %-10s │ %-30s %-5s %-7s │ %-14s %-5s %-7s\n",
		"DType", "fwd QAT SIMD-f→NatS-f", "×", "faster", "fwd winner", "×", "faster")
	fmt.Println("  ├──────────┼────────────────────────────────────┼────────────────────────")
	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		fmt.Printf("  │ %-10s │ %-30s %-5s %-7s │ %-14s %-5s %-7s\n",
			r.DType, r.duelFwd.label(), r.duelFwd.ratio(), r.duelFwd.fasterPct(),
			r.fwdWinner, r.fwdWinRatio, r.fwdWinFaster)
	}
	fmt.Println("  ├──────────┼────────────────────────────────────┼────────────────────────")
	fmt.Printf("  │ %-10s │ %-30s %-5s %-7s │ %-14s %-5s %-7s\n",
		"DType", "bwd QAT SIMD-b→NatS-b", "×", "faster", "bwd winner", "×", "faster")
	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		fmt.Printf("  │ %-10s │ %-30s %-5s %-7s │ %-14s %-5s %-7s\n",
			r.DType, r.duelBwd.label(), r.duelBwd.ratio(), r.duelBwd.fasterPct(),
			r.bwdWinner, r.bwdWinRatio, r.bwdWinFaster)
	}
	fmt.Println("  ├──────────┼────────────────────────────────────┼────────────────────────")
	fmt.Printf("  │ %-10s │ %-30s %-5s %-7s │ %-14s %-5s %-7s\n",
		"DType", "train QAT SIMD→NatS", "×", "faster", "train winner", "×", "faster")
	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		fmt.Printf("  │ %-10s │ %-30s %-5s %-7s │ %-14s %-5s %-7s\n",
			r.DType, r.duelTrain.label(), r.duelTrain.ratio(), r.duelTrain.fasterPct(),
			r.trainWinner, r.trainWinRatio, r.trainWinFaster)
	}
	fmt.Println("  └──────────┴────────────────────────────────────┴────────────────────────")
}

func printSimdDuelTrainTable(layerName string, rows []crossPathRow, epochs int) {
	fmt.Printf("\n  ┌─ %s SIMD duel — training (%d epochs) ───────────────────────────────\n", layerName, epochs)
	fmt.Printf("  │ %-10s │ %8s %8s %8s │ %-6s %-6s\n",
		"DType", "Loss₀", "QAT-SIMD", "Nat-SIMD", "QAT", "NatS")
	fmt.Println("  ├──────────────────────────────────────────────────────────────────────")
	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		fmt.Printf("  │ %-10s │ %8.4f %8.4f %8.4f │ %-6s %-6s\n",
			r.DType, r.LossInit, r.LossFinalSimd, r.LossFinalNativeSimd,
			markOK(r.TrainSimdOK), markOK(r.TrainNativeSimdOK))
	}
	fmt.Println("  └──────────────────────────────────────────────────────────────────────")
}

func printSimdDuelTestTally(layerName string, t *testTally) {
	fmt.Printf("\n  ┌─ %s SIMD duel test tally ─────────────────────────────────────────\n", layerName)
	cats := []string{
		"native.path",
		"tiled.fwd.simd", "tiled.bwd.simd",
		"native.fwd.simd", "native.bwd.simd",
		"train.simd", "train.native.simd",
	}
	for _, cat := range cats {
		v, ok := t.byCat[cat]
		if !ok || v[1] == 0 {
			continue
		}
		fmt.Printf("  │ %-28s %4d / %4d\n", cat, v[0], v[1])
	}
	fmt.Printf("  │ %-28s %4d / %4d\n", "TOTAL (gated)", t.passed, t.total)
	fmt.Println("  └──────────────────────────────────────────────────────────────────────")
}
