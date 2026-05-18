package poly

import (
	"fmt"
	"math"
)

// LayerActionRecord is one sub-layer step during a traced CPU forward.
type LayerActionRecord struct {
	TokenOrdinal int // -1 = prefill pass; 0+ = decode token index
	Step         int
	StepTotal    int
	Block        int // 0-based decoder block; -1 for final norm
	Label        string
	Repeated     bool
	Mean         float64
	Std          float64
	Min          float64
	Max          float64
	L2           float64
}

// layerTraceState is active for the duration of one Generate call when GenOptions.LayerTrace is set.
type layerTraceState struct {
	tokenOrdinal    int
	repeatBlock     int // 0-based decoder block to run twice; <0 disables
	printEach       bool
	recording       bool // false during untraced prefill
	records         []LayerActionRecord
	forwardPass     int
	stepsPerForward int
	numBlocks       int
}

func (t *Transformer[T]) cpuLayerTraceStepTotal() int {
	n := len(t.Network.Layers) / 4
	total := n * 6
	if t.finalNormLayer != nil {
		total++
	}
	return total
}

func (t *Transformer[T]) beginLayerTrace(opts GenOptions) {
	if !opts.LayerTrace {
		t.layerTrace = nil
		return
	}
	nb := len(t.Network.Layers) / 4
	t.layerTrace = &layerTraceState{
		tokenOrdinal:    -1,
		repeatBlock:     opts.RepeatDecoderBlock,
		printEach:       !opts.LayerTraceSilent,
		recording:       opts.LayerTracePrefill,
		records:         make([]LayerActionRecord, 0, t.cpuLayerTraceStepTotal()*8),
		stepsPerForward: t.cpuLayerTraceStepTotal(),
		numBlocks:       nb,
	}
}

func (t *Transformer[T]) setLayerTraceRecording(on bool) {
	if t.layerTrace != nil {
		t.layerTrace.recording = on
	}
}

func (t *Transformer[T]) layerTraceRecording() bool {
	return t.layerTrace != nil && t.layerTrace.recording
}

func (t *Transformer[T]) printLayerTraceBanner(phase string, promptTokens, decodeTokens int) {
	st := t.layerTrace
	if st == nil {
		return
	}
	steps := st.stepsPerForward
	switch phase {
	case "prefill":
		fmt.Printf("\n── prefill trace: %d prompt token(s) → %d sub-layer steps (%d blocks × 6 + final norm) ──\n",
			promptTokens, steps, st.numBlocks)
	case "prefill-skip":
		fmt.Printf("\n── prefill: %d prompt token(s) — not traced (%d sub-layer steps if you enable prefill trace) ──\n",
			promptTokens, steps)
	case "decode":
		fmt.Printf("\n── decode trace: %d new token(s) → %d sub-layer steps each (%d blocks × 6 + final norm) ──\n",
			decodeTokens, steps, st.numBlocks)
	}
}

func (t *Transformer[T]) endLayerTrace() {
	t.layerTrace = nil
}

func (t *Transformer[T]) layerTraceActive() bool {
	return t.layerTrace != nil
}

func (t *Transformer[T]) layerTraceSetTokenOrdinal(ord int) {
	if t.layerTrace != nil {
		t.layerTrace.tokenOrdinal = ord
		t.layerTrace.forwardPass++
	}
}

// LayerTraceRecords returns records from the last Generate call (empty if tracing was off).
func (t *Transformer[T]) LayerTraceRecords() []LayerActionRecord {
	if t.layerTrace == nil {
		return nil
	}
	out := make([]LayerActionRecord, len(t.layerTrace.records))
	copy(out, t.layerTrace.records)
	return out
}

func summarizeTensorLastRow[T Numeric](t *Tensor[T], hiddenSize int) (mean, std, min, max, l2 float64) {
	if t == nil || len(t.Data) == 0 || hiddenSize <= 0 {
		return 0, 0, 0, 0, 0
	}
	row := t.Data
	if n := len(t.Data); n >= hiddenSize {
		row = t.Data[n-hiddenSize:]
	}
	min = math.MaxFloat64
	max = -math.MaxFloat64
	var sum, sumSq float64
	for _, v := range row {
		f := float64(v)
		sum += f
		sumSq += f * f
		if f < min {
			min = f
		}
		if f > max {
			max = f
		}
	}
	mean = sum / float64(len(row))
	variance := sumSq/float64(len(row)) - mean*mean
	if variance < 0 {
		variance = 0
	}
	std = math.Sqrt(variance)
	l2 = math.Sqrt(sumSq)
	return mean, std, min, max, l2
}

func (t *Transformer[T]) recordLayerAction(step, stepTotal, block int, label string, repeated bool, hidden *Tensor[T]) {
	st := t.layerTrace
	if st == nil || !st.recording {
		return
	}
	mean, std, min, max, l2 := summarizeTensorLastRow(hidden, t.HiddenSize)
	rec := LayerActionRecord{
		TokenOrdinal: st.tokenOrdinal,
		Step:         step,
		StepTotal:    stepTotal,
		Block:        block,
		Label:        label,
		Repeated:     repeated,
		Mean:         mean,
		Std:          std,
		Min:          min,
		Max:          max,
		L2:           l2,
	}
	st.records = append(st.records, rec)
	if st.printEach {
		prefix := "layer trace"
		if repeated {
			prefix = "layer trace REPEAT"
		}
		tokLabel := "prefill"
		if st.tokenOrdinal >= 0 {
			tokLabel = fmt.Sprintf("decode#%d", st.tokenOrdinal)
		}
		fmt.Printf("[%s] %s | step %d/%d block=%d %s | last-row mean=%.5f std=%.5f min=%.4f max=%.4f L2=%.4f\n",
			prefix, tokLabel, step, stepTotal, block, label, mean, std, min, max, l2)
	}
}

func (t *Transformer[T]) runDecoderBlockTraced(b, numBlocks int, current *Tensor[T], repeated bool) *Tensor[T] {
	base := b * 4
	total := t.layerTrace.stepsPerForward
	step := b * 6

	prefix := ""
	if repeated {
		prefix = "REPEAT "
	}

	residual := current.Clone()

	lNorm1 := &t.Network.Layers[base+0]
	_, current = RMSNormForwardPolymorphic(lNorm1, current)
	step++
	t.recordLayerAction(step, total, b, fmt.Sprintf("%sblock %d/%d RMSNorm (pre-attn)", prefix, b+1, numBlocks), repeated, current)

	lMHA := &t.Network.Layers[base+1]
	_, current = MHAForwardPolymorphic(lMHA, current)
	step++
	t.recordLayerAction(step, total, b, fmt.Sprintf("%sblock %d/%d MHA", prefix, b+1, numBlocks), repeated, current)

	current.Add(residual)
	step++
	t.recordLayerAction(step, total, b, fmt.Sprintf("%sblock %d/%d residual (attn)", prefix, b+1, numBlocks), repeated, current)

	residual = current.Clone()

	lNorm2 := &t.Network.Layers[base+2]
	_, current = RMSNormForwardPolymorphic(lNorm2, current)
	step++
	t.recordLayerAction(step, total, b, fmt.Sprintf("%sblock %d/%d RMSNorm (pre-mlp)", prefix, b+1, numBlocks), repeated, current)

	lMLP := &t.Network.Layers[base+3]
	_, current = SwiGLUForwardPolymorphic(lMLP, current)
	step++
	t.recordLayerAction(step, total, b, fmt.Sprintf("%sblock %d/%d SwiGLU", prefix, b+1, numBlocks), repeated, current)

	current.Add(residual)
	step++
	t.recordLayerAction(step, total, b, fmt.Sprintf("%sblock %d/%d residual (mlp)", prefix, b+1, numBlocks), repeated, current)

	return current
}

// forwardOnCPUTraced runs the decoder on CPU with per-sub-layer recording (and optional block repeat).
func (t *Transformer[T]) forwardOnCPUTraced(input *Tensor[T]) *Tensor[T] {
	if t.hostWeightsReleased {
		fmt.Println("⚠️  CPU forward skipped (host weights released after GPU upload).")
		return NewTensor[T](input.Shape...)
	}
	current := input
	numBlocks := len(t.Network.Layers) / 4
	st := t.layerTrace

	for b := 0; b < numBlocks; b++ {
		current = t.runDecoderBlockTraced(b, numBlocks, current, false)
		if st != nil && st.repeatBlock == b {
			fmt.Printf("🔁 Repeating decoder block %d/%d (recording enabled for repeat pass)\n", b+1, numBlocks)
			current = t.runDecoderBlockTraced(b, numBlocks, current, true)
		}
	}

	if t.finalNormLayer != nil {
		_, current = RMSNormForwardPolymorphic(t.finalNormLayer, current)
		step := numBlocks*6 + 1
		total := t.cpuLayerTraceStepTotal()
		if st != nil {
			total = st.stepsPerForward
		}
		t.recordLayerAction(step, total, -1, "final RMSNorm", false, current)
	}

	return current
}
