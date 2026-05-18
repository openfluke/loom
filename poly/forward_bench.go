package poly

import "time"

// ForwardBenchResult holds timing for a single CPU forward pass (prefill-shaped).
type ForwardBenchResult struct {
	Mode         TransformerForwardMode
	SeqRows      int
	WallTime     time.Duration
	TokPerSec    float64
	Pipeline     PipelineForwardStats
	UsedPipeline bool
	// MaxHiddenAbsDiff vs a reference forward (set by ComparePrefillToNormal).
	MaxHiddenAbsDiff float32
}

// BenchmarkCPUPrefill runs ForwardFull on embedded input with the given forward mode.
// Resets KV first. For pipeline mode, fills Pipeline stats.
func (t *Transformer[T]) BenchmarkCPUPrefill(input *Tensor[T], mode TransformerForwardMode) ForwardBenchResult {
	res := ForwardBenchResult{Mode: mode}
	if t == nil || input == nil || len(input.Data) == 0 {
		return res
	}
	h := t.HiddenSize
	if h <= 0 {
		return res
	}
	rows := len(input.Data) / h
	if rows < 1 {
		return res
	}
	res.SeqRows = rows

	prev := t.ForwardMode
	t.ForwardMode = mode
	t.ForwardStepDebug = false
	t.QueueTickPause = nil
	t.PipelineTickPause = nil
	t.Reset()
	t.TakePipelineForwardStats()

	start := time.Now()
	_ = t.ForwardFull(input)
	res.WallTime = time.Since(start)
	if rows > 0 && res.WallTime > 0 {
		res.TokPerSec = float64(rows) / res.WallTime.Seconds()
	}
	if mode == TransformerForwardPipelineCPU {
		res.UsedPipeline = true
		res.Pipeline = t.TakePipelineForwardStats() // stats for this prefill forward only
	} else {
		t.TakePipelineForwardStats() // discard
	}
	t.ForwardMode = prev
	return res
}

// ComparePrefillToNormal runs normal then pipeline prefill on the same embeddings; returns pipeline bench stats and max |Δhidden|.
func (t *Transformer[T]) ComparePrefillToNormal(input *Tensor[T]) (bench ForwardBenchResult, maxDiff float32) {
	if t == nil || input == nil || len(input.Data) == 0 {
		return bench, 0
	}
	h := t.HiddenSize
	rows := len(input.Data) / h
	if rows < 1 {
		return bench, 0
	}
	prev := t.ForwardMode
	t.ForwardStepDebug = false
	t.QueueTickPause = nil
	t.PipelineTickPause = nil

	t.ForwardMode = TransformerForwardNormal
	t.Reset()
	t.TakePipelineForwardStats()
	startN := time.Now()
	outN := t.ForwardFull(input)
	wallN := time.Since(startN)

	t.ForwardMode = TransformerForwardPipelineCPU
	t.Reset()
	t.TakePipelineForwardStats()
	startP := time.Now()
	outP := t.ForwardFull(input)
	wallP := time.Since(startP)
	pipe := t.TakePipelineForwardStats()

	t.ForwardMode = prev

	bench = ForwardBenchResult{
		Mode:         TransformerForwardPipelineCPU,
		SeqRows:      rows,
		WallTime:     wallP,
		TokPerSec:    float64(rows) / wallP.Seconds(),
		Pipeline:     pipe,
		UsedPipeline: true,
	}
	_ = wallN

	if outN != nil && outP != nil {
		for i := 0; i < rows*h; i++ {
			d := float32(outP.Data[i]) - float32(outN.Data[i])
			if d < 0 {
				d = -d
			}
			if d > maxDiff {
				maxDiff = d
			}
		}
	}
	bench.MaxHiddenAbsDiff = maxDiff
	return bench, maxDiff
}
