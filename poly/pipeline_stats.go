package poly

import "fmt"

// PipelineForwardStats summarizes one pipeline (wavefront) forward pass.
// Only meaningful when ForwardMode == TransformerForwardPipelineCPU.
type PipelineForwardStats struct {
	PipelineTicks     uint64 // wavefront clock ticks (PipelineTick calls)
	SubLayerOps       int    // sub-layer labels executed across all ticks
	MaxActiveJobs     int    // peak concurrent in-flight token jobs
	MaxBlockSpread    int    // peak (max block − min block) among active jobs
	MaxDistinctBlocks int    // peak count of distinct blocks occupied at once
	MaxPendingTokens  int    // peak tokens waiting to enter block 0
	StallFallback     bool   // true if the pass fell back to fused normal forward

	// TokenDoneTick[i] = pipeline tick when prompt token i (batch-relative) finished all blocks.
	// Unset entries are -1. Length equals batch size on a full successful prefill.
	TokenDoneTick []int
}

// TokenTimelineSummary is a readable view of per-token completion skew.
type TokenTimelineSummary struct {
	NumTokens       int
	FirstDoneTick   int // token index 0
	LastDoneTick    int // token index NumTokens-1
	TickSpread      int // LastDoneTick - FirstDoneTick
	SampleIndices   []int
	SampleDoneTicks []int
}

func (p PipelineForwardStats) SummarizeTokenTimeline() TokenTimelineSummary {
	out := TokenTimelineSummary{}
	if len(p.TokenDoneTick) == 0 {
		return out
	}
	out.NumTokens = len(p.TokenDoneTick)
	out.FirstDoneTick = p.TokenDoneTick[0]
	out.LastDoneTick = p.TokenDoneTick[len(p.TokenDoneTick)-1]
	if out.FirstDoneTick >= 0 && out.LastDoneTick >= 0 {
		out.TickSpread = out.LastDoneTick - out.FirstDoneTick
	}
	// Quartile sample indices
	if out.NumTokens > 0 {
		for _, q := range []float64{0, 0.25, 0.5, 0.75, 1.0} {
			idx := int(float64(out.NumTokens-1) * q)
			if idx < 0 {
				idx = 0
			}
			if idx >= out.NumTokens {
				idx = out.NumTokens - 1
			}
			out.SampleIndices = append(out.SampleIndices, idx)
			tick := -1
			if idx < len(p.TokenDoneTick) {
				tick = p.TokenDoneTick[idx]
			}
			out.SampleDoneTicks = append(out.SampleDoneTicks, tick)
		}
	}
	return out
}

func (s TokenTimelineSummary) FormatComparison(normalWallSec float64, pipelineTicks uint64) string {
	if s.NumTokens == 0 {
		return "  (no per-token timeline recorded)\n"
	}
	msPerTick := 0.0
	if pipelineTicks > 0 && normalWallSec > 0 {
		// Rough scale: map pipeline tick spacing to normal wall time for intuition.
		_ = normalWallSec
	}
	var b string
	b += fmt.Sprintf("  Normal: all %d prompt tokens move through each block together (one batched MHA per block).\n", s.NumTokens)
	b += fmt.Sprintf("          They effectively finish the stack at the same time (~%.2fs wall).\n", normalWallSec)
	b += fmt.Sprintf("  Pipeline: each token is its own job; finishes the stack at different ticks.\n")
	if s.FirstDoneTick >= 0 && s.LastDoneTick >= 0 {
		b += fmt.Sprintf("          token[0] done @ tick %d  |  token[%d] done @ tick %d\n",
			s.FirstDoneTick, s.NumTokens-1, s.LastDoneTick)
		b += fmt.Sprintf("          stagger: %d ticks between first and last token finishing\n", s.TickSpread)
	}
	b += "          sample (token index → done tick):\n            "
	for i, idx := range s.SampleIndices {
		if i > 0 {
			b += "  "
		}
		tick := -1
		if i < len(s.SampleDoneTicks) {
			tick = s.SampleDoneTicks[i]
		}
		b += fmt.Sprintf("[%d]→%d", idx, tick)
	}
	b += "\n"
	if pipelineTicks > 0 && s.TickSpread > 0 {
		frac := float64(s.TickSpread) / float64(pipelineTicks)
		b += fmt.Sprintf("          first→last token spread is %.0f%% of total prefill pipeline ticks\n", frac*100)
	}
	if msPerTick > 0 {
		_ = msPerTick
	}
	return b
}

func cloneTokenDoneTick(src []int) []int {
	if len(src) == 0 {
		return nil
	}
	dst := make([]int, len(src))
	copy(dst, src)
	return dst
}

func (t *Transformer[T]) resetPipelineForwardStats() {
	t.pipeStatsCur = PipelineForwardStats{}
}

func (t *Transformer[T]) pipeMarkTokenDone(absPos int) {
	if t == nil || t.pipe == nil {
		return
	}
	p := t.pipe
	rel := absPos - p.batchStartPos
	if rel < 0 {
		return
	}
	st := &t.pipeStatsCur
	for len(st.TokenDoneTick) <= rel {
		st.TokenDoneTick = append(st.TokenDoneTick, -1)
	}
	if st.TokenDoneTick[rel] < 0 {
		st.TokenDoneTick[rel] = int(p.tick)
	}
}

func (t *Transformer[T]) pipeObserveWavefront() {
	if t == nil || t.pipe == nil {
		return
	}
	st := &t.pipeStatsCur
	p := t.pipe
	if len(p.pending) > st.MaxPendingTokens {
		st.MaxPendingTokens = len(p.pending)
	}
	active := 0
	minB, maxB := -1, -1
	blocks := make(map[int]struct{})
	for _, j := range p.active {
		if j == nil || j.done {
			continue
		}
		active++
		blocks[j.block] = struct{}{}
		if minB < 0 || j.block < minB {
			minB = j.block
		}
		if j.block > maxB {
			maxB = j.block
		}
	}
	if active > st.MaxActiveJobs {
		st.MaxActiveJobs = active
	}
	if len(blocks) > st.MaxDistinctBlocks {
		st.MaxDistinctBlocks = len(blocks)
	}
	if active > 1 && maxB >= minB {
		spread := maxB - minB
		if spread > st.MaxBlockSpread {
			st.MaxBlockSpread = spread
		}
	}
}

func (t *Transformer[T]) pipeRecordWavefrontSnapshot(opsThisTick int) {
	if t == nil || t.pipe == nil {
		return
	}
	st := &t.pipeStatsCur
	st.PipelineTicks = t.pipe.tick
	st.SubLayerOps += opsThisTick
	t.pipeObserveWavefront()
}

// TakePipelineForwardStats returns stats from the most recent pipeline forward and clears the accumulator.
func (t *Transformer[T]) TakePipelineForwardStats() PipelineForwardStats {
	if t == nil {
		return PipelineForwardStats{}
	}
	out := t.lastPipelineStats
	out.TokenDoneTick = cloneTokenDoneTick(out.TokenDoneTick)
	t.lastPipelineStats = PipelineForwardStats{}
	return out
}

func (t *Transformer[T]) finalizePipelineForwardStats() {
	if t == nil {
		return
	}
	t.lastPipelineStats = t.pipeStatsCur
	t.lastPipelineStats.TokenDoneTick = cloneTokenDoneTick(t.pipeStatsCur.TokenDoneTick)
}

// LastPipelineForwardStats returns stats without clearing.
func (t *Transformer[T]) LastPipelineForwardStats() PipelineForwardStats {
	if t == nil {
		return PipelineForwardStats{}
	}
	return t.lastPipelineStats
}
