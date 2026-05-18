package poly

import "fmt"

// TransformerForwardPipelineCPU runs a wavefront pipeline: multiple tokens can sit at
// different decoder blocks; each PipelineTick advances every ready job by one sub-layer.
// MHA at a block runs in increasing position order so per-layer KV caches stay valid.
const TransformerForwardPipelineCPU TransformerForwardMode = 4

type pipeJob[T Numeric] struct {
	hidden  *Tensor[T]
	pos     int
	block   int
	phase   int // 0..5 in-block; 6 = final RMSNorm
	resAttn *Tensor[T]
	resMLP  *Tensor[T]
	done    bool
}

type decoderPipelineState[T Numeric] struct {
	numBlocks int
	hasFinal  bool

	pending       []*Tensor[T]
	active        []*pipeJob[T]
	nextInjectPos int
	batchStartPos int // KV sequence index of first token in this forward batch
	batchRows     int // how many tokens in this batch

	mhaDone   [][]bool
	blockDone [][]bool

	completed []*Tensor[T] // indexed by absolute sequence position
	lastRow   *Tensor[T]

	tick uint64
}

// pipeKVPos returns the KV write cursor at block 0 MHA (all layers should match after balanced passes).
func (t *Transformer[T]) pipeKVPos() int {
	if len(t.Network.Layers) < 2 {
		return 0
	}
	return t.Network.Layers[1].KVOffset
}

func (t *Transformer[T]) ensurePipelineState() {
	nb := len(t.Network.Layers) / 4
	if t.pipe != nil && t.pipe.numBlocks == nb {
		return
	}
	t.pipe = &decoderPipelineState[T]{
		numBlocks: nb,
		hasFinal:  t.finalNormLayer != nil,
		pending:   make([]*Tensor[T], 0, 8),
		active:    make([]*pipeJob[T], 0, 8),
		completed: make([]*Tensor[T], 0, 8),
	}
}

func (t *Transformer[T]) pipeGrowPos(pos int) {
	p := t.pipe
	need := pos + 1
	for len(p.mhaDone) < t.pipe.numBlocks {
		p.mhaDone = append(p.mhaDone, nil)
		p.blockDone = append(p.blockDone, nil)
	}
	for b := 0; b < t.pipe.numBlocks; b++ {
		for len(p.mhaDone[b]) < need {
			p.mhaDone[b] = append(p.mhaDone[b], false)
			p.blockDone[b] = append(p.blockDone[b], false)
		}
	}
}

func (t *Transformer[T]) pipeCanInject() bool {
	p := t.pipe
	pos := p.nextInjectPos
	if len(p.pending) == 0 {
		return false
	}
	for _, j := range p.active {
		if j.pos == pos && !j.done {
			return false
		}
	}
	// Continuation after prefill / prior decode: block-0 KV cursor is already at pos.
	if pos == p.batchStartPos && pos > 0 {
		l0mha := &t.Network.Layers[1]
		if l0mha.KVOffset == pos {
			return true
		}
	}
	if pos == 0 {
		return true
	}
	t.pipeGrowPos(pos - 1)
	return p.blockDone[0][pos-1]
}

func (t *Transformer[T]) pipeTryInject() bool {
	if !t.pipeCanInject() {
		return false
	}
	p := t.pipe
	pos := p.nextInjectPos
	row := p.pending[0]
	p.pending = p.pending[1:]
	t.pipeGrowPos(pos)
	p.active = append(p.active, &pipeJob[T]{
		hidden: row,
		pos:    pos,
		block:  0,
		phase:  0,
	})
	p.nextInjectPos++
	return true
}

func (t *Transformer[T]) pipeCanAdvance(j *pipeJob[T]) bool {
	if j.done {
		return false
	}
	p := t.pipe
	b, pos, ph := j.block, j.pos, j.phase
	if ph == 1 {
		l1 := &t.Network.Layers[b*4+1]
		if l1.KVOffset != pos {
			return false
		}
		if pos > 0 {
			t.pipeGrowPos(pos - 1)
			if !p.mhaDone[b][pos-1] {
				// Same-batch ordering, or continuation: this block's KV already holds pos.. from prefill.
				if pos != p.batchStartPos || l1.KVOffset != pos {
					return false
				}
			}
		}
	}
	if ph == 0 && b > 0 {
		t.pipeGrowPos(pos)
		if pos >= len(p.blockDone[b-1]) || !p.blockDone[b-1][pos] {
			return false
		}
	}
	return true
}

func (t *Transformer[T]) pipeCompleteJob(j *pipeJob[T]) {
	p := t.pipe
	for len(p.completed) <= j.pos {
		p.completed = append(p.completed, nil)
	}
	p.completed[j.pos] = j.hidden
	p.lastRow = j.hidden
	j.done = true
	t.pipeMarkTokenDone(j.pos)
	t.pipeRemoveJob(j)
}

func (t *Transformer[T]) pipeRemoveJob(j *pipeJob[T]) {
	p := t.pipe
	for i, x := range p.active {
		if x == j {
			p.active = append(p.active[:i], p.active[i+1:]...)
			return
		}
	}
}

func (t *Transformer[T]) pipeAdvanceJob(j *pipeJob[T]) string {
	p := t.pipe
	b, pos := j.block, j.pos
	base := b * 4
	var label string

	switch j.phase {
	case 0:
		j.resAttn = j.hidden.Clone()
		l0 := &t.Network.Layers[base+0]
		_, j.hidden = RMSNormForwardPolymorphic(l0, j.hidden)
		j.phase = 1
		label = fmt.Sprintf("tok %d block %d/%d RMSNorm (pre-attn)", pos+1, b+1, p.numBlocks)
	case 1:
		l1 := &t.Network.Layers[base+1]
		_, j.hidden = MHAForwardPolymorphic(l1, j.hidden)
		t.pipeGrowPos(pos)
		p.mhaDone[b][pos] = true
		j.phase = 2
		label = fmt.Sprintf("tok %d block %d/%d MHA", pos+1, b+1, p.numBlocks)
	case 2:
		j.hidden.Add(j.resAttn)
		j.resAttn = nil
		j.phase = 3
		label = fmt.Sprintf("tok %d block %d/%d residual (attn)", pos+1, b+1, p.numBlocks)
	case 3:
		j.resMLP = j.hidden.Clone()
		l2 := &t.Network.Layers[base+2]
		_, j.hidden = RMSNormForwardPolymorphic(l2, j.hidden)
		j.phase = 4
		label = fmt.Sprintf("tok %d block %d/%d RMSNorm (pre-mlp)", pos+1, b+1, p.numBlocks)
	case 4:
		l3 := &t.Network.Layers[base+3]
		_, j.hidden = SwiGLUForwardPolymorphic(l3, j.hidden)
		j.phase = 5
		label = fmt.Sprintf("tok %d block %d/%d SwiGLU", pos+1, b+1, p.numBlocks)
	case 5:
		j.hidden.Add(j.resMLP)
		j.resMLP = nil
		t.pipeGrowPos(pos)
		p.blockDone[b][pos] = true
		label = fmt.Sprintf("tok %d block %d/%d residual (mlp)", pos+1, b+1, p.numBlocks)
		j.block++
		if j.block < p.numBlocks {
			j.phase = 0
		} else if p.hasFinal {
			j.phase = 6
		} else {
			t.pipeCompleteJob(j)
		}
	case 6:
		_, j.hidden = RMSNormForwardPolymorphic(t.finalNormLayer, j.hidden)
		t.pipeCompleteJob(j)
		label = fmt.Sprintf("tok %d final RMSNorm", pos+1)
	}
	return label
}

// PipelineTick advances the wavefront one clock (all ready jobs move one sub-layer).
func (t *Transformer[T]) PipelineTick() (labels []string, finished bool, err error) {
	if t.ForwardMode != TransformerForwardPipelineCPU {
		return nil, false, fmt.Errorf("PipelineTick: ForwardMode must be TransformerForwardPipelineCPU")
	}
	if t.hostWeightsReleased {
		return nil, false, fmt.Errorf("PipelineTick: host weights released")
	}
	t.ensurePipelineState()
	p := t.pipe

	_ = t.pipeTryInject()
	t.pipeObserveWavefront()

	mhaPicked := make(map[int]*pipeJob[T])
	for _, j := range p.active {
		if j.done || j.phase != 1 || !t.pipeCanAdvance(j) {
			continue
		}
		if cur, ok := mhaPicked[j.block]; !ok || j.pos < cur.pos {
			mhaPicked[j.block] = j
		}
	}

	for _, j := range p.active {
		if j.done {
			continue
		}
		if j.phase == 1 {
			if pick, ok := mhaPicked[j.block]; !ok || pick != j {
				continue
			}
		} else if !t.pipeCanAdvance(j) {
			continue
		}
		label := t.pipeAdvanceJob(j)
		if label != "" {
			labels = append(labels, label)
		}
	}

	_ = t.pipeTryInject()
	t.pipeObserveWavefront()

	p.tick++
	t.pipeRecordWavefrontSnapshot(len(labels))

	batchDone := p.batchRows > 0 && p.nextInjectPos == p.batchStartPos+p.batchRows
	lastCompleted := false
	if batchDone && p.batchRows > 0 {
		lastPos := p.batchStartPos + p.batchRows - 1
		if lastPos < len(p.completed) && p.completed[lastPos] != nil {
			lastCompleted = true
		}
	}
	finished = len(p.pending) == 0 && len(p.active) == 0 && p.lastRow != nil && batchDone && lastCompleted

	if t.PipelineTickPause != nil && len(labels) > 0 {
		summary := fmt.Sprintf("tick %d: %d op(s)", p.tick, len(labels))
		if len(labels) == 1 {
			summary = fmt.Sprintf("tick %d: %s", p.tick, labels[0])
		} else if len(labels) <= 3 {
			summary = fmt.Sprintf("tick %d: %v", p.tick, labels)
		}
		t.PipelineTickPause(int(p.tick), 0, summary)
	}

	return labels, finished, nil
}

// PipelineReset clears macro pipeline queues (call Transformer.Reset to clear KV).
func (t *Transformer[T]) PipelineReset() {
	t.pipe = nil
}

func (t *Transformer[T]) forwardCPUHiddenPipeline(input *Tensor[T]) *Tensor[T] {
	if input == nil || len(input.Data) == 0 {
		return NewTensor[T](1, t.HiddenSize)
	}
	h := t.HiddenSize
	if h <= 0 {
		return input
	}
	rows := len(input.Data) / h
	if rows < 1 {
		return NewTensor[T](1, h)
	}

	t.resetPipelineForwardStats()
	t.ensurePipelineState()
	p := t.pipe
	p.pending = p.pending[:0]
	p.active = p.active[:0]
	p.completed = p.completed[:0]
	p.lastRow = nil
	p.batchStartPos = t.pipeKVPos()
	p.batchRows = rows
	p.nextInjectPos = p.batchStartPos
	p.tick = 0
	p.mhaDone = nil
	p.blockDone = nil

	for i := 0; i < rows; i++ {
		row := NewTensor[T](1, h)
		copy(row.Data, input.Data[i*h:(i+1)*h])
		p.pending = append(p.pending, row)
	}

	stepN := 0
	totalHint := rows * t.cpuForwardStepCount()
	const maxTicks = 1_000_000
	for tick := 0; tick < maxTicks; tick++ {
		labels, done, err := t.PipelineTick()
		if err != nil {
			fmt.Printf("⚠️  %v\n", err)
			t.pipeStatsCur.StallFallback = true
			out := t.forwardOnCPUNormal(input)
			t.finalizePipelineForwardStats()
			return out
		}
		for _, lbl := range labels {
			stepN++
			t.emitCPUForwardStep(stepN, totalHint, lbl)
		}
		if done {
			break
		}
		if len(labels) == 0 {
			fmt.Println("⚠️  pipeline stall; falling back to fused forward for this pass")
			t.pipeStatsCur.StallFallback = true
			out := t.forwardOnCPUNormal(input)
			t.finalizePipelineForwardStats()
			return out
		}
	}

	if rows > 1 {
		out := NewTensor[T](rows, h)
		ok := true
		for i := 0; i < rows; i++ {
			abs := p.batchStartPos + i
			if abs >= len(p.completed) || p.completed[abs] == nil {
				ok = false
				break
			}
			copy(out.Data[i*h:(i+1)*h], p.completed[abs].Data)
		}
		if ok {
			p.lastRow = NewTensor[T](1, h)
			copy(p.lastRow.Data, out.Data[(rows-1)*h:rows*h])
			outFin := out
			t.finalizePipelineForwardStats()
			return outFin
		}
	}

	if p.lastRow != nil {
		out := p.lastRow.Clone()
		t.finalizePipelineForwardStats()
		return out
	}
	t.pipeStatsCur.StallFallback = true
	out := t.forwardOnCPUNormal(input)
	t.finalizePipelineForwardStats()
	return out
}
