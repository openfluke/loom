package poly

import "fmt"

// TransformerForwardMode selects how the transformer runs its decoder stack on CPU.
// GPU paths are skipped when forwardModeSkipsGPU() is true (stepped / queued).
type TransformerForwardMode int

const (
	// TransformerForwardNormal runs the fused CPU block loop (default).
	TransformerForwardNormal TransformerForwardMode = iota
	// TransformerForwardSteppedCPU runs the same ops as Normal but one sub-layer at a time
	// (optional logging via ForwardStepDebug + SetForwardStepObserver).
	TransformerForwardSteppedCPU
	// TransformerForwardQueuedCPU keeps partial progress in BeginCPUForwardQueue /
	// CPUForwardQueueTick. ForwardFull and forwardOne drain the queue each call.
	TransformerForwardQueuedCPU
	// TransformerForwardPipelineCPU is defined in transformer_pipeline.go (value 4).
)

// String returns a short label for UIs and logs.
func (m TransformerForwardMode) String() string {
	switch m {
	case TransformerForwardSteppedCPU:
		return "stepped"
	case TransformerForwardQueuedCPU:
		return "queued"
	case TransformerForwardPipelineCPU:
		return "pipeline"
	default:
		return "normal"
	}
}

// cpuForwardQueueState holds incremental CPU forward progress for TransformerForwardQueuedCPU.
type cpuForwardQueueState[T Numeric] struct {
	current   *Tensor[T]
	numBlocks int
	block     int
	phase     int // 0..5 in-block; 6 = final RMSNorm
	resAttn   *Tensor[T]
	resMLP    *Tensor[T]
	done      bool
}

// SetForwardStepObserver sets a callback for stepped or queued CPU forward when
// ForwardStepDebug is true and cb is non-nil.
func (t *Transformer[T]) SetForwardStepObserver(cb func(step, total int, label string)) {
	t.forwardStepCb = cb
}

func (t *Transformer[T]) emitCPUForwardStep(step, total int, label string) {
	if !t.ForwardStepDebug || t.forwardStepCb == nil {
		return
	}
	t.forwardStepCb(step, total, label)
}

func (t *Transformer[T]) cpuForwardStepCount() int {
	numBlocks := len(t.Network.Layers) / 4
	n := numBlocks * 6
	if t.finalNormLayer != nil {
		n++
	}
	return n
}

func (t *Transformer[T]) forwardModeSkipsGPU() bool {
	switch t.ForwardMode {
	case TransformerForwardSteppedCPU, TransformerForwardQueuedCPU, TransformerForwardPipelineCPU:
		return true
	default:
		return false
	}
}

// CPUForwardQueueStepTotal returns ticks needed for one full CPU forward.
func (t *Transformer[T]) CPUForwardQueueStepTotal() int {
	return t.cpuForwardStepCount()
}

// BeginCPUForwardQueue starts an incremental CPU forward (ForwardMode must be QueuedCPU).
func (t *Transformer[T]) BeginCPUForwardQueue(input *Tensor[T]) error {
	if t.ForwardMode != TransformerForwardQueuedCPU {
		return fmt.Errorf("BeginCPUForwardQueue: ForwardMode must be TransformerForwardQueuedCPU")
	}
	if t.hostWeightsReleased {
		return fmt.Errorf("BeginCPUForwardQueue: host weights released")
	}
	if input == nil {
		return fmt.Errorf("BeginCPUForwardQueue: nil input")
	}
	nb := len(t.Network.Layers) / 4
	if nb < 1 {
		return fmt.Errorf("BeginCPUForwardQueue: no decoder blocks (len(Layers)=%d)", len(t.Network.Layers))
	}
	t.cpuFQ = &cpuForwardQueueState[T]{
		current:   input,
		numBlocks: nb,
		block:     0,
		phase:     0,
		done:      false,
	}
	return nil
}

// CPUForwardQueueTick runs exactly one sub-layer step.
func (t *Transformer[T]) CPUForwardQueueTick() (done bool, completedLabel string, err error) {
	q := t.cpuFQ
	if q == nil {
		return false, "", fmt.Errorf("CPUForwardQueueTick: call BeginCPUForwardQueue first")
	}
	if q.done {
		return true, "", nil
	}
	if len(t.Network.Layers)%4 != 0 {
		return true, "", fmt.Errorf("CPUForwardQueueTick: len(Layers) %% 4 != 0")
	}
	base := q.block * 4
	switch q.phase {
	case 0:
		q.resAttn = q.current.Clone()
		l0 := &t.Network.Layers[base+0]
		_, q.current = RMSNormForwardPolymorphic(l0, q.current)
		q.phase = 1
		completedLabel = fmt.Sprintf("block %d/%d RMSNorm (pre-attn)", q.block+1, q.numBlocks)
	case 1:
		l1 := &t.Network.Layers[base+1]
		_, q.current = MHAForwardPolymorphic(l1, q.current)
		q.phase = 2
		completedLabel = fmt.Sprintf("block %d/%d MHA", q.block+1, q.numBlocks)
	case 2:
		q.current.Add(q.resAttn)
		q.resAttn = nil
		q.phase = 3
		completedLabel = fmt.Sprintf("block %d/%d residual (attn)", q.block+1, q.numBlocks)
	case 3:
		q.resMLP = q.current.Clone()
		l2 := &t.Network.Layers[base+2]
		_, q.current = RMSNormForwardPolymorphic(l2, q.current)
		q.phase = 4
		completedLabel = fmt.Sprintf("block %d/%d RMSNorm (pre-mlp)", q.block+1, q.numBlocks)
	case 4:
		l3 := &t.Network.Layers[base+3]
		_, q.current = SwiGLUForwardPolymorphic(l3, q.current)
		q.phase = 5
		completedLabel = fmt.Sprintf("block %d/%d SwiGLU", q.block+1, q.numBlocks)
	case 5:
		q.current.Add(q.resMLP)
		q.resMLP = nil
		completedLabel = fmt.Sprintf("block %d/%d residual (mlp)", q.block+1, q.numBlocks)
		q.block++
		if q.block < q.numBlocks {
			q.phase = 0
		} else if t.finalNormLayer != nil {
			q.phase = 6
		} else {
			q.done = true
			return true, completedLabel, nil
		}
	case 6:
		_, q.current = RMSNormForwardPolymorphic(t.finalNormLayer, q.current)
		completedLabel = "final RMSNorm"
		q.done = true
		return true, completedLabel, nil
	default:
		q.done = true
		return true, "", fmt.Errorf("CPUForwardQueueTick: invalid phase %d", q.phase)
	}
	return q.done, completedLabel, nil
}

// CPUForwardQueueResult returns the hidden tensor after the queue finished.
func (t *Transformer[T]) CPUForwardQueueResult() *Tensor[T] {
	if t.cpuFQ == nil || !t.cpuFQ.done {
		return nil
	}
	return t.cpuFQ.current
}

// CPUForwardQueueDiscard clears queue state.
func (t *Transformer[T]) CPUForwardQueueDiscard() {
	t.cpuFQ = nil
}

func (t *Transformer[T]) cpuForwardQueueDrain(input *Tensor[T]) *Tensor[T] {
	if err := t.BeginCPUForwardQueue(input); err != nil {
		fmt.Printf("⚠️  %v\n", err)
		if input != nil {
			return NewTensor[T](input.Shape...)
		}
		return nil
	}
	total := t.cpuForwardStepCount()
	step := 0
	for {
		done, label, err := t.CPUForwardQueueTick()
		if err != nil {
			fmt.Printf("⚠️  %v\n", err)
			t.cpuFQ = nil
			if input != nil {
				return NewTensor[T](input.Shape...)
			}
			return nil
		}
		step++
		t.emitCPUForwardStep(step, total, label)
		if t.QueueTickPause != nil {
			t.QueueTickPause(step, total, label)
		}
		if done {
			break
		}
	}
	out := t.CPUForwardQueueResult()
	t.cpuFQ = nil
	return out
}

func (t *Transformer[T]) forwardOnCPUNormal(input *Tensor[T]) *Tensor[T] {
	current := input
	numBlocks := len(t.Network.Layers) / 4

	for b := 0; b < numBlocks; b++ {
		base := b * 4

		residual := current.Clone()

		lNorm1 := &t.Network.Layers[base+0]
		_, current = RMSNormForwardPolymorphic(lNorm1, current)

		lMHA := &t.Network.Layers[base+1]
		_, current = MHAForwardPolymorphic(lMHA, current)

		current.Add(residual)

		residual = current.Clone()

		lNorm2 := &t.Network.Layers[base+2]
		_, current = RMSNormForwardPolymorphic(lNorm2, current)

		lMLP := &t.Network.Layers[base+3]
		_, current = SwiGLUForwardPolymorphic(lMLP, current)

		current.Add(residual)
	}

	if t.finalNormLayer != nil {
		_, current = RMSNormForwardPolymorphic(t.finalNormLayer, current)
	}

	return current
}

func (t *Transformer[T]) forwardOnCPUStepped(input *Tensor[T]) *Tensor[T] {
	current := input
	numBlocks := len(t.Network.Layers) / 4
	total := t.cpuForwardStepCount()
	step := 0

	for b := 0; b < numBlocks; b++ {
		base := b * 4

		residual := current.Clone()

		lNorm1 := &t.Network.Layers[base+0]
		_, current = RMSNormForwardPolymorphic(lNorm1, current)
		step++
		t.emitCPUForwardStep(step, total, fmt.Sprintf("block %d/%d RMSNorm (pre-attn)", b+1, numBlocks))

		lMHA := &t.Network.Layers[base+1]
		_, current = MHAForwardPolymorphic(lMHA, current)
		step++
		t.emitCPUForwardStep(step, total, fmt.Sprintf("block %d/%d MHA", b+1, numBlocks))

		current.Add(residual)
		step++
		t.emitCPUForwardStep(step, total, fmt.Sprintf("block %d/%d residual (attn)", b+1, numBlocks))

		residual = current.Clone()

		lNorm2 := &t.Network.Layers[base+2]
		_, current = RMSNormForwardPolymorphic(lNorm2, current)
		step++
		t.emitCPUForwardStep(step, total, fmt.Sprintf("block %d/%d RMSNorm (pre-mlp)", b+1, numBlocks))

		lMLP := &t.Network.Layers[base+3]
		_, current = SwiGLUForwardPolymorphic(lMLP, current)
		step++
		t.emitCPUForwardStep(step, total, fmt.Sprintf("block %d/%d SwiGLU", b+1, numBlocks))

		current.Add(residual)
		step++
		t.emitCPUForwardStep(step, total, fmt.Sprintf("block %d/%d residual (mlp)", b+1, numBlocks))
	}

	if t.finalNormLayer != nil {
		_, current = RMSNormForwardPolymorphic(t.finalNormLayer, current)
		step++
		t.emitCPUForwardStep(step, total, "final RMSNorm")
	}

	return current
}

// forwardCPUHidden runs the decoder on CPU respecting ForwardMode (not layer-trace).
func (t *Transformer[T]) forwardCPUHidden(input *Tensor[T]) *Tensor[T] {
	if t.hostWeightsReleased {
		fmt.Println("⚠️  CPU forward skipped (host weights released after GPU upload).")
		return NewTensor[T](input.Shape...)
	}
	switch t.ForwardMode {
	case TransformerForwardPipelineCPU:
		return t.forwardCPUHiddenPipeline(input)
	case TransformerForwardQueuedCPU:
		return t.cpuForwardQueueDrain(input)
	case TransformerForwardSteppedCPU:
		return t.forwardOnCPUStepped(input)
	default:
		return t.forwardOnCPUNormal(input)
	}
}
