package accel

// LayerBinding is a compiled vendor graph (init once, infer many).
type LayerBinding struct {
	Target   ExecTarget
	Desc     LayerDesc
	Compiled CompiledLayer
	CompileMs float64
	FirstInferMs float64
}

// Release frees the compiled handle.
func (b *LayerBinding) Release() {
	if b == nil || b.Compiled == nil {
		return
	}
	b.Compiled.Release()
	b.Compiled = nil
}

// Infer runs one forward through the binding.
func (b *LayerBinding) Infer(in, out []byte) (InferResult, error) {
	if b == nil || b.Compiled == nil {
		return InferResult{}, ErrUnavailable
	}
	return b.Compiled.Infer(in, out)
}

func (b *LayerBinding) InBytes() uintptr {
	if b == nil || b.Compiled == nil {
		return 0
	}
	return b.Compiled.InBytes()
}

func (b *LayerBinding) OutBytes() uintptr {
	if b == nil || b.Compiled == nil {
		return 0
	}
	return b.Compiled.OutBytes()
}
