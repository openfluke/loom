package accel

// ExecTarget selects where a layer runs during forward inference.
type ExecTarget int

const (
	ExecLoomCPU ExecTarget = iota
	ExecIntelCPU
	ExecIntelNPU
)

func (t ExecTarget) UseAccel() bool {
	return t == ExecIntelCPU || t == ExecIntelNPU
}

func (t ExecTarget) String() string {
	switch t {
	case ExecIntelCPU:
		return "Intel-CPU"
	case ExecIntelNPU:
		return "Intel-NPU"
	default:
		return "Loom-CPU"
	}
}
