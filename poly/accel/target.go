package accel

// ExecTarget selects where a layer runs during forward inference.
type ExecTarget int

const (
	ExecLoomCPU ExecTarget = iota
	ExecIntelCPU
	ExecIntelNPU
	ExecQualcommCPU
	ExecQualcommNPU
)

func (t ExecTarget) UseAccel() bool {
	return t == ExecIntelCPU || t == ExecIntelNPU ||
		t == ExecQualcommCPU || t == ExecQualcommNPU
}

func (t ExecTarget) String() string {
	switch t {
	case ExecIntelCPU:
		return "Intel-CPU"
	case ExecIntelNPU:
		return "Intel-NPU"
	case ExecQualcommCPU:
		return "Qualcomm-CPU"
	case ExecQualcommNPU:
		return "Qualcomm-NPU"
	default:
		return "Loom-CPU"
	}
}
