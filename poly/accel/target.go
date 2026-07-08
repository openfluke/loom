package accel

// ExecTarget selects where a layer runs during forward inference.
type ExecTarget int

const (
	ExecLoomCPU ExecTarget = iota
	ExecIntelCPU
	ExecIntelNPU
	ExecQualcommCPU
	ExecQualcommNPU
	ExecAppleCPU
	ExecAppleGPU
)

func (t ExecTarget) UseAccel() bool {
	return t == ExecIntelCPU || t == ExecIntelNPU ||
		t == ExecQualcommCPU || t == ExecQualcommNPU ||
		t == ExecAppleCPU || t == ExecAppleGPU
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
	case ExecAppleCPU:
		return "Apple-CPU"
	case ExecAppleGPU:
		return "Apple-GPU"
	default:
		return "Loom-CPU"
	}
}
