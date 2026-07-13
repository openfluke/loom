package adaptationsuite

import (
	"time"

	"github.com/openfluke/loom/poly"
)

// UpdateMode is the training/update strategy under test.
type UpdateMode int

const (
	UpdateNormalBP UpdateMode = iota
	UpdateStepBP
	UpdateTween
	UpdateTweenChain
	UpdateStepTween
	UpdateStepTweenChain
)

var AllUpdateModes = []UpdateMode{
	UpdateNormalBP,
	UpdateStepBP,
	UpdateTween,
	UpdateTweenChain,
	UpdateStepTween,
	UpdateStepTweenChain,
}

func (m UpdateMode) String() string {
	switch m {
	case UpdateNormalBP:
		return "NormalBP"
	case UpdateStepBP:
		return "Step+BP"
	case UpdateTween:
		return "Tween"
	case UpdateTweenChain:
		return "TweenChain"
	case UpdateStepTween:
		return "StepTween"
	case UpdateStepTweenChain:
		return "StepTweenChain"
	default:
		return "?"
	}
}

func (m UpdateMode) UsesTween() bool {
	switch m {
	case UpdateTween, UpdateTweenChain, UpdateStepTween, UpdateStepTweenChain:
		return true
	default:
		return false
	}
}

func (m UpdateMode) UsesTweenChain() bool {
	switch m {
	case UpdateTweenChain, UpdateStepTweenChain:
		return true
	default:
		return false
	}
}

func (m UpdateMode) PerStep() bool {
	switch m {
	case UpdateStepBP, UpdateStepTween, UpdateStepTweenChain:
		return true
	default:
		return false
	}
}

// Paradigm is QAT (tiled FP32) vs native exact storage.
type Paradigm int

const (
	ParadigmQAT Paradigm = iota
	ParadigmNative
)

func (p Paradigm) String() string {
	if p == ParadigmNative {
		return "Nat"
	}
	return "QAT"
}

// Path is one full adaptation run configuration.
type Path struct {
	Paradigm Paradigm
	SIMD     bool
	Mode     UpdateMode
}

func (p Path) Label() string {
	simd := "SC"
	if p.SIMD {
		simd = "SIMD"
	}
	return p.Paradigm.String() + "/" + simd + "/" + p.Mode.String()
}

// AllPaths returns every QAT/Nat × SC/SIMD × update-mode combination.
// Native paths are omitted when nativeExact is false.
func AllPaths(nativeExact bool) []Path {
	var out []Path
	paradigms := []Paradigm{ParadigmQAT}
	if nativeExact {
		paradigms = append(paradigms, ParadigmNative)
	}
	for _, par := range paradigms {
		for _, simd := range []bool{false, true} {
			for _, mode := range AllUpdateModes {
				out = append(out, Path{Paradigm: par, SIMD: simd, Mode: mode})
			}
		}
	}
	return out
}

// Config controls one adaptation benchmark run.
type Config struct {
	Steps          int
	Windows        int
	LearningRate   float32
	TrainInterval  time.Duration
	MaxBatch       int
	RecoveryThresh float64
	Deadline       time.Duration
	Seed           uint64
}

func DefaultConfig() Config {
	return Config{
		Steps:          450,
		Windows:        9,
		LearningRate:   0.02,
		TrainInterval:  50 * time.Millisecond,
		MaxBatch:       20,
		RecoveryThresh: 50.0,
		Deadline:       10 * time.Millisecond,
		Seed:           0xAD171700,
	}
}

// Scenario supplies inputs and phase-dependent targets for a layer stack.
type Scenario struct {
	Primary     poly.LayerType
	MakeInput   func() *poly.Tensor[float32]
	PhaseTarget func(phase int, outShape []int) *poly.Tensor[float32]
}
