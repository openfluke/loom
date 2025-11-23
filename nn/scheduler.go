package nn

import (
	"fmt"
	"math"
)

// LRScheduler interface defines learning rate scheduling strategies
type LRScheduler interface {
	// GetLR returns the learning rate for the given step
	GetLR(step int) float32

	// Reset resets the scheduler state
	Reset()

	// Name returns the scheduler name
	Name() string
}

// ============================================================================
// Constant Scheduler - Fixed learning rate
// ============================================================================

type ConstantScheduler struct {
	baseLR float32
}

func NewConstantScheduler(baseLR float32) *ConstantScheduler {
	return &ConstantScheduler{baseLR: baseLR}
}

func (s *ConstantScheduler) GetLR(step int) float32 {
	return s.baseLR
}

func (s *ConstantScheduler) Reset() {
	// No state to reset
}

func (s *ConstantScheduler) Name() string {
	return "Constant"
}

// ============================================================================
// Linear Decay Scheduler - Linear decay from initial to final LR
// ============================================================================

type LinearDecayScheduler struct {
	initialLR  float32
	finalLR    float32
	totalSteps int
}

func NewLinearDecayScheduler(initialLR, finalLR float32, totalSteps int) *LinearDecayScheduler {
	return &LinearDecayScheduler{
		initialLR:  initialLR,
		finalLR:    finalLR,
		totalSteps: totalSteps,
	}
}

func (s *LinearDecayScheduler) GetLR(step int) float32 {
	if step >= s.totalSteps {
		return s.finalLR
	}

	// Linear interpolation: lr = initialLR + (finalLR - initialLR) * (step / totalSteps)
	progress := float32(step) / float32(s.totalSteps)
	return s.initialLR + (s.finalLR-s.initialLR)*progress
}

func (s *LinearDecayScheduler) Reset() {
	// No state to reset
}

func (s *LinearDecayScheduler) Name() string {
	return "LinearDecay"
}

// ============================================================================
// Cosine Annealing Scheduler - Cosine decay with optional warm restarts
// ============================================================================

type CosineAnnealingScheduler struct {
	initialLR     float32
	minLR         float32
	totalSteps    int
	warmRestarts  bool
	restartPeriod int
}

func NewCosineAnnealingScheduler(initialLR, minLR float32, totalSteps int) *CosineAnnealingScheduler {
	return &CosineAnnealingScheduler{
		initialLR:     initialLR,
		minLR:         minLR,
		totalSteps:    totalSteps,
		warmRestarts:  false,
		restartPeriod: 0,
	}
}

func NewCosineAnnealingWithWarmRestarts(initialLR, minLR float32, restartPeriod int) *CosineAnnealingScheduler {
	return &CosineAnnealingScheduler{
		initialLR:     initialLR,
		minLR:         minLR,
		totalSteps:    0,
		warmRestarts:  true,
		restartPeriod: restartPeriod,
	}
}

func (s *CosineAnnealingScheduler) GetLR(step int) float32 {
	var progress float32

	if s.warmRestarts {
		// Warm restarts: reset cosine curve every restartPeriod steps
		cycleStep := step % s.restartPeriod
		progress = float32(cycleStep) / float32(s.restartPeriod)
	} else {
		// Standard cosine annealing
		if step >= s.totalSteps {
			return s.minLR
		}
		progress = float32(step) / float32(s.totalSteps)
	}

	// Cosine annealing: lr = minLR + (initialLR - minLR) * (1 + cos(π * progress)) / 2
	cosineDecay := (1.0 + float32(math.Cos(math.Pi*float64(progress)))) / 2.0
	return s.minLR + (s.initialLR-s.minLR)*cosineDecay
}

func (s *CosineAnnealingScheduler) Reset() {
	// No state to reset
}

func (s *CosineAnnealingScheduler) Name() string {
	if s.warmRestarts {
		return "CosineAnnealingWarmRestarts"
	}
	return "CosineAnnealing"
}

// ============================================================================
// Exponential Decay Scheduler - Exponential decay
// ============================================================================

type ExponentialDecayScheduler struct {
	initialLR  float32
	decayRate  float32
	decaySteps int
}

func NewExponentialDecayScheduler(initialLR, decayRate float32, decaySteps int) *ExponentialDecayScheduler {
	return &ExponentialDecayScheduler{
		initialLR:  initialLR,
		decayRate:  decayRate,
		decaySteps: decaySteps,
	}
}

func (s *ExponentialDecayScheduler) GetLR(step int) float32 {
	// Exponential decay: lr = initialLR * decayRate^(step / decaySteps)
	exponent := float64(step) / float64(s.decaySteps)
	return s.initialLR * float32(math.Pow(float64(s.decayRate), exponent))
}

func (s *ExponentialDecayScheduler) Reset() {
	// No state to reset
}

func (s *ExponentialDecayScheduler) Name() string {
	return "ExponentialDecay"
}

// ============================================================================
// Warmup Scheduler - Linear warmup followed by another scheduler
// ============================================================================

type WarmupScheduler struct {
	warmupSteps    int
	warmupLR       float32
	baseLR         float32
	afterScheduler LRScheduler
}

func NewWarmupScheduler(warmupSteps int, warmupLR, baseLR float32, afterScheduler LRScheduler) *WarmupScheduler {
	return &WarmupScheduler{
		warmupSteps:    warmupSteps,
		warmupLR:       warmupLR,
		baseLR:         baseLR,
		afterScheduler: afterScheduler,
	}
}

func (s *WarmupScheduler) GetLR(step int) float32 {
	if step < s.warmupSteps {
		// Linear warmup: lr = warmupLR + (baseLR - warmupLR) * (step / warmupSteps)
		progress := float32(step) / float32(s.warmupSteps)
		return s.warmupLR + (s.baseLR-s.warmupLR)*progress
	}

	// After warmup, use the base scheduler (adjusted for warmup steps)
	if s.afterScheduler != nil {
		return s.afterScheduler.GetLR(step - s.warmupSteps)
	}

	return s.baseLR
}

func (s *WarmupScheduler) Reset() {
	if s.afterScheduler != nil {
		s.afterScheduler.Reset()
	}
}

func (s *WarmupScheduler) Name() string {
	if s.afterScheduler != nil {
		return fmt.Sprintf("Warmup+%s", s.afterScheduler.Name())
	}
	return "Warmup"
}

// ============================================================================
// Step Decay Scheduler - Step-wise decay at specific milestones
// ============================================================================

type StepDecayScheduler struct {
	initialLR   float32
	decayFactor float32
	stepSize    int
}

func NewStepDecayScheduler(initialLR, decayFactor float32, stepSize int) *StepDecayScheduler {
	return &StepDecayScheduler{
		initialLR:   initialLR,
		decayFactor: decayFactor,
		stepSize:    stepSize,
	}
}

func (s *StepDecayScheduler) GetLR(step int) float32 {
	// Step decay: lr = initialLR * decayFactor^(step / stepSize)
	numDecays := step / s.stepSize
	return s.initialLR * float32(math.Pow(float64(s.decayFactor), float64(numDecays)))
}

func (s *StepDecayScheduler) Reset() {
	// No state to reset
}

func (s *StepDecayScheduler) Name() string {
	return "StepDecay"
}

// ============================================================================
// Polynomial Decay Scheduler - Polynomial decay
// ============================================================================

type PolynomialDecayScheduler struct {
	initialLR  float32
	finalLR    float32
	totalSteps int
	power      float32
}

func NewPolynomialDecayScheduler(initialLR, finalLR float32, totalSteps int, power float32) *PolynomialDecayScheduler {
	return &PolynomialDecayScheduler{
		initialLR:  initialLR,
		finalLR:    finalLR,
		totalSteps: totalSteps,
		power:      power,
	}
}

func (s *PolynomialDecayScheduler) GetLR(step int) float32 {
	if step >= s.totalSteps {
		return s.finalLR
	}

	// Polynomial decay: lr = (initialLR - finalLR) * (1 - step/totalSteps)^power + finalLR
	progress := float32(step) / float32(s.totalSteps)
	decay := float32(math.Pow(float64(1.0-progress), float64(s.power)))
	return (s.initialLR-s.finalLR)*decay + s.finalLR
}

func (s *PolynomialDecayScheduler) Reset() {
	// No state to reset
}

func (s *PolynomialDecayScheduler) Name() string {
	return "PolynomialDecay"
}
