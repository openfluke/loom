package nn

import (
	"fmt"
	"time"
)

// TrainingConfig holds configuration for high-level training with stepping
type SteppingTrainingConfig struct {
	// Optimizer settings
	Optimizer    string // "sgd", "adamw", "rmsprop", "sgd_momentum"
	LearningRate float32
	Beta1        float32 // AdamW/Adam (default: 0.9)
	Beta2        float32 // AdamW/Adam (default: 0.999)
	WeightDecay  float32 // AdamW (default: 0.01)
	Momentum     float32 // SGD/RMSprop (default: 0.9)
	Dampening    float32 // SGD momentum (default: 0.0)
	Nesterov     bool    // SGD Nesterov momentum (default: false)
	Alpha        float32 // RMSprop decay rate (default: 0.99)
	Epsilon      float32 // AdamW/RMSprop epsilon (default: 1e-8)

	// Scheduler settings
	LRSchedule  string  // "constant", "linear", "cosine", "exponential", "warmup", "step", "polynomial"
	WarmupSteps int     // Number of warmup steps (for warmup scheduler)
	TotalSteps  int     // Total training steps (for decay schedulers)
	MinLR       float32 // Minimum learning rate (for cosine/polynomial)
	DecayRate   float32 // Decay rate (for exponential/step schedulers)
	DecaySteps  int     // Decay steps (for exponential scheduler)
	StepSize    int     // Step size (for step scheduler)
	Power       float32 // Power (for polynomial scheduler, default: 1.0)

	// Training settings
	GradAccumSteps  int     // Number of steps to accumulate gradients (default: 1 = no accumulation)
	GradClipValue   float32 // Gradient clipping value (0 = no clipping)
	CheckpointEvery int     // Save checkpoint every N steps (0 = no checkpointing)
	LogEvery        int     // Log progress every N steps (default: 100)

	// Callbacks
	OnStep       func(step int, lr float32, loss float32) // Called after each step
	OnCheckpoint func(step int, network *Network)         // Called at checkpoints
}

// TrainingResult holds the results of training
type SteppingTrainingResult struct {
	TotalSteps     int
	FinalLoss      float32
	LossHistory    []float32
	LRHistory      []float32
	TotalTime      time.Duration
	StepsPerSecond float32
}

// DataLoader is a function that returns input and target for a given step
// Returns nil to signal end of data
type DataLoader func(step int) (input, target []float32, ok bool)

// TrainWithStepping provides a high-level training loop with stepping
// This integrates optimizers, schedulers, gradient accumulation, and checkpointing
func (n *Network) TrainWithStepping(
	config *SteppingTrainingConfig,
	dataLoader DataLoader,
	totalSteps int,
) (*SteppingTrainingResult, error) {

	// Set defaults
	if config.LogEvery == 0 {
		config.LogEvery = 100
	}
	if config.GradAccumSteps == 0 {
		config.GradAccumSteps = 1
	}

	// Initialize optimizer
	if err := n.initializeOptimizer(config); err != nil {
		return nil, err
	}

	// Initialize scheduler
	scheduler, err := n.initializeScheduler(config)
	if err != nil {
		return nil, err
	}

	// Initialize stepping state (assuming first data point for size)
	input, _, ok := dataLoader(0)
	if !ok {
		return nil, fmt.Errorf("dataLoader returned no data for step 0")
	}

	state := n.InitStepState(len(input))

	// Training loop
	result := &SteppingTrainingResult{
		TotalSteps:  totalSteps,
		LossHistory: make([]float32, 0, totalSteps/config.LogEvery),
		LRHistory:   make([]float32, 0, totalSteps/config.LogEvery),
	}

	startTime := time.Now()
	var accumulatedLoss float32
	gradAccumCount := 0

	for step := 0; step < totalSteps; step++ {
		// Get data
		input, target, ok := dataLoader(step)
		if !ok {
			break // End of data
		}

		// Get learning rate from scheduler
		lr := config.LearningRate
		if scheduler != nil {
			lr = scheduler.GetLR(step)
		}

		// Forward pass
		state.SetInput(input)
		n.StepForward(state)
		output := state.GetOutput()

		// Compute loss (MSE for now)
		loss := n.computeLoss(output, target)
		accumulatedLoss += loss

		// Compute gradients
		gradients := make([]float32, len(output))
		for i := range gradients {
			gradients[i] = 2.0 * (output[i] - target[i]) / float32(len(output))
		}

		// Backward pass
		n.StepBackward(state, gradients)

		gradAccumCount++

		// Apply gradients after accumulation steps
		if gradAccumCount >= config.GradAccumSteps {
			// Apply gradient clipping if configured
			if config.GradClipValue > 0 {
				n.clipGradients(config.GradClipValue)
			}

			// Update weights
			n.ApplyGradients(lr)

			// Reset accumulation
			gradAccumCount = 0
		}

		// Logging
		if (step+1)%config.LogEvery == 0 {
			avgLoss := accumulatedLoss / float32(config.LogEvery)
			result.LossHistory = append(result.LossHistory, avgLoss)
			result.LRHistory = append(result.LRHistory, lr)

			if config.OnStep != nil {
				config.OnStep(step+1, lr, avgLoss)
			}

			accumulatedLoss = 0
		}

		// Checkpointing
		if config.CheckpointEvery > 0 && (step+1)%config.CheckpointEvery == 0 {
			if config.OnCheckpoint != nil {
				config.OnCheckpoint(step+1, n)
			}
		}
	}

	// Final metrics
	result.TotalTime = time.Since(startTime)
	result.StepsPerSecond = float32(totalSteps) / float32(result.TotalTime.Seconds())
	if len(result.LossHistory) > 0 {
		result.FinalLoss = result.LossHistory[len(result.LossHistory)-1]
	}

	return result, nil
}

// initializeOptimizer sets up the optimizer based on config
func (n *Network) initializeOptimizer(config *SteppingTrainingConfig) error {
	switch config.Optimizer {
	case "sgd", "":
		// Simple SGD (default)
		n.SetOptimizer(NewSGDOptimizer())

	case "sgd_momentum":
		// SGD with momentum
		momentum := config.Momentum
		if momentum == 0 {
			momentum = 0.9
		}
		n.SetOptimizer(NewSGDOptimizerWithMomentum(momentum, config.Dampening, config.Nesterov))

	case "adamw":
		// AdamW optimizer
		beta1 := config.Beta1
		if beta1 == 0 {
			beta1 = 0.9
		}
		beta2 := config.Beta2
		if beta2 == 0 {
			beta2 = 0.999
		}
		epsilon := config.Epsilon
		if epsilon == 0 {
			epsilon = 1e-8
		}
		weightDecay := config.WeightDecay
		if weightDecay == 0 {
			weightDecay = 0.01
		}
		n.SetOptimizer(NewAdamWOptimizer(beta1, beta2, epsilon, weightDecay))

	case "rmsprop":
		// RMSprop optimizer
		alpha := config.Alpha
		if alpha == 0 {
			alpha = 0.99
		}
		epsilon := config.Epsilon
		if epsilon == 0 {
			epsilon = 1e-8
		}
		n.SetOptimizer(NewRMSpropOptimizer(alpha, epsilon, config.Momentum))

	default:
		return fmt.Errorf("unknown optimizer: %s", config.Optimizer)
	}

	return nil
}

// initializeScheduler sets up the learning rate scheduler based on config
func (n *Network) initializeScheduler(config *SteppingTrainingConfig) (LRScheduler, error) {
	switch config.LRSchedule {
	case "constant", "":
		// Constant learning rate (default)
		return NewConstantScheduler(config.LearningRate), nil

	case "linear":
		// Linear decay
		minLR := config.MinLR
		if minLR == 0 {
			minLR = config.LearningRate * 0.01 // Default to 1% of initial LR
		}
		return NewLinearDecayScheduler(config.LearningRate, minLR, config.TotalSteps), nil

	case "cosine":
		// Cosine annealing
		minLR := config.MinLR
		if minLR == 0 {
			minLR = 0
		}
		return NewCosineAnnealingScheduler(config.LearningRate, minLR, config.TotalSteps), nil

	case "exponential":
		// Exponential decay
		decayRate := config.DecayRate
		if decayRate == 0 {
			decayRate = 0.96 // Default decay rate
		}
		decaySteps := config.DecaySteps
		if decaySteps == 0 {
			decaySteps = 1000 // Default decay steps
		}
		return NewExponentialDecayScheduler(config.LearningRate, decayRate, decaySteps), nil

	case "step":
		// Step decay
		decayRate := config.DecayRate
		if decayRate == 0 {
			decayRate = 0.1 // Default decay factor
		}
		stepSize := config.StepSize
		if stepSize == 0 {
			stepSize = 10000 // Default step size
		}
		return NewStepDecayScheduler(config.LearningRate, decayRate, stepSize), nil

	case "polynomial":
		// Polynomial decay
		minLR := config.MinLR
		if minLR == 0 {
			minLR = 0
		}
		power := config.Power
		if power == 0 {
			power = 1.0 // Linear decay
		}
		return NewPolynomialDecayScheduler(config.LearningRate, minLR, config.TotalSteps, power), nil

	case "warmup":
		// Warmup + constant
		warmupLR := config.MinLR
		if warmupLR == 0 {
			warmupLR = 0
		}
		afterScheduler := NewConstantScheduler(config.LearningRate)
		return NewWarmupScheduler(config.WarmupSteps, warmupLR, config.LearningRate, afterScheduler), nil

	default:
		return nil, fmt.Errorf("unknown scheduler: %s", config.LRSchedule)
	}
}

// computeLoss computes MSE loss
func (n *Network) computeLoss(output, target []float32) float32 {
	var loss float32
	for i := range output {
		diff := output[i] - target[i]
		loss += diff * diff
	}
	return loss / float32(len(output))
}
