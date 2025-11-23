package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	nn "github.com/openfluke/loom/nn"
)

// ===========================
// TASK: Memory-Augmented Algorithmic Learning
// ===========================
// This is a challenging task where the network must learn to:
// 1. COPY: Read a sequence and reproduce it exactly
// 2. REPEAT COPY: Read a sequence and repeat it N times
// 3. ASSOCIATIVE RECALL: Store key-value pairs and retrieve values given keys
// 4. PRIORITY SORT: Read numbers and output them in sorted order
//
// This requires the network to develop internal memory mechanisms
// using only LSTM states and the stepping architecture.
// ===========================

// TaskType defines different algorithmic tasks
type TaskType int

const (
	TaskCopy TaskType = iota
	TaskRepeatCopy
	TaskAssociativeRecall
	TaskPrioritySort
)

func (t TaskType) String() string {
	switch t {
	case TaskCopy:
		return "Copy"
	case TaskRepeatCopy:
		return "RepeatCopy"
	case TaskAssociativeRecall:
		return "AssocRecall"
	case TaskPrioritySort:
		return "PrioritySort"
	default:
		return "Unknown"
	}
}

// SequenceTask represents a single algorithmic task instance
type SequenceTask struct {
	taskType      TaskType
	inputSeq      [][]float32 // Input sequence (each element is a vector)
	targetSeq     [][]float32 // Expected output sequence
	description   string
	currentInput  int // Current position in input
	currentTarget int // Current position in target
}

// TaskGenerator creates algorithmic tasks
type TaskGenerator struct {
	vectorSize int
	maxSeqLen  int
}

func NewTaskGenerator(vectorSize, maxSeqLen int) *TaskGenerator {
	return &TaskGenerator{
		vectorSize: vectorSize,
		maxSeqLen:  maxSeqLen,
	}
}

// GenerateCopyTask: Input a sequence, then output the same sequence
func (tg *TaskGenerator) GenerateCopyTask() *SequenceTask {
	seqLen := rand.Intn(tg.maxSeqLen-2) + 3 // 3 to maxSeqLen

	inputSeq := make([][]float32, seqLen+2)  // +2 for start/end markers
	targetSeq := make([][]float32, seqLen+1) // +1 for end marker

	// Start marker (first channel = 1.0)
	inputSeq[0] = make([]float32, tg.vectorSize)
	inputSeq[0][0] = 1.0

	// Random sequence
	for i := 1; i <= seqLen; i++ {
		inputSeq[i] = make([]float32, tg.vectorSize)
		for j := 2; j < tg.vectorSize; j++ { // Reserve channels 0,1 for markers
			inputSeq[i][j] = rand.Float32()
		}
	}

	// End marker (second channel = 1.0)
	inputSeq[seqLen+1] = make([]float32, tg.vectorSize)
	inputSeq[seqLen+1][1] = 1.0

	// Target: copy the sequence
	for i := 0; i < seqLen; i++ {
		targetSeq[i] = make([]float32, tg.vectorSize)
		copy(targetSeq[i], inputSeq[i+1])
	}

	// End marker in target
	targetSeq[seqLen] = make([]float32, tg.vectorSize)
	targetSeq[seqLen][1] = 1.0

	return &SequenceTask{
		taskType:    TaskCopy,
		inputSeq:    inputSeq,
		targetSeq:   targetSeq,
		description: fmt.Sprintf("Copy sequence of length %d", seqLen),
	}
}

// GenerateRepeatCopyTask: Input a sequence and a repeat count, output the sequence N times
func (tg *TaskGenerator) GenerateRepeatCopyTask() *SequenceTask {
	seqLen := rand.Intn(tg.maxSeqLen/3-1) + 2 // Shorter sequences for repeating
	repeatCount := rand.Intn(3) + 2           // Repeat 2-4 times

	inputSeq := make([][]float32, seqLen+3) // +3 for start, repeat marker, end
	targetSeq := make([][]float32, seqLen*repeatCount+1)

	// Start marker
	inputSeq[0] = make([]float32, tg.vectorSize)
	inputSeq[0][0] = 1.0

	// Random sequence
	sequence := make([][]float32, seqLen)
	for i := 0; i < seqLen; i++ {
		sequence[i] = make([]float32, tg.vectorSize)
		inputSeq[i+1] = make([]float32, tg.vectorSize)
		for j := 2; j < tg.vectorSize; j++ {
			val := rand.Float32()
			sequence[i][j] = val
			inputSeq[i+1][j] = val
		}
	}

	// Repeat count marker (encode in channel 2)
	inputSeq[seqLen+1] = make([]float32, tg.vectorSize)
	inputSeq[seqLen+1][2] = float32(repeatCount) / 10.0 // Normalize

	// End marker
	inputSeq[seqLen+2] = make([]float32, tg.vectorSize)
	inputSeq[seqLen+2][1] = 1.0

	// Target: repeat the sequence
	for r := 0; r < repeatCount; r++ {
		for i := 0; i < seqLen; i++ {
			targetSeq[r*seqLen+i] = make([]float32, tg.vectorSize)
			copy(targetSeq[r*seqLen+i], sequence[i])
		}
	}

	// End marker
	targetSeq[seqLen*repeatCount] = make([]float32, tg.vectorSize)
	targetSeq[seqLen*repeatCount][1] = 1.0

	return &SequenceTask{
		taskType:    TaskRepeatCopy,
		inputSeq:    inputSeq,
		targetSeq:   targetSeq,
		description: fmt.Sprintf("Repeat copy: %dx%d", seqLen, repeatCount),
	}
}

// GenerateAssociativeRecallTask: Store key-value pairs, then query with a key
func (tg *TaskGenerator) GenerateAssociativeRecallTask() *SequenceTask {
	numPairs := rand.Intn(3) + 2 // 2-4 pairs
	keySize := tg.vectorSize / 2

	inputSeq := make([][]float32, 0)

	// Generate key-value pairs
	keys := make([][]float32, numPairs)
	values := make([][]float32, numPairs)

	for i := 0; i < numPairs; i++ {
		// Start marker for pair
		pairStart := make([]float32, tg.vectorSize)
		pairStart[0] = 1.0
		inputSeq = append(inputSeq, pairStart)

		// Key
		key := make([]float32, tg.vectorSize)
		for j := 2; j < keySize; j++ {
			key[j] = rand.Float32()
		}
		keys[i] = key
		inputSeq = append(inputSeq, key)

		// Value
		value := make([]float32, tg.vectorSize)
		for j := keySize; j < tg.vectorSize; j++ {
			value[j] = rand.Float32()
		}
		values[i] = value
		inputSeq = append(inputSeq, value)
	}

	// Query marker
	queryMarker := make([]float32, tg.vectorSize)
	queryMarker[1] = 1.0
	inputSeq = append(inputSeq, queryMarker)

	// Random query key
	queryIdx := rand.Intn(numPairs)
	queryKey := make([]float32, tg.vectorSize)
	copy(queryKey, keys[queryIdx])
	inputSeq = append(inputSeq, queryKey)

	// End marker
	endMarker := make([]float32, tg.vectorSize)
	endMarker[2] = 1.0
	inputSeq = append(inputSeq, endMarker)

	// Target: the corresponding value
	targetSeq := make([][]float32, 2)
	targetSeq[0] = make([]float32, tg.vectorSize)
	copy(targetSeq[0], values[queryIdx])
	targetSeq[1] = make([]float32, tg.vectorSize)
	targetSeq[1][2] = 1.0 // End marker

	return &SequenceTask{
		taskType:    TaskAssociativeRecall,
		inputSeq:    inputSeq,
		targetSeq:   targetSeq,
		description: fmt.Sprintf("Recall: %d pairs, query #%d", numPairs, queryIdx),
	}
}

// GeneratePrioritySortTask: Input numbers and output them in sorted order
func (tg *TaskGenerator) GeneratePrioritySortTask() *SequenceTask {
	seqLen := rand.Intn(tg.maxSeqLen-2) + 3

	inputSeq := make([][]float32, seqLen+2)

	// Start marker
	inputSeq[0] = make([]float32, tg.vectorSize)
	inputSeq[0][0] = 1.0

	// Random numbers (stored in channel 2)
	numbers := make([]float32, seqLen)
	for i := 0; i < seqLen; i++ {
		numbers[i] = rand.Float32()
		inputSeq[i+1] = make([]float32, tg.vectorSize)
		inputSeq[i+1][2] = numbers[i]
	}

	// End marker
	inputSeq[seqLen+1] = make([]float32, tg.vectorSize)
	inputSeq[seqLen+1][1] = 1.0

	// Sort numbers
	sortedNumbers := make([]float32, seqLen)
	copy(sortedNumbers, numbers)
	for i := 0; i < seqLen; i++ {
		for j := i + 1; j < seqLen; j++ {
			if sortedNumbers[i] > sortedNumbers[j] {
				sortedNumbers[i], sortedNumbers[j] = sortedNumbers[j], sortedNumbers[i]
			}
		}
	}

	// Target: sorted sequence
	targetSeq := make([][]float32, seqLen+1)
	for i := 0; i < seqLen; i++ {
		targetSeq[i] = make([]float32, tg.vectorSize)
		targetSeq[i][2] = sortedNumbers[i]
	}
	targetSeq[seqLen] = make([]float32, tg.vectorSize)
	targetSeq[seqLen][1] = 1.0

	return &SequenceTask{
		taskType:    TaskPrioritySort,
		inputSeq:    inputSeq,
		targetSeq:   targetSeq,
		description: fmt.Sprintf("Sort %d numbers", seqLen),
	}
}

// GenerateRandomTask generates a random task
func (tg *TaskGenerator) GenerateRandomTask() *SequenceTask {
	taskType := TaskType(rand.Intn(4))
	switch taskType {
	case TaskCopy:
		return tg.GenerateCopyTask()
	case TaskRepeatCopy:
		return tg.GenerateRepeatCopyTask()
	case TaskAssociativeRecall:
		return tg.GenerateAssociativeRecallTask()
	case TaskPrioritySort:
		return tg.GeneratePrioritySortTask()
	default:
		return tg.GenerateCopyTask()
	}
}

// TargetQueue handles delayed targets for the stepping network
type TargetQueue struct {
	targets [][]float32
	maxSize int
}

func NewTargetQueue(size int) *TargetQueue {
	return &TargetQueue{
		targets: make([][]float32, 0, size),
		maxSize: size,
	}
}

func (q *TargetQueue) Push(target []float32) {
	q.targets = append(q.targets, target)
}

func (q *TargetQueue) Pop() []float32 {
	if len(q.targets) == 0 {
		return nil
	}
	t := q.targets[0]
	q.targets = q.targets[1:]
	return t
}

func (q *TargetQueue) IsFull() bool {
	return len(q.targets) >= q.maxSize
}

func (q *TargetQueue) Clear() {
	q.targets = q.targets[:0]
}

// Calculate sequence loss (MSE for continuous values)
func calculateSequenceLoss(output, target []float32) (float32, []float32) {
	loss := float32(0.0)
	grad := make([]float32, len(output))

	for i := 0; i < len(output); i++ {
		diff := output[i] - target[i]
		loss += diff * diff
		grad[i] = 2.0 * diff / float32(len(output))
	}

	return loss / float32(len(output)), grad
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  LOOM Stepping Neural Network v8: Algorithmic Learning            ║")
	fmt.Println("║  Memory-Augmented Sequence Tasks                                  ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Println("Network Architecture:")
	fmt.Println("  Input(16) → Dense(64) → LSTM(128) → LSTM(128) → Dense(16)")
	fmt.Println()
	fmt.Println("Tasks:")
	fmt.Println("  1. COPY: Read and reproduce sequences")
	fmt.Println("  2. REPEAT COPY: Read and repeat sequences N times")
	fmt.Println("  3. ASSOCIATIVE RECALL: Store key-value pairs and retrieve")
	fmt.Println("  4. PRIORITY SORT: Sort sequences of numbers")
	fmt.Println()

	// Network configuration
	vectorSize := 16

	networkJSON := `{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 5,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_height": 16,
				"output_height": 64,
				"activation": "relu"
			},
			{
				"type": "lstm",
				"input_size": 64,
				"hidden_size": 128,
				"seq_length": 1,
				"activation": "tanh"
			},
			{
				"type": "lstm",
				"input_size": 128,
				"hidden_size": 128,
				"seq_length": 1,
				"activation": "tanh"
			},
			{
				"type": "dense",
				"input_height": 128,
				"output_height": 64,
				"activation": "relu"
			},
			{
				"type": "dense",
				"input_height": 64,
				"output_height": 16,
				"activation": "tanh"
			}
		]
	}`

	net, err := nn.BuildNetworkFromJSON(networkJSON)
	if err != nil {
		log.Fatalf("Failed to build network: %v", err)
	}
	net.InitializeWeights()

	state := net.InitStepState(vectorSize)

	// Training configuration
	totalEpisodes := 5000
	targetDelay := 5 // 5-layer network
	targetQueue := NewTargetQueue(targetDelay)

	learningRate := float32(0.001)
	minLearningRate := float32(0.0001)
	decayRate := float32(0.9995)
	gradientClipValue := float32(5.0)

	taskGen := NewTaskGenerator(vectorSize, 8)

	fmt.Printf("Training Configuration:\n")
	fmt.Printf("  Episodes: %d\n", totalEpisodes)
	fmt.Printf("  Target delay: %d steps\n", targetDelay)
	fmt.Printf("  Learning rate: %.6f → %.6f\n", learningRate, minLearningRate)
	fmt.Printf("  Gradient clipping: %.2f\n", gradientClipValue)
	fmt.Println()

	// Training statistics
	taskSuccesses := make(map[TaskType]int)
	taskAttempts := make(map[TaskType]int)

	fmt.Println("Training Progress:")
	fmt.Println("─────────────────────────────────────────────────────────────────────")

	startTime := time.Now()

	for episode := 0; episode < totalEpisodes; episode++ {
		// Generate a random task
		task := taskGen.GenerateRandomTask()
		taskAttempts[task.taskType]++

		// Reset target queue for new episode
		targetQueue.Clear()

		// Process the task
		totalLoss := float32(0.0)
		lossCount := 0

		// Phase 1: Input phase - feed the input sequence
		for _, input := range task.inputSeq {
			state.SetInput(input)
			net.StepForward(state)

			// During input phase, target is zero (no output expected)
			zeroTarget := make([]float32, vectorSize)
			targetQueue.Push(zeroTarget)

			if targetQueue.IsFull() {
				delayedTarget := targetQueue.Pop()
				output := state.GetOutput()

				loss, grad := calculateSequenceLoss(output, delayedTarget)

				// Gradient clipping
				gradNorm := float32(0.0)
				for _, g := range grad {
					gradNorm += g * g
				}
				gradNorm = float32(math.Sqrt(float64(gradNorm)))

				if gradNorm > gradientClipValue {
					scale := gradientClipValue / gradNorm
					for i := range grad {
						grad[i] *= scale
					}
				}

				net.StepBackward(state, grad)
				net.ApplyGradients(learningRate)

				totalLoss += loss
				lossCount++
			}
		}

		// Phase 2: Output phase - expect the target sequence
		correctOutputs := 0
		for _, target := range task.targetSeq {
			// Zero input during output phase
			zeroInput := make([]float32, vectorSize)
			state.SetInput(zeroInput)
			net.StepForward(state)

			targetQueue.Push(target)

			if targetQueue.IsFull() {
				delayedTarget := targetQueue.Pop()
				output := state.GetOutput()

				loss, grad := calculateSequenceLoss(output, delayedTarget)

				// Check if output is close to target
				isCorrect := true
				for i := 0; i < vectorSize; i++ {
					if math.Abs(float64(output[i]-delayedTarget[i])) > 0.3 {
						isCorrect = false
						break
					}
				}
				if isCorrect {
					correctOutputs++
				}

				// Gradient clipping
				gradNorm := float32(0.0)
				for _, g := range grad {
					gradNorm += g * g
				}
				gradNorm = float32(math.Sqrt(float64(gradNorm)))

				if gradNorm > gradientClipValue {
					scale := gradientClipValue / gradNorm
					for i := range grad {
						grad[i] *= scale
					}
				}

				net.StepBackward(state, grad)
				net.ApplyGradients(learningRate)

				totalLoss += loss
				lossCount++
			}
		}

		// Flush remaining targets
		for targetQueue.IsFull() == false && lossCount < len(task.inputSeq)+len(task.targetSeq) {
			zeroInput := make([]float32, vectorSize)
			state.SetInput(zeroInput)
			net.StepForward(state)

			zeroTarget := make([]float32, vectorSize)
			targetQueue.Push(zeroTarget)

			if targetQueue.IsFull() {
				delayedTarget := targetQueue.Pop()
				output := state.GetOutput()

				loss, grad := calculateSequenceLoss(output, delayedTarget)

				gradNorm := float32(0.0)
				for _, g := range grad {
					gradNorm += g * g
				}
				gradNorm = float32(math.Sqrt(float64(gradNorm)))

				if gradNorm > gradientClipValue {
					scale := gradientClipValue / gradNorm
					for i := range grad {
						grad[i] *= scale
					}
				}

				net.StepBackward(state, grad)
				net.ApplyGradients(learningRate)

				totalLoss += loss
				lossCount++
			}
		}

		// Task success if most outputs are correct
		if float32(correctOutputs) >= float32(len(task.targetSeq))*0.8 {
			taskSuccesses[task.taskType]++
		}

		avgLoss := totalLoss / float32(lossCount)

		// Decay learning rate
		learningRate *= decayRate
		if learningRate < minLearningRate {
			learningRate = minLearningRate
		}

		// Logging
		if episode%100 == 0 {
			successRate := float32(0.0)
			if taskAttempts[task.taskType] > 0 {
				successRate = float32(taskSuccesses[task.taskType]) / float32(taskAttempts[task.taskType]) * 100
			}

			fmt.Printf("Ep %5d | %-15s | Loss: %.6f | Success: %2d/%2d (%.1f%%) | LR: %.6f\n",
				episode, task.description, avgLoss,
				correctOutputs, len(task.targetSeq), successRate, learningRate)
		}

		// Periodic summary
		if (episode+1)%1000 == 0 {
			fmt.Println()
			fmt.Printf("=== Summary at Episode %d ===\n", episode+1)
			for taskType := TaskCopy; taskType <= TaskPrioritySort; taskType++ {
				attempts := taskAttempts[taskType]
				successes := taskSuccesses[taskType]
				rate := float32(0.0)
				if attempts > 0 {
					rate = float32(successes) / float32(attempts) * 100
				}
				fmt.Printf("  %-20s: %4d/%4d (%.1f%%)\n",
					taskType.String(), successes, attempts, rate)
			}
			fmt.Println("─────────────────────────────────────────────────────────────────────")
		}
	}

	totalTime := time.Since(startTime)

	fmt.Println()
	fmt.Println("╔════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  Training Complete                                                 ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════════╝")
	fmt.Printf("Total Time: %v\n", totalTime)
	fmt.Printf("Speed: %.2f episodes/sec\n", float64(totalEpisodes)/totalTime.Seconds())
	fmt.Println()
	fmt.Println("Final Task Performance:")
	for taskType := TaskCopy; taskType <= TaskPrioritySort; taskType++ {
		attempts := taskAttempts[taskType]
		successes := taskSuccesses[taskType]
		rate := float32(0.0)
		if attempts > 0 {
			rate = float32(successes) / float32(attempts) * 100
		}

		bar := ""
		barLen := int(rate / 2) // 50 chars = 100%
		for i := 0; i < 50; i++ {
			if i < barLen {
				bar += "█"
			} else {
				bar += "░"
			}
		}

		fmt.Printf("  %-20s [%s] %.1f%%\n", taskType.String(), bar, rate)
	}
	fmt.Println()

	// Evaluation phase
	fmt.Println("╔════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  Evaluation on Test Tasks                                         ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	testTasks := []*SequenceTask{
		taskGen.GenerateCopyTask(),
		taskGen.GenerateRepeatCopyTask(),
		taskGen.GenerateAssociativeRecallTask(),
		taskGen.GeneratePrioritySortTask(),
	}

	for _, task := range testTasks {
		fmt.Printf("\nTask: %s (%s)\n", task.description, task.taskType)
		fmt.Println("Input sequence:")
		for i, input := range task.inputSeq {
			fmt.Printf("  Step %2d: [", i)
			for j := 0; j < min(5, len(input)); j++ {
				fmt.Printf("%.3f ", input[j])
			}
			fmt.Println("...]")
		}

		// Run the task
		targetQueue.Clear()
		outputs := make([][]float32, 0)

		// Input phase
		for _, input := range task.inputSeq {
			state.SetInput(input)
			net.StepForward(state)
		}

		// Output phase
		for range task.targetSeq {
			zeroInput := make([]float32, vectorSize)
			state.SetInput(zeroInput)
			net.StepForward(state)

			output := make([]float32, vectorSize)
			copy(output, state.GetOutput())
			outputs = append(outputs, output)
		}

		fmt.Println("\nExpected vs Actual:")
		correctCount := 0
		for i := 0; i < len(task.targetSeq); i++ {
			target := task.targetSeq[i]
			output := outputs[i]

			isCorrect := true
			for j := 0; j < vectorSize; j++ {
				if math.Abs(float64(output[j]-target[j])) > 0.3 {
					isCorrect = false
				}
			}

			mark := "✗"
			if isCorrect {
				mark = "✓"
				correctCount++
			}

			fmt.Printf("  %s Step %2d: Target [", mark, i)
			for j := 0; j < min(5, len(target)); j++ {
				fmt.Printf("%.3f ", target[j])
			}
			fmt.Printf("...] Output [")
			for j := 0; j < min(5, len(output)); j++ {
				fmt.Printf("%.3f ", output[j])
			}
			fmt.Println("...]")
		}

		fmt.Printf("\nAccuracy: %d/%d (%.1f%%)\n",
			correctCount, len(task.targetSeq),
			float32(correctCount)/float32(len(task.targetSeq))*100)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
