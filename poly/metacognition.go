package poly

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"sync"
	"time"
)

// ============================================================================
// HEURISTIC ENGINE
// ============================================================================

// MetaCondition defines what a heuristic rule checks.
type MetaCondition int

const (
	MetaCondNone          MetaCondition = iota
	MetaCondStdAbove                    // Output standard deviation exceeds threshold
	MetaCondStdBelow                    // Output standard deviation below threshold (vanishing)
	MetaCondAvgAbove                    // Output mean activation too high (exploding)
	MetaCondAvgBelow                    // Output mean activation too low
	MetaCondMaxAbove                    // Output max activation exceeds threshold
	MetaCondActiveBelow                 // Active neuron ratio below threshold (dead neurons)
	MetaCondGradientStorm               // Detects rapid std growth across history window
	MetaCondGainDrift                   // Output/Input magnitude ratio deviates from 1.0
)

// MetaRule is a single heuristic: if Condition(stats) crosses Threshold, fire Command.
type MetaRule struct {
	Condition MetaCondition
	Threshold float64
	Command   float32 // Command code: 70=Dense, 90=RMSNorm, 101=ResetIdentity, etc.
	Param     float32 // Optional parameter for the command
	Cooldown  int     // Minimum forward passes between firings (0 = fire every time)
	SelfOnly  bool    // If true, targets MetaObservedLayer. If false, targets grid position via Param.

	fireCount    int // internal: passes since last fired
	totalFirings int // internal: lifetime firing count
}

// DefaultStabilityRules returns a practical set of heuristic rules for general use.
func DefaultStabilityRules() []MetaRule {
	return []MetaRule{
		// If this layer's gain has drifted more than 5% from unity → reset weights
		{Condition: MetaCondGainDrift, Threshold: 0.05, Command: 101, SelfOnly: true},
		// If std deviation explodes → inject RMSNorm
		{Condition: MetaCondStdAbove, Threshold: 10.0, Command: 90, SelfOnly: true, Cooldown: 5},
		// If activations are all dead → reset to Dense
		{Condition: MetaCondActiveBelow, Threshold: 0.05, Command: 70, SelfOnly: true, Cooldown: 10},
	}
}

// evaluateCondition checks a single rule against input stats, output stats, and history.
func evaluateCondition(rule *MetaRule, inputStats, outputStats LayerStats, history []LayerStats) bool {
	switch rule.Condition {
	case MetaCondStdAbove:
		return float64(outputStats.Std) > rule.Threshold
	case MetaCondStdBelow:
		return float64(outputStats.Std) < rule.Threshold && outputStats.Std >= 0
	case MetaCondAvgAbove:
		return math.Abs(float64(outputStats.Avg)) > rule.Threshold
	case MetaCondAvgBelow:
		return math.Abs(float64(outputStats.Avg)) < rule.Threshold
	case MetaCondMaxAbove:
		return math.Abs(float64(outputStats.Max)) > rule.Threshold
	case MetaCondActiveBelow:
		if outputStats.Total == 0 {
			return false
		}
		return float64(outputStats.Active)/float64(outputStats.Total) < rule.Threshold
	case MetaCondGainDrift:
		// Compare output magnitude to input magnitude.
		// A healthy identity-like layer should have gain ≈ 1.0.
		// Threshold is the allowed deviation from 1.0 (e.g. 0.05 = 5% drift).
		inputMag := math.Abs(float64(inputStats.Avg))
		outputMag := math.Abs(float64(outputStats.Avg))

		// Skip if input is near zero (can't compute meaningful ratio)
		if inputMag < 1e-6 {
			return false
		}

		gain := outputMag / inputMag
		drift := math.Abs(gain - 1.0)
		return drift > rule.Threshold
	case MetaCondGradientStorm:
		if len(history) < 3 {
			return false
		}
		recent := history[len(history)-3:]
		if recent[0].Std == 0 {
			return false
		}
		growth := recent[2].Std / (recent[0].Std + 1e-8)
		return float64(growth) > rule.Threshold
	default:
		return false
	}
}

// ============================================================================
// TARGET RESOLUTION
// ============================================================================

// resolveCommandTarget finds the actual compute layer, unwrapping metacognition wrappers.
func resolveCommandTarget(net *VolumetricNetwork, z, y, x, l int) *VolumetricLayer {
	target := net.GetLayer(z, y, x, l)
	if target == nil {
		return nil
	}
	if target.Type == LayerMetacognition && target.MetaObservedLayer != nil {
		return target.MetaObservedLayer
	}
	return target
}

// ============================================================================
// PER-NETWORK EVENT HISTORY
// ============================================================================

var networkHistories sync.Map // map[*VolumetricNetwork][]LayerStats
const maxMetaHistory = 100

func recordMetaEvent(net *VolumetricNetwork, stats LayerStats) {
	val, _ := networkHistories.LoadOrStore(net, []LayerStats{})
	history := val.([]LayerStats)
	history = append(history, stats)
	if len(history) > maxMetaHistory {
		history = history[1:]
	}
	networkHistories.Store(net, history)
}

func getMetaHistory(net *VolumetricNetwork) []LayerStats {
	if net == nil {
		return nil
	}
	if val, ok := networkHistories.Load(net); ok {
		return val.([]LayerStats)
	}
	return nil
}

// ============================================================================
// FORWARD PASS
// ============================================================================

// MetacognitionForwardPolymorphic executes a meta-layer that wraps an observed layer.
// Execution order:
//  1. Run the observed layer to get its output
//  2. Compute input AND output stats (the diagnostic pair)
//  3. Evaluate heuristic rules comparing output vs input
//  4. If any rule fires → repair the observed layer, re-run it
//  5. If MetaNetwork exists, run it and apply MetaEffect instead
func MetacognitionForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {

	// Record input stats to network history
	inputStats := ComputeLayerStats(input)
	if layer.Network != nil {
		recordMetaEvent(layer.Network, inputStats)
	}
	history := getMetaHistory(layer.Network)

	// ── META-NETWORK PATH (learned behavior, bypasses heuristics) ──
	if layer.MetaNetwork != nil {
		metaInput := collectMetaInput(layer, input)
		metaF32 := ConvertTensor[T, float32](metaInput)
		metaOutF32, _, _ := ForwardPolymorphic(layer.MetaNetwork, metaF32)
		return applyMetaEffect(layer, input, metaOutF32)
	}

	// ── PURE HEURISTIC PATH ──
	if layer.MetaObservedLayer == nil {
		return input, input
	}

	// 1. Run the observed layer FIRST to see what it produces
	basePre, basePost := DispatchLayer(layer.MetaObservedLayer, input, nil)
	outputStats := ComputeLayerStats(basePost)

	// 2. Evaluate heuristic rules against the input→output diagnostic pair
	repaired := false
	if len(layer.MetaRules) > 0 {
		for i := range layer.MetaRules {
			rule := &layer.MetaRules[i]

			// Cooldown check
			if rule.Cooldown > 0 && rule.fireCount < rule.Cooldown {
				rule.fireCount++
				continue
			}

			if evaluateCondition(rule, inputStats, outputStats, history) {
				var target *VolumetricLayer
				if rule.SelfOnly {
					target = layer.MetaObservedLayer
				} else {
					target = resolveCommandTarget(layer.Network, 0, 0, int(rule.Param), 0)
				}

				if target != nil {
					executeMorphCommand(layer.Network, target, rule.Command, rule.Param)
					repaired = true
				}

				rule.fireCount = 0
				rule.totalFirings++
			} else {
				rule.fireCount++
			}
		}
	}

	// 3. If we repaired, re-run the (now fixed) observed layer
	if repaired {
		basePre, basePost = DispatchLayer(layer.MetaObservedLayer, input, nil)
	}

	return basePre, basePost
}

// ============================================================================
// META-INPUT COLLECTION
// ============================================================================

func collectMetaInput[T Numeric](layer *VolumetricLayer, input *Tensor[T]) *Tensor[T] {
	switch layer.MetaSource {
	case "input":
		return input
	case "stats":
		stats := ComputeLayerStats(input)
		// [Avg, Std, Max, Min, Active, Bias]
		return NewTensorFromSlice([]T{T(stats.Avg), T(stats.Std), T(stats.Max), T(stats.Min), T(stats.Active), 1.0}, 1, 6)
	case "history":
		const window = 10
		hData := make([]T, window*5+1)

		history := getMetaHistory(layer.Network)
		start := len(history) - window
		if start < 0 {
			start = 0
		}

		offset := 0
		for i := start; i < len(history); i++ {
			h := history[i]
			hData[offset] = T(h.Avg)
			hData[offset+1] = T(h.Std)
			hData[offset+2] = T(h.Max)
			hData[offset+3] = T(h.Min)
			hData[offset+4] = T(h.Active)
			offset += 5
		}
		hData[window*5] = 1.0
		return NewTensorFromSlice(hData, 1, len(hData))
	case "activations":
		data := make([]T, len(input.Data)+1)
		copy(data, input.Data)
		data[len(input.Data)] = 1.0
		return NewTensorFromSlice(data, 1, len(data))
	default:
		return input
	}
}

// ============================================================================
// MORPH COMMAND EXECUTION (shared by heuristics + autonomous_command)
// ============================================================================

func executeMorphCommand(net *VolumetricNetwork, target *VolumetricLayer, cmdVal float32, param float32) {
	if target == nil {
		return
	}

	switch {
	case cmdVal == 70: // MorphToDense
		if target.Type != LayerDense {
			fmt.Printf("[META] Heuristic: MorphToDense [WEIGHT PRESERVED]\n")
			oldW := target.WeightStore.Master
			target.Type = LayerDense
			target.Activation = ActivationLinear
			needed := target.InputHeight * target.OutputHeight
			target.WeightStore = NewWeightStore(needed)
			target.WeightStore.Master = make([]float32, needed)
			d := target.InputHeight
			if target.OutputHeight == d {
				for j := 0; j < d; j++ {
					target.WeightStore.Master[j*d+j] = 1.0
				}
			}
			if len(oldW) > 0 {
				n := len(target.WeightStore.Master)
				if len(oldW) < n {
					n = len(oldW)
				}
				copy(target.WeightStore.Master, oldW[:n])
			}
		}

	case cmdVal == 80: // MorphToCNN1
		if target.Type != LayerCNN1 && net != nil {
			fmt.Printf("[META] Heuristic: MorphToCNN1 [WEIGHT PRESERVED]\n")
			oldW := target.WeightStore.Master
			target.Type = LayerCNN1
			filters := 16
			kernelSize := 3
			needed := filters*target.InputHeight*kernelSize + filters
			target.WeightStore = NewWeightStore(needed)
			if len(oldW) > 0 {
				n := len(target.WeightStore.Master)
				if len(oldW) < n {
					n = len(oldW)
				}
				copy(target.WeightStore.Master, oldW[:n])
			}
		}

	case cmdVal == 81: // MorphToRNN
		if target.Type != LayerRNN && net != nil {
			fmt.Printf("[META] Heuristic: MorphToRNN [WEIGHT PRESERVED]\n")
			oldW := target.WeightStore.Master
			d := target.InputHeight
			target.Type = LayerRNN
			needed := d*d + d*d + d
			target.WeightStore = NewWeightStore(needed)
			if len(oldW) > 0 {
				n := len(target.WeightStore.Master)
				if len(oldW) < n {
					n = len(oldW)
				}
				copy(target.WeightStore.Master, oldW[:n])
			}
		}

	case cmdVal == 83: // MorphToSoftmax
		if target.Type != LayerSoftmax {
			fmt.Printf("[META] Heuristic: MorphToSoftmax\n")
			target.Type = LayerSoftmax
		}

	case cmdVal == 90: // MorphToRMSNorm
		if target.Type != LayerRMSNorm {
			fmt.Printf("[META] Heuristic: MorphToRMSNorm\n")
			target.Type = LayerRMSNorm
		}

	case cmdVal == 91: // MorphToSwiGLU
		if target.Type != LayerSwiGLU {
			fmt.Printf("[META] Heuristic: MorphToSwiGLU\n")
			target.Type = LayerSwiGLU
			inter := target.InputHeight * 4
			target.OutputHeight = inter
			target.WeightStore = NewWeightStore(target.InputHeight*inter*3 + inter*2 + target.InputHeight)
			if target.WeightStore != nil {
				target.WeightStore.Randomize(rand.Int63(), 0.1)
			}
		}

	case cmdVal == 92: // MorphToMHA
		if target.Type != LayerMultiHeadAttention && net != nil {
			fmt.Printf("[META] Heuristic: MorphToMHA [INTELLIGENCE PRESERVED]\n")
			oldW := target.WeightStore.Master
			d := target.InputHeight
			numHeads := 4
			target.Type = LayerMultiHeadAttention
			needed := 4 * d * d
			target.WeightStore = NewWeightStore(needed)
			if len(oldW) > 0 {
				gateSize := d * d
				for i := 0; i < 4; i++ {
					offset := i * gateSize
					if offset+gateSize <= len(target.WeightStore.Master) {
						n := gateSize
						if len(oldW) < n {
							n = len(oldW)
						}
						copy(target.WeightStore.Master[offset:], oldW[:n])
					}
				}
			}
			_ = numHeads
		}

	case cmdVal == 93: // MorphToLSTM
		if target.Type != LayerLSTM && net != nil {
			fmt.Printf("[META] Heuristic: MorphToLSTM [INTELLIGENCE PRESERVED]\n")
			oldW := target.WeightStore.Master
			d := target.InputHeight
			target.Type = LayerLSTM
			gateTotal := d*d + d*d + d
			needed := 4 * gateTotal
			target.WeightStore = NewWeightStore(needed)
			if len(oldW) > 0 {
				for i := 0; i < 4; i++ {
					offset := i * gateTotal
					if offset+d*d <= len(target.WeightStore.Master) {
						n := d * d
						if len(oldW) < n {
							n = len(oldW)
						}
						copy(target.WeightStore.Master[offset:], oldW[:n])
					}
				}
			}
		}

	case cmdVal == 98: // MorphToKMeans
		if target.Type != LayerKMeans && net != nil {
			numClusters := int(param)
			if numClusters <= 0 {
				numClusters = target.OutputHeight
			}
			if numClusters <= 0 {
				numClusters = 2 // Safety fallback
			}
			fmt.Printf("[META] Heuristic: MorphToKMeans(Clusters %d) [WEIGHT PRESERVED]\n", numClusters)
			oldW := target.WeightStore.Master
			target.Type = LayerKMeans
			target.NumClusters = numClusters
			target.OutputHeight = numClusters
			needed := numClusters * target.InputHeight
			target.WeightStore = NewWeightStore(needed)
			if len(oldW) > 0 {
				n := len(oldW)
				if n > needed { n = needed }
				copy(target.WeightStore.Master, oldW[:n])
			}
			if len(target.WeightStore.Master) > len(oldW) {
				// Randomize new centers
				target.WeightStore.Randomize(time.Now().UnixNano(), 0.1)
			}
			target.KMeansTemperature = 1.0
		}

	case cmdVal == 99: // MorphDType
		if target.DType != DType(int(param)) {
			fmt.Printf("[META] Heuristic: MorphDType(%d)\n", int(param))
			MorphLayer(target, DType(int(param)))
		}

	case cmdVal == 101: // ResetToIdentity — surgical weight repair
		if target.Type == LayerDense {
			d := target.InputHeight
			if target.OutputHeight == d && len(target.WeightStore.Master) == d*d {
				for j := 0; j < d*d; j++ {
					target.WeightStore.Master[j] = 0
				}
				for j := 0; j < d; j++ {
					target.WeightStore.Master[j*d+j] = 1.0
				}
			} else if target.OutputHeight == 1 && len(target.WeightStore.Master) >= d {
				for j := range target.WeightStore.Master {
					target.WeightStore.Master[j] = 0
				}
				target.WeightStore.Master[0] = 1.0
			}
		}
	}
}

// ============================================================================
// META-EFFECT APPLICATION (for meta-network path)
// ============================================================================

func applyMetaEffect[T Numeric](layer *VolumetricLayer, input *Tensor[T], metaOut *Tensor[float32]) (preAct, postAct *Tensor[T]) {
	var basePre, basePost *Tensor[T]
	if layer.MetaObservedLayer != nil {
		basePre, basePost = DispatchLayer(layer.MetaObservedLayer, input, nil)
	} else {
		basePre, basePost = input, input
	}

	preAct = basePre
	postAct = NewTensor[T](basePost.Shape...)

	switch layer.MetaEffect {
	case "gate":
		gate := float32(0)
		if len(metaOut.Data) > 0 {
			gate = 1.0 / (1.0 + float32(math.Exp(-float64(metaOut.Data[0]))))
		}
		for i := range basePost.Data {
			postAct.Data[i] = T(float32(basePost.Data[i]) * gate)
		}

	case "noise":
		scale := float32(0)
		if len(metaOut.Data) > 0 {
			scale = float32(math.Max(0, float64(metaOut.Data[0])))
		}
		for i := range basePost.Data {
			noise := (rand.Float32()*2 - 1) * scale
			postAct.Data[i] = T(float32(basePost.Data[i]) + noise)
		}

	case "intervention":
		intervene := false
		if len(metaOut.Data) > 0 && metaOut.Data[0] > 0.5 {
			intervene = true
		}
		if intervene {
			for i := range basePost.Data {
				if len(metaOut.Data) > i+1 {
					postAct.Data[i] = T(metaOut.Data[i+1])
				} else {
					postAct.Data[i] = 0
				}
			}
		} else {
			copy(postAct.Data, basePost.Data)
		}

	case "autonomous_command":
		if layer.Network != nil {
			for c := 0; c <= len(metaOut.Data)-3; c += 3 {
				cmdVal := metaOut.Data[c]
				targetIdx := int(metaOut.Data[c+1])
				param := metaOut.Data[c+2]

				if cmdVal == 0 {
					continue
				}

				target := resolveCommandTarget(layer.Network, 0, 0, targetIdx, 0)
				if target == nil {
					continue
				}

				if cmdVal >= 70 {
					executeMorphCommand(layer.Network, target, cmdVal, param)
				} else if cmdVal > 0 {
					cmdIdx := int(cmdVal)
					methods := layer.Network.ListMethods()
					if cmdIdx >= 0 && cmdIdx < len(methods) {
						methodName := methods[cmdIdx]
						networkVal := reflect.ValueOf(layer.Network)
						method := networkVal.MethodByName(methodName)
						if method.IsValid() {
							numParams := method.Type().NumIn()
							if numParams == 0 {
								method.Call(nil)
							} else if numParams == 2 {
								fmt.Printf("[META] Autonomous Command: %s(Layer %d, Value %f)\n", methodName, targetIdx, param)
								args := []reflect.Value{
									reflect.ValueOf(targetIdx),
									reflect.ValueOf(param),
								}
								method.Call(args)
							}
						}
					}
				}
			}
		}
		copyPostData(postAct, basePost)

	default:
		copyPostData(postAct, basePost)
	}

	if (len(postAct.Data) == 0 || postAct.Data == nil) && len(input.Data) > 0 {
		postAct.Data = make([]T, len(input.Data))
		copy(postAct.Data, input.Data)
		postAct.Shape = append([]int{}, input.Shape...)
	}

	return preAct, postAct
}

func copyPostData[T Numeric](dst, src *Tensor[T]) {
	if len(dst.Data) != len(src.Data) {
		dst.Data = make([]T, len(src.Data))
	}
	copy(dst.Data, src.Data)
}

// ============================================================================
// BACKWARD PASS
// ============================================================================

func MetacognitionBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	if layer.MetaObservedLayer != nil {
		return DispatchLayerBackward(layer.MetaObservedLayer, gradOutput, input, nil, preAct)
	}
	return gradOutput, nil
}

// ============================================================================
// CONVENIENCE: Wrap any network with metacognition
// ============================================================================

func WrapWithMetacognition(net *VolumetricNetwork, rules []MetaRule) {
	if rules == nil {
		rules = DefaultStabilityRules()
	}

	for i := range net.Layers {
		layer := &net.Layers[i]

		if layer.Type == LayerMetacognition || layer.IsDisabled {
			continue
		}

		// Full shallow copy — preserves ALL fields
		observed := new(VolumetricLayer)
		*observed = *layer

		// Convert the grid slot into a metacognition wrapper
		layer.Type = LayerMetacognition
		layer.MetaObservedLayer = observed
		layer.MetaSource = "stats"
		layer.MetaEffect = ""

		layerRules := make([]MetaRule, len(rules))
		copy(layerRules, rules)
		layer.MetaRules = layerRules
	}
}
