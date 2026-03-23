package poly

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"sync"
)

// Per-network event history for metacognition
var networkHistories sync.Map // map[*VolumetricNetwork][]LayerStats
const maxMetaHistory = 100

// MetacognitionForwardPolymorphic executes a sub-network that observes layer state and modulates the pass.
func MetacognitionForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if layer.MetaNetwork == nil {
		return input, input // Passthrough
	}

	// 1. RECORD CURRENT STATS TO NETWORK-SPECIFIC HISTORY
	currentStats := ComputeLayerStats(input)
	if layer.Network != nil {
		recordMetaEvent(layer.Network, currentStats)
	}

	// 2. COLLECT META-SOURCE
	metaInput := collectMetaInput(layer, input)

	// 3. DISPATCH META-NETWORK
	metaF32 := ConvertTensor[T, float32](metaInput)
	metaOutF32, _, _ := ForwardPolymorphic(layer.MetaNetwork, metaF32)

	// 4. APPLY META-EFFECT
	return applyMetaEffect(layer, input, metaOutF32)
}

func recordMetaEvent(net *VolumetricNetwork, stats LayerStats) {
	val, _ := networkHistories.LoadOrStore(net, []LayerStats{})
	history := val.([]LayerStats)
	history = append(history, stats)
	if len(history) > maxMetaHistory {
		history = history[1:]
	}
	networkHistories.Store(net, history)
}

func collectMetaInput[T Numeric](layer *VolumetricLayer, input *Tensor[T]) *Tensor[T] {
	switch layer.MetaSource {
	case "input":
		return input
	case "stats":
		stats := ComputeLayerStats(input)
		return NewTensorFromSlice([]T{T(stats.Avg), T(stats.Max), T(stats.Min), T(stats.Active), 1.0}, 1, 5)
	case "history":
		const window = 10
		hData := make([]T, window*4+1)
		
		var history []LayerStats
		if layer.Network != nil {
			if val, ok := networkHistories.Load(layer.Network); ok {
				history = val.([]LayerStats)
			}
		}
		
		start := len(history) - window
		if start < 0 { start = 0 }
		
		offset := 0
		for i := start; i < len(history); i++ {
			h := history[i]
			hData[offset] = T(h.Avg)
			hData[offset+1] = T(h.Max)
			hData[offset+2] = T(h.Min)
			hData[offset+3] = T(h.Active)
			offset += 4
		}
		hData[window*4] = 1.0 // Bias
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
			// Handle multiple commands in a single pass if metaOut is large enough
			for c := 0; c <= len(metaOut.Data)-3; c += 3 {
				cmdVal := metaOut.Data[c]
				targetIdx := int(metaOut.Data[c+1])
				param := metaOut.Data[c+2]

				if cmdVal == 0 { continue } // No-op

				// Handle Virtual Commands
				if cmdVal == 70 {
					l := layer.Network.GetLayer(0, 0, targetIdx, 0)
					if l != nil && l.Type != LayerDense {
						fmt.Printf("[META] Autonomous Command: MorphToDense(Layer %d) [WEIGHT PRESERVED]\n", targetIdx)
						oldW := l.WeightStore.Master
						layer.Network.InitDenseCell(0, 0, targetIdx, 0, l.InputHeight, ActivationLinear, 0.02)
						if len(oldW) > 0 {
							sliceLen := len(l.WeightStore.Master)
							if len(oldW) < sliceLen { sliceLen = len(oldW) }
							copy(l.WeightStore.Master, oldW[:sliceLen])
						}
					}
				} else if cmdVal == 80 {
					l := layer.Network.GetLayer(0, 0, targetIdx, 0)
					if l != nil && l.Type != LayerCNN1 {
						fmt.Printf("[META] Autonomous Command: MorphToCNN1(Layer %d) [WEIGHT PRESERVED]\n", targetIdx)
						oldW := l.WeightStore.Master
						layer.Network.InitCNNCell(0, 0, targetIdx, 0, LayerCNN1, l.InputHeight, 16, 3, DTypeFloat32, 0.02)
						if len(oldW) > 0 {
							sliceLen := len(l.WeightStore.Master)
							if len(oldW) < sliceLen { sliceLen = len(oldW) }
							copy(l.WeightStore.Master, oldW[:sliceLen])
						}
					}
				} else if cmdVal == 81 {
					l := layer.Network.GetLayer(0, 0, targetIdx, 0)
					if l != nil && l.Type != LayerRNN {
						fmt.Printf("[META] Autonomous Command: MorphToRNN(Layer %d) [WEIGHT PRESERVED]\n", targetIdx)
						oldW := l.WeightStore.Master
						layer.Network.InitRNNCell(0, 0, targetIdx, 0, l.InputHeight, 0.02)
						if len(oldW) > 0 {
							sliceLen := len(l.WeightStore.Master)
							if len(oldW) < sliceLen { sliceLen = len(oldW) }
							copy(l.WeightStore.Master, oldW[:sliceLen])
						}
					}
				} else if cmdVal == 82 {
					l := layer.Network.GetLayer(0, 0, targetIdx, 0)
					if l != nil && l.Type != LayerEmbedding {
						fmt.Printf("[META] Autonomous Command: MorphToEmbedding(Layer %d)\n", targetIdx)
						layer.Network.InitEmbeddingCell(0, 0, targetIdx, 0, 100, 32, DTypeFloat32)
					}
				} else if cmdVal == 83 {
					l := layer.Network.GetLayer(0, 0, targetIdx, 0)
					if l != nil && l.Type != LayerSoftmax {
						fmt.Printf("[META] Autonomous Command: MorphToSoftmax(Layer %d)\n", targetIdx)
						l.Type = LayerSoftmax
					}
				} else if cmdVal == 90 {
					l := layer.Network.GetLayer(0, 0, targetIdx, 0)
					if l != nil && l.Type != LayerRMSNorm {
						fmt.Printf("[META] Autonomous Command: MorphToRMSNorm(Layer %d)\n", targetIdx)
						l.Type = LayerRMSNorm
						l.InputHeight = l.Network.Layers[targetIdx].InputHeight
					}
				} else if cmdVal == 91 {
					l := layer.Network.GetLayer(0, 0, targetIdx, 0)
					if l != nil && l.Type != LayerSwiGLU {
						fmt.Printf("[META] Autonomous Command: MorphToSwiGLU(Layer %d)\n", targetIdx)
						l.Type = LayerSwiGLU
						inter := l.InputHeight * 4
						l.OutputHeight = inter
						l.WeightStore = NewWeightStore(l.InputHeight*inter*3 + inter*2 + l.InputHeight)
						if l.WeightStore != nil {
							l.WeightStore.Randomize(rand.Int63(), 0.1)
						}
					}
				} else if cmdVal == 92 {
					l := layer.Network.GetLayer(0, 0, targetIdx, 0)
					if l != nil && l.Type != LayerMultiHeadAttention {
						fmt.Printf("[META] Autonomous Command: MorphToMHA(Layer %d) [INTELLIGENCE PRESERVED]\n", targetIdx)
						oldW := l.WeightStore.Master
						layer.Network.InitMHACell(0, 0, targetIdx, 0, l.InputHeight, 4, 0.02)
						// Smart Seeding: Seed all Multi-Head projections (Q, K, V, O)
						if len(oldW) > 0 {
							gateSize := l.InputHeight * l.InputHeight
							for i := 0; i < 4; i++ {
								offset := i * gateSize
								if offset+gateSize <= len(l.WeightStore.Master) {
									copy(l.WeightStore.Master[offset:], oldW[:gateSize])
								}
							}
						}
					}
				} else if cmdVal == 93 {
					l := layer.Network.GetLayer(0, 0, targetIdx, 0)
					if l != nil && l.Type != LayerLSTM {
						fmt.Printf("[META] Autonomous Command: MorphToLSTM(Layer %d) [INTELLIGENCE PRESERVED]\n", targetIdx)
						oldW := l.WeightStore.Master
						layer.Network.InitLSTMCell(0, 0, targetIdx, 0, l.InputHeight, 0.02)
						// Smart Seeding: Seed all 4 LSTM gates with identity/old weights
						if len(oldW) > 0 {
							d := l.InputHeight
							gateTotal := d*d + d*d + d
							for i := 0; i < 4; i++ {
								offset := i * gateTotal
								// Seed input weights for this gate
								if offset+d*d <= len(l.WeightStore.Master) {
									copy(l.WeightStore.Master[offset:], oldW[:d*d])
								}
							}
						}
					}
				} else if cmdVal == 98 {
					l := layer.Network.GetLayer(0, 0, targetIdx, 0)
					if l != nil {
						// CRITICAL: Stability requires keeping OutputHeight consistent
						numClusters := l.OutputHeight 
						if l.Type != LayerKMeans {
							fmt.Printf("[META] Autonomous Command: MorphToKMeans(Layer %d, Clusters %d) [WEIGHT PRESERVED]\n", targetIdx, numClusters)
							// Seed clusters from existing Dense weights instead of randomizing
							oldW := l.WeightStore.Master
							layer.Network.InitKMeansCell(0, 0, targetIdx, 0, numClusters, l.InputHeight, DTypeFloat32)
							if len(oldW) >= numClusters*l.InputHeight {
								copy(l.WeightStore.Master, oldW[:numClusters*l.InputHeight])
							}
							l.KMeansTemperature = 1.0
						}
					}
				} else if cmdVal == 99 {
					targetLayer := layer.Network.GetLayer(0, 0, targetIdx, 0)
					if targetLayer != nil && targetLayer.DType != DType(int(param)) {
						fmt.Printf("[META] Autonomous Command: MorphLayer(Layer %d, DType %d)\n", targetIdx, int(param))
						MorphLayer(targetLayer, DType(int(param)))
					}
				} else if cmdVal > 0 {
					// Standard Methods (Skip 0 as NOP)
					cmdIdx := int(cmdVal)
					methods := layer.Network.ListMethods()
					if cmdIdx >= 0 && cmdIdx < len(methods) {
						methodName := methods[cmdIdx]
						networkVal := reflect.ValueOf(layer.Network)
						method := networkVal.MethodByName(methodName)
						if method.IsValid() {
							numParams := method.Type().NumIn()
							var args []reflect.Value
							// We only support (targetIdx int, param float32) or ()
							if numParams == 0 {
								method.Call(nil)
							} else if numParams == 2 {
								fmt.Printf("[META] Autonomous Command: %s(Layer %d, Value %f)\n", methodName, targetIdx, param)
								args = append(args, reflect.ValueOf(targetIdx))
								args = append(args, reflect.ValueOf(param))
								method.Call(args)
							}
						}
					}
				}
			}
		}
		if len(postAct.Data) != len(basePost.Data) {
			postAct.Data = make([]T, len(basePost.Data))
		}
		copy(postAct.Data, basePost.Data)
	default:
		if len(postAct.Data) != len(basePost.Data) {
			postAct.Data = make([]T, len(basePost.Data))
		}
		copy(postAct.Data, basePost.Data)
	}

	if (len(postAct.Data) == 0 || postAct.Data == nil) && len(input.Data) > 0 {
		// Recovery: Mirror input and shape to prevent downstream panics
		postAct.Data = make([]T, len(input.Data))
		copy(postAct.Data, input.Data)
		postAct.Shape = append([]int{}, input.Shape...)
	}

	return preAct, postAct
}
