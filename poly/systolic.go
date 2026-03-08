package poly

import (
	"sync"
	"time"
)

/*
SYSTOLIC GRID PROPAGATION (Discrete-Time Neural Mesh)
---------------------------------------------------
This technique treats the 3D Volumetric Grid as a living, breathing mesh.
Unlike standard "Sequential" propagation where an input flows through the 
entire network in one cycle, Systolic Propagation processes every 
coordinate simultaneously based on its current input buffer.

Key Features:
- Clock-Cycle Accuracy: Each 'Step' moves data exactly one layer deep.
- Double Buffering: Prevents race conditions during simultaneous updates.
- Temporal Feedback: Spatial hopping back to earlier layers creates 
  discrete-time recurrence (RNN behavior at scale).
- Systolic Backprop: Gradients flow backward through preserved temporal states.
*/

// SystolicState holds the temporal snapshot of the 3D grid.
type SystolicState[T Numeric] struct {
	// LayerData holds the current output of every layer in the grid.
	// Indexing follows VolumetricNetwork.GetIndex(z, y, x, l)
	LayerData []*Tensor[T]

	// BackwardContext stores pre-activations and inputs for backpropagation.
	// These are indexed by [Step][LayerIndex] to allow BPTT across clock cycles.
	HistoryIn  [][]*Tensor[T]
	HistoryPre [][]*Tensor[T]

	// Double buffering for simultaneous updates
	NextBuffer []*Tensor[T]

	// Grid Metadata
	StepCount uint64
	mu        sync.RWMutex

	// Neural Target Propagation Bridge
	tpState   *TargetPropState[T]
	lastInput *Tensor[T]
}

// NewSystolicState initializes a state for a specific Volumetric Network.
func NewSystolicState[T Numeric](n *VolumetricNetwork) *SystolicState[T] {
	total := len(n.Layers)
	return &SystolicState[T]{
		LayerData:  make([]*Tensor[T], total),
		NextBuffer: make([]*Tensor[T], total),
		HistoryIn:  make([][]*Tensor[T], 0),
		HistoryPre: make([][]*Tensor[T], 0),
	}
}

// SetInput injects data into the starting coordinate (0,0,0,0).
func (s *SystolicState[T]) SetInput(input *Tensor[T]) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastInput = input
	if len(s.LayerData) > 0 {
		s.LayerData[0] = input
	}
}

// SystolicForward executes one "Clock Cycle" across the entire 3D grid.
// Every layer processes its current input buffer and writes to the next buffer.
func SystolicForward[T Numeric](n *VolumetricNetwork, s *SystolicState[T], captureHistory bool) time.Duration {
	start := time.Now()
	s.mu.Lock()
	defer s.mu.Unlock()

	numLayers := len(n.Layers)
	
	// Prepare history capture if requested
	var currentIn, currentPre []*Tensor[T]
	if captureHistory {
		currentIn = make([]*Tensor[T], numLayers)
		currentPre = make([]*Tensor[T], numLayers)
	}

	// 1. Process all layers simultaneously
	for idx := range n.Layers {
		l := &n.Layers[idx]
		if l.IsDisabled {
			// Identity pass
			if idx > 0 {
				s.NextBuffer[idx] = s.LayerData[idx-1]
			} else {
				// Layer 0 stays what it was (Input source)
				s.NextBuffer[idx] = s.LayerData[idx]
			}
			continue
		}

		// Determine input source
		// In a systolic grid, input usually comes from the PREVIOUS index 
		// or a REMOTE spatial hop.
		var input *Tensor[T]
		if l.IsRemoteLink {
			targetIdx := n.GetIndex(l.TargetZ, l.TargetY, l.TargetX, l.TargetL)
			input = s.LayerData[targetIdx]
		} else if idx > 0 {
			input = s.LayerData[idx-1]
		} else {
			// Layer 0 uses its own persistent data (Injected via SetInput)
			input = s.LayerData[0]
		}

		if input == nil {
			continue
		}

		// Dispatch and capture
		pre, post := DispatchLayer(l, input, nil)
		s.NextBuffer[idx] = post

		if captureHistory {
			currentIn[idx] = input
			currentPre[idx] = pre
		}
	}

	// 2. Commit the buffer (Atomic Swap)
	for i := range s.LayerData {
		s.LayerData[i] = s.NextBuffer[i]
	}

	// 3. Update History
	if captureHistory {
		s.HistoryIn = append(s.HistoryIn, currentIn)
		s.HistoryPre = append(s.HistoryPre, currentPre)
	}

	s.StepCount++
	return time.Since(start)
}

// SystolicBackward propagates gradients backward through the systolic history.
// It walks backward through clock cycles, accurately routing gradients to their source coordinates.
func SystolicBackward[T Numeric](n *VolumetricNetwork, s *SystolicState[T], gradOutput *Tensor[T]) (gradIn *Tensor[T], layerGradients [][2]*Tensor[T], err error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(s.HistoryIn) == 0 {
		return nil, nil, nil
	}

	numSteps := len(s.HistoryIn)
	numLayers := len(n.Layers)
	
	layerGradients = make([][2]*Tensor[T], numLayers)
	
	// gradBuffers holds the current gradient "budget" for each coordinate.
	// In the systolic mesh, a coordinate's gradOutput is the sum of gradients 
	// from all coordinates that consumed its output in the previous cycle.
	gradBuffers := make([]*Tensor[T], numLayers)

	// Initialize final layer with external gradient
	gradBuffers[numLayers-1] = gradOutput

	// Walk backwards through time steps (The Temporal dimension)
	for step := numSteps - 1; step >= 0; step-- {
		stepIn := s.HistoryIn[step]
		stepPre := s.HistoryPre[step]
		
		// Next cycle's gradient accumulation buffer
		nextGradBuffers := make([]*Tensor[T], numLayers)

		// Walk backwards through spatial coordinates
		for idx := numLayers - 1; idx >= 0; idx-- {
			l := &n.Layers[idx]
			if l.IsDisabled {
				// Pass gradient back to source
				if gradBuffers[idx] != nil {
					accumulateMeshGrad(n, nextGradBuffers, idx, gradBuffers[idx])
				}
				continue
			}

			input := stepIn[idx]
			pre := stepPre[idx]
			currentGrad := gradBuffers[idx]

			if input == nil || pre == nil || currentGrad == nil {
				continue
			}

			// DISPATCH BACKWARD
			gIn, gW := DispatchLayerBackward(l, currentGrad, input, pre, nil)
			
			// Accumulate weight gradients for this layer
			if layerGradients[idx][1] == nil {
				layerGradients[idx] = [2]*Tensor[T]{gIn, gW}
			} else if gW != nil {
				for i := range layerGradients[idx][1].Data {
					layerGradients[idx][1].Data[i] += gW.Data[i]
				}
			}

			// Route gIn back to the coordinate that supplied the input for this layer
			if gIn != nil {
				accumulateMeshGrad(n, nextGradBuffers, idx, gIn)
			}
		}
		
		// Carry gradBuffers to the next (earlier in time) step
		gradBuffers = nextGradBuffers
	}

	return gradBuffers[0], layerGradients, nil
}

// SystolicApplyTargetProp bridges the Systolic state with the Target Propagation machinery.
// It uses the core 'Gap-Bridging' logic to update weights across the volumetric mesh.
func SystolicApplyTargetProp[T Numeric](n *VolumetricNetwork, s *SystolicState[T], globalTarget *Tensor[T], lr float32) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// 1. Initialize or update TargetPropState
	if s.tpState == nil {
		config := DefaultTargetPropConfig()
		config.UseChainRule = false // Default to Gap-Based for Systolic Mesh o_O
		s.tpState = NewTargetPropState[T](n, config)
	}

	// 2. Capture current mesh state as Forward Activations
	// Note: s.LayerData[idx] is the output of layer idx.
	// TargetProp expects s.ForwardActs[0] to be input, s.ForwardActs[1] to be output of Layer 0, etc.
	if len(s.LayerData) > 0 {
		s.tpState.ForwardActs[0] = s.lastInput
		for i := 0; i < len(s.LayerData); i++ {
			s.tpState.ForwardActs[i+1] = s.LayerData[i]
		}
	}

	// 3. SECONARY: If history is available, we could do BPTT-style Target Prop,
	// but for now, we do steady-state grid refinement.
	
	// Use the standard Target Prop Backward (sequential index assuming)
	// or implement mesh-aware variant.
	// Since n.Layers is the linear order of the grid, the standard backward 
	// usually works if the grid is designed sequentially.
	TargetPropBackward(n, s.tpState, globalTarget)

	// 4. Update Diagnostics
	s.tpState.CalculateLinkBudgets()

	// 5. Apply Gaps (Neural Weight Mutation)
	ApplyTargetPropGaps(n, s.tpState, lr)
}

// accumulateMeshGrad finds the source for layer 'idx' and adds 'grad' to its entry in 'buffers'.
func accumulateMeshGrad[T Numeric](n *VolumetricNetwork, buffers []*Tensor[T], idx int, grad *Tensor[T]) {
	l := &n.Layers[idx]
	var sourceIdx int
	
	if l.IsRemoteLink {
		sourceIdx = n.GetIndex(l.TargetZ, l.TargetY, l.TargetX, l.TargetL)
	} else if idx > 0 {
		sourceIdx = idx - 1
	} else {
		sourceIdx = 0 // Input site
	}

	if buffers[sourceIdx] == nil {
		buffers[sourceIdx] = grad.Clone()
	} else {
		for i := range buffers[sourceIdx].Data {
			buffers[sourceIdx].Data[i] += grad.Data[i]
		}
	}
}
