package poly

import (
	"fmt"
)

// GraftNetworksPolymorphic takes multiple heterogeneous VolumetricNetworks and grafts 
// their specific layers into a single parallel layer within a new network.
func GraftNetworksPolymorphic(networks []*VolumetricNetwork, combineMode string) (*VolumetricLayer, error) {
	if len(networks) == 0 {
		return nil, fmt.Errorf("no networks to graft")
	}

	var allBranches []VolumetricLayer
	
	for _, net := range networks {
		// Target the first layer of the first cell by default
		layer := net.GetLayer(0, 0, 0, 0)
		if layer == nil { continue }
		
		if layer.Type == LayerParallel {
			for _, b := range layer.ParallelBranches {
				allBranches = append(allBranches, b)
			}
		} else {
			allBranches = append(allBranches, *layer)
		}
	}
	
	if len(allBranches) == 0 {
		return nil, fmt.Errorf("no valid branches found in provided networks")
	}

	// Create the Super-Parallel layer
	parallelLayer := &VolumetricLayer{
		Type:             LayerParallel,
		ParallelBranches: allBranches,
		CombineMode:      combineMode,
	}
	
	if combineMode == "gated" {
		// Create a gate that selects based on input features
		// Default to first branch if no gate logic provided
		parallelLayer.KMeansOutputMode = "top1" 
	}
	
	return parallelLayer, nil
}

// CreateResidualGraft wraps a network in a residual block.
func CreateResidualGraft(main *VolumetricNetwork) *VolumetricLayer {
	l0 := main.GetLayer(0, 0, 0, 0)
	
	return &VolumetricLayer{
		Type: LayerParallel,
		ParallelBranches: []VolumetricLayer{
			{Type: LayerDense, InputHeight: l0.InputHeight, OutputHeight: l0.OutputHeight, IsDisabled: true}, // Identity branch
			*l0,
		},
		CombineMode: "add",
	}
}
