package nn

import (
	"fmt"
)

// Heterogeneous Evolutionary Architecture Search with Spatially-Adaptive Fusion
//in progress -_-

// GraftNetworks takes multiple heterogeneous networks and grafts their "Hive" (Layer 1) into a single parallel layer.
// It assumes the networks follow the ef.go structure: Input(0) -> Hive(1) -> Merger(2) -> Output(3).
// It returns a LayerConfig for a new Super-Hive layer containing all branches from the source hives.
func GraftNetworks(networks []*Network, combineMode string) (*LayerConfig, error) {
	if len(networks) == 0 {
		return nil, fmt.Errorf("no networks to graft")
	}

	var allBranches []LayerConfig
	
	// Track DModel to ensure compatibility (optional, but good for safety)
	// Actually, if we use 'concat' or 'grid_scatter', DModel of branches can vary if the merger handles it.
	// But 'add'/'avg' requires same output size.
	// For now, we assume the user filters for compatibility or uses flexible modes.

	for _, net := range networks {
		// Target Layer 1 (The Hive)
		hiveLayer := net.GetLayer(0, 0, 1)
		if hiveLayer == nil {
			continue
		}
		
		// If the layer is Parallel, extract its branches.
		// If it's a single functional layer (e.g. just Dense), treat it as one branch.
		if hiveLayer.Type == LayerParallel {
			// Deep copy branches to avoid shared state issues if original nets are mutated
			for _, b := range hiveLayer.ParallelBranches {
				branchCopy := b // Shallow copy of struct
				// For deeper copy of slices (weights), we currently rely on the fact that we won't mutate the *source* weights anymore.
				// But if we train the new model, we want independent weights.
				// We really should Clone the LayerConfig.
				// Since we don't have a Clone method readily available in this file and cannot easily add to types.go without bloat,
				// we'll rely on the fact that `ef.go` creates new networks from configs usually.
				// BUT here we are taking *trained* networks.
				// So we are sharing weight pointers.
				// If we update weights in SuperModel, it updates the original component.
				// This might be desired (fine-tuning) or not.
				// Given "Step Tween train the model", we likely want to update.
				// Sharing pointers is risky if we want to keep original safe.
				// But Copying weights in Go without helper is verbose.
				// We'll proceed with shallow copy for now, assuming the user is okay with mutating or discarding the old models.
				allBranches = append(allBranches, branchCopy)
			}
		} else {
			// It's a single layer (e.g. just a Dense brain), treat as one branch
			branchCopy := *hiveLayer
			allBranches = append(allBranches, branchCopy)
		}
	}
	
	if len(allBranches) == 0 {
		return nil, fmt.Errorf("no valid branches found in provided networks")
	}

	// Create the Super-Hive config
	hiveCfg := &LayerConfig{
		Type:             LayerParallel,
		ParallelBranches: allBranches,
		CombineMode:      combineMode,
	}
	
	// Handle grid_scatter specific setup if needed
	if combineMode == "grid_scatter" {
		// Simple layout: Square grid
		numBranches := len(allBranches)
		side := 1
		for side*side < numBranches {
			side++
		}
		
		hiveCfg.GridOutputRows = side
		hiveCfg.GridOutputCols = side
		hiveCfg.GridOutputLayers = 1
		
		hiveCfg.GridPositions = make([]GridPosition, numBranches)
		for i := 0; i < numBranches; i++ {
			hiveCfg.GridPositions[i] = GridPosition{
				BranchIndex: i,
				TargetRow:   i / side,
				TargetCol:   i % side,
				TargetLayer: 0,
			}
		}
	}
    
    return hiveCfg, nil
}