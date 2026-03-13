/*
DNA Engine: Hierarchical Spatial Correlation Engine
---------------------------------------------------
A topological reconstruction system for neural networks. Converts structural
signatures (LayerType, DType, weights) into 3D directional geometry for
high-fidelity comparison across diverse numerical families.
*/
package poly

import (
	"fmt"
	"math"
)

// LayerSignature represents the unique 3D topological "DNA" of a layer.
type LayerSignature struct {
	Z, Y, X, L int
	Type       LayerType
	DType      DType
	Weights    []float32 // Normalized, precision-simulated weights
}

// NetworkDNA is the complete genetic blueprint of a VolumetricNetwork.
type NetworkDNA []LayerSignature

// ExtractDNA generates the topological signatures for all layers in a network.
// It uses SimulatePrecision to ensure that comparison reflects the actual numerical behavior.
func ExtractDNA(n *VolumetricNetwork) NetworkDNA {
	dna := make(NetworkDNA, 0, len(n.Layers))
	for _, l := range n.Layers {
		var norm []float32
		if l.WeightStore != nil && len(l.WeightStore.Master) > 0 {
			// 1. Simulate the actual numerical behavior based on DType/Scale
			simulated := make([]float32, len(l.WeightStore.Master))
			scale := l.WeightStore.Scale
			if scale == 0 { scale = 1.0 }

			for i, w := range l.WeightStore.Master {
				simulated[i] = SimulatePrecision(w, l.DType, scale)
			}

			// 2. Normalize the vector for direction-based comparison (Cosine Similarity)
			norm = Normalize(simulated)
		} else {
			// Structural layer with no parameters (e.g., Softmax)
			// Give it a unique but neutral signature based on its presence
			norm = []float32{1.0}
		}

		dna = append(dna, LayerSignature{
			Z: l.Z, Y: l.Y, X: l.X, L: l.L,
			Type:    l.Type,
			DType:   l.DType,
			Weights: norm,
		})
	}
	return dna
}

// Normalize computes the unit vector of the input weight slice.
func Normalize(v []float32) []float32 {
	var sumSq float64
	for _, x := range v {
		sumSq += float64(x * x)
	}
	mag := float32(math.Sqrt(sumSq))
	if mag == 0 {
		return make([]float32, len(v))
	}
	res := make([]float32, len(v))
	for i, x := range v {
		res[i] = x / mag
	}
	return res
}

// CosineSimilarity acts as the "slider" (-1.0 to 1.0) for comparing two layer signatures.
func CosineSimilarity(s1, s2 LayerSignature) float32 {
	if s1.Type != s2.Type || s1.DType != s2.DType {
		return 0 // Architectural mismatch
	}
	if len(s1.Weights) != len(s2.Weights) {
		return 0 // Dimension mismatch
	}

	// SPECIAL CASE: Zero Vectors
	isZ1, isZ2 := true, true
	var dot float32
	for i := range s1.Weights {
		if s1.Weights[i] != 0 { isZ1 = false }
		if s2.Weights[i] != 0 { isZ2 = false }
		dot += s1.Weights[i] * s2.Weights[i]
	}
	
	if isZ1 && isZ2 { return 1.0 }
	if isZ1 || isZ2 { return 0.0 }

	return dot
}

// NetworkComparisonResult holds the hierarchical similarity metrics.
type NetworkComparisonResult struct {
	OverallOverlap  float32
	LayerOverlaps   map[string]float32 // "z,y,x,l" -> score
	LogicShifts     []LogicShift
}

// LogicShift identifies if a specific architectural pattern has moved in space.
type LogicShift struct {
	SourcePos string // "z,y,x,l"
	TargetPos string
	Overlap   float32
}

// CompareNetworks performs the hierarchical spatial correlation between two blueprints.
func CompareNetworks(dna1, dna2 NetworkDNA) NetworkComparisonResult {
	res := NetworkComparisonResult{
		LayerOverlaps: make(map[string]float32),
		LogicShifts:   []LogicShift{},
	}

	var totalOverlap float32
	var matchedCount int

	// 1. Hierarchical Direct Overlap (Medium/Average layer by layer)
	for _, sig1 := range dna1 {
		posKey := fmt.Sprintf("%d,%d,%d,%d", sig1.Z, sig1.Y, sig1.X, sig1.L)
		
		// Find corresponding layer in DNA2 at the same position
		for _, sig2 := range dna2 {
			if sig1.Z == sig2.Z && sig1.Y == sig2.Y && sig1.X == sig2.X && sig1.L == sig2.L {
				overlap := CosineSimilarity(sig1, sig2)
				res.LayerOverlaps[posKey] = overlap
				totalOverlap += overlap
				matchedCount++
				break
			}
		}
	}

	if matchedCount > 0 {
		res.OverallOverlap = totalOverlap / float32(matchedCount)
	}

	// 2. Cross-Depth Alignment (Identify Logic Shifts)
	// We check if a layer in DNA1 matches ANY layer in DNA2 better than its position-mate.
	for _, sig1 := range dna1 {
		bestOverlap := float32(-1.0)
		bestSig2 := LayerSignature{}
		found := false

		for _, sig2 := range dna2 {
			if len(sig1.Weights) != len(sig2.Weights) {
				continue
			}
			overlap := CosineSimilarity(sig1, sig2)
			if overlap > bestOverlap {
				bestOverlap = overlap
				bestSig2 = sig2
				found = true
			}
		}

		if found && bestOverlap > 0.8 { // Threshold for "logic match"
			pos1 := fmt.Sprintf("%d,%d,%d,%d", sig1.Z, sig1.Y, sig1.X, sig1.L)
			pos2 := fmt.Sprintf("%d,%d,%d,%d", bestSig2.Z, bestSig2.Y, bestSig2.X, bestSig2.L)
			if pos1 != pos2 {
				res.LogicShifts = append(res.LogicShifts, LogicShift{
					SourcePos: pos1,
					TargetPos: pos2,
					Overlap:   bestOverlap,
				})
			}
		}
	}

	return res
}
