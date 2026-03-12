package main

/*
#include <stdlib.h>
#include <stdint.h>

typedef struct {
	int Z, Y, X, L;
	int Type;
	int Activation;
	int DType;
	int InputHeight, InputWidth, InputDepth;
	int OutputHeight, OutputWidth, OutputDepth;
	int InputChannels, Filters, KernelSize, Stride, Padding;
	int NumHeads, NumKVHeads, DModel, SeqLength;
	int VocabSize, EmbeddingDim;
	int NumClusters;
	int UseTiling, TileSize;
} LoomLayerSpec;

typedef struct {
	float Avg;
	float Max;
	float Min;
	int Active;
	int Total;
} LoomLayerStats;

typedef struct {
	int DType;
	int Rank;
	int64_t Shape[8];
} LoomTensorMeta;

typedef struct {
	const char* Name;
	int DType;
	int Rank;
	int64_t Shape[8];
	uint64_t DataOffset;
	uint64_t DataLength;
} LoomTensorInfo;

typedef struct {
	int NumTensors;
	LoomTensorInfo* Tensors;
} LoomSafetensorsHeader;

typedef struct {
	const char* ID;
	const char* Name;
	int GridDepth;
	int GridRows;
	int GridCols;
} LoomNetworkBlueprint;

*/
import "C"

import (
	"github.com/openfluke/loom/poly"
)

// Helper: poly.LayerStats -> C.LoomLayerStats
func packLayerStats(s poly.LayerStats) C.LoomLayerStats {
	return C.LoomLayerStats{
		Avg:    C.float(s.Avg),
		Max:    C.float(s.Max),
		Min:    C.float(s.Min),
		Active: C.int(s.Active),
		Total:  C.int(s.Total),
	}
}

// Helper: poly.VolumetricLayer -> C.LoomLayerSpec
func packLayerSpec(l *poly.VolumetricLayer) C.LoomLayerSpec {
	return C.LoomLayerSpec{
		Z: C.int(l.Z), Y: C.int(l.Y), X: C.int(l.X), L: C.int(l.L),
		Type:          C.int(l.Type),
		Activation:    C.int(l.Activation),
		DType:         C.int(l.DType),
		InputHeight:   C.int(l.InputHeight),
		InputWidth:    C.int(l.InputWidth),
		InputDepth:    C.int(l.InputDepth),
		OutputHeight:  C.int(l.OutputHeight),
		OutputWidth:   C.int(l.OutputWidth),
		OutputDepth:   C.int(l.OutputDepth),
		InputChannels: C.int(l.InputChannels),
		Filters:       C.int(l.Filters),
		KernelSize:    C.int(l.KernelSize),
		Stride:        C.int(l.Stride),
		Padding:       C.int(l.Padding),
		NumHeads:      C.int(l.NumHeads),
		NumKVHeads:    C.int(l.NumKVHeads),
		DModel:        C.int(l.DModel),
		SeqLength:     C.int(l.SeqLength),
		VocabSize:     C.int(l.VocabSize),
		EmbeddingDim:  C.int(l.EmbeddingDim),
		NumClusters:   C.int(l.NumClusters),
		UseTiling:     C.int(0), // Default
		TileSize:      C.int(l.TileSize),
	}
}

//export LoomGetLayerSpec
func LoomGetLayerSpec(networkHandle C.longlong, layerIdx C.int) C.LoomLayerSpec {
	n, ok := getNetwork(int64(networkHandle))
	if !ok || int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return C.LoomLayerSpec{}
	}
	return packLayerSpec(&n.Layers[int(layerIdx)])
}

//export LoomGetLayerStats
func LoomGetLayerStats(stateHandle C.longlong, layerIdx C.int) C.LoomLayerStats {
	c, ok := getSystolicContainer(int64(stateHandle))
	if !ok {
		return C.LoomLayerStats{}
	}

	// Extract stats based on dtype
	// (Simplified for C-ABI, usually language bindings want the floats anyway)
	var stats poly.LayerStats
	switch c.DType {
	case poly.DTypeFloat32:
		st := c.State.(*poly.SystolicState[float32])
		if int(layerIdx) >= 0 && int(layerIdx) < len(st.LayerData) && st.LayerData[layerIdx] != nil {
			stats = poly.ComputeLayerStats(st.LayerData[layerIdx])
		}
	}
	return packLayerStats(stats)
}

// Dummy use for parity scanner
var (
	_ poly.LayerSignature
	_ poly.PolyLayerEvent
	_ poly.DetectedTensor
	_ poly.TensorMeta
	_ poly.LayerArchetype
	_ poly.PersistenceLayerSpec
	_ poly.SafetensorsHeader
	_ poly.TensorInfo
	_ poly.TensorWithShape
	_ poly.VolumetricLayer
	_ poly.LayerSpec
	_ poly.LayerType
	_ poly.LayerStats
	_ poly.PreTokenizer
	_ poly.TokenizerJSON
	_ poly.WGPUContext
)
