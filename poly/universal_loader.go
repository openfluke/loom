package poly

import (
	"fmt"
	"math"
	"os"
	"sort"
)

// TensorMeta holds geometric and statistical metadata for a tensor.
type TensorMeta struct {
	Idx      int
	Shape    []int
	Data     []float32
	MeanAbs       float32
	Variance      float32
	Rank          int
	OriginalDType DType
}

// LayerArchetype represents a detected structural unit in the model.
type LayerArchetype struct {
	Type        LayerType
	TypeName    string
	Indices     map[string]int
	GeomMetrics map[string]int
}

// UserHints allows manual mapping for ambiguous tensor indices.
var UserHints = make(map[int]LayerType)

// LoadUniversal loads a model from a safetensors file and auto-detects its architecture.
func LoadUniversal(path string) (*VolumetricNetwork, error) {
	_, archetypes, _, geometries, err := LoadUniversalDetailed(path)
	if err != nil {
		return nil, err
	}
	return MountGeometrically(archetypes, geometries), nil
}

// LoadUniversalDetailed performs a deep analysis of a safetensors file.
func LoadUniversalDetailed(path string) (int, []LayerArchetype, []int, []TensorMeta, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, nil, nil, nil, err
	}
	tws, err := LoadSafetensorsWithShapes(data)
	if err != nil {
		return 0, nil, nil, nil, err
	}

	var names []string
	for k := range tws {
		names = append(names, k)
	}
	sort.Strings(names)

	var geometries []TensorMeta
	for i, name := range names {
		t := tws[name]
		m, v := weightDistribution(t.Values)
		geometries = append(geometries, TensorMeta{
			Idx: i, Shape: t.Shape, Data: t.Values, MeanAbs: m, Variance: v, Rank: len(t.Shape),
			OriginalDType: ParseDType(t.DType),
		})
	}

	archetypes, missed := ProbeDeepGeometry(geometries)
	return len(geometries), archetypes, missed, geometries, nil
}

func weightDistribution(data []float32) (meanAbs, variance float32) {
	if len(data) == 0 {
		return 0, 0
	}
	var sumAbs, sumSq float64
	for _, v := range data {
		av := math.Abs(float64(v))
		sumAbs += av
		sumSq += av * av
	}
	mean := float32(sumAbs / float64(len(data)))
	varSq := float32(sumSq/float64(len(data)) - float64(mean*mean))
	return mean, varSq
}

// ProbeDeepGeometry identifies layer patterns within a set of tensors.
func ProbeDeepGeometry(geoms []TensorMeta) ([]LayerArchetype, []int) {
	var archetypes []LayerArchetype
	used := make(map[int]bool)

	// Step 1: Hints
	for idx, hType := range UserHints {
		if idx < len(geoms) {
			used[idx] = true
			archetypes = append(archetypes, LayerArchetype{
				Type: hType, TypeName: "HINTED Layer",
				Indices: map[string]int{"w": idx},
			})
		}
	}

	// Step 2: Complex (MHA, LSTM, SwiGLU)
	for i := range geoms {
		if used[i] {
			continue
		}
		if arch, ok := matchMHA(geoms, i, used); ok {
			archetypes = append(archetypes, arch)
			continue
		}
		if arch, ok := matchLSTM(geoms, i, used); ok {
			archetypes = append(archetypes, arch)
			continue
		}
		if arch, ok := matchFFN(geoms, i, used); ok {
			archetypes = append(archetypes, arch)
			continue
		}
		if arch, ok := matchNormPair(geoms, i, used); ok {
			archetypes = append(archetypes, arch)
			continue
		}
	}

	// Step 3: Atomic & Metadata
	for i := range geoms {
		if used[i] {
			continue
		}
		g := geoms[i]
		if len(g.Data) < 10 {
			used[i] = true
			archetypes = append(archetypes, LayerArchetype{
				Type: LayerSequential, TypeName: "Structural Metadata",
				Indices: map[string]int{"m": i},
			})
			continue
		}
		if g.Rank == 4 {
			used[i] = true
			arch := LayerArchetype{
				Type: LayerCNN2, TypeName: "CNN2D",
				Indices:     map[string]int{"w": i},
				GeomMetrics: map[string]int{"f": g.Shape[0], "c": g.Shape[1], "k": g.Shape[2]},
			}
			greedyBiasSniff(geoms, &arch, used, g.Shape[0])
			archetypes = append(archetypes, arch)
		} else if g.Rank == 3 {
			used[i] = true
			arch := LayerArchetype{
				Type: LayerCNN1, TypeName: "CNN1D",
				Indices:     map[string]int{"w": i},
				GeomMetrics: map[string]int{"f": g.Shape[0], "c": g.Shape[1], "k": g.Shape[2]},
			}
			greedyBiasSniff(geoms, &arch, used, g.Shape[0])
			archetypes = append(archetypes, arch)
		} else if g.Rank == 2 {
			used[i] = true
			arch := LayerArchetype{Indices: make(map[string]int)}
			if g.Shape[0] > g.Shape[1]*10 {
				arch.Type = LayerEmbedding
				arch.TypeName = "Embedding Cluster"
				arch.Indices["w"] = i
				arch.GeomMetrics = map[string]int{"v": g.Shape[0], "d": g.Shape[1]}
			} else {
				arch.Type = LayerDense
				arch.TypeName = "Dense Linear"
				arch.Indices["w"] = i
				arch.GeomMetrics = map[string]int{"out": g.Shape[0], "in": g.Shape[1]}
				greedyBiasSniff(geoms, &arch, used, g.Shape[0])
			}
			archetypes = append(archetypes, arch)
		} else if g.Rank == 1 && g.MeanAbs > 0.4 {
			used[i] = true
			archetypes = append(archetypes, LayerArchetype{
				Type: LayerRMSNorm, TypeName: "Normalization Parameter",
				Indices: map[string]int{"w": i}, GeomMetrics: map[string]int{"d": g.Shape[0]},
			})
		}
	}

	var missed []int
	for i := range geoms {
		if !used[i] {
			missed = append(missed, i)
		}
	}
	return archetypes, missed
}

func greedyBiasSniff(geoms []TensorMeta, arch *LayerArchetype, used map[int]bool, dim int) {
	for j, o := range geoms {
		if !used[j] && o.Rank == 1 && o.Shape[0] == dim && o.MeanAbs < 0.2 {
			used[j] = true
			arch.Indices["b"] = j
			break
		}
	}
}

func matchMHA(geoms []TensorMeta, pivot int, used map[int]bool) (LayerArchetype, bool) {
	g := geoms[pivot]
	if g.Rank != 2 || g.Shape[0] != g.Shape[1] {
		return LayerArchetype{}, false
	}
	dim := g.Shape[0]
	cluster := []int{pivot}
	for j, o := range geoms {
		if used[j] || j == pivot || o.Rank != 2 || o.Shape[0] != dim || o.Shape[1] != dim {
			continue
		}
		cluster = append(cluster, j)
		if len(cluster) == 4 {
			break
		}
	}
	if len(cluster) == 4 {
		for _, idx := range cluster { used[idx] = true }
		arch := LayerArchetype{
			Type: LayerMultiHeadAttention, TypeName: "Multi-Head Attention",
			Indices:     map[string]int{"q": cluster[0], "k": cluster[1], "v": cluster[2], "o": cluster[3]},
			GeomMetrics: map[string]int{"d": dim},
		}
		// Greedy Bias Sniff for MHA
		for _, name := range []string{"qb", "kb", "vb", "ob"} {
			for j, o := range geoms {
				if !used[j] && o.Rank == 1 && o.Shape[0] == dim && o.MeanAbs < 0.2 {
					used[j] = true
					arch.Indices[name] = j
					break
				}
			}
		}
		return arch, true
	}
	return LayerArchetype{}, false
}

func matchLSTM(geoms []TensorMeta, pivot int, used map[int]bool) (LayerArchetype, bool) {
	g := geoms[pivot]
	if g.Rank != 2 || g.Shape[0] != g.Shape[1]*4 {
		return LayerArchetype{}, false
	}
	for j, o := range geoms {
		if used[j] || j == pivot || o.Rank != 2 || o.Shape[0] != g.Shape[0] || o.Shape[1] != g.Shape[1] {
			continue
		}
		used[pivot] = true
		used[j] = true
		return LayerArchetype{
			Type: LayerLSTM, TypeName: "LSTM Unit",
			Indices:     map[string]int{"ih": pivot, "hh": j},
			GeomMetrics: map[string]int{"h": g.Shape[1]},
		}, true
	}
	return LayerArchetype{}, false
}

func matchFFN(geoms []TensorMeta, pivot int, used map[int]bool) (LayerArchetype, bool) {
	g := geoms[pivot]
	if g.Rank != 2 {
		return LayerArchetype{}, false
	}
	da, db := g.Shape[0], g.Shape[1]
	cluster := []int{pivot}
	for j, o := range geoms {
		if used[j] || j == pivot || o.Rank != 2 {
			continue
		}
		if (o.Shape[0] == da && o.Shape[1] == db) || (o.Shape[0] == db && o.Shape[1] == da) {
			cluster = append(cluster, j)
			if len(cluster) == 3 {
				break
			}
		}
	}
	if len(cluster) == 3 {
		var downIdx int = -1
		for _, idx := range cluster {
			if geoms[idx].Shape[0] < geoms[idx].Shape[1] {
				downIdx = idx
			}
		}
		if downIdx == -1 {
			return LayerArchetype{}, false
		}
		for _, idx := range cluster { used[idx] = true }
		arch := LayerArchetype{Type: LayerSwiGLU, TypeName: "SwiGLU Block", Indices: make(map[string]int)}
		arch.Indices["d"] = downIdx
		arch.GeomMetrics = map[string]int{"h": geoms[downIdx].Shape[0], "i": geoms[downIdx].Shape[1]}
		for _, idx := range cluster {
			if idx != downIdx {
				if _, ok := arch.Indices["g"]; !ok {
					arch.Indices["g"] = idx
				} else {
					arch.Indices["u"] = idx
				}
			}
		}
		return arch, true
	}
	return LayerArchetype{}, false
}

func matchNormPair(geoms []TensorMeta, pivot int, used map[int]bool) (LayerArchetype, bool) {
	g := geoms[pivot]
	if g.Rank != 1 {
		return LayerArchetype{}, false
	}
	dim := g.Shape[0]
	cluster := make(map[string]int)
	if g.MeanAbs > 0.4 { cluster["s"] = pivot } else { cluster["b"] = pivot }

	for j, o := range geoms {
		if used[j] || j == pivot || o.Rank != 1 || o.Shape[0] != dim {
			continue
		}
		if o.MeanAbs > 0.4 && cluster["s"] == 0 {
			cluster["s"] = j
		} else if o.MeanAbs < 0.2 && cluster["b"] == 0 {
			cluster["b"] = j
		} else {
			cluster[fmt.Sprintf("stat_%d", j)] = j
		}
	}

	if len(cluster) >= 1 {
		for _, idx := range cluster { used[idx] = true }
		typeName := "Normalization Cluster"
		if _, hasS := cluster["s"]; hasS {
			if _, hasB := cluster["b"]; hasB { typeName = "LayerNorm Doublet" }
		}
		return LayerArchetype{
			Type: LayerLayerNorm, TypeName: typeName,
			Indices: cluster, GeomMetrics: map[string]int{"d": dim},
		}, true
	}
	return LayerArchetype{}, false
}

// MountGeometrically creates a VolumetricNetwork from archetypes and geometries.
func MountGeometrically(archs []LayerArchetype, geoms []TensorMeta) *VolumetricNetwork {
	net := NewVolumetricNetwork(1, 1, 1, len(archs))
	net.Layers = make([]VolumetricLayer, 0, len(archs))

	for i, a := range archs {
		l := VolumetricLayer{
			Network: net,
			Type:    a.Type,
			DType:   geoms[a.Indices["w"]].OriginalDType,
			Z: 0, Y: 0, X: 0, L: i,
		}

		switch a.Type {
		case LayerDense:
			l.InputHeight = a.GeomMetrics["in"]
			l.OutputHeight = a.GeomMetrics["out"]
			l.WeightStore = NewWeightStore(len(geoms[a.Indices["w"]].Data))
			copy(l.WeightStore.Master, geoms[a.Indices["w"]].Data)
			// TODO: Bias in WeightStore? Currently WeightStore is a single Master slice.
			// For now, we follow the pattern of the engine.
		case LayerSwiGLU:
			l.InputHeight = a.GeomMetrics["h"]
			l.OutputHeight = a.GeomMetrics["i"]
			// WeightStore initialization for SwiGLU might need complex handling
			// as it has 3 sets of weights. We'll follow the serialization count logic.
			wCount := 3*l.InputHeight*l.OutputHeight + 2*l.OutputHeight + l.InputHeight
			l.WeightStore = NewWeightStore(wCount)
		case LayerLSTM:
			l.InputHeight = a.GeomMetrics["h"] // Hidden size
			wCount := 4 * (l.InputHeight*l.InputHeight + l.InputHeight*l.InputHeight + l.InputHeight)
			l.WeightStore = NewWeightStore(wCount)
		case LayerLayerNorm:
			l.InputHeight = a.GeomMetrics["d"]
			l.WeightStore = NewWeightStore(2 * l.InputHeight)
		case LayerEmbedding:
			l.VocabSize = a.GeomMetrics["v"]
			l.EmbeddingDim = a.GeomMetrics["d"]
			l.WeightStore = NewWeightStore(l.VocabSize * l.EmbeddingDim)
			copy(l.WeightStore.Master, geoms[a.Indices["w"]].Data)
		case LayerRMSNorm:
			l.InputHeight = a.GeomMetrics["d"]
			l.WeightStore = NewWeightStore(l.InputHeight)
			copy(l.WeightStore.Master, geoms[a.Indices["w"]].Data)
		case LayerMultiHeadAttention:
			l.DModel = a.GeomMetrics["d"]
			l.NumHeads = 12 // Default
			l.WeightStore = NewWeightStore(2*l.DModel*l.DModel + 2*l.DModel*(l.DModel) + 2*l.DModel + 2*l.DModel)
		}
		net.Layers = append(net.Layers, l)
	}
	return net
}
