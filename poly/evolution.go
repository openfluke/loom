/*
Evolution Engine: DNA Splice & NEAT-style Topology Evolution
------------------------------------------------------------
Extends the DNA Engine (dna.go) with two capabilities:

  1. DNA Splice / Genetic Crossover
     Takes two trained parent networks, compares their NetworkDNA,
     and produces a child network whose weights are blended from
     both parents, guided by per-layer cosine similarity scores.

  2. NEAT-style Topology Evolution
     Mutates a network's topology (layer types, activations,
     remote-link connections) and weights without destroying
     learned structure. Supports a full population-based
     evolution loop via NEATPopulation.
*/
package poly

import (
	"fmt"
	"math/rand"
	"time"
)

// ============================================================
// DNA Splice / Genetic Crossover
// ============================================================

// SpliceConfig controls how two parent networks are combined.
type SpliceConfig struct {
	// CrossoverMode: "uniform", "point", or "blend"
	CrossoverMode string
	// BlendAlpha: interpolation factor for "blend" mode (0=all A, 1=all B)
	BlendAlpha float32
	// SplitRatio: fraction of weights taken from parent A in "point" mode
	SplitRatio float64
	// FitnessA/B: optional fitness scores to bias crossover toward fitter parent
	FitnessA float64
	FitnessB float64
}

// DefaultSpliceConfig returns a balanced blend configuration.
func DefaultSpliceConfig() SpliceConfig {
	return SpliceConfig{
		CrossoverMode: "blend",
		BlendAlpha:    0.5,
		SplitRatio:    0.5,
	}
}

// SpliceResult holds the outcome of a DNA splice operation.
type SpliceResult struct {
	Child        *VolumetricNetwork
	ParentADNA   NetworkDNA
	ParentBDNA   NetworkDNA
	ChildDNA     NetworkDNA
	Similarities map[string]float32 // "z,y,x,l" -> cosine similarity used
	BlendedCount int                // number of layers actually blended
}

// SpliceDNA merges two trained parent networks into a child network.
//
// parentA is the structural template (grid dimensions, layer types are inherited).
// parentB contributes weights to matching layers, weighted by DNA similarity.
//
// For each layer at coordinate (z,y,x,l):
//   - If both parents have the layer and their types match, weights are blended.
//   - If parentB has no matching layer, the child keeps parentA's weights.
//
// The three blend strategies:
//
//	"blend"   — interpolate: child[i] = wA[i]*(1-α) + wB[i]*α
//	            α is modulated by cosine similarity and relative fitness.
//	"point"   — split at SplitRatio: first N weights from A, rest from B.
//	"uniform" — per-weight random pick from A or B, biased by fitness.
func SpliceDNA(parentA, parentB *VolumetricNetwork, cfg SpliceConfig) *VolumetricNetwork {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	dnaA := ExtractDNA(parentA)
	dnaB := ExtractDNA(parentB)

	// Build position map of parentB's DNA for O(1) lookup
	bSigMap := make(map[string]LayerSignature, len(dnaB))
	for _, sig := range dnaB {
		bSigMap[evoLayerKey(sig.Z, sig.Y, sig.X, sig.L)] = sig
	}

	// Build position map of parentA's DNA
	aSigMap := make(map[string]LayerSignature, len(dnaA))
	for _, sig := range dnaA {
		aSigMap[evoLayerKey(sig.Z, sig.Y, sig.X, sig.L)] = sig
	}

	// Child starts as a deep clone of parentA
	child := cloneNetwork(parentA)

	for i, layerA := range parentA.Layers {
		if layerA.WeightStore == nil {
			continue
		}

		key := evoLayerKey(layerA.Z, layerA.Y, layerA.X, layerA.L)

		sigB, hasB := bSigMap[key]
		if !hasB || len(sigB.Weights) == 0 {
			continue // No counterpart in B — keep A's weights
		}

		sigA := aSigMap[key]
		similarity := CosineSimilarity(sigA, sigB)

		wA := parentA.Layers[i].WeightStore.Master
		wB := evoFindWeights(parentB, layerA.Z, layerA.Y, layerA.X, layerA.L)

		if wB == nil || len(wB) != len(wA) {
			continue // Dimension mismatch — skip
		}

		childW := child.Layers[i].WeightStore.Master

		switch cfg.CrossoverMode {
		case "point":
			// First SplitRatio fraction from A, remainder from B
			split := int(float64(len(wA)) * cfg.SplitRatio)
			if split > len(wA) {
				split = len(wA)
			}
			copy(childW[:split], wA[:split])
			copy(childW[split:], wB[split:])

		case "uniform":
			// Per-weight random selection, biased toward fitter parent
			threshold := float32(0.5)
			if cfg.FitnessA+cfg.FitnessB > 0 {
				threshold = float32(cfg.FitnessA / (cfg.FitnessA + cfg.FitnessB))
			}
			for j := range childW {
				if rng.Float32() < threshold {
					childW[j] = wA[j]
				} else {
					childW[j] = wB[j]
				}
			}

		default: // "blend"
			// Interpolate weights; alpha is modulated by similarity and fitness
			alpha := cfg.BlendAlpha
			if cfg.FitnessA+cfg.FitnessB > 0 {
				// Bias toward fitter parent, scaled by similarity
				// High similarity → blend freely; low similarity → favor fitter parent
				fitnessAlpha := float32(cfg.FitnessB / (cfg.FitnessA + cfg.FitnessB))
				alpha = fitnessAlpha * (0.5 + 0.5*similarity)
			}
			for j := range childW {
				childW[j] = wA[j]*(1-alpha) + wB[j]*alpha
			}
		}

		// Invalidate stale cached DType versions
		child.Layers[i].WeightStore.Versions = make(map[DType]any)
		child.Layers[i].WeightStore.GPUWeights = make(map[DType]any)
	}

	return child
}

// SpliceDNAWithReport performs a splice and returns a full diagnostic report.
// Use this when you want to inspect per-layer similarity scores or log blend stats.
func SpliceDNAWithReport(parentA, parentB *VolumetricNetwork, cfg SpliceConfig) SpliceResult {
	dnaA := ExtractDNA(parentA)
	dnaB := ExtractDNA(parentB)

	bMap := make(map[string]LayerSignature, len(dnaB))
	for _, sig := range dnaB {
		bMap[evoLayerKey(sig.Z, sig.Y, sig.X, sig.L)] = sig
	}

	sims := make(map[string]float32)
	blended := 0

	for _, sigA := range dnaA {
		key := evoLayerKey(sigA.Z, sigA.Y, sigA.X, sigA.L)
		if sigB, ok := bMap[key]; ok {
			sim := CosineSimilarity(sigA, sigB)
			sims[key] = sim
			if sim > 0 {
				blended++
			}
		}
	}

	child := SpliceDNA(parentA, parentB, cfg)

	return SpliceResult{
		Child:        child,
		ParentADNA:   dnaA,
		ParentBDNA:   dnaB,
		ChildDNA:     ExtractDNA(child),
		Similarities: sims,
		BlendedCount: blended,
	}
}

// ============================================================
// NEAT-style Topology Evolution
// ============================================================

// NEATConfig controls which mutations are enabled and their probabilities.
type NEATConfig struct {
	// Probabilities (0.0–1.0)
	WeightPerturbRate  float64 // Perturb each layer's weights with noise
	WeightPerturbScale float32 // Noise magnitude (default 0.05)
	NodeMutateRate     float64 // Swap a layer's type (and reinitialize its weights)
	ConnectionAddRate  float64 // Add a remote link (spatial hop) between two layers
	ConnectionDropRate float64 // Remove an existing remote link
	ActivationMutRate  float64 // Swap a layer's activation function
	LayerToggleRate    float64 // Enable/disable a dormant layer cell

	// AllowedLayerTypes for node mutation (nil = use defaults)
	AllowedLayerTypes []LayerType
	// DModel used when reinitializing a mutated layer's weights
	DModel int

	// Defaults for layer types that need extra config when reinitializing
	DefaultNumHeads    int // MHA: number of attention heads (default 4)
	DefaultInChannels  int // CNN/ConvTransposed: input channels (default 1)
	DefaultFilters     int // CNN/ConvTransposed: output filters (default 8)
	DefaultKernelSize  int // CNN/ConvTransposed: kernel size (default 3)
	DefaultVocabSize   int // Embedding: vocabulary size (default 256)
	DefaultNumClusters int // KMeans: number of clusters (default 8)

	Seed int64
}

// DefaultNEATConfig returns conservative mutation rates supporting all 19 layer types.
func DefaultNEATConfig(dModel int) NEATConfig {
	numHeads := 4
	if dModel < 4 {
		numHeads = 1
	}
	filters := dModel / 4
	if filters < 1 {
		filters = 1
	}
	return NEATConfig{
		WeightPerturbRate:  0.8,
		WeightPerturbScale: 0.05,
		NodeMutateRate:     0.1,
		ConnectionAddRate:  0.05,
		ConnectionDropRate: 0.02,
		ActivationMutRate:  0.1,
		LayerToggleRate:    0.02,
		DModel:             dModel,
		AllowedLayerTypes: []LayerType{
			LayerDense, LayerRNN, LayerLSTM, LayerRMSNorm, LayerSwiGLU,
			LayerMultiHeadAttention, LayerLayerNorm,
			LayerCNN1, LayerCNN2, LayerCNN3,
			LayerConvTransposed1D, LayerConvTransposed2D, LayerConvTransposed3D,
			LayerEmbedding, LayerKMeans,
			LayerSoftmax, LayerResidual,
		},
		DefaultNumHeads:    numHeads,
		DefaultInChannels:  1,
		DefaultFilters:     filters,
		DefaultKernelSize:  3,
		DefaultVocabSize:   256,
		DefaultNumClusters: 8,
		Seed:               time.Now().UnixNano(),
	}
}

// NEATMutate applies NEAT-style structural and weight mutations to a copy of n.
// The original network is never modified — a clone is returned.
//
// Mutation sequence per layer:
//  1. Weight perturbation  — add small Gaussian noise to Master weights
//  2. Activation mutation  — randomly swap the activation function
//  3. Node mutation        — change layer type, reinitialize weights
//  4. Layer toggle         — flip IsDisabled (activate dormant / silence active)
//
// Network-level mutations (applied once after per-layer pass):
//  5. Connection add       — insert a remote link (IsRemoteLink spatial hop)
//  6. Connection drop      — remove an existing remote link
func NEATMutate(n *VolumetricNetwork, cfg NEATConfig) *VolumetricNetwork {
	child := cloneNetwork(n)
	rng := rand.New(rand.NewSource(cfg.Seed))

	for i := range child.Layers {
		l := &child.Layers[i]

		// 1. Weight perturbation
		if rng.Float64() < cfg.WeightPerturbRate && l.WeightStore != nil {
			neatPerturbWeights(l.WeightStore.Master, cfg.WeightPerturbScale, rng)
			l.WeightStore.Versions = make(map[DType]any)
			l.WeightStore.GPUWeights = make(map[DType]any)
		}

		// 2. Activation mutation
		if rng.Float64() < cfg.ActivationMutRate {
			l.Activation = neatRandomActivation(rng)
		}

		// 3. Node (layer type) mutation
		if rng.Float64() < cfg.NodeMutateRate {
			newType := neatRandomLayerType(cfg.AllowedLayerTypes, l.Type, rng)
			if newType != l.Type {
				neatReinitLayer(child, i, newType, cfg, rng)
			}
		}

		// 4. Layer toggle
		if rng.Float64() < cfg.LayerToggleRate {
			l.IsDisabled = !l.IsDisabled
		}
	}

	// 5. Connection add
	if rng.Float64() < cfg.ConnectionAddRate && len(child.Layers) >= 2 {
		neatAddConnection(child, rng)
	}

	// 6. Connection drop
	if rng.Float64() < cfg.ConnectionDropRate {
		neatDropConnection(child, rng)
	}

	return child
}

// ============================================================
// NEAT Population
// ============================================================

// NEATPopulation manages a pool of networks evolving over generations.
type NEATPopulation struct {
	Networks  []*VolumetricNetwork
	Fitnesses []float64
	Config    NEATConfig
	rng       *rand.Rand
}

// NewNEATPopulation creates an initial population by mutating a seed network.
// Each member starts as a NEATMutate of the seed, giving diversity from day 0.
func NewNEATPopulation(seed *VolumetricNetwork, size int, cfg NEATConfig) *NEATPopulation {
	pop := &NEATPopulation{
		Networks:  make([]*VolumetricNetwork, size),
		Fitnesses: make([]float64, size),
		Config:    cfg,
		rng:       rand.New(rand.NewSource(cfg.Seed)),
	}
	for i := range pop.Networks {
		mutCfg := cfg
		mutCfg.Seed = pop.rng.Int63()
		pop.Networks[i] = NEATMutate(seed, mutCfg)
	}
	return pop
}

// Evolve runs one generation:
//  1. Evaluate all networks with fitnessFn (higher = better)
//  2. Sort by fitness descending
//  3. Top 25% survive as elites
//  4. Remaining slots filled with SpliceDNA(elite pair) + NEATMutate offspring
//
// fitnessFn should return a positive float64 (e.g., accuracy, reward, 1/loss).
func (p *NEATPopulation) Evolve(fitnessFn func(*VolumetricNetwork) float64) {
	for i, net := range p.Networks {
		p.Fitnesses[i] = fitnessFn(net)
	}

	p.sortByFitness()

	size := len(p.Networks)
	eliteCount := size / 4
	if eliteCount < 1 {
		eliteCount = 1
	}

	next := make([]*VolumetricNetwork, size)

	// Elites carry over unchanged
	for i := 0; i < eliteCount; i++ {
		next[i] = p.Networks[i]
	}

	// Offspring: splice two elites, then mutate
	for i := eliteCount; i < size; i++ {
		aIdx := p.rng.Intn(eliteCount)
		bIdx := p.rng.Intn(eliteCount)
		for bIdx == aIdx && eliteCount > 1 {
			bIdx = p.rng.Intn(eliteCount)
		}

		spliceCfg := DefaultSpliceConfig()
		spliceCfg.FitnessA = p.Fitnesses[aIdx]
		spliceCfg.FitnessB = p.Fitnesses[bIdx]

		child := SpliceDNA(p.Networks[aIdx], p.Networks[bIdx], spliceCfg)

		mutCfg := p.Config
		mutCfg.Seed = p.rng.Int63()
		next[i] = NEATMutate(child, mutCfg)
	}

	p.Networks = next
}

// Best returns the highest-fitness network from the last Evolve call.
func (p *NEATPopulation) Best() *VolumetricNetwork {
	if len(p.Networks) == 0 {
		return nil
	}
	return p.Networks[0]
}

// BestFitness returns the fitness score of the top network.
func (p *NEATPopulation) BestFitness() float64 {
	if len(p.Fitnesses) == 0 {
		return 0
	}
	return p.Fitnesses[0]
}

// Summary prints a one-line diagnostic for the population.
func (p *NEATPopulation) Summary(generation int) string {
	if len(p.Fitnesses) == 0 {
		return fmt.Sprintf("Gen %d: empty population", generation)
	}
	best := p.Fitnesses[0]
	worst := p.Fitnesses[len(p.Fitnesses)-1]
	var sum float64
	for _, f := range p.Fitnesses {
		sum += f
	}
	avg := sum / float64(len(p.Fitnesses))
	return fmt.Sprintf("Gen %d | best=%.4f  avg=%.4f  worst=%.4f  pop=%d",
		generation, best, avg, worst, len(p.Networks))
}

// sortByFitness sorts both slices descending by fitness.
func (p *NEATPopulation) sortByFitness() {
	n := len(p.Networks)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-1-i; j++ {
			if p.Fitnesses[j] < p.Fitnesses[j+1] {
				p.Networks[j], p.Networks[j+1] = p.Networks[j+1], p.Networks[j]
				p.Fitnesses[j], p.Fitnesses[j+1] = p.Fitnesses[j+1], p.Fitnesses[j]
			}
		}
	}
}

// ============================================================
// Internal helpers
// ============================================================

// evoLayerKey returns the position string used as map keys throughout this file.
// Matches the format used by dna.go ("z,y,x,l").
func evoLayerKey(z, y, x, l int) string {
	return fmt.Sprintf("%d,%d,%d,%d", z, y, x, l)
}

// evoFindWeights returns the Master weights of the layer at (z,y,x,l) in n.
func evoFindWeights(n *VolumetricNetwork, z, y, x, l int) []float32 {
	layer := n.GetLayer(z, y, x, l)
	if layer == nil || layer.WeightStore == nil {
		return nil
	}
	return layer.WeightStore.Master
}

// cloneNetwork deep-copies a VolumetricNetwork including all weights.
// GPU state is intentionally not carried over — child starts CPU-resident.
func cloneNetwork(src *VolumetricNetwork) *VolumetricNetwork {
	dst := NewVolumetricNetwork(src.Depth, src.Rows, src.Cols, src.LayersPerCell)
	dst.UseTiling = src.UseTiling

	for i, sl := range src.Layers {
		dl := &dst.Layers[i]

		dl.Type = sl.Type
		dl.Activation = sl.Activation
		dl.DType = sl.DType
		dl.IsDisabled = sl.IsDisabled

		dl.InputHeight = sl.InputHeight
		dl.InputWidth = sl.InputWidth
		dl.InputDepth = sl.InputDepth
		dl.OutputHeight = sl.OutputHeight
		dl.OutputWidth = sl.OutputWidth
		dl.OutputDepth = sl.OutputDepth
		dl.InputChannels = sl.InputChannels
		dl.Filters = sl.Filters
		dl.KernelSize = sl.KernelSize
		dl.Stride = sl.Stride
		dl.Padding = sl.Padding
		dl.OutputPadding = sl.OutputPadding

		dl.NumHeads = sl.NumHeads
		dl.NumKVHeads = sl.NumKVHeads
		dl.HeadDim = sl.HeadDim
		dl.DModel = sl.DModel
		dl.SeqLength = sl.SeqLength
		dl.RoPEFreqBase = sl.RoPEFreqBase

		dl.VocabSize = sl.VocabSize
		dl.EmbeddingDim = sl.EmbeddingDim

		dl.NumClusters = sl.NumClusters
		dl.KMeansTemperature = sl.KMeansTemperature
		dl.KMeansOutputMode = sl.KMeansOutputMode

		dl.SoftmaxType = sl.SoftmaxType
		dl.Temperature = sl.Temperature
		dl.SoftmaxRows = sl.SoftmaxRows
		dl.SoftmaxCols = sl.SoftmaxCols
		dl.EntmaxAlpha = sl.EntmaxAlpha
		dl.GumbelNoise = sl.GumbelNoise

		dl.CombineMode = sl.CombineMode
		dl.IsRemoteLink = sl.IsRemoteLink
		dl.TargetZ = sl.TargetZ
		dl.TargetY = sl.TargetY
		dl.TargetX = sl.TargetX
		dl.TargetL = sl.TargetL

		// Deep copy parallel branches (structural only, no GPU state)
		if len(sl.ParallelBranches) > 0 {
			dl.ParallelBranches = make([]VolumetricLayer, len(sl.ParallelBranches))
			copy(dl.ParallelBranches, sl.ParallelBranches)
		}

		// Deep copy sequential layers
		if len(sl.SequentialLayers) > 0 {
			dl.SequentialLayers = make([]VolumetricLayer, len(sl.SequentialLayers))
			copy(dl.SequentialLayers, sl.SequentialLayers)
		}

		// Deep copy mask
		if len(sl.Mask) > 0 {
			dl.Mask = make([]bool, len(sl.Mask))
			copy(dl.Mask, sl.Mask)
		}

		// Deep copy hierarchy levels
		if len(sl.HierarchyLevels) > 0 {
			dl.HierarchyLevels = make([]int, len(sl.HierarchyLevels))
			copy(dl.HierarchyLevels, sl.HierarchyLevels)
		}

		// Deep copy weights (Master only; GPU buffers are intentionally dropped)
		if sl.WeightStore != nil {
			dl.WeightStore = NewWeightStore(len(sl.WeightStore.Master))
			copy(dl.WeightStore.Master, sl.WeightStore.Master)
			dl.WeightStore.Scale = sl.WeightStore.Scale
		}
	}

	return dst
}

func neatPerturbWeights(weights []float32, scale float32, rng *rand.Rand) {
	for i := range weights {
		weights[i] += (rng.Float32()*2 - 1) * scale
	}
}

func neatRandomActivation(rng *rand.Rand) ActivationType {
	acts := []ActivationType{
		ActivationReLU, ActivationSilu, ActivationGELU,
		ActivationTanh, ActivationSigmoid, ActivationLinear,
	}
	return acts[rng.Intn(len(acts))]
}

func neatRandomLayerType(allowed []LayerType, current LayerType, rng *rand.Rand) LayerType {
	candidates := make([]LayerType, 0, len(allowed))
	for _, t := range allowed {
		if t != current {
			candidates = append(candidates, t)
		}
	}
	if len(candidates) == 0 {
		return current
	}
	return candidates[rng.Intn(len(candidates))]
}

// neatReinitLayer replaces a layer's type and reinitializes its WeightStore.
// Handles all 19 LayerType values using config defaults for type-specific params.
func neatReinitLayer(n *VolumetricNetwork, idx int, newType LayerType, cfg NEATConfig, rng *rand.Rand) {
	l := &n.Layers[idx]
	dModel := cfg.DModel
	l.Type = newType
	l.InputHeight = dModel
	l.OutputHeight = dModel

	var wCount int

	switch newType {

	// ── Parameterless / structural layers ─────────────────────────────────
	case LayerSoftmax:
		// No weights — just set type, clear any stale store
		l.WeightStore = nil
		return

	case LayerResidual:
		// No weights
		l.WeightStore = nil
		return

	case LayerParallel, LayerSequential:
		// Structural containers — keep existing branches/children intact,
		// only perturb weights if present; don't blow away the structure.
		return

	// ── Dense ─────────────────────────────────────────────────────────────
	case LayerDense:
		wCount = dModel * dModel

	// ── Recurrent ─────────────────────────────────────────────────────────
	case LayerRNN:
		// Wx (in→hidden) + Wh (hidden→hidden) + bias
		l.InputChannels = dModel
		l.SeqLength = 1
		wCount = dModel*dModel + dModel*dModel + dModel

	case LayerLSTM:
		// 4 gates × (Wx + Wh + bias)
		l.InputChannels = dModel
		l.SeqLength = 1
		gate := dModel*dModel + dModel*dModel + dModel
		wCount = 4 * gate

	// ── Feed-forward / gating ─────────────────────────────────────────────
	case LayerSwiGLU:
		inter := dModel * 2
		l.OutputHeight = inter
		// gate + up + down projections + biases
		wCount = dModel*inter*3 + inter*2 + dModel

	// ── Normalization ─────────────────────────────────────────────────────
	case LayerRMSNorm:
		// Scale vector only
		wCount = dModel

	case LayerLayerNorm:
		// Gamma + Beta
		wCount = dModel * 2

	// ── Attention ─────────────────────────────────────────────────────────
	case LayerMultiHeadAttention:
		numHeads := cfg.DefaultNumHeads
		if numHeads < 1 {
			numHeads = 1
		}
		// Clamp so headDim is always a whole number
		for dModel%numHeads != 0 && numHeads > 1 {
			numHeads--
		}
		headDim := dModel / numHeads
		kv := numHeads * headDim // = dModel

		l.DModel = dModel
		l.NumHeads = numHeads
		l.NumKVHeads = numHeads
		l.HeadDim = headDim
		l.SeqLength = 1

		// Q·W_Q + K·W_K + V·W_V + O·W_O + biases (matches InitMHACell)
		wCount = 2*dModel*dModel + 2*dModel*kv + 2*dModel + 2*kv

	// ── Convolutional ─────────────────────────────────────────────────────
	case LayerCNN1, LayerCNN2:
		inCh := cfg.DefaultInChannels
		filters := cfg.DefaultFilters
		kSize := cfg.DefaultKernelSize
		if inCh < 1 {
			inCh = 1
		}
		if filters < 1 {
			filters = 1
		}
		if kSize < 1 {
			kSize = 3
		}
		l.InputChannels = inCh
		l.Filters = filters
		l.KernelSize = kSize
		// Matches InitCNNCell formula for CNN1 and CNN2
		wCount = filters * inCh * kSize * kSize

	case LayerCNN3:
		inCh := cfg.DefaultInChannels
		filters := cfg.DefaultFilters
		kSize := cfg.DefaultKernelSize
		if inCh < 1 {
			inCh = 1
		}
		if filters < 1 {
			filters = 1
		}
		if kSize < 1 {
			kSize = 3
		}
		l.InputChannels = inCh
		l.Filters = filters
		l.KernelSize = kSize
		wCount = filters * inCh * kSize * kSize * kSize

	// ── Transposed convolution ─────────────────────────────────────────────
	case LayerConvTransposed1D, LayerConvTransposed2D:
		inCh := cfg.DefaultInChannels
		filters := cfg.DefaultFilters
		kSize := cfg.DefaultKernelSize
		if inCh < 1 {
			inCh = 1
		}
		if filters < 1 {
			filters = 1
		}
		if kSize < 1 {
			kSize = 3
		}
		l.InputChannels = inCh
		l.Filters = filters
		l.KernelSize = kSize
		// Matches InitConvTransposedCell formula
		wCount = inCh * filters * kSize * kSize

	case LayerConvTransposed3D:
		inCh := cfg.DefaultInChannels
		filters := cfg.DefaultFilters
		kSize := cfg.DefaultKernelSize
		if inCh < 1 {
			inCh = 1
		}
		if filters < 1 {
			filters = 1
		}
		if kSize < 1 {
			kSize = 3
		}
		l.InputChannels = inCh
		l.Filters = filters
		l.KernelSize = kSize
		wCount = inCh * filters * kSize * kSize * kSize

	// ── Embedding ─────────────────────────────────────────────────────────
	case LayerEmbedding:
		vocabSize := cfg.DefaultVocabSize
		if vocabSize < 1 {
			vocabSize = 256
		}
		l.VocabSize = vocabSize
		l.EmbeddingDim = dModel
		l.InputHeight = vocabSize
		l.OutputHeight = dModel
		wCount = vocabSize * dModel

	// ── KMeans ────────────────────────────────────────────────────────────
	case LayerKMeans:
		k := cfg.DefaultNumClusters
		if k < 1 {
			k = 8
		}
		l.NumClusters = k
		l.KMeansTemperature = 1.0
		l.KMeansOutputMode = "probabilities"
		l.InputHeight = dModel
		l.OutputHeight = k
		wCount = k * dModel

	default:
		// Unknown future layer type — safe fallback
		wCount = dModel * dModel
	}

	l.WeightStore = NewWeightStore(wCount)
	l.WeightStore.Randomize(rng.Int63(), 0.05)
}

// neatAddConnection picks two random layers and adds a remote link from src → dst
// by appending an IsRemoteLink branch to src's ParallelBranches.
func neatAddConnection(n *VolumetricNetwork, rng *rand.Rand) {
	if len(n.Layers) < 2 {
		return
	}

	aIdx := rng.Intn(len(n.Layers))
	bIdx := rng.Intn(len(n.Layers))
	for bIdx == aIdx {
		bIdx = rng.Intn(len(n.Layers))
	}

	src := &n.Layers[aIdx]
	dst := &n.Layers[bIdx]

	remoteLink := VolumetricLayer{
		Network:      n,
		IsRemoteLink: true,
		TargetZ:      dst.Z,
		TargetY:      dst.Y,
		TargetX:      dst.X,
		TargetL:      dst.L,
	}

	src.ParallelBranches = append(src.ParallelBranches, remoteLink)
	if src.CombineMode == "" {
		src.CombineMode = "add"
	}
}

// neatDropConnection removes a random remote link from the network.
func neatDropConnection(n *VolumetricNetwork, rng *rand.Rand) {
	// Collect all layers that have at least one remote branch
	type candidate struct {
		layerIdx  int
		branchIdx int
	}
	var candidates []candidate

	for i := range n.Layers {
		for j, b := range n.Layers[i].ParallelBranches {
			if b.IsRemoteLink {
				candidates = append(candidates, candidate{i, j})
			}
		}
	}

	if len(candidates) == 0 {
		return
	}

	pick := candidates[rng.Intn(len(candidates))]
	l := &n.Layers[pick.layerIdx]
	l.ParallelBranches = append(
		l.ParallelBranches[:pick.branchIdx],
		l.ParallelBranches[pick.branchIdx+1:]...,
	)
}
