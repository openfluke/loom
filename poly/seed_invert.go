package poly

import "fmt"

// FindLayerSeedForWeights searches for a layer_seed whose He-init matches master exactly.
// Returns false when no exact seed is found within the search budget.
func FindLayerSeedForWeights(master []float32, inputSize int, hints ...uint64) (uint64, bool) {
	if len(master) == 0 {
		return 0, false
	}
	if inputSize <= 0 {
		inputSize = 1
	}
	try := make([]uint64, 0, len(hints)+4)
	try = append(try, hints...)
	try = append(try, SeedFrom("invert-hint", len(master), weightSliceFingerprint(master)))
	best := try[0]
	bestMiss := weightSeedMismatches(master, inputSize, best)
	if bestMiss == 0 {
		return best, true
	}
	rng := NewSeedRNG(SeedFrom("invert-search", best, uint64(len(master))))
	const trials = 50_000
	for t := 0; t < trials; t++ {
		candidate := best
		if t%50_000 == 0 && t > 0 {
			candidate = rng.Uint64()
		} else {
			candidate = mutateLayerSeed(candidate, rng.Uint64())
		}
		miss := weightSeedMismatches(master, inputSize, candidate)
		if miss < bestMiss {
			bestMiss = miss
			best = candidate
			if bestMiss == 0 {
				return best, true
			}
		}
	}
	return best, bestMiss == 0
}

func weightSeedMismatches(master []float32, inputSize int, seed uint64) int {
	tmp := NewWeightStore(len(master))
	InitWeightStoreHeSeeded(tmp, inputSize, seed)
	miss := 0
	for i := range master {
		if master[i] != tmp.Master[i] {
			miss++
		}
	}
	return miss
}

func weightSliceFingerprint(master []float32) uint64 {
	ws := NewWeightStore(len(master))
	copy(ws.Master, master)
	return weightStoreFingerprint(ws)
}

func mutateLayerSeed(seed, noise uint64) uint64 {
	switch noise % 3 {
	case 0:
		bit := (noise / 3) % 64
		return seed ^ (1 << bit)
	case 1:
		return seed + (noise>>6) + 1
	default:
		return seed ^ noise
	}
}

// LayerSeedFromTrainedLayer tries exact weights→layer_seed for one built layer.
func LayerSeedFromTrainedLayer(l *VolumetricLayer, hint uint64) (uint64, bool, error) {
	if l == nil || l.WeightStore == nil || len(l.WeightStore.Master) == 0 {
		return 0, false, fmt.Errorf("invert: empty layer")
	}
	in := seedLayerInputSize(l)
	seed, ok := FindLayerSeedForWeights(l.WeightStore.Master, in, hint)
	return seed, ok, nil
}
