package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"encoding/json"

	"github.com/openfluke/loom/poly"
)

// ─────────────────────────────────────────────────────────────────────────────
// DNA Splice / Crossover
// ─────────────────────────────────────────────────────────────────────────────

//export LoomDefaultSpliceConfig
func LoomDefaultSpliceConfig() *C.char {
	cfg := poly.DefaultSpliceConfig()
	data, _ := json.Marshal(cfg)
	return C.CString(string(data))
}

//export LoomSpliceDNA
func LoomSpliceDNA(handleA C.longlong, handleB C.longlong, cfgJSON *C.char) C.longlong {
	a, ok := getNetwork(int64(handleA))
	if !ok {
		return -1
	}
	b, ok := getNetwork(int64(handleB))
	if !ok {
		return -1
	}

	var cfg poly.SpliceConfig
	if err := json.Unmarshal([]byte(C.GoString(cfgJSON)), &cfg); err != nil {
		cfg = poly.DefaultSpliceConfig()
	}

	child := poly.SpliceDNA(a, b, cfg)

	networkMu.Lock()
	id := networkNextID
	networkNextID++
	networks[id] = child
	networkMu.Unlock()

	return C.longlong(id)
}

//export LoomSpliceDNAWithReport
func LoomSpliceDNAWithReport(handleA C.longlong, handleB C.longlong, cfgJSON *C.char) *C.char {
	a, ok := getNetwork(int64(handleA))
	if !ok {
		return errJSON("invalid network handle A")
	}
	b, ok := getNetwork(int64(handleB))
	if !ok {
		return errJSON("invalid network handle B")
	}

	var cfg poly.SpliceConfig
	if err := json.Unmarshal([]byte(C.GoString(cfgJSON)), &cfg); err != nil {
		cfg = poly.DefaultSpliceConfig()
	}

	result := poly.SpliceDNAWithReport(a, b, cfg)

	networkMu.Lock()
	childID := networkNextID
	networkNextID++
	networks[childID] = result.Child
	networkMu.Unlock()

	type reportOut struct {
		ChildHandle  int64              `json:"child_handle"`
		ParentADNA   poly.NetworkDNA    `json:"parent_a_dna"`
		ParentBDNA   poly.NetworkDNA    `json:"parent_b_dna"`
		ChildDNA     poly.NetworkDNA    `json:"child_dna"`
		Similarities map[string]float32 `json:"similarities"`
		BlendedCount int                `json:"blended_count"`
	}
	out := reportOut{
		ChildHandle:  childID,
		ParentADNA:   result.ParentADNA,
		ParentBDNA:   result.ParentBDNA,
		ChildDNA:     result.ChildDNA,
		Similarities: result.Similarities,
		BlendedCount: result.BlendedCount,
	}
	data, _ := json.Marshal(out)
	return C.CString(string(data))
}

// ─────────────────────────────────────────────────────────────────────────────
// NEAT Mutation
// ─────────────────────────────────────────────────────────────────────────────

//export LoomDefaultNEATConfig
func LoomDefaultNEATConfig(dModel C.int) *C.char {
	cfg := poly.DefaultNEATConfig(int(dModel))
	data, _ := json.Marshal(cfg)
	return C.CString(string(data))
}

//export LoomNEATMutate
func LoomNEATMutate(handle C.longlong, cfgJSON *C.char) C.longlong {
	n, ok := getNetwork(int64(handle))
	if !ok {
		return -1
	}

	var cfg poly.NEATConfig
	if err := json.Unmarshal([]byte(C.GoString(cfgJSON)), &cfg); err != nil {
		return -1
	}

	child := poly.NEATMutate(n, cfg)

	networkMu.Lock()
	id := networkNextID
	networkNextID++
	networks[id] = child
	networkMu.Unlock()

	return C.longlong(id)
}

// ─────────────────────────────────────────────────────────────────────────────
// NEAT Population
// ─────────────────────────────────────────────────────────────────────────────

//export LoomNewNEATPopulation
func LoomNewNEATPopulation(seedHandle C.longlong, size C.int, cfgJSON *C.char) C.longlong {
	seed, ok := getNetwork(int64(seedHandle))
	if !ok {
		return -1
	}

	var cfg poly.NEATConfig
	if err := json.Unmarshal([]byte(C.GoString(cfgJSON)), &cfg); err != nil {
		return -1
	}

	pop := poly.NewNEATPopulation(seed, int(size), cfg)

	networkMu.Lock()
	id := neatPopNextID
	neatPopNextID++
	neatPopulations[id] = pop
	networkMu.Unlock()

	return C.longlong(id)
}

//export LoomNEATPopulationSize
func LoomNEATPopulationSize(popHandle C.longlong) C.int {
	pop, ok := getNEATPopulation(int64(popHandle))
	if !ok {
		return -1
	}
	return C.int(len(pop.Networks))
}

// LoomNEATPopulationGetNetwork returns a network handle pointing into the
// population at the given index. The handle shares the underlying pointer —
// do NOT call LoomFreeNetwork on it while the population is alive.
//
//export LoomNEATPopulationGetNetwork
func LoomNEATPopulationGetNetwork(popHandle C.longlong, index C.int) C.longlong {
	pop, ok := getNEATPopulation(int64(popHandle))
	if !ok {
		return -1
	}
	idx := int(index)
	if idx < 0 || idx >= len(pop.Networks) {
		return -1
	}

	networkMu.Lock()
	id := networkNextID
	networkNextID++
	networks[id] = pop.Networks[idx]
	networkMu.Unlock()

	return C.longlong(id)
}

//export LoomNEATPopulationEvolveWithFitnesses
func LoomNEATPopulationEvolveWithFitnesses(popHandle C.longlong, fitnessesJSON *C.char) *C.char {
	pop, ok := getNEATPopulation(int64(popHandle))
	if !ok {
		return errJSON("invalid population handle")
	}

	var fitnesses []float64
	if err := json.Unmarshal([]byte(C.GoString(fitnessesJSON)), &fitnesses); err != nil {
		return errJSON("invalid fitnesses JSON")
	}

	idx := 0
	pop.Evolve(func(_ *poly.VolumetricNetwork) float64 {
		if idx < len(fitnesses) {
			f := fitnesses[idx]
			idx++
			return f
		}
		return 0
	})

	return C.CString(`{"status": "ok"}`)
}

//export LoomNEATPopulationBest
func LoomNEATPopulationBest(popHandle C.longlong) C.longlong {
	pop, ok := getNEATPopulation(int64(popHandle))
	if !ok {
		return -1
	}
	best := pop.Best()
	if best == nil {
		return -1
	}

	networkMu.Lock()
	id := networkNextID
	networkNextID++
	networks[id] = best
	networkMu.Unlock()

	return C.longlong(id)
}

//export LoomNEATPopulationBestFitness
func LoomNEATPopulationBestFitness(popHandle C.longlong) C.double {
	pop, ok := getNEATPopulation(int64(popHandle))
	if !ok {
		return C.double(-1)
	}
	return C.double(pop.BestFitness())
}

//export LoomNEATPopulationSummary
func LoomNEATPopulationSummary(popHandle C.longlong, generation C.int) *C.char {
	pop, ok := getNEATPopulation(int64(popHandle))
	if !ok {
		return C.CString("error: invalid population handle")
	}
	return C.CString(pop.Summary(int(generation)))
}

//export LoomFreeNEATPopulation
func LoomFreeNEATPopulation(popHandle C.longlong) {
	networkMu.Lock()
	delete(neatPopulations, int64(popHandle))
	networkMu.Unlock()
}
