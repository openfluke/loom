package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"fmt"
	"sync"

	"github.com/openfluke/loom/poly"
)

// Handle-based management maps
var networks = make(map[int64]*poly.VolumetricNetwork)
var networkNextID int64 = 1

// State containers for polymorphic types
type systolicContainer struct {
	State interface{}
	DType poly.DType
}

type targetPropContainer struct {
	State interface{}
	DType poly.DType
}

var systolicStates = make(map[int64]*systolicContainer)
var systolicNextID int64 = 1

var targetPropStates = make(map[int64]*targetPropContainer)
var targetPropNextID int64 = 1

var networkMu sync.RWMutex

var tokenizers = make(map[int64]*poly.Tokenizer)
var tokenizerNextID int64 = 1

// Helper: Error to C String
func errJSON(msg string) *C.char {
	return C.CString(fmt.Sprintf(`{"error": "%s"}`, msg))
}

// Helper: Convert Handle to Network
func getNetwork(handle int64) (*poly.VolumetricNetwork, bool) {
	networkMu.RLock()
	defer networkMu.RUnlock()
	n, ok := networks[handle]
	return n, ok
}

// Helper: Convert Handle to SystolicState Container
func getSystolicContainer(handle int64) (*systolicContainer, bool) {
	networkMu.RLock()
	defer networkMu.RUnlock()
	s, ok := systolicStates[handle]
	return s, ok
}

var (
	transformers      = make(map[int64]interface{}) // Using interface{} to handle multiple numeric types
	transfomrerNextID int64 = 1
)

func getTransformer(handle int64) (interface{}, bool) {
	networkMu.RLock()
	defer networkMu.RUnlock()
	tr, ok := transformers[handle]
	return tr, ok
}

var neatPopulations = make(map[int64]*poly.NEATPopulation)
var neatPopNextID int64 = 1

func getNEATPopulation(handle int64) (*poly.NEATPopulation, bool) {
	networkMu.RLock()
	defer networkMu.RUnlock()
	p, ok := neatPopulations[handle]
	return p, ok
}

// Helper: Convert Handle to TargetPropState Container
func getTargetPropContainer(handle int64) (*targetPropContainer, bool) {
	networkMu.RLock()
	defer networkMu.RUnlock()
	s, ok := targetPropStates[handle]
	return s, ok
}
