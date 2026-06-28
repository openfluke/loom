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
	"github.com/openfluke/loom/poly/accel"
)

// Handle-based management maps
var networks = make(map[int64]*poly.VolumetricNetwork)
var networkNextID int64 = 1

// State containers for polymorphic types
type stepContainer struct {
	State    interface{}
	DType    poly.DType
	Borrowed bool // GPU buffer owned by poly (e.g. layer weight); LoomFreeGPUBuffer must not Destroy
}

type tweenContainer struct {
	State interface{}
	DType poly.DType
}

var stepStates = make(map[int64]*stepContainer)
var stepNextID int64 = 1

var tweenStates = make(map[int64]*tweenContainer)
var tweenNextID int64 = 1

var networkMu sync.RWMutex

var tokenizers = make(map[int64]*poly.Tokenizer)
var tokenizerNextID int64 = 1

var tensors = make(map[int64]interface{}) // interface{} to handle poly.Tensor[float32], etc.
var tensorNextID int64 = 1


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

// Helper: Convert Handle to StepState container
func getStepContainer(handle int64) (*stepContainer, bool) {
	networkMu.RLock()
	defer networkMu.RUnlock()
	s, ok := stepStates[handle]
	return s, ok
}

var transformers = make(map[int64]interface{})
var transformerNextID int64 = 1

var entityTransformers = make(map[int64]*poly.EntityTransformer)
var entityTransformerNextID int64 = 1

func getTransformer(handle int64) (interface{}, bool) {
	networkMu.RLock()
	defer networkMu.RUnlock()
	tr, ok := transformers[handle]
	return tr, ok
}

func getEntityTransformer(handle int64) (*poly.EntityTransformer, bool) {
	networkMu.RLock()
	defer networkMu.RUnlock()
	et, ok := entityTransformers[handle]
	return et, ok
}

func registerTransformer(tr interface{}, _ poly.DType) int64 {
	networkMu.Lock()
	id := transformerNextID
	transformerNextID++
	transformers[id] = tr
	networkMu.Unlock()
	return id
}

func getTensor(handle int64) (interface{}, bool) {
	networkMu.RLock()
	defer networkMu.RUnlock()
	t, ok := tensors[handle]
	return t, ok
}



var neatPopulations = make(map[int64]*poly.NEATPopulation)
var neatPopNextID int64 = 1

var accelRegistries = make(map[int64]*accel.Registry)
var accelRegistryNextID int64 = 1

var entityFiles = make(map[int64]*poly.EntityFile)
var entityFileNextID int64 = 1

func getNEATPopulation(handle int64) (*poly.NEATPopulation, bool) {
	networkMu.RLock()
	defer networkMu.RUnlock()
	p, ok := neatPopulations[handle]
	return p, ok
}

func getAccelRegistry(handle int64) (*accel.Registry, bool) {
	networkMu.RLock()
	defer networkMu.RUnlock()
	r, ok := accelRegistries[handle]
	return r, ok
}

func getEntityFile(handle int64) (*poly.EntityFile, bool) {
	networkMu.RLock()
	defer networkMu.RUnlock()
	ef, ok := entityFiles[handle]
	return ef, ok
}

// Helper: Convert Handle to TweenState Container
func getTweenContainer(handle int64) (*tweenContainer, bool) {
	networkMu.RLock()
	defer networkMu.RUnlock()
	s, ok := tweenStates[handle]
	return s, ok
}
