package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"encoding/json"
	"time"
	"unsafe"

	"github.com/openfluke/loom/poly"
)

//export LoomCreateStepState
func LoomCreateStepState(networkHandle C.longlong, dtype C.int) C.longlong {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return -1 }

	dt := poly.DType(dtype)
	var state interface{}

	switch dt {
	case poly.DTypeFloat64:
		state = poly.NewStepState[float64](n)
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		state = poly.NewStepState[float32](n)
	case poly.DTypeInt64:
		state = poly.NewStepState[int64](n)
	case poly.DTypeInt32:
		state = poly.NewStepState[int32](n)
	case poly.DTypeInt16:
		state = poly.NewStepState[int16](n)
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		state = poly.NewStepState[int8](n)
	case poly.DTypeUint64:
		state = poly.NewStepState[uint64](n)
	case poly.DTypeUint32:
		state = poly.NewStepState[uint32](n)
	case poly.DTypeUint16:
		state = poly.NewStepState[uint16](n)
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		state = poly.NewStepState[uint8](n)
	default:
		state = poly.NewStepState[float32](n)
		dt = poly.DTypeFloat32
	}

	networkMu.Lock()
	id := stepNextID
	stepNextID++
	stepStates[id] = &stepContainer{State: state, DType: dt}
	networkMu.Unlock()

	return C.longlong(id)
}

//export LoomFreeStepState
func LoomFreeStepState(handle C.longlong) {
	networkMu.Lock()
	delete(stepStates, int64(handle))
	networkMu.Unlock()
}

//export LoomSetInput
func LoomSetInput(stateHandle C.longlong, data *C.float, length C.int) {
	c, ok := getStepContainer(int64(stateHandle))
	if !ok { return }

	ptr := unsafe.Pointer(data)
	slice := (*[1 << 30]float32)(ptr)[:length:length]

	l := int(length)
	switch c.DType {
	case poly.DTypeFloat64:
		st := c.State.(*poly.StepState[float64])
		f := make([]float64, l)
		for i, v := range slice { f[i] = float64(v) }
		st.SetInput(poly.NewTensorFromSlice(f, 1, l))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		st := c.State.(*poly.StepState[float32])
		ts := make([]float32, l)
		copy(ts, slice)
		st.SetInput(poly.NewTensorFromSlice(ts, 1, l))
	case poly.DTypeInt64:
		st := c.State.(*poly.StepState[int64])
		ts := make([]int64, l)
		for i, v := range slice { ts[i] = int64(v) }
		st.SetInput(poly.NewTensorFromSlice(ts, 1, l))
	case poly.DTypeInt32:
		st := c.State.(*poly.StepState[int32])
		ts := make([]int32, l)
		for i, v := range slice { ts[i] = int32(v) }
		st.SetInput(poly.NewTensorFromSlice(ts, 1, l))
	case poly.DTypeInt16:
		st := c.State.(*poly.StepState[int16])
		ts := make([]int16, l)
		for i, v := range slice { ts[i] = int16(v) }
		st.SetInput(poly.NewTensorFromSlice(ts, 1, l))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		st := c.State.(*poly.StepState[int8])
		ts := make([]int8, l)
		for i, v := range slice { ts[i] = int8(v) }
		st.SetInput(poly.NewTensorFromSlice(ts, 1, l))
	case poly.DTypeUint64:
		st := c.State.(*poly.StepState[uint64])
		ts := make([]uint64, l)
		for i, v := range slice { ts[i] = uint64(v) }
		st.SetInput(poly.NewTensorFromSlice(ts, 1, l))
	case poly.DTypeUint32:
		st := c.State.(*poly.StepState[uint32])
		ts := make([]uint32, l)
		for i, v := range slice { ts[i] = uint32(v) }
		st.SetInput(poly.NewTensorFromSlice(ts, 1, l))
	case poly.DTypeUint16:
		st := c.State.(*poly.StepState[uint16])
		ts := make([]uint16, l)
		for i, v := range slice { ts[i] = uint16(v) }
		st.SetInput(poly.NewTensorFromSlice(ts, 1, l))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		st := c.State.(*poly.StepState[uint8])
		ts := make([]uint8, l)
		for i, v := range slice { ts[i] = uint8(v) }
		st.SetInput(poly.NewTensorFromSlice(ts, 1, l))
	}
}

//export LoomStep
func LoomStep(networkHandle C.longlong, stateHandle C.longlong, captureHistory C.int) C.longlong {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return -1 }
	c, ok := getStepContainer(int64(stateHandle))
	if !ok { return -1 }

	var duration int64
	switch c.DType {
	case poly.DTypeFloat64:
		duration = poly.StepForward(n, c.State.(*poly.StepState[float64]), captureHistory != 0).Nanoseconds()
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		duration = poly.StepForward(n, c.State.(*poly.StepState[float32]), captureHistory != 0).Nanoseconds()
	case poly.DTypeInt64:
		duration = poly.StepForward(n, c.State.(*poly.StepState[int64]), captureHistory != 0).Nanoseconds()
	case poly.DTypeInt32:
		duration = poly.StepForward(n, c.State.(*poly.StepState[int32]), captureHistory != 0).Nanoseconds()
	case poly.DTypeInt16:
		duration = poly.StepForward(n, c.State.(*poly.StepState[int16]), captureHistory != 0).Nanoseconds()
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		duration = poly.StepForward(n, c.State.(*poly.StepState[int8]), captureHistory != 0).Nanoseconds()
	case poly.DTypeUint64:
		duration = poly.StepForward(n, c.State.(*poly.StepState[uint64]), captureHistory != 0).Nanoseconds()
	case poly.DTypeUint32:
		duration = poly.StepForward(n, c.State.(*poly.StepState[uint32]), captureHistory != 0).Nanoseconds()
	case poly.DTypeUint16:
		duration = poly.StepForward(n, c.State.(*poly.StepState[uint16]), captureHistory != 0).Nanoseconds()
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		duration = poly.StepForward(n, c.State.(*poly.StepState[uint8]), captureHistory != 0).Nanoseconds()
	}
	return C.longlong(duration)
}

//export LoomGetOutput
func LoomGetOutput(stateHandle C.longlong, layerIdx C.int) *C.char {
	c, ok := getStepContainer(int64(stateHandle))
	if !ok { return errJSON("invalid state handle") }

	networkMu.RLock()
	defer networkMu.RUnlock()
	
	marshalOutput := func(data any) *C.char {
		res, _ := json.Marshal(data)
		return C.CString(string(res))
	}

	switch c.DType {
	case poly.DTypeFloat64:
		s := c.State.(*poly.StepState[float64])
		if int(layerIdx) < 0 || int(layerIdx) >= len(s.LayerData) { return errJSON("oob") }
		if s.LayerData[layerIdx] == nil { return errJSON("no output") }
		return marshalOutput(s.LayerData[layerIdx].Data)
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		s := c.State.(*poly.StepState[float32])
		if int(layerIdx) < 0 || int(layerIdx) >= len(s.LayerData) { return errJSON("oob") }
		if s.LayerData[layerIdx] == nil { return errJSON("no output") }
		return marshalOutput(s.LayerData[layerIdx].Data)
	case poly.DTypeInt64:
		s := c.State.(*poly.StepState[int64])
		if int(layerIdx) < 0 || int(layerIdx) >= len(s.LayerData) { return errJSON("oob") }
		if s.LayerData[layerIdx] == nil { return errJSON("no output") }
		return marshalOutput(s.LayerData[layerIdx].Data)
	case poly.DTypeInt32:
		s := c.State.(*poly.StepState[int32])
		if int(layerIdx) < 0 || int(layerIdx) >= len(s.LayerData) { return errJSON("oob") }
		if s.LayerData[layerIdx] == nil { return errJSON("no output") }
		return marshalOutput(s.LayerData[layerIdx].Data)
	case poly.DTypeInt16:
		s := c.State.(*poly.StepState[int16])
		if int(layerIdx) < 0 || int(layerIdx) >= len(s.LayerData) { return errJSON("oob") }
		if s.LayerData[layerIdx] == nil { return errJSON("no output") }
		return marshalOutput(s.LayerData[layerIdx].Data)
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		s := c.State.(*poly.StepState[int8])
		if int(layerIdx) < 0 || int(layerIdx) >= len(s.LayerData) { return errJSON("oob") }
		if s.LayerData[layerIdx] == nil { return errJSON("no output") }
		return marshalOutput(s.LayerData[layerIdx].Data)
	case poly.DTypeUint64:
		s := c.State.(*poly.StepState[uint64])
		if int(layerIdx) < 0 || int(layerIdx) >= len(s.LayerData) { return errJSON("oob") }
		if s.LayerData[layerIdx] == nil { return errJSON("no output") }
		return marshalOutput(s.LayerData[layerIdx].Data)
	case poly.DTypeUint32:
		s := c.State.(*poly.StepState[uint32])
		if int(layerIdx) < 0 || int(layerIdx) >= len(s.LayerData) { return errJSON("oob") }
		if s.LayerData[layerIdx] == nil { return errJSON("no output") }
		return marshalOutput(s.LayerData[layerIdx].Data)
	case poly.DTypeUint16:
		s := c.State.(*poly.StepState[uint16])
		if int(layerIdx) < 0 || int(layerIdx) >= len(s.LayerData) { return errJSON("oob") }
		if s.LayerData[layerIdx] == nil { return errJSON("no output") }
		return marshalOutput(s.LayerData[layerIdx].Data)
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		s := c.State.(*poly.StepState[uint8])
		if int(layerIdx) < 0 || int(layerIdx) >= len(s.LayerData) { return errJSON("oob") }
		if s.LayerData[layerIdx] == nil { return errJSON("no output") }
		return marshalOutput(s.LayerData[layerIdx].Data)
	}
	return errJSON("unsupported dtype")
}

//export LoomSequentialForward
func LoomSequentialForward(networkHandle C.longlong, inputData *C.float, inputLen C.int) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}

	// Convert C target to Tensor
	ptr := unsafe.Pointer(inputData)
	slice := (*[1 << 30]float32)(ptr)[:inputLen:inputLen]

	// Determine dispatch type (usually from first layer or network default)
	dtype := poly.DTypeFloat32
	if len(n.Layers) > 0 {
		dtype = n.Layers[0].DType
	}

	var out *poly.Tensor[float32] // We'll convert result back to float32 for JSON/C return
	l := int(inputLen)

	switch dtype {
	case poly.DTypeFloat64:
		f := make([]float64, l)
		for i, v := range slice { f[i] = float64(v) }
		t := poly.NewTensorFromSlice(f, 1, l)
		o, _, _ := poly.ForwardPolymorphic(n, t)
		out = poly.ConvertTensor[float64, float32](o)
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		t := poly.NewTensorFromSlice(slice, 1, l)
		o, _, _ := poly.ForwardPolymorphic(n, t)
		out = o
	case poly.DTypeInt64:
		ts := make([]int64, l)
		for i, v := range slice { ts[i] = int64(v) }
		t := poly.NewTensorFromSlice(ts, 1, l)
		o, _, _ := poly.ForwardPolymorphic(n, t)
		out = poly.ConvertTensor[int64, float32](o)
	case poly.DTypeInt32:
		ts := make([]int32, l)
		for i, v := range slice { ts[i] = int32(v) }
		t := poly.NewTensorFromSlice(ts, 1, l)
		o, _, _ := poly.ForwardPolymorphic(n, t)
		out = poly.ConvertTensor[int32, float32](o)
	case poly.DTypeInt16:
		ts := make([]int16, l)
		for i, v := range slice { ts[i] = int16(v) }
		t := poly.NewTensorFromSlice(ts, 1, l)
		o, _, _ := poly.ForwardPolymorphic(n, t)
		out = poly.ConvertTensor[int16, float32](o)
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		ts := make([]int8, l)
		for i, v := range slice { ts[i] = int8(v) }
		t := poly.NewTensorFromSlice(ts, 1, l)
		o, _, _ := poly.ForwardPolymorphic(n, t)
		out = poly.ConvertTensor[int8, float32](o)
	case poly.DTypeUint64:
		ts := make([]uint64, l)
		for i, v := range slice { ts[i] = uint64(v) }
		t := poly.NewTensorFromSlice(ts, 1, l)
		o, _, _ := poly.ForwardPolymorphic(n, t)
		out = poly.ConvertTensor[uint64, float32](o)
	case poly.DTypeUint32:
		ts := make([]uint32, l)
		for i, v := range slice { ts[i] = uint32(v) }
		t := poly.NewTensorFromSlice(ts, 1, l)
		o, _, _ := poly.ForwardPolymorphic(n, t)
		out = poly.ConvertTensor[uint32, float32](o)
	case poly.DTypeUint16:
		ts := make([]uint16, l)
		for i, v := range slice { ts[i] = uint16(v) }
		t := poly.NewTensorFromSlice(ts, 1, l)
		o, _, _ := poly.ForwardPolymorphic(n, t)
		out = poly.ConvertTensor[uint16, float32](o)
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		ts := make([]uint8, l)
		for i, v := range slice { ts[i] = uint8(v) }
		t := poly.NewTensorFromSlice(ts, 1, l)
		o, _, _ := poly.ForwardPolymorphic(n, t)
		out = poly.ConvertTensor[uint8, float32](o)
	default:
		return errJSON("unsupported network dtype for sequential forward")
	}

	data, _ := json.Marshal(out.Data)
	return C.CString(string(data))
}

//export LoomConfigureTrainingMode
func LoomConfigureTrainingMode(networkHandle C.longlong, mode C.int) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if err := poly.ConfigureNetworkForMode(n, poly.TrainingMode(mode)); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"ok":true}`)
}

//export LoomForwardPolymorphic
// shapeJSON: e.g. [4,16] — tensor layout for inputData (float32 host buffer).
func LoomForwardPolymorphic(networkHandle C.longlong, inputData *C.float, inputLen C.int, shapeJSON *C.char) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	var shape []int
	if s := C.GoString(shapeJSON); s != "" {
		if err := json.Unmarshal([]byte(s), &shape); err != nil {
			return errJSON("invalid shape JSON: " + err.Error())
		}
	}
	if len(shape) == 0 {
		shape = []int{int(inputLen)}
	}
	slice := (*[1 << 30]float32)(unsafe.Pointer(inputData))[:inputLen:inputLen]
	t := poly.NewTensorFromSlice(slice, shape...)
	out, _, _ := poly.ForwardPolymorphic(n, t)
	if out == nil {
		return errJSON("forward returned nil")
	}
	data, _ := json.Marshal(out.Data)
	return C.CString(string(data))
}

//export LoomBackwardPolymorphic
// One full-network backward (MSE). Returns JSON {"dx":[],"dw":[],"dur_ns":int64}.
func LoomBackwardPolymorphic(
	networkHandle C.longlong,
	inputData *C.float, inputLen C.int, inputShapeJSON *C.char,
	targetData *C.float, targetLen C.int, targetShapeJSON *C.char,
) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	var inShape, tgtShape []int
	if err := json.Unmarshal([]byte(C.GoString(inputShapeJSON)), &inShape); err != nil {
		return errJSON("invalid input shape JSON")
	}
	if err := json.Unmarshal([]byte(C.GoString(targetShapeJSON)), &tgtShape); err != nil {
		return errJSON("invalid target shape JSON")
	}
	inSlice := (*[1 << 30]float32)(unsafe.Pointer(inputData))[:inputLen:inputLen]
	tgtSlice := (*[1 << 30]float32)(unsafe.Pointer(targetData))[:targetLen:targetLen]
	input := poly.NewTensorFromSlice(inSlice, inShape...)
	target := poly.NewTensorFromSlice(tgtSlice, tgtShape...)

	for i := range n.Layers {
		n.Layers[i].ResetState()
	}
	histIn := make([]*poly.Tensor[float32], len(n.Layers))
	histPre := make([]*poly.Tensor[float32], len(n.Layers))
	curr := input
	for i := range n.Layers {
		l := &n.Layers[i]
		if l.IsDisabled {
			continue
		}
		histIn[i] = curr
		pre, post := poly.DispatchLayer(l, curr, nil)
		histPre[i] = pre
		curr = post
	}
	gradOut := poly.ComputeLossGradient(curr, target, "mse")
	t0 := time.Now()
	_, layerGrads, _ := poly.BackwardPolymorphic(n, gradOut, histIn, histPre)
	dur := time.Since(t0)
	var dx, dw []float32
	if len(layerGrads) > 0 && layerGrads[0][0] != nil {
		dx = layerGrads[0][0].Data
	}
	for _, g := range layerGrads {
		if g[1] != nil {
			dw = append(dw, g[1].Data...)
		}
	}
	resp, _ := json.Marshal(map[string]interface{}{
		"dx": dx, "dw": dw, "dur_ns": dur.Nanoseconds(),
	})
	return C.CString(string(resp))
}

//export LoomMorphLayer
func LoomMorphLayer(networkHandle C.longlong, layerIdx C.int, targetDType C.int) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	
	l := &n.Layers[int(layerIdx)]
	err := poly.MorphLayer(l, poly.DType(targetDType))
	if err != nil {
		return errJSON(err.Error())
	}
	
	return C.CString(`{"status": "ok"}`)
}

//export LoomTokensToTensor
func LoomTokensToTensor(transformerHandle C.longlong, tokens *C.uint, count C.int) C.longlong {
	tr, ok := getTransformer(int64(transformerHandle))
	if !ok { return -1 }

	ptr := unsafe.Pointer(tokens)
	slice := (*[1 << 30]uint32)(ptr)[:count:count]



	// The current CABI LoomCreateTransformer only makes float32 transformers.
	t := tr.(*poly.Transformer[float32])
	res := t.TokensToTensor(slice)

	networkMu.Lock()
	id := tensorNextID
	tensorNextID++
	tensors[id] = res
	networkMu.Unlock()

	return C.longlong(id)
}

//export LoomForwardFull
func LoomForwardFull(transformerHandle C.longlong, inputTensorHandle C.longlong) C.longlong {
	tr, ok := getTransformer(int64(transformerHandle))
	if !ok { return -1 }
	ts, ok := getTensor(int64(inputTensorHandle))
	if !ok { return -1 }

	t := tr.(*poly.Transformer[float32])
	in := ts.(*poly.Tensor[float32])
	res := t.ForwardFull(in)

	networkMu.Lock()
	id := tensorNextID
	tensorNextID++
	tensors[id] = res
	networkMu.Unlock()

	return C.longlong(id)
}

//export LoomFreeTensor
func LoomFreeTensor(handle C.longlong) {
	networkMu.Lock()
	delete(tensors, int64(handle))
	networkMu.Unlock()
}

// Parity dummies for scanner
func _parityInf() {
	var t poly.Transformer[float32]
	_ = t.TokensToTensor
	_ = t.ForwardFull
}


