package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"encoding/json"
	"unsafe"

	"github.com/openfluke/loom/poly"
)

//export LoomCreateSystolicState
func LoomCreateSystolicState(networkHandle C.longlong, dtype C.int) C.longlong {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return -1 }

	dt := poly.DType(dtype)
	var state interface{}

	switch dt {
	case poly.DTypeFloat64:
		state = poly.NewSystolicState[float64](n)
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		state = poly.NewSystolicState[float32](n)
	case poly.DTypeInt64:
		state = poly.NewSystolicState[int64](n)
	case poly.DTypeInt32:
		state = poly.NewSystolicState[int32](n)
	case poly.DTypeInt16:
		state = poly.NewSystolicState[int16](n)
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		state = poly.NewSystolicState[int8](n)
	case poly.DTypeUint64:
		state = poly.NewSystolicState[uint64](n)
	case poly.DTypeUint32:
		state = poly.NewSystolicState[uint32](n)
	case poly.DTypeUint16:
		state = poly.NewSystolicState[uint16](n)
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		state = poly.NewSystolicState[uint8](n)
	default:
		state = poly.NewSystolicState[float32](n)
		dt = poly.DTypeFloat32
	}

	networkMu.Lock()
	id := systolicNextID
	systolicNextID++
	systolicStates[id] = &systolicContainer{State: state, DType: dt}
	networkMu.Unlock()

	return C.longlong(id)
}

//export LoomFreeSystolicState
func LoomFreeSystolicState(handle C.longlong) {
	networkMu.Lock()
	delete(systolicStates, int64(handle))
	networkMu.Unlock()
}

//export LoomSetInput
func LoomSetInput(stateHandle C.longlong, data *C.float, length C.int) {
	c, ok := getSystolicContainer(int64(stateHandle))
	if !ok { return }

	ptr := unsafe.Pointer(data)
	slice := (*[1 << 30]float32)(ptr)[:length:length]

	l := int(length)
	switch c.DType {
	case poly.DTypeFloat64:
		st := c.State.(*poly.SystolicState[float64])
		f := make([]float64, l)
		for i, v := range slice { f[i] = float64(v) }
		st.SetInput(poly.NewTensorFromSlice(f, 1, l))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		st := c.State.(*poly.SystolicState[float32])
		ts := make([]float32, l)
		copy(ts, slice)
		st.SetInput(poly.NewTensorFromSlice(ts, 1, l))
	case poly.DTypeInt64:
		st := c.State.(*poly.SystolicState[int64])
		ts := make([]int64, l)
		for i, v := range slice { ts[i] = int64(v) }
		st.SetInput(poly.NewTensorFromSlice(ts, 1, l))
	case poly.DTypeInt32:
		st := c.State.(*poly.SystolicState[int32])
		ts := make([]int32, l)
		for i, v := range slice { ts[i] = int32(v) }
		st.SetInput(poly.NewTensorFromSlice(ts, 1, l))
	case poly.DTypeInt16:
		st := c.State.(*poly.SystolicState[int16])
		ts := make([]int16, l)
		for i, v := range slice { ts[i] = int16(v) }
		st.SetInput(poly.NewTensorFromSlice(ts, 1, l))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		st := c.State.(*poly.SystolicState[int8])
		ts := make([]int8, l)
		for i, v := range slice { ts[i] = int8(v) }
		st.SetInput(poly.NewTensorFromSlice(ts, 1, l))
	case poly.DTypeUint64:
		st := c.State.(*poly.SystolicState[uint64])
		ts := make([]uint64, l)
		for i, v := range slice { ts[i] = uint64(v) }
		st.SetInput(poly.NewTensorFromSlice(ts, 1, l))
	case poly.DTypeUint32:
		st := c.State.(*poly.SystolicState[uint32])
		ts := make([]uint32, l)
		for i, v := range slice { ts[i] = uint32(v) }
		st.SetInput(poly.NewTensorFromSlice(ts, 1, l))
	case poly.DTypeUint16:
		st := c.State.(*poly.SystolicState[uint16])
		ts := make([]uint16, l)
		for i, v := range slice { ts[i] = uint16(v) }
		st.SetInput(poly.NewTensorFromSlice(ts, 1, l))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		st := c.State.(*poly.SystolicState[uint8])
		ts := make([]uint8, l)
		for i, v := range slice { ts[i] = uint8(v) }
		st.SetInput(poly.NewTensorFromSlice(ts, 1, l))
	}
}

//export LoomSystolicStep
func LoomSystolicStep(networkHandle C.longlong, stateHandle C.longlong, captureHistory C.int) C.longlong {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return -1 }
	c, ok := getSystolicContainer(int64(stateHandle))
	if !ok { return -1 }

	var duration int64
	switch c.DType {
	case poly.DTypeFloat64:
		duration = poly.SystolicForward(n, c.State.(*poly.SystolicState[float64]), captureHistory != 0).Nanoseconds()
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		duration = poly.SystolicForward(n, c.State.(*poly.SystolicState[float32]), captureHistory != 0).Nanoseconds()
	case poly.DTypeInt64:
		duration = poly.SystolicForward(n, c.State.(*poly.SystolicState[int64]), captureHistory != 0).Nanoseconds()
	case poly.DTypeInt32:
		duration = poly.SystolicForward(n, c.State.(*poly.SystolicState[int32]), captureHistory != 0).Nanoseconds()
	case poly.DTypeInt16:
		duration = poly.SystolicForward(n, c.State.(*poly.SystolicState[int16]), captureHistory != 0).Nanoseconds()
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		duration = poly.SystolicForward(n, c.State.(*poly.SystolicState[int8]), captureHistory != 0).Nanoseconds()
	case poly.DTypeUint64:
		duration = poly.SystolicForward(n, c.State.(*poly.SystolicState[uint64]), captureHistory != 0).Nanoseconds()
	case poly.DTypeUint32:
		duration = poly.SystolicForward(n, c.State.(*poly.SystolicState[uint32]), captureHistory != 0).Nanoseconds()
	case poly.DTypeUint16:
		duration = poly.SystolicForward(n, c.State.(*poly.SystolicState[uint16]), captureHistory != 0).Nanoseconds()
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		duration = poly.SystolicForward(n, c.State.(*poly.SystolicState[uint8]), captureHistory != 0).Nanoseconds()
	}
	return C.longlong(duration)
}

//export LoomGetOutput
func LoomGetOutput(stateHandle C.longlong, layerIdx C.int) *C.char {
	c, ok := getSystolicContainer(int64(stateHandle))
	if !ok { return errJSON("invalid state handle") }

	networkMu.RLock()
	defer networkMu.RUnlock()
	
	marshalOutput := func(data any) *C.char {
		res, _ := json.Marshal(data)
		return C.CString(string(res))
	}

	switch c.DType {
	case poly.DTypeFloat64:
		s := c.State.(*poly.SystolicState[float64])
		if int(layerIdx) < 0 || int(layerIdx) >= len(s.LayerData) { return errJSON("oob") }
		if s.LayerData[layerIdx] == nil { return errJSON("no output") }
		return marshalOutput(s.LayerData[layerIdx].Data)
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		s := c.State.(*poly.SystolicState[float32])
		if int(layerIdx) < 0 || int(layerIdx) >= len(s.LayerData) { return errJSON("oob") }
		if s.LayerData[layerIdx] == nil { return errJSON("no output") }
		return marshalOutput(s.LayerData[layerIdx].Data)
	case poly.DTypeInt64:
		s := c.State.(*poly.SystolicState[int64])
		if int(layerIdx) < 0 || int(layerIdx) >= len(s.LayerData) { return errJSON("oob") }
		if s.LayerData[layerIdx] == nil { return errJSON("no output") }
		return marshalOutput(s.LayerData[layerIdx].Data)
	case poly.DTypeInt32:
		s := c.State.(*poly.SystolicState[int32])
		if int(layerIdx) < 0 || int(layerIdx) >= len(s.LayerData) { return errJSON("oob") }
		if s.LayerData[layerIdx] == nil { return errJSON("no output") }
		return marshalOutput(s.LayerData[layerIdx].Data)
	case poly.DTypeInt16:
		s := c.State.(*poly.SystolicState[int16])
		if int(layerIdx) < 0 || int(layerIdx) >= len(s.LayerData) { return errJSON("oob") }
		if s.LayerData[layerIdx] == nil { return errJSON("no output") }
		return marshalOutput(s.LayerData[layerIdx].Data)
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		s := c.State.(*poly.SystolicState[int8])
		if int(layerIdx) < 0 || int(layerIdx) >= len(s.LayerData) { return errJSON("oob") }
		if s.LayerData[layerIdx] == nil { return errJSON("no output") }
		return marshalOutput(s.LayerData[layerIdx].Data)
	case poly.DTypeUint64:
		s := c.State.(*poly.SystolicState[uint64])
		if int(layerIdx) < 0 || int(layerIdx) >= len(s.LayerData) { return errJSON("oob") }
		if s.LayerData[layerIdx] == nil { return errJSON("no output") }
		return marshalOutput(s.LayerData[layerIdx].Data)
	case poly.DTypeUint32:
		s := c.State.(*poly.SystolicState[uint32])
		if int(layerIdx) < 0 || int(layerIdx) >= len(s.LayerData) { return errJSON("oob") }
		if s.LayerData[layerIdx] == nil { return errJSON("no output") }
		return marshalOutput(s.LayerData[layerIdx].Data)
	case poly.DTypeUint16:
		s := c.State.(*poly.SystolicState[uint16])
		if int(layerIdx) < 0 || int(layerIdx) >= len(s.LayerData) { return errJSON("oob") }
		if s.LayerData[layerIdx] == nil { return errJSON("no output") }
		return marshalOutput(s.LayerData[layerIdx].Data)
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		s := c.State.(*poly.SystolicState[uint8])
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
