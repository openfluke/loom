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

//export LoomApplyGradients
func LoomApplyGradients(networkHandle C.longlong, layerIdx C.int, gradWeightsJSON *C.char, learningRate C.float) {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return }
	
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return
	}
	
	l := &n.Layers[int(layerIdx)]
	if l.WeightStore == nil { return }
	
	var grad poly.Tensor[float32]
	if err := json.Unmarshal([]byte(C.GoString(gradWeightsJSON)), &grad); err != nil {
		return
	}
	
	l.WeightStore.ApplyGradients(&grad, float32(learningRate))
}

//export LoomSystolicBackward
func LoomSystolicBackward(networkHandle C.longlong, stateHandle C.longlong, gradOutputData *C.float, gradOutputLen C.int) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(stateHandle))
	if !ok { return errJSON("invalid state handle") }

	ptr := unsafe.Pointer(gradOutputData)
	slice := (*[1 << 30]float32)(ptr)[:gradOutputLen:gradOutputLen]

	marshalGrad := func(gIn any, dt poly.DType) *C.char {
		if gIn == nil { return errJSON("no gradient") }
		switch dt {
		case poly.DTypeFloat64:
			data, _ := json.Marshal(gIn.(*poly.Tensor[float64]).Data)
			return C.CString(string(data))
		case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
			data, _ := json.Marshal(gIn.(*poly.Tensor[float32]).Data)
			return C.CString(string(data))
		case poly.DTypeInt64:
			data, _ := json.Marshal(gIn.(*poly.Tensor[int64]).Data)
			return C.CString(string(data))
		case poly.DTypeInt32:
			data, _ := json.Marshal(gIn.(*poly.Tensor[int32]).Data)
			return C.CString(string(data))
		case poly.DTypeInt16:
			data, _ := json.Marshal(gIn.(*poly.Tensor[int16]).Data)
			return C.CString(string(data))
		case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
			data, _ := json.Marshal(gIn.(*poly.Tensor[int8]).Data)
			return C.CString(string(data))
		case poly.DTypeUint64:
			data, _ := json.Marshal(gIn.(*poly.Tensor[uint64]).Data)
			return C.CString(string(data))
		case poly.DTypeUint32:
			data, _ := json.Marshal(gIn.(*poly.Tensor[uint32]).Data)
			return C.CString(string(data))
		case poly.DTypeUint16:
			data, _ := json.Marshal(gIn.(*poly.Tensor[uint16]).Data)
			return C.CString(string(data))
		case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
			data, _ := json.Marshal(gIn.(*poly.Tensor[uint8]).Data)
			return C.CString(string(data))
		}
		return errJSON("marshal error")
	}

	switch c.DType {
	case poly.DTypeFloat64:
		s := c.State.(*poly.SystolicState[float64])
		f64 := make([]float64, gradOutputLen)
		for i, v := range slice { f64[i] = float64(v) }
		gradOut := poly.NewTensorFromSlice(f64, int(gradOutputLen))
		gIn, _, err := poly.SystolicBackward(n, s, gradOut)
		if err != nil { return errJSON(err.Error()) }
		return marshalGrad(gIn, c.DType)
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		s := c.State.(*poly.SystolicState[float32])
		gradOut := poly.NewTensorFromSlice(slice, int(gradOutputLen))
		gIn, _, err := poly.SystolicBackward(n, s, gradOut)
		if err != nil { return errJSON(err.Error()) }
		return marshalGrad(gIn, c.DType)
	case poly.DTypeInt64:
		s := c.State.(*poly.SystolicState[int64])
		ts := make([]int64, gradOutputLen)
		for i, v := range slice { ts[i] = int64(v) }
		gradOut := poly.NewTensorFromSlice(ts, int(gradOutputLen))
		gIn, _, err := poly.SystolicBackward(n, s, gradOut)
		if err != nil { return errJSON(err.Error()) }
		return marshalGrad(gIn, poly.DTypeInt64)
	case poly.DTypeInt32:
		s := c.State.(*poly.SystolicState[int32])
		ts := make([]int32, gradOutputLen)
		for i, v := range slice { ts[i] = int32(v) }
		gradOut := poly.NewTensorFromSlice(ts, int(gradOutputLen))
		gIn, _, err := poly.SystolicBackward(n, s, gradOut)
		if err != nil { return errJSON(err.Error()) }
		return marshalGrad(gIn, poly.DTypeInt32)
	case poly.DTypeInt16:
		s := c.State.(*poly.SystolicState[int16])
		ts := make([]int16, gradOutputLen)
		for i, v := range slice { ts[i] = int16(v) }
		gradOut := poly.NewTensorFromSlice(ts, int(gradOutputLen))
		gIn, _, err := poly.SystolicBackward(n, s, gradOut)
		if err != nil { return errJSON(err.Error()) }
		return marshalGrad(gIn, poly.DTypeInt16)
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		s := c.State.(*poly.SystolicState[int8])
		ts := make([]int8, gradOutputLen)
		for i, v := range slice { ts[i] = int8(v) }
		gradOut := poly.NewTensorFromSlice(ts, int(gradOutputLen))
		gIn, _, err := poly.SystolicBackward(n, s, gradOut)
		if err != nil { return errJSON(err.Error()) }
		return marshalGrad(gIn, poly.DTypeInt8)
	case poly.DTypeUint64:
		s := c.State.(*poly.SystolicState[uint64])
		ts := make([]uint64, gradOutputLen)
		for i, v := range slice { ts[i] = uint64(v) }
		gradOut := poly.NewTensorFromSlice(ts, int(gradOutputLen))
		gIn, _, err := poly.SystolicBackward(n, s, gradOut)
		if err != nil { return errJSON(err.Error()) }
		return marshalGrad(gIn, poly.DTypeUint64)
	case poly.DTypeUint32:
		s := c.State.(*poly.SystolicState[uint32])
		ts := make([]uint32, gradOutputLen)
		for i, v := range slice { ts[i] = uint32(v) }
		gradOut := poly.NewTensorFromSlice(ts, int(gradOutputLen))
		gIn, _, err := poly.SystolicBackward(n, s, gradOut)
		if err != nil { return errJSON(err.Error()) }
		return marshalGrad(gIn, poly.DTypeUint32)
	case poly.DTypeUint16:
		s := c.State.(*poly.SystolicState[uint16])
		ts := make([]uint16, gradOutputLen)
		for i, v := range slice { ts[i] = uint16(v) }
		gradOut := poly.NewTensorFromSlice(ts, int(gradOutputLen))
		gIn, _, err := poly.SystolicBackward(n, s, gradOut)
		if err != nil { return errJSON(err.Error()) }
		return marshalGrad(gIn, poly.DTypeUint16)
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		s := c.State.(*poly.SystolicState[uint8])
		ts := make([]uint8, gradOutputLen)
		for i, v := range slice { ts[i] = uint8(v) }
		gradOut := poly.NewTensorFromSlice(ts, int(gradOutputLen))
		gIn, _, err := poly.SystolicBackward(n, s, gradOut)
		if err != nil { return errJSON(err.Error()) }
		return marshalGrad(gIn, poly.DTypeUint8)
	}
	return errJSON("unsupported backward dtype")
}

//export LoomApplyTargetProp
func LoomApplyTargetProp(networkHandle C.longlong, stateHandle C.longlong, targetData *C.float, targetLen C.int, learningRate C.float) {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return }
	c, ok := getSystolicContainer(int64(stateHandle))
	if !ok { return }

	ptr := unsafe.Pointer(targetData)
	slice := (*[1 << 30]float32)(ptr)[:targetLen:targetLen]

	switch c.DType {
	case poly.DTypeFloat64:
		s := c.State.(*poly.SystolicState[float64])
		f := make([]float64, targetLen)
		for i, v := range slice { f[i] = float64(v) }
		poly.SystolicApplyTargetProp(n, s, poly.NewTensorFromSlice(f, int(targetLen)), float32(learningRate))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		s := c.State.(*poly.SystolicState[float32])
		poly.SystolicApplyTargetProp(n, s, poly.NewTensorFromSlice(slice, int(targetLen)), float32(learningRate))
	case poly.DTypeInt64:
		s := c.State.(*poly.SystolicState[int64])
		ts := make([]int64, targetLen)
		for i, v := range slice { ts[i] = int64(v) }
		poly.SystolicApplyTargetProp(n, s, poly.NewTensorFromSlice(ts, int(targetLen)), float32(learningRate))
	case poly.DTypeInt32:
		s := c.State.(*poly.SystolicState[int32])
		ts := make([]int32, targetLen)
		for i, v := range slice { ts[i] = int32(v) }
		poly.SystolicApplyTargetProp(n, s, poly.NewTensorFromSlice(ts, int(targetLen)), float32(learningRate))
	case poly.DTypeInt16:
		s := c.State.(*poly.SystolicState[int16])
		ts := make([]int16, targetLen)
		for i, v := range slice { ts[i] = int16(v) }
		poly.SystolicApplyTargetProp(n, s, poly.NewTensorFromSlice(ts, int(targetLen)), float32(learningRate))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		s := c.State.(*poly.SystolicState[int8])
		ts := make([]int8, targetLen)
		for i, v := range slice { ts[i] = int8(v) }
		poly.SystolicApplyTargetProp(n, s, poly.NewTensorFromSlice(ts, int(targetLen)), float32(learningRate))
	case poly.DTypeUint64:
		s := c.State.(*poly.SystolicState[uint64])
		ts := make([]uint64, targetLen)
		for i, v := range slice { ts[i] = uint64(v) }
		poly.SystolicApplyTargetProp(n, s, poly.NewTensorFromSlice(ts, int(targetLen)), float32(learningRate))
	case poly.DTypeUint32:
		s := c.State.(*poly.SystolicState[uint32])
		ts := make([]uint32, targetLen)
		for i, v := range slice { ts[i] = uint32(v) }
		poly.SystolicApplyTargetProp(n, s, poly.NewTensorFromSlice(ts, int(targetLen)), float32(learningRate))
	case poly.DTypeUint16:
		s := c.State.(*poly.SystolicState[uint16])
		ts := make([]uint16, targetLen)
		for i, v := range slice { ts[i] = uint16(v) }
		poly.SystolicApplyTargetProp(n, s, poly.NewTensorFromSlice(ts, int(targetLen)), float32(learningRate))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		s := c.State.(*poly.SystolicState[uint8])
		ts := make([]uint8, targetLen)
		for i, v := range slice { ts[i] = uint8(v) }
		poly.SystolicApplyTargetProp(n, s, poly.NewTensorFromSlice(ts, int(targetLen)), float32(learningRate))
	}
}

//export LoomCreateTargetPropState
func LoomCreateTargetPropState(networkHandle C.longlong, dtype C.int) C.longlong {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return -1 }
	
	dt := poly.DType(dtype)
	var state interface{}

	switch dt {
	case poly.DTypeFloat64:
		state = poly.NewTargetPropState[float64](n, nil)
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		state = poly.NewTargetPropState[float32](n, nil)
	case poly.DTypeInt64:
		state = poly.NewTargetPropState[int64](n, nil)
	case poly.DTypeInt32:
		state = poly.NewTargetPropState[int32](n, nil)
	case poly.DTypeInt16:
		state = poly.NewTargetPropState[int16](n, nil)
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		state = poly.NewTargetPropState[int8](n, nil)
	case poly.DTypeUint64:
		state = poly.NewTargetPropState[uint64](n, nil)
	case poly.DTypeUint32:
		state = poly.NewTargetPropState[uint32](n, nil)
	case poly.DTypeUint16:
		state = poly.NewTargetPropState[uint16](n, nil)
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		state = poly.NewTargetPropState[uint8](n, nil)
	default:
		state = poly.NewTargetPropState[float32](n, nil)
		dt = poly.DTypeFloat32
	}
	
	networkMu.Lock()
	id := targetPropNextID
	targetPropNextID++
	targetPropStates[id] = &targetPropContainer{State: state, DType: dt}
	networkMu.Unlock()
	
	return C.longlong(id)
}

//export LoomTargetPropForward
func LoomTargetPropForward(networkHandle C.longlong, tpHandle C.longlong, inputData *C.float, inputLen C.int) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	
	networkMu.RLock()
	c, ok := targetPropStates[int64(tpHandle)]
	networkMu.RUnlock()
	if !ok { return errJSON("invalid targetprop handle") }
	
	ptr := unsafe.Pointer(inputData)
	slice := (*[1 << 30]float32)(ptr)[:inputLen:inputLen]
	
	switch c.DType {
	case poly.DTypeFloat64:
		tp := c.State.(*poly.TargetPropState[float64])
		f := make([]float64, inputLen)
		for i, v := range slice { f[i] = float64(v) }
		out := poly.TargetPropForward(n, tp, poly.NewTensorFromSlice(f, int(inputLen)))
		res, _ := json.Marshal(out.Data)
		return C.CString(string(res))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		tp := c.State.(*poly.TargetPropState[float32])
		out := poly.TargetPropForward(n, tp, poly.NewTensorFromSlice(slice, int(inputLen)))
		res, _ := json.Marshal(out.Data)
		return C.CString(string(res))
	case poly.DTypeInt64:
		tp := c.State.(*poly.TargetPropState[int64])
		ts := make([]int64, inputLen)
		for i, v := range slice { ts[i] = int64(v) }
		out := poly.TargetPropForward(n, tp, poly.NewTensorFromSlice(ts, int(inputLen)))
		res, _ := json.Marshal(out.Data)
		return C.CString(string(res))
	case poly.DTypeInt32:
		tp := c.State.(*poly.TargetPropState[int32])
		ts := make([]int32, inputLen)
		for i, v := range slice { ts[i] = int32(v) }
		out := poly.TargetPropForward(n, tp, poly.NewTensorFromSlice(ts, int(inputLen)))
		res, _ := json.Marshal(out.Data)
		return C.CString(string(res))
	case poly.DTypeInt16:
		tp := c.State.(*poly.TargetPropState[int16])
		ts := make([]int16, inputLen)
		for i, v := range slice { ts[i] = int16(v) }
		out := poly.TargetPropForward(n, tp, poly.NewTensorFromSlice(ts, int(inputLen)))
		res, _ := json.Marshal(out.Data)
		return C.CString(string(res))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		tp := c.State.(*poly.TargetPropState[int8])
		ts := make([]int8, inputLen)
		for i, v := range slice { ts[i] = int8(v) }
		out := poly.TargetPropForward(n, tp, poly.NewTensorFromSlice(ts, int(inputLen)))
		res, _ := json.Marshal(out.Data)
		return C.CString(string(res))
	case poly.DTypeUint64:
		tp := c.State.(*poly.TargetPropState[uint64])
		ts := make([]uint64, inputLen)
		for i, v := range slice { ts[i] = uint64(v) }
		out := poly.TargetPropForward(n, tp, poly.NewTensorFromSlice(ts, int(inputLen)))
		res, _ := json.Marshal(out.Data)
		return C.CString(string(res))
	case poly.DTypeUint32:
		tp := c.State.(*poly.TargetPropState[uint32])
		ts := make([]uint32, inputLen)
		for i, v := range slice { ts[i] = uint32(v) }
		out := poly.TargetPropForward(n, tp, poly.NewTensorFromSlice(ts, int(inputLen)))
		res, _ := json.Marshal(out.Data)
		return C.CString(string(res))
	case poly.DTypeUint16:
		tp := c.State.(*poly.TargetPropState[uint16])
		ts := make([]uint16, inputLen)
		for i, v := range slice { ts[i] = uint16(v) }
		out := poly.TargetPropForward(n, tp, poly.NewTensorFromSlice(ts, int(inputLen)))
		res, _ := json.Marshal(out.Data)
		return C.CString(string(res))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		tp := c.State.(*poly.TargetPropState[uint8])
		ts := make([]uint8, inputLen)
		for i, v := range slice { ts[i] = uint8(v) }
		out := poly.TargetPropForward(n, tp, poly.NewTensorFromSlice(ts, int(inputLen)))
		res, _ := json.Marshal(out.Data)
		return C.CString(string(res))
	}
	return errJSON("unsupported tp forward dtype")
}

//export LoomTargetPropBackward
func LoomTargetPropBackward(networkHandle C.longlong, tpHandle C.longlong, targetData *C.float, targetLen C.int) {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return }
	
	networkMu.RLock()
	c, ok := targetPropStates[int64(tpHandle)]
	networkMu.RUnlock()
	if !ok { return }
	
	ptr := unsafe.Pointer(targetData)
	slice := (*[1 << 30]float32)(ptr)[:targetLen:targetLen]
	
	switch c.DType {
	case poly.DTypeFloat64:
		tp := c.State.(*poly.TargetPropState[float64])
		ts := make([]float64, targetLen)
		for i, v := range slice { ts[i] = float64(v) }
		poly.TargetPropBackward(n, tp, poly.NewTensorFromSlice(ts, int(targetLen)))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		tp := c.State.(*poly.TargetPropState[float32])
		poly.TargetPropBackward(n, tp, poly.NewTensorFromSlice(slice, int(targetLen)))
	case poly.DTypeInt64:
		tp := c.State.(*poly.TargetPropState[int64])
		ts := make([]int64, targetLen)
		for i, v := range slice { ts[i] = int64(v) }
		poly.TargetPropBackward(n, tp, poly.NewTensorFromSlice(ts, int(targetLen)))
	case poly.DTypeInt32:
		tp := c.State.(*poly.TargetPropState[int32])
		ts := make([]int32, targetLen)
		for i, v := range slice { ts[i] = int32(v) }
		poly.TargetPropBackward(n, tp, poly.NewTensorFromSlice(ts, int(targetLen)))
	case poly.DTypeInt16:
		tp := c.State.(*poly.TargetPropState[int16])
		ts := make([]int16, targetLen)
		for i, v := range slice { ts[i] = int16(v) }
		poly.TargetPropBackward(n, tp, poly.NewTensorFromSlice(ts, int(targetLen)))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		tp := c.State.(*poly.TargetPropState[int8])
		ts := make([]int8, targetLen)
		for i, v := range slice { ts[i] = int8(v) }
		poly.TargetPropBackward(n, tp, poly.NewTensorFromSlice(ts, int(targetLen)))
	case poly.DTypeUint64:
		tp := c.State.(*poly.TargetPropState[uint64])
		ts := make([]uint64, targetLen)
		for i, v := range slice { ts[i] = uint64(v) }
		poly.TargetPropBackward(n, tp, poly.NewTensorFromSlice(ts, int(targetLen)))
	case poly.DTypeUint32:
		tp := c.State.(*poly.TargetPropState[uint32])
		ts := make([]uint32, targetLen)
		for i, v := range slice { ts[i] = uint32(v) }
		poly.TargetPropBackward(n, tp, poly.NewTensorFromSlice(ts, int(targetLen)))
	case poly.DTypeUint16:
		tp := c.State.(*poly.TargetPropState[uint16])
		ts := make([]uint16, targetLen)
		for i, v := range slice { ts[i] = uint16(v) }
		poly.TargetPropBackward(n, tp, poly.NewTensorFromSlice(ts, int(targetLen)))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		tp := c.State.(*poly.TargetPropState[uint8])
		ts := make([]uint8, targetLen)
		for i, v := range slice { ts[i] = uint8(v) }
		poly.TargetPropBackward(n, tp, poly.NewTensorFromSlice(ts, int(targetLen)))
	}
}

//export LoomDefaultTargetPropConfig
func LoomDefaultTargetPropConfig() *C.char {
	config := poly.DefaultTargetPropConfig()
	data, _ := json.Marshal(config)
	return C.CString(string(data))
}

//export LoomApplyTargetPropGaps
func LoomApplyTargetPropGaps(networkHandle C.longlong, tpHandle C.longlong, learningRate C.float) {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return }
	
	networkMu.RLock()
	c, ok := targetPropStates[int64(tpHandle)]
	networkMu.RUnlock()
	if !ok { return }
	
	switch c.DType {
	case poly.DTypeFloat64:
		poly.ApplyTargetPropGaps(n, c.State.(*poly.TargetPropState[float64]), float32(learningRate))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		poly.ApplyTargetPropGaps(n, c.State.(*poly.TargetPropState[float32]), float32(learningRate))
	case poly.DTypeInt64:
		poly.ApplyTargetPropGaps(n, c.State.(*poly.TargetPropState[int64]), float32(learningRate))
	case poly.DTypeInt32:
		poly.ApplyTargetPropGaps(n, c.State.(*poly.TargetPropState[int32]), float32(learningRate))
	case poly.DTypeInt16:
		poly.ApplyTargetPropGaps(n, c.State.(*poly.TargetPropState[int16]), float32(learningRate))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		poly.ApplyTargetPropGaps(n, c.State.(*poly.TargetPropState[int8]), float32(learningRate))
	case poly.DTypeUint64:
		poly.ApplyTargetPropGaps(n, c.State.(*poly.TargetPropState[uint64]), float32(learningRate))
	case poly.DTypeUint32:
		poly.ApplyTargetPropGaps(n, c.State.(*poly.TargetPropState[uint32]), float32(learningRate))
	case poly.DTypeUint16:
		poly.ApplyTargetPropGaps(n, c.State.(*poly.TargetPropState[uint16]), float32(learningRate))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		poly.ApplyTargetPropGaps(n, c.State.(*poly.TargetPropState[uint8]), float32(learningRate))
	}
}

//export LoomComputeLossGradient
func LoomComputeLossGradient(outputHandle C.longlong, targetHandle C.longlong, lossType *C.char) *C.char {
	cOut, ok := getSystolicContainer(int64(outputHandle))
	if !ok { return errJSON("invalid output handle") }
	cTar, ok := getSystolicContainer(int64(targetHandle))
	if !ok { return errJSON("invalid target handle") }
	
	lt := C.GoString(lossType)
	
	var grad interface{}
	switch cOut.DType {
	case poly.DTypeFloat64:
		grad = poly.ComputeLossGradient(cOut.State.(*poly.Tensor[float64]), cTar.State.(*poly.Tensor[float64]), lt)
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		grad = poly.ComputeLossGradient(cOut.State.(*poly.Tensor[float32]), cTar.State.(*poly.Tensor[float32]), lt)
	case poly.DTypeInt64:
		grad = poly.ComputeLossGradient(cOut.State.(*poly.Tensor[int64]), cTar.State.(*poly.Tensor[int64]), lt)
	case poly.DTypeInt32:
		grad = poly.ComputeLossGradient(cOut.State.(*poly.Tensor[int32]), cTar.State.(*poly.Tensor[int32]), lt)
	case poly.DTypeInt16:
		grad = poly.ComputeLossGradient(cOut.State.(*poly.Tensor[int16]), cTar.State.(*poly.Tensor[int16]), lt)
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		grad = poly.ComputeLossGradient(cOut.State.(*poly.Tensor[int8]), cTar.State.(*poly.Tensor[int8]), lt)
	case poly.DTypeUint64:
		grad = poly.ComputeLossGradient(cOut.State.(*poly.Tensor[uint64]), cTar.State.(*poly.Tensor[uint64]), lt)
	case poly.DTypeUint32:
		grad = poly.ComputeLossGradient(cOut.State.(*poly.Tensor[uint32]), cTar.State.(*poly.Tensor[uint32]), lt)
	case poly.DTypeUint16:
		grad = poly.ComputeLossGradient(cOut.State.(*poly.Tensor[uint16]), cTar.State.(*poly.Tensor[uint16]), lt)
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		grad = poly.ComputeLossGradient(cOut.State.(*poly.Tensor[uint8]), cTar.State.(*poly.Tensor[uint8]), lt)
	default:
		return errJSON("unsupported dtype for loss gradient")
	}

	data, _ := json.Marshal(grad)
	return C.CString(string(data))
}

//export LoomDenseBackward
func LoomDenseBackward(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	
	cGOut, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	cIn, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	cPre, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var gIn, gW interface{}
	switch cIn.DType {
	case poly.DTypeFloat64:
		gIn, gW = poly.DenseBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float64]), cIn.State.(*poly.Tensor[float64]), cPre.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		gIn, gW = poly.DenseBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float32]), cIn.State.(*poly.Tensor[float32]), cPre.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		gIn, gW = poly.DenseBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int64]), cIn.State.(*poly.Tensor[int64]), cPre.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		gIn, gW = poly.DenseBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int32]), cIn.State.(*poly.Tensor[int32]), cPre.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		gIn, gW = poly.DenseBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int16]), cIn.State.(*poly.Tensor[int16]), cPre.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		gIn, gW = poly.DenseBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int8]), cIn.State.(*poly.Tensor[int8]), cPre.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		gIn, gW = poly.DenseBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint64]), cIn.State.(*poly.Tensor[uint64]), cPre.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		gIn, gW = poly.DenseBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint32]), cIn.State.(*poly.Tensor[uint32]), cPre.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		gIn, gW = poly.DenseBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint16]), cIn.State.(*poly.Tensor[uint16]), cPre.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		gIn, gW = poly.DenseBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint8]), cIn.State.(*poly.Tensor[uint8]), cPre.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for dense backward")
	}

	res := map[string]interface{}{"gradInput": gIn, "gradWeights": gW}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomRMSNormBackward
func LoomRMSNormBackward(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	cGOut, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	cIn, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	cPre, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var gIn, gW interface{}
	switch cIn.DType {
	case poly.DTypeFloat64:
		gIn, gW = poly.RMSNormBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float64]), cIn.State.(*poly.Tensor[float64]), cPre.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		gIn, gW = poly.RMSNormBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float32]), cIn.State.(*poly.Tensor[float32]), cPre.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		gIn, gW = poly.RMSNormBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int64]), cIn.State.(*poly.Tensor[int64]), cPre.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		gIn, gW = poly.RMSNormBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int32]), cIn.State.(*poly.Tensor[int32]), cPre.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		gIn, gW = poly.RMSNormBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int16]), cIn.State.(*poly.Tensor[int16]), cPre.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		gIn, gW = poly.RMSNormBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int8]), cIn.State.(*poly.Tensor[int8]), cPre.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		gIn, gW = poly.RMSNormBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint64]), cIn.State.(*poly.Tensor[uint64]), cPre.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		gIn, gW = poly.RMSNormBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint32]), cIn.State.(*poly.Tensor[uint32]), cPre.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		gIn, gW = poly.RMSNormBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint16]), cIn.State.(*poly.Tensor[uint16]), cPre.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		gIn, gW = poly.RMSNormBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint8]), cIn.State.(*poly.Tensor[uint8]), cPre.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for rmsnorm backward")
	}

	res := map[string]interface{}{"gradInput": gIn, "gradWeights": gW}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomLayerNormBackward
func LoomLayerNormBackward(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	cGOut, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	cIn, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	cPre, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var gIn, gW interface{}
	switch cIn.DType {
	case poly.DTypeFloat64:
		gIn, gW = poly.LayerNormBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float64]), cIn.State.(*poly.Tensor[float64]), cPre.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		gIn, gW = poly.LayerNormBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float32]), cIn.State.(*poly.Tensor[float32]), cPre.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		gIn, gW = poly.LayerNormBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int64]), cIn.State.(*poly.Tensor[int64]), cPre.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		gIn, gW = poly.LayerNormBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int32]), cIn.State.(*poly.Tensor[int32]), cPre.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		gIn, gW = poly.LayerNormBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int16]), cIn.State.(*poly.Tensor[int16]), cPre.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		gIn, gW = poly.LayerNormBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int8]), cIn.State.(*poly.Tensor[int8]), cPre.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		gIn, gW = poly.LayerNormBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint64]), cIn.State.(*poly.Tensor[uint64]), cPre.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		gIn, gW = poly.LayerNormBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint32]), cIn.State.(*poly.Tensor[uint32]), cPre.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		gIn, gW = poly.LayerNormBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint16]), cIn.State.(*poly.Tensor[uint16]), cPre.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		gIn, gW = poly.LayerNormBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint8]), cIn.State.(*poly.Tensor[uint8]), cPre.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for layernorm backward")
	}

	res := map[string]interface{}{"gradInput": gIn, "gradWeights": gW}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomMHABackward
func LoomMHABackward(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	cGOut, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	cIn, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	cPre, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var gIn, gW interface{}
	switch cIn.DType {
	case poly.DTypeFloat64:
		gIn, gW = poly.MHABackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float64]), cIn.State.(*poly.Tensor[float64]), cPre.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		gIn, gW = poly.MHABackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float32]), cIn.State.(*poly.Tensor[float32]), cPre.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		gIn, gW = poly.MHABackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int64]), cIn.State.(*poly.Tensor[int64]), cPre.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		gIn, gW = poly.MHABackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int32]), cIn.State.(*poly.Tensor[int32]), cPre.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		gIn, gW = poly.MHABackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int16]), cIn.State.(*poly.Tensor[int16]), cPre.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		gIn, gW = poly.MHABackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int8]), cIn.State.(*poly.Tensor[int8]), cPre.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		gIn, gW = poly.MHABackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint64]), cIn.State.(*poly.Tensor[uint64]), cPre.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		gIn, gW = poly.MHABackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint32]), cIn.State.(*poly.Tensor[uint32]), cPre.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		gIn, gW = poly.MHABackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint16]), cIn.State.(*poly.Tensor[uint16]), cPre.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		gIn, gW = poly.MHABackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint8]), cIn.State.(*poly.Tensor[uint8]), cPre.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for mha backward")
	}

	res := map[string]interface{}{"gradInput": gIn, "gradWeights": gW}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomSoftmaxBackward
func LoomSoftmaxBackward(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, postActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	cGOut, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	cIn, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	cPost, ok := getSystolicContainer(int64(postActHandle))
	if !ok { return errJSON("invalid postAct handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var gIn, gW interface{}
	switch cIn.DType {
	case poly.DTypeFloat64:
		gIn, gW = poly.SoftmaxBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float64]), cIn.State.(*poly.Tensor[float64]), cPost.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		gIn, gW = poly.SoftmaxBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float32]), cIn.State.(*poly.Tensor[float32]), cPost.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		gIn, gW = poly.SoftmaxBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int64]), cIn.State.(*poly.Tensor[int64]), cPost.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		gIn, gW = poly.SoftmaxBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int32]), cIn.State.(*poly.Tensor[int32]), cPost.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		gIn, gW = poly.SoftmaxBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int16]), cIn.State.(*poly.Tensor[int16]), cPost.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		gIn, gW = poly.SoftmaxBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int8]), cIn.State.(*poly.Tensor[int8]), cPost.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		gIn, gW = poly.SoftmaxBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint64]), cIn.State.(*poly.Tensor[uint64]), cPost.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		gIn, gW = poly.SoftmaxBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint32]), cIn.State.(*poly.Tensor[uint32]), cPost.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		gIn, gW = poly.SoftmaxBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint16]), cIn.State.(*poly.Tensor[uint16]), cPost.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		gIn, gW = poly.SoftmaxBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint8]), cIn.State.(*poly.Tensor[uint8]), cPost.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for softmax backward")
	}

	res := map[string]interface{}{"gradInput": gIn, "gradWeights": gW}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomSwiGLUBackward
func LoomSwiGLUBackward(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	cGOut, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	cIn, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	cPre, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var gIn, gW interface{}
	switch cIn.DType {
	case poly.DTypeFloat64:
		gIn, gW = poly.SwiGLUBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float64]), cIn.State.(*poly.Tensor[float64]), cPre.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		gIn, gW = poly.SwiGLUBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float32]), cIn.State.(*poly.Tensor[float32]), cPre.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		gIn, gW = poly.SwiGLUBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int64]), cIn.State.(*poly.Tensor[int64]), cPre.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		gIn, gW = poly.SwiGLUBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int32]), cIn.State.(*poly.Tensor[int32]), cPre.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		gIn, gW = poly.SwiGLUBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int16]), cIn.State.(*poly.Tensor[int16]), cPre.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		gIn, gW = poly.SwiGLUBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int8]), cIn.State.(*poly.Tensor[int8]), cPre.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		gIn, gW = poly.SwiGLUBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint64]), cIn.State.(*poly.Tensor[uint64]), cPre.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		gIn, gW = poly.SwiGLUBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint32]), cIn.State.(*poly.Tensor[uint32]), cPre.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		gIn, gW = poly.SwiGLUBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint16]), cIn.State.(*poly.Tensor[uint16]), cPre.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		gIn, gW = poly.SwiGLUBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint8]), cIn.State.(*poly.Tensor[uint8]), cPre.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for swiglu backward")
	}

	res := map[string]interface{}{"gradInput": gIn, "gradWeights": gW}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomEmbeddingBackward
func LoomEmbeddingBackward(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	cGOut, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	cIn, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	cPre, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var gIn, gW interface{}
	switch cIn.DType {
	case poly.DTypeFloat64:
		gIn, gW = poly.EmbeddingBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float64]), cIn.State.(*poly.Tensor[float64]), cPre.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		gIn, gW = poly.EmbeddingBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float32]), cIn.State.(*poly.Tensor[float32]), cPre.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		gIn, gW = poly.EmbeddingBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int64]), cIn.State.(*poly.Tensor[int64]), cPre.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		gIn, gW = poly.EmbeddingBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int32]), cIn.State.(*poly.Tensor[int32]), cPre.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		gIn, gW = poly.EmbeddingBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int16]), cIn.State.(*poly.Tensor[int16]), cPre.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		gIn, gW = poly.EmbeddingBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int8]), cIn.State.(*poly.Tensor[int8]), cPre.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		gIn, gW = poly.EmbeddingBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint64]), cIn.State.(*poly.Tensor[uint64]), cPre.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		gIn, gW = poly.EmbeddingBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint32]), cIn.State.(*poly.Tensor[uint32]), cPre.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		gIn, gW = poly.EmbeddingBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint16]), cIn.State.(*poly.Tensor[uint16]), cPre.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		gIn, gW = poly.EmbeddingBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint8]), cIn.State.(*poly.Tensor[uint8]), cPre.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for embedding backward")
	}

	res := map[string]interface{}{"gradInput": gIn, "gradWeights": gW}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomResidualBackward
func LoomResidualBackward(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	cGOut, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	cIn, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	cPre, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var gIn, gW interface{}
	switch cIn.DType {
	case poly.DTypeFloat64:
		gIn, gW = poly.ResidualBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float64]), cIn.State.(*poly.Tensor[float64]), cPre.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		gIn, gW = poly.ResidualBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float32]), cIn.State.(*poly.Tensor[float32]), cPre.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		gIn, gW = poly.ResidualBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int64]), cIn.State.(*poly.Tensor[int64]), cPre.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		gIn, gW = poly.ResidualBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int32]), cIn.State.(*poly.Tensor[int32]), cPre.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		gIn, gW = poly.ResidualBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int16]), cIn.State.(*poly.Tensor[int16]), cPre.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		gIn, gW = poly.ResidualBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int8]), cIn.State.(*poly.Tensor[int8]), cPre.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		gIn, gW = poly.ResidualBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint64]), cIn.State.(*poly.Tensor[uint64]), cPre.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		gIn, gW = poly.ResidualBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint32]), cIn.State.(*poly.Tensor[uint32]), cPre.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		gIn, gW = poly.ResidualBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint16]), cIn.State.(*poly.Tensor[uint16]), cPre.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		gIn, gW = poly.ResidualBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint8]), cIn.State.(*poly.Tensor[uint8]), cPre.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for residual backward")
	}

	res := map[string]interface{}{"gradInput": gIn, "gradWeights": gW}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomKMeansBackward
func LoomKMeansBackward(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	cGOut, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	cIn, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	cPre, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var gIn, gW interface{}
	switch cIn.DType {
	case poly.DTypeFloat64:
		gIn, gW = poly.KMeansBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float64]), cIn.State.(*poly.Tensor[float64]), cPre.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		gIn, gW = poly.KMeansBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float32]), cIn.State.(*poly.Tensor[float32]), cPre.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		gIn, gW = poly.KMeansBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int64]), cIn.State.(*poly.Tensor[int64]), cPre.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		gIn, gW = poly.KMeansBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int32]), cIn.State.(*poly.Tensor[int32]), cPre.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		gIn, gW = poly.KMeansBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int16]), cIn.State.(*poly.Tensor[int16]), cPre.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		gIn, gW = poly.KMeansBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int8]), cIn.State.(*poly.Tensor[int8]), cPre.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		gIn, gW = poly.KMeansBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint64]), cIn.State.(*poly.Tensor[uint64]), cPre.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		gIn, gW = poly.KMeansBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint32]), cIn.State.(*poly.Tensor[uint32]), cPre.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		gIn, gW = poly.KMeansBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint16]), cIn.State.(*poly.Tensor[uint16]), cPre.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		gIn, gW = poly.KMeansBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint8]), cIn.State.(*poly.Tensor[uint8]), cPre.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for kmeans backward")
	}

	res := map[string]interface{}{"gradInput": gIn, "gradWeights": gW}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}
//export LoomRNNBackward
func LoomRNNBackward(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	cGOut, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	cIn, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	cPre, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var gIn, gW interface{}
	switch cIn.DType {
	case poly.DTypeFloat64:
		gIn, gW = poly.RNNBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float64]), cIn.State.(*poly.Tensor[float64]), cPre.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		gIn, gW = poly.RNNBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float32]), cIn.State.(*poly.Tensor[float32]), cPre.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		gIn, gW = poly.RNNBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int64]), cIn.State.(*poly.Tensor[int64]), cPre.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		gIn, gW = poly.RNNBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int32]), cIn.State.(*poly.Tensor[int32]), cPre.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		gIn, gW = poly.RNNBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int16]), cIn.State.(*poly.Tensor[int16]), cPre.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		gIn, gW = poly.RNNBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int8]), cIn.State.(*poly.Tensor[int8]), cPre.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		gIn, gW = poly.RNNBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint64]), cIn.State.(*poly.Tensor[uint64]), cPre.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		gIn, gW = poly.RNNBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint32]), cIn.State.(*poly.Tensor[uint32]), cPre.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		gIn, gW = poly.RNNBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint16]), cIn.State.(*poly.Tensor[uint16]), cPre.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		gIn, gW = poly.RNNBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint8]), cIn.State.(*poly.Tensor[uint8]), cPre.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for rnn backward")
	}

	res := map[string]interface{}{"gradInput": gIn, "gradWeights": gW}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomLSTMBackward
func LoomLSTMBackward(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	cGOut, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	cIn, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	cPre, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var gIn, gW interface{}
	switch cIn.DType {
	case poly.DTypeFloat64:
		gIn, gW = poly.LSTMBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float64]), cIn.State.(*poly.Tensor[float64]), cPre.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		gIn, gW = poly.LSTMBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float32]), cIn.State.(*poly.Tensor[float32]), cPre.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		gIn, gW = poly.LSTMBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int64]), cIn.State.(*poly.Tensor[int64]), cPre.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		gIn, gW = poly.LSTMBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int32]), cIn.State.(*poly.Tensor[int32]), cPre.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		gIn, gW = poly.LSTMBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int16]), cIn.State.(*poly.Tensor[int16]), cPre.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		gIn, gW = poly.LSTMBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int8]), cIn.State.(*poly.Tensor[int8]), cPre.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		gIn, gW = poly.LSTMBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint64]), cIn.State.(*poly.Tensor[uint64]), cPre.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		gIn, gW = poly.LSTMBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint32]), cIn.State.(*poly.Tensor[uint32]), cPre.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		gIn, gW = poly.LSTMBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint16]), cIn.State.(*poly.Tensor[uint16]), cPre.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		gIn, gW = poly.LSTMBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint8]), cIn.State.(*poly.Tensor[uint8]), cPre.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for lstm backward")
	}

	res := map[string]interface{}{"gradInput": gIn, "gradWeights": gW}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomCNN1Backward
func LoomCNN1Backward(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	cGOut, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	cIn, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	cPre, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var gIn, gW interface{}
	switch cIn.DType {
	case poly.DTypeFloat64:
		gIn, gW = poly.CNN1BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float64]), cIn.State.(*poly.Tensor[float64]), cPre.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		gIn, gW = poly.CNN1BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float32]), cIn.State.(*poly.Tensor[float32]), cPre.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		gIn, gW = poly.CNN1BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int64]), cIn.State.(*poly.Tensor[int64]), cPre.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		gIn, gW = poly.CNN1BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int32]), cIn.State.(*poly.Tensor[int32]), cPre.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		gIn, gW = poly.CNN1BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int16]), cIn.State.(*poly.Tensor[int16]), cPre.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		gIn, gW = poly.CNN1BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int8]), cIn.State.(*poly.Tensor[int8]), cPre.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		gIn, gW = poly.CNN1BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint64]), cIn.State.(*poly.Tensor[uint64]), cPre.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		gIn, gW = poly.CNN1BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint32]), cIn.State.(*poly.Tensor[uint32]), cPre.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		gIn, gW = poly.CNN1BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint16]), cIn.State.(*poly.Tensor[uint16]), cPre.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		gIn, gW = poly.CNN1BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint8]), cIn.State.(*poly.Tensor[uint8]), cPre.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for cnn1 backward")
	}

	res := map[string]interface{}{"gradInput": gIn, "gradWeights": gW}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomCNN2Backward
func LoomCNN2Backward(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	cGOut, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	cIn, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	cPre, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var gIn, gW interface{}
	switch cIn.DType {
	case poly.DTypeFloat64:
		gIn, gW = poly.CNN2BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float64]), cIn.State.(*poly.Tensor[float64]), cPre.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		gIn, gW = poly.CNN2BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float32]), cIn.State.(*poly.Tensor[float32]), cPre.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		gIn, gW = poly.CNN2BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int64]), cIn.State.(*poly.Tensor[int64]), cPre.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		gIn, gW = poly.CNN2BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int32]), cIn.State.(*poly.Tensor[int32]), cPre.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		gIn, gW = poly.CNN2BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int16]), cIn.State.(*poly.Tensor[int16]), cPre.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		gIn, gW = poly.CNN2BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int8]), cIn.State.(*poly.Tensor[int8]), cPre.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		gIn, gW = poly.CNN2BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint64]), cIn.State.(*poly.Tensor[uint64]), cPre.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		gIn, gW = poly.CNN2BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint32]), cIn.State.(*poly.Tensor[uint32]), cPre.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		gIn, gW = poly.CNN2BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint16]), cIn.State.(*poly.Tensor[uint16]), cPre.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		gIn, gW = poly.CNN2BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint8]), cIn.State.(*poly.Tensor[uint8]), cPre.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for cnn2 backward")
	}

	res := map[string]interface{}{"gradInput": gIn, "gradWeights": gW}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomCNN3Backward
func LoomCNN3Backward(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	cGOut, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	cIn, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	cPre, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var gIn, gW interface{}
	switch cIn.DType {
	case poly.DTypeFloat64:
		gIn, gW = poly.CNN3BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float64]), cIn.State.(*poly.Tensor[float64]), cPre.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		gIn, gW = poly.CNN3BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float32]), cIn.State.(*poly.Tensor[float32]), cPre.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		gIn, gW = poly.CNN3BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int64]), cIn.State.(*poly.Tensor[int64]), cPre.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		gIn, gW = poly.CNN3BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int32]), cIn.State.(*poly.Tensor[int32]), cPre.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		gIn, gW = poly.CNN3BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int16]), cIn.State.(*poly.Tensor[int16]), cPre.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		gIn, gW = poly.CNN3BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int8]), cIn.State.(*poly.Tensor[int8]), cPre.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		gIn, gW = poly.CNN3BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint64]), cIn.State.(*poly.Tensor[uint64]), cPre.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		gIn, gW = poly.CNN3BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint32]), cIn.State.(*poly.Tensor[uint32]), cPre.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		gIn, gW = poly.CNN3BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint16]), cIn.State.(*poly.Tensor[uint16]), cPre.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		gIn, gW = poly.CNN3BackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint8]), cIn.State.(*poly.Tensor[uint8]), cPre.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for cnn3 backward")
	}

	res := map[string]interface{}{"gradInput": gIn, "gradWeights": gW}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomConvTransposed1DBackward
func LoomConvTransposed1DBackward(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	cGOut, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	cIn, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	cPre, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var gIn, gW interface{}
	switch cIn.DType {
	case poly.DTypeFloat64:
		gIn, gW = poly.ConvTransposed1DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float64]), cIn.State.(*poly.Tensor[float64]), cPre.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		gIn, gW = poly.ConvTransposed1DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float32]), cIn.State.(*poly.Tensor[float32]), cPre.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		gIn, gW = poly.ConvTransposed1DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int64]), cIn.State.(*poly.Tensor[int64]), cPre.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		gIn, gW = poly.ConvTransposed1DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int32]), cIn.State.(*poly.Tensor[int32]), cPre.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		gIn, gW = poly.ConvTransposed1DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int16]), cIn.State.(*poly.Tensor[int16]), cPre.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		gIn, gW = poly.ConvTransposed1DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int8]), cIn.State.(*poly.Tensor[int8]), cPre.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		gIn, gW = poly.ConvTransposed1DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint64]), cIn.State.(*poly.Tensor[uint64]), cPre.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		gIn, gW = poly.ConvTransposed1DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint32]), cIn.State.(*poly.Tensor[uint32]), cPre.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		gIn, gW = poly.ConvTransposed1DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint16]), cIn.State.(*poly.Tensor[uint16]), cPre.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		gIn, gW = poly.ConvTransposed1DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint8]), cIn.State.(*poly.Tensor[uint8]), cPre.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for convtransposed1d backward")
	}

	res := map[string]interface{}{"gradInput": gIn, "gradWeights": gW}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomConvTransposed2DBackward
func LoomConvTransposed2DBackward(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	cGOut, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	cIn, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	cPre, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var gIn, gW interface{}
	switch cIn.DType {
	case poly.DTypeFloat64:
		gIn, gW = poly.ConvTransposed2DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float64]), cIn.State.(*poly.Tensor[float64]), cPre.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		gIn, gW = poly.ConvTransposed2DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float32]), cIn.State.(*poly.Tensor[float32]), cPre.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		gIn, gW = poly.ConvTransposed2DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int64]), cIn.State.(*poly.Tensor[int64]), cPre.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		gIn, gW = poly.ConvTransposed2DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int32]), cIn.State.(*poly.Tensor[int32]), cPre.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		gIn, gW = poly.ConvTransposed2DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int16]), cIn.State.(*poly.Tensor[int16]), cPre.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		gIn, gW = poly.ConvTransposed2DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int8]), cIn.State.(*poly.Tensor[int8]), cPre.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		gIn, gW = poly.ConvTransposed2DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint64]), cIn.State.(*poly.Tensor[uint64]), cPre.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		gIn, gW = poly.ConvTransposed2DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint32]), cIn.State.(*poly.Tensor[uint32]), cPre.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		gIn, gW = poly.ConvTransposed2DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint16]), cIn.State.(*poly.Tensor[uint16]), cPre.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		gIn, gW = poly.ConvTransposed2DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint8]), cIn.State.(*poly.Tensor[uint8]), cPre.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for convtransposed2d backward")
	}

	res := map[string]interface{}{"gradInput": gIn, "gradWeights": gW}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomConvTransposed3DBackward
func LoomConvTransposed3DBackward(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	cGOut, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	cIn, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	cPre, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var gIn, gW interface{}
	switch cIn.DType {
	case poly.DTypeFloat64:
		gIn, gW = poly.ConvTransposed3DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float64]), cIn.State.(*poly.Tensor[float64]), cPre.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		gIn, gW = poly.ConvTransposed3DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[float32]), cIn.State.(*poly.Tensor[float32]), cPre.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		gIn, gW = poly.ConvTransposed3DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int64]), cIn.State.(*poly.Tensor[int64]), cPre.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		gIn, gW = poly.ConvTransposed3DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int32]), cIn.State.(*poly.Tensor[int32]), cPre.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		gIn, gW = poly.ConvTransposed3DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int16]), cIn.State.(*poly.Tensor[int16]), cPre.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		gIn, gW = poly.ConvTransposed3DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[int8]), cIn.State.(*poly.Tensor[int8]), cPre.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		gIn, gW = poly.ConvTransposed3DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint64]), cIn.State.(*poly.Tensor[uint64]), cPre.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		gIn, gW = poly.ConvTransposed3DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint32]), cIn.State.(*poly.Tensor[uint32]), cPre.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		gIn, gW = poly.ConvTransposed3DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint16]), cIn.State.(*poly.Tensor[uint16]), cPre.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		gIn, gW = poly.ConvTransposed3DBackwardPolymorphic(l, cGOut.State.(*poly.Tensor[uint8]), cIn.State.(*poly.Tensor[uint8]), cPre.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for convtransposed3d backward")
	}

	res := map[string]interface{}{"gradInput": gIn, "gradWeights": gW}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}
