package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"encoding/base64"
	"encoding/json"

	"github.com/openfluke/loom/poly"
	"github.com/openfluke/loom/poly/accel"
)

//export LoomDiscoverAccel
func LoomDiscoverAccel(intelSO *C.char) C.longlong {
	cfg := accel.AccelConfig{}
	if intelSO != nil {
		cfg.IntelSO = C.GoString(intelSO)
	}
	reg, err := poly.DiscoverAccel(cfg)
	if err != nil {
		return -1
	}
	networkMu.Lock()
	id := accelRegistryNextID
	accelRegistryNextID++
	accelRegistries[id] = reg
	networkMu.Unlock()
	return C.longlong(id)
}

//export LoomFreeAccelRegistry
func LoomFreeAccelRegistry(accelHandle C.longlong) {
	networkMu.Lock()
	if reg, ok := accelRegistries[int64(accelHandle)]; ok {
		reg.Close()
		delete(accelRegistries, int64(accelHandle))
	}
	networkMu.Unlock()
}

//export LoomNetworkAttachAccel
func LoomNetworkAttachAccel(networkHandle C.longlong, accelHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	reg, ok := getAccelRegistry(int64(accelHandle))
	if !ok {
		return errJSON("invalid accel registry handle")
	}
	n.Accel = reg
	return C.CString(`{"status":"ok"}`)
}

//export LoomNetworkReleaseAccel
func LoomNetworkReleaseAccel(networkHandle C.longlong) {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return
	}
	n.ReleaseAccel()
}

//export LoomSetLayerExecTarget
func LoomSetLayerExecTarget(networkHandle C.longlong, layerIdx C.int, target C.int) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	n.Layers[int(layerIdx)].ExecTarget = accel.ExecTarget(target)
	return C.CString(`{"status":"ok"}`)
}

//export LoomSyncToAccel
func LoomSyncToAccel(networkHandle C.longlong, sizeLabel *C.char) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	label := "medium"
	if sizeLabel != nil {
		if s := C.GoString(sizeLabel); s != "" {
			label = s
		}
	}
	if err := n.SyncToAccel(label); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status":"ok"}`)
}

//export LoomLayerWeightBytesForAccel
func LoomLayerWeightBytesForAccel(networkHandle C.longlong, layerIdx C.int) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	b := poly.LayerWeightBytesForAccel(&n.Layers[int(layerIdx)])
	out, _ := json.Marshal(map[string]interface{}{
		"bytes": base64.StdEncoding.EncodeToString(b),
		"len":   len(b),
	})
	return C.CString(string(out))
}

//export LoomDispatchAccelForward
func LoomDispatchAccelForward(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	c, ok := getStepContainer(int64(inputHandle))
	if !ok {
		return errJSON("invalid input handle")
	}
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	var ran bool
	switch c.DType {
	case poly.DTypeFloat64:
		var p, o *poly.Tensor[float64]
		p, o, ran = poly.DispatchAccelForward(l, c.State.(*poly.Tensor[float64]))
		pre, post = p, o
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		var p, o *poly.Tensor[float32]
		p, o, ran = poly.DispatchAccelForward(l, c.State.(*poly.Tensor[float32]))
		pre, post = p, o
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		var p, o *poly.Tensor[int8]
		p, o, ran = poly.DispatchAccelForward(l, c.State.(*poly.Tensor[int8]))
		pre, post = p, o
	default:
		var p, o *poly.Tensor[float32]
		if t, ok := c.State.(*poly.Tensor[float32]); ok {
			p, o, ran = poly.DispatchAccelForward(l, t)
			pre, post = p, o
		} else {
			return errJSON("unsupported dtype for accel forward")
		}
	}
	if !ran {
		return errJSON("accel forward not available for layer")
	}
	res, _ := json.Marshal(map[string]interface{}{"pre": pre, "post": post})
	return C.CString(string(res))
}
