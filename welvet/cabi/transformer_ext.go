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

func getTransformerF32(handle int64) (*poly.Transformer[float32], bool) {
	tr, ok := getTransformer(handle)
	if !ok {
		return nil, false
	}
	t, ok := tr.(*poly.Transformer[float32])
	return t, ok
}

//export LoomSetTransformerForwardMode
func LoomSetTransformerForwardMode(transformerHandle C.longlong, mode C.int) *C.char {
	tr, ok := getTransformerF32(int64(transformerHandle))
	if !ok {
		return errJSON("invalid transformer handle")
	}
	tr.ForwardMode = poly.TransformerForwardMode(mode)
	return C.CString(`{"status":"ok"}`)
}

//export LoomGetTransformerForwardMode
func LoomGetTransformerForwardMode(transformerHandle C.longlong) C.int {
	tr, ok := getTransformerF32(int64(transformerHandle))
	if !ok {
		return -1
	}
	return C.int(tr.ForwardMode)
}

//export LoomTransformerCPUForwardQueueStepTotal
func LoomTransformerCPUForwardQueueStepTotal(transformerHandle C.longlong) C.int {
	tr, ok := getTransformerF32(int64(transformerHandle))
	if !ok {
		return -1
	}
	return C.int(tr.CPUForwardQueueStepTotal())
}

//export LoomTransformerBeginCPUForwardQueue
func LoomTransformerBeginCPUForwardQueue(transformerHandle C.longlong, inputJSON *C.char) *C.char {
	tr, ok := getTransformerF32(int64(transformerHandle))
	if !ok {
		return errJSON("invalid transformer handle")
	}
	var in poly.Tensor[float32]
	if err := json.Unmarshal([]byte(C.GoString(inputJSON)), &in); err != nil {
		return errJSON("invalid input tensor JSON")
	}
	if err := tr.BeginCPUForwardQueue(&in); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status":"ok"}`)
}

//export LoomTransformerCPUForwardQueueTick
func LoomTransformerCPUForwardQueueTick(transformerHandle C.longlong) *C.char {
	tr, ok := getTransformerF32(int64(transformerHandle))
	if !ok {
		return errJSON("invalid transformer handle")
	}
	done, label, err := tr.CPUForwardQueueTick()
	if err != nil {
		return errJSON(err.Error())
	}
	out, _ := json.Marshal(map[string]interface{}{
		"done":  done,
		"label": label,
	})
	return C.CString(string(out))
}

//export LoomTransformerCPUForwardQueueResult
func LoomTransformerCPUForwardQueueResult(transformerHandle C.longlong) *C.char {
	tr, ok := getTransformerF32(int64(transformerHandle))
	if !ok {
		return errJSON("invalid transformer handle")
	}
	res := tr.CPUForwardQueueResult()
	if res == nil {
		return errJSON("queue not finished or not started")
	}
	b, err := json.Marshal(res)
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomTransformerCPUForwardQueueDiscard
func LoomTransformerCPUForwardQueueDiscard(transformerHandle C.longlong) *C.char {
	tr, ok := getTransformerF32(int64(transformerHandle))
	if !ok {
		return errJSON("invalid transformer handle")
	}
	tr.CPUForwardQueueDiscard()
	return C.CString(`{"status":"ok"}`)
}

//export LoomTransformerComparePrefillToNormal
func LoomTransformerComparePrefillToNormal(transformerHandle C.longlong, inputJSON *C.char) *C.char {
	tr, ok := getTransformerF32(int64(transformerHandle))
	if !ok {
		return errJSON("invalid transformer handle")
	}
	var in poly.Tensor[float32]
	if err := json.Unmarshal([]byte(C.GoString(inputJSON)), &in); err != nil {
		return errJSON("invalid input tensor JSON")
	}
	bench, maxDiff := tr.ComparePrefillToNormal(&in)
	out, _ := json.Marshal(map[string]interface{}{
		"bench":   bench,
		"maxDiff": maxDiff,
	})
	return C.CString(string(out))
}

//export LoomTransformerTakePipelineForwardStats
func LoomTransformerTakePipelineForwardStats(transformerHandle C.longlong) *C.char {
	tr, ok := getTransformerF32(int64(transformerHandle))
	if !ok {
		return errJSON("invalid transformer handle")
	}
	stats := tr.TakePipelineForwardStats()
	b, err := json.Marshal(stats)
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomTransformerLastPipelineForwardStats
func LoomTransformerLastPipelineForwardStats(transformerHandle C.longlong) *C.char {
	tr, ok := getTransformerF32(int64(transformerHandle))
	if !ok {
		return errJSON("invalid transformer handle")
	}
	stats := tr.LastPipelineForwardStats()
	b, err := json.Marshal(stats)
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

//export LoomForwardBenchResultJSON
func LoomForwardBenchResultJSON() *C.char {
	b, _ := json.Marshal(poly.ForwardBenchResult{})
	return C.CString(string(b))
}

//export LoomLayerActionRecordJSON
func LoomLayerActionRecordJSON() *C.char {
	b, _ := json.Marshal(poly.LayerActionRecord{})
	return C.CString(string(b))
}

//export LoomSetTransformerForwardStepObserver
func LoomSetTransformerForwardStepObserver(transformerHandle C.longlong, debug C.int) *C.char {
	tr, ok := getTransformerF32(int64(transformerHandle))
	if !ok {
		return errJSON("invalid transformer handle")
	}
	if debug == 0 {
		tr.SetForwardStepObserver(nil)
		tr.ForwardStepDebug = false
	} else {
		tr.ForwardStepDebug = true
		tr.SetForwardStepObserver(func(step, total int, label string) {
			_ = step
			_ = total
			_ = label
		})
	}
	return C.CString(`{"status":"ok"}`)
}

//export LoomTransformerLayerTraceRecords
func LoomTransformerLayerTraceRecords(transformerHandle C.longlong) *C.char {
	tr, ok := getTransformerF32(int64(transformerHandle))
	if !ok {
		return errJSON("invalid transformer handle")
	}
	recs := tr.LayerTraceRecords()
	b, err := json.Marshal(recs)
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}
