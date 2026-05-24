package sevenlayer

import (
	"fmt"
	"runtime"

	"github.com/openfluke/loom/poly"
)

// memSnapshot is a point-in-time Go heap / runtime memory reading.
type memSnapshot struct {
	HeapAlloc uint64
	Sys       uint64
}

func readMemSnapshot() memSnapshot {
	var m runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m)
	return memSnapshot{HeapAlloc: m.HeapAlloc, Sys: m.Sys}
}

func formatBytes(n uint64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.2f GiB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.2f MiB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.2f KiB", float64(n)/(1<<10))
	default:
		return fmt.Sprintf("%d B", n)
	}
}

func networkWeightBytes(net *poly.VolumetricNetwork) uint64 {
	if net == nil {
		return 0
	}
	var total uint64
	for i := range net.Layers {
		total += layerWeightBytes(&net.Layers[i])
	}
	return total
}

func layerWeightBytes(l *poly.VolumetricLayer) uint64 {
	var n uint64
	if l.WeightStore != nil {
		n += uint64(len(l.WeightStore.Master)) * 4
		n += nativeVersionsBytes(l.WeightStore, l.DType)
	}
	for i := range l.ParallelBranches {
		n += layerWeightBytes(&l.ParallelBranches[i])
	}
	for i := range l.SequentialLayers {
		n += layerWeightBytes(&l.SequentialLayers[i])
	}
	if l.MetaObservedLayer != nil {
		n += layerWeightBytes(l.MetaObservedLayer)
	}
	return n
}

func nativeVersionsBytes(ws *poly.WeightStore, dt poly.DType) uint64 {
	if ws == nil {
		return 0
	}
	active := ws.Versions[dt]
	if active == nil {
		return 0
	}
	switch w := active.(type) {
	case []float64:
		return uint64(len(w)) * 8
	case []int64:
		return uint64(len(w)) * 8
	case []uint64:
		return uint64(len(w)) * 8
	case []int32:
		return uint64(len(w)) * 4
	case []uint32:
		return uint64(len(w)) * 4
	case []int16:
		return uint64(len(w)) * 2
	case []uint16:
		return uint64(len(w)) * 2
	case []uint8:
		return uint64(len(w))
	default:
		return 0
	}
}
