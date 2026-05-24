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
	return poly.NetworkAccountingWeightBytes(net)
}

func layerWeightBytes(l *poly.VolumetricLayer) uint64 {
	var n uint64
	if l.WeightStore != nil {
		n += l.WeightStore.AccountingWeightBytes(l.DType)
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
