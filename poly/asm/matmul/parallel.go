package matmul

import (
	"runtime"
	"sync"
)

// OverOutputTiles runs fn on disjoint [o0,o1) output ranges in parallel.
func OverOutputTiles(outDim, tileSize int, fn func(o0, o1 int)) {
	if tileSize <= 0 {
		tileSize = 32
	}
	workers := runtime.NumCPU()
	sem := make(chan struct{}, workers)
	var wg sync.WaitGroup

	for oTile := 0; oTile < outDim; oTile += tileSize {
		o0 := oTile
		o1 := oTile + tileSize
		if o1 > outDim {
			o1 = outDim
		}
		sem <- struct{}{}
		wg.Add(1)
		go func(o0, o1 int) {
			defer func() { <-sem; wg.Done() }()
			fn(o0, o1)
		}(o0, o1)
	}
	wg.Wait()
}
