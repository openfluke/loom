package matmul

var (
	dotTileF32 = dotTileF32Go
	dotTileF64 = dotTileF64Go
)

// ForwardTiledF32 computes preAct = input @ weights^T using the same 2D output/input
// tiling and float64 accumulation as poly.denseForwardTiledSerial/Parallel.
func ForwardTiledF32(preAct, input, weights []float32, batch, inDim, outDim int, mc bool, tileSize int) {
	forwardTiledF32(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTileF32)
}

// ForwardTiledF64 is the float64 element variant of ForwardTiledF32.
func ForwardTiledF64(preAct, input, weights []float64, batch, inDim, outDim int, mc bool, tileSize int) {
	forwardTiledF64(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTileF64)
}

// ForwardTiledNumeric is the integer element variant (float64 accumulate, cast to T).
func ForwardTiledNumeric[T Numeric](preAct, input, weights []T, batch, inDim, outDim int, mc bool, tileSize int) {
	if tileSize <= 0 {
		tileSize = 32
	}
	run := func(oTile, oEnd int) {
		for iTile := 0; iTile < inDim; iTile += tileSize {
			iEnd := iTile + tileSize
			if iEnd > inDim {
				iEnd = inDim
			}
			for b := 0; b < batch; b++ {
				inRow := input[b*inDim : (b+1)*inDim]
				outRow := preAct[b*outDim : (b+1)*outDim]
				for o := oTile; o < oEnd; o++ {
					prev := float64(0)
					if iTile > 0 {
						prev = float64(outRow[o])
					}
					wRow := weights[o*inDim : (o+1)*inDim]
					sum := dotTileNumeric(inRow, wRow, iTile, iEnd, prev)
					outRow[o] = T(sum)
				}
			}
		}
	}
	if mc {
		OverOutputTiles(outDim, tileSize, run)
	} else {
		run(0, outDim)
	}
}

type dotTileFnF32 func(inRow, wRow []float32, i0, i1 int, prev float64) float64
type dotTileFnF64 func(inRow, wRow []float64, i0, i1 int, prev float64) float64

func forwardTiledF32(
	preAct, input, weights []float32,
	batch, inDim, outDim int,
	mc bool, tileSize int,
	dotTile dotTileFnF32,
) {
	if tileSize <= 0 {
		tileSize = 32
	}
	run := func(oTile, oEnd int) {
		for iTile := 0; iTile < inDim; iTile += tileSize {
			iEnd := iTile + tileSize
			if iEnd > inDim {
				iEnd = inDim
			}
			for b := 0; b < batch; b++ {
				inRow := input[b*inDim : (b+1)*inDim]
				outRow := preAct[b*outDim : (b+1)*outDim]
				for o := oTile; o < oEnd; o++ {
					prev := float64(0)
					if iTile > 0 {
						prev = float64(outRow[o])
					}
					wRow := weights[o*inDim : (o+1)*inDim]
					outRow[o] = float32(dotTile(inRow, wRow, iTile, iEnd, prev))
				}
			}
		}
	}
	if mc {
		OverOutputTiles(outDim, tileSize, run)
	} else {
		run(0, outDim)
	}
}

func forwardTiledF64(
	preAct, input, weights []float64,
	batch, inDim, outDim int,
	mc bool, tileSize int,
	dotTile dotTileFnF64,
) {
	if tileSize <= 0 {
		tileSize = 32
	}
	run := func(oTile, oEnd int) {
		for iTile := 0; iTile < inDim; iTile += tileSize {
			iEnd := iTile + tileSize
			if iEnd > inDim {
				iEnd = inDim
			}
			for b := 0; b < batch; b++ {
				inRow := input[b*inDim : (b+1)*inDim]
				outRow := preAct[b*outDim : (b+1)*outDim]
				for o := oTile; o < oEnd; o++ {
					prev := float64(0)
					if iTile > 0 {
						prev = outRow[o]
					}
					wRow := weights[o*inDim : (o+1)*inDim]
					outRow[o] = dotTile(inRow, wRow, iTile, iEnd, prev)
				}
			}
		}
	}
	if mc {
		OverOutputTiles(outDim, tileSize, run)
	} else {
		run(0, outDim)
	}
}

func dotTileF32Go(inRow, wRow []float32, i0, i1 int, prev float64) float64 {
	sum := prev
	for i := i0; i < i1; i++ {
		sum += float64(inRow[i]) * float64(wRow[i])
	}
	return sum
}

func dotTileF64Go(inRow, wRow []float64, i0, i1 int, prev float64) float64 {
	sum := prev
	for i := i0; i < i1; i++ {
		sum += inRow[i] * wRow[i]
	}
	return sum
}

func dotTileNumeric[T Numeric](inRow, wRow []T, i0, i1 int, prev float64) float64 {
	sum := prev
	for i := i0; i < i1; i++ {
		sum += float64(inRow[i]) * float64(wRow[i])
	}
	return sum
}
