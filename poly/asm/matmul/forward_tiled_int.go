package matmul

import "github.com/openfluke/loom/poly/asm/dot"

var (
	dotTileI8  = dotTileI8Go
	dotTileI16 = dotTileI16Go
	dotTileI32 = dotTileI32Go
	dotTileI64 = dotTileI64Go
	dotTileU8  = dotTileU8Go
	dotTileU16 = dotTileU16Go
	dotTileU32 = dotTileU32Go
	dotTileU64 = dotTileU64Go
)

func ForwardTiledI8(preAct, input, weights []int8, batch, inDim, outDim int, mc bool, tileSize int) {
	forwardTiledInt8(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTileI8)
}

func ForwardTiledI16(preAct, input, weights []int16, batch, inDim, outDim int, mc bool, tileSize int) {
	forwardTiledInt16(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTileI16)
}

func ForwardTiledI32(preAct, input, weights []int32, batch, inDim, outDim int, mc bool, tileSize int) {
	forwardTiledInt32(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTileI32)
}

func ForwardTiledI64(preAct, input, weights []int64, batch, inDim, outDim int, mc bool, tileSize int) {
	forwardTiledInt64(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTileI64)
}

func ForwardTiledU8(preAct, input, weights []uint8, batch, inDim, outDim int, mc bool, tileSize int) {
	forwardTiledUint8(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTileU8)
}

func ForwardTiledU16(preAct, input, weights []uint16, batch, inDim, outDim int, mc bool, tileSize int) {
	forwardTiledUint16(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTileU16)
}

func ForwardTiledU32(preAct, input, weights []uint32, batch, inDim, outDim int, mc bool, tileSize int) {
	forwardTiledUint32(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTileU32)
}

func ForwardTiledU64(preAct, input, weights []uint64, batch, inDim, outDim int, mc bool, tileSize int) {
	forwardTiledUint64(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTileU64)
}

type dotTileFnI8 func(inRow, wRow []int8, i0, i1 int, prev float64) float64
type dotTileFnI16 func(inRow, wRow []int16, i0, i1 int, prev float64) float64
type dotTileFnI32 func(inRow, wRow []int32, i0, i1 int, prev float64) float64
type dotTileFnI64 func(inRow, wRow []int64, i0, i1 int, prev float64) float64
type dotTileFnU8 func(inRow, wRow []uint8, i0, i1 int, prev float64) float64
type dotTileFnU16 func(inRow, wRow []uint16, i0, i1 int, prev float64) float64
type dotTileFnU32 func(inRow, wRow []uint32, i0, i1 int, prev float64) float64
type dotTileFnU64 func(inRow, wRow []uint64, i0, i1 int, prev float64) float64

func forwardTiledInt8(preAct, input, weights []int8, batch, inDim, outDim int, mc bool, tileSize int, dotTile dotTileFnI8) {
	forwardTiledInt(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTile)
}

func forwardTiledInt16(preAct, input, weights []int16, batch, inDim, outDim int, mc bool, tileSize int, dotTile dotTileFnI16) {
	forwardTiledInt(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTile)
}

func forwardTiledInt32(preAct, input, weights []int32, batch, inDim, outDim int, mc bool, tileSize int, dotTile dotTileFnI32) {
	forwardTiledInt(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTile)
}

func forwardTiledInt64(preAct, input, weights []int64, batch, inDim, outDim int, mc bool, tileSize int, dotTile dotTileFnI64) {
	forwardTiledInt(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTile)
}

func forwardTiledUint8(preAct, input, weights []uint8, batch, inDim, outDim int, mc bool, tileSize int, dotTile dotTileFnU8) {
	forwardTiledInt(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTile)
}

func forwardTiledUint16(preAct, input, weights []uint16, batch, inDim, outDim int, mc bool, tileSize int, dotTile dotTileFnU16) {
	forwardTiledInt(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTile)
}

func forwardTiledUint32(preAct, input, weights []uint32, batch, inDim, outDim int, mc bool, tileSize int, dotTile dotTileFnU32) {
	forwardTiledInt(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTile)
}

func forwardTiledUint64(preAct, input, weights []uint64, batch, inDim, outDim int, mc bool, tileSize int, dotTile dotTileFnU64) {
	forwardTiledInt(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTile)
}

func forwardTiledInt[T Numeric](
	preAct, input, weights []T,
	batch, inDim, outDim int,
	mc bool, tileSize int,
	dotTile func(inRow, wRow []T, i0, i1 int, prev float64) float64,
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
					outRow[o] = T(dotTile(inRow, wRow, iTile, iEnd, prev))
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

func dotTileI8Go(inRow, wRow []int8, i0, i1 int, prev float64) float64 {
	return prev + dot.I8TileAccF64(inRow[i0:i1], wRow[i0:i1], i1-i0)
}

func dotTileI16Go(inRow, wRow []int16, i0, i1 int, prev float64) float64 {
	return prev + dot.I16TileAccF64(inRow[i0:i1], wRow[i0:i1], i1-i0)
}

func dotTileI32Go(inRow, wRow []int32, i0, i1 int, prev float64) float64 {
	return prev + dot.I32TileAccF64(inRow[i0:i1], wRow[i0:i1], i1-i0)
}

func dotTileI64Go(inRow, wRow []int64, i0, i1 int, prev float64) float64 {
	return prev + dot.I64TileAccF64(inRow[i0:i1], wRow[i0:i1], i1-i0)
}

func dotTileU8Go(inRow, wRow []uint8, i0, i1 int, prev float64) float64 {
	return prev + dot.U8TileAccF64(inRow[i0:i1], wRow[i0:i1], i1-i0)
}

func dotTileU16Go(inRow, wRow []uint16, i0, i1 int, prev float64) float64 {
	return prev + dot.U16TileAccF64(inRow[i0:i1], wRow[i0:i1], i1-i0)
}

func dotTileU32Go(inRow, wRow []uint32, i0, i1 int, prev float64) float64 {
	return prev + dot.U32TileAccF64(inRow[i0:i1], wRow[i0:i1], i1-i0)
}

func dotTileU64Go(inRow, wRow []uint64, i0, i1 int, prev float64) float64 {
	return prev + dot.U64TileAccF64(inRow[i0:i1], wRow[i0:i1], i1-i0)
}
