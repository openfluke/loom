package matmul

// ForwardNumeric runs ForwardSC or ForwardMC for integer element types.
func ForwardNumeric(preAct, input, weights any, batch, inDim, outDim int, mc bool, tileSize int) {
	switch w := weights.(type) {
	case []int:
		p, in := preAct.([]int), input.([]int)
		if mc {
			ForwardMC(p, in, w, batch, inDim, outDim, tileSize)
		} else {
			ForwardSC(p, in, w, batch, inDim, outDim)
		}
	case []int8:
		p, in := preAct.([]int8), input.([]int8)
		if mc {
			ForwardMC(p, in, w, batch, inDim, outDim, tileSize)
		} else {
			ForwardSC(p, in, w, batch, inDim, outDim)
		}
	case []int16:
		p, in := preAct.([]int16), input.([]int16)
		if mc {
			ForwardMC(p, in, w, batch, inDim, outDim, tileSize)
		} else {
			ForwardSC(p, in, w, batch, inDim, outDim)
		}
	case []int32:
		p, in := preAct.([]int32), input.([]int32)
		if mc {
			ForwardMC(p, in, w, batch, inDim, outDim, tileSize)
		} else {
			ForwardSC(p, in, w, batch, inDim, outDim)
		}
	case []int64:
		p, in := preAct.([]int64), input.([]int64)
		if mc {
			ForwardMC(p, in, w, batch, inDim, outDim, tileSize)
		} else {
			ForwardSC(p, in, w, batch, inDim, outDim)
		}
	case []uint:
		p, in := preAct.([]uint), input.([]uint)
		if mc {
			ForwardMC(p, in, w, batch, inDim, outDim, tileSize)
		} else {
			ForwardSC(p, in, w, batch, inDim, outDim)
		}
	case []uint8:
		p, in := preAct.([]uint8), input.([]uint8)
		if mc {
			ForwardMC(p, in, w, batch, inDim, outDim, tileSize)
		} else {
			ForwardSC(p, in, w, batch, inDim, outDim)
		}
	case []uint16:
		p, in := preAct.([]uint16), input.([]uint16)
		if mc {
			ForwardMC(p, in, w, batch, inDim, outDim, tileSize)
		} else {
			ForwardSC(p, in, w, batch, inDim, outDim)
		}
	case []uint32:
		p, in := preAct.([]uint32), input.([]uint32)
		if mc {
			ForwardMC(p, in, w, batch, inDim, outDim, tileSize)
		} else {
			ForwardSC(p, in, w, batch, inDim, outDim)
		}
	case []uint64:
		p, in := preAct.([]uint64), input.([]uint64)
		if mc {
			ForwardMC(p, in, w, batch, inDim, outDim, tileSize)
		} else {
			ForwardSC(p, in, w, batch, inDim, outDim)
		}
	}
}
