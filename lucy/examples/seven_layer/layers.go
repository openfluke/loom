package sevenlayer

import (
	"fmt"
	"math"
	"strings"

	"github.com/openfluke/loom/poly"
)

const sevenLayersPerCell = 7

// sevenEndpoints must have length sevenLayersPerCell+1 (one endpoint per layer boundary).
func sevenEndpoints(v []int) []int {
	if len(v) != sevenLayersPerCell+1 {
		panic(fmt.Sprintf("seven_layer: want %d endpoints for %d layers, got %d", sevenLayersPerCell+1, sevenLayersPerCell, len(v)))
	}
	return v
}

func sinInput(shape ...int) *poly.Tensor[float32] {
	t := poly.NewTensor[float32](shape...)
	for i := range t.Data {
		t.Data[i] = 0.2 * float32(math.Sin(float64(i)*0.11+0.3))
	}
	return t
}

func sinTarget(net *poly.VolumetricNetwork, input *poly.Tensor[float32]) *poly.Tensor[float32] {
	out, _, _ := poly.ForwardPolymorphic(net, input)
	tgt := poly.NewTensor[float32](out.Shape...)
	for i := range tgt.Data {
		tgt.Data[i] = 0.5 + 0.3*float32(math.Sin(float64(i)*0.17))
	}
	return tgt
}

// flatEndpoints repeats width at every layer boundary so a multi-cell forward stack
// (z→y→x across the grid) never hands a mismatched activation to the next cell.
func flatEndpoints(width int) []int {
	v := make([]int, sevenLayersPerCell+1)
	for i := range v {
		v[i] = width
	}
	return sevenEndpoints(v)
}

func denseEndpoints(g GridSpec) []int {
	switch g.Cells() {
	case 1:
		return sevenEndpoints([]int{16, 24, 32, 48, 64, 48, 32, 8})
	case 8:
		// dim≥16 needed for SIMD crossover (see poly.DenseSimdMinDim).
		return flatEndpoints(16)
	default:
		// dim=32 for AVX2 crossover; MorphScaleForStackDepth keeps Uint wide types stable.
		return flatEndpoints(32)
	}
}

func swigluEndpoints(g GridSpec) []int {
	switch g.Cells() {
	case 1:
		return sevenEndpoints([]int{32, 32, 32, 32, 32, 32, 32, 16})
	case 8:
		// dim≥16 for SIMD crossover (see poly.DenseSimdMinDim).
		return flatEndpoints(16)
	default:
		// 3³: dim=32 for AVX2/NEON wins (same as Dense flat stacks).
		return flatEndpoints(32)
	}
}

func rnnEndpoints(g GridSpec) []int {
	switch g.Cells() {
	case 1:
		return sevenEndpoints([]int{16, 24, 32, 32, 32, 24, 16, 8})
	case 8:
		return flatEndpoints(8)
	default:
		return flatEndpoints(4)
	}
}

func cnnChannelEndpoints(g GridSpec) []int {
	switch g.Cells() {
	case 1:
		return sevenEndpoints([]int{3, 6, 8, 8, 8, 16, 16, 16})
	case 8:
		// inC=8 → kernelVol=24 at k=3 (AVX2 crossover).
		return flatEndpoints(8)
	default:
		// 3³: inC=16 → kernelVol=48; MorphScaleForStackDepth keeps Uint stacks stable.
		return flatEndpoints(16)
	}
}

func cnn3ChannelEndpoints(g GridSpec) []int {
	switch g.Cells() {
	case 1:
		return sevenEndpoints([]int{2, 4, 4, 4, 8, 8, 8, 8})
	default:
		return flatEndpoints(2)
	}
}

func cnnSpatial(g GridSpec) int {
	switch g.Cells() {
	case 1:
		return 16
	case 8:
		return 8
	default:
		return 4
	}
}

func cnn3Spatial(g GridSpec) (d, h, w int) {
	switch g.Cells() {
	case 1:
		return 8, 8, 8
	case 8:
		return 6, 6, 6
	default:
		// 3³ × 7 layers/cell: keep voxels modest (4³) so 189-layer stacks finish in-menu.
		return 4, 4, 4
	}
}

type mhaShape struct {
	dModel, heads, seq int
}

func mhaShapeFor(g GridSpec) mhaShape {
	switch g.Cells() {
	case 1:
		return mhaShape{64, 4, 8}
	case 8:
		return mhaShape{16, 2, 4}
	default:
		// d_model=32 head_dim=8 — SIMD crossover on projections (see poly.DenseSimdMinDim).
		return mhaShape{32, 4, 4}
	}
}

func embeddingDims(g GridSpec) []int {
	switch g.Cells() {
	case 1:
		return []int{32, 32, 32, 24, 16, 12, 8}
	case 8:
		w := 8
		return []int{w, w, w, w, w, w, w}
	default:
		w := 4
		return []int{w, w, w, w, w, w, w}
	}
}

func embeddingVocab(g GridSpec) int {
	switch g.Cells() {
	case 1:
		return 50
	default:
		return 20
	}
}

func embeddingSeqLen(g GridSpec) int {
	if g.Cells() == 1 {
		return 8
	}
	return 4
}

func residualDim(g GridSpec) int {
	switch g.Cells() {
	case 1:
		return 32
	case 8:
		return 16
	default:
		return 8
	}
}

func RunDense() bool {
	return runAllGrids(func(g GridSpec) LayerSuite {
		dims := denseEndpoints(g)
		acts := []string{"LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR"}
		return LayerSuite{
			Name:          "Dense",
			Grid:          g,
			PrimaryType:   poly.LayerDense,
			CheckpointTag: "seven_dense" + gridCheckpointSuffix(g),
			Banner: fmt.Sprintf("  Grid %s · pyramid %v (%d layers/cell, %d stack)",
				g, dims, sevenLayersPerCell, g.StackLayers()),
			BuildJSON: func(jsonDType string) []byte {
				var b strings.Builder
				writeNetworkHeader(&b, "loom-seven-dense", g)
				first := true
				forEachGridCell(g, func(z, y, x int) {
					for i := 0; i < sevenLayersPerCell; i++ {
						appendLayerJSON(&b, &first, fmt.Sprintf(
							`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"DENSE","activation":"%s","dtype":"%s","input_height":%d,"output_height":%d}`,
							z, y, x, i, acts[i], jsonDType, dims[i], dims[i+1],
						))
					}
				})
				b.WriteString(`]}`)
				return []byte(b.String())
			},
			MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, dims[0]) },
			MakeTarget: sinTarget,
		}
	})
}

func RunSwiGLU() bool {
	return runAllGrids(func(g GridSpec) LayerSuite {
		dims := swigluEndpoints(g)
		return LayerSuite{
			Name:          "SwiGLU",
			Grid:          g,
			PrimaryType:   poly.LayerSwiGLU,
			CheckpointTag: "seven_swiglu" + gridCheckpointSuffix(g),
			Banner: fmt.Sprintf("  Grid %s · 7 SwiGLU/cell (%d stack) — Plan 9 SIMD when GOARCH supports it",
				g, g.StackLayers()),
			BuildJSON: func(jsonDType string) []byte {
				var b strings.Builder
				writeNetworkHeader(&b, "loom-seven-swiglu", g)
				first := true
				forEachGridCell(g, func(z, y, x int) {
					for i := 0; i < sevenLayersPerCell; i++ {
						appendLayerJSON(&b, &first, fmt.Sprintf(
							`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"SWIGLU","activation":"RELU","dtype":"%s","input_height":%d,"output_height":%d}`,
							z, y, x, i, jsonDType, dims[i], dims[i+1],
						))
					}
				})
				b.WriteString(`]}`)
				return []byte(b.String())
			},
			MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, dims[0]) },
			MakeTarget: sinTarget,
		}
	})
}

func RunMHA() bool {
	return runAllGrids(func(g GridSpec) LayerSuite {
		m := mhaShapeFor(g)
		return LayerSuite{
			Name:          "MHA",
			Grid:          g,
			PrimaryType:   poly.LayerMultiHeadAttention,
			CheckpointTag: "seven_mha" + gridCheckpointSuffix(g),
			Banner: fmt.Sprintf("  Grid %s · 7 MHA/cell d=%d h=%d seq=%d — Plan 9 SIMD when GOARCH supports it",
				g, m.dModel, m.heads, m.seq),
			BuildJSON: func(jsonDType string) []byte {
				var b strings.Builder
				writeNetworkHeader(&b, "loom-seven-mha", g)
				first := true
				forEachGridCell(g, func(z, y, x int) {
					for i := 0; i < sevenLayersPerCell; i++ {
						appendLayerJSON(&b, &first, fmt.Sprintf(
							`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"MHA","activation":"RELU","dtype":"%s","d_model":%d,"num_heads":%d,"seq_length":%d}`,
							z, y, x, i, jsonDType, m.dModel, m.heads, m.seq,
						))
					}
				})
				b.WriteString(`]}`)
				return []byte(b.String())
			},
			MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, m.seq, m.dModel) },
			MakeTarget: sinTarget,
		}
	})
}

func RunCNN1() bool {
	return runGrids(CNN1Grids, func(g GridSpec) LayerSuite {
		ch := cnnChannelEndpoints(g)
		sp := cnnSpatial(g)
		return LayerSuite{
			Name:          "CNN1",
			Grid:          g,
			PrimaryType:   poly.LayerCNN1,
			CheckpointTag: "seven_cnn1" + gridCheckpointSuffix(g),
			Banner:        fmt.Sprintf("  Grid %s · 7 CNN1/cell %d×%d spatial — Plan 9 SIMD (AVX2/NEON)", g, sp, sp),
			BuildJSON: func(jsonDType string) []byte {
				var b strings.Builder
				writeNetworkHeader(&b, "loom-seven-cnn1", g)
				first := true
				forEachGridCell(g, func(z, y, x int) {
					for i := 0; i < sevenLayersPerCell; i++ {
						appendLayerJSON(&b, &first, fmt.Sprintf(
							`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"CNN1","activation":"RELU","dtype":"%s","input_channels":%d,"filters":%d,"input_height":%d,"output_height":%d,"kernel_size":3,"stride":1,"padding":1}`,
							z, y, x, i, jsonDType, ch[i], ch[i+1], sp, sp,
						))
					}
				})
				b.WriteString(`]}`)
				return []byte(b.String())
			},
			MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, ch[0], sp) },
			MakeTarget: sinTarget,
		}
	})
}

func RunCNN2() bool {
	return runGrids(CNN2Grids, func(g GridSpec) LayerSuite {
		ch := cnnChannelEndpoints(g)
		sp := cnnSpatial(g)
		return LayerSuite{
			Name:          "CNN2",
			Grid:          g,
			PrimaryType:   poly.LayerCNN2,
			CheckpointTag: "seven_cnn2" + gridCheckpointSuffix(g),
			Banner:        fmt.Sprintf("  Grid %s · 7 CNN2/cell %d×%d spatial — Plan 9 SIMD (AVX2/NEON)", g, sp, sp),
			BuildJSON: func(jsonDType string) []byte {
				var b strings.Builder
				writeNetworkHeader(&b, "loom-seven-cnn2", g)
				first := true
				forEachGridCell(g, func(z, y, x int) {
					for i := 0; i < sevenLayersPerCell; i++ {
						appendLayerJSON(&b, &first, fmt.Sprintf(
							`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"CNN2","activation":"RELU","dtype":"%s","input_channels":%d,"filters":%d,"input_height":%d,"input_width":%d,"output_height":%d,"output_width":%d,"kernel_size":3,"stride":1,"padding":1}`,
							z, y, x, i, jsonDType, ch[i], ch[i+1], sp, sp, sp, sp,
						))
					}
				})
				b.WriteString(`]}`)
				return []byte(b.String())
			},
			MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, ch[0], sp, sp) },
			MakeTarget: sinTarget,
		}
	})
}

func RunCNN3() bool {
	return runGrids(CNN3Grids, func(g GridSpec) LayerSuite {
		ch := cnn3ChannelEndpoints(g)
		d, h, w := cnn3Spatial(g)
		return LayerSuite{
			Name:          "CNN3",
			Grid:          g,
			PrimaryType:   poly.LayerCNN3,
			CheckpointTag: "seven_cnn3" + gridCheckpointSuffix(g),
			Banner:        fmt.Sprintf("  Grid %s · 7 CNN3/cell %d×%d×%d — Plan 9 SIMD (AVX2/NEON)", g, d, h, w),
			BuildJSON: func(jsonDType string) []byte {
				var b strings.Builder
				writeNetworkHeader(&b, "loom-seven-cnn3", g)
				first := true
				forEachGridCell(g, func(z, y, x int) {
					for i := 0; i < sevenLayersPerCell; i++ {
						appendLayerJSON(&b, &first, fmt.Sprintf(
							`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"CNN3","activation":"RELU","dtype":"%s","input_channels":%d,"filters":%d,"input_depth":%d,"input_height":%d,"input_width":%d,"output_depth":%d,"output_height":%d,"output_width":%d,"kernel_size":3,"stride":1,"padding":1}`,
							z, y, x, i, jsonDType, ch[i], ch[i+1], d, h, w, d, h, w,
						))
					}
				})
				b.WriteString(`]}`)
				return []byte(b.String())
			},
			MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, ch[0], d, h, w) },
			MakeTarget: sinTarget,
		}
	})
}

func RunRNN() bool {
	return runAllGrids(func(g GridSpec) LayerSuite {
		dims := rnnEndpoints(g)
		return LayerSuite{
			Name:          "RNN",
			Grid:          g,
			PrimaryType:   poly.LayerRNN,
			CheckpointTag: "seven_rnn" + gridCheckpointSuffix(g),
			Banner:        fmt.Sprintf("  Grid %s · 7 RNN/cell — Plan 9 SIMD (AVX2/NEON)", g),
			BuildJSON: func(jsonDType string) []byte {
				var b strings.Builder
				writeNetworkHeader(&b, "loom-seven-rnn", g)
				first := true
				forEachGridCell(g, func(z, y, x int) {
					for i := 0; i < sevenLayersPerCell; i++ {
						appendLayerJSON(&b, &first, fmt.Sprintf(
							`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"RNN","activation":"TANH","dtype":"%s","input_height":%d,"output_height":%d}`,
							z, y, x, i, jsonDType, dims[i], dims[i+1],
						))
					}
				})
				b.WriteString(`]}`)
				return []byte(b.String())
			},
			MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, dims[0]) },
			MakeTarget: sinTarget,
		}
	})
}

func RunLSTM() bool {
	return runAllGrids(func(g GridSpec) LayerSuite {
		dims := rnnEndpoints(g)
		return LayerSuite{
			Name:          "LSTM",
			Grid:          g,
			PrimaryType:   poly.LayerLSTM,
			CheckpointTag: "seven_lstm" + gridCheckpointSuffix(g),
			Banner:        fmt.Sprintf("  Grid %s · 7 LSTM/cell — Plan 9 SIMD (AVX2/NEON)", g),
			BuildJSON: func(jsonDType string) []byte {
				var b strings.Builder
				writeNetworkHeader(&b, "loom-seven-lstm", g)
				first := true
				forEachGridCell(g, func(z, y, x int) {
					for i := 0; i < sevenLayersPerCell; i++ {
						appendLayerJSON(&b, &first, fmt.Sprintf(
							`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"LSTM","activation":"TANH","dtype":"%s","input_height":%d,"output_height":%d}`,
							z, y, x, i, jsonDType, dims[i], dims[i+1],
						))
					}
				})
				b.WriteString(`]}`)
				return []byte(b.String())
			},
			MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, dims[0]) },
			MakeTarget: sinTarget,
		}
	})
}

func RunEmbedding() bool {
	return runAllGrids(embeddingSuite)
}

func embeddingSuite(g GridSpec) LayerSuite {
	{
		vocab := embeddingVocab(g)
		acts := []string{"RELU", "RELU", "RELU", "RELU", "RELU", "SIGMOID"}
		// Multi-cell: embedding only at stack origin; later cells are float→float DENSE.
		var banner string
		switch g.Cells() {
		case 1:
			banner = fmt.Sprintf("  Grid %s · EMBEDDING + 6 DENSE — ASM not implemented", g)
		default:
			banner = fmt.Sprintf("  Grid %s · EMBEDDING@(0,0,0)+6 DENSE; other cells 7× DENSE — ASM not implemented", g)
		}
		return LayerSuite{
			Name:          "Embedding",
			Grid:          g,
			PrimaryType:   poly.LayerEmbedding,
			CheckpointTag: "seven_embedding" + gridCheckpointSuffix(g),
			Banner:        banner,
			BuildJSON: func(jsonDType string) []byte {
				dims := embeddingDims(g)
				// Non-origin cells must stay at the embedding output width so the
				// cross-cell forward (origin → next cell) never hands a mismatched
				// activation to the next Dense layer.
				denseOnly := flatEndpoints(dims[len(dims)-1])
				var b strings.Builder
				writeNetworkHeader(&b, "loom-seven-embedding", g)
				first := true
				forEachGridCell(g, func(z, y, x int) {
					if isStackOrigin(z, y, x) {
						appendLayerJSON(&b, &first, fmt.Sprintf(
							`{"z":%d,"y":%d,"x":%d,"l":0,"type":"EMBEDDING","dtype":"%s","vocab_size":%d,"embedding_dim":%d}`,
							z, y, x, jsonDType, vocab, dims[0],
						))
						for i := 0; i < len(dims)-1; i++ {
							appendLayerJSON(&b, &first, fmt.Sprintf(
								`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"DENSE","activation":"%s","dtype":"%s","input_height":%d,"output_height":%d}`,
								z, y, x, i+1, acts[i], jsonDType, dims[i], dims[i+1],
							))
						}
						return
					}
					for i := 0; i < sevenLayersPerCell; i++ {
						appendLayerJSON(&b, &first, fmt.Sprintf(
							`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"DENSE","activation":"%s","dtype":"%s","input_height":%d,"output_height":%d}`,
							z, y, x, i, acts[i%len(acts)], jsonDType, denseOnly[i], denseOnly[i+1],
						))
					}
				})
				b.WriteString(`]}`)
				return []byte(b.String())
			},
			MakeInput: func() *poly.Tensor[float32] {
				seq := embeddingSeqLen(g)
				t := poly.NewTensor[float32](seq, 1)
				for i := range t.Data {
					t.Data[i] = float32(i % vocab)
				}
				return t
			},
			MakeTarget: sinTarget,
		}
	}
}

func RunResidual() bool {
	return runAllGrids(func(g GridSpec) LayerSuite {
		dim := residualDim(g)
		return LayerSuite{
			Name:          "Residual",
			Grid:          g,
			PrimaryType:   poly.LayerResidual,
			CheckpointTag: "seven_residual" + gridCheckpointSuffix(g),
			Banner: fmt.Sprintf("  Grid %s · 7 RESIDUAL/cell %d→%d (no nested sequential_layers)",
				g, dim, dim),
			BuildJSON: func(jsonDType string) []byte {
				var b strings.Builder
				writeNetworkHeader(&b, "loom-seven-residual", g)
				first := true
				forEachGridCell(g, func(z, y, x int) {
					for i := 0; i < sevenLayersPerCell; i++ {
						appendLayerJSON(&b, &first, fmt.Sprintf(
							`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"RESIDUAL","dtype":"%s","input_height":%d,"output_height":%d}`,
							z, y, x, i, jsonDType, dim, dim,
						))
					}
				})
				b.WriteString(`]}`)
				return []byte(b.String())
			},
			MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, dim) },
			MakeTarget: sinTarget,
		}
	})
}
