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

func writeNetworkHeader(b *strings.Builder, id string) {
	b.WriteString(fmt.Sprintf(
		`{"id":"%s","depth":1,"rows":1,"cols":1,"layers_per_cell":%d,"layers":[`,
		id, sevenLayersPerCell,
	))
}

func RunDense() bool {
	dims := sevenEndpoints([]int{16, 24, 32, 48, 64, 48, 32, 8})
	// LINEAR stack: deep ReLU pyramids stall gradients; scaling in prepareTrainingNet helps 7-wide cells.
	acts := []string{"LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR"}
	return RunLayerSuite(LayerSuite{
		Name:          "Dense",
		PrimaryType:   poly.LayerDense,
		CheckpointTag: "seven_dense",
		Banner:        fmt.Sprintf("  Pyramid %v (%d flat DENSE layers)", dims, sevenLayersPerCell),
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "loom-seven-dense")
			for i := 0; i < sevenLayersPerCell; i++ {
				if i > 0 {
					b.WriteByte(',')
				}
				b.WriteString(fmt.Sprintf(
					`{"z":0,"y":0,"x":0,"l":%d,"type":"DENSE","activation":"%s","dtype":"%s","input_height":%d,"output_height":%d}`,
					i, acts[i], jsonDType, dims[i], dims[i+1],
				))
			}
			b.WriteString(`]}`)
			return []byte(b.String())
		},
		MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, dims[0]) },
		MakeTarget: sinTarget,
	})
}

func RunSwiGLU() bool {
	dims := sevenEndpoints([]int{32, 32, 32, 32, 32, 32, 32, 16})
	return RunLayerSuite(LayerSuite{
		Name:          "SwiGLU",
		PrimaryType:   poly.LayerSwiGLU,
		CheckpointTag: "seven_swiglu",
		Banner:        "  7 flat SwiGLU — ASM forward/backward not implemented",
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "loom-seven-swiglu")
			for i := 0; i < sevenLayersPerCell; i++ {
				if i > 0 {
					b.WriteByte(',')
				}
				b.WriteString(fmt.Sprintf(
					`{"z":0,"y":0,"x":0,"l":%d,"type":"SWIGLU","activation":"RELU","dtype":"%s","input_height":%d,"output_height":%d}`,
					i, jsonDType, dims[i], dims[i+1],
				))
			}
			b.WriteString(`]}`)
			return []byte(b.String())
		},
		MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, dims[0]) },
		MakeTarget: sinTarget,
	})
}

func RunMHA() bool {
	return RunLayerSuite(LayerSuite{
		Name:          "MHA",
		PrimaryType:   poly.LayerMultiHeadAttention,
		CheckpointTag: "seven_mha",
		Banner:        "  7 flat MHA — ASM not implemented",
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "loom-seven-mha")
			for i := 0; i < sevenLayersPerCell; i++ {
				if i > 0 {
					b.WriteByte(',')
				}
				b.WriteString(fmt.Sprintf(
					`{"z":0,"y":0,"x":0,"l":%d,"type":"MHA","activation":"RELU","dtype":"%s","d_model":64,"num_heads":4,"seq_length":8}`,
					i, jsonDType,
				))
			}
			b.WriteString(`]}`)
			return []byte(b.String())
		},
		MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, 8, 64) },
		MakeTarget: sinTarget,
	})
}

func RunCNN1() bool {
	ch := sevenEndpoints([]int{3, 6, 8, 8, 8, 16, 16, 16})
	return RunLayerSuite(LayerSuite{
		Name:          "CNN1",
		PrimaryType:   poly.LayerCNN1,
		CheckpointTag: "seven_cnn1",
		Banner:        "  7 flat CNN1 — ASM not implemented",
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "loom-seven-cnn1")
			for i := 0; i < sevenLayersPerCell; i++ {
				if i > 0 {
					b.WriteByte(',')
				}
				b.WriteString(fmt.Sprintf(
					`{"z":0,"y":0,"x":0,"l":%d,"type":"CNN1","activation":"RELU","dtype":"%s","input_channels":%d,"filters":%d,"input_height":32,"output_height":32,"kernel_size":3,"stride":1,"padding":1}`,
					i, jsonDType, ch[i], ch[i+1],
				))
			}
			b.WriteString(`]}`)
			return []byte(b.String())
		},
		MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, ch[0], 32) },
		MakeTarget: sinTarget,
	})
}

func RunCNN2() bool {
	ch := sevenEndpoints([]int{3, 6, 8, 8, 16, 16, 16, 16})
	return RunLayerSuite(LayerSuite{
		Name:          "CNN2",
		PrimaryType:   poly.LayerCNN2,
		CheckpointTag: "seven_cnn2",
		Banner:        "  7 flat CNN2 — ASM not implemented",
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "loom-seven-cnn2")
			for i := 0; i < sevenLayersPerCell; i++ {
				if i > 0 {
					b.WriteByte(',')
				}
				b.WriteString(fmt.Sprintf(
					`{"z":0,"y":0,"x":0,"l":%d,"type":"CNN2","activation":"RELU","dtype":"%s","input_channels":%d,"filters":%d,"input_height":32,"output_height":32,"kernel_size":3,"stride":1,"padding":1}`,
					i, jsonDType, ch[i], ch[i+1],
				))
			}
			b.WriteString(`]}`)
			return []byte(b.String())
		},
		MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, ch[0], 32, 32) },
		MakeTarget: sinTarget,
	})
}

func RunCNN3() bool {
	ch := sevenEndpoints([]int{2, 4, 4, 4, 8, 8, 8, 8})
	return RunLayerSuite(LayerSuite{
		Name:          "CNN3",
		PrimaryType:   poly.LayerCNN3,
		CheckpointTag: "seven_cnn3",
		Banner:        "  7 flat CNN3 — ASM not implemented",
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "loom-seven-cnn3")
			for i := 0; i < sevenLayersPerCell; i++ {
				if i > 0 {
					b.WriteByte(',')
				}
				b.WriteString(fmt.Sprintf(
					`{"z":0,"y":0,"x":0,"l":%d,"type":"CNN3","activation":"RELU","dtype":"%s","input_channels":%d,"filters":%d,"input_height":16,"input_width":16,"output_height":16,"output_width":16,"kernel_size":3,"stride":1,"padding":1}`,
					i, jsonDType, ch[i], ch[i+1],
				))
			}
			b.WriteString(`]}`)
			return []byte(b.String())
		},
		MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, ch[0], 16, 16) },
		MakeTarget: sinTarget,
	})
}

func RunRNN() bool {
	dims := sevenEndpoints([]int{16, 24, 32, 32, 32, 24, 16, 8})
	return RunLayerSuite(LayerSuite{
		Name:          "RNN",
		PrimaryType:   poly.LayerRNN,
		CheckpointTag: "seven_rnn",
		Banner:        "  7 flat RNN — ASM not implemented",
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "loom-seven-rnn")
			for i := 0; i < sevenLayersPerCell; i++ {
				if i > 0 {
					b.WriteByte(',')
				}
				b.WriteString(fmt.Sprintf(
					`{"z":0,"y":0,"x":0,"l":%d,"type":"RNN","activation":"TANH","dtype":"%s","input_height":%d,"output_height":%d}`,
					i, jsonDType, dims[i], dims[i+1],
				))
			}
			b.WriteString(`]}`)
			return []byte(b.String())
		},
		MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, dims[0]) },
		MakeTarget: sinTarget,
	})
}

func RunLSTM() bool {
	dims := sevenEndpoints([]int{16, 24, 32, 32, 32, 24, 16, 8})
	return RunLayerSuite(LayerSuite{
		Name:          "LSTM",
		PrimaryType:   poly.LayerLSTM,
		CheckpointTag: "seven_lstm",
		Banner:        "  7 flat LSTM — ASM not implemented",
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "loom-seven-lstm")
			for i := 0; i < sevenLayersPerCell; i++ {
				if i > 0 {
					b.WriteByte(',')
				}
				b.WriteString(fmt.Sprintf(
					`{"z":0,"y":0,"x":0,"l":%d,"type":"LSTM","activation":"TANH","dtype":"%s","input_height":%d,"output_height":%d}`,
					i, jsonDType, dims[i], dims[i+1],
				))
			}
			b.WriteString(`]}`)
			return []byte(b.String())
		},
		MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, dims[0]) },
		MakeTarget: sinTarget,
	})
}

func RunEmbedding() bool {
	dims := []int{32, 32, 32, 24, 16, 12, 8}
	acts := []string{"RELU", "RELU", "RELU", "RELU", "RELU", "SIGMOID"}
	return RunLayerSuite(LayerSuite{
		Name:          "Embedding",
		PrimaryType:   poly.LayerEmbedding,
		CheckpointTag: "seven_embedding",
		Banner:        "  EMBEDDING + 6 DENSE — ASM not implemented",
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "loom-seven-embedding")
			b.WriteString(fmt.Sprintf(
				`{"z":0,"y":0,"x":0,"l":0,"type":"EMBEDDING","dtype":"%s","vocab_size":50,"embedding_dim":32}`,
				jsonDType,
			))
			for i := 0; i < len(dims)-1; i++ {
				b.WriteByte(',')
				b.WriteString(fmt.Sprintf(
					`{"z":0,"y":0,"x":0,"l":%d,"type":"DENSE","activation":"%s","dtype":"%s","input_height":%d,"output_height":%d}`,
					i+1, acts[i], jsonDType, dims[i], dims[i+1],
				))
			}
			b.WriteString(`]}`)
			return []byte(b.String())
		},
		MakeInput: func() *poly.Tensor[float32] {
			t := poly.NewTensor[float32](8, 1)
			for i := range t.Data {
				t.Data[i] = float32(i % 50)
			}
			return t
		},
		MakeTarget: sinTarget,
	})
}

func RunResidual() bool {
	const residualDim = 32
	return RunLayerSuite(LayerSuite{
		Name:          "Residual",
		PrimaryType:   poly.LayerResidual,
		CheckpointTag: "seven_residual",
		Banner:        fmt.Sprintf("  7 flat RESIDUAL %d→%d (no nested sequential_layers)", residualDim, residualDim),
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "loom-seven-residual")
			for i := 0; i < sevenLayersPerCell; i++ {
				if i > 0 {
					b.WriteByte(',')
				}
				b.WriteString(fmt.Sprintf(
					`{"z":0,"y":0,"x":0,"l":%d,"type":"RESIDUAL","dtype":"%s","input_height":%d,"output_height":%d}`,
					i, jsonDType, residualDim, residualDim,
				))
			}
			b.WriteString(`]}`)
			return []byte(b.String())
		},
		MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, residualDim) },
		MakeTarget: sinTarget,
	})
}
