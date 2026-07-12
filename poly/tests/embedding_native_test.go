package poly_test

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestEmbeddingNativeBackwardFloat64(t *testing.T) {
	json := `{"id":"test","depth":1,"rows":1,"cols":1,"layers_per_cell":7,"layers":[` +
		`{"z":0,"y":0,"x":0,"l":0,"type":"EMBEDDING","dtype":"Float64","vocab_size":50,"embedding_dim":32},` +
		`{"z":0,"y":0,"x":0,"l":1,"type":"DENSE","activation":"RELU","dtype":"Float64","input_height":32,"output_height":32},` +
		`{"z":0,"y":0,"x":0,"l":2,"type":"DENSE","activation":"RELU","dtype":"Float64","input_height":32,"output_height":32},` +
		`{"z":0,"y":0,"x":0,"l":3,"type":"DENSE","activation":"RELU","dtype":"Float64","input_height":32,"output_height":24},` +
		`{"z":0,"y":0,"x":0,"l":4,"type":"DENSE","activation":"RELU","dtype":"Float64","input_height":24,"output_height":16},` +
		`{"z":0,"y":0,"x":0,"l":5,"type":"DENSE","activation":"RELU","dtype":"Float64","input_height":16,"output_height":12},` +
		`{"z":0,"y":0,"x":0,"l":6,"type":"DENSE","activation":"SIGMOID","dtype":"Float64","input_height":12,"output_height":8}` +
		`]}`
	net, err := poly.BuildNetworkFromJSON([]byte(json))
	if err != nil {
		t.Fatal(err)
	}
	net.UseExactDType = true
	for i := range net.Layers {
		net.Layers[i].Network = net
	}
	seq := 8
	input := poly.NewTensor[float32](seq, 1)
	for i := range input.Data {
		input.Data[i] = float32(i % 50)
	}
	target := poly.NewTensor[float32](seq, 8)
	for i := range target.Data {
		target.Data[i] = 0.5
	}

	histIn := make([]*poly.Tensor[float32], len(net.Layers))
	histPre := make([]*poly.Tensor[float32], len(net.Layers))
	curr := input
	for i := range net.Layers {
		histIn[i] = curr
		var pre, post *poly.Tensor[float32]
		switch net.Layers[i].Type {
		case poly.LayerEmbedding:
			pre, post = poly.EmbeddingForwardPolymorphic(&net.Layers[i], curr)
		default:
			pre, post = poly.DenseForwardPolymorphic(&net.Layers[i], curr)
		}
		histPre[i] = pre
		curr = post
	}
	gradOut := poly.ComputeLossGradient(curr, target, "mse")
	_, layerGrads, _ := poly.BackwardPolymorphic(net, gradOut, histIn, histPre)
	if layerGrads[0][1] == nil || len(layerGrads[0][1].Data) == 0 {
		t.Fatal("expected embedding weight grads")
	}
}
