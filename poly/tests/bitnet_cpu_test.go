package poly_test

import (
	"math"
	"testing"

	. "github.com/openfluke/loom/poly"
)

func deterministicWeights(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		v := float32((i%11)-5) * 0.13
		if i%7 == 0 {
			v *= -0.5
		}
		out[i] = v
	}
	return out
}

func maxAbsDiffSlice(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	var maxDiff float64
	for i := range a {
		diff := math.Abs(float64(a[i] - b[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	return maxDiff
}

func allFinite(data []float32) bool {
	for _, v := range data {
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return false
		}
	}
	return true
}

func bitNetTestScale(weights []float32) float32 {
	if len(weights) == 0 {
		return 1.0
	}
	var sumAbs float64
	for _, v := range weights {
		sumAbs += math.Abs(float64(v))
	}
	scale := float32(sumAbs / float64(len(weights)))
	if scale == 0 {
		return 1.0
	}
	return scale
}

func bitNetTestQuantValue(v, scale float32) uint8 {
	if scale == 0 {
		scale = 1.0
	}
	q := int(math.Round(float64(v / scale)))
	if q > 1 {
		q = 1
	}
	if q < -1 {
		q = -1
	}
	return uint8(int8(q))
}

func dequantBitNetWeights(weights []float32) []float32 {
	scale := bitNetTestScale(weights)
	out := make([]float32, len(weights))
	for i, v := range weights {
		out[i] = float32(int8(bitNetTestQuantValue(v, scale))) * scale
	}
	return out
}

func bitNetTestQuantizeActivation(input []float32) ([]int8, float32) {
	maxAbs := float32(0)
	for _, v := range input {
		a := float32(math.Abs(float64(v)))
		if a > maxAbs {
			maxAbs = a
		}
	}
	if maxAbs < 1e-5 {
		maxAbs = 1e-5
	}
	out := make([]int8, len(input))
	scale := 127.0 / float64(maxAbs)
	for i, v := range input {
		q := int(math.Round(float64(v) * scale))
		if q < -128 {
			q = -128
		}
		if q > 127 {
			q = 127
		}
		out[i] = int8(q)
	}
	return out, maxAbs
}

func bitNetTestMatVecQuantized(matrix *BitNetTernaryMatrix, xq []int8, activationMax float32, out []float64) bool {
	if matrix == nil || len(xq) < matrix.Cols || len(out) < matrix.Rows {
		return false
	}
	outputScale := float64(matrix.Scale) * float64(activationMax) / 127.0
	if outputScale == 0 {
		outputScale = 1.0
	}
	rowWords := matrix.RowWords
	if rowWords <= 0 {
		rowWords = (matrix.Cols + 15) / 16
	}
	for r := 0; r < matrix.Rows; r++ {
		var sum int32
		for c := 0; c < matrix.Cols; c++ {
			word := matrix.Words[r*rowWords+c/16]
			code := (word >> uint((c%16)*2)) & 0x03
			sum += int32(xq[c]) * (int32(code) - 1)
		}
		out[r] = float64(sum) * outputScale
	}
	return true
}

func TestBitNetTernaryQuantizerUsesAbsMean(t *testing.T) {
	ws := NewWeightStore(4)
	copy(ws.Master, []float32{-0.2, -0.8, 0.1, 0.8})

	ws.MorphBitNetTernary()

	if math.Abs(float64(ws.Scale-0.475)) > 1e-6 {
		t.Fatalf("scale = %v, want absmean 0.475", ws.Scale)
	}
	got, ok := ws.Versions[DTypeTernary].([]uint8)
	if !ok {
		t.Fatalf("ternary version type = %T, want []uint8", ws.Versions[DTypeTernary])
	}
	want := []int8{0, -1, 0, 1}
	for i := range want {
		if int8(got[i]) != want[i] {
			t.Fatalf("q[%d] = %d, want %d", i, int8(got[i]), want[i])
		}
	}
}

func TestBitNetNativeTernaryUsesUnitScale(t *testing.T) {
	net := NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.Type = LayerDense
	l.InputHeight = 4
	l.OutputHeight = 2
	l.WeightStore = NewWeightStore(8)
	copy(l.WeightStore.Master, []float32{-0.2, -0.8, 0.1, 0.8, 0.4, -0.3, 0.0, 0.9})

	if err := MorphLayerBitNetNativeTernary(l); err != nil {
		t.Fatal(err)
	}
	if l.WeightStore.Scale != 1.0 {
		t.Fatalf("scale = %v, want 1", l.WeightStore.Scale)
	}
	matrix, ok := l.WeightStore.GetBitNetTernaryMatrix(0, 2, 4)
	if !ok {
		t.Fatal("packed native ternary matrix missing")
	}
	if matrix.Scale != 1.0 {
		t.Fatalf("matrix scale = %v, want 1", matrix.Scale)
	}
	for i, v := range l.WeightStore.Master {
		if v != -1 && v != 0 && v != 1 {
			t.Fatalf("master[%d] = %v, want ternary", i, v)
		}
	}
}

func TestPackedTernaryDenseMatchesDequantizedPath(t *testing.T) {
	weights := deterministicWeights(15)
	refNet := NewVolumetricNetwork(1, 1, 1, 1)
	refLayer := refNet.GetLayer(0, 0, 0, 0)
	refLayer.Type = LayerDense
	refLayer.Activation = ActivationLinear
	refLayer.DType = DTypeFloat32
	refLayer.InputHeight = 5
	refLayer.OutputHeight = 3
	refLayer.WeightStore = NewWeightStore(refLayer.InputHeight * refLayer.OutputHeight)
	copy(refLayer.WeightStore.Master, dequantBitNetWeights(weights))

	net := NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.Type = LayerDense
	l.Activation = ActivationLinear
	l.DType = DTypeTernary
	l.InputHeight = 5
	l.OutputHeight = 3
	l.WeightStore = NewWeightStore(l.InputHeight * l.OutputHeight)
	copy(l.WeightStore.Master, weights)
	input := NewTensorFromSlice([]float32{0.1, -0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8, -0.9, 1.0}, 2, 5)

	_, ref := DenseForwardPolymorphic(refLayer, input)
	net.UseExactDType = true
	_, got := DenseForwardPolymorphic(l, input)

	if diff := maxAbsDiffSlice(ref.Data, got.Data); diff > 5e-3 {
		t.Fatalf("packed dense diff = %g", diff)
	}
}

func TestBitNetTernaryKernelHandlesTailColumns(t *testing.T) {
	rows, cols := 3, 17
	weights := deterministicWeights(rows * cols)
	ws := NewWeightStore(rows * cols)
	copy(ws.Master, weights)
	matrix, ok := ws.GetBitNetTernaryMatrix(0, rows, cols)
	if !ok {
		t.Fatal("failed to pack tail-column matrix")
	}
	input := deterministicWeights(cols)
	xq, activationMax := bitNetTestQuantizeActivation(input)
	got := make([]float64, rows)
	if !bitNetTestMatVecQuantized(matrix, xq, activationMax, got) {
		t.Fatal("kernel returned false")
	}
	outputScale := float64(matrix.Scale) * float64(activationMax) / 127.0
	for r := 0; r < rows; r++ {
		var sum int32
		for c := 0; c < cols; c++ {
			q := int8(bitNetTestQuantValue(weights[r*cols+c], matrix.Scale))
			if q > 0 {
				sum += int32(xq[c])
			} else if q < 0 {
				sum -= int32(xq[c])
			}
		}
		want := float64(sum) * outputScale
		if math.Abs(got[r]-want) > 1e-6 {
			t.Fatalf("row %d = %g, want %g", r, got[r], want)
		}
	}
}

func TestPrepareBitNetTernaryCPUReleasesProjectionMaster(t *testing.T) {
	net := NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.Type = LayerDense
	l.Activation = ActivationLinear
	l.DType = DTypeFloat32
	l.InputHeight = 17
	l.OutputHeight = 3
	l.WeightStore = NewWeightStore(l.InputHeight * l.OutputHeight)
	copy(l.WeightStore.Master, deterministicWeights(len(l.WeightStore.Master)))

	if err := PrepareNetworkBitNetTernaryCPU(net); err != nil {
		t.Fatal(err)
	}
	if len(l.WeightStore.Master) != 0 {
		t.Fatalf("master len = %d, want released", len(l.WeightStore.Master))
	}
	matrix, ok := l.WeightStore.GetBitNetTernaryMatrix(0, l.OutputHeight, l.InputHeight)
	if !ok {
		t.Fatal("cached packed matrix unavailable after master release")
	}
	if matrix.RowWords != 2 {
		t.Fatalf("rowWords = %d, want 2", matrix.RowWords)
	}
	if got := l.WeightStore.SizeInBytes(DTypeTernary); got != len(matrix.Words)*4 {
		t.Fatalf("SizeInBytes = %d, want packed bytes %d", got, len(matrix.Words)*4)
	}
	input := NewTensorFromSlice(deterministicWeights(l.InputHeight), 1, l.InputHeight)
	net.UseExactDType = true
	_, out := DenseForwardPolymorphic(l, input)
	if len(out.Data) != l.OutputHeight || !allFinite(out.Data) {
		t.Fatalf("packed dense after release produced invalid output: %v", out.Data)
	}
}

func TestMicrosoftOfflinePackedBitNetRowsDecode(t *testing.T) {
	ws := NewWeightStore(8 * 2)
	packed := []float32{
		0b10100001, 0b00011000,
		0b10010000, 0b00001010,
	}

	if !ws.SetMicrosoftBitNetPackedMatrix(0, 8, 2, packed) {
		t.Fatal("failed to set Microsoft packed matrix")
	}
	ws.SetBitNetPackedScale(0, 0.5)
	matrix, ok := ws.GetBitNetTernaryMatrix(0, 8, 2)
	if !ok {
		t.Fatal("packed matrix missing")
	}

	xq := []int8{2, -3}
	got := make([]float64, 8)
	if !bitNetTestMatVecQuantized(matrix, xq, 127, got) {
		t.Fatal("kernel returned false")
	}

	want := []float64{1.5, -2.5, -2.5, -2.5, 1, 1.5, 2.5, 2.5}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-6 {
			t.Fatalf("got[%d] = %g, want %g", i, got[i], want[i])
		}
	}
}

func TestPackedTernaryMHAProducesFiniteOutput(t *testing.T) {
	weights := deterministicWeights(80)
	net := NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.Type = LayerMultiHeadAttention
	l.DType = DTypeTernary
	l.DModel = 4
	l.NumHeads = 2
	l.NumKVHeads = 2
	l.HeadDim = 2
	l.QueryDim = 4
	l.MaxSeqLen = 8
	mhaSize := 4*4 + 4*4 + 4*4 + 4*4 + 4 + 4 + 4 + 4
	l.WeightStore = NewWeightStore(mhaSize)
	copy(l.WeightStore.Master, weights)
	input := NewTensorFromSlice([]float32{0.1, -0.2, 0.3, -0.4, 0.2, -0.1, 0.4, -0.3}, 2, 4)

	net.UseExactDType = true
	l.ResetState()
	_, got := MHAForwardPolymorphic(l, input)

	if len(got.Data) != 8 || !allFinite(got.Data) {
		t.Fatalf("packed mha produced invalid output: len=%d data=%v", len(got.Data), got.Data)
	}
}

func TestPackedTernarySwiGLUProducesFiniteOutput(t *testing.T) {
	weights := deterministicWeights(88)
	net := NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.Type = LayerSwiGLU
	l.DType = DTypeTernary
	l.InputHeight = 4
	l.OutputHeight = 6
	wCount := 3*l.InputHeight*l.OutputHeight + 2*l.OutputHeight + l.InputHeight
	l.WeightStore = NewWeightStore(wCount)
	copy(l.WeightStore.Master, weights)
	input := NewTensorFromSlice([]float32{0.1, -0.2, 0.3, -0.4, 0.2, -0.1, 0.4, -0.3}, 2, 4)

	net.UseExactDType = true
	_, got := SwiGLUForwardPolymorphic(l, input)

	if len(got.Data) != 8 || !allFinite(got.Data) {
		t.Fatalf("packed swiglu produced invalid output: len=%d data=%v", len(got.Data), got.Data)
	}
}

func TestTernaryNativePersistenceRoundTripKeepsActiveVersion(t *testing.T) {
	net := NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.Type = LayerDense
	l.Activation = ActivationLinear
	l.DType = DTypeFloat32
	l.InputHeight = 4
	l.OutputHeight = 3
	l.WeightStore = NewWeightStore(l.InputHeight * l.OutputHeight)
	copy(l.WeightStore.Master, deterministicWeights(len(l.WeightStore.Master)))
	if err := MorphLayerBitNetTernary(l); err != nil {
		t.Fatal(err)
	}

	data, err := SerializeNetwork(net)
	if err != nil {
		t.Fatal(err)
	}
	roundTrip, err := DeserializeNetwork(data)
	if err != nil {
		t.Fatal(err)
	}
	rtLayer := roundTrip.GetLayer(0, 0, 0, 0)
	if rtLayer.DType != DTypeTernary {
		t.Fatalf("dtype = %v, want ternary", rtLayer.DType)
	}
	if _, ok := rtLayer.WeightStore.Versions[DTypeTernary].([]uint8); !ok {
		t.Fatalf("version type = %T, want []uint8", rtLayer.WeightStore.Versions[DTypeTernary])
	}
	if active := rtLayer.WeightStore.GetActive(DTypeTernary); active == nil {
		t.Fatal("GetActive(DTypeTernary) returned nil after round trip")
	}
}

func TestPackedTernaryLMHeadProducesLogits(t *testing.T) {
	net := NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.DType = DTypeTernary
	net.UseExactDType = true
	tr := NewTransformer[float32](net, make([]float32, 12), deterministicWeights(12), nil, Template{})
	tr.HiddenSize = 4
	tr.VocabSize = 3

	logits := tr.ApplyLMHead([]float32{0.2, -0.1, 0.4, -0.3})
	if len(logits) != 3 {
		t.Fatalf("logits len = %d, want 3", len(logits))
	}
}

func TestPackedTernaryLMHeadSkipsTiedEmbeddings(t *testing.T) {
	net := NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.DType = DTypeTernary
	net.UseExactDType = true
	embeddings := deterministicWeights(12)
	tr := NewTransformer[float32](net, embeddings, embeddings, nil, Template{})
	tr.HiddenSize = 4
	tr.VocabSize = 3

	logits := tr.ApplyLMHead([]float32{0.2, -0.1, 0.4, -0.3})
	if len(logits) != 3 {
		t.Fatalf("logits len = %d, want 3", len(logits))
	}
}

func TestFP32LMHeadMatchesSequential(t *testing.T) {
	net := NewVolumetricNetwork(1, 1, 1, 1)
	weights := deterministicWeights(16 * 8)
	tr := NewTransformer[float32](net, nil, weights, nil, Template{})
	tr.HiddenSize = 8
	tr.VocabSize = 16
	hidden := deterministicWeights(8)

	got := tr.ApplyLMHead(hidden)
	want := make([]float32, tr.VocabSize)
	for v := 0; v < tr.VocabSize; v++ {
		var sum float64
		for d := 0; d < tr.HiddenSize; d++ {
			sum += float64(hidden[d]) * float64(weights[v*tr.HiddenSize+d])
		}
		want[v] = float32(sum)
	}
	if diff := maxAbsDiffSlice(want, got); diff > 1e-6 {
		t.Fatalf("lm_head diff = %g", diff)
	}
}

func BenchmarkPackedTernaryDenseForward(b *testing.B) {
	net := NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.Type = LayerDense
	l.Activation = ActivationLinear
	l.DType = DTypeTernary
	l.InputHeight = 512
	l.OutputHeight = 512
	l.WeightStore = NewWeightStore(l.InputHeight * l.OutputHeight)
	copy(l.WeightStore.Master, deterministicWeights(len(l.WeightStore.Master)))
	input := NewTensorFromSlice(deterministicWeights(512), 1, 512)
	net.UseExactDType = true
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		DenseForwardPolymorphic(l, input)
	}
}

func BenchmarkBitNetTernaryKernel1536(b *testing.B) {
	rows, cols := 1536, 1536
	ws := NewWeightStore(rows * cols)
	copy(ws.Master, deterministicWeights(rows*cols))
	matrix, ok := ws.GetBitNetTernaryMatrix(0, rows, cols)
	if !ok {
		b.Fatal("failed to pack matrix")
	}
	xq, activationMax := bitNetTestQuantizeActivation(deterministicWeights(cols))
	out := make([]float64, rows)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if !bitNetTestMatVecQuantized(matrix, xq, activationMax, out) {
			b.Fatal("kernel failed")
		}
	}
}

func BenchmarkFP32LMHeadTiedBitNetSize(b *testing.B) {
	net := NewVolumetricNetwork(1, 1, 1, 1)
	hiddenSize := 1536
	vocabSize := 32002
	weights := deterministicWeights(hiddenSize * vocabSize)
	tr := NewTransformer[float32](net, nil, weights, nil, Template{})
	tr.HiddenSize = hiddenSize
	tr.VocabSize = vocabSize
	hidden := deterministicWeights(hiddenSize)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = tr.ApplyLMHead(hidden)
	}
}
