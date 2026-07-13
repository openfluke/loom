package poly

import "fmt"

// RunAllSwiGLUDTypeRoundTrips runs 21 dtypes on a single SwiGLU block.
func RunAllSwiGLUDTypeRoundTrips(tag string) []DTypeRoundTripResult {
	specs := []SwiGLUSpec{{Hidden: 8, Intermediate: 16}}
	out := make([]DTypeRoundTripResult, 0, 21)
	for _, dt := range SeedDTypesAll() {
		out = append(out, runSwiGLUDTypeRoundTrip(tag, specs, dt))
	}
	return out
}

func runSwiGLUDTypeRoundTrip(tag string, specs []SwiGLUSpec, dt DType) DTypeRoundTripResult {
	res := DTypeRoundTripResult{Family: "swiglu", DType: dt, DTypeName: dt.String()}
	dtype := dt.String()
	topo := layerDTypeTopo(SwiGLUTopologySeed(tag, specs), dt)
	manifest, err := BuildSwiGLUManifest(topo, specs, []string{dtype})
	if err != nil {
		res.Err = err.Error()
		return res
	}
	res.TopologySeed = topo
	if len(manifest.Layers) > 0 {
		res.LayerSeed = manifest.Layers[0].LayerSeed
		res.WeightFP = manifest.Layers[0].WeightFP
	}
	rebuilt, err := RebuildSwiGLUManifest(manifest)
	if err != nil {
		res.Err = "rebuild: " + err.Error()
		return res
	}
	netA, err := BuildSwiGLUVolumetricFromManifest(manifest)
	if err != nil {
		res.Err = "build A: " + err.Error()
		return res
	}
	netB, err := BuildSwiGLUVolumetricFromManifest(rebuilt)
	if err != nil {
		res.Err = "build B: " + err.Error()
		return res
	}
	hashA, err := volumetricForwardHash(netA, specs[0].Hidden)
	if err != nil {
		res.Err = "forward A: " + err.Error()
		return res
	}
	hashB, err := volumetricForwardHash(netB, specs[0].Hidden)
	if err != nil {
		res.Err = "forward B: " + err.Error()
		return res
	}
	extracted, err := ManifestFromSwiGLUNetwork(netA, topo, specs, []string{dtype})
	if err != nil {
		res.Err = "extract: " + err.Error()
		return res
	}
	extSeed := uint64(0)
	if len(extracted.Layers) > 0 {
		extSeed = extracted.Layers[0].LayerSeed
	}
	_ = rebuilt
	return finishDTypeRoundTrip(res, nil, manifest.NetworkFP, res.LayerSeed,
		hashA, hashB, extracted.NetworkFP, extSeed, extracted.ForwardFP)
}

// RunAllMHADTypeRoundTrips runs 21 dtypes on a single MHA block.
func RunAllMHADTypeRoundTrips(tag string) []DTypeRoundTripResult {
	specs := []MHASpec{{DModel: 8, NumHeads: 2, NumKVHeads: 2, HeadDim: 4, QueryDim: 8}}
	out := make([]DTypeRoundTripResult, 0, 21)
	for _, dt := range SeedDTypesAll() {
		out = append(out, runMHADTypeRoundTrip(tag, specs, dt))
	}
	return out
}

func runMHADTypeRoundTrip(tag string, specs []MHASpec, dt DType) DTypeRoundTripResult {
	res := DTypeRoundTripResult{Family: "mha", DType: dt, DTypeName: dt.String()}
	dtype := dt.String()
	topo := layerDTypeTopo(MHATopologySeed(tag, specs), dt)
	manifest, err := BuildMHAManifest(topo, specs, []string{dtype})
	if err != nil {
		res.Err = err.Error()
		return res
	}
	res.TopologySeed = topo
	if len(manifest.Layers) > 0 {
		res.LayerSeed = manifest.Layers[0].LayerSeed
		res.WeightFP = manifest.Layers[0].WeightFP
	}
	if _, err := RebuildMHAManifest(manifest); err != nil {
		res.Err = "rebuild: " + err.Error()
		return res
	}
	netA, err := BuildMHAVolumetricFromManifest(manifest)
	if err != nil {
		res.Err = "build A: " + err.Error()
		return res
	}
	rebuilt, err := RebuildMHAManifest(manifest)
	if err != nil {
		res.Err = "rebuild: " + err.Error()
		return res
	}
	netB, err := BuildMHAVolumetricFromManifest(rebuilt)
	if err != nil {
		res.Err = "build B: " + err.Error()
		return res
	}
	hashA, err := volumetricForwardHash(netA, specs[0].DModel)
	if err != nil {
		res.Err = "forward A: " + err.Error()
		return res
	}
	hashB, err := volumetricForwardHash(netB, specs[0].DModel)
	if err != nil {
		res.Err = "forward B: " + err.Error()
		return res
	}
	extracted, err := ManifestFromMHANetwork(netA, topo, specs, []string{dtype})
	if err != nil {
		res.Err = "extract: " + err.Error()
		return res
	}
	extSeed := uint64(0)
	if len(extracted.Layers) > 0 {
		extSeed = extracted.Layers[0].LayerSeed
	}
	return finishDTypeRoundTrip(res, nil, manifest.NetworkFP, res.LayerSeed,
		hashA, hashB, extracted.NetworkFP, extSeed, extracted.ForwardFP)
}

// RunAllRNNDTypeRoundTrips runs 21 dtypes on a single RNN layer.
func RunAllRNNDTypeRoundTrips(tag string) []DTypeRoundTripResult {
	sizes := []int{4, 8}
	out := make([]DTypeRoundTripResult, 0, 21)
	for _, dt := range SeedDTypesAll() {
		out = append(out, runRNNDTypeRoundTrip(tag, sizes, dt))
	}
	return out
}

func runRNNDTypeRoundTrip(tag string, sizes []int, dt DType) DTypeRoundTripResult {
	res := DTypeRoundTripResult{Family: "rnn", DType: dt, DTypeName: dt.String()}
	dtype := dt.String()
	topo := layerDTypeTopo(RNNTopologySeed(tag, sizes), dt)
	manifest, err := BuildRNNManifest(topo, sizes, []string{dtype})
	if err != nil {
		res.Err = err.Error()
		return res
	}
	res.TopologySeed = topo
	if len(manifest.Layers) > 0 {
		res.LayerSeed = manifest.Layers[0].LayerSeed
		res.WeightFP = manifest.Layers[0].WeightFP
	}
	if _, err := RebuildRNNManifest(manifest); err != nil {
		res.Err = "rebuild: " + err.Error()
		return res
	}
	netA, err := BuildRNNVolumetricFromManifest(manifest)
	if err != nil {
		res.Err = "build A: " + err.Error()
		return res
	}
	rebuilt, err := RebuildRNNManifest(manifest)
	if err != nil {
		res.Err = "rebuild: " + err.Error()
		return res
	}
	netB, err := BuildRNNVolumetricFromManifest(rebuilt)
	if err != nil {
		res.Err = "build B: " + err.Error()
		return res
	}
	hashA, err := volumetricForwardHash(netA, sizes[0])
	if err != nil {
		res.Err = "forward A: " + err.Error()
		return res
	}
	hashB, err := volumetricForwardHash(netB, sizes[0])
	if err != nil {
		res.Err = "forward B: " + err.Error()
		return res
	}
	extracted, err := ManifestFromRNNNetwork(netA, topo, sizes, []string{dtype})
	if err != nil {
		res.Err = "extract: " + err.Error()
		return res
	}
	extSeed := uint64(0)
	if len(extracted.Layers) > 0 {
		extSeed = extracted.Layers[0].LayerSeed
	}
	return finishDTypeRoundTrip(res, nil, manifest.NetworkFP, res.LayerSeed,
		hashA, hashB, extracted.NetworkFP, extSeed, extracted.ForwardFP)
}

// RunAllLSTMDTypeRoundTrips runs 21 dtypes on a single LSTM layer.
func RunAllLSTMDTypeRoundTrips(tag string) []DTypeRoundTripResult {
	sizes := []int{4, 8}
	out := make([]DTypeRoundTripResult, 0, 21)
	for _, dt := range SeedDTypesAll() {
		out = append(out, runLSTMDTypeRoundTrip(tag, sizes, dt))
	}
	return out
}

func runLSTMDTypeRoundTrip(tag string, sizes []int, dt DType) DTypeRoundTripResult {
	res := DTypeRoundTripResult{Family: "lstm", DType: dt, DTypeName: dt.String()}
	dtype := dt.String()
	topo := layerDTypeTopo(LSTMTopologySeed(tag, sizes), dt)
	manifest, err := BuildLSTMManifest(topo, sizes, []string{dtype})
	if err != nil {
		res.Err = err.Error()
		return res
	}
	res.TopologySeed = topo
	if len(manifest.Layers) > 0 {
		res.LayerSeed = manifest.Layers[0].LayerSeed
		res.WeightFP = manifest.Layers[0].WeightFP
	}
	if _, err := RebuildLSTMManifest(manifest); err != nil {
		res.Err = "rebuild: " + err.Error()
		return res
	}
	netA, err := BuildLSTMVolumetricFromManifest(manifest)
	if err != nil {
		res.Err = "build A: " + err.Error()
		return res
	}
	rebuilt, err := RebuildLSTMManifest(manifest)
	if err != nil {
		res.Err = "rebuild: " + err.Error()
		return res
	}
	netB, err := BuildLSTMVolumetricFromManifest(rebuilt)
	if err != nil {
		res.Err = "build B: " + err.Error()
		return res
	}
	hashA, err := volumetricForwardHash(netA, sizes[0])
	if err != nil {
		res.Err = "forward A: " + err.Error()
		return res
	}
	hashB, err := volumetricForwardHash(netB, sizes[0])
	if err != nil {
		res.Err = "forward B: " + err.Error()
		return res
	}
	extracted, err := ManifestFromLSTMNetwork(netA, topo, sizes, []string{dtype})
	if err != nil {
		res.Err = "extract: " + err.Error()
		return res
	}
	extSeed := uint64(0)
	if len(extracted.Layers) > 0 {
		extSeed = extracted.Layers[0].LayerSeed
	}
	return finishDTypeRoundTrip(res, nil, manifest.NetworkFP, res.LayerSeed,
		hashA, hashB, extracted.NetworkFP, extSeed, extracted.ForwardFP)
}

// RunAllCNNDTypeRoundTrips runs 21 dtypes on one CNN layer of the given dim (1/2/3).
func RunAllCNNDTypeRoundTrips(tag string, dim int) []DTypeRoundTripResult {
	spatial := 8
	if dim == 3 {
		spatial = 6
	}
	specs := []CNNSpec{{Dim: dim, InputChannels: 2, Filters: 4, Spatial: spatial, KernelSize: 3}}
	family := fmt.Sprintf("cnn%d", dim)
	out := make([]DTypeRoundTripResult, 0, 21)
	for _, dt := range SeedDTypesAll() {
		out = append(out, runCNNDTypeRoundTrip(family, tag, specs, dt))
	}
	return out
}

func runCNNDTypeRoundTrip(family, tag string, specs []CNNSpec, dt DType) DTypeRoundTripResult {
	res := DTypeRoundTripResult{Family: family, DType: dt, DTypeName: dt.String()}
	dtype := dt.String()
	topo := layerDTypeTopo(CNNTopologySeed(tag, specs), dt)
	manifest, err := BuildCNNManifest(topo, specs, []string{dtype})
	if err != nil {
		res.Err = err.Error()
		return res
	}
	res.TopologySeed = topo
	if len(manifest.Layers) > 0 {
		res.LayerSeed = manifest.Layers[0].LayerSeed
		res.WeightFP = manifest.Layers[0].WeightFP
	}
	if _, err := RebuildCNNManifest(manifest); err != nil {
		res.Err = "rebuild: " + err.Error()
		return res
	}
	rebuilt, err := RebuildCNNManifest(manifest)
	if err != nil {
		res.Err = "rebuild: " + err.Error()
		return res
	}
	hashA, err := cnnManifestForwardHash(manifest)
	if err != nil {
		res.Err = "forward A: " + err.Error()
		return res
	}
	hashB, err := cnnManifestForwardHash(rebuilt)
	if err != nil {
		res.Err = "forward B: " + err.Error()
		return res
	}
	netA, err := BuildCNNVolumetricFromManifest(manifest)
	if err != nil {
		res.Err = "build: " + err.Error()
		return res
	}
	extracted, err := ManifestFromCNNNetwork(netA, topo, specs, []string{dtype})
	if err != nil {
		res.Err = "extract: " + err.Error()
		return res
	}
	extSeed := uint64(0)
	if len(extracted.Layers) > 0 {
		extSeed = extracted.Layers[0].LayerSeed
	}
	return finishDTypeRoundTrip(res, nil, manifest.NetworkFP, res.LayerSeed,
		hashA, hashB, extracted.NetworkFP, extSeed, extracted.ForwardFP)
}

func cnnManifestForwardHash(m *CNNWeightManifest) (uint64, error) {
	net, err := BuildCNNVolumetricFromManifest(m)
	if err != nil {
		return 0, err
	}
	in := CNNDemoInput(m.Specs[0])
	if in == nil {
		return 0, fmt.Errorf("cnn: nil input")
	}
	out, _, _ := ForwardPolymorphic(net, in)
	if out == nil {
		return 0, fmt.Errorf("cnn: forward nil")
	}
	return seedOutputHash(out.Data), nil
}

// RunAllEmbeddingDTypeRoundTrips runs 21 dtypes on a single embedding table.
func RunAllEmbeddingDTypeRoundTrips(tag string) []DTypeRoundTripResult {
	specs := []EmbeddingSpec{{VocabSize: 32, EmbeddingDim: 8, SeqLen: 8}}
	out := make([]DTypeRoundTripResult, 0, 21)
	for _, dt := range SeedDTypesAll() {
		out = append(out, runEmbeddingDTypeRoundTrip(tag, specs, dt))
	}
	return out
}

func runEmbeddingDTypeRoundTrip(tag string, specs []EmbeddingSpec, dt DType) DTypeRoundTripResult {
	res := DTypeRoundTripResult{Family: "embedding", DType: dt, DTypeName: dt.String()}
	dtype := dt.String()
	topo := layerDTypeTopo(EmbeddingTopologySeed(tag, specs), dt)
	manifest, err := BuildEmbeddingManifest(topo, specs, []string{dtype})
	if err != nil {
		res.Err = err.Error()
		return res
	}
	res.TopologySeed = topo
	if len(manifest.Layers) > 0 {
		res.LayerSeed = manifest.Layers[0].LayerSeed
		res.WeightFP = manifest.Layers[0].WeightFP
	}
	if _, err := RebuildEmbeddingManifest(manifest); err != nil {
		res.Err = "rebuild: " + err.Error()
		return res
	}
	rebuilt, err := RebuildEmbeddingManifest(manifest)
	if err != nil {
		res.Err = "rebuild: " + err.Error()
		return res
	}
	hashA, err := embeddingManifestForwardHash(manifest)
	if err != nil {
		res.Err = "forward A: " + err.Error()
		return res
	}
	hashB, err := embeddingManifestForwardHash(rebuilt)
	if err != nil {
		res.Err = "forward B: " + err.Error()
		return res
	}
	netA, err := BuildEmbeddingVolumetricFromManifest(manifest)
	if err != nil {
		res.Err = "build: " + err.Error()
		return res
	}
	extracted, err := ManifestFromEmbeddingNetwork(netA, topo, specs, []string{dtype})
	if err != nil {
		res.Err = "extract: " + err.Error()
		return res
	}
	extSeed := uint64(0)
	if len(extracted.Layers) > 0 {
		extSeed = extracted.Layers[0].LayerSeed
	}
	return finishDTypeRoundTrip(res, nil, manifest.NetworkFP, res.LayerSeed,
		hashA, hashB, extracted.NetworkFP, extSeed, extracted.ForwardFP)
}

func embeddingManifestForwardHash(m *EmbeddingWeightManifest) (uint64, error) {
	out, err := ForwardEmbeddingManifest(m)
	if err != nil {
		return 0, err
	}
	return seedOutputHash(out), nil
}

// RunAllResidualDTypeRoundTrips runs 21 dtypes on a dense+residual block.
func RunAllResidualDTypeRoundTrips(tag string) []DTypeRoundTripResult {
	spec := ResidualSpec{In: 8, Out: 8}
	out := make([]DTypeRoundTripResult, 0, 21)
	for _, dt := range SeedDTypesAll() {
		out = append(out, runResidualDTypeRoundTrip(tag, spec, dt))
	}
	return out
}

func runResidualDTypeRoundTrip(tag string, spec ResidualSpec, dt DType) DTypeRoundTripResult {
	res := DTypeRoundTripResult{Family: "residual", DType: dt, DTypeName: dt.String()}
	dtype := dt.String()
	topo := layerDTypeTopo(ResidualTopologySeed(tag, spec), dt)
	manifest, err := BuildResidualManifest(topo, spec, dtype)
	if err != nil {
		res.Err = err.Error()
		return res
	}
	res.TopologySeed = topo
	res.LayerSeed = manifest.DenseSeed
	res.WeightFP = manifest.DenseWeightFP
	if _, err := RebuildResidualManifest(manifest); err != nil {
		res.Err = "rebuild: " + err.Error()
		return res
	}
	rebuilt, err := RebuildResidualManifest(manifest)
	if err != nil {
		res.Err = "rebuild: " + err.Error()
		return res
	}
	hashA, err := residualManifestForwardHash(manifest)
	if err != nil {
		res.Err = "forward A: " + err.Error()
		return res
	}
	hashB, err := residualManifestForwardHash(rebuilt)
	if err != nil {
		res.Err = "forward B: " + err.Error()
		return res
	}
	netA, err := BuildResidualVolumetricFromManifest(manifest)
	if err != nil {
		res.Err = "build: " + err.Error()
		return res
	}
	extracted, err := ManifestFromResidualNetwork(netA, topo, spec, dtype)
	if err != nil {
		res.Err = "extract: " + err.Error()
		return res
	}
	return finishDTypeRoundTrip(res, nil, manifest.DenseWeightFP, manifest.DenseSeed,
		hashA, hashB, extracted.DenseWeightFP, extracted.DenseSeed, extracted.ForwardFP)
}

func residualManifestForwardHash(m *ResidualWeightManifest) (uint64, error) {
	out, err := ForwardResidualManifest(m)
	if err != nil {
		return 0, err
	}
	return seedOutputHash(out), nil
}

// RunAllNumericalLayerDTypeMatrix runs 21 dtypes across every seed round-trip layer family.
func RunAllNumericalLayerDTypeMatrix(tag string) []LayerDTypeMatrix {
	sizes := []int{4, 8}
	return []LayerDTypeMatrix{
		{Family: "dense", Results: RunAllDenseDTypeRoundTrips(tag, sizes)},
		{Family: "swiglu", Results: RunAllSwiGLUDTypeRoundTrips(tag)},
		{Family: "mha", Results: RunAllMHADTypeRoundTrips(tag)},
		{Family: "rnn", Results: RunAllRNNDTypeRoundTrips(tag)},
		{Family: "lstm", Results: RunAllLSTMDTypeRoundTrips(tag)},
		{Family: "cnn1", Results: RunAllCNNDTypeRoundTrips(tag, 1)},
		{Family: "cnn2", Results: RunAllCNNDTypeRoundTrips(tag, 2)},
		{Family: "cnn3", Results: RunAllCNNDTypeRoundTrips(tag, 3)},
		{Family: "embedding", Results: RunAllEmbeddingDTypeRoundTrips(tag)},
		{Family: "residual", Results: RunAllResidualDTypeRoundTrips(tag)},
	}
}

// MatrixDTypeRoundTripSummary totals pass/fail across all families.
func MatrixDTypeRoundTripSummary(matrix []LayerDTypeMatrix) (pass, fail int, familyFails []string) {
	for _, block := range matrix {
		p, f, report := DTypeRoundTripSummary(block.Results)
		pass += p
		fail += f
		if f > 0 {
			familyFails = append(familyFails, fmt.Sprintf("%s (%d/%d): %s", block.Family, p, p+f, report))
		}
	}
	return pass, fail, familyFails
}
