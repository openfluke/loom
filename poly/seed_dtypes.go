package poly

import (
	"fmt"
	"strings"
)

// SeedDTypesAll returns the 21 canonical weight dtypes (float64 … binary).
func SeedDTypesAll() []DType {
	out := make([]DType, 0, 21)
	for d := DTypeFloat64; d <= DTypeBinary; d++ {
		out = append(out, d)
	}
	return out
}

// DTypeRoundTripResult is one dtype round-trip outcome on a layer family.
type DTypeRoundTripResult struct {
	Family       string
	DType        DType
	DTypeName    string
	TopologySeed uint64
	LayerSeed    uint64
	WeightFP     uint64
	ForwardFP    uint64
	OK           bool
	Err          string
}

// DenseDTypeTopologySeed is a per-dtype topology seed for dtype sweep tests.
func DenseDTypeTopologySeed(tag string, sizes []int, dt DType) uint64 {
	base := DenseTopologySeed(tag, sizes)
	return SeedFrom(base, "dtype", dt.String())
}

// RunDenseDTypeRoundTrip verifies seeds→weights→forward→seeds for one dtype.
func RunDenseDTypeRoundTrip(tag string, sizes []int, dt DType) DTypeRoundTripResult {
	res := DTypeRoundTripResult{Family: "dense", DType: dt, DTypeName: dt.String()}
	if len(sizes) < 2 {
		res.Err = "need at least two sizes"
		return res
	}
	topo := DenseDTypeTopologySeed(tag, sizes, dt)
	dtype := dt.String()
	manifest, err := BuildDenseManifest(topo, sizes, []string{dtype})
	if err != nil {
		res.Err = err.Error()
		return res
	}
	res.TopologySeed = topo
	if len(manifest.Layers) > 0 {
		res.LayerSeed = manifest.Layers[0].LayerSeed
		res.WeightFP = manifest.Layers[0].WeightFP
	}

	rebuilt, err := RebuildDenseManifest(manifest)
	if err != nil {
		res.Err = "rebuild: " + err.Error()
		return res
	}
	if rebuilt.NetworkFP != manifest.NetworkFP {
		res.Err = fmt.Sprintf("network fp mismatch 0x%x vs 0x%x", rebuilt.NetworkFP, manifest.NetworkFP)
		return res
	}

	netA, err := BuildDenseVolumetricFromManifest(manifest)
	if err != nil {
		res.Err = "build A: " + err.Error()
		return res
	}
	netB, err := BuildDenseVolumetricFromManifest(rebuilt)
	if err != nil {
		res.Err = "build B: " + err.Error()
		return res
	}
	hashA, err := denseForwardOutputHash(netA, sizes[0])
	if err != nil {
		res.Err = "forward A: " + err.Error()
		return res
	}
	hashB, err := denseForwardOutputHash(netB, sizes[0])
	if err != nil {
		res.Err = "forward B: " + err.Error()
		return res
	}

	extracted, err := ManifestFromDenseNetwork(netA, topo, sizes, []string{dtype})
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

// RunAllDenseDTypeRoundTrips runs every canonical dtype through dense round trip.
func RunAllDenseDTypeRoundTrips(tag string, sizes []int) []DTypeRoundTripResult {
	results := make([]DTypeRoundTripResult, 0, 21)
	for _, dt := range SeedDTypesAll() {
		results = append(results, RunDenseDTypeRoundTrip(tag, sizes, dt))
	}
	return results
}

// DTypeRoundTripSummary counts passes and builds a failure report.
func DTypeRoundTripSummary(results []DTypeRoundTripResult) (pass, fail int, report string) {
	var fails []string
	for _, r := range results {
		if r.OK {
			pass++
		} else {
			fail++
			msg := r.Err
			if msg == "" {
				msg = "unknown"
			}
			label := r.DTypeName
			if r.Family != "" {
				label = r.Family + "/" + label
			}
			fails = append(fails, fmt.Sprintf("%s: %s", label, msg))
		}
	}
	return pass, fail, strings.Join(fails, "; ")
}

// LayerDTypeMatrix is one layer family × 21 dtype results.
type LayerDTypeMatrix struct {
	Family  string
	Results []DTypeRoundTripResult
}

func layerDTypeTopo(base uint64, dt DType) uint64 {
	return SeedFrom(base, "dtype", dt.String())
}

func finishDTypeRoundTrip(
	res DTypeRoundTripResult,
	rebuildErr error,
	manifestNetworkFP uint64,
	manifestLayerSeed uint64,
	hashA, hashB uint64,
	extractedNetworkFP uint64,
	extractedLayerSeed uint64,
	extractedForwardFP uint64,
) DTypeRoundTripResult {
	if rebuildErr != nil {
		res.Err = "rebuild: " + rebuildErr.Error()
		return res
	}
	if hashA != hashB {
		res.Err = fmt.Sprintf("forward mismatch 0x%x vs 0x%x", hashA, hashB)
		return res
	}
	if extractedNetworkFP != manifestNetworkFP {
		res.Err = "extracted network fp mismatch"
		return res
	}
	if extractedLayerSeed != manifestLayerSeed {
		res.Err = fmt.Sprintf("seed mismatch 0x%x vs 0x%x", extractedLayerSeed, manifestLayerSeed)
		return res
	}
	if extractedForwardFP != hashA {
		res.Err = fmt.Sprintf("forward fp mismatch extract=0x%x hash=0x%x", extractedForwardFP, hashA)
		return res
	}
	res.OK = true
	return res
}

func volumetricForwardHash(net *VolumetricNetwork, inputDim int) (uint64, error) {
	return denseForwardOutputHash(net, inputDim)
}

func denseForwardOutputHash(net *VolumetricNetwork, inputDim int) (uint64, error) {
	if net == nil {
		return 0, fmt.Errorf("nil network")
	}
	in := NewTensorFromSlice(seedDemoForwardInput(inputDim), 1, inputDim)
	out, _, _ := ForwardPolymorphic(net, in)
	if out == nil {
		return 0, fmt.Errorf("forward nil")
	}
	return seedOutputHash(out.Data), nil
}
