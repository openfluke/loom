package sevenlayer

import (
	"fmt"
	"strings"

	"github.com/openfluke/loom/poly"
)

func denseSuiteForGrid(g GridSpec) LayerSuite {
	dims := denseEndpoints(g)
	acts := []string{"LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR"}
	return LayerSuite{
		Name:        "Dense",
		Grid:        g,
		PrimaryType: poly.LayerDense,
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "diag-dense", g)
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
}

func swigluSuiteForGrid(g GridSpec) LayerSuite {
	dims := swigluEndpoints(g)
	return LayerSuite{
		Name:        "SwiGLU",
		Grid:        g,
		PrimaryType: poly.LayerSwiGLU,
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "diag-swiglu", g)
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
}
