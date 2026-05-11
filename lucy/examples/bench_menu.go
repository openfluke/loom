package examples

import (
	"bufio"

	denseadaptation "github.com/openfluke/loom/lucy/examples/dense_adaptation"
)

// RunTestsMenu runs the dense mid-stream adaptation benchmark.
func RunTestsMenu(reader *bufio.Reader) {
	denseadaptation.RunDenseAdaptationBenchmark(reader)
}
