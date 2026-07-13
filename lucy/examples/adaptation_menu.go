package examples

import (
	"bufio"

	sevenlayer "github.com/openfluke/loom/lucy/examples/seven_layer"
)

// RunAdaptationMenu runs Lucy menu [17]: mid-stream adaptation across all layers/dtypes/paths.
func RunAdaptationMenu(reader *bufio.Reader) {
	sevenlayer.RunAdaptationMenu(reader)
}
