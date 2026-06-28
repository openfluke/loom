package examples

import (
	"bufio"

	ninelayer "github.com/openfluke/loom/lucy/examples/nine_layer"
)

// RunNineLayerMenu runs the Intel NPU CABI bridge suite (Lucy menu [9]).
func RunNineLayerMenu(reader *bufio.Reader) {
	ninelayer.RunMenu(reader)
}
