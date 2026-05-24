package examples

import (
	"bufio"

	sevenlayer "github.com/openfluke/loom/lucy/examples/seven_layer"
)

// RunSevenLayerMenu runs the seven-layer CPU suite (Lucy menu [7]).
func RunSevenLayerMenu(reader *bufio.Reader) {
	sevenlayer.RunMenu(reader)
}
