package examples

import (
	"bufio"

	"github.com/openfluke/loom/lucy/examples/five_layer"
)

// RunFiveLayerMenu runs per-layer Loom tutorials (Lucy menu [6]).
func RunFiveLayerMenu(reader *bufio.Reader) {
	fivelayer.RunMenu(reader)
}
