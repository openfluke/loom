package examples

import (
	"bufio"

	snapdragon "github.com/openfluke/loom/lucy/examples/snapdragon"
)

// RunSnapdragonMenu runs the Qualcomm (Hexagon) NPU CABI bridge suite (Lucy menu [12]).
func RunSnapdragonMenu(reader *bufio.Reader) {
	snapdragon.RunMenu(reader)
}
