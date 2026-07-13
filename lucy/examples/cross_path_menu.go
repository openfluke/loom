package examples

import (
	"bufio"

	sevenlayer "github.com/openfluke/loom/lucy/examples/seven_layer"
)

// RunCrossPathMenu runs Lucy menu [15]: SC/MC/SIMD vs native vs native-SIMD on all layers.
func RunCrossPathMenu(reader *bufio.Reader) {
	sevenlayer.RunCrossPathMenu(reader)
}
