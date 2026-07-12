package examples

import (
	"bufio"

	sevenlayer "github.com/openfluke/loom/lucy/examples/seven_layer"
)

// RunDenseNativeMenu runs Lucy menu [14]: native layer suite (all layer types).
func RunDenseNativeMenu(reader *bufio.Reader) {
	sevenlayer.RunNativeMenu(reader)
}
