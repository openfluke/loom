package examples

import (
	"bufio"

	sevenlayer "github.com/openfluke/loom/lucy/examples/seven_layer"
)

// RunTweenNativeMenu runs Lucy menu [16]: native tween SC vs native-SIMD on all layers.
func RunTweenNativeMenu(reader *bufio.Reader) {
	sevenlayer.RunTweenNativeMenu(reader)
}
