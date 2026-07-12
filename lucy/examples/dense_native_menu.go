package examples

import (
	"bufio"

	sevenlayer "github.com/openfluke/loom/lucy/examples/seven_layer"
)

// RunDenseNativeMenu runs Lucy menu [14]: dense native-exact forward/backward × 21 dtypes.
func RunDenseNativeMenu(reader *bufio.Reader) {
	sevenlayer.RunDenseNativeMenu(reader)
}
