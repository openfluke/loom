package examples

import (
	"bufio"

	apple "github.com/openfluke/loom/lucy/examples/apple"
)

// RunAppleMenu runs the Apple (Metal / MPSGraph) GPU CABI bridge suite (Lucy menu [13]).
func RunAppleMenu(reader *bufio.Reader) {
	apple.RunMenu(reader)
}
