package examples

import (
	"bufio"

	tfsimdbench "github.com/openfluke/loom/lucy/examples/transformer_simd_bench"
)

// RunTransformerSimdBenchMenu runs the CPU transformer SIMD-vs-MC decode
// benchmark over quantized .entity checkpoints (Lucy menu [11]).
func RunTransformerSimdBenchMenu(reader *bufio.Reader) {
	tfsimdbench.RunMenu(reader)
}
