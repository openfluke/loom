package examples

import "bufio"

// RunTestsMenu runs the dense stack benchmark: 100 samples, forward vs step.
func RunTestsMenu(reader *bufio.Reader) {
	RunDenseBench(reader)
}
