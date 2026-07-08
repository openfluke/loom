package examples

import (
	"bufio"

	contextsuite "github.com/openfluke/loom/lucy/examples/context_suite"
)

// RunContextSuiteMenu runs the long-context / multi-prompt suite (Lucy menu [10]).
func RunContextSuiteMenu(reader *bufio.Reader) {
	contextsuite.RunMenu(reader)
}

// RunContextSuiteAuto runs the full matrix without prompts (LOOM_CONTEXT_SUITE=1).
func RunContextSuiteAuto() {
	contextsuite.RunAuto()
}
