package pods

import (
	"errors"
	"sort"
)

// Runner is a simple demo/algorithm entry point.
type Runner func() error

var registry = map[string]Runner{}

func Register(name string, fn Runner) { registry[name] = fn }

func Names() []string {
	out := make([]string, 0, len(registry))
	for k := range registry {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func Run(name string) error {
	fn, ok := registry[name]
	if !ok {
		return errors.New("unknown pod: " + name)
	}
	return fn()
}
