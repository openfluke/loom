package main

import (
	"encoding/json"
	"fmt"
	"os"
)

type APIItem struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

func main() {
	data, err := os.ReadFile("expected_api.json")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	var api map[string][]APIItem
	if err := json.Unmarshal(data, &api); err != nil {
		fmt.Printf("Error unmarshaling: %v\n", err)
		return
	}

	f, _ := os.Create("parity_data.go")
	fmt.Fprintln(f, "package main")
	fmt.Fprintln(f, "")
	fmt.Fprintln(f, "// Auto-generated parity list for 100% C-ABI verification")
	fmt.Fprintln(f, "var totalParityItems = []string{")
	for _, items := range api {
		for _, item := range items {
			fmt.Fprintf(f, "\t\"%s\",\n", item.Name)
		}
	}
	fmt.Fprintln(f, "}")
	f.Close()
	fmt.Println("Generated parity_data.go")
}
