package main

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
)

type APIItem struct {
	Name string `json:"name"`
	Type string `json:"type"` // "Struct", "Func", "Method"
}

func main() {
	polyPath := `c:\git\soul\loom\poly`
	cabiPath := `c:\git\soul\loom\welvet\cabi`

	// 1. Extract Core API from poly/
	coreAPI := scanPackage(polyPath)

	// 2. Extract C-ABI Exports from main.go
	cabiExports := scanCABI(cabiPath)

	categories := map[string][]string{
		"CORE MECHANICS": {"VolumetricNetwork", "Forward", "Backward", "Step", "Layer", "Tensor"},
		"ACCELERATION":   {"WGPU", "GPU", "Sync", "Dispatch", "Shader"},
		"LEARNING/DNA":   {"Tween", "DNA", "Compare", "Refit", "Gradient"},
		"IO/UTIL":        {"JSON", "Safetensors", "Load", "Extract", "Tokenizer"},
	}

	result := make(map[string][]APIItem)
	for cat, keywords := range categories {
		var catItems []APIItem
		for _, coreItem := range coreAPI {
			match := false
			for _, kw := range keywords {
				if strings.Contains(strings.ToLower(coreItem.Name), strings.ToLower(kw)) {
					match = true
					break
				}
			}
			if !match {
				continue
			}

			found := false
			for _, export := range cabiExports {
				if strings.Contains(strings.ToLower(export), strings.ToLower(coreItem.Name)) {
					found = true
					break
				}
			}

			if found {
				catItems = append(catItems, coreItem)
			}
		}
		result[cat] = catItems
	}

	b, _ := json.MarshalIndent(result, "", "  ")
	os.WriteFile("expected_api.json", b, 0644)
	fmt.Printf("Generated expected_api.json in current directory.\n")
}

func scanPackage(path string) []APIItem {
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, path, nil, parser.ParseComments)
	if err != nil {
		fmt.Printf("Error parsing poly dir: %v\n", err)
		return nil
	}

	var items []APIItem
	for _, pkg := range pkgs {
		for _, file := range pkg.Files {
			ast.Inspect(file, func(n ast.Node) bool {
				switch x := n.(type) {
				case *ast.TypeSpec:
					if ast.IsExported(x.Name.Name) {
						items = append(items, APIItem{Name: x.Name.Name, Type: "Struct"})
					}
				case *ast.FuncDecl:
					if ast.IsExported(x.Name.Name) {
						typeStr := "Func"
						if x.Recv != nil {
							typeStr = "Method"
						}
						items = append(items, APIItem{Name: x.Name.Name, Type: typeStr})
					}
				}
				return true
			})
		}
	}
	return items
}

func scanCABI(path string) []string {
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, path, nil, parser.ParseComments)
	if err != nil {
		fmt.Printf("Error parsing C-ABI dir: %v\n", err)
		return nil
	}

	var exports []string
	for _, pkg := range pkgs {
		for _, file := range pkg.Files {
			for _, cg := range file.Comments {
				for _, c := range cg.List {
					if strings.HasPrefix(c.Text, "//export ") {
						parts := strings.Fields(c.Text)
						if len(parts) >= 2 {
							exports = append(exports, parts[1])
						}
					}
				}
			}
			ast.Inspect(file, func(n ast.Node) bool {
				if id, ok := n.(*ast.Ident); ok {
					exports = append(exports, id.Name)
				}
				return true
			})
		}
	}
	return exports
}
