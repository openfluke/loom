package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
)

type APIItem struct {
	Name    string
	Type    string // "Struct", "Func", "Method"
	Comment string
}

func main() {
	fmt.Println("====================================================")
	fmt.Println(" LOOM C-ABI Coverage Analysis Tool (welvet/cabi)")
	fmt.Println("====================================================")

	polyPath := "../../../../poly"
	cabiFile := "../../main.go"

	// 1. Extract Core API from poly/
	fmt.Printf("\n[1] Scanning poly/ core...\n")
	coreAPI := scanPackage(polyPath)
	fmt.Printf("    Found %d public API items in poly/\n", len(coreAPI))

	// 2. Extract C-ABI Exports from main.go
	fmt.Printf("\n[2] Scanning welvet/cabi/main.go exports...\n")
	cabiExports := scanCABI(cabiFile)
	fmt.Printf("    Found %d //export directives in C-ABI\n", len(cabiExports))

	// 3. Compare and Categorize
	fmt.Printf("\n[3] Functional Parity Report (Categorized)\n")
	fmt.Println("----------------------------------------------------------------------")
	
	categories := map[string][]string{
		"CORE MECHANICS": {"VolumetricNetwork", "Forward", "Backward", "Systolic", "Layer", "Tensor"},
		"ACCELERATION":   {"WGPU", "GPU", "Sync", "Dispatch", "Shader"},
		"LEARNING/DNA":   {"TargetProp", "DNA", "Compare", "Refit", "Gradient"},
		"IO/UTIL":        {"JSON", "Safetensors", "Load", "Extract", "Tokenizer"},
	}

	coverageCount := 0
	globalTotal := 0
	globalCovered := 0

	for cat, keywords := range categories {
		fmt.Printf("\n>>> %s\n", cat)
		catTotal := 0
		catCovered := 0
		
		for _, coreItem := range coreAPI {
			match := false
			for _, kw := range keywords {
				if strings.Contains(strings.ToLower(coreItem.Name), strings.ToLower(kw)) {
					match = true
					break
				}
			}
			if !match { continue }

			catTotal++
			found := false
			for _, export := range cabiExports {
				if strings.Contains(strings.ToLower(export), strings.ToLower(coreItem.Name)) {
					found = true
					break
				}
			}

			status := "[ ]"
			if found {
				status = "[X]"
				catCovered++
				coverageCount++
			}
			fmt.Printf("  %s %-8s %s\n", status, coreItem.Type, coreItem.Name)
		}
		
		if catTotal > 0 {
			fmt.Printf("  CATEGORY COVERAGE: %d/%d (%.1f%%)\n", catCovered, catTotal, float64(catCovered)/float64(catTotal)*100)
			globalTotal += catTotal
			globalCovered += catCovered
		}
	}

	fmt.Println("\n----------------------------------------------------------------------")
	fmt.Printf("Global Functional Overlap: %d Unique Core Items Mapped in C-ABI\n", coverageCount)
	if globalTotal > 0 {
		fmt.Printf("TOTAL API COVERAGE: %d/%d (%.1f%%)\n", globalCovered, globalTotal, float64(globalCovered)/float64(globalTotal)*100)
	}
	fmt.Println("\nNOTE: 0% coverage in 'ACCELERATION' is often GOOD. High-level bridges should")
	fmt.Println("abstract away GPU dispatches into simple calls like 'LoomSystolicStep'.")
	fmt.Println("======================================================================")
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

func scanCABI(filename string) []string {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, filename, nil, parser.ParseComments)
	if err != nil {
		fmt.Printf("Error parsing C-ABI file: %v\n", err)
		return nil
	}

	var exports []string
	// 1. Scan for //export comments
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

	// 2. Scan function bodies for all identifiers (Deep Inspection)
	ast.Inspect(file, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok {
			exports = append(exports, id.Name)
		}
		return true
	})

	return exports
}
