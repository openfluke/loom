#!/bin/bash

# Output file
OUTPUT_FILE="all_tests_output.txt"

# Clear/Init output file
echo "==========================================" > "$OUTPUT_FILE"
echo "      NEURAL TWEEN TEST SUITE RUN         " >> "$OUTPUT_FILE"
echo "      Date: $(date)                       " >> "$OUTPUT_FILE"
echo "==========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# List of files to run
files=(
    "main.go"
    "test13_step_tween.go"
    "test14_step_tween_large.go"
    "test15_step_tween_deep.go"
)

# Colors for console output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}Running $file...${NC}"
        
        # Header in the file
        echo "################################################################################" >> "$OUTPUT_FILE"
        echo ">>> EXECUTING: $file" >> "$OUTPUT_FILE"
        echo "################################################################################" >> "$OUTPUT_FILE"
        
        # Run and append output (stdout and stderr)
        go run "$file" >> "$OUTPUT_FILE" 2>&1
        
        echo "Finished $file"
        echo "" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
    else
        echo "Warning: $file not found!"
        echo "Warning: $file not found!" >> "$OUTPUT_FILE"
    fi
done

echo -e "${GREEN}All tests completed. Output saved to $OUTPUT_FILE${NC}"
