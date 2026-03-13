#!/bin/bash
set -e

echo "Building Loom C ABI shared library..."
go build -o libloom.so -buildmode=c-shared main.go

echo "Compiling Universal Test Suite..."
gcc -o universal_test universal_test.c ./libloom.so -I. -lm

echo "Running Tests..."
export LD_LIBRARY_PATH=.
./universal_test
