#!/bin/bash
echo "=== Cleaning and Compiling CUDA Pipeline ==="
make clean
make build

if [ $? -eq 0 ]; then
    echo "=== Launching Batched Processing Execution ==="
    ./image_pipeline.exe -b 1.3 -t 140
    echo "=== Execution Log Verification ==="
    cat execution_proof.log
else
    echo "Compilation failed."
    exit 1
fi
