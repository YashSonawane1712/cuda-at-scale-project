# cuda-at-scale-project
The purpose of this project is to develop a high-throughput, GPU-accelerated batch data processing pipeline capable of manipulating massive collections of parallel data inputs concurrently on Windows systems. Rather than processing signals sequentially on the CPU, this implementation dynamically transfers large matrix blocks into GPU VRAM.

# Batched CUDA Image Processing Pipeline
This project processes hundreds of data frames concurrently using custom 2D CUDA execution blocks.

### Execution Instructions:
1. Compile the workspace: `make build`
2. Run the application pipeline: `./image_pipeline.exe -b 1.3 -t 140`
