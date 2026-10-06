NVCC = nvcc
CXXFLAGS = --std=c++17 -Wno-deprecated-gpu-targets
OPENCV_FLAGS = `pkg-config opencv4 --cflags --libs`
TARGET = image_pipeline.exe

build: main.cu
	(NVCC) main.cu (CXXFLAGS) \((OPENCV_FLAGS) -o\)(TARGET) -lcuda

clean:
	rm -f \$(TARGET) execution_proof.log
