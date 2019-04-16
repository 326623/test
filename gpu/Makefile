add: add.cpp
	g++ $^ -o $@

add_cuda: add.cu
	nvcc $^ -o $@

# To run the profile without sudo
# nvprof --unified-memory-profiling off ./add_cuda
