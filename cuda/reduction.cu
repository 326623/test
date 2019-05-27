#include <glog/logging.h>
#include <random>

#include <curand.h>

// __global__ void RandomArray(float* array, int n) {

// }

__global__ void FindMax(const float* array, int n,
                        // output parameter
                        int* max_index) {
  int index = 
}

void FindMax(const float* array, int n, int* max_index) {
  if (n <= 0) return;
  int max = array[0];
  for (int i = 0; i < n; ++i) {
    if (array[i] < max) {
      *max_index = i;
      max = array[i];
    }
  }
}

#define CUDA_CALL(x)                                  \
  do {                                                \
    if ((x) != cudaSuccess) {                         \
      printf("Error at %s:%d\n", __FILE__, __LINE__); \
      return EXIT_FAILURE;                            \
    }                                                 \
  } while (0)

#define CURAND_CALL(x)                                \
  do {                                                \
    if ((x) != CURAND_STATUS_SUCCESS) {               \
      printf("Error at %s:%d\n", __FILE__, __LINE__); \
      return EXIT_FAILURE;                            \
    }                                                 \
  } while (0)

int main(int argc, char* argv[]) {
  if (argc != 2) return -1;
  int n = std::atoi(argv[1]);
  float* device_array;
  int* device_max_index;
  float* array;
  int* max_index;
  curandGenerator_t gen;

  CUDA_CALL(cudaMalloc(device_array, n * sizeof(float)));
  CUDA_CALL(cudaMalloc(device_max_index, sizeof(int)));
  malloc(array, n * sizeof(float));
  malloc(max_index, sizeof(int));

  // Random number generation
  std::random_device rd;
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rd()));
  CURAND_CALL(curandGenerateUniform(gen, device_array, n));
  CURAND_CALL(curandDestroyGenerator(gen));

  FindMax<<<>>>(device_array, n, device_max_index);
  CUDA_CALL(CudaMemcpy(device_array, array, n * sizeof(float),
                       cudaMemcpyDeviceToHost));
  CUDA_CALL(CudaMemcpy(device_max_index, max_index, sizeof(int),
                       cudaMemcpyDeviceToHost));

  int kernel_max_index = *max_index;
  FindMax(array, n, max_index);
  CHECK_EQ(*max_index, kernel_max_index);

  cudaFree(device_array);
  cudaFree(device_max_index);
  free(array);
  free(max_index);
}