#include <glog/logging.h>
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <absl/strings/str_format.h>

#include <random>
#include <iostream>

#include <curand.h>

// Compile with nvcc reduction.cu -lcurand -lglog -lgflags

// __global__ void RandomArray(float* array, int n) {

// }

// FindMax Assumes that 1D block and 1D thread block. Also the number of
// elements to be processed should exceed the number of workers, which means
// that n >= stride. Otherwise, the behaviour is undefined.
__global__ void FindMax0(const float* array, int n,
                        // output parameter
                        float* out_array) {
  // What happens if the shared memory is not enough? Runtime error?
  extern __shared__ float shared_array[];
  // First, for each thread of each block, look for max
  int thread_id = threadIdx.x;
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // float local_max = array[id];
  // Find max for each thread
  // // This not only requires commutative property and associative property
  // for (int i = id + stride; i < n; i += stride) {
  //   if (local_max < array[i]) {
  //     local_max = array[i];
  //   }
  // }

  // Dispatch workload evenly
  int workload = n / stride;
  int start = workload * id;
  int end = workload * (id + 1);
  // last one takes up more work
  if (id == stride - 1)
    end = n;

  float local_max = array[start];

  for (int i = start; i < end; ++i) {
    if (local_max < array[i]) {
      local_max = array[i];
    }
  }

  // Each thread loads its max into each block's shared mem
  shared_array[thread_id] = local_max;
  __syncthreads();

  // Perform reduction upon each block
  for (int num_workers = blockDim.x / 2, step_size = 1; num_workers > 0;
       num_workers /= 2, step_size *= 2) {
    if (thread_id < num_workers) {
      int index = 2 * step_size * thread_id;
      float num1 = shared_array[index];
      float num2 = shared_array[index + step_size];

      if (num1 < num2)
        shared_array[index] = num2;
      else
        shared_array[index] = num1;
    }
    __syncthreads();
  }
  // All blocks have its max in shared_mem[0]
  // Write to global memory for synchronization
  if (thread_id == 0)
    out_array[blockIdx.x] = shared_array[0];
}

// FindMax Assumes that 1D block and 1D thread block. Also the number of
// elements to be processed should exceed the number of workers, which means
// that n >= stride. Otherwise, the behaviour is undefined.
__global__ void FindMax1(const float* array, int n,
                         // output parameter
                         float* out_array) {
  // What happens if the shared memory is not enough? Runtime error?
  extern __shared__ float shared_array[];
  // First, for each thread of each block, look for max
  int thread_id = threadIdx.x;
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  float local_max = array[id];
  // Find max for each thread
  // This not only requires commutative property and associative property
  for (int i = id + stride; i < n; i += stride) {
    if (local_max < array[i]) {
      local_max = array[i];
    }
  }

  // Each thread loads its max into each block's shared mem
  shared_array[thread_id] = local_max;
  __syncthreads();

  // Perform reduction upon each block
  for (int num_workers = blockDim.x / 2, step_size = 1; num_workers > 0;
       num_workers /= 2, step_size *= 2) {
    if (thread_id < num_workers) {
      int index = 2 * step_size * thread_id;
      float num1 = shared_array[index];
      float num2 = shared_array[index + step_size];

      if (num1 < num2)
        shared_array[index] = num2;
      else
        shared_array[index] = num1;
    }
    __syncthreads();
  }
  // All blocks have its max in shared_mem[0]
  // Write to global memory for synchronization
  if (thread_id == 0)
    out_array[blockIdx.x] = shared_array[0];
}

float HostFindMax(const float* array, int n) {
  if (n <= 0) return -1;
  float max = array[0];
  for (int i = 0; i < n; ++i) {
    if (array[i] > max) {
      max = array[i];
    }
  }
  return max;
}

// #define CUDA_CALL(x)                                  \
//   do {                                                \
//     if ((x) != cudaSuccess) {                         \
//       printf("Error at %s:%d\n", __FILE__, __LINE__); \
//       return EXIT_FAILURE;                            \
//     }                                                 \
//   } while (0)

#define CUDA_CALL(x)            \
  do {                          \
    CHECK_EQ((x), cudaSuccess); \
  } while (0)

#define CURAND_CALL(x)                    \
  do {                                    \
    CHECK_EQ((x), CURAND_STATUS_SUCCESS); \
  } while (0)

// #define CUDA_CALL(x)                                              \
//   do {                                                            \
//     CHECK_EQ((x), cudaSuccess)                                    \
//         << absl::StrFormat("Error at %s:%d", __FILE__, __LINE__); \
//   } while (0)

// #define CURAND_CALL(x)                                              \
//   do {                                                              \
//     CHECK_EQ((x), CURAND_STATUS_SUCCESS)                            \
//         << absl::StrFormat("Error at %s:%d\n", __FILE__, __LINE__); \
//   } while (0)

// #define CURAND_CALL(x)                                \
//   do {                                                \
//     if ((x) != CURAND_STATUS_SUCCESS) {               \
//       printf("Error at %s:%d\n", __FILE__, __LINE__); \
//       return EXIT_FAILURE;                            \
//     }                                                 \
//   } while (0)

// What we are going to do next:
// Learning to combine the great google benchmark with cuda GPU execution
// static void ReductionTiming(benchmark::State& state) {
//   // int microseonds = state.range(0);
//   // std::chrono::duration<double, std::micro> sleep_duration {
//   //   static_cast<double>(microseonds);
//   // }
//   for (auto _ : state) {
//     auto start = std::chrono::high_resolution_clock;
//     // work
//     auto end = std::chrono::high_resolution_clock;
//     auto elapsed_seconds =
//         std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//     state.SetIterationTime(elapsed_seconds.count());
//   }
// }

static void BM_RandomNumberGeneration(benchmark::State& state) {
  float* device_array;
  int n = state.range(0);
  CUDA_CALL(cudaMalloc(&device_array, n * sizeof(float)));
  std::random_device rd;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  curandGenerator_t gen;

  for (auto _ : state) {
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rd()));
    cudaEventRecord(start);
    CURAND_CALL(curandGenerateUniform(gen, device_array, n));
    cudaEventRecord(stop);
    CURAND_CALL(curandDestroyGenerator(gen));
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    state.SetIterationTime(static_cast<double>(milliseconds));    
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * state.range(0) * 4);
  cudaFree(device_array);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

// 2^(28+2) requires 2GB of VRAM
BENCHMARK(BM_RandomNumberGeneration)
    ->RangeMultiplier(2)
    ->Range(1 << 12, 1 << 28)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

static void BM_FindMax(benchmark::State& state) {
  float* device_array;
  int n = state.range(0);
  float* array = new float[n];
  CUDA_CALL(cudaMalloc(&device_array, n * sizeof(float)));
  std::random_device rd;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  curandGenerator_t gen;
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rd()));

  for (auto _ : state) {
    // Generate random number
    CURAND_CALL(curandGenerateUniform(gen, device_array, n));
    CUDA_CALL(cudaMemcpy(array, device_array, n * sizeof(float),
                         cudaMemcpyDeviceToHost));

    cudaEventRecord(start);
    FindMax0<<<64, 64, 64 * sizeof(float)>>>(device_array, n, device_array);
    FindMax0<<<1, 64, 64 * sizeof(float)>>>(device_array, n, device_array);
    cudaEventRecord(stop);
    float max_host = HostFindMax(array, n);
    CUDA_CALL(cudaMemcpy(array, device_array, 1 * sizeof(float),
                         cudaMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(max_host, array[0]);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    state.SetIterationTime(milliseconds);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * state.range(0) * 4);
  CURAND_CALL(curandDestroyGenerator(gen));
  cudaFree(device_array);
  delete array;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

BENCHMARK(BM_FindMax)
    ->RangeMultiplier(2)
    ->Range(1 << 12, 1 << 28)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

// int main(int argc, char* argv[]) {
//   if (argc != 2) return -1;
//   int n = std::atoi(argv[1]);
//   float* device_array;
//   float* array;
//   curandGenerator_t gen;

//   CUDA_CALL(cudaMalloc(&device_array, n * sizeof(float)));
//   array = new float[n];// malloc(n * sizeof(float));
//   // max_index = new int; // malloc(max_index, sizeof(int));

//   // Random number generation
//   std::random_device rd;
//   CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937));
//   CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rd()));
//   CURAND_CALL(curandGenerateUniform(gen, device_array, n));
//   CURAND_CALL(curandDestroyGenerator(gen));
//   CUDA_CALL(cudaDeviceSynchronize());

//   // Since the FindMax would alter device_array, first copy to host
//   CUDA_CALL(cudaMemcpy(array, device_array, n * sizeof(float),
//                        cudaMemcpyDeviceToHost));
//   // for (int i = 0; i < n-1; ++i)
//   //   std::cout << array[i] << ' ';
//   // std::cout << array[n-1] << '\n';
//   float max_host = HostFindMax(array, n);

//   // First FindMax() would assign the local maxima to the first 64 of device
//   // array, which is the block size of the first call
//   FindMax<<<64, 64, 64 * sizeof(float)>>>(device_array, n, device_array);
//   FindMax<<<1, 64, 64 * sizeof(float)>>>(device_array, n, device_array);
//   CUDA_CALL(cudaDeviceSynchronize());
//   CUDA_CALL(cudaMemcpy(array, device_array, 1 * sizeof(float),
//                        cudaMemcpyDeviceToHost));

//   float max_device = array[0];
//   CHECK_EQ(max_host, max_device);
//   std::cout << max_host << '\n';
//   // for (int i = 0; i < n-1; ++i)
//   //   std::cout << array[i] << ' ';
//   // std::cout << array[n-1] << '\n';

//   // int kernel_max_index = *max_index;
//   cudaFree(device_array);
//   //  cudaFree(device_max_index);
//   free(array);
//   // free(max_index);
// }