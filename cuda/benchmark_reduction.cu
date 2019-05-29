#include <glog/logging.h>
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <absl/strings/str_format.h>

#include <random>
#include <iostream>

#include <curand.h>

#include "cuda_macro.h"
#include "reduction.cuh"

class RandomArrayFixture : public benchmark::Fixture {
 public:
  void SetUp(const ::benchmark::State& state) {
    n = state.range(0);
    CUDA_CALL(cudaMalloc(&device_array, n * sizeof(float)));
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
  }

  void TearDown(const ::benchmark::State& state) {
    CUDA_CALL(cudaFree(device_array));
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));
  }
 protected:
  float* device_array;
  int n;
  cudaEvent_t start, stop;
  curandGenerator_t gen;
  std::random_device rd;
};

BENCHMARK_DEFINE_F(RandomArrayFixture, BM_RandomNumberGeneration)(benchmark::State& state) {
  for (auto _ : state) {
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rd()));
    CUDA_CALL(cudaEventRecord(start));
    CURAND_CALL(curandGenerateUniform(gen, device_array, n));
    CUDA_CALL(cudaEventRecord(stop));
    CURAND_CALL(curandDestroyGenerator(gen));
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    state.SetIterationTime(static_cast<double>(milliseconds / 1.0e3));
  }
}

BENCHMARK_REGISTER_F(RandomArrayFixture, BM_RandomNumberGeneration)
    ->RangeMultiplier(2)
    ->Range(1 << 12, 1 << 28)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

// static void BM_RandomNumberGeneration(benchmark::State& state) {
//   float* device_array;
//   int n = state.range(0);
//   CUDA_CALL(cudaMalloc(&device_array, n * sizeof(float)));
//   std::random_device rd;
//   cudaEvent_t start, stop;
//   CUDA_CALL(cudaEventCreate(&start));
//   CUDA_CALL(cudaEventCreate(&stop));
//   curandGenerator_t gen;

//   for (auto _ : state) {
//     CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937));
//     CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rd()));
//     CUDA_CALL(cudaEventRecord(start));
//     CURAND_CALL(curandGenerateUniform(gen, device_array, n));
//     CUDA_CALL(cudaEventRecord(stop));
//     CURAND_CALL(curandDestroyGenerator(gen));
//     cudaEventSynchronize(stop);
//     float milliseconds = 0;
//     CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
//     state.SetIterationTime(static_cast<double>(milliseconds / 1.0e3));
//   }

//   state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * state.range(0) * 4);
//   CUDA_CALL(cudaFree(device_array));
//   CUDA_CALL(cudaEventDestroy(start));
//   CUDA_CALL(cudaEventDestroy(stop));
// }

// 2^(28+2) requires 2GB of VRAM
// BENCHMARK(BM_RandomNumberGeneration)
//     ->RangeMultiplier(2)
//     ->Range(1 << 12, 1 << 28)
//     ->UseManualTime()
//     ->Unit(benchmark::kMillisecond);
class RandomArrayFixture : public benchmark::Fixture {
 public:
  void SetUp(const ::benchmark::State& state) {
    n = state.range(0);
    CUDA_CALL(cudaMalloc(&device_array, n * sizeof(float)));
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
  }

  void TearDown(const ::benchmark::State& state) {
    CUDA_CALL(cudaFree(device_array));
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));
  }
 protected:
  float* device_array;
  int n;
  cudaEvent_t start, stop;
  curandGenerator_t gen;
  std::random_device rd;
};

static void BM_FindMax(benchmark::State& state) {
  float* device_array;
  int n = state.range(0);
  float* array = new float[n];
  CUDA_CALL(cudaMalloc(&device_array, n * sizeof(float)));
  std::random_device rd;
  cudaEvent_t start, stop;
  CUDA_CALL(cudaEventCreate(&start));
  CUDA_CALL(cudaEventCreate(&stop));
  curandGenerator_t gen;
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rd()));

  for (auto _ : state) {
    // Generate random number
    CURAND_CALL(curandGenerateUniform(gen, device_array, n));
    CUDA_CALL(cudaMemcpy(array, device_array, n * sizeof(float),
                         cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaEventRecord(start));
    FindMax1<<<64, 64, 64 * sizeof(float)>>>(device_array, n, device_array);
    CUDA_CALL(cudaEventRecord(stop));
    FindMax1<<<1, 64, 64 * sizeof(float)>>>(device_array, n, device_array);

    float max_host = HostFindMax(array, n);
    CUDA_CALL(cudaMemcpy(array, device_array, 1 * sizeof(float),
                         cudaMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(max_host, array[0]);

    float milliseconds = 0;
    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    // Takes second
    state.SetIterationTime(milliseconds / 1.0e3);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * state.range(0) * 4);
  CURAND_CALL(curandDestroyGenerator(gen));
  CUDA_CALL(cudaFree(device_array));
  delete array;
  CUDA_CALL(cudaEventDestroy(start));
  CUDA_CALL(cudaEventDestroy(stop));
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