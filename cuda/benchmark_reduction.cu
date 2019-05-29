#include <absl/strings/str_format.h>
#include <benchmark/benchmark.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <iostream>
#include <random>

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

BENCHMARK_DEFINE_F(RandomArrayFixture, BM_RandomNumberGeneration)
(benchmark::State& state) {
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
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          state.range(0));
}

class ReductionFixture : public benchmark::Fixture {
 public:
  void SetUp(const ::benchmark::State& state) {
    n = state.range(0);
    array = new float[n];
    CUDA_CALL(cudaMalloc(&device_array, n * sizeof(float)));
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rd()));
  }

  void TearDown(const ::benchmark::State& state) {
    delete array;
    CURAND_CALL(curandDestroyGenerator(gen));
    CUDA_CALL(cudaFree(device_array));
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));
  }

 protected:
  float* device_array;
  float* array;
  int n;
  cudaEvent_t start, stop;
  curandGenerator_t gen;
  std::random_device rd;
};

BENCHMARK_DEFINE_F(ReductionFixture, BM_FindMax0)
(benchmark::State& state) {
  for (auto _ : state) {
    // Generate random number
    CURAND_CALL(curandGenerateUniform(gen, device_array, n));
    CUDA_CALL(cudaMemcpy(array, device_array, n * sizeof(float),
                         cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaEventRecord(start));
    FindMax0<<<64, 64, 64 * sizeof(float)>>>(device_array, n, device_array);
    CUDA_CALL(cudaEventRecord(stop));
    FindMax0<<<1, 64, 64 * sizeof(float)>>>(device_array, n, device_array);

    float max_host = HostFindMax(array, n);
    CUDA_CALL(cudaMemcpy(array, device_array, 1 * sizeof(float),
                         cudaMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(max_host, array[0]);

    float milliseconds = 0;
    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    // Takes second
    state.SetIterationTime(milliseconds / 1.0e3);
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          state.range(0) * 4);
}

BENCHMARK_DEFINE_F(ReductionFixture, BM_FindMax1)
(benchmark::State& state) {
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
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          state.range(0) * 4);
}

// 2^(28+2) requires 2GB of VRAM
BENCHMARK_REGISTER_F(RandomArrayFixture, BM_RandomNumberGeneration)
    ->RangeMultiplier(2)
    ->Range(1 << 12, 1 << 28)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(ReductionFixture, BM_FindMax0)
    ->RangeMultiplier(2)
    ->Range(1 << 12, 1 << 28)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(ReductionFixture, BM_FindMax1)
    ->RangeMultiplier(2)
    ->Range(1 << 12, 1 << 28)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);
