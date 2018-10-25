#include <benchmark/benchmark.h>

static void BM_Simple(benchmark::State &state) {
  for (auto _ : state) {
    // benchmark code
  }
}

// passing parameters
BENCHMARK(BM_Simple)->RangeMultiplier(2)->Range(1, 32);

BENCHMARK_MAIN();
