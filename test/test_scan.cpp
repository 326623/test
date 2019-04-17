/* Copyright (C) 2018 New Joy - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GPLv3
 *
 *
 * You should have received a copy of the GPLv3 license with
 * this file. If not, please visit https://www.gnu.org/licenses/gpl-3.0.en.html
 *
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: yangqp5@outlook.com (New Joy)
 *
 */
#include <benchmark/benchmark.h>
#include <iostream>
#include <vector>
#include <memory>

#include "scan.cuh"

// note: this implementation does not disable this overload for array types
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// int main() {
//   const int kN = 100;
//   std::size_t size = N * sizeof(float);
//   auto h_A = make_unique<float>(size);
//   float& d_A;
//   cudaMalloc(&d_A, size);
//   cudaError_t error = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
//   // if (error != cudaSuccess) {
//   // }
//   inclusive_prefix_sum<16 * 16, 16 * 16>(
//       d_A, d_A + size, 0, [](float a, float b) { return a + b; });
// }
TEST(TEST_SCAN, TEST1) {
  const int kN = 100;
  std::size_t size = N * sizeof(float);
  auto h_A = make_unique<float>(size);
  float& d_A;
  cudaMalloc(&d_A, size);
  cudaError_t error = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaEvent_t stop;
  // if (error != cudaSuccess) {
  // }
  inclusive_prefix_sum<<<16 * 16, 16 * 16>>>(
      d_A, d_A + size, 0, [](float a, float b) { return a + b; });

  error = cudaEventSynchronize(&stop);
}
