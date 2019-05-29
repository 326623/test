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
#ifndef _TEST_REDUCTION_CUH_
#define _TEST_REDUCTION_CUH_
#include "cuda_macro.h"

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
  int workload, start, end;
  workload = n / stride;
  start = workload * id;
  // last one takes up more work
  if (id == stride - 1)
    end = n;
  else
    end = workload * (id + 1);

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

#endif /* _TEST_REDUCTION_CUH_ */
