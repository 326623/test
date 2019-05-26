#include <stdio.h>
__device__ float* hello;

__global__ void TryHello(float* hello) {
  // float hi = *hello;
  // printf("%f\n", hi);
  printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

__global__ void setHello(float* hello, float num) {
  *hello = num;
}

// int main() {
//   TryHello<<<1, 5>>>(&hello);
//   cudaDeviceSynchronize();
//   setHello<<<1, 5>>>(&hello, 99.0f);
//   cudaDeviceSynchronize();
//   TryHello<<<1, 5>>>(&hello);
//   cudaDeviceSynchronize();
// }

__global__ void helloCUDA(float f)
{
    printf("Hello thread %d, f=%f\n", threadIdx.x, f);
}

int main()
{
    helloCUDA<<<1, 5>>>(1.2345f);
    cudaDeviceSynchronize();
    // setHello<<<1, 5>>>(hello, 99.0f);
    TryHello<<<1, 5>>>(hello);
    cudaDeviceSynchronize();
    return 0;
}