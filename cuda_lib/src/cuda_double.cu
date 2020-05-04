/*
 * @Author: Xu.Wang
 * @Date: 2020-04-24 00:06:07
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-05-03 18:22:24
 */
#include <cuda_double.h>

__global__ void kernel_double(int *in, int *out, const int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    out[i] = in[i] * 2;
  }
}

void cudaDouble(int *hIn, int *hOut, const int n) {
  int *dIn;
  int *dOut;
  cudaMallocHost((void **)&dIn, n * sizeof(int));
  cudaMallocHost((void **)&dOut, n * sizeof(int));
  cudaMemcpy(dIn, hIn, n * sizeof(int), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int numBlocks = ceilDiv(n, blockSize);
  kernel_double<<<numBlocks, blockSize>>>(dIn, dOut, n);
  cudaDeviceSynchronize();

  cudaMemcpy(hOut, dOut, n * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(dIn);
  cudaFree(dOut);
}