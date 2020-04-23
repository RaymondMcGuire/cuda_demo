/*
 * @Author: Xu.Wang 
 * @Date: 2020-04-24 00:06:07 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-04-24 01:14:48
 */
#include <cuda_runtime.h>
#include <cuda_double.h>
#include<iostream>

__global__ void kernel_double(int *in, int *out, const int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=index;i<n;i+=stride){
        out[i] = in[i]*2;
    }
}

void cudaDouble(int *hIn, int *hOut,const int n)
{
    int *dIn;
    int *dOut;
    cudaMallocHost((void**)&dIn, n * sizeof(int));
    cudaMallocHost((void**)&dOut, n * sizeof(int));
    cudaMemcpy(dIn, hIn, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    kernel_double<<<numBlocks,blockSize>>>(dIn, dOut, n);
    cudaDeviceSynchronize();

    cudaMemcpy(hOut, dOut, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dIn);
    cudaFree(dOut);
}