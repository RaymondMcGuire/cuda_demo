/*
 * @Author: Xu.Wang 
 * @Date: 2020-04-24 00:06:02 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-04-24 01:17:50
 */
#include <cuda_runtime.h>
#include <cuda_add.h>

__global__ void kernel_add(float *a, float* b,int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=index;i<n;i+=stride){
        b[i] = a[i] + b[i];
    }
}

void cudaAdd(float *a, float* b,int n)
{   
    float*x;
    float*y;
    cudaMallocHost((void**)&x, n * sizeof(float));
    cudaMallocHost((void**)&y, n * sizeof(float));
    cudaMemcpy(x, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y, b, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    kernel_add<<<numBlocks,blockSize>>>(x, y, n);
    cudaDeviceSynchronize();

    cudaMemcpy(b, y, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(x);
    cudaFree(y);
}