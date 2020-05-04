/*
 * @Author: Xu.Wang
 * @Date: 2020-05-03 18:03:29
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-05-03 18:29:35
 */

#ifndef _CUDA_COMMON_H_
#define _CUDA_COMMON_H_

#include <cuda_runtime.h>
#include <iostream>

using namespace std;

// print demo
void printDemoTitle(string demo_name);
void printDemoEnd();

// check error
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func,
                const char *const file, int const line);

// compute grid size
__host__ __device__ inline int ceilDiv(int n, int b) {
  return (int)((n + b - 1) / b);
}

#endif /* _CUDA_COMMON_H_ */