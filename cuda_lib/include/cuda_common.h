/*
 * @Author: Xu.Wang
 * @Date: 2020-05-03 18:03:29
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-05-03 18:29:35
 */

#ifndef _CUDA_COMMON_H_
#define _CUDA_COMMON_H_

#include <iostream>

// cuda library
#include <vector_functions.h>
#include <vector_types.h>

#include <cuda_helper/helper_math.h>
#include <cuda_helper/helper_cuda.h>

using namespace std;

// print demo
void printDemoTitle(string demo_name);
void printDemoEnd();

// compute grid size
__host__ __device__ inline int ceilDiv(int n, int b)
{
    return (int)((n + b - 1) / b);
}

#endif /* _CUDA_COMMON_H_ */