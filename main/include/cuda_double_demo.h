/*
 * @Author: Xu.Wang 
 * @Date: 2020-04-24 00:00:48 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-04-24 01:20:15
 */

#include <iostream>
#include <math.h>
#include <cuda_double.h>

void CudaDoubleDemo()
{
    std::cout << "----------------------" << std::endl;
    std::cout << "CUDA Double Demo " << std::endl;
    const int n = 1 << 20;
    std::cout << "n: " << n << std::endl;
    int *in = new int[n];
    int *out = new int[n];
    int *answer = new int[n];

    for (int i = 0; i < n; i++)
        in[i] = rand() % 100;
    for (int i = 0; i < n; i++)
        answer[i] = in[i] * 2;

    cudaDouble(in, out, n);

    float maxError = 0.0f;
    for (int i = 0; i < n; i++)
    {
        maxError = fmax(maxError, fabs(out[i] - answer[i]));
    }
    std::cout << "Max error: " << maxError << std::endl;
    delete[] in;
    delete[] out;
    delete[] answer;
}