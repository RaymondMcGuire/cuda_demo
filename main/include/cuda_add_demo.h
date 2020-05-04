/*
 * @Author: Xu.Wang 
 * @Date: 2020-04-24 00:03:00 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-05-03 18:42:39
 */
#include <math.h>
#include <cuda_add.h>

void CudaAddDemo()
{
    printDemoTitle("CUDA Add Demo");

    int n = 1 << 20;
    float *x = new float[n];
    float *y = new float[n];
    for (int i = 0; i < n; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaAdd(x, y, n);

    float maxError = 0.0f;
    for (int i = 0; i < n; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    cout << "Max error: " << maxError << endl;

    printDemoEnd();
    delete[] x;
    delete[] y;
}