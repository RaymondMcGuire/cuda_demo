/*
 * @Author: Xu.Wang 
 * @Date: 2020-04-24 00:00:48 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-05-03 18:42:52
 */

#include <math.h>
#include <cuda_double.h>

void CudaDoubleDemo()
{
    printDemoTitle("CUDA Double Demo");
    const int n = 1 << 20;
    cout << "n: " << n << endl;
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
        maxError = fmax(maxError, fabs(float(out[i] - answer[i])));
    }
    cout << "Max error: " << maxError << endl;
    printDemoEnd();
    delete[] in;
    delete[] out;
    delete[] answer;
}