/*
 * @Author: Xu.Wang 
 * @Date: 2020-04-23 20:59:32 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-04-24 01:20:21
 * @Ref:https://devblogs.nvidia.com/even-easier-introduction-cuda/
 */
#include <cuda_add_demo.h>
#include <cuda_double_demo.h>

int main(void)
{
    CudaDoubleDemo();

    CudaAddDemo();
    return 0;
}