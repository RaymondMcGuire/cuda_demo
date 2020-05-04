/*
 * @Author: Xu.Wang
 * @Date: 2020-04-23 20:59:32
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-05-03 18:31:47
 * @Ref:https://devblogs.nvidia.com/even-easier-introduction-cuda/
 */
#include <cuda_add_demo.h>
#include <cuda_double_demo.h>
#include <cuda_thrust.h>
#include <cuda_math.h>

enum DEMO_TYPES
{
    CUDA_NO_LIB = 0,
    THRUST_LIB = 1,
    CUDA_MATH_LIB = 2
};

int main(void)
{
    DEMO_TYPES type = CUDA_MATH_LIB;
    switch (type)
    {
    case CUDA_NO_LIB:
        CudaDoubleDemo();

        CudaAddDemo();
        break;
    case THRUST_LIB:

        ceilDivTest(10000, 128);

        thrustVersion();

        thrustHostDeviceVector();

        thrustCopyFill();

        thrustInitByStl();
        break;
    case CUDA_MATH_LIB:
        mathVec3();
        break;
    default:
        break;
    }
    return 0;
}