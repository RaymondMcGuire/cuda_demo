/*
 * @Author: Xu.Wang
 * @Date: 2020-05-03 17:47:56
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-05-04 01:25:33
 */

#ifndef _CUDA_THRUST_H_
#define _CUDA_THRUST_H_

#include <cuda_common.h>

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/version.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstdlib>
#include <list>
#include <vector>

struct CalcSqrtDim2
    : public thrust::binary_function<const double, const double, double> {
  __host__ __device__ CalcSqrtDim2() {}

  __device__ double operator()(const double &x, const double &y) const {
    return sqrt(x * x + y * y);
  }
};

struct calcGridHash {
  float3 lowerCorner;
  float3 cellWidth;
  int3 gridSize;
  __host__ __device__ calcGridHash(const float3 &_lowerCorner,
                                   const float3 &_cellWidth, int3 &_gridSize)
      : lowerCorner(_lowerCorner), cellWidth(_cellWidth), gridSize(_gridSize) {}

  __device__ int operator()(float3 pos) {
    float3 diff = pos - lowerCorner;
    int x = min(max(int(diff.x / cellWidth.x), 0), gridSize.x - 1),
        y = min(max(int(diff.y / cellWidth.y), 0), gridSize.y - 1),
        z = min(max(int(diff.z / cellWidth.z), 0), gridSize.z - 1);
    return (int)(x * gridSize.y * gridSize.z + y * gridSize.z + z);
  }
};

void ceilDivTest(int totalNum, int block_size);

void thrustVersion();
void thrustHostDeviceVector();
void thrustCopyFill();
void thrustInitByStl();

void thrustTransform();
void thrustTransformDevicePtr();
void thrustSortByKey();

#endif /* _CUDA_THRUST_H_ */