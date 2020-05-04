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

#include <algorithm>
#include <cstdlib>
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>

#include <list>
#include <vector>

void thrustVersion();
void thrustHostDeviceVector();
void thrustCopyFill();
void thrustInitByStl();
void ceilDivTest(int totalNum, int block_size);

#endif /* _CUDA_THRUST_H_ */