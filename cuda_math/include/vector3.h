/*
 * @Author: Xu.Wang 
 * @Date: 2020-04-25 01:38:31 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-04-25 02:56:35
 */
#pragma once
#ifndef _CUDA_MATH_VECTOR3_H_
#define _CUDA_MATH_VECTOR3_H_
#include <math.h>

namespace cuda_math
{
    class Vector3
    {

    public:
        __host__ __device__ Vector3() {}
        __host__ __device__ Vector3(float e0, float e1, float e2)
        {
            e[0] = e0;
            e[1] = e1;
            e[2] = e2;
        }
        __host__ __device__ float x() const { return e[0]; }
        __host__ __device__ float y() const { return e[1]; }
        __host__ __device__ float z() const { return e[2]; }
        __host__ __device__ float r() const { return e[0]; }
        __host__ __device__ float g() const { return e[1]; }
        __host__ __device__ float b() const { return e[2]; }

        __host__ __device__ const Vector3 &operator+() const { return *this; }
        __host__ __device__ Vector3 operator-() const { return Vector3(-e[0], -e[1], -e[2]); }
        __host__ __device__ float operator[](int i) const { return e[i]; }
        __host__ __device__ float &operator[](int i) { return e[i]; };

        __host__ __device__ Vector3 &operator+=(const Vector3 &v2);
        __host__ __device__ Vector3 &operator-=(const Vector3 &v2);
        __host__ __device__ Vector3 &operator*=(const Vector3 &v2);
        __host__ __device__ Vector3 &operator/=(const Vector3 &v2);
        __host__ __device__ Vector3 &operator*=(const float t);
        __host__ __device__ Vector3 &operator/=(const float t);

        __host__ __device__ float length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
        __host__ __device__ float squared_length() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
        __host__ __device__ void make_unit_vector();

        float e[3];
    };
} // namespace cuda_math

#include "inl/vector3-inl.h"

#endif