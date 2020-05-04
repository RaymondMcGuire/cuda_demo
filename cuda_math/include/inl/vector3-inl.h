/*
 * @Author: Xu.Wang 
 * @Date: 2020-04-25 01:43:33 
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-04-25 02:56:38
 */
#ifndef _CUDA_MATH_VECTOR3_INL_H_
#define _CUDA_MATH_VECTOR3_INL_H_

namespace cuda_math
{
    __host__ __device__ void Vector3::make_unit_vector()
    {
        float k = 1.0f / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
        e[0] *= k;
        e[1] *= k;
        e[2] *= k;
    }

    __host__ __device__ Vector3 operator+(const Vector3 &v1, const Vector3 &v2)
    {
        return Vector3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
    }

    __host__ __device__ Vector3 operator-(const Vector3 &v1, const Vector3 &v2)
    {
        return Vector3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
    }

    __host__ __device__ Vector3 operator*(const Vector3 &v1, const Vector3 &v2)
    {
        return Vector3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
    }

    __host__ __device__ Vector3 operator/(const Vector3 &v1, const Vector3 &v2)
    {
        return Vector3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
    }

    __host__ __device__ Vector3 operator*(float t, const Vector3 &v)
    {
        return Vector3(t * v.e[0], t * v.e[1], t * v.e[2]);
    }

    __host__ __device__ Vector3 operator/(Vector3 v, float t)
    {
        return Vector3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
    }

    __host__ __device__ Vector3 operator*(const Vector3 &v, float t)
    {
        return Vector3(t * v.e[0], t * v.e[1], t * v.e[2]);
    }

    __host__ __device__ float dot(const Vector3 &v1, const Vector3 &v2)
    {
        return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
    }

    __host__ __device__ Vector3 cross(const Vector3 &v1, const Vector3 &v2)
    {
        return Vector3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
                       (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
                       (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
    }

    __host__ __device__ Vector3 &Vector3::operator+=(const Vector3 &v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ Vector3 &Vector3::operator*=(const Vector3 &v)
    {
        e[0] *= v.e[0];
        e[1] *= v.e[1];
        e[2] *= v.e[2];
        return *this;
    }

    __host__ __device__ Vector3 &Vector3::operator/=(const Vector3 &v)
    {
        e[0] /= v.e[0];
        e[1] /= v.e[1];
        e[2] /= v.e[2];
        return *this;
    }

    __host__ __device__ Vector3 &Vector3::operator-=(const Vector3 &v)
    {
        e[0] -= v.e[0];
        e[1] -= v.e[1];
        e[2] -= v.e[2];
        return *this;
    }

    __host__ __device__ Vector3 &Vector3::operator*=(const float t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ Vector3 &Vector3::operator/=(const float t)
    {
        float k = 1.0f / t;

        e[0] *= k;
        e[1] *= k;
        e[2] *= k;
        return *this;
    }

    __host__ __device__ Vector3 unit_vector(Vector3 v)
    {
        return v / v.length();
    }

} // namespace cuda_math

#endif