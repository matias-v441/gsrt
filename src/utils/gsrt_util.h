#pragma once
#include "vec_math.h"
#include "Matrix.h"
using namespace util;

namespace gsrt_util
{

HOSTDEVICE inline 
Matrix3x3 construct_rotation(float4 vec){
    float4 q = normalize(vec);
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;
    Matrix3x3 R({
        1.f - 2.f * (y*y + z*z), 2.f * (x*y - r*z),       2.f * (x*z + r*y),
        2.f * (x*y + r*z),       1.f - 2.f * (x*x + z*z), 2.f * (y*z - r*x),
        2.f * (x*z - r*y),       2.f * (y*z + r*x), 1.f   -2.f * (x*x + y*y)});
    return R;
}

namespace icosahedron{
    const float x = sqrt(.4f*(5.f+sqrt(5.f))) *.5f;
    const float y = x*(1.f+sqrt(5.f))*.5f;
    const int n_verts = 12;
    const int n_faces = 20;
    inline float3 vertices[n_verts] = {
        make_float3(-x,y,0.f),make_float3(x,y,0.f),make_float3(-x,-y,0.f),make_float3(x,-y,0.f),
        make_float3(0.f,-x,y),make_float3(0.f,x,y),make_float3(0.f,-x,-y),make_float3(0.f,x,-y),
        make_float3(y,0.f,-x),make_float3(y,0.f,x),make_float3(-y,0.f,-x),make_float3(-y,0.f,x),
        };
    inline uint3 triangles[n_faces] = {
        make_uint3(0,11,5), make_uint3(0,5,1), make_uint3(0,1,7), make_uint3(0,7,10), make_uint3(0,10,11),
        make_uint3(1,5,9), make_uint3(5,11,4), make_uint3(11,10,2), make_uint3(10,7,6), make_uint3(7,1,8),
        make_uint3(3,9,4), make_uint3(3,4,2), make_uint3(3,2,6), make_uint3(3,6,8), make_uint3(3,8,9),
        make_uint3(4,9,5), make_uint3(2,4,11), make_uint3(6,2,10), make_uint3(8,6,7), make_uint3(9,8,1),
    };
}

}