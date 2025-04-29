#include "primitives.h"
#include <math.h>
#include <unistd.h>
#include <cstdint>
#include "utils/vec_math.h"
#include "utils/auxiliary.h"
#include "assert.h"
#include <iostream>

namespace icosahedron{
    constexpr int n_verts = 12;
    constexpr int n_faces = 20;
    struct verts{float3 data[n_verts];};
    struct triag{uint3 data[n_faces];};
    __device__ const verts vertices(){
        const float x = sqrt(.4f*(5.f+sqrt(5.f))) *.5f;
        const float y = x*(1.f+sqrt(5.f))*.5f;
        return {
        make_float3(-x,y,0.f),make_float3(x,y,0.f),make_float3(-x,-y,0.f),make_float3(x,-y,0.f),
        make_float3(0.f,-x,y),make_float3(0.f,x,y),make_float3(0.f,-x,-y),make_float3(0.f,x,-y),
        make_float3(y,0.f,-x),make_float3(y,0.f,x),make_float3(-y,0.f,-x),make_float3(-y,0.f,x),
        };}
    __device__ const triag triangles(){return{
        make_uint3(0,11,5), make_uint3(0,5,1), make_uint3(0,1,7), make_uint3(0,7,10), make_uint3(0,10,11),
        make_uint3(1,5,9), make_uint3(5,11,4), make_uint3(11,10,2), make_uint3(10,7,6), make_uint3(7,1,8),
        make_uint3(3,9,4), make_uint3(3,4,2), make_uint3(3,2,6), make_uint3(3,6,8), make_uint3(3,8,9),
        make_uint3(4,9,5), make_uint3(2,4,11), make_uint3(6,2,10), make_uint3(8,6,7), make_uint3(9,8,1),
    };}
}

namespace icosahedron_grut{
    constexpr int n_verts = 12;
    constexpr int n_faces = 20;
    constexpr float goldenRatio = 1.618033988749895;
    constexpr float icosaEdge     = 1.323169076499215;
    constexpr float icosaVrtScale = 0.5 * icosaEdge;
    struct verts{float3 data[n_verts];};
    struct triag{uint3 data[n_faces];};
    __device__ static const verts vertices(){return {
        make_float3(-1, goldenRatio, 0)*icosaVrtScale, make_float3(1, goldenRatio, 0)*icosaVrtScale, make_float3(0, 1, -goldenRatio)*icosaVrtScale,
        make_float3(-goldenRatio, 0, -1)*icosaVrtScale, make_float3(-goldenRatio, 0, 1)*icosaVrtScale, make_float3(0, 1, goldenRatio)*icosaVrtScale,
        make_float3(goldenRatio, 0, 1)*icosaVrtScale, make_float3(0, -1, goldenRatio)*icosaVrtScale, make_float3(-1, -goldenRatio, 0)*icosaVrtScale,
        make_float3(0, -1, -goldenRatio)*icosaVrtScale, make_float3(goldenRatio, 0, -1)*icosaVrtScale, make_float3(1, -goldenRatio, 0)*icosaVrtScale
        };}
    __device__ static const triag triangles(){return{
        make_uint3(0, 1, 2), make_uint3(0, 2, 3), make_uint3(0, 3, 4), make_uint3(0, 4, 5), make_uint3(0, 5, 1),
        make_uint3(6, 1, 5), make_uint3(6, 5, 7), make_uint3(6, 7, 11), make_uint3(6, 11, 10), make_uint3(6, 10, 1),
        make_uint3(8, 4, 3), make_uint3(8, 3, 9), make_uint3(8, 9, 11), make_uint3(8, 11, 7), make_uint3(8, 7, 4),
        make_uint3(9, 3, 2), make_uint3(9, 2, 10), make_uint3(9, 10, 11),
        make_uint3(5, 4, 7), make_uint3(1, 10, 2)
    };}
}

namespace primitive = icosahedron_grut;


void alloc_buffers(const int n, float3** vertices, uint32_t& nvert,
     uint3** triangles, uint32_t& ntriag){
    nvert = primitive::n_verts*n;
    ntriag = primitive::n_faces*n;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(vertices),nvert*sizeof(float3)),true);
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(triangles),ntriag*sizeof(uint3)),true);
}

__device__ Matrix3x3 _construct_rotation(float4 vec){
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


__global__ void _construct_primitives(const int numgs, const float3* xyz, const float* opacity,
     const float3* scaling, const float4* rotation,
     float3* vertices, uint3* triangles){

    int i = blockDim.x*blockIdx.x+threadIdx.x;
    if(i < numgs){
        using primitive::n_verts;
        using primitive::n_faces;
        const float3* prim_verts = primitive::vertices().data;
        const uint3* prim_triag = primitive::triangles().data;

        const Matrix3x3 R = _construct_rotation(rotation[i]);

        constexpr float alpha_min = 0.0113f;
        float adaptive_scale = sqrtf(-2.*logf(fminf(alpha_min / opacity[i], 0.97f)));
        //float adaptive_scale = sqrtf(2.*logf(opacity[i]/alpha_min));
        float3 s = scaling[i]*adaptive_scale;

        for(int j = 0; j < n_verts; ++j){
            float3 v = prim_verts[j];
            float3 w = R*(v*s)+xyz[i];
            vertices[j+i*n_verts] = w; 
        }

        for(int j = 0; j < n_faces; ++j){
            uint3 triag = prim_triag[j];
            triag += make_uint3(i*n_verts);
            triangles[j+i*n_faces] = triag;
        }
    }
}


void construct_primitives(const int numgs, const float3* xyz, const float* opacity,
    const float3* scaling, const float4* rotation,
    float3* vertices, uint3* triangles){
    uint32_t threads = 1024;
    uint32_t blocks = (numgs+threads-1)/threads;
    _construct_primitives<<< blocks,threads >>>(numgs,xyz,opacity,scaling,rotation,vertices,triangles);
    CHECK_CUDA(,true);
}
