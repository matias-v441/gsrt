#include "primitives.h"
#include <math.h>
#include <unistd.h>
#include <cstdint>
#include "utils/vec_math.h"
#include "utils/auxiliary.h"
#include "assert.h"
#include <iostream>
#include "utils/geom.h"

namespace primitive = util::geom::icosahedron_3dgrut;

__global__ void _construct_icosahedra(const int numgs, const float3* xyz, const float* opacity,
     const float3* scaling, const float4* rotation,
     float3* vertices, uint3* triangles){

    int i = blockDim.x*blockIdx.x+threadIdx.x;
    if(i < numgs){
        using primitive::n_verts;
        using primitive::n_faces;
        const float3* prim_verts = primitive::vertices().data;
        const uint3* prim_triag = primitive::triangles().data;

        const util::Matrix3x3 R = util::geom::construct_rotation(rotation[i]);

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

namespace util::geom::cuda {

    void alloc_buffers(const int n, float3** vertices, uint32_t& nvert,
        uint3** triangles, uint32_t& ntriag){
        nvert = primitive::n_verts*n;
        ntriag = primitive::n_faces*n;
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(vertices),nvert*sizeof(float3)),true);
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(triangles),ntriag*sizeof(uint3)),true);
    }

    void construct_icosahedra(const int numgs, const float3* xyz, const float* opacity,
        const float3* scaling, const float4* rotation,
        float3* vertices, uint3* triangles){
        uint32_t threads = 1024;
        uint32_t blocks = (numgs+threads-1)/threads;
        _construct_icosahedra<<< blocks,threads >>>(numgs,xyz,opacity,scaling,rotation,vertices,triangles);
        CHECK_CUDA(,true);
    }

}