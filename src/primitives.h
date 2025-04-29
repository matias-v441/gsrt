#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils/Matrix.h"
using namespace util;

void alloc_buffers(const int n, float3** vertices, uint32_t& nvert, uint3** triangles, uint32_t& ntriag);

void construct_primitives(const int numgs, const float3* xyz, const float* opacity, const float3* scaling, const float4* rotation,
                        float3* vertices, uint3* triangles);