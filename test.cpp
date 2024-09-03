#include "gaussians_tracer.h"
#include <iostream>
#include <cuda_runtime.h>
#include "cnpy.h"
#include "utils/vec_math.h"

int main(int argc, char** argv){

    int device;
    cudaGetDevice(&device);
    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    std::cout << props.name << std::endl;

    GaussiansTracer tracer(device);

    cnpy::npz_t gs_npz = cnpy::npz_load("data/drums.npz");
    GaussiansData data;
    data.numgs = gs_npz["xyz"].shape[0];
    data.xyz = gs_npz["xyz"].data<float3>();
    data.rotation = gs_npz["rotation"].data<float4>();
    data.scaling = gs_npz["scaling"].data<float3>();
    data.opacity = gs_npz["opacity"].data<float>();
    tracer.load_gaussians(data);

    return 0;
}