#include <optix.h>

#include "optix_types.h"
#include "utils/vec_math.h"

extern "C" {
__constant__ Params params;
constexpr float eps = 1e-6;
}

extern "C" __global__ void __raygen__rg() {
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    const float3 ray_origin = params.ray_origins[idx.x];
    const float3 ray_direction = params.ray_directions[idx.x];

    // Trace the ray against our scene hierarchy
    unsigned int p0 = 0;
    optixTrace(
        params.handle,
        ray_origin,
        ray_direction,
        0.0f,                      // Min intersection distance
        1e16f,                     // Max intersection distance
        0.0f,                      // rayTime -- used for motion blur
        OptixVisibilityMask(255),  // Specify always visible
        OPTIX_RAY_FLAG_NONE,
        0,  // SBT offset   -- See SBT discussion
        1,  // SBT stride   -- See SBT discussion
        0,  // missSBTIndex -- See SBT discussion
        p0);
}

extern "C" __global__ void __miss__ms() {
}

extern "C" __global__ void __closesthit__ms() {
}

extern "C" __global__ void __anyhit__ms() {
    const unsigned int num_triangles = optixGetPayload_0();
    //if (num_triangles >= params.max_ray_triangles - 1) {
    //    optixTerminateRay();
    //    return;
    //}

    // When built-in triangle intersection is used, a number of fundamental
    // attributes are provided by the OptiX API, indlucing barycentric coordinates.
    // const float2 barycentrics = optixGetTriangleBarycentrics();
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const unsigned int current_triangle = optixGetPrimitiveIndex();
    //params.hit_distances[idx.x * params.max_ray_triangles + num_triangles] = optixGetRayTmax();

    // setPayload(make_float3(barycentrics, 1.0f));
    optixSetPayload_0(num_triangles + 1);
    optixIgnoreIntersection();
}