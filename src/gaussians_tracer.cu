#include <optix.h>

#include "optix_types.h"
#include "utils/Matrix.h"

#include <vector>

using namespace util;

extern "C" {
__constant__ Params params;
constexpr float eps = 1e-6;
}

struct Payload{
    float3 radiance;
    float transmittance;
};

__device__ __forceinline__ Payload getPayload(){
    Payload payload;
    payload.radiance.x = __uint_as_float(optixGetPayload_0());
    payload.radiance.y = __uint_as_float(optixGetPayload_1());
    payload.radiance.z = __uint_as_float(optixGetPayload_2());
    payload.transmittance = __uint_as_float(optixGetPayload_3());
    return payload;
}

__device__ __forceinline__ void setPayload(const Payload& payload){
    optixSetPayload_0(__float_as_uint(payload.radiance.x));
    optixSetPayload_1(__float_as_uint(payload.radiance.y));
    optixSetPayload_2(__float_as_uint(payload.radiance.z));
    optixSetPayload_3(__float_as_uint(payload.transmittance));
}

extern "C" __global__ void __raygen__rg() {

    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    const float3 ray_origin = params.ray_origins[idx.x];
    const float3 ray_direction = params.ray_directions[idx.x];

    Payload payload{};
    payload.radiance = make_float3(0.f);
    payload.transmittance = 1.f;

    unsigned int* p = reinterpret_cast<unsigned int *>(&payload);

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
        p[0],p[1],p[2],p[3]);
    params.radiance[idx.x] = payload.radiance;
    params.transmittance[idx.x] = payload.transmittance;
}

extern "C" __global__ void __miss__ms() {
    const uint3 idx = optixGetLaunchIndex();
    //params.radiance[idx.x] = make_float3(0.,0.,1.);
}

extern "C" __global__ void __closesthit__ms() {
}

__device__ Matrix3x3 construct_rotation(float4 vec){
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

__device__ float computeResponse(unsigned int gs_id){
    const uint3 idx = optixGetLaunchIndex();
    const float3 o = params.ray_origins[idx.x];
    const float3 d = params.ray_directions[idx.x];
    const float3 mu = params.gs_xyz[gs_id];
    const float3 s = params.gs_scaling[gs_id];
    Matrix3x3 R = construct_rotation(params.gs_rotation[gs_id]).transpose();
    R.setRow(0,R.getRow(0)/(s.x+eps));
    R.setRow(1,R.getRow(1)/(s.y+eps));
    R.setRow(2,R.getRow(2)/(s.z+eps));
    float3 og = R*(mu-o);
    float3 dg = R*d;
    float tmax = dot(og,dg)/(dot(dg,dg)+eps);
    float3 samp = o+tmax*d;
    float3 x = R*(samp-mu);
    float resp = params.gs_opacity[gs_id]*exp(-dot(x,x));
    return resp;
}

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__device__ float3 computeRadiance(unsigned int gs_id){
    const uint3 idx = optixGetLaunchIndex();

    //const float3 dir = -params.ray_directions[idx.x];
    const float3 mu = params.gs_xyz[gs_id];
    const float3 o = params.ray_origins[idx.x];
    const float3 dir = normalize(mu-o);

    const float3* sh = params.gs_sh + gs_id*16;
    const int deg = params.sh_deg;

	float3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += make_float3(.5f);

    return result;
}

extern "C" __global__ void __anyhit__ms() {

    Payload payload = getPayload();

    //if (num_triangles >= params.max_ray_triangles - 1) {
    //    optixTerminateRay();
    //    return;
    //}
    const int triagPerParticle = 20;
    const float Tmin = 0.001;
    const float resp_min = 0.01f;

    // When built-in triangle intersection is used, a number of fundamental
    // attributes are provided by the OptiX API, indlucing barycentric coordinates.
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const uint3 idx = optixGetLaunchIndex();

    const unsigned int hitParticle = optixGetPrimitiveIndex()/triagPerParticle;
    const float resp = computeResponse(hitParticle);
    if(resp > resp_min){
        const float3 rad = computeRadiance(hitParticle); 
        payload.radiance += payload.transmittance*resp*rad;
        payload.transmittance *= (1-resp);
    }

    setPayload(payload);

    if(payload.transmittance > Tmin){
        optixIgnoreIntersection();
    }
}