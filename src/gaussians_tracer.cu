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

const int triagPerParticle = 20;
const float Tmin = .001;
const float resp_min = 0.01f;

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
        10000.f,//1e16f,                     // Max intersection distance
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
    const uint3 dim = optixGetLaunchDimensions();
    Payload payload = getPayload();
    if(idx.x%800 <= 400)
        payload.radiance = make_float3(1.);
    else 
        payload.radiance = make_float3(0.);
    //setPayload(payload);
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
    const float3 o = optixGetWorldRayOrigin();
    const float3 d = optixGetWorldRayDirection();
    const float3 mu = params.gs_xyz[gs_id];
    const float3 s = params.gs_scaling[gs_id];
    Matrix3x3 Rt = construct_rotation(params.gs_rotation[gs_id]).transpose();
    Rt.setRow(0,Rt.getRow(0)/(s.x+eps));
    Rt.setRow(1,Rt.getRow(1)/(s.y+eps));
    Rt.setRow(2,Rt.getRow(2)/(s.z+eps));
    float3 og = Rt*(mu-o);
    float3 dg = Rt*d;
    float tmax = dot(og,dg)/(dot(dg,dg)+eps);
    float3 samp = o+tmax*d;
    float3 x = Rt*(samp-mu);
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

    //const float3 dir = optixGetWorldRayDirection();//-params.ray_directions[idx.x];
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

extern "C" __global__ void __closesthit__ms() {

    const float2 barycentrics = optixGetTriangleBarycentrics();
    const uint3 idx = optixGetLaunchIndex();

    const unsigned int hitParticle = optixGetPrimitiveIndex()/triagPerParticle;
    float3 radiance = make_float3(0.); 
    const float resp = computeResponse(hitParticle);
    if(resp > resp_min)
    {
        const float3 rad = computeRadiance(hitParticle); 
        radiance = rad;
    }

    Payload payload = getPayload();
    //payload.radiance = radiance;
    //payload.radiance = make_float3(1.-barycentrics.x-barycentrics.y,barycentrics.x,barycentrics.y);
    payload.radiance = make_float3(0.,1.,0.)*resp*payload.transmittance;

    //setPayload(payload);

    //params.radiance[idx.x] = radiance;
    //params.radiance[idx.x] = make_float3(1.-barycentrics.x-barycentrics.y,barycentrics.x,barycentrics.y);
}

extern "C" __global__ void __anyhit__ms() {

    Payload payload = getPayload();

    //if (num_triangles >= params.max_ray_triangles - 1) {
    //    optixTerminateRay();
    //    return;
    //}


    // When built-in triangle intersection is used, a number of fundamental
    // attributes are provided by the OptiX API, indlucing barycentric coordinates.

//#define MESH
#ifndef MESH

    const unsigned int prim_id = optixGetPrimitiveIndex();
    const unsigned int hitParticle = prim_id/triagPerParticle;

    const float3 d = optixGetWorldRayDirection();
    // const float3 s = params.gs_scaling[hitParticle];
    // Matrix3x3 R = construct_rotation(params.gs_rotation[hitParticle]).transpose();
    // R.setRow(0,R.getRow(0)/(s.x+eps));
    // R.setRow(1,R.getRow(1)/(s.y+eps));
    // R.setRow(2,R.getRow(2)/(s.z+eps));
    // const float3 its = R*(optixGetWorldRayOrigin()+optixGetRayTmax()*d);
    // const float3 part_center = R*(its - params.gs_xyz[hitParticle]);
    // if(dot(part_center,d) > 0.){
    //     optixIgnoreIntersection();
    //     return;
    // }
    float3 normal = params.gs_normals[prim_id];
    if(dot(normal,d)>0.)
    {
        optixIgnoreIntersection();
        return;
    }

    const float resp = computeResponse(hitParticle);
    if(resp > resp_min){
        const float3 rad = computeRadiance(hitParticle); 
        payload.radiance += rad*resp*payload.transmittance;
        payload.transmittance *= (1.-resp);
        //payload.transmittance += resp;
    }

    setPayload(payload);

    if(payload.transmittance > Tmin)
    {
        optixIgnoreIntersection();
    }
    else{
        optixTerminateRay();
    }
#else

    const float2 barycentrics = optixGetTriangleBarycentrics();

    const float3 d = optixGetWorldRayDirection();
    
    const unsigned int prim_id = optixGetPrimitiveIndex();
    float3 normal = params.gs_normals[prim_id];

    if(dot(normal,d)<0.)
    {
        payload.radiance = make_float3(1.-barycentrics.x-barycentrics.y,barycentrics.x,barycentrics.y);
    }
    setPayload(payload);
    //optixIgnoreIntersection();

    //const uint3 idx = optixGetLaunchIndex();
    //params.radiance[idx.x] = make_float3(1.-barycentrics.x-barycentrics.y,barycentrics.x,barycentrics.y);
#endif
}