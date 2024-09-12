#include <optix.h>

#include "optix_types.h"
#include "utils/vec_math.h"

#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <vector>

extern "C" {
__constant__ Params params;
constexpr float eps = 1e-6;
}

struct Payload{
    float3 radiance;
    float transmittance;
};

//__device__ __forceinline__ Payload getPayload(){
//    unsigned int p[4];
//    p[0] = optixGetPayload_0();
//    p[1] = optixGetPayload_1();
//    p[2] = optixGetPayload_2();
//    p[3] = optixGetPayload_3();
//    return *reinterpret_cast<Payload*>(p);
//}
//
//__device__ __forceinline__ void setPayload(Payload payload){
//    unsigned int* p = reinterpret_cast<unsigned int*>(&payload);
//    optixSetPayload_0(p[0]);
//    optixSetPayload_1(p[1]);
//    optixSetPayload_2(p[2]);
//    optixSetPayload_3(p[3]);
//}

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
    //__float_as_uint(payload.radiance.x);
    //__float_as_uint(payload.radiance.y);
    //__float_as_uint(payload.radiance.z);
    //__float_as_uint(payload.transmittance);
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

__device__ glm::mat3 construct_rotation(float4 vec){
    glm::vec4 q = glm::normalize(glm::vec4(vec.x,vec.y,vec.z,vec.w));
    glm::mat3 R(0.0f);
    float r = q[0];
    float x = q[1];
    float y = q[2];
    float z = q[3];
    R[0][0] = 1. - 2. * (y*y + z*z);
    R[1][0] = 2. * (x*y - r*z);
    R[2][0] = 2. * (x*z + r*y);
    R[0][1] = 2. * (x*y + r*z);
    R[1][1] = 1. - 2. * (x*x + z*z);
    R[2][1] = 2. * (y*z - r*x);
    R[0][2] = 2. * (x*z - r*y);
    R[1][2] = 2. * (y*z + r*x);
    R[2][2] = 1. - 2. * (x*x + y*y);
    return R;
}

__device__ float computeResponse(unsigned int gs_id){
    const uint3 idx = optixGetLaunchIndex();
    const float3 ray_origin = params.ray_origins[idx.x];
    const float3 ray_direction = params.ray_directions[idx.x];
    const float3 xyz = params.gs_xyz[gs_id];
    const glm::vec3 o(ray_origin.x,ray_origin.y,ray_origin.z);
    const glm::vec3 d(ray_direction.x,ray_direction.y,ray_direction.z);
    const glm::vec3 mu(xyz.x,xyz.y,xyz.z);
    //glm::mat3 R = construct_rotation(params.gs_rotation[gs_id]);
    //R[0] /= params.gs_scaling[gs_id].x + eps;
    //R[1] /= params.gs_scaling[gs_id].y + eps;
    //R[2] /= params.gs_scaling[gs_id].z + eps;
    //R = glm::transpose(R);
    //glm::vec3 og = R*(mu-o);
    //glm::vec3 dg = R*d;
    //float tmax = og*og;
    glm::vec3 g(1.);
    //float k = g*g;
    //printf("%f ", k);
    //float tmax = (og*dg)/(dg*dg+eps);
    //glm::vec3 samp = o+tmax*d;
    //glm::vec3 x = R*(samp-mu);
    //float tmax = glm(og,dg)/(glm::dot(dg,dg)+eps);
    //glm::vec3 samp = o+tmax*d;
    //glm::vec3 x = R*(samp-mu);


    //printf("| %f %f %f |",x.x,x.y,x.z);
    //printf("%f ",glm::dot(x,x));
    //float resp = params.gs_opacity[gs_id]*exp(-x*x);
    //float resp = glm::dot(x,x);
    //printf("%f",tmax);
    //printf("| %d |",CUDA_VERSION);
    //glm::vec3 f = g*g;
    float k = glm::dot(g,g);
    //float l = g.x*g.x+g.y*g.y+g.z*g.z;
    //printf("|%f, %f, %f|",f.x, k, l);
    printf("%f",k);
    return 0.;
}

__device__ float3 computeRadiance(unsigned int gs_id){
    return make_float3(1.,1.,1.);
}

extern "C" __global__ void __anyhit__ms() {

    Payload payload = getPayload();

    //unsigned int p[4];
    //p[0] = optixGetPayload_0();
    //p[1] = optixGetPayload_1();
    //p[2] = optixGetPayload_2();
    //p[3] = optixGetPayload_3();

    //unsigned int p[2];
    //p[0] = optixGetPayload_0();
    //p[1] = optixGetPayload_1();
    //Payload *payload = reinterpret_cast<Payload *>(p);

    //float transm = __uint_as_float(optixGetPayload_0());
    //float transm_updated = transm+1;
    
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
    //const uint3 dim = optixGetLaunchDimensions();

    //float* T = reinterpret_cast<float*>(&p);
    //float T = payload->transmittance;
    //float3 radiance = payload->radiance;
    const unsigned int hitParticle = optixGetPrimitiveIndex()/triagPerParticle;
    const float resp = computeResponse(hitParticle);
    if(resp > resp_min){
        const float3 rad = computeRadiance(hitParticle); 
        payload.radiance += payload.transmittance*resp*rad;
        payload.transmittance *= (1-resp);
        //transm *= (1-resp);
        //transm_updated = transm * (1 - resp);
        // printf("%d",p);
        //*T *= (1-resp);
    }
    //printf("%f",resp);
    //payload->radiance = make_float3(1.-barycentrics.x-barycentrics.y, barycentrics);//radiance;
    //payload->transmittance = T;
    //optixSetPayload_0(__float_as_uint(transm_updated));

    //params.hit_distances[idx.x * params.max_ray_triangles + num_triangles] = optixGetRayTmax();
    //params.radiance[idx.x] = make_float3(1.-barycentrics.x-barycentrics.y, barycentrics);
    //payload.radiance = make_float3(1.-barycentrics.x-barycentrics.y, barycentrics);
    // setPayload(make_float3(barycentrics, 1.0f));

    //float rx = radiance.x;
    //float ry = radiance.y;
    //float rz = radiance.z;
    //float tr = T;
    //unsigned int prx = 1;//*reinterpret_cast<unsigned int*>(&rx);
    //unsigned int pry = 1;//*reinterpret_cast<unsigned int*>(&rx);
    //unsigned int prz = 1;//*reinterpret_cast<unsigned int*>(&rx);
    //unsigned int ptr = 1;//*reinterpret_cast<unsigned int*>(&tr);
    //optixSetPayload_0(prx);
    //optixSetPayload_1(pry);
    //optixSetPayload_2(prz);
    //optixSetPayload_3(ptr);

    setPayload(payload);

    if(payload.transmittance > Tmin){
        optixIgnoreIntersection();
    }
}