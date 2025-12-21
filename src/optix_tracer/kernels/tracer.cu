#include <optix.h>
#include "optix_types.h"
#include <float.h>

extern "C" {
__constant__ Params params;
}

#include "rendering/bwd.cuh"

struct Hit{
    int id;
    float thit;
};

constexpr int triagPerParticle = 20;

constexpr unsigned int chunk_size = 16;

constexpr float Tmin = 0.001;

#define SAMPLE_BASED_ORDER false

#define ENSURE_CORRECT_ORDER false

extern "C" __global__ void __raygen__rg() {

    // Lookup our location within the launch grid
    const uint3 idxy = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const int id = idxy.x + idxy.y*dim.x;

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    const float3 ray_origin = params.ray_origins[id];
    const float3 ray_direction = params.ray_directions[id];

    Hit hits[chunk_size];

    unsigned int p_hits[2];
    Hit* hits_ptr = reinterpret_cast<Hit*>(hits);
    memcpy(p_hits, &hits_ptr, sizeof(void*));

    unsigned int hits_size = 0;

    float max_dist = 1e16f; 
    float min_dist = 0.f;
    constexpr float epsT = 1e-9f;

    if(params.compute_grad){
        max_dist = params.distance[id]+epsT;
    }

    Acc acc{};
    acc.radiance = make_float3(0.f);
    acc.transmittance = 1.f;
    unsigned int* uip_acc = reinterpret_cast<unsigned int *>(&acc);

    Acc acc_full{};
    acc_full.radiance = params.radiance[id];
    acc_full.transmittance = params.transmittance[id];
    // int last_hit = -1; -> skip repeating hit
    while((min_dist < max_dist) && (acc.transmittance > Tmin || params.compute_grad)){
        for(int i=0; i<chunk_size;++i){
            hits[i].thit = FLT_MAX;
        }
        optixTrace(
            OPTIX_PAYLOAD_TYPE_ID_0,
            params.handle,
            ray_origin,
            ray_direction,
            min_dist+epsT,                      // Min intersection distance
            max_dist,                     // Max intersection distance
            0.0f,                      // rayTime -- used for motion blur
            OptixVisibilityMask(255),  // Specify always visible
            OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
            //OPTIX_RAY_FLAG_NONE,
            0,  // SBT offset   -- See SBT discussion
            2,  // SBT stride   -- See SBT discussion
            0,  // missSBTIndex -- See SBT discussion
            uip_acc[0],uip_acc[1],uip_acc[2],uip_acc[3],
            p_hits[0],p_hits[1],hits_size,hits_size);

        if(hits[0].thit == FLT_MAX){
            break;
        }
#pragma unroll
        for(int i=0; i<chunk_size;++i){
            const Hit chit = hits[i];
            // if(last_hit == chit.id) continue; -> skip repeating hit
            if((chit.thit != FLT_MAX) && (acc.transmittance > Tmin || params.compute_grad)){
                float resp,thit; 
                bool accept = compute_response(ray_origin,
                                ray_direction,
                                params.gs_xyz[chit.id],
                                params.gs_opacity[chit.id],
                                params.gs_rotation[chit.id],params.gs_scaling[chit.id], 
                                resp,thit);
                if(accept)
                {
                    float3 rad; bool clamped[3];
                    compute_radiance(params.gs_sh,params.sh_deg,chit.id,ray_origin,ray_direction,rad,clamped);
                    acc.radiance += rad*resp*acc.transmittance;
                    if(params.compute_grad){
                        add_grad_at(params.gs_rotation,params.gs_scaling,params.gs_xyz,params.gs_opacity,
                            params.gs_sh,params.sh_deg,params.grad_sh,
                            params.white_background,
                            params.dL_dC[id],
                            params.grad_rotation,params.grad_scale,params.grad_xyz,params.grad_xyz_2d,params.grad_opacity,
                            acc,rad,acc_full,chit.id,
                            ray_origin+ray_direction*thit,
                            resp, ray_origin, ray_direction, clamped);
                    }
                    acc.transmittance *= (1.-resp);
                }
                min_dist = chit.thit; // same as fmaxf(min_dist, chit.thit);
                // last_hit = chit.id; -> skip repeating hit
            }
        }
        if(hits[chunk_size-1].thit == FLT_MAX) break; // buffer is not full->finish
    }
    if(!params.compute_grad){
        params.radiance[id] = acc.radiance;
        params.transmittance[id] = acc.transmittance;
        params.distance[id] = min_dist;
    }
}

extern "C" __global__ void __anyhit__fwd() {

    //atomicAdd(params.num_its,1ull);

    optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0);

    unsigned int p_hitq[2];
    p_hitq[0] = optixGetPayload_4();
    p_hitq[1] = optixGetPayload_5();
    Hit* hitq;
    memcpy(&hitq, p_hitq, sizeof(p_hitq));

    const unsigned int hit_id = optixGetPrimitiveIndex()/triagPerParticle;

    Hit hit;
    hit.id = hit_id;
#if SAMPLE_BASED_ORDER
    float _resp,_thit; 
    if(!compute_response(optixGetWorldRayOrigin(),
        optixGetWorldRayDirection(),
        params.gs_xyz[hit_id],
        params.gs_opacity[hit_id],
        params.gs_rotation[hit_id],params.gs_scaling[hit_id],
        _resp,_thit)){
        optixIgnoreIntersection();
        return;
    }
    hit.thit = fmaxf(_thit,optixGetRayTmax()); // -> skip repeating hit
#else
    float _thit = optixGetRayTmax();
    hit.thit = _thit;
#endif
    
    if(hit.thit < hitq[chunk_size-1].thit)
    {
#pragma unroll
        for(int i = 0; i < chunk_size; ++i){
            Hit hitH = hitq[i];
            if(hit.thit < hitH.thit){
                hitq[i] = hit;
                hit = hitH;
            }
        }

#if !ENSURE_CORRECT_ORDER
        if(_thit < hitq[chunk_size-1].thit)
            optixIgnoreIntersection();
#endif
    }
#if ENSURE_CORRECT_ORDER
    optixIgnoreIntersection();
#endif
}

extern "C" __global__ void __anyhit__bwd() {


    optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_1);

}
