#include <optix.h>
#include "optix_types.h"
#include <float.h>
#include "rendering/bwd.cuh"

extern "C" {
__constant__ Params params;
}

struct Hit{
    int id;
    float thit;
    float resp;
};

template<typename H>
__device__ __forceinline__ void hitq_push(H* hitq, unsigned int& hitq_size, const H& hit){
    int j = hitq_size;
    int i = (j-1)>>1;
    while(j!=0 && hitq[i].thit > hit.thit){
        hitq[j] = hitq[i];
        j = i;
        i = (j-1)>>1;
    }
    hitq[j] = hit;
    hitq_size++;
}

template<typename H>
__device__ __forceinline__ void hitq_pop(H* hitq, unsigned int& hitq_size){
    int i = 0;
    int j = 1;
    int bott = hitq_size-1;
    float bott_val = hitq[bott].thit;
    if(j<bott && hitq[j].thit > hitq[j+1].thit) j++;
    while(j<=bott && hitq[j].thit < bott_val){
        hitq[i] = hitq[j];
        i = j;
        j = (i<<1)+1;
        if(j<bott && hitq[j].thit > hitq[j+1].thit) j++;
    }
    hitq[i] = hitq[bott];
    hitq_size--;
}

constexpr int triagPerParticle = 20;

constexpr unsigned int chunk_size = 512;

constexpr float Tmin = 0.001;

constexpr int num_recasts = 1;
constexpr int hits_max_capacity = chunk_size*num_recasts;

__device__ __forceinline__ void add_samp(Acc& acc, const float3& rad, const Hit& chit){
    acc.radiance += rad*chit.resp*acc.transmittance;
    acc.transmittance *= (1.-chit.resp);
}

extern "C" __global__ void __raygen__rg() {

    // Lookup our location within the launch grid
    const uint3 idxy = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const int id = idxy.x + idxy.y*dim.x;

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    const float3 ray_origin = params.ray_origins[id];
    const float3 ray_direction = params.ray_directions[id];

    Hit hits[hits_max_capacity];
    //HitBwd hits_bwd[hits_max_capacity];

    unsigned int p_hits[2];
    Hit* hits_ptr = reinterpret_cast<Hit*>(hits);
    memcpy(p_hits, &hits_ptr, sizeof(void*));

    unsigned int hits_size = 0;

    constexpr float max_dist = 1e16f; 
    float min_dist = 0.f;

    if(!params.compute_grad){ // forward

        Acc acc{};
        acc.radiance = make_float3(0.f);
        acc.transmittance = 1.f;
        unsigned int* uip_acc = reinterpret_cast<unsigned int *>(&acc);

        unsigned int hits_capacity = chunk_size;
        while(hits_capacity <= hits_max_capacity){
            unsigned int n_hits_capacity = hits_capacity;
            optixTrace(
                OPTIX_PAYLOAD_TYPE_ID_0,
                params.handle,
                ray_origin,
                ray_direction,
                min_dist,                      // Min intersection distance
                max_dist,                     // Max intersection distance
                0.0f,                      // rayTime -- used for motion blur
                OptixVisibilityMask(255),  // Specify always visible
                OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,//OPTIX_RAY_FLAG_NONE,
                0,  // SBT offset   -- See SBT discussion
                2,  // SBT stride   -- See SBT discussion
                0,  // missSBTIndex -- See SBT discussion
                uip_acc[0],uip_acc[1],uip_acc[2],uip_acc[3],
                p_hits[0],p_hits[1],hits_size,n_hits_capacity);

            if(n_hits_capacity == hits_capacity){
                break;   
            }
            atomicMax(params.num_its, n_hits_capacity);
            hits_capacity = n_hits_capacity;
            hits_size = 0;
            acc.radiance = make_float3(0.f);
            acc.transmittance = 1.f;
        }
        //atomicMax(params.num_its, hits_size);
        while(hits_size!=0 && acc.transmittance > Tmin){
            const Hit& chit = hits[0];
            float3 rad; bool clamped[3];
            float3 pos = ray_origin+ray_direction*chit.thit;
            compute_radiance(params.gs_sh,params.sh_deg,chit.id,ray_origin,ray_direction,rad,clamped);
            add_samp(acc,rad,chit);
            hitq_pop(hits,hits_size);
        }
        params.radiance[id] = acc.radiance;
        params.transmittance[id] = acc.transmittance;

    } else { // backward

        Acc acc_bwd{};
        acc_bwd.radiance = make_float3(0.f);
        acc_bwd.transmittance = 1.f;
        unsigned int* uip_acc_bwd = reinterpret_cast<unsigned int *>(&acc_bwd);

        Acc acc_full{};
        acc_full.radiance = params.radiance[id];
        acc_full.transmittance = params.transmittance[id];
        unsigned int* uip_acc_full = reinterpret_cast<unsigned int *>(&acc_full);

        optixTrace(
            OPTIX_PAYLOAD_TYPE_ID_1,
            params.handle,
            ray_origin,
            ray_direction,
            min_dist,                      // Min intersection distance
            max_dist,                     // Max intersection distance
            0.0f,                      // rayTime -- used for motion blur
            OptixVisibilityMask(255),  // Specify always visible
            OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,//OPTIX_RAY_FLAG_NONE,
            1,  // SBT offset   -- See SBT discussion
            2,  // SBT stride   -- See SBT discussion
            0,  // missSBTIndex -- See SBT discussion
            uip_acc_bwd[0],uip_acc_bwd[1],uip_acc_bwd[2],uip_acc_bwd[3],
            p_hits[0],p_hits[1],hits_size,
            uip_acc_full[0],uip_acc_full[1],uip_acc_full[2],uip_acc_full[3]
            );
        while(hits_size!=0 && acc_bwd.transmittance > Tmin){
            const Hit& chit = hits[0];
            float3 rad; bool clamped[3];

            float3 pos = ray_origin+ray_direction*chit.thit;
            compute_radiance(params.gs_sh,params.sh_deg,chit.id,ray_origin,ray_direction,rad,clamped);
            acc_bwd.radiance += rad*chit.resp*acc_bwd.transmittance;

            add_grad_at(params.gs_rotation,params.gs_scaling,params.gs_xyz,params.gs_opacity,
                params.gs_sh,params.sh_deg,params.grad_sh,
                params.white_background,
                params.dL_dC[id],
                params.grad_rotation,params.grad_scale,params.grad_xyz,params.grad_opacity,
                acc_bwd,rad,acc_full,chit.id,
                ray_origin+ray_direction*chit.thit,
                chit.resp, ray_origin, ray_direction, clamped);

            acc_bwd.transmittance *= (1.-chit.resp);
            hitq_pop(hits,hits_size);
        }
    }
}

extern "C" __global__ void __anyhit__fwd() {

    optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0);

    Acc acc;
    acc.radiance.x = __uint_as_float(optixGetPayload_0());
    acc.radiance.y = __uint_as_float(optixGetPayload_1());
    acc.radiance.z = __uint_as_float(optixGetPayload_2());
    acc.transmittance = __uint_as_float(optixGetPayload_3());

    unsigned int p_hitq[2];
    p_hitq[0] = optixGetPayload_4();
    p_hitq[1] = optixGetPayload_5();
    Hit* hitq;
    memcpy(&hitq, p_hitq, sizeof(p_hitq));

    unsigned int hitq_size = optixGetPayload_6();
    unsigned int hitq_capacity = optixGetPayload_7();

    const unsigned int hit_id = optixGetPrimitiveIndex()/triagPerParticle;
    float resp,thit; 
    if(!compute_response(optixGetWorldRayOrigin(),
                    optixGetWorldRayDirection(),
                    params.gs_xyz[hit_id],
                    params.gs_opacity[hit_id],
                    construct_inv_RS(params.gs_rotation[hit_id],params.gs_scaling[hit_id]),
                    resp,thit)){
        optixIgnoreIntersection();
        return;
    }

    if(hitq_size == hitq_capacity){
        const Hit &chit = hitq[0];
        
        if(hitq_capacity != hits_max_capacity && thit < chit.thit){
            optixSetPayload_7(hitq_capacity+chunk_size);
            return;
        }
        float3 rad; bool clamped[3];
        compute_radiance(params.gs_sh,params.sh_deg,chit.id,optixGetWorldRayOrigin(),optixGetWorldRayDirection(),rad,clamped);
        add_samp(acc,rad,chit);

        optixSetPayload_0(__float_as_uint(acc.radiance.x));
        optixSetPayload_1(__float_as_uint(acc.radiance.y));
        optixSetPayload_2(__float_as_uint(acc.radiance.z));
        optixSetPayload_3(__float_as_uint(acc.transmittance));
        if(acc.transmittance < Tmin){
            return;
        }
        hitq_pop(hitq,hitq_size);
        optixSetPayload_6(hitq_size);
    }

    Hit hit;
    hit.id = hit_id;
    hit.resp = resp;
    hit.thit = thit;
    hitq_push(hitq,hitq_size,hit);
    optixSetPayload_6(hitq_size);

    optixIgnoreIntersection();
}

extern "C" __global__ void __anyhit__bwd() {


    optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_1);

    Acc acc;
    acc.radiance.x = __uint_as_float(optixGetPayload_0());
    acc.radiance.y = __uint_as_float(optixGetPayload_1());
    acc.radiance.z = __uint_as_float(optixGetPayload_2());
    acc.transmittance = __uint_as_float(optixGetPayload_3());

    unsigned int p_hitq[2];
    p_hitq[0] = optixGetPayload_4();
    p_hitq[1] = optixGetPayload_5();
    Hit* hitq;
    memcpy(&hitq, p_hitq, sizeof(p_hitq));

    unsigned int hitq_size = optixGetPayload_6();

    Acc acc_full;
    acc_full.radiance.x = __uint_as_float(optixGetPayload_7());
    acc_full.radiance.y = __uint_as_float(optixGetPayload_8());
    acc_full.radiance.z = __uint_as_float(optixGetPayload_9());
    acc_full.transmittance = __uint_as_float(optixGetPayload_10());

    const float3 origin = optixGetWorldRayOrigin();
    const float3 direction = optixGetWorldRayDirection();

    if(hitq_size == chunk_size){
        const Hit &chit = hitq[0];

        float3 rad; bool clamped[3];
        compute_radiance(params.gs_sh,params.sh_deg,chit.id,origin,direction,rad,clamped);

        acc.radiance += rad*chit.resp*acc.transmittance;
        {
            const uint3 idxy = optixGetLaunchIndex();
            const uint3 dim = optixGetLaunchDimensions();
            const int id = idxy.x + idxy.y*dim.x;
            add_grad_at(params.gs_rotation,params.gs_scaling,params.gs_xyz,params.gs_opacity,
                params.gs_sh,params.sh_deg,params.grad_sh,
                params.white_background,
                params.dL_dC[id],
                params.grad_rotation,params.grad_scale,params.grad_xyz,params.grad_opacity,
                acc,rad,acc_full,chit.id,
                origin+direction*chit.thit,
                chit.resp,origin,direction,clamped);
        }
        acc.transmittance *= (1.-chit.resp);
        optixSetPayload_0(__float_as_uint(acc.radiance.x));
        optixSetPayload_1(__float_as_uint(acc.radiance.y));
        optixSetPayload_2(__float_as_uint(acc.radiance.z));
        optixSetPayload_3(__float_as_uint(acc.transmittance));

        if(acc.transmittance < Tmin){
            return;
        }
        
        hitq_pop(hitq,hitq_size);
        optixSetPayload_6(hitq_size);
    }

    const unsigned int hit_id = optixGetPrimitiveIndex()/triagPerParticle;
    const Matrix3x3 inv_RS = construct_inv_RS(params.gs_rotation[hit_id],params.gs_scaling[hit_id]);
    float resp,thit; 
    if(!compute_response(origin,
                    direction,
                    params.gs_xyz[hit_id],
                    params.gs_opacity[hit_id],
                    inv_RS,
                    resp,thit)){
        optixIgnoreIntersection();
        return;
    }

    Hit hit;
    hit.id = hit_id;
    hit.resp = resp;
    hit.thit = thit;
    hitq_push(hitq,hitq_size,hit);
    optixSetPayload_6(hitq_size);

    optixIgnoreIntersection();
}