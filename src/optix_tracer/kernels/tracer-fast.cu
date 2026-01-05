#include <optix.h>
#include "optix_types.h"
#include <float.h>
#include "rendering/bwd.cuh"

extern "C" {
__constant__ Params params;
}

struct Hit{
    unsigned int id;
    unsigned int thit; // float
};

constexpr float Tmin = 0.001;

extern "C" __global__ void __raygen__rg() {

    const uint3 idxy = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const int id = idxy.x + idxy.y*dim.x;

    const float3 ray_origin = params.ray_origins[id];
    const float3 ray_direction = params.ray_directions[id];

    Hit hits[16];

    float max_dist = 1e16f; 
    float min_dist = 0.f;
    constexpr float epsT = 1e-9f;

    Acc acc{};
    acc.radiance = make_float3(0.f);
    acc.transmittance = 1.f;

    const unsigned int uint_flt_max = __float_as_uint(FLT_MAX);

    while((min_dist < max_dist) && acc.transmittance > Tmin){
//#pragma unroll
//        for(int i=0; i<16;++i){
//            hits[i].thit = uint_flt_max;
//        }
//#define h(i) hits[i].id,hits[i].thit
        unsigned int r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
        r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
        r1=r3=r5=r7=r9=r11=r13=r15=r17=r19=r21=r23=r25=r27=r29=r31 = uint_flt_max;
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
            //h(0),h(1),h(2),h(3),h(4),h(5),h(6),h(7),h(8),h(9),h(10),h(11),h(12),h(13),h(14),h(15)
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
            r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31
            );
#define r(i,rj) hits[i>>1].id=r##i, hits[i>>1].thit=rj
        r(0, r1), r(2, r3), r(4, r5), r(6, r7), r(8, r9), r(10, r11), r(12, r13), r(14, r15),
        r(16, r17), r(18, r19), r(20, r21), r(22, r23), r(24, r25), r(26, r27), r(28, r29), r(30, r31);

        if(hits[0].thit == uint_flt_max){
            break;
        }
#pragma unroll
        for(int i=0; i<16;++i){
            Hit chit = hits[i];
            chit.id /= 20;
            // if(last_hit == chit.id) continue; -> skip repeating hit
            if((chit.thit != uint_flt_max) && (acc.transmittance > Tmin)){
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
                    acc.transmittance *= (1.-resp);
                }
                min_dist = __uint_as_float(chit.thit); // same as fmaxf(min_dist, chit.thit);
            }
        }
        if(hits[15].thit == uint_flt_max) break; // buffer is not full->finish
    }
    params.radiance[id] = acc.radiance;
    params.transmittance[id] = acc.transmittance;
    params.distance[id] = min_dist;
}


extern "C" __global__ void __raygen__bwd() {

    const uint3 idxy = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const int id = idxy.x + idxy.y*dim.x;

    const float3 ray_origin = params.ray_origins[id];
    const float3 ray_direction = params.ray_directions[id];

    Hit hits[16];

    constexpr float epsT = 1e-9f;
    float max_dist = params.distance[id]+epsT;
    float min_dist = 0.f;

    Acc acc{};
    acc.radiance = make_float3(0.f);
    acc.transmittance = 1.f;

    Acc acc_full{};
    acc_full.radiance = params.radiance[id];
    acc_full.transmittance = params.transmittance[id];

    const unsigned int uint_flt_max = __float_as_uint(FLT_MAX);

    while(min_dist < max_dist){
//#pragma unroll
//        for(int i=0; i<16;++i){
//            hits[i].thit = uint_flt_max;
//        }
//#define h(i) hits[i].id,hits[i].thit
        unsigned int r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
        r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
        r1=r3=r5=r7=r9=r11=r13=r15=r17=r19=r21=r23=r25=r27=r29=r31 = uint_flt_max;
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
            //h(0),h(1),h(2),h(3),h(4),h(5),h(6),h(7),h(8),h(9),h(10),h(11),h(12),h(13),h(14),h(15)
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
            r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31
            );
#define r(i,rj) hits[i>>1].id=r##i, hits[i>>1].thit=rj
        r(0, r1), r(2, r3), r(4, r5), r(6, r7), r(8, r9), r(10, r11), r(12, r13), r(14, r15),
        r(16, r17), r(18, r19), r(20, r21), r(22, r23), r(24, r25), r(26, r27), r(28, r29), r(30, r31);

        if(hits[0].thit == uint_flt_max){
            break;
        }
#pragma unroll
        for(int i=0; i<16;++i){
            Hit chit = hits[i];
            chit.id /= 20;
            // if(last_hit == chit.id) continue; -> skip repeating hit
            if(chit.thit != uint_flt_max){
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
                    add_grad_at(params.gs_rotation,params.gs_scaling,params.gs_xyz,params.gs_opacity,
                        params.gs_sh,params.sh_deg,params.grad_sh,
                        params.white_background,
                        params.dL_dC[id],
                        params.grad_rotation,params.grad_scale,params.grad_xyz,params.grad_xyz_2d,params.grad_opacity,
                        acc,rad,acc_full,chit.id,
                        ray_origin+ray_direction*thit,
                        resp, ray_origin, ray_direction, clamped);
                    acc.transmittance *= (1.-resp);
                }
                min_dist = __uint_as_float(chit.thit); // same as fmaxf(min_dist, chit.thit);
            }
        }
        if(hits[15].thit == uint_flt_max) break; // buffer is not full->finish
    }
}

#define compareAndSwapHitPayloadValue(g_id, g_thit, s_id, s_thit)\
    {\
        const float thit = __uint_as_float(g_thit());\
        if (last_thit < thit) {\
            unsigned int id = g_id();\
            s_thit(__float_as_uint(last_thit));\
            s_id(last_id);\
            last_thit = thit;\
            last_id = id;\
        }\
    }

extern "C" __global__ void __anyhit__fwd() {

    optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0);

    unsigned int last_id = optixGetPrimitiveIndex();
    float last_thit = optixGetRayTmax();

    if (last_thit < __uint_as_float(optixGetPayload_31())) {

        compareAndSwapHitPayloadValue(optixGetPayload_0,  optixGetPayload_1, optixSetPayload_0,  optixSetPayload_1);
        compareAndSwapHitPayloadValue(optixGetPayload_2,  optixGetPayload_3, optixSetPayload_2,  optixSetPayload_3);
        compareAndSwapHitPayloadValue(optixGetPayload_4,  optixGetPayload_5, optixSetPayload_4,  optixSetPayload_5);
        compareAndSwapHitPayloadValue(optixGetPayload_6,  optixGetPayload_7, optixSetPayload_6,  optixSetPayload_7);
        compareAndSwapHitPayloadValue(optixGetPayload_8,  optixGetPayload_9, optixSetPayload_8,  optixSetPayload_9);
        compareAndSwapHitPayloadValue(optixGetPayload_10, optixGetPayload_11, optixSetPayload_10, optixSetPayload_11);
        compareAndSwapHitPayloadValue(optixGetPayload_12, optixGetPayload_13, optixSetPayload_12, optixSetPayload_13);
        compareAndSwapHitPayloadValue(optixGetPayload_14, optixGetPayload_15, optixSetPayload_14, optixSetPayload_15);
        compareAndSwapHitPayloadValue(optixGetPayload_16, optixGetPayload_17, optixSetPayload_16, optixSetPayload_17);
        compareAndSwapHitPayloadValue(optixGetPayload_18, optixGetPayload_19, optixSetPayload_18, optixSetPayload_19);
        compareAndSwapHitPayloadValue(optixGetPayload_20, optixGetPayload_21, optixSetPayload_20, optixSetPayload_21);
        compareAndSwapHitPayloadValue(optixGetPayload_22, optixGetPayload_23, optixSetPayload_22, optixSetPayload_23);
        compareAndSwapHitPayloadValue(optixGetPayload_24, optixGetPayload_25, optixSetPayload_24, optixSetPayload_25);
        compareAndSwapHitPayloadValue(optixGetPayload_26, optixGetPayload_27, optixSetPayload_26, optixSetPayload_27);
        compareAndSwapHitPayloadValue(optixGetPayload_28, optixGetPayload_29, optixSetPayload_28, optixSetPayload_29);
        compareAndSwapHitPayloadValue(optixGetPayload_30, optixGetPayload_31, optixSetPayload_30, optixSetPayload_31);

        // ignore all inserted hits, expect if the last one
        if (__uint_as_float(optixGetPayload_31()) > optixGetRayTmax()) {
            optixIgnoreIntersection();
        }
    }
}


extern "C" __global__ void __anyhit__bwd() {

    optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_1);
}