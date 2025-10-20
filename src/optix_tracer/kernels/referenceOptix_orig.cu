#include <optix.h>
#include "optix_types.h"
#include "utils/auxiliary.h"
#include "utils/vec_math.h"
#include <stdint.h>

extern "C" {
__constant__ PipelineParameters params;
}

struct RayHit {
    unsigned int particleId;
    float distance;

    static constexpr unsigned int InvalidParticleId = 0xFFFFFFFF;
    static constexpr float InfiniteDistance         = 1e20f;
};
using RayPayload = RayHit[PipelineParameters::MaxNumHitPerTrace];

static __device__ __inline__ float2 intersectAABB(const OptixAabb& aabb, const float3& rayOri, const float3& rayDir) {
    const float3 t0   = (make_float3(aabb.minX, aabb.minY, aabb.minZ) - rayOri) / rayDir;
    const float3 t1   = (make_float3(aabb.maxX, aabb.maxY, aabb.maxZ) - rayOri) / rayDir;
    const float3 tmax = make_float3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));
    const float3 tmin = make_float3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
    return float2{fmaxf(0.f, fmaxf(tmin.x, fmaxf(tmin.y, tmin.z))), fminf(tmax.x, fminf(tmax.y, tmax.z))};
}

static __device__ __inline__ uint32_t optixPrimitiveIndex() {
    return PipelineParameters::InstancePrimitive ? optixGetInstanceIndex() : (PipelineParameters::CustomPrimitive ? optixGetPrimitiveIndex() : static_cast<uint32_t>(optixGetPrimitiveIndex() / params.gPrimNumTri));
}

static __device__ __inline__ void trace(
    RayPayload& rayPayload,
    const float3& rayOri,
    const float3& rayDir,
    const float tmin,
    const float tmax) {
    uint32_t r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
        r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
    r0 = r2 = r4 = r6 = r8 = r10 = r12 = r14 = r16 = r18 = r20 = r22 = r24 = r26 = r28 = r30 = RayHit::InvalidParticleId;
    r1 = r3 = r5 = r7 = r9 = r11 = r13 = r15 = r17 = r19 = r21 = r23 = r25 = r27 = r29 = r31 = __float_as_int(RayHit::InfiniteDistance);

    // Trace the ray against our scene hierarchy
    optixTrace(params.handle, rayOri, rayDir,
               tmin,                     // Min intersection distance
               tmax,                     // Max intersection distance
               0.0f,                     // rayTime -- used for motion blur
               OptixVisibilityMask(255), // Specify always visible
               OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | (PipelineParameters::SurfelPrimitive ? OPTIX_RAY_FLAG_NONE : OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES),
               0, // SBT offset   -- See SBT discussion
               1, // SBT stride   -- See SBT discussion
               0, // missSBTIndex -- See SBT discussion
               r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
               r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31);

    rayPayload[0].particleId  = r0;
    rayPayload[0].distance    = __uint_as_float(r1);
    rayPayload[1].particleId  = r2;
    rayPayload[1].distance    = __uint_as_float(r3);
    rayPayload[2].particleId  = r4;
    rayPayload[2].distance    = __uint_as_float(r5);
    rayPayload[3].particleId  = r6;
    rayPayload[3].distance    = __uint_as_float(r7);
    rayPayload[4].particleId  = r8;
    rayPayload[4].distance    = __uint_as_float(r9);
    rayPayload[5].particleId  = r10;
    rayPayload[5].distance    = __uint_as_float(r11);
    rayPayload[6].particleId  = r12;
    rayPayload[6].distance    = __uint_as_float(r13);
    rayPayload[7].particleId  = r14;
    rayPayload[7].distance    = __uint_as_float(r15);
    rayPayload[8].particleId  = r16;
    rayPayload[8].distance    = __uint_as_float(r17);
    rayPayload[9].particleId  = r18;
    rayPayload[9].distance    = __uint_as_float(r19);
    rayPayload[10].particleId = r20;
    rayPayload[10].distance   = __uint_as_float(r21);
    rayPayload[11].particleId = r22;
    rayPayload[11].distance   = __uint_as_float(r23);
    rayPayload[12].particleId = r24;
    rayPayload[12].distance   = __uint_as_float(r25);
    rayPayload[13].particleId = r26;
    rayPayload[13].distance   = __uint_as_float(r27);
    rayPayload[14].particleId = r28;
    rayPayload[14].distance   = __uint_as_float(r29);
    rayPayload[15].particleId = r30;
    rayPayload[15].distance   = __uint_as_float(r31);
}

__device__ inline bool processHit(
    const float3& rayOrigin,
    const float3& rayDirection,
    const int32_t particleIdx,
    const ParticleDensity* particlesDensity,
    const float* particlesSphCoefficients,
    const float minParticleKernelDensity,
    const float minParticleAlpha,
    const int32_t sphEvalDegree,
    float* transmittance,
    float3* radiance,
    float* depth,
    float3* normal) {
    float3 particlePosition;
    float3 particleScale;
    float33 particleInvRotation;
    float particleDensity;

    fetchParticleDensity(
        particleIdx,
        particlesDensity,
        particlePosition,
        particleScale,
        particleInvRotation,
        particleDensity);

    const float3 giscl   = make_float3(1 / particleScale.x, 1 / particleScale.y, 1 / particleScale.z);
    const float3 gposc   = (rayOrigin - particlePosition);
    const float3 gposcr  = (gposc * particleInvRotation);
    const float3 gro     = giscl * gposcr;
    const float3 rayDirR = rayDirection * particleInvRotation;
    const float3 grdu    = giscl * rayDirR;
    const float3 grd     = safe_normalize(grdu);

    const float3 gcrod   = SurfelPrimitive ? gro + grd * -gro.z / grd.z : cross(grd, gro);
    const float grayDist = dot(gcrod, gcrod);

    const float gres   = particleResponse<ParticleKernelDegree>(grayDist);
    const float galpha = fminf(0.99f, gres * particleDensity);

    const bool acceptHit = (gres > minParticleKernelDensity) && (galpha > minParticleAlpha);
    if (acceptHit) {
        const float weight = galpha * (*transmittance);

        // distance to the gaussian center projection on the ray
        const float3 grds = particleScale * grd * (SurfelPrimitive ? -gro.z / grd.z : dot(grd, -1 * gro));
        const float hitT  = sqrtf(dot(grds, grds));

        // radiance from sph coefficients
        float3 sphCoefficients[SPH_MAX_NUM_COEFFS];
        fetchParticleSphCoefficients(
            particleIdx,
            particlesSphCoefficients,
            &sphCoefficients[0]);
        const float3 grad = radianceFromSpH(sphEvalDegree, &sphCoefficients[0], rayDirection);

        *radiance += grad * weight;
        *transmittance *= (1 - galpha);
        *depth += hitT * weight;

        if (normal) {
            constexpr float ellispoidSqRadius = 9.0f;
            const float3 particleScaleRotated = (particleInvRotation * particleScale);
            *normal += weight * (SurfelPrimitive ? make_float3(0, 0, (grd.z > 0 ? 1 : -1) * particleScaleRotated.z) : safe_normalize((gro + grd * (dot(grd, -1 * gro) - sqrtf(ellispoidSqRadius - grayDist))) * particleScaleRotated));
        }
    }

    return acceptHit;
}
extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    if ((idx.x > params.frameBounds.x) || (idx.y > params.frameBounds.y)) {
        return;
    }

    float3 rayOrigin    = params.rayWorldOrigin(idx);
    float3 rayDirection = params.rayWorldDirection(idx);

    float3 rayRadiance     = make_float3(0.0f);
    float rayTransmittance = 1.0f;
    float rayHitDistance   = 0.f;
#ifdef ENABLE_NORMALS
    float3 rayNormal = make_float3(0.f);
#endif
#ifdef ENABLE_HIT_COUNTS
    float rayHitsCount = 0.f;
#endif

    //float2 minMaxT       = intersectAABB(params.aabb, rayOrigin, rayDirection);
    float2 minMaxT = {0.f,1e20f};
    constexpr float epsT = 1e-9;

    float rayLastHitDistance = fmaxf(0.0f, minMaxT.x - epsT);
    RayPayload rayPayload;

    while ((rayLastHitDistance <= minMaxT.y) && (rayTransmittance > params.minTransmittance)) {
        trace(rayPayload, rayOrigin, rayDirection, rayLastHitDistance + epsT, minMaxT.y + epsT);
        if (rayPayload[0].particleId == RayHit::InvalidParticleId) {
            break;
        }

#pragma unroll
        for (int i = 0; i < PipelineParameters::MaxNumHitPerTrace; i++) {
            const RayHit rayHit = rayPayload[i];

            if ((rayHit.particleId != RayHit::InvalidParticleId) && (rayTransmittance > params.minTransmittance)) {
                const bool acceptedHit = processHit<PipelineParameters::ParticleKernelDegree, PipelineParameters::SurfelPrimitive>(
                    rayOrigin,
                    rayDirection,
                    rayHit.particleId,
                    params.particleDensity,
                    params.particleRadiance,
                    params.hitMinGaussianResponse,
                    params.alphaMinThreshold,
                    params.sphDegree,
                    &rayTransmittance,
                    &rayRadiance,
                    &rayHitDistance,
#ifdef ENABLE_NORMALS
                    &rayNormal
#else
                    nullptr
#endif
                );

                rayLastHitDistance = fmaxf(rayLastHitDistance, rayHit.distance);

#ifdef ENABLE_HIT_COUNTS
                rayHitsCount += acceptedHit ? 1.0f : 0.f;
#endif
            }
        }
    }

    params.rayRadiance[idx.z][idx.y][idx.x][0]    = rayRadiance.x;
    params.rayRadiance[idx.z][idx.y][idx.x][1]    = rayRadiance.y;
    params.rayRadiance[idx.z][idx.y][idx.x][2]    = rayRadiance.z;
    params.rayDensity[idx.z][idx.y][idx.x][0]     = 1 - rayTransmittance;
    params.rayHitDistance[idx.z][idx.y][idx.x][0] = rayHitDistance;
    params.rayHitDistance[idx.z][idx.y][idx.x][1] = rayLastHitDistance;
#ifdef ENABLE_NORMALS
    params.rayNormal[idx.z][idx.y][idx.x][0] = rayNormal.x;
    params.rayNormal[idx.z][idx.y][idx.x][1] = rayNormal.y;
    params.rayNormal[idx.z][idx.y][idx.x][2] = rayNormal.z;
#endif
#ifdef ENABLE_HIT_COUNTS
    params.rayHitsCount[idx.z][idx.y][idx.x][0] = rayHitsCount;
#endif
}

extern "C" __global__ void __intersection__is() {
    float hitDistance;
    const bool intersect = PipelineParameters::InstancePrimitive ? intersectInstanceParticle(optixGetObjectRayOrigin(),
                                                                                             optixGetObjectRayDirection(),
                                                                                             optixGetInstanceIndex(),
                                                                                             optixGetRayTmin(),
                                                                                             optixGetRayTmax(),
                                                                                             params.hitMaxParticleSquaredDistance,
                                                                                             hitDistance)
                                                                 : intersectCustomParticle(optixGetWorldRayOrigin(),
                                                                                           optixGetWorldRayDirection(),
                                                                                           optixGetPrimitiveIndex(),
                                                                                           params.particleDensity,
                                                                                           optixGetRayTmin(),
                                                                                           optixGetRayTmax(),
                                                                                           params.hitMaxParticleSquaredDistance,
                                                                                           hitDistance);
    if (intersect) {
        optixReportIntersection(hitDistance, 0);
    }
}

#define compareAndSwapHitPayloadValue(hit, i_id, i_distance)                      \
    {                                                                             \
        const float distance = __uint_as_float(optixGetPayload_##i_distance##()); \
        if (hit.distance < distance) {                                            \
            optixSetPayload_##i_distance##(__float_as_uint(hit.distance));        \
            const uint32_t id = optixGetPayload_##i_id##();                       \
            optixSetPayload_##i_id##(hit.particleId);                             \
            hit.distance   = distance;                                            \
            hit.particleId = id;                                                  \
        }                                                                         \
    }

extern "C" __global__ void __anyhit__ah() {

    RayHit hit = RayHit{optixPrimitiveIndex(), optixGetRayTmax()};

    if (hit.distance < __uint_as_float(optixGetPayload_31())) {
        compareAndSwapHitPayloadValue(hit, 0, 1);
        compareAndSwapHitPayloadValue(hit, 2, 3);
        compareAndSwapHitPayloadValue(hit, 4, 5);
        compareAndSwapHitPayloadValue(hit, 6, 7);
        compareAndSwapHitPayloadValue(hit, 8, 9);
        compareAndSwapHitPayloadValue(hit, 10, 11);
        compareAndSwapHitPayloadValue(hit, 12, 13);
        compareAndSwapHitPayloadValue(hit, 14, 15);
        compareAndSwapHitPayloadValue(hit, 16, 17);
        compareAndSwapHitPayloadValue(hit, 18, 19);
        compareAndSwapHitPayloadValue(hit, 20, 21);
        compareAndSwapHitPayloadValue(hit, 22, 23);
        compareAndSwapHitPayloadValue(hit, 24, 25);
        compareAndSwapHitPayloadValue(hit, 26, 27);
        compareAndSwapHitPayloadValue(hit, 28, 29);
        compareAndSwapHitPayloadValue(hit, 30, 31);

        // ignore all inserted hits, expect if the last one
        if (__uint_as_float(optixGetPayload_31()) > optixGetRayTmax()) {
            optixIgnoreIntersection();
        }
    }
}
