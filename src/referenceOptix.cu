#include <optix.h>
#include "optix_types.h"
#include "utils/auxiliary.h"
#include "utils/vec_math.h"
#include <stdint.h>

extern "C" {
__constant__ Params params;
}

struct RayHit {
    unsigned int particleId;
    float distance;

    static constexpr unsigned int InvalidParticleId = 0xFFFFFFFF;
    static constexpr float InfiniteDistance         = 1e20f;
};
//using RayPayload = RayHit[PipelineParameters::MaxNumHitPerTrace];
using RayPayload = RayHit[16];

// static __device__ __inline__ float2 intersectAABB(const OptixAabb& aabb, const float3& rayOri, const float3& rayDir) {
//     const float3 t0   = (make_float3(aabb.minX, aabb.minY, aabb.minZ) - rayOri) / rayDir;
//     const float3 t1   = (make_float3(aabb.maxX, aabb.maxY, aabb.maxZ) - rayOri) / rayDir;
//     const float3 tmax = make_float3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));
//     const float3 tmin = make_float3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
//     return float2{fmaxf(0.f, fmaxf(tmin.x, fmaxf(tmin.y, tmin.z))), fminf(tmax.x, fminf(tmax.y, tmax.z))};
// }

static __device__ __inline__ uint32_t optixPrimitiveIndex() {
    return static_cast<uint32_t>(optixGetPrimitiveIndex() / 20);
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
    optixTrace(OPTIX_PAYLOAD_TYPE_ID_0,params.handle, rayOri, rayDir,
               tmin,                     // Min intersection distance
               tmax,                     // Max intersection distance
               0.0f,                     // rayTime -- used for motion blur
               OptixVisibilityMask(255), // Specify always visible
               OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
               0, // SBT offset   -- See SBT discussion
               2, // SBT stride   -- See SBT discussion
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

using float33 = float3[3];

static __device__ inline float3 operator*(const float33& m, const float3& p) {
    return make_float3(
        dot(make_float3(m[0].x, m[1].x, m[2].x), p),
        dot(make_float3(m[0].y, m[1].y, m[2].y), p),
        dot(make_float3(m[0].z, m[1].z, m[2].z), p));
}

static __device__ inline float3 operator*(const float3& p, const float33& m) {
    return make_float3(dot(m[0], p), dot(m[1], p), dot(m[2], p));
}

static __device__ inline float3 matmul_bw_vec(const float33& m, const float3& gdt) {
    return make_float3(
        gdt.x * m[0].x + gdt.y * m[1].x + gdt.z * m[2].x,
        gdt.x * m[0].y + gdt.y * m[1].y + gdt.z * m[2].y,
        gdt.x * m[0].z + gdt.y * m[1].z + gdt.z * m[2].z);
}


static __device__ inline float4 matmul_bw_quat(const float3& p, const float3& g, const float4& q) {
    float33 dmat;
    dmat[0] = g.x * p;
    dmat[1] = g.y * p;
    dmat[2] = g.z * p;

    const float r = q.x;
    const float x = q.y;
    const float y = q.z;
    const float z = q.w;

    float dr = 0;
    float dx = 0;
    float dy = 0;
    float dz = 0;

    // m[0] = make_float3((1.f - 2.f * (y * y + z * z)), 2.f * (x * y + r * z), 2.f * (x * z - r * y));

    // m[0].x = (1.f - 2.f * (y * y + z * z))
    dy += -4 * y * dmat[0].x;
    dz += -4 * z * dmat[0].x;
    // m[0].y = 2.f * (x * y + r * z)
    dr += 2 * z * dmat[0].y;
    dx += 2 * y * dmat[0].y;
    dy += 2 * x * dmat[0].y;
    dz += 2 * r * dmat[0].y;
    // m[0].z = 2.f * (x * z - r * y)
    dr += -2 * y * dmat[0].z;
    dx += 2 * z * dmat[0].z;
    dy += -2 * r * dmat[0].z;
    dz += 2 * x * dmat[0].z;

    // m[1] = make_float3(2.f * (x * y - r * z), (1.f - 2.f * (x * x + z * z)), 2.f * (y * z + r * x));

    // m[1].x = 2.f * (x * y - r * z)
    dr += -2 * z * dmat[1].x;
    dx += 2 * y * dmat[1].x;
    dy += 2 * x * dmat[1].x;
    dz += -2 * r * dmat[1].x;
    // m[1].y = (1.f - 2.f * (x * x + z * z))
    dx += -4 * x * dmat[1].y;
    dz += -4 * z * dmat[1].y;
    // m[1].z = 2.f * (y * z + r * x))
    dr += 2 * x * dmat[1].z;
    dx += 2 * r * dmat[1].z;
    dy += 2 * z * dmat[1].z;
    dz += 2 * y * dmat[1].z;

    // m[2] = make_float3(2.f * (x * z + r * y), 2.f * (y * z - r * x), (1.f - 2.f * (x * x + y * y)));

    // m[2].x = 2.f * (x * z + r * y)
    dr += 2 * y * dmat[2].x;
    dx += 2 * z * dmat[2].x;
    dy += 2 * r * dmat[2].x;
    dz += 2 * x * dmat[2].x;
    // m[2].y = 2.f * (y * z - r * x)
    dr += -2 * x * dmat[2].y;
    dx += -2 * r * dmat[2].y;
    dy += 2 * z * dmat[2].y;
    dz += 2 * y * dmat[2].y;
    // m[2].z = (1.f - 2.f * (x * x + y * y))
    dx += -4 * x * dmat[2].z;
    dy += -4 * y * dmat[2].z;

    return make_float4(dr, dx, dy, dz);
}

__device__ void rotationMatrixTranspose(const float4& q, float33& ret) {
    const float r = q.x;
    const float x = q.y;
    const float y = q.z;
    const float z = q.w;

    const float xx = x * x;
    const float yy = y * y;
    const float zz = z * z;
    const float xy = x * y;
    const float xz = x * z;
    const float yz = y * z;
    const float rx = r * x;
    const float ry = r * y;
    const float rz = r * z;

    // Compute rotation matrix from quaternion
    ret[0] = make_float3((1.f - 2.f * (yy + zz)), 2.f * (xy + rz), 2.f * (xz - ry));
    ret[1] = make_float3(2.f * (xy - rz), (1.f - 2.f * (xx + zz)), 2.f * (yz + rx));
    ret[2] = make_float3(2.f * (xz + ry), 2.f * (yz - rx), (1.f - 2.f * (xx + yy)));
}

#include "math.h"

static __device__ inline float3 safe_normalize(float3 v) {
    const float l = v.x * v.x + v.y * v.y + v.z * v.z;
    return l > 0.0f ? (v * rsqrtf(l)) : v;
}

static __device__ inline float3 safe_normalize_bw(const float3& v, const float3& d_out) {
    const float l = v.x * v.x + v.y * v.y + v.z * v.z;
    if (l > 0.0f) {
        const float il  = rsqrtf(l);
        const float il3 = (il * il * il);
        return il * d_out - il3 * make_float3(d_out.x * (v.x * v.x) + d_out.y * (v.y * v.x) + d_out.z * (v.z * v.x),
                                              d_out.x * (v.x * v.y) + d_out.y * (v.y * v.y) + d_out.z * (v.z * v.y),
                                              d_out.x * (v.x * v.z) + d_out.y * (v.y * v.z) + d_out.z * (v.z * v.z));
    }
    return make_float3(0);
}

#define SPH_MAX_NUM_COEFFS 16

static inline __device__ void fetchParticleSphCoefficients(
    const int32_t particleIdx,
    const float* particlesSphCoefficients,
    float3* sphCoefficients) {
    const uint32_t particleOffset = particleIdx * SPH_MAX_NUM_COEFFS * 3;
#pragma unroll
    for (unsigned int i = 0; i < SPH_MAX_NUM_COEFFS; ++i) {
        const int offset   = i * 3;
        sphCoefficients[i] = make_float3(
            particlesSphCoefficients[particleOffset + offset + 0],
            particlesSphCoefficients[particleOffset + offset + 1],
            particlesSphCoefficients[particleOffset + offset + 2]);
    }
}


static inline __device__ float3
radianceFromSpH(int deg, const float3* sphCoefficients, const float3& rdir, bool clamped = true) {
    float3 rad = SH_C0 * sphCoefficients[0];
    if (deg > 0) {
        const float3& dir = rdir;

        const float x = dir.x;
        const float y = dir.y;
        const float z = dir.z;
        rad           = rad - SH_C1 * y * sphCoefficients[1] + SH_C1 * z * sphCoefficients[2] -
              SH_C1 * x * sphCoefficients[3];

        if (deg > 1) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, yz = y * z, xz = x * z;
            rad = rad + SH_C2[0] * xy * sphCoefficients[4] + SH_C2[1] * yz * sphCoefficients[5] +
                  SH_C2[2] * (2.0f * zz - xx - yy) * sphCoefficients[6] +
                  SH_C2[3] * xz * sphCoefficients[7] + SH_C2[4] * (xx - yy) * sphCoefficients[8];

            if (deg > 2) {
                rad = rad + SH_C3[0] * y * (3.0f * xx - yy) * sphCoefficients[9] +
                      SH_C3[1] * xy * z * sphCoefficients[10] +
                      SH_C3[2] * y * (4.0f * zz - xx - yy) * sphCoefficients[11] +
                      SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sphCoefficients[12] +
                      SH_C3[4] * x * (4.0f * zz - xx - yy) * sphCoefficients[13] +
                      SH_C3[5] * z * (xx - yy) * sphCoefficients[14] +
                      SH_C3[6] * x * (xx - 3.0f * yy) * sphCoefficients[15];
            }
        }
    }
    rad += make_float3(0.5f);
    return clamped ? fmaxf(rad, make_float3(0.0f)) : rad;
}

static inline __device__ void addSphCoeffGrd(float3* sphCoefficientsGrad, int idx, const float3& val) {
    atomicAdd(&sphCoefficientsGrad[idx].x, val.x);
    atomicAdd(&sphCoefficientsGrad[idx].y, val.y);
    atomicAdd(&sphCoefficientsGrad[idx].z, val.z);
}

static inline __device__ float3 radianceFromSpHBwd(
    int deg, const float3* sphCoefficients, const float3& rdir, float weight, const float3& rayRadGrd, float3* sphCoefficientsGrad) {
    // radiance unclamped
    const float3 gradu = radianceFromSpH(deg, sphCoefficients, rdir, false);

    // clamped radiance
    float3 grad = make_float3(gradu.x > 0.0f ? gradu.x : 0.0f,
                              gradu.y > 0.0f ? gradu.y : 0.0f,
                              gradu.z > 0.0f ? gradu.z : 0.0f);

    //
    float3 dL_dRGB = rayRadGrd * weight;
    dL_dRGB.x *= (gradu.x > 0.0f ? 1 : 0);
    dL_dRGB.y *= (gradu.y > 0.0f ? 1 : 0);
    dL_dRGB.z *= (gradu.z > 0.0f ? 1 : 0);

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // ---> rayRad = weight * grad = weight * explu(gsph0 * SH_C0 +
    // 0.5,SHRadMinBound) with explu(x,a) = x if x > a else a*e(x-a)
    // ===> d_rayRad / d_gsph0 =   weight * SH_C0
    addSphCoeffGrd(sphCoefficientsGrad, 0, SH_C0 * dL_dRGB);

    if (deg > 0) {
        // const float3 sphdiru = gpos - rori;
        // const float3 sphdir = safe_normalize(sphdiru);
        const float3& sphdir = rdir;

        float x = sphdir.x;
        float y = sphdir.y;
        float z = sphdir.z;

        float dRGBdsh1 = -SH_C1 * y;
        float dRGBdsh2 = SH_C1 * z;
        float dRGBdsh3 = -SH_C1 * x;

        addSphCoeffGrd(sphCoefficientsGrad, 1, dRGBdsh1 * dL_dRGB);
        addSphCoeffGrd(sphCoefficientsGrad, 2, dRGBdsh2 * dL_dRGB);
        addSphCoeffGrd(sphCoefficientsGrad, 3, dRGBdsh3 * dL_dRGB);

        if (deg > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;

            float dRGBdsh4 = SH_C2[0] * xy;
            float dRGBdsh5 = SH_C2[1] * yz;
            float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
            float dRGBdsh7 = SH_C2[3] * xz;
            float dRGBdsh8 = SH_C2[4] * (xx - yy);

            addSphCoeffGrd(sphCoefficientsGrad, 4, dRGBdsh4 * dL_dRGB);
            addSphCoeffGrd(sphCoefficientsGrad, 5, dRGBdsh5 * dL_dRGB);
            addSphCoeffGrd(sphCoefficientsGrad, 6, dRGBdsh6 * dL_dRGB);
            addSphCoeffGrd(sphCoefficientsGrad, 7, dRGBdsh7 * dL_dRGB);
            addSphCoeffGrd(sphCoefficientsGrad, 8, dRGBdsh8 * dL_dRGB);

            if (deg > 2) {
                float dRGBdsh9  = SH_C3[0] * y * (3.f * xx - yy);
                float dRGBdsh10 = SH_C3[1] * xy * z;
                float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
                float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
                float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
                float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
                float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);

                addSphCoeffGrd(sphCoefficientsGrad, 9, dRGBdsh9 * dL_dRGB);
                addSphCoeffGrd(sphCoefficientsGrad, 10, dRGBdsh10 * dL_dRGB);
                addSphCoeffGrd(sphCoefficientsGrad, 11, dRGBdsh11 * dL_dRGB);
                addSphCoeffGrd(sphCoefficientsGrad, 12, dRGBdsh12 * dL_dRGB);
                addSphCoeffGrd(sphCoefficientsGrad, 13, dRGBdsh13 * dL_dRGB);
                addSphCoeffGrd(sphCoefficientsGrad, 14, dRGBdsh14 * dL_dRGB);
                addSphCoeffGrd(sphCoefficientsGrad, 15, dRGBdsh15 * dL_dRGB);
            }
        }
    }

    return grad;
}


#include "utils/Matrix.h"
using namespace util;

constexpr float eps = 1e-6;

__device__ bool compute_response(
    const float3& o, const float3& d, const float3& mu,
    const float opacity, const Matrix3x3& inv_RS,unsigned int chit_id,
    float& alpha, float& tmax){

    constexpr float min_kernel_density = 0.0113f;
    constexpr float min_alpha = 1/255.f; //0.01f;

    float3 og = inv_RS*(mu-o);
    float3 dg = inv_RS*d;

    tmax = dot(og,dg)/max(eps,dot(dg,dg));
    //tmax = dot(og,dg)/(eps+dot(dg,dg));
    float3 c_samp = o+tmax*d;
    float3 v = inv_RS*(c_samp-mu);
    float resp = exp(-.5f*dot(v,v));
    if(resp < min_kernel_density) return false;
    alpha = min(0.99f,opacity*resp);
    
    return (alpha > min_alpha) && (resp > min_kernel_density);
}

__device__ Matrix3x3 construct_rotation(float4 q){
    q = normalize(q);
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

__device__ __forceinline__ void inv_S_times_M_(const float3 s, Matrix3x3& M){
    M.setRow(0,M.getRow(0)/max(s.x,eps));
    M.setRow(1,M.getRow(1)/max(s.y,eps));
    M.setRow(2,M.getRow(2)/max(s.z,eps));
}

__device__ __forceinline__ Matrix3x3 construct_inv_RS(const float4& rot, const float3& s){
    Matrix3x3 RT = construct_rotation(rot).transpose();
    inv_S_times_M_(s,RT);
    return RT;
}

__device__ inline bool processHit(
    const float3& rayOrigin,
    const float3& rayDirection,
    const int32_t particleIdx,
    const float minParticleKernelDensity,
    const float minParticleAlpha,
    float* transmittance,
    float3* radiance) {

    float3 particlePosition = params.gs_xyz[particleIdx];
    float3 particleScale = params.gs_scaling[particleIdx];
    float33 particleInvRotation;
    rotationMatrixTranspose(params.gs_rotation[particleIdx],particleInvRotation);
    float particleDensity = params.gs_opacity[particleIdx];

    const float3 giscl   = make_float3(1 / particleScale.x, 1 / particleScale.y, 1 / particleScale.z);
    const float3 gposc   = (rayOrigin - particlePosition);
    const float3 gposcr  = (gposc * particleInvRotation);
    const float3 gro     = giscl * gposcr;
    const float3 rayDirR = rayDirection * particleInvRotation;
    const float3 grdu    = giscl * rayDirR;
    const float3 grd     = safe_normalize(grdu);
    const float3 gcrod   =  cross(grd, gro);
    const float grayDist = dot(gcrod, gcrod);
    const float gres   = expf(-0.5f*grayDist);//particleResponse<ParticleKernelDegree>(grayDist);
    const float galpha = fminf(0.99f, gres * particleDensity);

    const bool acceptHit = (gres > minParticleKernelDensity) && (galpha > minParticleAlpha);
    if (acceptHit) {
        const float weight = galpha * (*transmittance);

        // distance to the gaussian center projection on the ray
        // const float3 grds = particleScale * grd * (dot(grd, -1 * gro));
        // const float hitT  = sqrtf(dot(grds, grds));

        // radiance from sph coefficients
        float3 sphCoefficients[SPH_MAX_NUM_COEFFS];
        fetchParticleSphCoefficients(
            particleIdx,
            (float*)params.gs_sh,
            &sphCoefficients[0]);
        const float3 grad = radianceFromSpH(params.sh_deg, &sphCoefficients[0], rayDirection);
        //float3 grad; bool clamped[3];
        //compute_radiance(particleIdx,rayOrigin,rayDirection,grad,clamped);

        *radiance += grad * weight;
        *transmittance *= (1 - galpha);
    }

    return acceptHit;
}

__device__ inline void processHitBwd(
    const float3& rayOrigin,
    const float3& rayDirection,
    int32_t particleIdx,
    const float* particleRadiancePtr,
    float* particleRadianceGradPtr,
    float minParticleKernelDensity,
    float minParticleAlpha,
    float minTransmittance,
    int32_t sphEvalDegree,
    float integratedTransmittance,
    float& transmittance,
    //float transmittanceGrad,
    float3 integratedRadiance,
    float3& radiance,
    float3 radianceGrad) {

    float3 particlePosition;
    float3 gscl;
    float33 particleInvRotation;
    float particleDensity;
    float4 grot;

    {
        particlePosition                   = params.gs_xyz[particleIdx];
        gscl                               = params.gs_scaling[particleIdx];
        grot                               = params.gs_rotation[particleIdx];
        rotationMatrixTranspose(grot, particleInvRotation);
        particleDensity = params.gs_opacity[particleIdx];
    }

    // project ray in the gaussian
    const float3 giscl   = make_float3(1 / gscl.x, 1 / gscl.y, 1 / gscl.z);
    const float3 gposc   = (rayOrigin - particlePosition);
    const float3 gposcr  = (gposc * particleInvRotation);
    const float3 gro     = giscl * gposcr;
    const float3 rayDirR = rayDirection * particleInvRotation;
    const float3 grdu    = giscl * rayDirR;
    const float3 grd     = safe_normalize(grdu);
    const float3 gcrod   = cross(grd, gro);
    const float grayDist = dot(gcrod, gcrod);

    const float gres   = expf(-0.5f*grayDist);//particleResponse<ParticleKernelDegree>(grayDist);
    const float galpha = fminf(0.99f, gres * particleDensity);

    if ((gres > minParticleKernelDensity) && (galpha > minParticleAlpha))
    {

        const float weight = galpha * transmittance;

        const float nextTransmit = (1 - galpha) * transmittance;


        float3 sphCoefficients[SPH_MAX_NUM_COEFFS];
        fetchParticleSphCoefficients(
            particleIdx,
            particleRadiancePtr,
            &sphCoefficients[0]);
        const float3 grad = radianceFromSpHBwd(sphEvalDegree, &sphCoefficients[0], rayDirection, weight, radianceGrad, (float3*)&particleRadianceGradPtr[particleIdx * SPH_MAX_NUM_COEFFS * 3]);

        // >>> rayRadiance = accumulatedRayRad + weigth * rayRad + (1-galpha)*transmit * residualRayRad
        const float3 rayRad = weight * grad;
        radiance += rayRad;
        const float3 residualRayRad = fmaxf((nextTransmit <= minTransmittance ? make_float3(0) : (integratedRadiance - radiance) / nextTransmit),
                                            make_float3(0));
       atomicAdd(
            &params.grad_opacity[particleIdx],
            gres * (/*galphaRayHitGrd + galphaRayDnsGrd +*/ transmittance * (grad.x - residualRayRad.x) * radianceGrad.x +
                    transmittance * (grad.y - residualRayRad.y) * radianceGrad.y +
                    transmittance * (grad.z - residualRayRad.z) * radianceGrad.z));

       const float gresGrd =
            particleDensity * (/*galphaRayHitGrd +  galphaRayDnsGrd +*/ transmittance * (grad.x - residualRayRad.x) * radianceGrad.x +
                               transmittance * (grad.y - residualRayRad.y) * radianceGrad.y +
                               transmittance * (grad.z - residualRayRad.z) * radianceGrad.z);

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> gres = exp(-0.0555 * grayDist * grayDist)
        // ===> d_gres / d_grayDist = -0.111 * grayDist * exp(-0.555 * grayDist * grayDist)
        //                          = -0.111 * grayDist * gres
        const float grayDistGrd = -0.5f*gres*gresGrd;// particleResponseGrd<PARTICLE_KERNEL_DEGREE>(grayDist, gres, gresGrd);

        float3 grdGrd, groGrd;
        {
            const float3 gcrod = cross(grd, gro);

            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // ---> grayDist = dot(gcrod, gcrod)
            //               = gcrod.x^2 + gcrod.y^2 + gcrod.z^2
            // ===> d_grayDist / d_gcrod = 2*gcrod
            const float3 gcrodGrd = 2 * gcrod * grayDistGrd;

            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // ---> gcrod = cross(grd, gro)
            // ---> gcrod.x = grd.y * gro.z - grd.z * gro.y
            // ---> gcrod.y = grd.z * gro.x - grd.x * gro.z
            // ---> gcrod.z = grd.x * gro.y - grd.y * gro.x
            grdGrd = make_float3(gcrodGrd.z * gro.y - gcrodGrd.y * gro.z,
                                 gcrodGrd.x * gro.z - gcrodGrd.z * gro.x,
                                 gcrodGrd.y * gro.x - gcrodGrd.x * gro.y);
            groGrd = make_float3(gcrodGrd.y * grd.z - gcrodGrd.z * grd.y,
                                 gcrodGrd.z * grd.x - gcrodGrd.x * grd.z,
                                 gcrodGrd.x * grd.y - gcrodGrd.y * grd.x);
            //groGrd *= 0.f;
        }

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> gro = (1/gscl)*gposcr
        // ===> d_gro / d_gscl = -gposcr/(gscl*gscl)
        // ===> d_gro / d_gposcr = (1/gscl)
        const float3 gsclGrdGro = make_float3((-gposcr.x / (gscl.x * gscl.x)),
                                              (-gposcr.y / (gscl.y * gscl.y)),
                                              (-gposcr.z / (gscl.z * gscl.z))) *
                                  (groGrd/* + groRayHitGrd*/);
        const float3 gposcrGrd = giscl * (groGrd/* + groRayHitGrd*/);

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> gposcr = matmul(gposc, grotMat)
        // ===> d_gposcr / d_gposc = matmul_bw_vec(grotMat)
        // ===> d_gposcr / d_grotmat = matmul_bw_mat(gposc)
        const float3 gposcGrd     = matmul_bw_vec(particleInvRotation, gposcrGrd);
        const float4 grotGrdPoscr = matmul_bw_quat(gposc, gposcrGrd, grot);

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> gposc = rayOri - gpos
        // ===> d_gposc / d_gpos = -1
        const float3 rayMoGPosGrd = -gposcGrd;
        atomicAdd(&params.grad_xyz[particleIdx].x, rayMoGPosGrd.x);
        atomicAdd(&params.grad_xyz[particleIdx].y, rayMoGPosGrd.y);
        atomicAdd(&params.grad_xyz[particleIdx].z, rayMoGPosGrd.z);

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> grd = safe_normalize(grdu)
        // ===> d_grd / d_grdu = safe_normalize_bw(grd)
        const float3 grduGrd = safe_normalize_bw(grdu, grdGrd /* + grdRayHitGrd*/);

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> grdu = (1/gscl)*rayDirR
        // ===> d_grdu / d_gscl = -rayDirR/(gscl*gscl)
        // ===> d_grdu / d_rayDirR = (1/gscl)
        atomicAdd(&params.grad_scale[particleIdx].x, /*gsclRayHitGrd.x +*/ gsclGrdGro.x + (-rayDirR.x / (gscl.x * gscl.x)) * grduGrd.x);
        atomicAdd(&params.grad_scale[particleIdx].y, /*gsclRayHitGrd.y +*/ gsclGrdGro.y + (-rayDirR.y / (gscl.y * gscl.y)) * grduGrd.y);
        atomicAdd(&params.grad_scale[particleIdx].z, /*gsclRayHitGrd.z +*/ gsclGrdGro.z + (-rayDirR.z / (gscl.z * gscl.z)) * grduGrd.z);
        const float3 rayDirRGrd = giscl * grduGrd;

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> rayDirR = matmul(rayDir, grotMat)
        // ===> d_rayDirR / d_grotmat = matmul_bw_mat(rayDir, grotMat)
        const float4 grotGrdRayDirR = matmul_bw_quat(rayDirection, rayDirRGrd, grot);
        atomicAdd(&params.grad_rotation[particleIdx].x, grotGrdPoscr.x + grotGrdRayDirR.x);
        atomicAdd(&params.grad_rotation[particleIdx].y, grotGrdPoscr.y + grotGrdRayDirR.y);
        atomicAdd(&params.grad_rotation[particleIdx].z, grotGrdPoscr.z + grotGrdRayDirR.z);
        atomicAdd(&params.grad_rotation[particleIdx].w, grotGrdPoscr.w + grotGrdRayDirR.w);

        transmittance = nextTransmit;
    }
}


extern "C" __global__ void __raygen__rg() {
    const uint3 idxy = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const int id = idxy.x + idxy.y*dim.x;

    float3 rayOrigin    = params.ray_origins[id];
    float3 rayDirection = params.ray_directions[id];

    float3 rayRadiance     = make_float3(0.0f);
    float rayTransmittance = 1.0f;
    float rayHitDistance   = 0.f;

    //float2 minMaxT       = intersectAABB(params.aabb, rayOrigin, rayDirection);
    float2 minMaxT = {0.f,1e20f};
    constexpr float epsT = 1e-9;

    if(!params.compute_grad){

        float rayHitCount = 0.f;

        float rayLastHitDistance = fmaxf(0.0f, minMaxT.x - epsT);
        RayPayload rayPayload;

        while ((rayLastHitDistance <= minMaxT.y) && (rayTransmittance > params.minTransmittance)) {
            trace(rayPayload, rayOrigin, rayDirection, rayLastHitDistance + epsT, minMaxT.y + epsT);
            if (rayPayload[0].particleId == RayHit::InvalidParticleId) {
                break;
            }

    #pragma unroll
            for (int i = 0; i < 16; i++) {
                const RayHit rayHit = rayPayload[i];

                if ((rayHit.particleId != RayHit::InvalidParticleId) && (rayTransmittance > params.minTransmittance)) {
                    
                    bool accepted = processHit(
                        rayOrigin,
                        rayDirection,
                        rayHit.particleId,
                        0.0113f,
                        1/255.f,
                        &rayTransmittance,
                        &rayRadiance
                    );
                    rayHitCount += accepted? 1.f : 0.f;

                    rayLastHitDistance = fmaxf(rayLastHitDistance, rayHit.distance);
                }
            }
        }

        params.radiance[id] = rayRadiance;
        params.transmittance[id] = rayTransmittance;
        params.distance[id] = rayLastHitDistance;
        params.debug_map_0[id].x = rayHitCount;
    }else{

        float3 rayIntegratedRadiance     = params.radiance[id];
        float rayIntegratedTransmittance = params.transmittance[id];
        float rayMaxHitDistance          = params.distance[id];
    
        float3 rayRadianceGrad     = params.dL_dC[id];
        //float rayTransmittanceGrad = -1.0f * params.rayDensityGrad[idx.z][idx.y][idx.x][0];
        //float rayHitDistanceGrad   = params.rayHitDistanceGrad[idx.z][idx.y][idx.x][0];

        float startT     = fmaxf(0.0f, minMaxT.x - epsT);
        const float endT = fminf(rayMaxHitDistance, minMaxT.y) + epsT;
    
        float3 rayRadiance     = make_float3(0.f);
        float rayTransmittance = 1.f;
        float rayHitDistance   = 0.f;
    
        RayPayload rayPayload;
    
        while (startT < endT) {
            trace(rayPayload, rayOrigin, rayDirection, startT + epsT, endT);
            if (rayPayload[0].particleId == RayHit::InvalidParticleId) {
                break;
            }
    
    #pragma unroll
            for (int i = 0; i < 16; i++) {
                const RayHit rayHit = rayPayload[i];
    
                if (rayHit.particleId != RayHit::InvalidParticleId) {
                    processHitBwd(
                        rayOrigin,
                        rayDirection,
                        rayHit.particleId,
                        (float*)params.gs_sh,
                        (float*)params.grad_sh,
                        0.0113f,
                        1/255.f,
                        params.minTransmittance,
                        params.sh_deg,
                        rayIntegratedTransmittance,
                        rayTransmittance,
                        //rayTransmittanceGrad,
                        rayIntegratedRadiance,
                        rayRadiance,
                        rayRadianceGrad);
    
                    startT = fmaxf(startT, rayHit.distance);
                }
            }
        }
    }
}


#define compareAndSwapHitPayloadValue(hit, g_id, g_distance, s_id, s_distance)                      \
    {                                                                             \
        const float distance = __uint_as_float(g_distance()); \
        if (hit.distance < distance) {                                            \
            s_distance(__float_as_uint(hit.distance));        \
            const uint32_t id = g_id();                       \
            s_id(hit.particleId);                             \
            hit.distance   = distance;                                            \
            hit.particleId = id;                                                  \
        }                                                                         \
    }

extern "C" __global__ void __anyhit__fwd() {

    optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0);

    RayHit hit = RayHit{optixPrimitiveIndex(), optixGetRayTmax()};

    if (hit.distance < __uint_as_float(optixGetPayload_31())) {

        compareAndSwapHitPayloadValue(hit, optixGetPayload_0,  optixGetPayload_1, optixSetPayload_0,  optixSetPayload_1);
        compareAndSwapHitPayloadValue(hit, optixGetPayload_2,  optixGetPayload_3, optixSetPayload_2,  optixSetPayload_3);
        compareAndSwapHitPayloadValue(hit, optixGetPayload_4,  optixGetPayload_5, optixSetPayload_4,  optixSetPayload_5);
        compareAndSwapHitPayloadValue(hit, optixGetPayload_6,  optixGetPayload_7, optixSetPayload_6,  optixSetPayload_7);
        compareAndSwapHitPayloadValue(hit, optixGetPayload_8,  optixGetPayload_9, optixSetPayload_8,  optixSetPayload_9);
        compareAndSwapHitPayloadValue(hit, optixGetPayload_10, optixGetPayload_11, optixSetPayload_10, optixSetPayload_11);
        compareAndSwapHitPayloadValue(hit, optixGetPayload_12, optixGetPayload_13, optixSetPayload_12, optixSetPayload_13);
        compareAndSwapHitPayloadValue(hit, optixGetPayload_14, optixGetPayload_15, optixSetPayload_14, optixSetPayload_15);
        compareAndSwapHitPayloadValue(hit, optixGetPayload_16, optixGetPayload_17, optixSetPayload_16, optixSetPayload_17);
        compareAndSwapHitPayloadValue(hit, optixGetPayload_18, optixGetPayload_19, optixSetPayload_18, optixSetPayload_19);
        compareAndSwapHitPayloadValue(hit, optixGetPayload_20, optixGetPayload_21, optixSetPayload_20, optixSetPayload_21);
        compareAndSwapHitPayloadValue(hit, optixGetPayload_22, optixGetPayload_23, optixSetPayload_22, optixSetPayload_23);
        compareAndSwapHitPayloadValue(hit, optixGetPayload_24, optixGetPayload_25, optixSetPayload_24, optixSetPayload_25);
        compareAndSwapHitPayloadValue(hit, optixGetPayload_26, optixGetPayload_27, optixSetPayload_26, optixSetPayload_27);
        compareAndSwapHitPayloadValue(hit, optixGetPayload_28, optixGetPayload_29, optixSetPayload_28, optixSetPayload_29);
        compareAndSwapHitPayloadValue(hit, optixGetPayload_30, optixGetPayload_31, optixSetPayload_30, optixSetPayload_31);

        // ignore all inserted hits, expect if the last one
        if (__uint_as_float(optixGetPayload_31()) > optixGetRayTmax()) {
            optixIgnoreIntersection();
        }
    }
}


extern "C" __global__ void __anyhit__bwd() {

    optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_1);
}
