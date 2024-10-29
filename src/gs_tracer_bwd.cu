#include <optix.h>

#include "optix_types.h"
#include "utils/Matrix.h"

#include <vector>
#include <float.h>

using namespace util;

extern "C" {
__constant__ Params params;
constexpr float eps = 1e-6;
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

__device__ float3 compute_radiance(unsigned int gs_id, const float3 &ray_origin){
    const uint3 idx = optixGetLaunchIndex();

    //const float3 dir = -params.ray_directions[idx.x];
    const float3 mu = params.gs_xyz[gs_id];
    const float3 dir = normalize(mu-ray_origin);

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

struct Hit{
    int id;
    float thit;
    float resp;
};

struct HitBwd{
    int id;
    float thit;
    float resp;
    Matrix3x3 inv_RS;
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

__device__ __forceinline__ Matrix3x3 construct_inv_RS(const float4& rot, const float3& s){
    Matrix3x3 R = construct_rotation(rot).transpose();
    R.setRow(0,R.getRow(0)/(s.x+eps));
    R.setRow(1,R.getRow(1)/(s.y+eps));
    R.setRow(2,R.getRow(2)/(s.z+eps));
    return R;
}

__device__ void compute_response(
    const float3& o, const float3& d, const float3& mu,
    const float opacity, const Matrix3x3& inv_RS,
    float& resp, float& tmax){

    float3 og = inv_RS*(mu-o);
    float3 dg = inv_RS*d;
    tmax = dot(og,dg)/(dot(dg,dg)+eps);
    float3 c_samp = o+tmax*d;
    float3 v = inv_RS*(c_samp-mu);
    resp = opacity*exp(-dot(v,v));
}

__device__ __forceinline__ void compute_grad(
    const float &prev_acc_trans, const float3 &acc_rad,
    const float3 &particle_rad, const float3 &full_rad,
    const float3 &c_samp, const float3 &pos, const float &opacity,
    const Matrix3x3 &inv_RS,
    float3 &grad_pos, float &grad_opac){

    const float3 d_rad_resp = prev_acc_trans*(particle_rad-full_rad+acc_rad);
    const float3 x = c_samp-pos;
    const float3 v = inv_RS*x;
    const float g = exp(-dot(v,v));
    const float3 d_resp_pos = 2*opacity*g*(inv_RS*inv_RS.transpose())*x;
    const float cs = d_rad_resp.x + d_rad_resp.y + d_rad_resp.z;
    grad_pos = cs*d_resp_pos;
    grad_opac = cs*g;
}


struct Acc{
    float3 radiance;
    float transmittance;
};


__device__ __forceinline__ void add_samp(Acc& acc, const float3& rad, const Hit& chit){
    acc.radiance += rad*chit.resp*acc.transmittance;
    acc.transmittance *= (1.-chit.resp);
}

__device__ __forceinline__ void add_grad(const Acc& acc, const float3& rad, const Acc& acc_full,
                                        int chit_id, const float3& c_samp){
    const Matrix3x3 inv_RS = construct_inv_RS(params.gs_rotation[chit_id],params.gs_scaling[chit_id]);
    float3 grad_pos;
    float grad_opac;
    compute_grad(acc.transmittance,acc.radiance,rad,acc_full.radiance,
        c_samp,
        params.gs_xyz[chit_id],
        params.gs_opacity[chit_id],
        inv_RS,
        grad_pos,grad_opac
    );
    atomicAdd(&params.grad_opacity[chit_id],grad_opac);
    atomicAdd(&params.grad_xyz[chit_id].x,grad_pos.x);
    atomicAdd(&params.grad_xyz[chit_id].y,grad_pos.y);
    atomicAdd(&params.grad_xyz[chit_id].z,grad_pos.z);
}


constexpr int chunk_size = 512;

constexpr int num_recasts = 2;
constexpr int hits_max_capacity = chunk_size*num_recasts;

constexpr int triagPerParticle = 20;
constexpr float Tmin = 0.001;
constexpr float respMin = 0.01f;

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

    Acc acc{};
    acc.radiance = make_float3(0.f);
    acc.transmittance = 1.f;
    unsigned int* uip_acc = reinterpret_cast<unsigned int *>(&acc);

    constexpr float max_dist = 1e16f; 
    float min_dist = 0.f;
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
            OPTIX_RAY_FLAG_NONE,
            0,  // SBT offset   -- See SBT discussion
            2,  // SBT stride   -- See SBT discussion
            0,  // missSBTIndex -- See SBT discussion
            uip_acc[0],uip_acc[1],uip_acc[2],uip_acc[3],
            p_hits[0],p_hits[1],hits_size,n_hits_capacity);

        if(n_hits_capacity == hits_capacity){
            break;   
        }
        hits_capacity = n_hits_capacity;
        hits_size = 0;
        acc.radiance = make_float3(0.f);
        acc.transmittance = 1.f;
    }
    while(hits_size!=0 && acc.transmittance > Tmin){
        const Hit& chit = hits[0];
        const float3 rad = compute_radiance(chit.id,ray_origin);
        add_samp(acc,rad,chit);
        hitq_pop(hits,hits_size);
    }
    params.radiance[id] = acc.radiance;
    params.transmittance[id] = acc.transmittance;

    if(params.compute_grad){

        Acc acc_bwd{};
        acc_bwd.radiance = make_float3(0.f);
        acc_bwd.transmittance = 1.f;
        unsigned int* uip_acc_bwd = reinterpret_cast<unsigned int *>(&acc_bwd);

        optixTrace(
            OPTIX_PAYLOAD_TYPE_ID_1,
            params.handle,
            ray_origin,
            ray_direction,
            min_dist,                      // Min intersection distance
            max_dist,                     // Max intersection distance
            0.0f,                      // rayTime -- used for motion blur
            OptixVisibilityMask(255),  // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            1,  // SBT offset   -- See SBT discussion
            2,  // SBT stride   -- See SBT discussion
            0,  // missSBTIndex -- See SBT discussion
            uip_acc_bwd[0],uip_acc_bwd[1],uip_acc_bwd[2],uip_acc_bwd[3],
            p_hits[0],p_hits[1],hits_size,
            uip_acc[0],uip_acc[1],uip_acc[2],uip_acc[3]
            );
        while(hits_size!=0 && acc_bwd.transmittance > Tmin){
            const Hit& chit = hits[0];
            const float3 rad = compute_radiance(chit.id,ray_origin);
            acc.radiance += rad*chit.resp*acc.transmittance;
            add_grad(acc_bwd,rad,acc,chit.id,ray_origin+ray_direction*chit.thit);
            acc.transmittance *= (1.-chit.resp);
            hitq_pop(hits,hits_size);
        }
    }
}

extern "C" __global__ void __anyhit__fwd() {

    atomicAdd(params.num_its,1ull);

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

    if(hitq_size == hitq_capacity){
        const Hit &chit = hitq[0];
        const float3 rad = compute_radiance(chit.id,optixGetWorldRayOrigin());
        add_samp(acc,rad,chit);

        optixSetPayload_0(__float_as_uint(acc.radiance.x));
        optixSetPayload_1(__float_as_uint(acc.radiance.y));
        optixSetPayload_2(__float_as_uint(acc.radiance.z));
        optixSetPayload_3(__float_as_uint(acc.transmittance));

        if(acc.transmittance < Tmin){
            return;
        }
        if(hitq_capacity != hits_max_capacity && optixGetRayTmax() < chit.thit){
            printf("recast %d\n",hitq_capacity+chunk_size);
            optixSetPayload_7(hitq_capacity+chunk_size);
            return;
        }
        hitq_pop(hitq,hitq_size);
        optixSetPayload_6(hitq_size);
    }


    const unsigned int prim_id = optixGetPrimitiveIndex();
    float3 normal = params.gs_normals[prim_id];
    if(dot(normal,optixGetWorldRayDirection())>0.){
        optixIgnoreIntersection();
        return;
    }

    const unsigned int hit_id = optixGetPrimitiveIndex()/triagPerParticle;
    float resp,thit; 
    compute_response(optixGetWorldRayOrigin(),
                    optixGetWorldRayDirection(),
                    params.gs_xyz[hit_id],
                    params.gs_opacity[hit_id],
                    construct_inv_RS(params.gs_rotation[hit_id],params.gs_scaling[hit_id]),
                    resp,thit);
    
    if(resp < respMin){
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

extern "C" __global__ void __anyhit__bwd() {

    atomicAdd(params.num_its_bwd,1ull);

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

    const unsigned int prim_id = optixGetPrimitiveIndex();
    float3 normal = params.gs_normals[prim_id];
    if(dot(normal,direction)>0.){
        optixIgnoreIntersection();
        return;
    }

    if(hitq_size == chunk_size){
        const Hit &chit = hitq[0];

        const float3 rad = compute_radiance(chit.id,origin);

        acc.radiance += rad*chit.resp*acc.transmittance;
        add_grad(acc,rad,acc_full,chit.id,origin+direction*chit.thit);
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
    compute_response(origin,
                    direction,
                    params.gs_xyz[hit_id],
                    params.gs_opacity[hit_id],
                    inv_RS,
                    resp,thit);
    
    if(resp < respMin){
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
