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

struct Hit{
    float thit = FLT_MAX;
    int primId;
    float resp;
};

constexpr int max_hit_id = 31;
constexpr int max_chunk_size = 512;

constexpr int triagPerParticle = 20;
constexpr float Tmin = 0.001;
constexpr float respMin = 0.01f;

__device__ float3 computeRadiance(unsigned int gs_id, const float3& ray_origin);

extern "C" __global__ void __raygen__rg() {

    // Lookup our location within the launch grid
    const uint3 idxy = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const int id = idxy.x + idxy.y*dim.x;

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    const float3 ray_origin = params.ray_origins[id];
    const float3 ray_direction = params.ray_directions[id];

    Hit hits[max_chunk_size];

    unsigned int p_hits[2];
    Hit* hits_ptr = reinterpret_cast<Hit*>(hits);
    memcpy(p_hits, &hits_ptr, sizeof(void*));

    //for(int i = 0; i < max_chunk_size; ++i){
    //    hits[i].thit = FLT_MAX;
    //}
    
    int hit_ids[2][max_chunk_size];
    for(int i = 0; i < max_chunk_size; ++i){
        hit_ids[0][i] = i;
        hit_ids[1][i] = i;
    }

    Payload payload{};
    payload.radiance = make_float3(0.f);
    payload.transmittance = 1.f;
    unsigned int* p = reinterpret_cast<unsigned int *>(&payload);

    float3 radiance = make_float3(0.f);
    float T = 1.f;
    constexpr float scene_max_dist = 100.f;//1e16f; 
    constexpr float delta_dist = 10.f;
    float min_dist = 0.f;
    int chunk_id = 0;
    int k = 0;
    while(T > Tmin && scene_max_dist > min_dist){
        float max_dist = min_dist + delta_dist;
        optixTrace(
            params.handle,
            ray_origin,
            ray_direction,
            min_dist,                      // Min intersection distance
            max_dist,                     // Max intersection distance
            0.0f,                      // rayTime -- used for motion blur
            OptixVisibilityMask(255),  // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,  // SBT offset   -- See SBT discussion
            1,  // SBT stride   -- See SBT discussion
            0,  // missSBTIndex -- See SBT discussion
            p_hits[0],p_hits[1],p[2],p[3]);
        k++;
        
        for(int i = 0; i < 4; ++i){
            int nchunks = 1<<(4-i);
            int chunk_size = max_chunk_size/nchunks;
            int id_in = i&1;
            int id_out = !id_in;
            for(int j = 0; j < nchunks; j += 2){
                int sl = chunk_size*j;
                int sr = sl+chunk_size;
                int el = sr;
                int er = sr+chunk_size;
                if(hits[hit_ids[id_in][sr-1]].thit < hits[hit_ids[id_in][sr]].thit)
                    continue;
                for(int k = sl; k < er; ++k){
                    if(sl == el){
                        hit_ids[id_out][k] = hit_ids[id_in][sr++];
                        continue;
                    }
                    if(sr == er){
                        hit_ids[id_out][k] = hit_ids[id_in][sl++];
                        continue;
                    }
                    if(hits[hit_ids[id_in][sr]].thit < hits[hit_ids[id_in][sl]].thit){
                        hit_ids[id_out][k] = hit_ids[id_in][sr++];
                    }else{
                        hit_ids[id_out][k] = hit_ids[id_in][sl++];
                    }
                }
            }
        }
        //Hit hits_std[max_chunk_size];
        //for(int i = 0; i < max_chunk_size; ++i){
        //    hits_std[i] = hits[hit_ids[4][i]];
        //}
        float last_thit = FLT_MAX;
        for(int i = 0; i < max_chunk_size; ++i){
            // if(hits[i].thit == FLT_MAX // && i == chunk_size-1
            // ){
            //     if(chunk_id == 0){ //miss
            //         //radiance = make_float3(1.,1.,1.);
            //     }
            //     goto end_while;
            // }
            int id = hit_ids[0][i];
            if(hits[id].thit == FLT_MAX) break;
            last_thit = hits[id].thit;
            if(hits[id].resp < respMin) continue;
            radiance += hits[id].resp*T*computeRadiance(hits[id].primId,ray_origin);
            T *= (1.-hits[id].resp);
            hits[id].thit = FLT_MAX;
        }

        //for(int i = 0; i < max_chunk_size; ++i){
        //    hits[i].thit = FLT_MAX;
        //}
        min_dist = min(last_thit, max_dist);
        chunk_id++;
        
    } end_while:
    //printf("%d\n",k);
    params.radiance[id] = radiance;
    params.transmittance[id] = T;
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

__device__ void computeResponse(unsigned int gs_id, float& resp, float& tmax){
    const float3 o = optixGetWorldRayOrigin();
    const float3 d = optixGetWorldRayDirection();
    const float3 mu = params.gs_xyz[gs_id];
    const float3 s = params.gs_scaling[gs_id];
    Matrix3x3 R = construct_rotation(params.gs_rotation[gs_id]).transpose();
    R.setRow(0,R.getRow(0)/(s.x+eps));
    R.setRow(1,R.getRow(1)/(s.y+eps));
    R.setRow(2,R.getRow(2)/(s.z+eps));
    float3 og = R*(mu-o);
    float3 dg = R*d;
    tmax = dot(og,dg)/(dot(dg,dg)+eps);
    float3 samp = o+tmax*d;
    float3 x = R*(samp-mu);
    resp = params.gs_opacity[gs_id]*exp(-dot(x,x));
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

__device__ float3 computeRadiance(unsigned int gs_id, const float3 &ray_origin){
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

extern "C" __global__ void __anyhit__ms() {


    unsigned int p_hits[2];
    p_hits[0] = optixGetPayload_0();
    p_hits[1] = optixGetPayload_1();
    Hit* hits;
    memcpy(&hits, p_hits, sizeof(p_hits));

    (*params.num_its)++;

    const unsigned int hitParticle = optixGetPrimitiveIndex()/triagPerParticle;

    const float3 d = optixGetWorldRayDirection();
    const float3 its = optixGetWorldRayOrigin()+optixGetRayTmax()*d;
    
    const unsigned int prim_id = optixGetPrimitiveIndex();
    float3 normal = params.gs_normals[prim_id];
    if(dot(normal,d)>0.)
    {
        optixIgnoreIntersection();
        return;
    }

    //if(optixGetRayTmax() > hits[max_chunk_size-1].thit){
    //    optixIgnoreIntersection();
    //    return;
    //}

    float resp,thit; 
    computeResponse(hitParticle,resp,thit);

    //if(thit > hits[max_chunk_size-1].thit){
    //    optixIgnoreIntersection();
    //    return;
    //}
    //
    if(resp < respMin){
        optixIgnoreIntersection();
        return;
    }


    Hit hit;
    hit.primId = hitParticle;
    hit.resp = resp;
    //hit.thit = optixGetRayTmax();
    hit.thit = thit;
    int last_id = 0;
    int end = 32;
    constexpr int n = max_chunk_size/32;
    for(int i = 2; i <= n; ++i){
        if(hits[i*32-1].thit == FLT_MAX){
            end = i*32;
            break;
        }
    }
    int start = end-32;
    for(int i = 0; i < 32; ++i){
        int id = start+i;
        if(hit.thit < hits[id].thit){
            Hit h = hit;
            hit = hits[id];
            hits[id] = h;
        }
        last_id = i;
        if(hit.thit == FLT_MAX) break;
    }
    const uint3 idxy = optixGetLaunchIndex();
    
    //optixIgnoreIntersection();
    if(hits[max_chunk_size-1].thit == FLT_MAX){
        //printf("ign %d %d %d %f\n",idxy.x,idxy.y,last_id,hits[last_id-1].thit - optixGetRayTmin());
        optixIgnoreIntersection();
    }else{
        //printf("full %d %d %d %f\n",idxy.x,idxy.y,last_id,hits[max_chunk_size-1].thit - optixGetRayTmin());
    }
}