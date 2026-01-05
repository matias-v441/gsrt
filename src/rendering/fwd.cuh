#pragma once
#include "utils/Matrix.h"
using namespace util;

constexpr float eps = 1e-6;
constexpr float min_kernel_density = 0.0113f;
constexpr float min_alpha = 1/255.f; 
constexpr float max_alpha = 0.99f;

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

__device__ bool compute_response_naive(
    const float3& o, const float3& d, const float3& mu,
    const float opacity, const float4& rotation, const float3& scaling,
    float& alpha, float& tmax){
	const Matrix3x3 inv_RS = construct_inv_RS(rotation, scaling);
    float3 og = inv_RS*(mu-o);
    float3 dg = inv_RS*d;
    tmax = dot(og,dg)/max(eps,dot(dg,dg));
    //tmax = dot(og,dg)/(eps+dot(dg,dg));
    float3 c_samp = o+tmax*d;
    float3 v = inv_RS*(c_samp-mu);
    //float resp = exp(-.5f*dot(v,v));
    float resp = exp(-.5f*tmax);
    //if(resp < min_kernel_density) return false;
    alpha = min(max_alpha,opacity*resp);

    //return (alpha > min_alpha) && (resp > min_kernel_density);
	return alpha > min_alpha;
}

static __device__ inline float3 safe_normalize(float3 v) {
    const float l = v.x * v.x + v.y * v.y + v.z * v.z;
    return l > 0.0f ? (v * rsqrtf(l)) : v;
}

__device__ bool compute_response(
    const float3& o, const float3& d, const float3& mu,
    const float opacity, const float4& rotation, const float3& scaling,
    float& alpha, float& tmax){

	Matrix3x3 RT = construct_rotation(rotation).transpose();
	const float3 iscl = make_float3(1/scaling.x,1/scaling.y,1/scaling.z);
	const float3 og = iscl*(RT*(mu-o));
	const float3 dg_unorm = iscl*(RT*d);
	const float3 dg = safe_normalize(dg_unorm);
	const float3 dg_x_og = cross(dg,og);
	tmax = dot(og,dg_unorm)/max(eps,dot(dg_unorm,dg_unorm));
    float G = expf(-.5f*dot(dg_x_og,dg_x_og));
    alpha = min(max_alpha, opacity*G);
    return (alpha > min_alpha) && (G > min_kernel_density);
	//return alpha > min_alpha;
	//return alpha > min_kernel_density;
}

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

__device__ void compute_radiance(const float3* shs, int sh_deg, unsigned int gs_id, const float3 &ray_origin,
     const float3& ray_direction, float3& rad, bool *clamped){

    const float3 dir = ray_direction;

    const float3* sh = shs + gs_id*16;

	float3 result = SH_C0 * sh[0];

	if (sh_deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (sh_deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (sh_deg > 2)
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

	clamped[0] = (result.x < 0);
	clamped[1] = (result.y < 0);
	clamped[2] = (result.z < 0);

	rad = {max(result.x,0.f),max(result.y,0.f),max(result.z,0.f)};
}
