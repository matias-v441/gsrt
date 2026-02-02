#pragma once
#include "preprocessor.h"
#include "vec_math.h"
#include "Matrix.h"
#include "geom.h"

namespace util::gs_tracing
{

#ifdef __CUDA_ARCH__
	#define ENABLE_DEVICE __device__
#else
	#define ENABLE_DEVICE
#endif
ENABLE_DEVICE const float SH_C0 = 0.28209479177387814f;
ENABLE_DEVICE const float SH_C1 = 0.4886025119029199f;
ENABLE_DEVICE const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
ENABLE_DEVICE const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,		
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

HOSTDEVICE static void computeResponse(const float3* xyz, const float4* rotation,
	 const float3* scaling, const float* opacity, unsigned int gs_id,
     const float3 o, const float3 d, float& resp, float& tmax){

    const float3 mu = xyz[gs_id];
    const float3 s = scaling[gs_id];
    Matrix3x3 R = geom::construct_rotation(rotation[gs_id]).transpose();
    constexpr float eps = 1e-6;
    R.setRow(0,R.getRow(0)/(s.x+eps));
    R.setRow(1,R.getRow(1)/(s.y+eps));
    R.setRow(2,R.getRow(2)/(s.z+eps));
    float3 og = R*(mu-o);
    float3 dg = R*d;
    tmax = dot(og,dg)/(dot(dg,dg)+eps);
    float3 samp = o+tmax*d;
    float3 x = R*(samp-mu);
    resp = opacity[gs_id]*exp(-dot(x,x));
}

HOSTDEVICE static float3 computeRadiance(const float3* sh, int sh_deg, const float3 mu, const float3 &ray_origin){

    //const float3 dir = -params.ray_directions[idx.x];
    const float3 dir = normalize(mu-ray_origin);

    const int deg = sh_deg;

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
	
} // namespace gsrt::util