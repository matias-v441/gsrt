#pragma once
#include "fwd.cuh"

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__device__ __forceinline__ void atomicAdd_float3(float3 &acc, const float3 &val){
    atomicAdd(&acc.x,val.x);
    atomicAdd(&acc.y,val.y);
    atomicAdd(&acc.z,val.z);
}

__device__ void compute_radiance_bwd_at(const float3* shs, int sh_deg, float3* grad_sh,
	 int gs_id, const float3& ray_origin, const float3& ray_direction,
    const float3& dL_dcolor, float3& grad_xyz, const bool* clamped)
{
    // atomicAdd_float3(params.grad_color[gs_id], dL_dcolor);
    // grad_xyz = make_float3(0.f);
    // if(params.sh_deg == -1) return;

	// same as forward -----

    //const float3 dir = -params.ray_directions[idx.x];
    //const float3 mu = params.gs_xyz[gs_id];
    //const float3 dir = normalize(mu-ray_origin);
    const float3 dir = ray_direction;

    const float3* sh = shs + gs_id*16;
    const int deg = sh_deg;
    // ---------------------

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	float3 dL_dRGB = dL_dcolor;
	dL_dRGB.x *= clamped[0] ? 0.f : 1.f;
	dL_dRGB.y *= clamped[1] ? 0.f : 1.f;
	dL_dRGB.z *= clamped[2] ? 0.f : 1.f;

	float3 dRGBdx{0.f, 0.f, 0.f};
	float3 dRGBdy{0.f, 0.f, 0.f};
	float3 dRGBdz{0.f, 0.f, 0.f};
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
    float3 *dL_dsh = grad_sh + gs_id*16;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;

	//dL_dsh[0] = dRGBdsh0 * dL_dRGB;
    atomicAdd_float3(dL_dsh[0], dRGBdsh0 * dL_dRGB);

	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;

        // If color function is view dependant we might accumulate grad_color for each view instead
        // if it is ray dependant we should accumulate grad_sh directly

		//dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		//dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		//dL_dsh[3] = dRGBdsh3 * dL_dRGB;

        atomicAdd_float3(dL_dsh[1], dRGBdsh1 * dL_dRGB);
		atomicAdd_float3(dL_dsh[2], dRGBdsh2 * dL_dRGB);
		atomicAdd_float3(dL_dsh[3], dRGBdsh3 * dL_dRGB);

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			//dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			//dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			//dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			//dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			//dL_dsh[8] = dRGBdsh8 * dL_dRGB;

            atomicAdd_float3(dL_dsh[4], dRGBdsh4 * dL_dRGB);
			atomicAdd_float3(dL_dsh[5], dRGBdsh5 * dL_dRGB);
			atomicAdd_float3(dL_dsh[6], dRGBdsh6 * dL_dRGB);
			atomicAdd_float3(dL_dsh[7], dRGBdsh7 * dL_dRGB);
			atomicAdd_float3(dL_dsh[8], dRGBdsh8 * dL_dRGB);

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				//dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				//dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				//dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				//dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				//dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				//dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				//dL_dsh[15] = dRGBdsh15 * dL_dRGB;

                atomicAdd_float3(dL_dsh[9], dRGBdsh9 * dL_dRGB);
				atomicAdd_float3(dL_dsh[10], dRGBdsh10 * dL_dRGB);
				atomicAdd_float3(dL_dsh[11], dRGBdsh11 * dL_dRGB);
				atomicAdd_float3(dL_dsh[12], dRGBdsh12 * dL_dRGB);
				atomicAdd_float3(dL_dsh[13], dRGBdsh13 * dL_dRGB);
				atomicAdd_float3(dL_dsh[14], dRGBdsh14 * dL_dRGB);
				atomicAdd_float3(dL_dsh[15], dRGBdsh15 * dL_dRGB);

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	float3 dL_ddir{dot(dRGBdx, dL_dRGB), dot(dRGBdy, dL_dRGB), dot(dRGBdz, dL_dRGB)};

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir.x, dir.y, dir.z },
                               float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	grad_xyz = make_float3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

struct Acc{
    float3 radiance;
    float transmittance;
};

__device__ __forceinline__ void add_grad_at(
    const float4* gs_rotation, const float3* gs_scaling, const float3* gs_xyz, const float* gs_opacity,
    const float3* shs, int sh_deg, float3* grad_sh,
    bool white_background,
    float3 dL_dC,
    float4* grad_rotation, float3* grad_scaling, float3* grad_xyz, float* grad_opacity,
    const Acc& acc, const float3& rad, const Acc& acc_full,
    int chit_id, const float3& csamp,
    const float resp, 
    const float3 ray_origin,
    const float3 ray_direction,
    const bool* clamped
    ){

    //atomicAdd(params.num_its_bwd,1ull);

    const float4 quat = gs_rotation[chit_id];
    const float3 scale = gs_scaling[chit_id];
    const float3 pos = gs_xyz[chit_id];
    const float opacity = gs_opacity[chit_id];

    float3 background = white_background? make_float3(1.f) : make_float3(0.f);
    const float3 dC_dresp = acc.transmittance*rad - (acc_full.radiance - acc.radiance)/max(eps,1.f-resp)
                            - background*acc_full.transmittance/max(eps,1.f-resp);

    const Matrix3x3 inv_RS = construct_inv_RS(quat,scale);

    const float3 csamp_pos = csamp-pos;
    const float3 xg = inv_RS*csamp_pos;
    const float G = __expf(-0.5f*dot(xg,xg));

    const float3 dresp_dxg = -opacity*G*xg;//^T
    const Matrix3x3 dxg_dcsamp = inv_RS;
    const float3 dg = inv_RS * ray_direction;
    const float3 o_pos = ray_origin-pos;
    const float3 og = inv_RS * o_pos;
    float dg2 = max(eps,dot(dg,dg));
    const float3 dcsamp_dt = ray_direction;
    const float dresp_dt = dot(dresp_dxg,dxg_dcsamp*dcsamp_dt);
    const float3 dt_dog = dg/dg2;//^T
    const float3 dt_ddg = og/dg2 - 2.f*dot(og,dg)/max(dg2*dg2,eps) * dg;//^T
    const float3 dresp_dog = dresp_dt*dt_dog;//^T
    const float3 dresp_ddg = dresp_dt*dt_ddg;//^T

    const Matrix3x3 dog_dmu = (-1.f)*inv_RS;
    const Matrix3x3 dxg_dmu = (-1.f)*inv_RS;
    const float3 dresp_dmu = dxg_dmu.transpose()*dresp_dxg + dog_dmu.transpose()*dresp_dog;//^T

    Matrix3x3 inv_RSS = inv_RS;
    inv_S_times_M_(scale,inv_RSS);
    const float3 dxg_ds_diag = (-1.f)*inv_RSS*csamp_pos;
    const float3 dog_ds_diag = (-1.f)*inv_RSS*o_pos;
    const float3 ddg_ds_diag = (-1.f)*inv_RSS*ray_direction;
    const float3 dresp_ds = dresp_dxg*dxg_ds_diag + dresp_dog*dog_ds_diag + dresp_ddg*ddg_ds_diag;

    const float qr=quat.x, qi=quat.y, qj=quat.z, qk=quat.w;
    Matrix3x3 dR_dqr({
        0.f, qk, -qj,
        -qk, 0.f, qi,
        qj, -qi, 0.f
    });
    //dR_dqr *= 2.f; // do it in the end
    Matrix3x3 dR_dqi({
        0.f, qj, qk,
        qj, -2.f*qi, qr,
        qk, -qr, -2.f*qi
    });
    //dR_dqi *= 2.f;
    Matrix3x3 dR_dqj({
        -2.f*qj, qi, -qr,
        qi, 0.f, qk,
        qr, qk, -2.f*qj
    });
    //dR_dqj *= 2.f;
    Matrix3x3 dR_dqk({
        -2.f*qk, qr, qi,
        -qr, -2.f*qk, qj,
        qi, qj, 0.f
    });
    //dR_dqk *= 2.f;

    inv_S_times_M_(scale,dR_dqr);
    inv_S_times_M_(scale,dR_dqi);
    inv_S_times_M_(scale,dR_dqj);
    inv_S_times_M_(scale,dR_dqk);

    const float3 dxg_dqr = dR_dqr*csamp_pos;
    const float3 dxg_dqi = dR_dqi*csamp_pos;
    const float3 dxg_dqj = dR_dqj*csamp_pos;
    const float3 dxg_dqk = dR_dqk*csamp_pos;

    const float3 dog_dqr = dR_dqr*o_pos;
    const float3 dog_dqi = dR_dqi*o_pos;
    const float3 dog_dqj = dR_dqj*o_pos;
    const float3 dog_dqk = dR_dqk*o_pos;

    const float3 ddg_dqr = dR_dqr*ray_direction;
    const float3 ddg_dqi = dR_dqi*ray_direction;
    const float3 ddg_dqj = dR_dqj*ray_direction;
    const float3 ddg_dqk = dR_dqk*ray_direction;

    const float4 dresp_dq = {
        dot(dresp_dxg,dxg_dqr)+dot(dresp_dog,dog_dqr)+dot(dresp_ddg,ddg_dqr),
        dot(dresp_dxg,dxg_dqi)+dot(dresp_dog,dog_dqi)+dot(dresp_ddg,ddg_dqi),
        dot(dresp_dxg,dxg_dqj)+dot(dresp_dog,dog_dqj)+dot(dresp_ddg,ddg_dqj),
        dot(dresp_dxg,dxg_dqk)+dot(dresp_dog,dog_dqk)+dot(dresp_ddg,ddg_dqk)
    };

    // const uint3 launch_idxy = optixGetLaunchIndex();
    // const uint3 launch_dim = optixGetLaunchDimensions();
    // const int launch_id = launch_idxy.x + launch_idxy.y*launch_dim.x;
    // const float3 dL_dC = params.dL_dC[launch_id];

    const float dL_dresp = dot(dL_dC,dC_dresp);
    //atomicAdd(&params.grad_resp[chit_id],dL_dresp);

    const float dL_dopac = dL_dresp*G;
    atomicAdd(&grad_opacity[chit_id],dL_dopac);
    //grad_opacity += dL_dopac;

    float3 dL_dmu = dL_dresp * dresp_dmu;

    const float3 dC_dcolor_diag = make_float3(resp*acc.transmittance);
    const float3 dL_dcolor = dL_dC * dC_dcolor_diag;
    float3 sh_grad_xyz;
    compute_radiance_bwd_at(shs,sh_deg,grad_sh,chit_id,ray_origin,ray_direction,dL_dcolor,sh_grad_xyz,clamped);
    //dL_dmu += sh_grad_xyz;

    atomicAdd_float3(grad_xyz[chit_id],dL_dmu);
    //grad_xyz += dL_dmu;

    const float3 dL_ds = dL_dresp*dresp_ds;
    atomicAdd_float3(grad_scaling[chit_id],dL_ds);
    //grad_scaling += dL_ds;

    const float4 dL_dq = dL_dresp*dresp_dq *2.f;
    atomicAdd(&grad_rotation[chit_id].x,dL_dq.x);
    atomicAdd(&grad_rotation[chit_id].y,dL_dq.y);
    atomicAdd(&grad_rotation[chit_id].z,dL_dq.z);
    atomicAdd(&grad_rotation[chit_id].w,dL_dq.w);
    //grad_rotation += dL_dq;
}

