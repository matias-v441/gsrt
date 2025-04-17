#include <optix.h>

#include "optix_types.h"
#include "utils/Matrix.h"
#include "utils/auxiliary.h"

#include <vector>
#include <float.h>

using namespace util;

extern "C" {
__constant__ Params params;
constexpr float eps = 1e-6;
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

__device__ void compute_radiance(unsigned int gs_id, const float3 &ray_origin,
     const float3& ray_direction, float3& rad, bool *clamped){

    const int deg = params.sh_deg;
    if(deg == -1){
        rad = params.gs_color[gs_id];
        return;
    }
    //const float3 dir = -ray_direction;
    const float3 dir = ray_direction;
    //const float3 mu = params.gs_xyz[gs_id];
    //const float3 dir = normalize(mu-ray_origin);

    const float3* sh = params.gs_sh + gs_id*16;

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

    //printf("RAD_res %f %f %f\n", result.x,result.y,result.z);
	clamped[0] = (result.x < 0);
	clamped[1] = (result.y < 0);
	clamped[2] = (result.z < 0);

	rad = {max(result.x,0.f),max(result.y,0.f),max(result.z,0.f)};
    //printf("RAD %f %f %f\n", rad.x,rad.y,rad.z);
}

__device__ __forceinline__ void atomicAdd_float3(float3 &acc, const float3 &val){
    atomicAdd(&acc.x,val.x);
    atomicAdd(&acc.y,val.y);
    atomicAdd(&acc.z,val.z);
}


// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
// __device__ void compute_radiance_bwd(int idx, int deg, int max_coeffs,
//                      const glm::vec3* means, glm::vec3 campos, const float* shs,
//                      const bool* clamped, const glm::vec3* dL_dcolor,
//                      glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
__device__ void compute_radiance_bwd(int gs_id, const float3& ray_origin, const float3& ray_direction,
    const float3& dL_dcolor, float3& grad_xyz, const bool* clamped)
{
    atomicAdd_float3(params.grad_color[gs_id], dL_dcolor);
    grad_xyz = make_float3(0.f);
    if(params.sh_deg == -1) return;

	// same as forward -----

    //const float3 dir = -params.ray_directions[idx.x];
    //const float3 mu = params.gs_xyz[gs_id];
    //const float3 dir = normalize(mu-ray_origin);
    const float3 dir = ray_direction;

    const float3* sh = params.gs_sh + gs_id*16;
    const int deg = params.sh_deg;
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
    float3 *dL_dsh = params.grad_sh + gs_id*16;

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

struct Hit{
    int id;
    float thit;
};

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


struct Acc{
    float3 radiance;
    float transmittance;
};

__device__ __forceinline__ Matrix3x3 outer(const float3& a, const float3& b){
    Matrix3x3 M({
        a.x*b.x, a.x*b.y, a.x*b.z,
        a.y*b.x, a.y*b.y, a.y*b.z,
        a.z*b.x, a.z*b.y, a.z*b.z
    });
    return M;
}

__device__ __forceinline__ Matrix4x4 outer(const float4& a, const float4& b){
    Matrix4x4 M({
        a.x*b.x, a.x*b.y, a.x*b.z, a.x*b.w,
        a.y*b.x, a.y*b.y, a.y*b.z, a.y*b.w,
        a.z*b.x, a.z*b.y, a.z*b.z, a.z*b.w,
        a.w*b.x, a.w*b.y, a.w*b.z, a.w*b.w,
    });
    return M;
}

__device__ __forceinline__ float dot(const Matrix3x3& a, const Matrix3x3& b){
    float res = 0;
    const float* pa = a.getData();
    const float* pb = b.getData();
    for(int i = 0; i < 9; ++i)
        res += pa[i]*pb[i];
    return res;
}

__device__ __forceinline__ Matrix3x3 TxM(const Matrix3x3& M, 
    const Matrix3x3& T11, const Matrix3x3& T12,const Matrix3x3& T13,
    const Matrix3x3& T21, const Matrix3x3& T22,const Matrix3x3& T23,
    const Matrix3x3& T31, const Matrix3x3& T32,const Matrix3x3& T33
){
    const float3 m1 = M.getCol(0);
    const float3 m2 = M.getCol(1);
    const float3 m3 = M.getCol(2);
    return m1.x*T11+m2.x*T12+m3.x*T13
          +m1.y*T21+m2.y*T22+m3.y*T23
          +m1.z*T31+m2.z*T32+m3.z*T33; 
}

__device__ __forceinline__ void add_grad_0(const Acc& acc, const float3& rad, const Acc& acc_full,
                                        int chit_id, const float3& c_samp,
                                        const float resp, 
                                        const float3 ray_origin,
                                        const float3 ray_direction,
                                        const bool* clamped
                                        ){

    atomicAdd(params.num_its_bwd,1ull);

    const float4& quat = params.gs_rotation[chit_id];
    const float3& scale = params.gs_scaling[chit_id];
    const Matrix3x3 inv_RS = construct_inv_RS(quat,scale);
    
    const float &prev_acc_trans = acc.transmittance;
    const float3 &acc_rad = acc.radiance;
    const float3 &particle_rad = rad;
    const float3 &full_rad = acc_full.radiance;
    const float3 &pos = params.gs_xyz[chit_id];
    const float &opacity = params.gs_opacity[chit_id];
    
    float3 background{1.f,1.f,1.f};
    const float3 dC_dresp = prev_acc_trans*(particle_rad-full_rad+acc_rad); // - background*acc_full.transmittance/max(eps,1.f-resp);
    const float3 x = c_samp-pos;
    const float3 v = inv_RS*x;
    const float G = exp(-dot(v,v));
    const float3 d_resp_pos = 2*opacity*G*(inv_RS*inv_RS.transpose())*x;

    //const float3 dL_dC{1.f,1.f,1.f};
    const uint3 launch_idxy = optixGetLaunchIndex();
    const uint3 launch_dim = optixGetLaunchDimensions();
    const int launch_id = launch_idxy.x + launch_idxy.y*launch_dim.x;
    const float3 dL_dC = params.dL_dC[launch_id];

    //printf("dC_dresp %f %f %f\n",dC_dresp.x,dC_dresp.y,dC_dresp.z);
    const float dL_dresp = dot(dL_dC,dC_dresp);

    atomicAdd(&params.grad_resp[chit_id],dL_dresp);

    //printf("dL_dresp %f\n",dL_dresp);
    //printf("G %f\n", G);
    const float grad_opac = dL_dresp*G;
    float3 grad_pos = dL_dresp*d_resp_pos;

    atomicAdd(&params.grad_opacity[chit_id],grad_opac);

    const float3 dC_dcolor_diag = make_float3(resp*prev_acc_trans);

    const float3 dL_dcolor = dL_dC * dC_dcolor_diag;

    // SH
    float3 sh_grad_xyz;
    compute_radiance_bwd(chit_id,ray_origin,ray_direction,dL_dcolor,sh_grad_xyz,clamped);
    grad_pos += sh_grad_xyz;

    atomicAdd_float3(params.grad_xyz[chit_id],grad_pos);

    const Matrix3x3 dresp_dinvSigma = - opacity*G*outer(x,x);

    // Scale
    const Matrix3x3 R = construct_rotation(quat);
    const float3 r1 = R.getCol(0);
    const float3 r2 = R.getCol(1);
    const float3 r3 = R.getCol(2);
    const Matrix3x3 dinvSigma_ds1 = -2.f/pow(scale.x,3)*outer(r1,r1);
    const Matrix3x3 dinvSigma_ds2 = -2.f/pow(scale.y,3)*outer(r2,r2);
    const Matrix3x3 dinvSigma_ds3 = -2.f/pow(scale.z,3)*outer(r3,r3);
    const float3 dL_ds{
        dL_dresp*dot(dresp_dinvSigma,dinvSigma_ds1),
        dL_dresp*dot(dresp_dinvSigma,dinvSigma_ds2),
        dL_dresp*dot(dresp_dinvSigma,dinvSigma_ds3)
    }; 
    atomicAdd_float3(params.grad_scale[chit_id],dL_ds);

    // Rotation
    const float r11=r1.x, r12=r1.y, r13=r1.z;
    const float r21=r2.x, r22=r2.y, r23=r2.z;
    const float r31=r3.x, r32=r3.y, r33=r3.z;
    Matrix3x3 dinvSigma_dr11({
        2.f*r11, r12, r13,
        r12, 0.f, 0.f,
        r13, 0.f, 0.f 
        }
    );
    Matrix3x3 dinvSigma_dr21({
        0.f, r11, 0.f,
        r11, 2.f*r12, r13,
        0.f, r13, 0.f 
        }
    );
    Matrix3x3 dinvSigma_dr31({
        0.f, 0.f, r11,
        0.f, 0.f, r12,
        r11, r12, 2.f*r13 
        }
    );
    const float isx_2 = 1.f/(scale.x*scale.x);
    dinvSigma_dr11 *= isx_2;
    dinvSigma_dr21 *= isx_2;
    dinvSigma_dr31 *= isx_2;

    Matrix3x3 dinvSigma_dr12({
        2.f*r21, r22, r23,
        r22, 0.f, 0.f,
        r23, 0.f, 0.f 
        }
    );
    Matrix3x3 dinvSigma_dr22({
        0.f, r21, 0.f,
        r21, 2.f*r22, r23,
        0.f, r23, 0.f 
        }
    );
    Matrix3x3 dinvSigma_dr32({
        0.f, 0.f, r21,
        0.f, 0.f, r22,
        r21, r22, 2.f*r23 
        }
    );
    const float isy_2 = 1.f/(scale.y*scale.y);
    dinvSigma_dr12 *= isy_2;
    dinvSigma_dr22 *= isy_2;
    dinvSigma_dr32 *= isy_2;

    Matrix3x3 dinvSigma_dr13({
        2.f*r31, r32, r33,
        r32, 0.f, 0.f,
        r33, 0.f, 0.f 
        }
    );
    Matrix3x3 dinvSigma_dr23({
        0.f, r31, 0.f,
        r31, 2.f*r32, r33,
        0.f, r33, 0.f 
        }
    );
    Matrix3x3 dinvSigma_dr33({
        0.f, 0.f, r31,
        0.f, 0.f, r32,
        r31, r32, 2.f*r33 
        }
    );
    const float isz_2 = 1.f/(scale.z*scale.z);
    dinvSigma_dr13 *= isz_2;
    dinvSigma_dr23 *= isz_2;
    dinvSigma_dr33 *= isz_2;

    const float4 nquat = normalize(quat);
    const float qr=nquat.x, qi=nquat.y, qj=nquat.z, qk=nquat.w;
    Matrix3x3 dR_dnqr({
        0.f, -qk, qj,
        qk, 0.f, -qi,
        -qj, qi, 0.f
    });
    dR_dnqr *= 2.f;
    Matrix3x3 dR_dnqi({
        0.f, qj, qk,
        qj, -2.f*qi, -qr,
        qk, qr, -2.f*qi
    });
    dR_dnqi *= 2.f;
    Matrix3x3 dR_dnqj({
        -2.f*qj, qi, qr,
        qi, 0.f, qk,
        -qr, qk, -2.f*qj
    });
    dR_dnqj *= 2.f;
    Matrix3x3 dR_dnqk({
        -2.f*qk, -qr, qi,
        qr, -2.f*qk, qj,
        qi, qj, 0.f
    });
    dR_dnqk *= 2.f;

    Matrix4x4 dnq_dq = outer(quat,quat*(-1.f));
    float* p_dnq_dq = dnq_dq.getData();
    float qdq = dot(quat,quat);
    // main diag
    p_dnq_dq[0] = qdq-quat.x*quat.x;
    p_dnq_dq[5] = qdq-quat.y*quat.y;
    p_dnq_dq[10] = qdq-quat.z*quat.z;
    p_dnq_dq[15] = qdq-quat.w*quat.w;
    //---
    dnq_dq *= powf(qdq,-1.5f);

    float dL_dq[4];
    for(int i = 0; i < 4; ++i){
        const float4 dnq_dqi = dnq_dq.getCol(i);    
        const Matrix3x3 dR_dqi = dR_dnqr*dnq_dqi.x
                                +dR_dnqi*dnq_dqi.y
                                +dR_dnqj*dnq_dqi.z
                                +dR_dnqk*dnq_dqi.w;
        const Matrix3x3 dinvSigma_dqi = TxM(dR_dqi, 
            dinvSigma_dr11, dinvSigma_dr12, dinvSigma_dr13,
            dinvSigma_dr21, dinvSigma_dr22, dinvSigma_dr23,
            dinvSigma_dr31, dinvSigma_dr32, dinvSigma_dr33
        );
        dL_dq[i] = dL_dresp*dot(dresp_dinvSigma,dinvSigma_dqi);
    }
    atomicAdd(&params.grad_rotation[chit_id].x,dL_dq[0]);
    atomicAdd(&params.grad_rotation[chit_id].y,dL_dq[1]);
    atomicAdd(&params.grad_rotation[chit_id].z,dL_dq[2]);
    atomicAdd(&params.grad_rotation[chit_id].w,dL_dq[3]);
}

__device__ __forceinline__ void add_grad_I(const Acc& acc, const float3& rad, const Acc& acc_full,
                                        int chit_id, const float3& csamp,
                                        const float resp, 
                                        const float3 ray_origin,
                                        const float3 ray_direction,
                                        const bool* clamped
                                        ){

    atomicAdd(params.num_its_bwd,1ull);

    const float4 quat = params.gs_rotation[chit_id];
    const float3 scale = params.gs_scaling[chit_id];
    const float3 pos = params.gs_xyz[chit_id];
    const float opacity = params.gs_opacity[chit_id];

    float3 background{1.f,1.f,1.f};
    //const float3 dC_dresp = (acc.transmittance*rad - (acc_full.radiance - acc.radiance)/*/max(eps,1.f-resp)*/) - background*acc_full.transmittance/max(eps,1.f-resp);
    const float3 dC_dresp = acc.transmittance*(rad - acc_full.radiance + acc.radiance)/*/max(eps,1.f-resp)*/ - background*acc_full.transmittance/max(eps,1.f-resp);

    const Matrix3x3 inv_RS = construct_inv_RS(quat,scale);

    const float3 csamp_pos = csamp-pos;
    const float3 xg = inv_RS*csamp_pos;
    const float G = __expf(-dot(xg,xg));

    const float3 dresp_dxg = -2.f*opacity*G*xg;//^T
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
    const Matrix3x3 dxg_dmu = (1.f)*inv_RS;
    const float3 dresp_dmu = dxg_dmu.transpose()*dresp_dxg + dog_dmu.transpose()*dresp_dog;//^T

    Matrix3x3 inv_RSS = inv_RS;
    inv_S_times_M_(scale,inv_RSS);
    const float3 dxg_ds_diag = (-1.f)*inv_RSS*csamp_pos;
    const float3 dog_ds_diag = (-1.f)*inv_RSS*o_pos;
    const float3 ddg_ds_diag = (-1.f)*inv_RSS*ray_direction;
    const float3 dresp_ds = dresp_dxg*dxg_ds_diag + dresp_dog*dog_ds_diag + dresp_ddg*ddg_ds_diag;

    const float qr=quat.x, qi=quat.y, qj=quat.z, qk=quat.w;
    Matrix3x3 dR_dqr({
        0.f, -qk, qj,
        qk, 0.f, -qi,
        -qj, qi, 0.f
    });
    dR_dqr *= 2.f;
    Matrix3x3 dR_dqi({
        0.f, qj, qk,
        qj, -2.f*qi, -qr,
        qk, qr, -2.f*qi
    });
    dR_dqi *= 2.f;
    Matrix3x3 dR_dqj({
        -2.f*qj, qi, qr,
        qi, 0.f, qk,
        -qr, qk, -2.f*qj
    });
    dR_dqj *= 2.f;
    Matrix3x3 dR_dqk({
        -2.f*qk, -qr, qi,
        qr, -2.f*qk, qj,
        qi, qj, 0.f
    });
    dR_dqk *= 2.f;

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

    const uint3 launch_idxy = optixGetLaunchIndex();
    const uint3 launch_dim = optixGetLaunchDimensions();
    const int launch_id = launch_idxy.x + launch_idxy.y*launch_dim.x;
    const float3 dL_dC = params.dL_dC[launch_id];

    const float dL_dresp = dot(dL_dC,dC_dresp);
    atomicAdd(&params.grad_resp[chit_id],dL_dresp);

    const float dL_dopac = dL_dresp*G;
    atomicAdd(&params.grad_opacity[chit_id],dL_dopac);

    float3 dL_dmu = dL_dresp * dresp_dmu;

    const float3 dC_dcolor_diag = make_float3(resp*acc.transmittance);
    const float3 dL_dcolor = dL_dC * dC_dcolor_diag;
    float3 sh_grad_xyz;
    compute_radiance_bwd(chit_id,ray_origin,ray_direction,dL_dcolor,sh_grad_xyz,clamped);
    //dL_dmu += sh_grad_xyz;

    atomicAdd_float3(params.grad_xyz[chit_id],dL_dmu);

    const float3 dL_ds = dL_dresp*dresp_ds;
    atomicAdd_float3(params.grad_scale[chit_id],dL_ds);

    const float4 dL_dq = dL_dresp*dresp_dq;
    atomicAdd(&params.grad_rotation[chit_id].x,dL_dq.x);
    atomicAdd(&params.grad_rotation[chit_id].y,dL_dq.y);
    atomicAdd(&params.grad_rotation[chit_id].z,dL_dq.z);
    atomicAdd(&params.grad_rotation[chit_id].w,dL_dq.w);

    //#define _norm_(x) sqrt(dot(x,x))
    //if(sqrt(dot(grad_pos,grad_pos)) > 10.)
    //    printf("POS GRAD NORM %f %f %f %f %f %f\n",
    //        _norm_(d_resp_pos),
    //        dL_dresp,
    //        G,
    //        _norm_(x),
    //        _norm_((inv_RS*inv_RS.transpose())*x),
    //        opacity);
}


__device__ __forceinline__ void add_grad_II(const Acc& acc, const float3& rad, const Acc& acc_full,
                                        int chit_id, const float3& csamp,
                                        const float resp, 
                                        const float3 ray_origin,
                                        const float3 ray_direction,
                                        const bool* clamped
                                        ){
    
    const float4 quat = params.gs_rotation[chit_id];
    const float3 scale = params.gs_scaling[chit_id];
    const float3 mu = params.gs_xyz[chit_id];
    const float opacity = params.gs_opacity[chit_id];
    const uint3 launch_idxy = optixGetLaunchIndex();
    const uint3 launch_dim = optixGetLaunchDimensions();
    const int launch_id = launch_idxy.x + launch_idxy.y*launch_dim.x;
    const float3 dL_dC = params.dL_dC[launch_id];
    const Matrix3x3 inv_RS = construct_inv_RS(quat,scale);

    //float3 background{1.f,1.f,1.f};
    float3 background{0.f,0.f,0.f};
    const float3 dC_dresp = acc.transmittance*rad - (acc_full.radiance - acc.radiance)/max(eps,1.f-resp)
                            - background*acc_full.transmittance/max(eps,1.f-resp);
    const float dL_dresp = dot(dL_dC,dC_dresp); 

    const float3 x_mu = csamp-mu;
    const float3 xg = inv_RS*x_mu;
    const float G = __expf(-0.5*dot(xg,xg));

    const float3 dresp_dxg = -opacity*G*xg;//^T
    const Matrix3x3 dxg_dcsamp = inv_RS;
    const float3 dg = inv_RS * ray_direction;
    const float3 o_pos = ray_origin-mu;
    const float3 og = inv_RS * o_pos;
    float dg2 = max(eps,dot(dg,dg));
    const float3 dcsamp_dt = ray_direction;
    const float dresp_dt = dot(dresp_dxg,dxg_dcsamp*dcsamp_dt);
    const float3 dt_dog = -dg/dg2;//^T
    const float3 dt_ddg = -og/dg2 + 2.f*dot(og,dg)/max(dg2*dg2,eps) * dg;//^T
    const float3 dresp_dog = dresp_dt*dt_dog;//^T
    const float3 dresp_ddg = dresp_dt*dt_ddg;//^T

    const Matrix3x3 dog_dmu = (-1.f)*inv_RS;
    const Matrix3x3 dxg_dmu = (-1.f)*inv_RS;
    const float3 dresp_dmu = dxg_dmu.transpose()*dresp_dxg + dog_dmu.transpose()*dresp_dog;//^T

    const float3 dL_dmu = dL_dresp*dresp_dmu;
    atomicAdd_float3(params.grad_xyz[chit_id],dL_dmu);

    const float dL_dopac = dL_dresp*G;
    atomicAdd(&params.grad_opacity[chit_id],dL_dopac);

    const Matrix3x3 dL_dinvRS = dL_dresp*(outer(dresp_dxg,x_mu) + outer(dresp_dog,o_pos) + outer(dresp_ddg,ray_direction));
    float* grad_invRS = (float*)(params.grad_invRS + chit_id);
	for (int i = 0; i < 9; ++i){
        atomicAdd(grad_invRS+i, dL_dinvRS[i]);
    }

    const float3 dC_dcolor_diag = make_float3(resp*acc.transmittance);
    const float3 dL_dcolor = dL_dC * dC_dcolor_diag;
    float3 sh_grad_xyz;
    compute_radiance_bwd(chit_id,ray_origin,ray_direction,dL_dcolor,sh_grad_xyz,clamped);
    //dL_dmu += sh_grad_xyz;
}


__device__ __forceinline__ void add_grad_III(const Acc& acc, const float3& rad, const Acc& acc_full,
                                        int chit_id, const float3& csamp,
                                        const float resp, 
                                        const float3 ray_origin,
                                        const float3 ray_direction,
                                        const bool* clamped
                                        ){
    
    const float4 quat = params.gs_rotation[chit_id];
    const float3 scale = params.gs_scaling[chit_id];
    const float3 mean3D = params.gs_xyz[chit_id];
    const float o = params.gs_opacity[chit_id];
    const uint3 launch_idxy = optixGetLaunchIndex();
    const uint3 launch_dim = optixGetLaunchDimensions();
    const int launch_id = launch_idxy.x + launch_idxy.y*launch_dim.x;
    const float3 dL_dC = params.dL_dC[launch_id];
    const Matrix3x3 SinvR = construct_inv_RS(quat,scale);
    float3 ray_o = ray_origin;
    float3 ray_d = ray_direction;
    float3 grad_colors = dL_dC;

    float3 background{1.f,1.f,1.f};
    // const float3 dC_dresp = acc.transmittance*rad - (acc_full.radiance - acc.radiance)/max(eps,1.f-resp)
    //                         - background*acc_full.transmittance/max(eps,1.f-resp);
    // const float dL_dresp = dot(dL_dC,dC_dresp); 

    // Compute intersection point
    float3 ray_o_mean3D = ray_o - mean3D;
    float3 o_g = SinvR * ray_o_mean3D; 
    float3 d_g = SinvR * ray_d;
    float dot_dg_dg = max(1e-6f, dot(d_g, d_g));
    float d = -dot(o_g, d_g) / dot_dg_dg;

    float3 pos = ray_o + d * ray_d;
    float3 mean_pos = mean3D - pos;
    float3 p_g = SinvR * mean_pos; 

    float G = __expf(-0.5f * dot(p_g, p_g));
    // float alpha = min(0.99f, o * G);
    // if (alpha<params.alpha_min) continue;

    // glm::vec3 c = computeColorFromSH_forward(params.deg, ray_d, params.shs + gs_idx * params.max_coeffs);

    // float w = T * alpha;
    // C += w * c;
    // D += w * d;
    // O += w;

    // T *= (1 - alpha);
    float3 c = rad;
    float3 C = acc.radiance;
    float3 C_final = acc_full.radiance;
    float alpha = resp;
    float T = acc.transmittance * (1.f - alpha);

    //float3 dL_dc = grad_colors * w;
    float dL_dd = 0;//grad_depths * w;
    float dL_dalpha = (
        dot(grad_colors, T * c - (C_final - C)) //+
        //grad_depths * (T * d - (D_final - D)) + 
        //-dot(grad_colors,background*acc_full.transmittance)//grad_alpha * (1 - O_final)
    ) / max(1e-6f, 1 - alpha);
    //computeColorFromSH_backward(params.deg, ray_d, params.shs + gs_idx * params.max_coeffs, dL_dc, params.grad_shs + gs_idx * params.max_coeffs);
    float dL_do = dL_dalpha * G;
    float dL_dG = dL_dalpha * o;
    float3 dL_dpg = -dL_dG * G * p_g;
    Matrix3x3 dL_dSinvR = outer(dL_dpg, mean_pos);
    
    float3 dL_dmean_pos = SinvR.transpose() * dL_dpg;
    float3 dL_dmean3D = dL_dmean_pos;

    dL_dd -= dot(dL_dmean_pos, ray_d);

    float3 dL_dog = -dL_dd / dot_dg_dg * d_g;
    float3 dL_ddg = -dL_dd / dot_dg_dg * o_g + 2 * dL_dd * dot(o_g, d_g) / max(1e-6f, dot_dg_dg * dot_dg_dg) * d_g;

    dL_dSinvR += outer(dL_dog, ray_o_mean3D);
    dL_dmean3D -= SinvR.transpose() * dL_dog;
    dL_dSinvR += outer(dL_ddg, ray_d);

    //atomic_add((float*)(params.grad_means3D+gs_idx), dL_dmean3D);
    atomicAdd_float3(params.grad_xyz[chit_id],dL_dmean3D);
    //atomicAdd(params.grad_opacity+gs_idx, dL_do);
    atomicAdd(&params.grad_opacity[chit_id],dL_do);

    // float* grad_SinvR = (float*)(params.grad_SinvR + gs_idx);
    // for (int j=0; j<9;++j){
    //     atomicAdd(grad_SinvR+j, dL_dSinvR[j/3][j%3]);
    // }

    float* grad_SinvR = (float*)(params.grad_invRS + chit_id);
	for (int i = 0; i < 9; ++i){
        atomicAdd(grad_SinvR+i, dL_dSinvR[i]);
    }

    const float3 dC_dcolor_diag = make_float3(resp*acc.transmittance);
    const float3 dL_dcolor = dL_dC * dC_dcolor_diag;
    float3 sh_grad_xyz;
    compute_radiance_bwd(chit_id,ray_origin,ray_direction,dL_dcolor,sh_grad_xyz,clamped);
    //dL_dmu += sh_grad_xyz;
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

constexpr int triagPerParticle = 20;
constexpr float Tmin = 0.001;
constexpr float min_alpha = 1/255.f;//0.01f;
constexpr float max_alpha = 0.99f;
constexpr float min_kernel_density = 0.0113f;

__device__ inline void add_grad/*(
    const float3& rayOrigin,
    const float3& rayDirection,
    int32_t particleIdx,
    const ParticleDensity* particleDensityPtr,
    ParticleDensity* particleDensityGradPtr,
    const float* particleRadiancePtr,
    float* particleRadianceGradPtr,
    float minParticleKernelDensity,
    float minParticleAlpha,
    float minTransmittance,
    int32_t sphEvalDegree,
    float integratedTransmittance,
    float& transmittance,
    float transmittanceGrad,
    float3 integratedRadiance,
    float3& radiance,
    float3 radianceGrad,
    )*/(const Acc& acc, const float3& rad, const Acc& acc_full,
                                        int chit_id, const float3& csamp,
                                        const float resp, 
                                        const float3 ray_origin,
                                        const float3 ray_direction,
                                        const bool* clamped
                                        ) {

    float3 particlePosition;
    float3 gscl;
    float33 particleInvRotation;
    float particleDensity;
    float4 grot;

    float3 radianceGrad;

    // {
    //     const ParticleDensity particleData = particleDensityPtr[particleIdx];
    //     particlePosition                   = particleData.position;
    //     gscl                               = particleData.scale;
    //     grot                               = particleData.quaternion;
    //     rotationMatrixTranspose(grot, particleInvRotation);
    //     particleDensity = particleData.density;
    // }
    {
        grot = params.gs_rotation[chit_id];
        gscl = params.gs_scaling[chit_id];
        particlePosition = params.gs_xyz[chit_id];
        particleDensity = params.gs_opacity[chit_id];
        const uint3 launch_idxy = optixGetLaunchIndex();
        const uint3 launch_dim = optixGetLaunchDimensions();
        const int launch_id = launch_idxy.x + launch_idxy.y*launch_dim.x;
        radianceGrad = params.dL_dC[launch_id];
        rotationMatrixTranspose(grot, particleInvRotation);
    }

    // project ray in the gaussian
    const float3 giscl   = make_float3(1 / gscl.x, 1 / gscl.y, 1 / gscl.z);
    const float3 gposc   = (ray_origin - particlePosition);
    const float3 gposcr  = (gposc * particleInvRotation);
    const float3 gro     = giscl * gposcr;
    const float3 rayDirR = ray_direction * particleInvRotation;
    const float3 grdu    = giscl * rayDirR;
    const float3 grd     = safe_normalize(grdu);
    const float3 gcrod   = cross(grd, gro);
    const float grayDist = dot(gcrod, gcrod);

    const float gres   = expf(-0.5f*grayDist);//particleResponse<ParticleKernelDegree>(grayDist);
    const float galpha = fminf(0.99f, gres * particleDensity);
    // if(galpha != resp){
    //     printf("resp %f\n",galpha-resp);
    // }

    if ((gres > min_kernel_density) && (galpha > min_alpha))
    {

        const float weight = galpha * acc.transmittance;

        const float nextTransmit = (1 - galpha) * acc.transmittance;


        // float3 sphCoefficients[SPH_MAX_NUM_COEFFS];
        // fetchParticleSphCoefficients(
        //     particleIdx,
        //     particleRadiancePtr,
        //     &sphCoefficients[0]);
        // const float3 grad = radianceFromSpHBwd(sphEvalDegree, &sphCoefficients[0], rayDirection, weight, radianceGrad, (float3*)&particleRadianceGradPtr[particleIdx * SPH_MAX_NUM_COEFFS * 3]);
        //const float3 dC_dcolor_diag = make_float3(resp*acc.transmittance);
        const float3 dL_dcolor = radianceGrad *weight;//radianceGrad * dC_dcolor_diag;
        float3 sh_grad_xyz;
        compute_radiance_bwd(chit_id,ray_origin,ray_direction,dL_dcolor,sh_grad_xyz,clamped);
        // if (sh_grad_xyz.x != 0.f || sh_grad_xyz.y != 0.f || sh_grad_xyz.z != 0.f)
        //     printf("sh_grad_xyz %f %f %f\n",sh_grad_xyz.x, sh_grad_xyz.y, sh_grad_xyz.z);

        const float3 grad = rad;

        // >>> rayRadiance = accumulatedRayRad + weigth * rayRad + (1-galpha)*transmit * residualRayRad
        const float3 rayRad = weight * grad;
        
        const float3 residualRayRad = fmaxf((nextTransmit <= 0.001 ? make_float3(0) : (acc_full.radiance - acc.radiance) / nextTransmit),
                                            make_float3(0));

        atomicAdd(
            &params.grad_opacity[chit_id],
            gres * (/*galphaRayHitGrd + galphaRayDnsGrd +*/ acc.transmittance * (grad.x - residualRayRad.x) * radianceGrad.x +
                    acc.transmittance * (grad.y - residualRayRad.y) * radianceGrad.y +
                    acc.transmittance * (grad.z - residualRayRad.z) * radianceGrad.z));

        const float gresGrd =
            particleDensity * (/*galphaRayHitGrd +  galphaRayDnsGrd +*/ acc.transmittance * (grad.x - residualRayRad.x) * radianceGrad.x +
                               acc.transmittance * (grad.y - residualRayRad.y) * radianceGrad.y +
                               acc.transmittance * (grad.z - residualRayRad.z) * radianceGrad.z);

        const float grayDistGrd = -0.5f*gres*gresGrd;//particleResponseGrd<PARTICLE_KERNEL_DEGREE>(grayDist, gres, gresGrd);

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
        atomicAdd(&params.grad_xyz[chit_id].x, rayMoGPosGrd.x);
        atomicAdd(&params.grad_xyz[chit_id].y, rayMoGPosGrd.y);
        atomicAdd(&params.grad_xyz[chit_id].z, rayMoGPosGrd.z);

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> grd = safe_normalize(grdu)
        // ===> d_grd / d_grdu = safe_normalize_bw(grd)
        const float3 grduGrd = safe_normalize_bw(grdu, grdGrd /* + grdRayHitGrd*/);

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> grdu = (1/gscl)*rayDirR
        // ===> d_grdu / d_gscl = -rayDirR/(gscl*gscl)
        // ===> d_grdu / d_rayDirR = (1/gscl)
        atomicAdd(&params.grad_scale[chit_id].x, /*gsclRayHitGrd.x +*/ gsclGrdGro.x + (-rayDirR.x / (gscl.x * gscl.x)) * grduGrd.x);
        atomicAdd(&params.grad_scale[chit_id].y, /*gsclRayHitGrd.y +*/ gsclGrdGro.y + (-rayDirR.y / (gscl.y * gscl.y)) * grduGrd.y);
        atomicAdd(&params.grad_scale[chit_id].z, /*gsclRayHitGrd.z +*/ gsclGrdGro.z + (-rayDirR.z / (gscl.z * gscl.z)) * grduGrd.z);
        const float3 rayDirRGrd = giscl * grduGrd;

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> rayDirR = matmul(rayDir, grotMat)
        // ===> d_rayDirR / d_grotmat = matmul_bw_mat(rayDir, grotMat)
        const float4 grotGrdRayDirR = matmul_bw_quat(ray_direction, rayDirRGrd, grot);
        atomicAdd(&params.grad_rotation[chit_id].x, grotGrdPoscr.x + grotGrdRayDirR.x);
        atomicAdd(&params.grad_rotation[chit_id].y, grotGrdPoscr.y + grotGrdRayDirR.y);
        atomicAdd(&params.grad_rotation[chit_id].z, grotGrdPoscr.z + grotGrdRayDirR.z);
        atomicAdd(&params.grad_rotation[chit_id].w, grotGrdPoscr.w + grotGrdRayDirR.w);
    }
}

constexpr unsigned int chunk_size = 16;

constexpr int num_recasts = 1;
constexpr int hits_max_capacity = chunk_size*num_recasts;


__device__ bool compute_response(
    const float3& o, const float3& d, const float3& mu,
    const float opacity, const Matrix3x3& inv_RS,unsigned int chit_id,
    float& alpha, float& tmax){

    // float3 og = inv_RS*(mu-o);
    // float3 dg = inv_RS*d;
    // tmax = dot(og,dg)/max(eps,dot(dg,dg));
    // //tmax = dot(og,dg)/(eps+dot(dg,dg));
    // float3 c_samp = o+tmax*d;
    // float3 v = inv_RS*(c_samp-mu);
    // float resp = exp(-.5f*dot(v,v));
    // if(resp < min_kernel_density) return false;
    // alpha = min(0.99f,opacity*resp);

    float3 particlePosition;
    float3 gscl;
    float33 particleInvRotation;
    float particleDensity;
    float4 grot;
    {
        grot = params.gs_rotation[chit_id];
        gscl = params.gs_scaling[chit_id];
        particlePosition = params.gs_xyz[chit_id];
        particleDensity = params.gs_opacity[chit_id];
        const uint3 launch_idxy = optixGetLaunchIndex();
        const uint3 launch_dim = optixGetLaunchDimensions();
        const int launch_id = launch_idxy.x + launch_idxy.y*launch_dim.x;
        rotationMatrixTranspose(grot, particleInvRotation);
    }
    // project ray in the gaussian
    const float3 giscl   = make_float3(1 / gscl.x, 1 / gscl.y, 1 / gscl.z);
    const float3 gposc   = (o - particlePosition);
    const float3 gposcr  = (gposc * particleInvRotation);
    const float3 gro     = giscl * gposcr;
    const float3 rayDirR = d * particleInvRotation;
    const float3 grdu    = giscl * rayDirR;
    const float3 grd     = safe_normalize(grdu);
    const float3 gcrod   = cross(grd, gro);
    const float grayDist = dot(gcrod, gcrod);
    const float resp   = expf(-0.5f*grayDist);//particleResponse<ParticleKernelDegree>(grayDist);
    alpha = fminf(0.99f, resp * particleDensity);
    
    //resp = opacity*exp(-dot(v,v));
    return (alpha > min_alpha) && (resp > min_kernel_density);
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

    constexpr float max_dist = 100.f;//1e16f; 
    float min_dist = 0.f;
    constexpr float epsT = 1e-9f;

    Acc acc{};
    acc.radiance = make_float3(0.f);
    acc.transmittance = 1.f;
    unsigned int* uip_acc = reinterpret_cast<unsigned int *>(&acc);

    Acc acc_full{};
    acc_full.radiance = params.radiance[id];
    acc_full.transmittance = params.transmittance[id];
    unsigned int* uip_acc_full = reinterpret_cast<unsigned int *>(&acc_full);

    while(min_dist <= max_dist && acc.transmittance > Tmin){
        for(int i=0; i<chunk_size;++i){
            hits[i].thit = FLT_MAX;
        }
        optixTrace(
            OPTIX_PAYLOAD_TYPE_ID_0,
            params.handle,
            ray_origin,
            ray_direction,
            min_dist,                      // Min intersection distance
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
            if(chit.thit != FLT_MAX && acc.transmittance > Tmin){
                float resp,thit; 
                bool accept = compute_response(ray_origin,
                                ray_direction,
                                params.gs_xyz[chit.id],
                                params.gs_opacity[chit.id],
                                construct_inv_RS(params.gs_rotation[chit.id],params.gs_scaling[chit.id]),
                                chit.id,
                                resp,thit);
                if(accept)
                {
                    float3 rad; bool clamped[3];
                    float3 pos = ray_origin+ray_direction*chit.thit;
                    compute_radiance(chit.id,ray_origin,ray_direction,rad,clamped);
                    acc.radiance += rad*resp*acc.transmittance;
                    if(params.compute_grad){
                        add_grad(acc,rad,acc_full,chit.id,
                            ray_origin+ray_direction*chit.thit,
                            resp, ray_origin, ray_direction, clamped);
                    }
                    acc.transmittance *= (1.-resp);
                }
                min_dist = fmaxf(min_dist,chit.thit);
            }
        }
        if(hits[hits_max_capacity-1].thit == FLT_MAX) break;
    }
    params.radiance[id] = acc.radiance;
    params.transmittance[id] = acc.transmittance;
}


__device__ inline bool intersectInstanceParticle(
    const float3& particleRayOrigin,
    const float3& particleRayDirection,
    const int32_t particleIdx,
    const float minHitDistance,
    const float maxHitDistance,
    const float maxParticleSquaredDistance,
    float& hitDistance) {
    const float numerator   = -dot(particleRayOrigin, particleRayDirection);
    const float denominator = 1.f / dot(particleRayDirection, particleRayDirection);
    hitDistance             = numerator * denominator;
    if ((hitDistance > minHitDistance) && (hitDistance < maxHitDistance)) {
        const float3 gcrod = cross(safe_normalize(particleRayDirection), particleRayOrigin);
        return (dot(gcrod, gcrod) * denominator < maxParticleSquaredDistance);
    }
    return false;
}

extern "C" __global__ void __intersection__is() {
    float hitDistance;
    printf("intersection\n");
    bool intersect = intersectInstanceParticle(optixGetObjectRayOrigin(),
                                optixGetObjectRayDirection(),
                                optixGetInstanceIndex(),
                                optixGetRayTmin(),
                                optixGetRayTmax(),
                                9.f,
                                hitDistance);
    if(intersect){
        optixReportIntersection(0.f,0);
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

    const unsigned int prim_id = optixGetPrimitiveIndex();

    const unsigned int hit_id = optixGetPrimitiveIndex()/triagPerParticle;

    Hit hit;
    hit.id = hit_id;
    float _resp,_thit; 
    // compute_response(optixGetWorldRayOrigin(),
    //                 optixGetWorldRayDirection(),
    //                 params.gs_xyz[hit_id],
    //                 params.gs_opacity[hit_id],
    //                 construct_inv_RS(params.gs_rotation[hit_id],params.gs_scaling[hit_id]),
    //                 _resp,_thit);
    _thit = optixGetRayTmax();
    hit.thit = _thit;//optixGetRayTmax();
    //if(_resp < respMin){
    //    optixIgnoreIntersection();
    //    return;
    //}
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

        if(_thit < hitq[chunk_size-1].thit)
            optixIgnoreIntersection();
    }
}

extern "C" __global__ void __anyhit__bwd() {


    optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_1);

    unsigned int p_hitq[2];
    p_hitq[0] = optixGetPayload_4();
    p_hitq[1] = optixGetPayload_5();
    Hit* hitq;
    memcpy(&hitq, p_hitq, sizeof(p_hitq));

    const unsigned int prim_id = optixGetPrimitiveIndex();
    float3 normal = params.gs_normals[prim_id];
    if(dot(normal,optixGetWorldRayDirection())>0.){
        optixIgnoreIntersection();
        return;
    }

    const unsigned int hit_id = optixGetPrimitiveIndex()/triagPerParticle;

    Hit hit;
    hit.id = hit_id;
    hit.thit = optixGetRayTmax();

    for(int i = 0; i < chunk_size; ++i){
        float thit = hitq[i].thit;
        if(thit > hit.thit){
            hitq[i] = hit;
            hit.thit = thit;
        }
    }

    if(optixGetRayTmax() < hitq[chunk_size-1].thit)
        optixIgnoreIntersection();
}
