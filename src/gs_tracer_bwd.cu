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

__device__ void compute_radiance(unsigned int gs_id, const float3 &ray_origin,
     const float3& ray_direction, float3& rad, bool *clamped){

    const int deg = params.sh_deg;
    if(deg == -1){
        rad = params.gs_color[gs_id];
        return;
        //printf("%d\n",deg);
    }
    //const float3 dir = -ray_direction;
    const float3 mu = params.gs_xyz[gs_id];
    const float3 dir = normalize(mu-ray_origin);

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
__device__ void compute_radiance_bwd(int gs_id, const float3& ray_origin, 
    const float3& dL_dcolor, float3& grad_xyz, const bool* clamped)
{
    atomicAdd_float3(params.grad_color[gs_id], dL_dcolor);
    grad_xyz = make_float3(0.f);
    if(params.sh_deg == -1) return;

	// same as forward -----

    //const float3 dir = -params.ray_directions[idx.x];
    const float3 mu = params.gs_xyz[gs_id];
    const float3 dir = normalize(mu-ray_origin);

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
    float resp;
};

//struct HitBwd{
//    int id;
//    float thit;
//    float resp;
//    Matrix3x3 inv_RS;
//};

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

struct Acc{
    float3 radiance;
    float transmittance;
};


__device__ __forceinline__ void add_samp(Acc& acc, const float3& rad, const Hit& chit){
    acc.radiance += rad*chit.resp*acc.transmittance;
    acc.transmittance *= (1.-chit.resp);
}

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

__device__ __forceinline__ void add_grad(const Acc& acc, const float3& rad, const Acc& acc_full,
                                        int chit_id, const float3& c_samp,
                                        const float resp, 
                                        const float3 ray_origin,
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

    const float3 dC_dresp = prev_acc_trans*(particle_rad-full_rad+acc_rad);
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
    compute_radiance_bwd(chit_id,ray_origin,dL_dcolor,sh_grad_xyz,clamped);
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


constexpr int chunk_size = 1024;

constexpr int num_recasts = 1;
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
        float3 rad; bool clamped[3];
        float3 pos = ray_origin+ray_direction*chit.thit;
        //printf("HIT %f %f %f \n", pos.x, pos.y, pos.z);
        compute_radiance(chit.id,ray_origin,ray_direction,rad,clamped);
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
            float3 rad; bool clamped[3];

            float3 pos = ray_origin+ray_direction*chit.thit;
            //printf("HIT %f %f %f \n", pos.x, pos.y, pos.z);
            compute_radiance(chit.id,ray_origin,ray_direction,rad,clamped);
            acc_bwd.radiance += rad*chit.resp*acc.transmittance;
            add_grad(acc_bwd,rad,acc,chit.id,
                ray_origin+ray_direction*chit.thit,
                chit.resp, ray_origin, clamped);
            acc_bwd.transmittance *= (1.-chit.resp);
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
        float3 rad; bool clamped[3];
        compute_radiance(chit.id,optixGetWorldRayOrigin(),optixGetWorldRayDirection(),rad,clamped);
        add_samp(acc,rad,chit);

        optixSetPayload_0(__float_as_uint(acc.radiance.x));
        optixSetPayload_1(__float_as_uint(acc.radiance.y));
        optixSetPayload_2(__float_as_uint(acc.radiance.z));
        optixSetPayload_3(__float_as_uint(acc.transmittance));

        if(acc.transmittance < Tmin){
            return;
        }
        //if(hitq_capacity != hits_max_capacity && optixGetRayTmax() < chit.thit){
        //    printf("recast %d\n",hitq_capacity+chunk_size);
        //    optixSetPayload_7(hitq_capacity+chunk_size);
        //    return;
        //}
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

        float3 rad; bool clamped[3];
        compute_radiance(chit.id,origin,direction,rad,clamped);

        acc.radiance += rad*chit.resp*acc.transmittance;
        add_grad(acc,rad,acc_full,chit.id,
            origin+direction*chit.thit,
            chit.resp,origin,clamped);
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
