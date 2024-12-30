#include "tracer_cu.cuh"
#include "tracer_custom.h"

//#include "utils/exception.h"
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include "utils/vec_math.h"
#include "utils/Matrix.h"
//
using namespace util;
using namespace kdtree_impl;

#include "utils/auxiliary.h"
#include <assert.h>
#include <algorithm>

__device__ Matrix3x3 _construct_rotation(float4 vec){
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

__device__ void _computeResponse(const GaussiansData& data, unsigned int gs_id,
                     const float3 o, const float3 d, float& resp, float& tmax){
    const float3 mu = data.xyz[gs_id];
    const float3 s = data.scaling[gs_id];
    Matrix3x3 R = _construct_rotation(data.rotation[gs_id]).transpose();
    constexpr float eps = 1e-6;
    R.setRow(0,R.getRow(0)/(s.x+eps));
    R.setRow(1,R.getRow(1)/(s.y+eps));
    R.setRow(2,R.getRow(2)/(s.z+eps));
    float3 og = R*(mu-o);
    float3 dg = R*d;
    tmax = dot(og,dg)/(dot(dg,dg)+eps);
    float3 samp = o+tmax*d;
    float3 x = R*(samp-mu);
    resp = data.opacity[gs_id]*exp(-dot(x,x));
}

__device__ float3 _computeRadiance(const float3* sh, int sh_deg, const float3 mu, const float3 &ray_origin){

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

__device__ bool _intersectAABB(const float3 min, const float3 max,
                    const float3 orig, const float3 dir,
                    float& tmin, float& tmax){
    float3 idir = 1.f/dir;
    float3 tc1 = (min-orig)*idir;
    float3 tc2 = (max-orig)*idir;
    float3 tc_min = fminf(tc1,tc2);
    float3 tc_max = fmaxf(tc1,tc2);
    tmin = fmax(tc_min.x,fmax(tc_min.y,tc_min.z));
    tmax = fmin(tc_max.x,fmin(tc_max.y,tc_max.z));
    return tmin < tmax;
}

#define BLOCK_DIM 16


struct Hit{int id; float resp, thit;};

__device__ __forceinline__ void pushHit(Hit* hitq, unsigned int& hitq_size, const Hit& hit){
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

__device__ __forceinline__ void popHit(Hit* hitq, unsigned int& hitq_size){
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


__device__ void process_hit(Hit hit, const GaussiansData& data, float& trans, float3& rad, float3 orig){

	rad += hit.resp*trans*_computeRadiance(data.sh+hit.id*16, data.sh_deg, data.xyz[hit.id], orig);
	trans *= (1-hit.resp);
}

__global__ void _rcast_kd_restart(
	TracingParams params, AABB scene_vol,
	const AABB* aabbs, const Node* nodes, const int* leaves_data,
	GaussiansData data
	) {

	int i = (blockDim.y*blockIdx.y + threadIdx.y) * params.width
			+ blockDim.x*blockIdx.x + threadIdx.x;
	if(i >= params.width*params.height) return;
	
	const float3 orig = params.ray_origins[i];
	const float3 dir = params.ray_directions[i];

	float sceneMin,sceneMax;
	if(!_intersectAABB(scene_vol.min,scene_vol.max,orig,dir,sceneMin,sceneMax))
		return;

	float tmin,tmax;
	tmin = tmax = sceneMin;
	int iroot = 0; 
	float3 acc_rad = make_float3(0.);
	float acc_trans = 1.f;
	char leaf_plane_mask = 0;
	constexpr float Tmin = 0.001;
	constexpr float respMin = 0.01f;
	while(tmax < sceneMax){
		int inode = iroot;
		Node node = nodes[inode];
		tmin = tmax;
		tmax = sceneMax;
		bool pushdown = true;
		char n_leaf_plane_mask = 0;
		while(!node.isleaf()){
			//if(tracing_params.draw_kd)
			//	draw_aabb(tracing_params,inode,i);
			int ax = node.axis();
			float tsplit = (node.cplane()-vec_c(orig,ax))/vec_c(dir,ax);
			int ifirst = inode+1;
			int isecond = node.has_right()? node.right_id(): -1;
			//if(vec_c(orig,ax)>node.cplane()) std::swap(ifirst,isecond);
			if(vec_c(dir,ax)<0){
				int s = isecond; isecond = ifirst; ifirst = s;
			}
			if(tsplit >= tmax){// || tsplit < 0){
				inode = ifirst; 
			}else if(tsplit <= tmin){
				inode = isecond; 
			}else{
				inode = ifirst;
				tmax = tsplit;
				pushdown = false;
				bool ismax = vec_c(dir,ax) > 0;
				// ignore-mask for the particles on the opposite side
				n_leaf_plane_mask = 1 << (ax*2 + !ismax); 
			}
			if(pushdown){
				iroot=inode;
			}
			if(inode == -1) break;
			node = nodes[inode];
		}
		if(inode != -1 && node.isleaf()){
			//if(tracing_params.draw_kd)
			//	draw_aabb(tracing_params,inode,i);
			if(node.is_leaf_empty()) continue;
			constexpr long hits_cap = 2048;
			Hit hits[hits_cap];
			unsigned int hitq_size = 0;
			// get particle intersections
			int leaf_size = *(leaves_data+node.data_id());
			const int *part_ids = leaves_data+node.data_id()+1;
			const char* plane_masks = reinterpret_cast<const char*>(leaves_data+node.data_id()+1+leaf_size);

			//params.radiance[i] = float3{1.,0.,0.};
			for(int k = 0; k < leaf_size; ++k){
				if(plane_masks[i] & leaf_plane_mask)
				 	continue; // -> this particle was processed in the previous leaf
				
				int part_id = part_ids[k];
				float part_tmin,part_tmax;
				if(_intersectAABB(aabbs[part_id].min,aabbs[part_id].max,orig,dir,part_tmin,part_tmax))
				{
					//params.radiance[i] = float3{0.,1.,0.};
					if(hitq_size == hits_cap){
						process_hit(hits[0],data,acc_trans,acc_rad,orig);
						popHit(hits,hitq_size);
						if(acc_trans < Tmin){
							tmax = sceneMax;
							break;
						}
					}
					float resp,tmaxresp;
					_computeResponse(data,part_id,orig,dir,resp,tmaxresp);
					if(resp < respMin) continue;
					Hit hit{part_id,resp,tmaxresp};
					pushHit(hits,hitq_size,hit);
				}
			}
			while(hitq_size != 0){
				process_hit(hits[0],data,acc_trans,acc_rad,orig);
				popHit(hits,hitq_size);
				if(acc_trans < Tmin){
					tmax = sceneMax;
					break;
				}
			}
			// set the particle mask for the next leaf
			leaf_plane_mask = n_leaf_plane_mask;
			// process hits
			// for(int k = 0; k < hits; ++k){
			// 	for(int c = 0; c < hits; ++c){}
			// }
			//for(const Hit& hit : hits){
			//	float3 rad = hit.rad;//make_float3(1.);
			//	acc_rad += rad*hit.resp*acc_trans;
			//	acc_trans *= (1-hit.resp);
			//	(*tracing_params.num_its)++;
			//}
		}
	}
	params.radiance[i] += acc_rad;
}


void CUDA_Traversal::rcast_kd_restart(const TracingParams& params){
#ifdef PRESORT
	float3 orig;
	cudaMemcpy(&orig,params.ray_origins,sizeof(float3),cudaMemcpyDeviceToHost);
	for(auto& node : nodes){
		if(!node.isleaf()) continue;
		int leaf_size = *(leaves_data.data()+node.data_id());
		int *part_ids = leaves_data.data()+node.data_id()+1;
		char* plane_masks = reinterpret_cast<char*>(leaves_data.data()+node.data_id()+1+leaf_size);
		std::vector<float> dists(leaf_size);
		std::vector<int> parts_asort(leaf_size);
		for(int i = 0; i < leaf_size; ++i){
			assert(part_ids[i] < gaussians.numgs && part_ids[i]>=0);
			dists[i] = length(gaussians.xyz[part_ids[i]]-orig);
			parts_asort[i] = i;
		}
		std::sort(parts_asort.begin(),parts_asort.end(),
			[&](int i, int j){
				return dists[i] < dists[j];
			});
		std::vector<int> n_part_ids(leaf_size);
		std::vector<char> n_plane_masks(leaf_size);
		for(int i = 0; i < leaf_size; ++i){
			n_part_ids[i] = part_ids[parts_asort[i]];
			n_plane_masks[i] = plane_masks[parts_asort[i]];
		}
		memcpy(part_ids,n_part_ids.data(),leaf_size*sizeof(int));
		memcpy(plane_masks,n_plane_masks.data(),leaf_size);
	}
	if(d_leaves == 0){
		CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_leaves), leaves_data.size()*sizeof(int)),true);
	}
	CHECK_CUDA(cudaMemcpy(d_leaves, leaves_data.data(),
						 leaves_data.size()*sizeof(int),cudaMemcpyHostToDevice),true);
#endif

	dim3 blocks((params.width+BLOCK_DIM-1)/BLOCK_DIM,(params.height+BLOCK_DIM-1)/BLOCK_DIM);
	dim3 threads(BLOCK_DIM,BLOCK_DIM);

	printf("gpu_trace %dx%d %dx%d \n", blocks.x,blocks.y, threads.x,threads.y);
	_rcast_kd_restart<<<blocks,threads>>>(
	 	params,scene_vol,d_aabbs,d_nodes,d_leaves,d_gaussians
		);
	CHECK_CUDA(,true);
}

CUDA_Traversal::CUDA_Traversal(GaussiansKDTree& kdtree){

	std::cout << "initializing cuda" << std::endl;

    auto toDevice = [&](auto& dst, void* src, size_t size){
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dst), size),true);
        CHECK_CUDA(cudaMemcpy(reinterpret_cast<void*>(dst), src, size, cudaMemcpyHostToDevice),true);
    };
	const auto& data = kdtree.data;
    d_gaussians.numgs = data.numgs;
    toDevice(d_gaussians.xyz, data.xyz, data.numgs*sizeof(float3));
    toDevice(d_gaussians.rotation, data.rotation, data.numgs*sizeof(float4));
    toDevice(d_gaussians.scaling, data.scaling, data.numgs*sizeof(float3));
    toDevice(d_gaussians.opacity, data.opacity, data.numgs*sizeof(float));
    toDevice(d_gaussians.sh, data.sh, data.numgs*sizeof(float3)*16);
	toDevice(d_aabbs, kdtree.aabbs.data(), kdtree.aabbs.size()*sizeof(AABB));

	scene_vol = kdtree.scene_vol;

	std::vector<int> leaves_data;
	int offt = 0;
	std::vector<int> n_leaf_data_ids; 
	for(const auto& ld : kdtree.leaves_data){
		int start = offt;
		n_leaf_data_ids.push_back(start);
		int num_part = ld.part_ids.size();
		int masks_size = (num_part+3)>>2;
		offt += 1+num_part+masks_size;
		leaves_data.resize(offt);
		memcpy(leaves_data.data()+start,&num_part,sizeof(int));
		memcpy(leaves_data.data()+start+1,ld.part_ids.data(),num_part*sizeof(int));
		memcpy(leaves_data.data()+start+1+num_part,ld.plane_masks.data(),num_part);
	}
	assert(leaves_data.size()==offt);
	std::vector<Node> nodes(kdtree.nodes.begin(),kdtree.nodes.end());
	for(auto& node : nodes){
		if(node.isleaf()){
			node.set_data_id(n_leaf_data_ids[node.data_id()]);
		}
	}
	toDevice(d_nodes, nodes.data(), nodes.size()*sizeof(Node));
	toDevice(d_leaves, leaves_data.data(), leaves_data.size()*sizeof(int));
	this->nodes = std::move(nodes);
	this->leaves_data = std::move(leaves_data);
	this->gaussians = kdtree.data;
}

CUDA_Traversal::~CUDA_Traversal(){

    CHECK_CUDA(cudaFree(d_gaussians.xyz),true);
    CHECK_CUDA(cudaFree(d_gaussians.rotation),true);
    CHECK_CUDA(cudaFree(d_gaussians.scaling),true);
    CHECK_CUDA(cudaFree(d_gaussians.opacity),true);
    CHECK_CUDA(cudaFree(d_gaussians.sh),true);
    CHECK_CUDA(cudaFree(d_aabbs),true);
    CHECK_CUDA(cudaFree(d_nodes),true);
    CHECK_CUDA(cudaFree(d_leaves),true);
}

