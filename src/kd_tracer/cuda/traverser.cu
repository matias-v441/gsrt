#include "traverser.h"

#include <stdio.h>
#include <assert.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "utils/Matrix.h"
#include "utils/gs_tracing.h"
#include "utils/exception_cuda.h"

using namespace util;
using namespace util::gs_tracing;
using namespace gsrt;
using namespace gsrt::kd_tracer;
using namespace gsrt::kd_tracer::cuda::traverser;

#define BLOCK_DIM 16

namespace gsrt::kd_tracer::cuda::traverser {

namespace {

__device__ bool _intersectAABB(const float3 min, const float3 max,
                    const float3 orig, const float3 dir,
                    float& tmin, float& tmax, char& closest_plane_mask){
    float3 idir = 1.f/dir;
    float3 tc1 = (min-orig)*idir;
    float3 tc2 = (max-orig)*idir;
    float3 tc_min = fminf(tc1,tc2);
    float3 tc_max = fmaxf(tc1,tc2);
    tmin = fmax(tc_min.x,fmax(tc_min.y,tc_min.z));
	int ax = 0;
	if(tmin == tc_min.y) ax = 1;
	if(tmin == tc_min.z) ax = 2;
	closest_plane_mask = 1 << (ax*2 + (vec_c(dir,ax) > 0));
    tmax = fmin(tc_max.x,fmin(tc_max.y,tc_max.z));
    return tmin < tmax;
}

__device__ void draw_aabb(const RayData& rays, const RenderOutput& output, const AABB* node_aabbs, int node_id, int ray_id, float3 color){
    const float3 orig = rays.ray_origins[ray_id];
    const float3 dir = rays.ray_directions[ray_id];
    float aabb_tmin,aabb_tmax;char _;
    if(_intersectAABB(node_aabbs[node_id].min,node_aabbs[node_id].max,orig,dir,aabb_tmin,aabb_tmax,_)){
        for(float t : {aabb_tmin,aabb_tmax}){
            float3 pos = t*dir+orig;
            float3 dmin = pos-node_aabbs[node_id].min;
            float3 dmax = node_aabbs[node_id].max-pos;
            const float thr = .005f;
            int cmin = 0;
            int cmax = 0;
            for(int k = 0; k < 3; ++k){
                if(vec_c(dmin,k)<thr){
                    ++cmin;
                }
				if(vec_c(dmax,k)<thr){
                    ++cmax;
                }
            }
            if(cmin == 2 || cmax == 2){
                output.radiance[ray_id] = color;
            }
        }
        //tracing_params.radiance[ray_id] += float3{1.,0.,0.};
    }
}

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

	rad += hit.resp*trans*computeRadiance(data.sh+hit.id*16, data.sh_deg, data.xyz[hit.id], orig);
	trans *= (1-hit.resp);
}
}

__global__ void _rcast_kd_restart(ASData_Device::DevData as, RayData rays, RenderOutput output) {

	int i = (blockDim.y*blockIdx.y + threadIdx.y) * rays.width
			+ blockDim.x*blockIdx.x + threadIdx.x;
	if(i >= rays.width*rays.height) return;

	const float3 orig = rays.ray_origins[i];
	const float3 dir = rays.ray_directions[i];

	float sceneMin,sceneMax;char _;
	if(!_intersectAABB(as.scene_vol.min,as.scene_vol.max,orig,dir,sceneMin,sceneMax,_))
		return;

	float tmin,tmax;
	tmin = tmax = sceneMin;
	int iroot = 0; 
	float3 acc_rad = make_float3(0.);
	float acc_trans = 1.f;
	char leaf_plane_mask = 0;
	constexpr float trans_min = 0.001;
	constexpr float resp_min = 0.01f;
	bool first_leaf = true;
	while(tmax < sceneMax){
		int inode = iroot;
		Node node = as.nodes[inode];
		tmin = tmax;
		tmax = sceneMax;
		bool pushdown = true;
		//char n_leaf_plane_mask = init_plane_mask;
		unsigned long long n_traversal_steps = 0;
		while(!node.isleaf()){
			//if(params.draw_kd) draw_aabb(params,node_aabbs,inode,i,{1.,0.,0.});
			int ax = node.axis();
			float tsplit = (node.cplane()-vec_c(orig,ax))/vec_c(dir,ax);
			int ifirst = inode+1;
			int isecond = node.right_id();//node.has_right()? node.right_id(): -1;
#define DIR_BASED
//#define ORIGIN_BASED
#ifdef DIR_BASED
			if(vec_c(dir,ax)<0){
				int s = isecond; isecond = ifirst; ifirst = s;
			}
			if(tsplit >= tmax){// || tsplit < 0){
				inode = ifirst; 
			}else if(tsplit <= tmin){
				inode = isecond; 
			}else
#endif
#ifdef ORIGIN_BASED
			if(vec_c(orig,ax)>node.cplane()) {
				int s = isecond; isecond = ifirst; ifirst = s;
			}
			if(tsplit >= tmax || tsplit < 0){
				inode = ifirst;
			}else if(tsplit <= tmin){
				inode = isecond;
			}else
#endif
			{
				inode = ifirst;
				tmax = tsplit;
				pushdown = false;
				// ignore-mask for the particles on the opposite side
				//n_leaf_plane_mask = 1 << (ax*2 + (vec_c(dir,ax) <= 0)); 
			}
			if(pushdown){
				iroot=inode;
			}
			if(inode == -1) break;
			node = as.nodes[inode];
			n_traversal_steps++;
		}
		if(inode != -1 && node.isleaf()){
			float leaf_tmin,leaf_tmax; char closest_plane_mask;
			if(!_intersectAABB(as.node_aabbs[inode].min,as.node_aabbs[inode].max,orig,dir,leaf_tmin,leaf_tmax,closest_plane_mask)){
				assert(false);
				//params.radiance[i] = float3{1.f,0.f,0.f};return;
			}
			if(!first_leaf){
				leaf_plane_mask = closest_plane_mask;
			}
			first_leaf = false;
			if(node.is_leaf_empty()){
				//if(params.draw_kd) draw_aabb(params,node_aabbs,inode,i,{0.,0.,1.});
				continue;
			}
			Hit hits[HIT_BUFFER_SIZE];
			unsigned int hitq_size = 0;
			// get particle intersections
			int leaf_size = *(as.leaves+node.data_id());
			const int *part_ids = as.leaves+node.data_id()+1;
			const char* plane_masks = reinterpret_cast<const char*>(as.leaves+node.data_id()+1+leaf_size);

			for(int k = 0; k < leaf_size; ++k){
				//if(plane_masks[i] & leaf_plane_mask)
				// 	continue; // -> this particle was processed in the previous leaf
				
				int part_id = part_ids[k];
				float part_tmin,part_tmax;char _;
				if(_intersectAABB(as.aabbs[part_id].min,as.aabbs[part_id].max,orig,dir,part_tmin,part_tmax,_)
					//&& part_tmin < leaf_tmin
				)
				{
					float resp,tmaxresp;
					computeResponse(as.data.xyz,as.data.rotation,as.data.scaling,as.data.opacity,
						part_id,orig,dir,resp,tmaxresp);
					if(tmaxresp >= leaf_tmin && tmaxresp <= leaf_tmax && 
						resp >= resp_min){
						if(hitq_size == HIT_BUFFER_SIZE){
							process_hit(hits[0],as.data,acc_trans,acc_rad,orig);
							popHit(hits,hitq_size);
							if(acc_trans < trans_min){
								tmax = sceneMax;
								break;
							}
						}
						atomicAdd(output.num_its,1ull);
						Hit hit{part_id,resp,tmaxresp};
						pushHit(hits,hitq_size,hit);
					}
				}
			}
			while(hitq_size != 0){
				process_hit(hits[0],as.data,acc_trans,acc_rad,orig);
				popHit(hits,hitq_size);
				if(acc_trans < trans_min){
					tmax = sceneMax;
					break;
				}
			}
			//draw_aabb(params,node_aabbs,inode,i,{0.,1.,0.});
			// set the particle mask for the next leaf
			//leaf_plane_mask = n_leaf_plane_mask;
			//leaf_plane_mask = next_plane_mask;
		}
	}
	output.radiance[i] = acc_rad;
}

void rcast_kd_restart(const ASData_Device& data, const RayData& rays, const RenderOutput& output){
//#define PRESORT
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

	dim3 blocks((rays.width+BLOCK_DIM-1)/BLOCK_DIM,(rays.height+BLOCK_DIM-1)/BLOCK_DIM);
	dim3 threads(BLOCK_DIM,BLOCK_DIM);

	_rcast_kd_restart<<<blocks,threads>>>(data._dd,rays,output);
}

} //namespace gsrt::kd_tracer::cuda::traverser
