#include "traverser.h"
#include "utils/vec_math.h"
#include "utils/geom.h"
#include "utils/gs_tracing.h"
#include <cmath>
#include <limits>
#include <iostream>
#include <algorithm>
using namespace util;
using namespace util::gs_tracing;

namespace {

bool intersectAABB(const float3 min, const float3 max,
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

int its2planeId(const float3 its, const float3 min, const float3 max){
    int side = 0;
    float mindist = __FLT_MAX__;
    for(int k = 0; k < 3; ++k){
        float c = vec_c(its,k);
        if(std::abs(vec_c(min,k)-c) < mindist){
            side = k*2;
        }
        if(std::abs(vec_c(max,k)-c) < mindist){
            side = k*2+1;
        }
    }
    return side;
}

bool __intersectAABB(const float3 min, const float3 max,
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
}

namespace gsrt::kd_tracer::host::traverser {

void rcast_kd_restart(const ASData_Host& as, const TracingParams &tracing_params){
    std::cout << "tracing..." << std::endl; 

    std::cout << "num. nodes: " << as.nodes.size() << std::endl;
    for(int i = 0; i < tracing_params.rays.num_rays; ++i){
        const float3 orig = tracing_params.rays.ray_origins[i];
        const float3 dir = tracing_params.rays.ray_directions[i];
        float sceneMin,sceneMax;
        if(!intersectAABB(as.scene_vol.min, as.scene_vol.max, orig, dir, sceneMin, sceneMax))
            continue;
        float tmin,tmax;
        tmin = tmax = sceneMin;
        int iroot = 0; 
        float3 acc_rad = make_float3(0.);
        float acc_trans = 1.f;
        //std::vector<int> stack;

        char leaf_plane_mask = 0;
        while(tmax < sceneMax){
            int inode = iroot;
            Node node = as.nodes.at(inode);
            tmin = tmax;
            tmax = sceneMax;
            bool pushdown = true;
            char n_leaf_plane_mask = 0;
            //std::cout << "restart " << inode << std::endl;
            while(!node.isleaf()){
                //if(tracing_params.draw_kd)
                //    draw_aabb(tracing_params,inode,i);
                int ax = node.axis();
                float tsplit = (node.cplane()-vec_c(orig,ax))/vec_c(dir,ax);
                int ifirst = inode+1;
                int isecond = node.has_right()? node.right_id(): -1;
                //if(vec_c(orig,ax)>node.cplane()) std::swap(ifirst,isecond);
                if(vec_c(dir,ax)<0) std::swap(ifirst,isecond);
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
                node = as.nodes.at(inode);
            }
            if(inode != -1 && node.isleaf()){
                //if(tracing_params.draw_kd)
                //    draw_aabb(tracing_params,inode,i);
                if(node.is_leaf_empty()) continue;
                //std::cout << "leaf" << std::endl;
                const auto& leaf_data = as.leaves_data[node.data_id()];
                struct Hit{int id; float resp, tmax; float3 rad;};
                std::vector<Hit> hits;
                // get particle intersections
                for(int i = 0; i < leaf_data.part_ids.size(); ++i){

                    if(leaf_data.plane_masks[i] & leaf_plane_mask){ 
                        continue; // -> this particle was processed in the previous leaf
                    }
                    
                    int part_id = leaf_data.part_ids[i];
                    float part_tmin,part_tmax;
                    if(intersectAABB(as.aabbs[part_id].min,as.aabbs[part_id].max,orig,dir,part_tmin,part_tmax)
                        // && part_tmin >= tmin
                        )
                         {
                        float resp,tmaxresp;
                        computeResponse(as.data.xyz,as.data.rotation,as.data.scaling,as.data.opacity,
                            part_id,orig,dir,resp,tmaxresp);
                        float3 rad = computeRadiance(as.data.sh+part_id*16, as.data.sh_deg, as.data.xyz[part_id],orig);
                        // if(part_tmin < tmin){
                        //      rad = float3{0.f,1.f,0.f};
                        //      resp = 1.;
                        // }
                        //if(leaf_data.plane_masks[i] != 0) std::cout << leaf_data.plane_masks[i] << std::endl;
                        // if(leaf_data.plane_masks[i] & leaf_plane_mask){
                        //     rad = float3{0.f,1.f,0.f};
                        //     resp = 1.;
                        // }
                        hits.push_back({part_id,resp,tmaxresp,rad});
                    }
                }
                // set the particle mask for the next leaf
                leaf_plane_mask = n_leaf_plane_mask;
                // process hits
                std::sort(hits.begin(),hits.end(),
                    [](const Hit& a, const Hit& b){return a.tmax<b.tmax;});
                for(const Hit& hit : hits){
                    float3 rad = hit.rad;//make_float3(1.);
                    acc_rad += rad*hit.resp*acc_trans;
                    acc_trans *= (1-hit.resp);
                    (*tracing_params.output.num_its)++;
                }
            }
        }
        tracing_params.output.radiance[i] += acc_rad;
    }
}

} //namespace gsrt::kd_tracer::host::traverser