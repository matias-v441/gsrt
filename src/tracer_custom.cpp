#include "tracer_custom.h"

#include "utils/exception.h"
#include "utils/vec_math.h"
#include "utils/gsrt_util.h"
#include "utils/Matrix.h"

#include <cassert>

#include <algorithm>

using namespace util;


void GaussiansKDTree::build(){
    std::cout << "building..." << std::endl;

    using gsrt_util::icosahedron::n_verts;
    const float alpha_min = .01;
    scene_vol = AABB{make_float3(__FLT_MAX__),make_float3(-__FLT_MAX__)};
    aabbs.resize(data.numgs,scene_vol);
    for(int i = 0; i < data.numgs; ++i){
        const Matrix3x3 R = gsrt_util::construct_rotation(data.rotation[i]);
        float adaptive_scale = sqrt(2.*log(data.opacity[i]/alpha_min));
        float3 s = data.scaling[i]*adaptive_scale;
        for(int j = 0; j < n_verts; ++j){
            float3 v = gsrt_util::icosahedron::vertices[j];
            float3 w = R*(s*v)+data.xyz[i];
            aabbs[i].min = fminf(aabbs[i].min,w);
            aabbs[i].max = fmaxf(aabbs[i].max,w);
        }
        scene_vol.min = fminf(aabbs[i].min,scene_vol.min);
        scene_vol.max = fmaxf(aabbs[i].max,scene_vol.max);
    }
    std::array<std::vector<int>,3> axis_events_asort;
    for(int axis = 0; axis < 3; ++axis){
        std::vector<float> events(data.numgs*2);
        std::vector<int> event_ids(data.numgs*2);
        for(int i = 0; i < data.numgs; ++i){
            int si = i;
            int ei = i+data.numgs;
            events[si] = vec_c(aabbs[i].min,axis);
            events[ei] = vec_c(aabbs[i].max,axis);
            event_ids[si] = si;
            event_ids[ei] = ei;
        }
        std::sort(event_ids.begin(),event_ids.end(),
            [&](int i, int j){return events[i]<events[j];});
        axis_events_asort[axis] = std::move(event_ids);
        axis_events[axis] = std::move(events);
    }
    build_rec(scene_vol,axis_events_asort,data.numgs,0);
    build_check();
    std::cout << "num. nodes: " << nodes.size() << std::endl;
    std::cout << "num. leaves: " << leaves_data.size() << std::endl;
    int max_leaf_size = 0;
    for(auto& leaf : leaves_data){
        max_leaf_size = max(leaf.part_ids.size(),max_leaf_size);
    }
    std::cout << "max leaf size: " << max_leaf_size << std::endl;

    cuda_traversal = std::make_unique<CUDA_Traversal>(*this);
}

void GaussiansKDTree::build_rec(AABB V,
                                axis_ev_ids evs,
                                int num_part,
                                int depth){
    
    int axis = depth%3; // round-robin

    int node_id = nodes.size();
    nodes.emplace_back();

    node_aabbs.push_back(V);

    if(num_part == 0){
        nodes[node_id].set_leaf_empty();
        return;
    }

    SplitPlane best_plane = find_best_plane(axis, V, evs, num_part);
    
    if(depth == maxdepth() ||
        best_plane.coord == vec_c(V.min,axis) || // same as no split
        best_plane.cost > cost(0.f,1.f,0,num_part) // split is worse than no split
        ){
        nodes[node_id].set_data_id(leaves_data.size());
        fill_leaf_particles(leaves_data.emplace_back(),V,evs);
        return;
    }
    nodes[node_id].set_axis(axis);
    nodes[node_id].set_cplane(best_plane.coord);

    auto [Vleft,Vright] = split_volume(axis,best_plane.coord,V);
    auto [evs_left,evs_right] = split_events(evs,axis,best_plane.coord);
    
                        // dont add empty leaves
    bool go_left = true;//best_plane.num_left != 0;
    bool go_right = true;//best_plane.num_right != 0;
    if(go_left){
        // printf("left volume min %f max %f\n",vec_c(Vleft.min,axis),vec_c(Vleft.max,axis));
        build_rec(Vleft,evs_left,best_plane.num_left,depth+1);
        if(go_right){
            nodes[node_id].set_right_id(nodes.size());
        }
    }
    if(go_right){
        // printf("right volume min %f max %f\n",vec_c(Vright.min,axis),vec_c(Vright.max,axis));
        build_rec(Vright,evs_right,best_plane.num_right,depth+1);
    }
}


void GaussiansKDTree::build_check(){
    for(int i = 0; i < nodes.size(); ++i){
        if(nodes[i].isleaf()) continue;
        assert(nodes[i].has_right());
        int ax = nodes[i].axis();
        assert(nodes[i].cplane()==vec_c(node_aabbs[i+1].max,ax));
        assert(nodes[i].cplane()==vec_c(node_aabbs[nodes[i].right_id()].min,ax));
    }
}

std::pair<axis_ev_ids,axis_ev_ids> GaussiansKDTree::split_events(
    axis_ev_ids& evs,int split_ax,float csplit) const{

    axis_ev_ids left;
    axis_ev_ids right;
    for(int k = 1; k < 3; ++k){
        int ax = (split_ax+k)%3;
        for(int ev_id : evs[ax]){
            auto aabb = aabbs[ev2part_id(ev_id)];
            float start_ev = vec_c(aabb.min,split_ax);
            float end_ev = vec_c(aabb.max,split_ax);
            if(start_ev<csplit){
                left[ax].push_back(ev_id);
            }
            if(end_ev>csplit){
                right[ax].push_back(ev_id);
            }
        }
    }
    for(int ev_id : evs[split_ax]){
        float ev = axis_events[split_ax][ev_id];
        if(ev < csplit || (ev == csplit && is_end_ev(ev_id))){
            left[split_ax].push_back(ev_id);
        }else{
            right[split_ax].push_back(ev_id);
        }
    }
    return {left,right};
}

void GaussiansKDTree::fill_leaf_particles(LeafData &leaf, AABB V,
             const axis_ev_ids& evs) const{

    // for(int ev_id : evs[0]){
    //     int part_id = ev2part_id(ev_id);
    //     if(is_start_ev(ev_id) || aabbs[part_id].min.x < V.min.x){
    //         leaf.part_ids.push_back(part_id);
    //     }
    // }
    for(int ev_id : evs[0]){
        int part_id = ev2part_id(ev_id);
        if(is_start_ev(ev_id) || aabbs[part_id].min.x < V.min.x){
            // add particle
            leaf.part_ids.push_back(part_id);
            // calculate plane mask
            char plane_mask = 0;
            for(int k = 0; k < 3; ++k){
                if(vec_c(aabbs[part_id].min,k)<vec_c(V.min,k)){
                    plane_mask |= 1<<(k*2);
                }
                if(vec_c(aabbs[part_id].max,k)>vec_c(V.max,k)){
                    plane_mask |= 1<<(k*2+1);
                }
            }
            leaf.plane_masks.push_back(plane_mask);
        }
    }
}

GaussiansKDTree::SplitPlane GaussiansKDTree::find_best_plane(int axis, AABB V, axis_ev_ids evs, int num_part) const{
    // printf("FBP vol_min=%f vol_max=%f\n",evs[0].size(),evs[1].size(),evs[2].size(),
    //     vec_c(V.min,axis),vec_c(V.max,axis));
    // for(int ev : evs[axis]){
    //     printf("%f ",axis_events[axis][ev]);
    // }
    // std::cout << std::endl;
    // for(int ev : evs[axis]){
    //     if(ev < data.numgs)
    //         printf("%ds ",ev2part_id(ev));
    //     else
    //         printf("%de ",ev2part_id(ev));
    // }
    // std::cout << std::endl;
    // for(int ev_id : evs[axis]){ 
    //     auto aabb = aabbs[ev2part_id(ev_id)];
    //     printf("%f ", vec_c(aabb.min,axis));
    // }
    // std::cout << std::endl;
    int num_left=0,num_right=num_part;
    // add all nodes started outside the volume to the left side
    for(int ev_id : evs[axis]){ 
        auto aabb = aabbs[ev2part_id(ev_id)];
        if(vec_c(aabb.min,axis)<vec_c(V.min,axis)){
            ++num_left;
        }
    }
    SplitPlane best_plane;
    best_plane.cost = __FLT_MAX__;
    // find split with minimum SAH
    for(int ev_id : evs[axis]){
        if(is_end_ev(ev_id)){
            --num_right;
        }
        float csplit = axis_events[axis][ev_id];
        // printf("SAH ARGS %d %f %d %d\n",axis,csplit,num_left,num_right);
        float cost = SAH(axis,csplit,V,num_left,num_right);
        // std::cout << cost << std::endl;
        if(cost<best_plane.cost){
            best_plane.cost = cost;
            best_plane.coord = csplit;
            best_plane.num_left = num_left;
            best_plane.num_right = num_right;
        }
        if(is_start_ev(ev_id)){
            ++num_left;
        }
    }
    // printf("FBP cost=%f num_left=%d num_right=%d\n",
    //     best_plane.cost,best_plane.num_left,best_plane.num_right);
    return best_plane;
}

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

void computeResponse(const GaussiansData& data, unsigned int gs_id,
                     const float3 o, const float3 d, float& resp, float& tmax){
    const float3 mu = data.xyz[gs_id];
    const float3 s = data.scaling[gs_id];
    Matrix3x3 R = gsrt_util::construct_rotation(data.rotation[gs_id]).transpose();
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

const float SH_C0 = 0.28209479177387814f;
const float SH_C1 = 0.4886025119029199f;
const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

float3 computeRadiance(const float3* sh, int sh_deg, const float3 mu, const float3 &ray_origin){

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

void GaussiansKDTree::rcast_linear(const TracingParams &tracing_params){
    std::cout << "tracing..." << std::endl; 
    if(tracing_params.draw_kd)
        rcast_draw_kd(tracing_params);
    for(int i = 0; i < tracing_params.num_rays; ++i){
        const float3 orig = tracing_params.ray_origins[i];
        const float3 dir = tracing_params.ray_directions[i];
        float sceneMin,sceneMax;
        if(!intersectAABB(scene_vol.min,scene_vol.max,orig,dir,sceneMin,sceneMax))
            continue;
        struct Hit{int id; float resp, tmax;};
        std::vector<Hit> hits;
        //for(int j = 0; j < data.numgs; ++j)
        for(const auto& leaf: leaves_data)
        for(int j : leaf.part_ids)
        {
            float tmin,tmax;
            if(intersectAABB(aabbs[j].min,aabbs[j].max,orig,dir,tmin,tmax)){
                float resp,tmax;
                computeResponse(data,j,orig,dir,resp,tmax);
                hits.push_back({j,resp,tmax});
            }
        }
        std::sort(hits.begin(),hits.end(),
            [](const Hit& a, const Hit& b){return a.tmax<b.tmax;});
        float3 rad_acc = make_float3(0.);
        float trans = 1.f;
        float3 color = make_float3(1.);
        for(const Hit& hit : hits){
            float3 rad = computeRadiance(data.sh+hit.id*16, data.sh_deg, data.xyz[hit.id],orig);
            rad_acc += rad*hit.resp*trans;
            trans *= (1-hit.resp);
            (*tracing_params.num_its)++;
        }
        tracing_params.radiance[i] += rad_acc;
    }
}

void GaussiansKDTree::rcast_kd(const TracingParams &tracing_params){
    std::cout << "tracing..." << std::endl; 
    //rcast_draw_kd(tracing_params);
    std::cout << "num. nodes: " << nodes.size() << std::endl;
    for(int i = 0; i < tracing_params.num_rays; ++i)
    {  
        const float3 orig = tracing_params.ray_origins[i];
        const float3 dir = tracing_params.ray_directions[i];
        float sceneMin,sceneMax;
        if(!intersectAABB(scene_vol.min,scene_vol.max,orig,dir,sceneMin,sceneMax)){
            continue;
        }
        float3 acc_rad = make_float3(0.);
        float acc_trans = 1.f;
        struct stack_node{
            int i;
            float tmin,tmax;
        };
        std::vector<stack_node> stack;
        stack.push_back({0,sceneMin,sceneMax});
        while(!stack.empty()){
            stack_node snode = stack.back();
            Node node = nodes[snode.i];
            stack.pop_back();
            
            int depth = 0;
            while(!node.isleaf()){
                depth++;
                //if(tracing_params.draw_kd && depth <= 2)
                //    draw_aabb(tracing_params,snode.i,i);
                int ax = node.axis();
                float tsplit = (node.cplane()-vec_c(orig,ax))/vec_c(dir,ax);
                int ifirst = snode.i+1;
                int isecond = node.has_right()? node.right_id(): -1;
                float3 orig_ = orig + dir*snode.tmin*.5f;
                //if(vec_c(orig_,ax)>node.cplane()) std::swap(ifirst,isecond);
                if(vec_c(dir,ax)<0) std::swap(ifirst,isecond);

                if(tsplit >= snode.tmax)// || tsplit < 0)
                {
                    snode.i = ifirst;
                }else if(tsplit <= snode.tmin){
                    snode.i = isecond; 
                }else
                {
                    //if(isecond != -1)
                        stack.push_back({isecond,tsplit,snode.tmax});
                    snode.i = ifirst; 
                    snode.tmax = tsplit;
                }
                //if(snode.i == -1) break;
                node = nodes[snode.i];
            }
            if(snode.i != -1 && node.isleaf()){
                if(node.is_leaf_empty()) continue;
                // if(tracing_params.draw_kd)
                //     draw_aabb(tracing_params,snode.i,i);
                struct Hit{int id; float resp, tmax;};
                std::vector<Hit> hits;
                for(int part_id : leaves_data.at(node.data_id()).part_ids){
                    float tmin,tmax;
                    if(intersectAABB(aabbs[part_id].min,aabbs[part_id].max,orig,dir,tmin,tmax) && tmin >= snode.tmin){
                        float resp,tmax;
                        computeResponse(data,part_id,orig,dir,resp,tmax);
                        hits.push_back({part_id,resp,tmax});
                    }
                }
                std::sort(hits.begin(),hits.end(),
                    [](const Hit& a, const Hit& b){return a.tmax<b.tmax;});
                for(const Hit& hit : hits){
                    float3 rad = computeRadiance(data.sh+hit.id*16, data.sh_deg, data.xyz[hit.id],orig);
                    acc_rad += rad*hit.resp*acc_trans;
                    acc_trans *= (1-hit.resp);
                    (*tracing_params.num_its)++;
                }
            }
        }
        tracing_params.radiance[i] += acc_rad;
    }
}


void GaussiansKDTree::rcast_gpu(const TracingParams& params){
    std::cout << "rcast_gpu" << std::endl;
    if(cuda_traversal != nullptr){
        cuda_traversal->rcast_kd_restart(params);
    }else{
        std::cout << "not initialized" << std::endl;
    }
}

void GaussiansKDTree::rcast_kd_restart(const TracingParams &tracing_params){
    std::cout << "tracing..." << std::endl; 

    std::cout << "num. nodes: " << nodes.size() << std::endl;
    for(int i = 0; i < tracing_params.num_rays; ++i){
        const float3 orig = tracing_params.ray_origins[i];
        const float3 dir = tracing_params.ray_directions[i];
        float sceneMin,sceneMax;
        if(!intersectAABB(scene_vol.min,scene_vol.max,orig,dir,sceneMin,sceneMax))
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
            Node node = nodes.at(inode);
            tmin = tmax;
            tmax = sceneMax;
            bool pushdown = true;
            char n_leaf_plane_mask = 0;
            //std::cout << "restart " << inode << std::endl;
            while(!node.isleaf()){
                if(tracing_params.draw_kd)
                    draw_aabb(tracing_params,inode,i);
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
                node = nodes.at(inode);
            }
            if(inode != -1 && node.isleaf()){
                if(tracing_params.draw_kd)
                    draw_aabb(tracing_params,inode,i);
                if(node.is_leaf_empty()) continue;
                //std::cout << "leaf" << std::endl;
                const auto& leaf_data = leaves_data[node.data_id()];
                struct Hit{int id; float resp, tmax; float3 rad;};
                std::vector<Hit> hits;
                // get particle intersections
                for(int i = 0; i < leaf_data.part_ids.size(); ++i){

                    if(leaf_data.plane_masks[i] & leaf_plane_mask){ 
                        continue; // -> this particle was processed in the previous leaf
                    }
                    
                    int part_id = leaf_data.part_ids[i];
                    float part_tmin,part_tmax;
                    if(intersectAABB(aabbs[part_id].min,aabbs[part_id].max,orig,dir,part_tmin,part_tmax)
                        // && part_tmin >= tmin
                        )
                         {
                        float resp,tmaxresp;
                        computeResponse(data,part_id,orig,dir,resp,tmaxresp);
                        float3 rad = computeRadiance(data.sh+part_id*16, data.sh_deg, data.xyz[part_id],orig);
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
                    (*tracing_params.num_its)++;
                }
            }
        }
        tracing_params.radiance[i] += acc_rad;
    }
}

void GaussiansKDTree::rcast_draw_kd(const TracingParams& tracing_params){
    for(int i = 0; i < tracing_params.num_rays; ++i){
        const float3 orig = tracing_params.ray_origins[i];
        const float3 dir = tracing_params.ray_directions[i];
        for(int j = 0; j < nodes.size(); ++j){
            draw_aabb(tracing_params,j,i);
        }
    }
}

void GaussiansKDTree::draw_aabb(const TracingParams& tracing_params, int node_id, int ray_id){
    const float3 orig = tracing_params.ray_origins[ray_id];
    const float3 dir = tracing_params.ray_directions[ray_id];
    float aabb_tmin,aabb_tmax;
    if(intersectAABB(node_aabbs[node_id].min,node_aabbs[node_id].max,orig,dir,aabb_tmin,aabb_tmax)){
        // for(float t : {aabb_tmin,aabb_tmax}){
        //     float3 pos = t*dir+orig;
        //     float3 dmin = pos-node_aabbs[node_id].min;
        //     float3 dmax = node_aabbs[node_id].max-pos;
        //     const float thr = .1f;
        //     int c = 0;
        //     for(int k = 0; k < 3; ++k){
        //         if(vec_c(dmin,k)<thr || vec_c(dmax,k)<thr){
        //             ++c;
        //         }
        //     }
        //     if(c == 2){
        //         tracing_params.radiance[ray_id] += float3{1.,0.,0.};
        //     }
        // }
        tracing_params.radiance[ray_id] += float3{1.,0.,0.};
    }
}