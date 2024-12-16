#include "tracer_custom.h"

#include "utils/exception.h"
#include "utils/vec_math.h"
#include "utils/gsrt_util.h"
#include "utils/Matrix.h"

#include <algorithm>

using namespace util;


void GaussiansKDTree::build(){
    std::cout << "building..." << std::endl;

    using gsrt_util::icosahedron::n_verts;
    const float alpha_min = .01;
    AABB scene_vol{make_float3(__FLT_MAX__),make_float3(-__FLT_MAX__)};
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
    std::cout << "axis events" << std::endl;
    for(int a=0; a<3; ++a){
        std::cout << "axis " << a << std::endl;
        for(int ev : axis_events_asort[a]){
            std::cout << axis_events[a][ev] << " ";
        }
        std::cout << std::endl;
    }
    build_rec(scene_vol,axis_events_asort,data.numgs,0);

    std::cout << "num. leaves: " << leaves_data.size() << std::endl;
    int max_leaf_size = 0;
    for(auto& leaf : leaves_data){
        max_leaf_size = max(leaf.part_ids.size(),max_leaf_size);
    }
    std::cout << "max leaf size: " << max_leaf_size << std::endl;
}

void GaussiansKDTree::build_rec(AABB V,
                                axis_ev_ids evs,
                                int num_part,
                                int depth){
    printf("----------- depth %d\n",depth);
    printf("num_part %d\n",num_part);
    //if(evs[0].size()<=2) return;
    int axis = depth%3; // round-robin
    SplitPlane best_plane = find_best_plane(axis, V, evs, num_part);
    std::cout << "cost nosplit " << cost(0.f,1.f,0,num_part) << std::endl;
    int node_id = nodes.size();
    nodes.emplace_back();
    if(best_plane.num_left==0 || best_plane.num_right==0 // no reason to split
        || best_plane.cost > cost(0.f,1.f,0,num_part) // split is worse than no split
        || num_part > MAX_LEAF_SIZE // if the leaf is too big we split anyway
        ){
        std::cout << "leaf" << std::endl;
        nodes[node_id].set_data_id(leaves_data.size());
        fill_leaf_particles(leaves_data.emplace_back(),V,evs);
        return;
    }
    std::cout << "split" << std::endl;
    nodes[node_id].set_axis(axis);
    nodes[node_id].set_cplane(best_plane.coord);
    
    auto [Vleft,Vright] = split_volume(axis,best_plane.coord,V);
    auto [evs_left,evs_right] = split_events(evs,axis,best_plane.coord);
    int num_left = best_plane.num_left;
    int num_right = best_plane.num_right;
    if(num_left==0){
        std::swap(num_left,num_right);
        std::swap(evs_left,evs_right);
    }
    build_rec(Vleft,evs_left,num_left,depth+1);
    if(num_right!=0){
        nodes[node_id].set_right_id(nodes.size());
        build_rec(Vright,evs_right,num_right,depth+1);
    }
}

std::pair<axis_ev_ids,axis_ev_ids> GaussiansKDTree::split_events(
    axis_ev_ids& evs,int split_ax,float csplit) const{

    printf("SEV_in %d split=%f\n",evs[split_ax].size(),csplit);
    for(int ev : evs[split_ax]){
        printf("%f ",axis_events[split_ax][ev]);
    }
    std::cout << std::endl;

    axis_ev_ids left;
    axis_ev_ids right;
    for(int ax = 0; ax < 3; ++ax){
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
    printf("SEV %d %d\n",left[0].size(),right[0].size());
    printf("SEV %d %d\n",left[1].size(),right[1].size());
    printf("SEV %d %d\n",left[2].size(),right[2].size());
    return {left,right};
}

void GaussiansKDTree::fill_leaf_particles(LeafData &leaf, AABB V,
             const axis_ev_ids& evs) const{

    for(int ev_id : evs[0]){
        int part_id = ev2part_id(ev_id);
        if(is_start_ev(ev_id) || aabbs[part_id].min.x < V.min.x){
            leaf.part_ids.push_back(part_id);
        }
    }
}

GaussiansKDTree::SplitPlane GaussiansKDTree::find_best_plane(int axis, AABB V, axis_ev_ids evs, int num_part) const{
    printf("FBP vol_min=%f vol_max=%f\n",evs[0].size(),evs[1].size(),evs[2].size(),
        vec_c(V.min,axis),vec_c(V.max,axis));
    for(int ev : evs[axis]){
        printf("%f ",axis_events[axis][ev]);
    }
    std::cout << std::endl;
    for(int ev : evs[axis]){
        if(ev < data.numgs)
            printf("%ds ",ev2part_id(ev));
        else
            printf("%de ",ev2part_id(ev));
    }
    std::cout << std::endl;
    for(int ev_id : evs[axis]){ 
        auto aabb = aabbs[ev2part_id(ev_id)];
        printf("%f ", vec_c(aabb.min,axis));
    }
    std::cout << std::endl;
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
        //std::cout << "SAH_ARGS " << num_left << " " << num_right << std::endl;
        float cost = SAH(axis,csplit,V,num_left,num_right);
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
    printf("FBP cost=%f num_left=%d num_right=%d\n",
        best_plane.cost,best_plane.num_left,best_plane.num_right);
    return best_plane;
}

bool intersectAABB(const float3 min, const float3 max, const float3 orig, const float3 dir){
    float3 idir = 1.f/dir;
    float3 tc1 = (min-orig)*idir;
    float3 tc2 = (max-orig)*idir;
    float3 tc_min = fminf(tc1,tc2);
    float3 tc_max = fmaxf(tc1,tc2);
    float tmin = fmax(tc_min.x,fmax(tc_min.y,tc_min.z));
    float tmax = fmin(tc_max.x,fmin(tc_max.y,tc_max.z));
    return tmin < tmax;
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

void GaussiansKDTree::traverse(const TracingParams &tracing_params){
    std::cout << "tracing..." << std::endl; 

    for(int i = 0; i < tracing_params.num_rays; ++i){
        const float3 orig = tracing_params.ray_origins[i];
        const float3 dir = tracing_params.ray_directions[i];
        struct Hit{int id; float resp, tmax;};
        std::vector<Hit> hits;
        for(int j = 0; j < data.numgs; ++j){
            if(intersectAABB(aabbs[j].min,aabbs[j].max,orig,dir)){
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
            rad_acc += data.opacity[hit.id]*color*hit.resp*trans;
            trans *= (1-hit.resp);
        }
        tracing_params.radiance[i] = rad_acc;
    }
}
