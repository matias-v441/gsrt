#include "builder.h"

#include "utils/vec_math.h"
#include "utils/geom.h"
#include "utils/Matrix.h"
#include <cassert>
#include <algorithm>
#include <vector>
#include <array>

using namespace util;
using namespace gsrt;
using namespace gsrt::kd_tracer;

namespace gsrt::kd_tracer::host::builder {

namespace {

typedef std::array<std::vector<int>,3> axes_ev_ids_t;
typedef std::array<std::vector<float>,3> axes_evs_t;

struct events_t {
    axes_evs_t axes_events;
    int num_part;
    bool is_start_ev(int ev_id) const{
        return ev_id < num_part;
    }
    bool is_end_ev(int ev_id) const{
        return !is_start_ev(ev_id);
    }
    int ev2part_id(int ev_id) const{
        return ev_id % num_part;
    }
};

struct SplitPlane{
    float coord;
    float cost;
    int num_left;
    int num_right;
};

SplitPlane find_best_plane(const events_t& evs, const axes_ev_ids_t& axes_ev_ids, int axis, AABB V, int num_part) {
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
    //for(int ev_id : evs[axis]){ 
    //    auto aabb = aabbs[ev2part_id(ev_id)];
    //    if(vec_c(aabb.min,axis)<vec_c(V.min,axis)){
    //        ++num_left;
    //    }
    //}
    SplitPlane best_plane;
    best_plane.cost = __FLT_MAX__;
    // find split with minimum SAH
    for(int ev_id : axes_ev_ids[axis]){
        if(evs.is_end_ev(ev_id)){
            --num_right;
        }
        float csplit = evs.axes_events[axis][ev_id];
        if(csplit > vec_c(V.min,axis) && csplit < vec_c(V.max,axis)){
            // printf("SAH ARGS %d %f %d %d\n",axis,csplit,num_left,num_right);
            //float cost = SAH(axis,csplit,V,num_left,num_right);
            // std::cout << cost << std::endl;
            //if(cost<best_plane.cost){
            float cost = 0;
            if(num_left >= num_right){
                best_plane.cost = cost;
                best_plane.coord = csplit;
                best_plane.num_left = num_left;
                best_plane.num_right = num_right;
                break;
            }
        }
        if(evs.is_start_ev(ev_id)){
            ++num_left;
        }
    }
    // printf("FBP cost=%f num_left=%d num_right=%d\n",
    //     best_plane.cost,best_plane.num_left,best_plane.num_right);
    return best_plane;
}

std::pair<axes_ev_ids_t,axes_ev_ids_t> split_events(const ASData_Host& as, const events_t& evs,
     const axes_ev_ids_t& axes_ev_ids, int split_ax, float csplit) {

    axes_ev_ids_t left;
    axes_ev_ids_t right;
    //for(int k = 1; k < 3; ++k){
    //    int ax = (split_ax+k)%3;
    for(int ax = 0; ax < 3; ++ax){
        for(int ev_id : axes_ev_ids[ax]){
            auto aabb = as.aabbs[evs.ev2part_id(ev_id)];
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
    //for(int ev_id : evs[split_ax]){
    //    float ev = axis_events[split_ax][ev_id];
    //    if(ev < csplit || (ev == csplit && is_end_ev(ev_id))){
    //        left[split_ax].push_back(ev_id);
    //    }else{
    //        right[split_ax].push_back(ev_id);
    //    }
    //}
    return {left,right};
}

void create_leaf(ASData_Host &as, const events_t& evs, const axes_ev_ids_t& axes_ev_ids, AABB V) {
    // for(int ev_id : evs[0]){
    //     int part_id = ev2part_id(ev_id);
    //     if(is_start_ev(ev_id) || aabbs[part_id].min.x < V.min.x){
    //         leaf.part_ids.push_back(part_id);
    //     }
    // }
    LeafData &leaf = as.leaves_data.emplace_back();
    for(int ev_id : axes_ev_ids[0]){
        int part_id = evs.ev2part_id(ev_id);
        if(evs.is_start_ev(ev_id) || as.aabbs[part_id].min.x < V.min.x){
            // add particle
            leaf.part_ids.push_back(part_id);
            // calculate plane mask
            char plane_mask = 0;
            for(int k = 0; k < 3; ++k){
                if(vec_c(as.aabbs[part_id].min,k)<vec_c(V.min,k)){
                    plane_mask |= 1<<(k*2);
                }
                if(vec_c(as.aabbs[part_id].max,k)>vec_c(V.max,k)){
                    plane_mask |= 1<<(k*2+1);
                }
            }
            leaf.plane_masks.push_back(plane_mask);
        }
    }
}

static float sa(AABB aabb) {
    float3 d = aabb.max-aabb.min;
    if(d.x==0.f || d.y==0.f || d.z==0.f) return 0.f;
    return 2*(d.x*d.y+d.x*d.z+d.y*d.z);
};

static std::pair<AABB,AABB> split_volume(int axis, float p, AABB aabb) {
    AABB left=aabb,right=aabb;
    vec_c(left.max,axis) = p;
    vec_c(right.min,axis) = p;
    return {left,right};
}

/*
inline int maxdepth() {
    return params.k1+params.k2*logf(static_cast<float>(data.numgs));
}

float cost(float Pl, float Pr, int Nl, int Nr) {
    float K_T = params.K_T;
    float K_I = params.K_I;
    switch (params.cf_type)
    {
    case CFType::Default :
    {
        return K_T + K_I*(Pl*Nl+Pr*Nr); 
    }
    case CFType::EmptySpaceBias :
    {
        float lambda = (Nl==0 || Nr==0)? 0.8 : 1.; // bias towards empty splits
        return lambda*(K_T + K_I*(Pl*Nl+Pr*Nr));
    }
    case CFType::Sorting :
    {
        return K_T + K_I*(Pl*(Nl+Nl*logf(static_cast<float>(Nl)))
                + Pr*(Nr+Nr*logf(static_cast<float>(Nr)))); 
    }
    case CFType::SomethingElse :
    {
        return K_T + K_I*(Pl*(Nl+Nl*Nl+exp(static_cast<float>(Nl)))
                + Pr*(Nr+Nr*Nr+exp(static_cast<float>(Nr)))); 
    }
    }
}//

float SAH(int axis, float p, AABB V, int Nl, int Nr) const {
    auto [Vleft,Vright] = split_volume(axis,p,V);
    // printf("SAH V %f max %f\n",vec_c(V.min,axis),vec_c(V.max,axis));
    // printf("SAH Vleft %f max %f\n",vec_c(Vleft.min,axis),vec_c(Vleft.max,axis));
    // printf("SAH Vright %f max %f\n",vec_c(Vright.min,axis),vec_c(Vright.max,axis));
    float prob_left = sa(Vleft)/sa(V);
    float prob_right = sa(Vright)/sa(V);
    // printf("SAH cost %f %f %d %d\n",prob_left,prob_right,Nl,Nr);
    float c = cost(prob_left,prob_right,Nl,Nr);
    return c;
};
*/

//enum class EventType:bool{START,END};
//struct Event{
//    EventType type;
//    float p;
//    int bb_id;
//};
// events sorted along each axis
//std::array<std::vector<Event>,3> evs_axis_sort;


void build_rec(ASData_Host& as, const KdParams& params, const events_t& evs, const axes_ev_ids_t& axes_ev_ids,
     AABB V, int num_part, int depth){
    
    int axis = depth%3; // round-robin

    as.max_depth = max(as.max_depth, depth);

    int node_id = as.nodes.size();
    as.nodes.emplace_back();

    as.node_aabbs.push_back(V);

    if(axes_ev_ids[0].size()+axes_ev_ids[1].size()+axes_ev_ids[2].size() == 0){
        as.nodes[node_id].set_leaf_empty();
        return;
    }

    SplitPlane best_plane = find_best_plane(evs, axes_ev_ids, axis, V, num_part);
    
    if(//depth == maxdepth()
        //|| best_plane.coord == vec_c(V.min,axis) || // same as no split
        //best_plane.cost > cost(0.f,1.f,0,num_part) // split is worse than no split
        // ||
         num_part <= params.max_leaf_size // leaf is small enough -> make leaf
        ){
        as.nodes[node_id].set_data_id(as.leaves_data.size());
        create_leaf(as, evs, axes_ev_ids, V);
        return;
    }
    as.nodes[node_id].set_axis(axis);
    as.nodes[node_id].set_cplane(best_plane.coord);

    auto [Vleft,Vright] = split_volume(axis, best_plane.coord, V);
    auto [evs_left,evs_right] = split_events(as, evs, axes_ev_ids, axis, best_plane.coord);
    
                        // dont add empty leaves
    bool go_left = true;//best_plane.num_left != 0;
    bool go_right = true;//best_plane.num_right != 0;
    if(go_left){
        // printf("left volume min %f max %f\n",vec_c(Vleft.min,axis),vec_c(Vleft.max,axis));
        build_rec(as, params, evs, evs_left, Vleft, best_plane.num_left, depth+1);
        if(go_right){
            as.nodes[node_id].set_right_id(as.nodes.size());
        }
    }
    if(go_right){
        // printf("right volume min %f max %f\n",vec_c(Vright.min,axis),vec_c(Vright.max,axis));
        build_rec(as, params, evs, evs_right, Vright, best_plane.num_right, depth+1);
    }
}
}

void build(ASData_Host& as, const KdParams& params){

    using util::geom::icosahedron::n_verts;
    const float alpha_min = .01;
    as.scene_vol = AABB{make_float3(__FLT_MAX__),make_float3(-__FLT_MAX__)};
    const auto& data = as.data;
    as.aabbs.resize(data.numgs,as.scene_vol);
    for(int i = 0; i < data.numgs; ++i){
        const Matrix3x3 R = util::geom::construct_rotation(data.rotation[i]);
        float adaptive_scale = sqrt(2.*log(data.opacity[i]/alpha_min));
        float3 s = data.scaling[i]*adaptive_scale;
        for(int j = 0; j < n_verts; ++j){
            float3 v = util::geom::icosahedron::vertices[j];
            float3 w = R*(s*v)+data.xyz[i];
            as.aabbs[i].min = fminf(as.aabbs[i].min,w);
            as.aabbs[i].max = fmaxf(as.aabbs[i].max,w);
        }
        as.scene_vol.min = fminf(as.aabbs[i].min,as.scene_vol.min);
        as.scene_vol.max = fmaxf(as.aabbs[i].max,as.scene_vol.max);
    }
    events_t evs;
    evs.num_part = data.numgs;
    axes_ev_ids_t axes_ev_ids;
    for(int axis = 0; axis < 3; ++axis){
        std::vector<float> events(data.numgs*2);
        std::vector<int> event_ids(data.numgs*2);
        for(int i = 0; i < data.numgs; ++i){
            int si = i;
            int ei = i+data.numgs;
            events[si] = vec_c(as.aabbs[i].min,axis);
            events[ei] = vec_c(as.aabbs[i].max,axis);
            event_ids[si] = si;
            event_ids[ei] = ei;
        }
        std::sort(event_ids.begin(),event_ids.end(),
            [&](int i, int j){return events[i]<events[j];});
        axes_ev_ids[axis] = std::move(event_ids);
        evs.axes_events[axis] = std::move(events);
    }
    build_rec(as, params, evs, axes_ev_ids, as.scene_vol, data.numgs, 0);
    as.validate();
}

}// namespace gsrt::kd_tracer::host::builder