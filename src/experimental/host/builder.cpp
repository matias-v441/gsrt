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

std::tuple<axes_ev_ids_t,axes_ev_ids_t,AABB,AABB,int,int>
split(const ASData_Host& as, const events_t& evs,
     const axes_ev_ids_t& axes_ev_ids, int split_ax, float csplit, AABB V) {

    axes_ev_ids_t evs_left;
    axes_ev_ids_t evs_right;
    for(int ax = 0; ax < 3; ++ax){
        for(int ev_id : axes_ev_ids[ax]){
            auto aabb = as.aabbs[evs.ev2part_id(ev_id)];
            float start_ev = vec_c(aabb.min,split_ax);
            float end_ev = vec_c(aabb.max,split_ax);
            if(start_ev<csplit){
                evs_left[ax].push_back(ev_id);
            }
            if(end_ev>csplit){
                evs_right[ax].push_back(ev_id);
            }
        }
    }
    AABB vol_left=V,vol_right=V;
    vec_c(vol_left.max,split_ax) = csplit;
    vec_c(vol_right.min,split_ax) = csplit;
    int num_left = 0;
    int num_right = 0;
    int cnt = 0;
    for(int ev_id : axes_ev_ids[0]){
        if(evs.is_end_ev(ev_id)) continue;
        cnt++;
        int part_id = evs.ev2part_id(ev_id);
        auto aabb = as.aabbs[part_id];
        if(aabb.min.x == __FLT_MAX__) continue;
        bool inleft = true;
        bool inright = true;
        for(int ax = 0; ax < 3; ++ax){
            if(vec_c(aabb.min, ax) < vec_c(vol_left.min, ax) || vec_c(aabb.max, ax) > vec_c(vol_left.max, ax)){
                inleft = false;
            }
            if(vec_c(aabb.min, ax) < vec_c(vol_right.min, ax) || vec_c(aabb.max, ax) > vec_c(vol_right.max, ax)){
                inright = false;
            }
        }
        num_left += inleft;
        num_right += inright;
    }
    return {evs_left, evs_right, vol_left, vol_right, num_left, num_right};
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

    float median = evs.axes_events[axis][axes_ev_ids[axis][axes_ev_ids[axis].size()/2]];
    //float median = 0.5f*(vec_c(V.min,axis)+vec_c(V.max,axis));
    
    if(num_part <= params.max_leaf_size // leaf is small enough -> make leaf
        || median < vec_c(V.min,axis) + 1e-6f
        || median > vec_c(V.max,axis) - 1e-6f
        ){
        as.nodes[node_id].set_data_id(as.leaves_data.size());
        create_leaf(as, evs, axes_ev_ids, V);
        return;
    }
    as.nodes[node_id].set_axis(axis);

    as.nodes[node_id].set_cplane(median);
    auto [evs_left, evs_right, Vleft, Vright, num_left, num_right] = split(as, evs, axes_ev_ids, axis, median, V);
    std::cout << vec_c(V.min,axis) << " " << vec_c(V.max,axis) << " " << median << " " << num_part
              << " num_left " << num_left << " num_right " << num_right <<std::endl;
    assert(num_left+num_right <= num_part);

                        // dont add empty leaves
    bool go_left = true;//best_plane.num_left != 0;
    bool go_right = true;//best_plane.num_right != 0;
    if(go_left){
        // printf("left volume min %f max %f\n",vec_c(Vleft.min,axis),vec_c(Vleft.max,axis));
        build_rec(as, params, evs, evs_left, Vleft, num_left, depth+1);
        if(go_right){
            as.nodes[node_id].set_right_id(as.nodes.size());
        }
    }
    if(go_right){
        // printf("right volume min %f max %f\n",vec_c(Vright.min,axis),vec_c(Vright.max,axis));
        build_rec(as, params, evs, evs_right, Vright, num_right, depth+1);
    }
}
}

void build(ASData_Host& as, const KdParams& params){

    using util::geom::icosahedron_3dgrut::n_verts;
    const float alpha_min = .0113f;
    as.scene_vol = AABB{make_float3(__FLT_MAX__),make_float3(-__FLT_MAX__)};
    const auto& data = as.data;
    as.aabbs.resize(data.numgs,as.scene_vol);
    for(int i = 0; i < data.numgs; ++i){
        const Matrix3x3 R = util::geom::construct_rotation(data.rotation[i]);
        float adaptive_scale = sqrtf(-2.*logf(fminf(alpha_min / data.opacity[i], 0.97f)));
        //float adaptive_scale = sqrt(2.*log(data.opacity[i]/alpha_min));
        float3 s = data.scaling[i]*adaptive_scale;
        for(int j = 0; j < n_verts; ++j){
            float3 v = util::geom::icosahedron_3dgrut::vertices().data[j];
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