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
    AABB scene_vol{make_float3(__FLT_MAX__),make_float3(__FLT_MIN__)};
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
    for(int axis = 0; axis < 3; ++axis){
        std::vector<float> events(data.numgs*2);
        std::vector<int> event_ids(data.numgs*2);
        for(int i = 0; i < data.numgs; ++i){
            int si = i;
            int ei = i+data.numgs;
            events[si] = vec_c(aabbs[i].min,i);
            events[ei] = vec_c(aabbs[i].max,i);
            event_ids[si] = si;
            event_ids[ei] = ei;
        }
        std::sort(event_ids.begin(),event_ids.end(),
            [&](int i, int j){return events[i]<events[j];});

        // add links to start event for each particle
        aabb_start_asort_ids[axis].resize(data.numgs);
        for(int i = 0; i < event_ids.size(); ++i){
            if(event_ids[i] < data.numgs) // start event
                aabb_start_asort_ids[axis][event_ids[i]] = i;
        }

        axis_events_asort[axis] = std::move(event_ids);
        axis_events[axis] = std::move(events);
    }
    const int n = data.numgs*2;
    std::array<int2,3> event_ranges{make_int2(0,n),make_int2(0,n),make_int2(0,n)};
    build_rec(scene_vol,event_ranges,data.numgs,0);

    std::cout << "num. leaves: " << leaves_data.size() << std::endl;
    int max_leaf_size = 0;
    for(auto& leaf : leaves_data){
        max_leaf_size = max(leaf.part_ids.size(),max_leaf_size);
    }
    std::cout << "max leaf size: " << max_leaf_size << std::endl;
}

void GaussiansKDTree::build_rec(AABB V,
                                std::array<std::vector<int>,3> event_ids,
                                int num_part,
                                int depth){
    int axis = depth%3; // round-robin
    SplitPlane best_plane = find_best_plane(axis, V, events_ranges[axis], num_part);
    Node& node = nodes.emplace_back();
    if(best_plane.cost < cost(0.f,1.f,0,num_part) // split is better than no split
        && num_part < MAX_LEAF_SIZE // if the leaf is too big we split anyway
        ){
        node.set_data_id(leaves_data.size());
        fill_leaf_particles(leaves_data.emplace_back(), events_ranges);
        return;
    }
    node.set_axis(axis);
    node.set_cplane(best_plane.coord);
    auto [Vleft,Vright] = split(axis,best_plane.coord,V);

    events_ranges[axis] = make_int2(events_ranges[axis].x,best_plane.event_id);
    build_rec(Vleft,events_ranges,best_plane.num_left,depth+1);

    node.set_right_id(nodes.size());

    events_ranges[axis] = make_int2(best_plane.event_id,events_ranges[axis].y);
    build_rec(Vright,events_ranges,best_plane.num_right,depth+1);
}

void GaussiansKDTree::fill_leaf_particles(LeafData &leaf, const std::vector<int>& axis_event_ids) const{
    // add all particles corresponding to start events and all that 
    // correspond to end events with the start event not lying in leaf range
    // particles on the splitting plane get added to both adjacent volumes
    for(int i = event_ids.x; i < event_ids.y; ++i){
        int ev_id = axis_events_asort[0][i];
        int part_id = ev2part_id(ev_id);
        if(is_start_ev(ev_id)
            || (aabb_start_asort_ids[0][part_id]<event_ranges[0].x)){
            leaf.part_ids.push_back(part_id);
        }
    }
}

GaussiansKDTree::SplitPlane GaussiansKDTree::find_best_plane(int axis, AABB V, int2 events_range, int num_part) const{
    
    int num_left=0,num_right=num_part;
    // add all nodes started outside the volume to the left side
    for(int i = events_range.x+1; i < events_range.y;++i){
        int id = axis_events_asort[axis][i];
        float ev = axis_events[axis][id];
        if(is_end_ev(id) && aabb_start_asort_ids[axis][ev2part_id(id)] < events_range.x){
            ++num_left;
        }
    }
    SplitPlane best_plane;
    best_plane.cost = __FLT_MAX__;
    // find split with minimum SAH
    for(int i = events_range.x+1; i < events_range.y;++i){
        int id = axis_events_asort[axis][i];
        float ev = axis_events[axis][id];
        if(is_end_ev(id)){
            --num_right;
        }
        float cost = SAH(axis,ev,V,num_left,num_right);
        if(is_start_ev(id)){
            ++num_left;
        }
        if(cost<best_plane.cost){
            best_plane.cost = cost;
            best_plane.event_id = i;
            best_plane.coord = ev;
            best_plane.num_left = num_left;
            best_plane.num_right = num_right;
        }
    }
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
