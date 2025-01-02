#pragma once
#include "gaussians_tracer.h"
#include "utils/vec_math.h"
#include <vector>
#include <iostream>
#include <array>
#include <memory>

#include "tracer_cu.cuh"

typedef std::array<std::vector<int>,3> axis_ev_ids ;

enum class CFType{
    Default,
    EmptySpaceBias,
    Sorting,
    SomethingElse
};

struct ASParams{
    CFType cf_type;
    float K_T = 2.;
    float K_I = 3.;
    float k1 = 3.;
    float k2 = 1.25;
};

namespace kdtree_impl
{

struct AABB;

struct LeafData
{
    std::vector<int> part_ids;
    std::vector<char> plane_masks;
};

// class SmallNode{
//     float p;
//     int data = __INT_MAX__; 
// public:
//     int axis() const{
//         return data & 3;
//     }
//     bool isleaf() const{
//         return data <= 0;
//     }
//     int data_id() const{
//         return -data;
//     }
//     int right_id() const{
//         return data >> 2;
//     }
//     float cplane() const{
//         return p;
//     }
//     void set_axis(int axis){
//         data += axis%3;
//     }
//     void set_right_id(int id){
//         data = (id<<2) + axis();
//     }
//     void set_data_id(int id){
//         data = -id;
//     }
//     void set_cplane(float c){
//         p = c;
//     }
//     bool has_right(){
//         return data == __INT_MAX__;
//     }
// };
// 
class Node{
    int _axis;
    bool _isleaf;
    int dataid=-1;
    int right; 
    bool hasright;
    float p;
public:
    HOSTDEVICE int axis() const{
        return _axis;
    }
    HOSTDEVICE bool isleaf() const{
        return _isleaf;
    }
    HOSTDEVICE int data_id() const{
        return dataid;
    }
    HOSTDEVICE int right_id() const{
        return right;
    }
    HOSTDEVICE float cplane() const{
        return p;
    }
    void set_axis(int a){
        _axis = a; 
    }
    void set_right_id(int id){
        hasright = true;
        right = id;
    }
    void set_data_id(int id){
        _isleaf = true;
        dataid = id;
    }
    void set_cplane(float c){
        p = c;
    }
    HOSTDEVICE bool has_right(){
        return hasright;
    }
    void set_leaf_empty(){
        _isleaf = true;
        dataid = -1;
    }
    HOSTDEVICE bool is_leaf_empty(){
        return dataid == -1;
    }
};

class CUDA_Traversal;

class GaussiansKDTree{
public:
    GaussiansKDTree() noexcept{}
    GaussiansKDTree(const GaussiansData& data, const ASParams& params)
        : data(data),params(params){
        build();
    }
    ~GaussiansKDTree() noexcept(true){}
    GaussiansKDTree(const GaussiansAS &) = delete;
    GaussiansKDTree &operator=(const GaussiansAS &) = delete;
    GaussiansKDTree(GaussiansKDTree &&other) noexcept{
        aabbs = std::move(other.aabbs);
        data = other.data;
        nodes = other.nodes;
        leaves_data = other.leaves_data;
        scene_vol = other.scene_vol;
        node_aabbs = other.node_aabbs;
        cuda_traversal = std::move(other.cuda_traversal);
        params = other.params;
    }
    GaussiansKDTree &operator=(GaussiansKDTree &&other) {
        using std::swap;
        if (this != &other) {
            GaussiansKDTree tmp(std::move(other));
            swap(tmp, *this);
        }
        return *this;
    }
    friend void swap(GaussiansKDTree &first, GaussiansKDTree &second) {
        using std::swap;
        swap(first.data, second.data);
        swap(first.aabbs, second.aabbs);
        swap(first.nodes, second.nodes);
        swap(first.leaves_data, second.leaves_data);
        swap(first.scene_vol, second.scene_vol);
        swap(first.node_aabbs, second.node_aabbs);
        swap(first.cuda_traversal, second.cuda_traversal);
        swap(first.params, second.params);
    }
    void rcast_linear(const TracingParams& params);
    void rcast_kd(const TracingParams& params);
    void rcast_kd_restart(const TracingParams& params);
    void rcast_draw_kd(const TracingParams& params);
    void draw_aabb(const TracingParams& params, int node_id, int ray_id);
    void rcast_gpu(const TracingParams& params);

    const GaussiansData& get_scene()const {
        return data;
    }

    static constexpr int MAX_LEAF_SIZE = 1024;

private:


    void build();

    void build_rec(AABB V, axis_ev_ids evs,int num_part,int depth);

    void build_check();

    void init_device();

    struct SplitPlane{
        float coord;
        float cost;
        int num_left;
        int num_right;
    };
    SplitPlane find_best_plane(int axis, AABB vol, axis_ev_ids evs, int num_part) const;

    std::pair<axis_ev_ids,axis_ev_ids> split_events(axis_ev_ids& evs,int split_ax,float csplit) const;

    void fill_leaf_particles(LeafData &leaf, AABB V, const axis_ev_ids& evs) const;

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

    int maxdepth() const{
        return params.k1+params.k2*logf(static_cast<float>(data.numgs));
    }

    float cost(float Pl, float Pr, int Nl, int Nr) const {
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
 
    std::vector<AABB> aabbs;
    AABB scene_vol;

    std::vector<Node> nodes;

    std::vector<AABB> node_aabbs;
    
    std::vector<LeafData> leaves_data;

    std::unique_ptr<CUDA_Traversal> cuda_traversal;

    ASParams params;

    //enum class EventType:bool{START,END};
    //struct Event{
    //    EventType type;
    //    float p;
    //    int bb_id;
    //};
    // events sorted along each axis
    //std::array<std::vector<Event>,3> evs_axis_sort;

    std::array<std::vector<float>,3> axis_events;

    bool is_start_ev(int ev_id) const{
        return ev_id < data.numgs;
    }
    bool is_end_ev(int ev_id) const{
        return !is_start_ev(ev_id);
    }
    int ev2part_id(int ev_id) const{
        return ev_id % data.numgs;
    }
    GaussiansData data;

    friend CUDA_Traversal;
};

}

using namespace kdtree_impl;

class TracerCustom{
public:
    TracerCustom(int8_t device){}
    ~TracerCustom() noexcept(true){}
    void load_gaussians(const GaussiansData& data, const ASParams& params) {
        scene_as = std::move(GaussiansKDTree(data,params));
    }

    void trace_rays(const TracingParams &tracing_params){
        if(tracing_params.tracer_type == 1)
            scene_as.rcast_kd_restart(tracing_params);
        if(tracing_params.tracer_type == 2)
            scene_as.rcast_kd(tracing_params);
        if(tracing_params.tracer_type == 0)
            scene_as.rcast_linear(tracing_params);
        if(tracing_params.tracer_type == 3)
            scene_as.rcast_draw_kd(tracing_params);
        if(tracing_params.tracer_type == 5)
            scene_as.rcast_gpu(tracing_params);
        }
private:
    kdtree_impl::GaussiansKDTree scene_as;
};