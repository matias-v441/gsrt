#include "gaussians_tracer.h"
#include "utils/vec_math.h"
#include <vector>
#include <iostream>
#include <array>

inline float& vec_c(float3& vec,int a){
    return (&vec.x)[a];
};

class GaussiansKDTree{
public:
    GaussiansKDTree() noexcept{}
    GaussiansKDTree(const GaussiansData& data):data(data){
        build();
    }
    ~GaussiansKDTree() noexcept(true){}
    GaussiansKDTree(const GaussiansAS &) = delete;
    GaussiansKDTree &operator=(const GaussiansAS &) = delete;
    GaussiansKDTree(GaussiansKDTree &&other) noexcept{
        aabbs = std::move(other.aabbs);
        data = other.data;
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
    }
    void traverse(const TracingParams& params);

    const GaussiansData& get_scene()const {
        return data;
    }
private:

    struct AABB { float3 min, max; };

    struct LeafData
    {
        std::vector<int> part_ids;
    };

    class Node{
        float p;
        int data; 
    public:
        int axis() const{
            return data & 3;
        }
        bool is_leaf() const{
            return data <= 0;
        }
        int data_id() const{
            return std::abs(data);
        }
        int right_id() const{
            return data >> 2;
        }
        float cplane() const{
            return p;
        }
        void set_axis(int axis){
            data += axis%3;
        }
        void set_right_id(int id){
            data = (id<<2) + axis();
        }
        void set_data_id(int id){
            data = -id;
        }
        float set_cplane(float c){
            p = c;
        }
    };

    void build();

    void build_rec(AABB V,std::array<std::vector<int>,3> axis_events_ids,int num_part,int depth);

    struct SplitPlane{
        float coord;
        int event_id;
        float cost;
        int num_left;
        int num_right;
    };
    SplitPlane find_best_plane(int axis, AABB vol, int2 events_range, int num_part) const;

    void fill_leaf_particles(LeafData &leaf, const std::array<int2,3>& event_ranges) const;

    float sa(AABB aabb) const {
        float3 d = aabb.max-aabb.min;
        return 2*(d.x*d.y+d.x*d.z+d.y*d.z);
    };
    std::pair<AABB,AABB> split(int axis, float p, AABB aabb) const{
        AABB left=aabb,right=aabb;
        vec_c(left.max,axis) = p;
        vec_c(right.min,axis) = p;
        return {left,right};
    }
    static constexpr float K_T = 2.;
    static constexpr float K_I = 3.;
    static constexpr int MAX_LEAF_SIZE = 1024;

    float cost(float Pl, float Pr, int Nl, int Nr) const {
        float lambda = (Nl==0 || Nr==0)? 0.8 : 1.; // bias towards empty splits
        return lambda*(K_T + K_I*(Pl*Nl+Pr*Nr));
    }
    float SAH(int axis, float p, AABB V, int Nl, int Nr) const {
        auto [Vleft,Vright] = split(axis,p,V);
        float prob_left = sa(Vleft)/sa(V);
        float prob_right = sa(Vright)/sa(V);
        float c = cost(prob_left,prob_right,Nl,Nr);
        return c;
    };
 
    std::vector<AABB> aabbs;

    std::vector<Node> nodes;
    
    std::vector<LeafData> leaves_data;

    //enum class EventType:bool{START,END};
    //struct Event{
    //    EventType type;
    //    float p;
    //    int bb_id;
    //};
    // events sorted along each axis
    //std::array<std::vector<Event>,3> evs_axis_sort;

    std::array<std::vector<float>,3> axis_events;
    std::array<std::vector<int>,3> aabb_start_asort_ids;

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
};

class TracerCustom{
public:
    TracerCustom(int8_t device){}
    ~TracerCustom() noexcept(true){}
    void load_gaussians(const GaussiansData& data) {
        scene_as = std::move(GaussiansKDTree(data));
    }

    void trace_rays(const TracingParams &tracing_params){
        scene_as.traverse(tracing_params);
    }
private:
    GaussiansKDTree scene_as;
};