#include "gaussians_tracer.h"
#include "utils/vec_math.h"
#include <vector>
#include <iostream>
#include <array>

inline float& vec_c(float3& vec,int a){
    return (&vec.x)[a];
};

typedef std::array<std::vector<int>,3> axis_ev_ids ;

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
        void set_cplane(float c){
            p = c;
        }
    };

    void build();

    void build_rec(AABB V, axis_ev_ids evs,int num_part,int depth);

    struct SplitPlane{
        float coord;
        float cost;
        int num_left;
        int num_right;
    };
    SplitPlane find_best_plane(int axis, AABB vol, axis_ev_ids evs, int num_part) const;

    std::pair<axis_ev_ids,axis_ev_ids> split_events(axis_ev_ids& evs,int split_ax,float csplit) const;

    void fill_leaf_particles(LeafData &leaf, AABB V, const axis_ev_ids& evs) const;

    float sa(AABB aabb) const {
        float3 d = aabb.max-aabb.min;
        return 2*(d.x*d.y+d.x*d.z+d.y*d.z);
    };
    std::pair<AABB,AABB> split_volume(int axis, float p, AABB aabb) const{
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
        auto [Vleft,Vright] = split_volume(axis,p,V);
        float prob_left = sa(Vleft)/sa(V);
        float prob_right = sa(Vright)/sa(V);
        if(Nl==1 && Nr==1){
            printf("SAH %f %f\n",prob_left,prob_right);
        }
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