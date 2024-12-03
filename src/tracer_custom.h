#include "gaussians_tracer.h"
#include "utils/vec_math.h"
#include <vector>
#include <iostream>

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
    void build();

    struct AABB { float3 min, max; };
    std::vector<AABB> aabbs;
    struct Node{
        int axis; // leaf: -1
        int right; // leaf: data id
    };
    std::vector<Node> nodes;
    std::vector<std::vector<int>> leaf_data_ids;

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