#include "gaussians_tracer.h"
#include "utils/vec_math.h"
#include <vector>

class GaussiansKDTree{
public:
    GaussiansKDTree() noexcept;
    GaussiansKDTree(const GaussiansData& data){
        build(data);
    }
    ~GaussiansKDTree() noexcept(true);
    GaussiansKDTree(const GaussiansAS &) = delete;
    GaussiansKDTree &operator=(const GaussiansAS &) = delete;
    GaussiansKDTree(GaussiansKDTree &&other) noexcept;
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
        //swap(first.context, second.context);
    }
private:
    void build(const GaussiansData& data);

    struct AABB { float3 min, max; };
    std::vector<AABB> aabbs;
};

class TracerCustom{
public:
    TracerCustom(int8_t device);
    ~TracerCustom() noexcept(true);
    void load_gaussians(const GaussiansData& data) {
        gaussians_structure = std::move(GaussiansKDTree(data));
    }

    void trace_rays(const TracingParams &tracing_params);
private:
    GaussiansKDTree gaussians_structure;
};