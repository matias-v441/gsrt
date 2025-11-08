#pragma once
#include "utils/Matrix.h"
#ifndef __CUDA_ARCH__
    #include <optional>
    #include <variant>
#endif

namespace gsrt {

struct GaussiansData
{
    size_t numgs;
    float3 *xyz;
    float4 *rotation;
    float3 *scaling;
    float *opacity;
    float3 *sh;
    //float3 *color;
    int sh_deg;
};

struct ASParamsBase {
    float alpha_min; 
};

using OptixASParams = ASParamsBase;

struct KdParams: ASParamsBase {
    int8_t device = -1; // -1: CPU, >=0: GPU id
    bool no_rebuild;
    size_t max_leaf_size = 4096;
};

#ifndef __CUDA_ARCH__
using ASParams = std::variant<OptixASParams, KdParams>;
#endif

struct BPInput {
    float3 *dL_dC; // gradient of loss w.r.t. radiance
    // outputs of forward pass
    float3 *radiance;
    float *transmittance;
    float *distance;
};

struct BPOutput {
    float3 *grad_xyz;
    float4 *grad_rotation;
    float3 *grad_scale;
    float *grad_opacity;
    float *grad_resp;
    float3 *grad_sh;
    //float3 *grad_color;
    //Matrix3x3 *grad_invRS;
};

struct RayData {
    size_t num_rays;
    int width;
    int height;
    float3 *ray_origins;
    float3 *ray_directions;
};

struct RenderOutput {
    float3 *radiance;
    float *transmittance;
    float *distance;
    float3 *debug_map_0;
    float3 *debug_map_1;
    unsigned long long *num_its;
    unsigned long long *num_its_bwd;
};

struct BaseRenderSettings {
    bool compute_grad;
    bool white_background;
};

using OptixRenderSettings = BaseRenderSettings;

enum class KdTracerType {
    Restart,
    Leaves
};

struct KdRenderSettings : BaseRenderSettings {
    int8_t device = -1; // -1: CPU, >=0: GPU id
    bool draw_kd;
    KdTracerType tracer;
};

#ifndef __CUDA_ARCH__

using RenderSettings = std::variant<OptixRenderSettings, KdRenderSettings>;

struct TracingParams
{
    RayData rays;
    RenderOutput output;
    RenderSettings settings;
    std::optional<BPInput> bp_in;
    std::optional<BPOutput> bp_out;
};

#endif // __CUDA_ARCH__

} // namespace gsrt