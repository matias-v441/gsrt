#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <optix.h>
#include <stdio.h>
#include <unistd.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

extern unsigned char ptx_code_file[];

struct GaussiansData
{
    size_t numgs;
    float3 *xyz;
    float4 *rotation;
    float3 *scaling;
    float *opacity;
    float3 *sh;
    float3 *color;

    float3 *normals;

    int sh_deg;
};

struct TracingParams
{
    size_t num_rays;
    size_t width;
    size_t height;
    float3 *ray_origins;
    float3 *ray_directions;

    float3 *radiance;
    float *transmittance;
    float3 *debug_map_0;
    float3 *debug_map_1;
    unsigned long long *num_its;
    unsigned long long *num_its_bwd;

    bool compute_grad;
    float3 *grad_xyz;
    float4 *grad_rotation;
    float3 *grad_scale;
    float *grad_opacity;
    float *grad_resp;
    float3 *grad_sh;
    float3 *grad_color;

    float3* dL_dC;
};

struct SceneBuffers{

    bool* rad_clamped;
    float3* rad_sh;
};

class GaussiansAS {
   public:
    GaussiansAS() noexcept;
    GaussiansAS(const OptixDeviceContext &context, const uint8_t device) : device(device), context(context) {}
    GaussiansAS(
        const OptixDeviceContext &context,
        const uint8_t device,
        const GaussiansData& data) : GaussiansAS(context, device) {
        build(data);
    }

    ~GaussiansAS() noexcept(false);
    GaussiansAS(const GaussiansAS &) = delete;
    GaussiansAS &operator=(const GaussiansAS &) = delete;
    GaussiansAS(GaussiansAS &&other) noexcept;
    GaussiansAS &operator=(GaussiansAS &&other) {
        using std::swap;
        if (this != &other) {
            GaussiansAS tmp(std::move(other));
            swap(tmp, *this);
        }
        return *this;
    }
    friend void swap(GaussiansAS &first, GaussiansAS &second) {
        using std::swap;
        swap(first.context, second.context);
        swap(first.device, second.device);
        swap(first.gas_handle_, second.gas_handle_);
        swap(first.d_gas_output_buffer, second.d_gas_output_buffer);
        swap(first.d_vertices, second.d_vertices);
        swap(first.d_triangles, second.d_triangles);
        swap(first.d_gaussians, second.d_gaussians);
        //swap(first.d_scene_buffers, second.d_scene_buffers);
    }

    OptixTraversableHandle gas_handle() const {
        if (!defined()) {
            throw std::runtime_error("TetrahedraStructure is not initialized");
        }
        return gas_handle_;
    }

    bool defined() const {
        return gas_handle_ != 0;
    }

    const GaussiansData& device_gaussians() const{
        return d_gaussians;
    }

    const SceneBuffers& device_scene_buffers() const{
        return d_scene_buffers;
    }

   private:
    void build(const GaussiansData& gaussians);

    void release();
    OptixDeviceContext context = nullptr;
    int8_t device = -1;
    OptixTraversableHandle gas_handle_ = 0;
    CUdeviceptr d_gas_output_buffer = 0;
    CUdeviceptr d_vertices = 0;
    CUdeviceptr d_triangles = 0;
    GaussiansData d_gaussians{};
    SceneBuffers d_scene_buffers{};
};


class TraceRaysPipeline {
   public:
    TraceRaysPipeline() = default;
    TraceRaysPipeline(const OptixDeviceContext &context, int8_t device);
    TraceRaysPipeline(const TraceRaysPipeline &) = delete;
    TraceRaysPipeline &operator=(const TraceRaysPipeline &) = delete;
    TraceRaysPipeline(TraceRaysPipeline &&other) noexcept;
    TraceRaysPipeline &operator=(TraceRaysPipeline &&other) {
        using std::swap;
        if (this != &other) {
            TraceRaysPipeline tmp(std::move(other));
            swap(tmp, *this);
        }
        return *this;
    }
    ~TraceRaysPipeline() noexcept(false);

    friend void swap(TraceRaysPipeline &first, TraceRaysPipeline &second) {
        using std::swap;
        swap(first.context, second.context);
        swap(first.device, second.device);
        swap(first.module, second.module);
        swap(first.sbt, second.sbt);
        swap(first.pipeline, second.pipeline);
        swap(first.d_param, second.d_param);
        swap(first.stream, second.stream);
        swap(first.raygen_prog_group, second.raygen_prog_group);
        swap(first.miss_prog_group, second.miss_prog_group);
        swap(first.hitgroup_prog_group, second.hitgroup_prog_group);
    }

    void trace_rays(const GaussiansAS *gaussians_structure,
                    const TracingParams &tracing_params
                    );

   private:
    // Context, streams, and accel structures are inherited
    OptixDeviceContext context = nullptr;
    int8_t device = -1;

    // Local fields used for this pipeline
    OptixModule module = nullptr;
    OptixShaderBindingTable sbt = {};
    OptixPipeline pipeline = nullptr;
    CUdeviceptr d_param = 0;
    CUstream stream = nullptr;

    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;
    OptixProgramGroup bwd_hitgroup_prog_group = nullptr;

    static std::string load_ptx_data() {
        return std::string((char *)ptx_code_file);
    }
};


class GaussiansTracer {
   public:
    GaussiansTracer(int8_t device);
    ~GaussiansTracer() noexcept(false);
    GaussiansTracer(const GaussiansTracer &) = delete;
    GaussiansTracer &operator=(const GaussiansTracer &) = delete;
    GaussiansTracer &operator=(GaussiansTracer &&other) {
        using std::swap;
        if (this != &other) {
            GaussiansTracer tmp(std::move(other));
            swap(tmp, *this);
        }
        return *this;
    }
    GaussiansTracer(GaussiansTracer &&other);

    void load_gaussians(const GaussiansData& data) {
        gaussians_structure = std::move(GaussiansAS(context, device, data));
    }

    void trace_rays(const TracingParams &tracing_params) {
        trace_rays_pipeline.trace_rays(
            &gaussians_structure,
            tracing_params);
    }

   private:
    // Global contexts
    int8_t device;
    OptixDeviceContext context;

    GaussiansAS gaussians_structure;
    TraceRaysPipeline trace_rays_pipeline;
};


