#include <optix.h>
#include <stdio.h>
#include <unistd.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include "as.h"

extern unsigned char ptx_code_file[];

namespace gsrt::optix_tracer {

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
    ~TraceRaysPipeline();

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

}  // namespace gsrt::optix_tracer
