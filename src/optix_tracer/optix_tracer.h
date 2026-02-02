#include <optix.h>
#include "../tracer.h"
#include "as.h"
#include "pipeline.h"

namespace gsrt::optix_tracer {

class OptixTracer : public Tracer {
public:
    explicit OptixTracer(int8_t device);
    ~OptixTracer() noexcept(false) override;

    void load_gaussians(const GaussiansData& data, const ASParams&) override {
        gaussians_structure = std::move(GaussiansAS(context, device, data));
    }

    void trace_rays(const TracingParams &tracing_params) override {
        trace_rays_pipeline.trace_rays(
            &gaussians_structure,
            tracing_params);
    }

    struct AS_Stats{
        size_t gas_size;
    };
    AS_Stats as_stats(){
        return {gaussians_structure.gas_size()};
    }

private:
    int8_t device;
    OptixDeviceContext context;

    GaussiansAS gaussians_structure;
    TraceRaysPipeline trace_rays_pipeline;
};

}  // namespace gsrt::optix_tracer