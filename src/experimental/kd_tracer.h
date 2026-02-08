#pragma once
#include "../tracer.h"
#include "host/traverser.h"
#include "host/builder.h"
#include "cuda/traverser.h"
#include "structure.h"

namespace gsrt::kd_tracer {

class KdTracer : public Tracer {
public:

    ~KdTracer() noexcept(true){}

    void load_gaussians(const GaussiansData& data, const ASParams& params) override;

    inline int get_size() const {
        return scene_as.has_value() ? scene_as->get_size() : 0;
    }

    void trace_rays(const TracingParams &tracing_params) override;

private:
   std::optional<ASData_Host> scene_as;
   std::optional<ASData_Device> scene_as_device;
};

}  // namespace gsrt::kd_tracer