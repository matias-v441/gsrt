#pragma once
#include "types.h"

namespace gsrt {

class Tracer {
   public:
    Tracer() = default;
    virtual ~Tracer() noexcept(false) = default;
    Tracer(const Tracer &) = delete;
    Tracer &operator=(const Tracer &) = delete;
    Tracer &operator=(Tracer &&other) = delete;
    Tracer(Tracer &&other) = delete;

    virtual void load_gaussians(const GaussiansData& data, const ASParams& as_params) = 0;

    virtual void trace_rays(const TracingParams &tracing_params) = 0;
};

}  // namespace gsrt
