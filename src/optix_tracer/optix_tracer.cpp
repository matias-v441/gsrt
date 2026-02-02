#include "optix_tracer.h"
#include "tracer.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils/exception.h"

#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <iomanip>
#include <cstring>

using namespace gsrt::optix_tracer;

static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << "\n";
}

OptixTracer::OptixTracer(int8_t device)
    : device(device){

    // Initialize fields
    context = nullptr;

    // Switch to active device
    CUDA_CHECK(cudaSetDevice(device));

    // Load PTX first to make sure it exists

    char log[2048];  // For error reporting from OptiX creation functions

    //
    // Initialize CUDA and create OptiX context
    //
    {
        // Initialize CUDA
        // Warning: CUDA should have been already initialized at this point!!
        CUDA_CHECK(cudaFree(0));

        // Initialize the OptiX API, loading all API entry points
        OPTIX_CHECK(optixInit());

        // Specify context options
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;

        // Associate a CUDA context (and therefore a specific GPU) with this
        // device context
        CUcontext cuCtx = 0;  // zero means take the current context
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
    }

    trace_rays_pipeline = std::move(TraceRaysPipeline(context, device));
    gaussians_structure = std::move(GaussiansAS(context, device));
}

OptixTracer::~OptixTracer() noexcept(false) {
    // We call the destructor manually here to ensure correct destruction order
    gaussians_structure.~GaussiansAS();
    trace_rays_pipeline.~TraceRaysPipeline();

    if (context != nullptr && device != -1) {
        CUDA_CHECK(cudaSetDevice(device));
        OPTIX_CHECK(optixDeviceContextDestroy(std::exchange(context, nullptr)));
    }
}