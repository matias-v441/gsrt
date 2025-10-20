#include "kd_tracer.h"

#include <stdexcept>
struct not_implemented : std::logic_error {
    using std::logic_error::logic_error;
};        
using namespace gsrt;
using namespace gsrt::kd_tracer;

void KdTracer::load_gaussians(const GaussiansData& data, const ASParams& params) {
    const auto& kd_params = std::get<KdParams>(params);
    if(kd_params.device >= 0){
        //std::cout << "Building KdTree on cuda:" << static_cast<int>(kd_params.device) << "..." << std::endl;
        //scene_as_device.emplace();
        //scene_as_device->data = data;
        //cuda::builder::build(scene_as_device, kd_params);
        std::cout << "Building KdTree on host..." << std::endl;
        ASData_Device device_as(kd_params.device);
        device_as._dd.data = data;
        ASData_Host host_as(device_as);
        host::builder::build(host_as, kd_params);
        device_as.copy_from_host(host_as);
        scene_as_device.emplace(std::move(device_as));
    }else
    {
        std::cout << "Building KdTree on host..." << std::endl;
        scene_as.emplace();
        scene_as->data = data;
        host::builder::build(*scene_as, kd_params);
    }
}

void KdTracer::trace_rays(const TracingParams &tracing_params) {
    auto& settings = std::get<KdRenderSettings>(tracing_params.settings);
    if(!scene_as.has_value() && !scene_as_device.has_value()){
        throw std::runtime_error("KdTracer: build the acceleration structure before tracing.");
    }
    if(settings.device >=0){
        if(!scene_as_device.has_value()){
            scene_as_device.emplace(*scene_as);
        }
        cuda::traverser::rcast_kd_restart(*scene_as_device,tracing_params.rays,tracing_params.output);
    }else{
        if(!scene_as.has_value()){
            throw not_implemented("KdTracer: build the acceleration structure on host before tracing on host.");
        }
        std::cout << "Tracing on CPU (defaulting to KdTracerType::Restart)" << std::endl;
        host::traverser::rcast_kd_restart(*scene_as,tracing_params);
    }
    
}
