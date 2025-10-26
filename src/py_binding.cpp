#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <iostream>
#include <memory>
#include <string>
#include <chrono>
#include <variant>

#include "tracer.h"
#include "kd_tracer/kd_tracer.h"
#include "optix_tracer/optix_tracer.h"
namespace kd = gsrt::kd_tracer;
namespace optix = gsrt::optix_tracer;
using namespace gsrt;

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the `_a` literal

#define CHECK_DEVICE(dev,x) TORCH_CHECK(x.device() == device, #x " must be on the same device")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must have float32 type")
#define CHECK_INPUT(dev,x,d) \
    CHECK_CONTIGUOUS(x);    \
    CHECK_DEVICE(dev,x);        \
    CHECK_FLOAT(x);         \
    TORCH_CHECK(x.size(-1) == d, #x " must have last dimension with size " #d)


struct PyTracer {

    PyTracer(){}

    void load_gaussians(
        const torch::Tensor &xyz,
        const torch::Tensor &rotation,
        const torch::Tensor &scaling,
        const torch::Tensor &opacity,
        const torch::Tensor &sh,
        const int sh_deg,
        const py::dict &as_params
        ) {

        torch::Device device = xyz.device();
        CHECK_INPUT(device,xyz,3);
        CHECK_INPUT(device,rotation,4);
        CHECK_INPUT(device,scaling,3);
        CHECK_INPUT(device,opacity,1);
        CHECK_INPUT(device,sh,3);

        gsrt::ASParams params;
        if(!tracer){
            assert(as_params.contains("type"));
            const std::string type = as_params["type"].cast<std::string>();
            if(type == "optix"){
                params = gsrt::OptixASParams{};
                tracer = std::make_unique<optix::OptixTracer>(device.index());
            }else if(type == "kd"){
                gsrt::KdParams kd_params;
                if(as_params.contains("no_rebuild"))
                    kd_params.no_rebuild = as_params["no_rebuild"].cast<bool>();
                if(as_params.contains("max_leaf_size"))
                    kd_params.max_leaf_size = as_params["max_leaf_size"].cast<size_t>();
                kd_params.device = device.is_cpu() ? -1 : device.index();
                if(device.is_cpu()){
                    std::cout << "KdTracer: tracing on CPU" << std::endl;
                }else{
                    std::cout << "KdTracer: tracing on GPU " << kd_params.device << std::endl;
                }
                params = kd_params;
                tracer = std::make_unique<kd::KdTracer>();
            }else{
                throw std::runtime_error("Unknown ASParams type: " + type);
            }
        }

        this->xyz = xyz;
        this->rotation = rotation;
        this->scaling = scaling;
        this->opacity = opacity;
        this->sh = sh;
        
        gsrt::GaussiansData gs;
        gs.numgs = xyz.numel() / 3;
        gs.xyz = reinterpret_cast<float3 *>(xyz.data_ptr());
        gs.rotation = reinterpret_cast<float4 *>(rotation.data_ptr());
        gs.scaling = reinterpret_cast<float3 *>(scaling.data_ptr());
        gs.opacity = reinterpret_cast<float *>(opacity.data_ptr());
        gs.sh = reinterpret_cast<float3 *>(sh.data_ptr());
        gs.sh_deg = sh_deg;
        //gs.color = reinterpret_cast<float3*>(color.data_ptr());
        
        tracer->load_gaussians(gs, params);
    }

    py::dict trace_fwd(const torch::Tensor &ray_origins,
                        const torch::Tensor &ray_directions,
                        int width, int height,
                        bool white_background
                    ) {

        torch::Device device = ray_origins.device();
        CHECK_INPUT(device,ray_origins,3);
        CHECK_INPUT(device,ray_directions,3);
        
        torch::AutoGradMode enable_grad(false);
        const size_t num_rays = ray_origins.numel() / 3;
        gsrt::TracingParams tracing_params{};
        RayData rays{};
        rays.num_rays = num_rays;
        rays.width = width;
        rays.height = height;
        rays.ray_origins = reinterpret_cast<float3 *>(ray_origins.data_ptr());
        rays.ray_directions = reinterpret_cast<float3 *>(ray_directions.data_ptr());
        RenderOutput output{};
        torch::Tensor num_its = torch::zeros({1,1}, torch::device(device).dtype(torch::kInt64));
        torch::Tensor radiance = torch::zeros({(long)num_rays, 3}, torch::device(device).dtype(torch::kFloat32));
        torch::Tensor transmittance = torch::zeros({(long)num_rays, 1}, torch::device(device).dtype(torch::kFloat32));
        torch::Tensor distance = torch::zeros({(long)num_rays, 1}, torch::device(device).dtype(torch::kFloat32));
        torch::Tensor debug_map_0 = torch::zeros({(long)num_rays, 3}, torch::device(device).dtype(torch::kFloat32));
        output.num_its = reinterpret_cast<unsigned long long*>(num_its.data_ptr());
        output.radiance = reinterpret_cast<float3 *>(radiance.data_ptr());
        output.transmittance = reinterpret_cast<float *>(transmittance.data_ptr());
        output.distance = reinterpret_cast<float*>(distance.data_ptr());
        output.debug_map_0 = reinterpret_cast<float3 *>(debug_map_0.data_ptr());
        output.debug_map_1 = nullptr;//reinterpret_cast<float3 *>(debug_map_1.data_ptr());

        if(dynamic_cast<kd::KdTracer*>(tracer.get())){
            auto settings = gsrt::KdRenderSettings{};
            settings.white_background = white_background;
            settings.device = device.is_cpu() ? -1 : device.index();
            tracing_params.settings = settings;
        }else{
            auto settings = gsrt::OptixRenderSettings{};
            settings.white_background = white_background;
            tracing_params.settings = settings;
        }
        tracing_params.rays = rays;
        tracing_params.output = output;

        using namespace std::chrono;
        const auto frame_start = high_resolution_clock::now();

        tracer->trace_rays(tracing_params);

        const double ms_frame = duration_cast<milliseconds>(high_resolution_clock::now()-frame_start).count();

        return py::dict("radiance"_a = radiance, "transmittance"_a = transmittance,
                        "debug_map_0"_a = debug_map_0,
                        "time_ms"_a = ms_frame,
                        "num_its"_a = *reinterpret_cast<unsigned long*>(num_its.cpu().data_ptr()),
                        "distance"_a = distance
                        );
    }

    py::dict trace_bwd(const torch::Tensor &ray_origins,
                        const torch::Tensor &ray_directions,
                        int width, int height,
                        bool white_background,
                        const torch::Tensor &dL_dC,
                        const torch::Tensor &out_rad,
                        const torch::Tensor &out_trans,
                        const torch::Tensor &out_dist
                    ) {

        torch::Device device = ray_origins.device();
        CHECK_INPUT(device,ray_origins,3);
        CHECK_INPUT(device,ray_directions,3);
        CHECK_INPUT(device,dL_dC,3);
        CHECK_INPUT(device,out_rad,3);
        CHECK_INPUT(device,out_trans,1);
        CHECK_INPUT(device,out_dist,1);

        torch::AutoGradMode enable_grad(false);
        const size_t num_rays = ray_origins.numel() / 3;
        gsrt::TracingParams tracing_params{};
        RayData rays{};
        rays.num_rays = num_rays;
        rays.width = width;
        rays.height = height;
        rays.ray_origins = reinterpret_cast<float3 *>(ray_origins.data_ptr());
        rays.ray_directions = reinterpret_cast<float3 *>(ray_directions.data_ptr());
        tracing_params.rays = rays;
        RenderOutput output{};
        torch::Tensor num_its_bwd = torch::zeros({1,1}, torch::device(device).dtype(torch::kInt64));
        output.num_its_bwd = reinterpret_cast<unsigned long long*>(num_its_bwd.data_ptr());
        BaseRenderSettings settings{};
        settings.compute_grad = true;
        settings.white_background = white_background;

        gsrt::BPInput bp_in{};
        bp_in.dL_dC = reinterpret_cast<float3 *>(dL_dC.data_ptr());
        bp_in.radiance = reinterpret_cast<float3 *>(out_rad.data_ptr());
        bp_in.transmittance = reinterpret_cast<float *>(out_trans.data_ptr());
        bp_in.distance = reinterpret_cast<float *>(out_dist.data_ptr());
        gsrt::BPOutput bp_out{};
        long N = xyz.numel() / 3;
        torch::Tensor grad_xyz = torch::zeros({N,3}, torch::device(device).dtype(torch::kFloat32));
        torch::Tensor grad_opacity = torch::zeros({N}, torch::device(device).dtype(torch::kFloat32));
        torch::Tensor grad_sh = torch::zeros({N,16,3}, torch::device(device).dtype(torch::kFloat32));
        torch::Tensor grad_scale = torch::zeros({N,3}, torch::device(device).dtype(torch::kFloat32));
        torch::Tensor grad_rotation = torch::zeros({N,4}, torch::device(device).dtype(torch::kFloat32));
        assert(grad_xyz.is_contiguous());
        assert(grad_opacity.is_contiguous());
        assert(grad_sh.is_contiguous());
        assert(grad_scale.is_contiguous());
        assert(grad_rotation.is_contiguous());
        //grad_color = torch::zeros({(long)particles.numgs,3}, torch::device(device).dtype(torch::kFloat32));
        //grad_invRS = torch::zeros({(long)particles.numgs,3,3}, torch::device(device).dtype(torch::kFloat32));
        bp_out.grad_xyz = reinterpret_cast<float3 *>(grad_xyz.data_ptr());
        bp_out.grad_opacity = reinterpret_cast<float*>(grad_opacity.data_ptr());
        bp_out.grad_sh = reinterpret_cast<float3*>(grad_sh.data_ptr());
        bp_out.grad_scale = reinterpret_cast<float3*>(grad_scale.data_ptr());
        bp_out.grad_rotation = reinterpret_cast<float4*>(grad_rotation.data_ptr());
        tracing_params.bp_in = bp_in;
        tracing_params.bp_out = bp_out;
    
        tracing_params.settings = settings;
        tracing_params.rays = rays;
        tracing_params.output = output;

        using namespace std::chrono;
        const auto frame_start = high_resolution_clock::now();

        tracer->trace_rays(tracing_params);

        const double ms_frame = duration_cast<milliseconds>(high_resolution_clock::now()-frame_start).count();

        return py::dict("time_ms"_a = ms_frame,
                        "num_its_bwd"_a = *reinterpret_cast<unsigned long*>(num_its_bwd.cpu().data_ptr()),
                        "grad_xyz"_a = grad_xyz,
                        "grad_opacity"_a = grad_opacity,
                        "grad_sh"_a = grad_sh,
                        "grad_scale"_a = grad_scale,
                        "grad_rot"_a = grad_rotation
                        //"grad_color"_a = grad_color,
                        //"grad_invRS"_a = grad_invRS
                        );
    }

    bool has_gaussians() const {
        return xyz.defined() && rotation.defined() && scaling.defined() && opacity.defined() && sh.defined();
    }

   private:
    std::unique_ptr<gsrt::Tracer> tracer;
    torch::Tensor xyz;
    torch::Tensor rotation;
    torch::Tensor scaling;
    torch::Tensor opacity;
    torch::Tensor sh;
};

PYBIND11_MODULE(gsrt_cpp_extension, m) {
    py::class_<PyTracer>(m, "Tracer")
        .def(py::init<>())
        .def("trace_fwd", &PyTracer::trace_fwd)
        .def("trace_bwd", &PyTracer::trace_bwd)
        .def("has_gaussians", &PyTracer::has_gaussians)
        .def("load_gaussians", &PyTracer::load_gaussians)
        ;
}
