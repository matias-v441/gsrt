#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <iostream>
#include <memory>
#include <string>
#include <chrono>

#include "gaussians_tracer.h"
#include "utils/exception.h"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the `_a` literal

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_DEVICE(x) TORCH_CHECK(x.device() == this->device, #x " must be on the same device")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must have float32 type")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT_DIM(x,d) \
    CHECK_INPUT(x);         \
    CHECK_DEVICE(x);        \
    CHECK_FLOAT(x);         \
    TORCH_CHECK(x.size(-1) == d, #x " must have last dimension with size d")
#define CHECK_HOST_FLOAT_DIM(x,d) \
    CHECK_CONTIGUOUS(x)     \
    CHECK_FLOAT(x);         \
    TORCH_CHECK(x.size(-1) == d, #x " must have last dimension with size d")


struct PyGaussiansTracer {
    PyGaussiansTracer(const torch::Device &device) : device(device) {
        if (!device.is_cuda()) {
            throw Exception("The device argument must be a CUDA device.");
        }
        tracer = std::make_unique<GaussiansTracer>(device.index());
    }
    ~PyGaussiansTracer() {
        tracer.reset();
    }

    py::dict trace_rays(const torch::Tensor &ray_origins,
                        const torch::Tensor &ray_directions,
                        int width, int height,
                        bool compute_grad,
                        const torch::Tensor &dL_dC) {

        torch::AutoGradMode enable_grad(false);
        CHECK_FLOAT_DIM(ray_origins,3);
        CHECK_FLOAT_DIM(ray_directions,3);
        const size_t num_rays = ray_origins.numel() / 3;
        const auto radiance = torch::zeros({(long)num_rays, 3}, torch::device(device).dtype(torch::kFloat32));
        const auto transmittance = torch::zeros({(long)num_rays}, torch::device(device).dtype(torch::kFloat32));

        const auto debug_map_0 = torch::zeros({(long)num_rays, 3}, torch::device(device).dtype(torch::kFloat32));
        const auto debug_map_1 = torch::zeros({(long)num_rays, 3}, torch::device(device).dtype(torch::kFloat32));

        const auto num_its = torch::zeros({1,1}, torch::device(device).dtype(torch::kInt64));
        const auto num_its_bwd = torch::zeros({1,1}, torch::device(device).dtype(torch::kInt64));
        using namespace std::chrono;
        const auto frame_start = high_resolution_clock::now();
        TracingParams tracing_params{};
        tracing_params.num_rays = num_rays;
        tracing_params.width = width;
        tracing_params.height = height;
        tracing_params.ray_origins = reinterpret_cast<float3 *>(ray_origins.data_ptr());
        tracing_params.ray_directions = reinterpret_cast<float3 *>(ray_directions.data_ptr());
        tracing_params.radiance = reinterpret_cast<float3 *>(radiance.data_ptr());
        tracing_params.transmittance = reinterpret_cast<float *>(transmittance.data_ptr());
        tracing_params.debug_map_0 = nullptr;//reinterpret_cast<float3 *>(debug_map_0.data_ptr());
        tracing_params.debug_map_1 = nullptr;//reinterpret_cast<float3 *>(debug_map_1.data_ptr());
        tracing_params.num_its = reinterpret_cast<unsigned long long*>(num_its.data_ptr());
        tracing_params.num_its_bwd = reinterpret_cast<unsigned long long*>(num_its_bwd.data_ptr());

        tracing_params.compute_grad = compute_grad;
        tracing_params.dL_dC = reinterpret_cast<float3*>(dL_dC.data_ptr());

        torch::Tensor grad_xyz,grad_opacity,grad_sh,grad_scale,grad_rot;
        if(compute_grad){
            grad_xyz = torch::zeros({(long)particles.numgs,3}, torch::device(device).dtype(torch::kFloat32));
            grad_opacity = torch::zeros({(long)particles.numgs}, torch::device(device).dtype(torch::kFloat32));
            grad_sh = torch::zeros({(long)particles.numgs,16,3}, torch::device(device).dtype(torch::kFloat32));
            grad_sh = grad_sh.contiguous();
            grad_scale = torch::zeros({(long)particles.numgs,3}, torch::device(device).dtype(torch::kFloat32));
            grad_rot = torch::zeros({(long)particles.numgs,4}, torch::device(device).dtype(torch::kFloat32));
            tracing_params.grad_xyz = reinterpret_cast<float3 *>(grad_xyz.data_ptr());
            tracing_params.grad_opacity = reinterpret_cast<float*>(grad_opacity.data_ptr());
            tracing_params.grad_sh = reinterpret_cast<float3*>(grad_sh.data_ptr());
            tracing_params.grad_scale = reinterpret_cast<float3*>(grad_scale.data_ptr());
            tracing_params.grad_rotation = reinterpret_cast<float4*>(grad_rot.data_ptr());
        }
        tracer->trace_rays(tracing_params);

        const auto frame_end = high_resolution_clock::now();
        const double ms_frame = duration_cast<milliseconds>(frame_end-frame_start).count();

        return py::dict("radiance"_a = radiance, "transmittance"_a = transmittance,
                        "debug_map_0"_a = debug_map_0, "debug_map_1"_a = debug_map_1,
                        "time_ms"_a = ms_frame,
                        "num_its"_a = *reinterpret_cast<unsigned long*>(num_its.cpu().data_ptr()),
                        "num_its_bwd"_a = *reinterpret_cast<unsigned long*>(num_its_bwd.cpu().data_ptr()),
                        "grad_xyz"_a = grad_xyz,
                        "grad_opacity"_a = grad_opacity,
                        "grad_sh"_a = grad_sh,
                        "grad_scale"_a = grad_scale,
                        "grad_rot"_a = grad_rot
                        );
    }

    void load_gaussians(
        const torch::Tensor &xyz,
        const torch::Tensor &rotation,
        const torch::Tensor &scaling,
        const torch::Tensor &opacity,
        const torch::Tensor &sh,
        const int sh_deg) {

        CHECK_HOST_FLOAT_DIM(xyz,3);
        CHECK_HOST_FLOAT_DIM(rotation,4);
        CHECK_HOST_FLOAT_DIM(scaling,3);
        CHECK_HOST_FLOAT_DIM(opacity,1);
        CHECK_HOST_FLOAT_DIM(sh,3);
        
        particles.numgs = xyz.numel() / 3;
        particles.xyz = reinterpret_cast<float3 *>(xyz.data_ptr());
        particles.rotation = reinterpret_cast<float4 *>(rotation.data_ptr());
        particles.scaling = reinterpret_cast<float3 *>(scaling.data_ptr());
        particles.opacity = reinterpret_cast<float *>(opacity.data_ptr());
        particles.sh = reinterpret_cast<float3 *>(sh.data_ptr());
        particles.sh_deg = sh_deg;
        tracer->load_gaussians(particles);
    }

    const torch::Device &get_device() const {
        return this->device;
    }

   private:
    std::unique_ptr<GaussiansTracer> tracer;
    GaussiansData particles;
    torch::Device device;
};

PYBIND11_MODULE(gsrt_cpp_extension, m) {
    py::class_<PyGaussiansTracer>(m, "GaussiansTracer")
        .def(py::init<const torch::Device &>())
        .def_property_readonly("device", &PyGaussiansTracer::get_device)
        .def("trace_rays", &PyGaussiansTracer::trace_rays)
        .def("load_gaussians", &PyGaussiansTracer::load_gaussians);
}
